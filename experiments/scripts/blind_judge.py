#!/usr/bin/env python3
"""Blind re-judge of the formal5 answer bundles.

The original judge (build_all_and_score_formal5.py) scored all 17 methods for a
question in ONE list-wise call, labelled by their real names, with
`signpost.full` always first.  That leaks the ablation direction ("full" vs
"no_*") and pins our method to the primacy slot.

This script keeps EVERYTHING else identical -- same system prompt, same user
template, same 4/3/3 rubric, same model, temperature 0 -- and changes exactly
two things:

  * method names are replaced by opaque labels (系统 A, 系统 B, ...)
  * the order of the method lines is permuted per question (seeded)

so the only information removed is identity and position.
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urlrequest

SEED = 20260919
LABELS = [f"系统 {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

SYSTEM_PROMPT = (
    "你是一位严谨、客观且具备领域知识和算法评测经验的问答评测专家。"
    "你的任务是基于同一个问题的标准答案，对不同方法生成的回答进行逐一评估和打分。"
)

USER_PROMPT_TEMPLATE = """# 角色设定
你是一位严谨、客观且具备领域知识和算法评测经验的问答评测专家。你的任务是对一组针对同一问题的不同回答进行评估、打分，并给出具体反馈。

# 背景信息
- **任务目标**：评估不同检索增强/问答方法对同一问题生成答案的质量。
- **数据集**：{dataset}
- **标准参考**：`GOLD` 行是标准答案。请以标准答案为主要依据，同时结合问题本身判断各方法回答是否正确、完整、简洁。
- **待评估对象**：只评估 `GOLD` 之后、`QUESTION` 之前的各方法回答；不要给 `GOLD` 本身打分。

# 评分标准（满分 10 分）
请严格按照以下维度对每一个方法进行评估：
1. **准确性（4分）**：回答是否正确解决了问题？是否有事实性错误、幻觉、逻辑漏洞，或与标准答案矛盾？
2. **完整性（3分）**：是否涵盖标准答案中的所有必要信息、关键条件和原因？是否有重要遗漏？
3. **简洁性（3分）**：表达是否清晰直接？是否过于冗余、含混、绕远，或加入无关内容？

# 扣分规则
- 如果某个方法获得满分（10分），请简要说明它准确、完整、简洁的优秀之处。
- 如果某个方法没有获得满分，必须明确指出扣分项，并解释它在哪里出了问题、遗漏了什么，或者哪里可以优化。
- 如果回答说证据不足、拒答、空泛泛化，且标准答案可回答，应在准确性和完整性上明显扣分。
- 如果回答包含标准答案以外的大量无关信息，即使部分正确，也应在简洁性上扣分。

# 待评估的方法数据
{qa_text}

# 输出格式要求
请严格使用以下 Markdown 格式输出对每一个方法的评估结果。不要输出开场白、总结、JSON、代码块或任何额外信息。

### 方法 [方法名]
- **总分**：[X]/10
- **各维度得分**：准确性 [X]/4，完整性 [X]/3，简洁性 [X]/3
- **评价与扣分原因**：
  - [如果满分，说明优秀之处；如果不满分，具体说明扣分点]
- **改进建议**：[一两句话说明如何改进能拿满分]

---
请现在开始评估。
"""

DIM_RE = re.compile(r"准确性\s*([0-9.]+)\s*/\s*4[，,]\s*完整性\s*([0-9.]+)\s*/\s*3[，,]\s*简洁性\s*([0-9.]+)\s*/\s*3")
ACC_RE = re.compile(r"准确性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*4")
CMP_RE = re.compile(r"完整性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3")
CON_RE = re.compile(r"简洁性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3")


def parse_bundle(path: Path) -> dict | None:
    """Split an `all/<dataset>/NNN_<qid>.txt` bundle into gold / methods / meta."""
    gold, methods, meta = None, [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, _, val = line.partition("\t")
        if key == "GOLD":
            gold = val
        elif key in ("QUESTION", "QUESTION_ID", "DATASET"):
            meta[key] = val
        else:
            methods.append((key, val))
    if gold is None or not methods:
        return None
    return {"gold": gold, "methods": methods, **meta}


def parse_scores(text: str, label_to_method: dict[str, str]) -> dict[str, dict]:
    out = {}
    for section in re.split(r"^### 方法\s+", text, flags=re.M)[1:]:
        head, _, rest = section.partition("\n")
        label = head.strip().strip("[]").replace("**", "").strip()
        method = label_to_method.get(label)
        if method is None:                      # tolerate "系统A" / "系统 A"
            squashed = label.replace(" ", "")
            for lab, m in label_to_method.items():
                if lab.replace(" ", "") == squashed:
                    method = m
                    break
        if method is None:
            continue
        m = DIM_RE.search(rest)
        if m:
            acc, cmp_, con = (float(g) for g in m.groups())
        else:
            a, c, k = ACC_RE.search(rest), CMP_RE.search(rest), CON_RE.search(rest)
            if not (a and c and k):
                continue
            acc, cmp_, con = float(a.group(1)), float(c.group(1)), float(k.group(1))
        out[method] = {
            "llm_accuracy_score": acc,
            "llm_completeness_score": cmp_,
            "llm_conciseness_score": con,
            "llm_total_score": acc + cmp_ + con,
        }
    return out


def call_gateway(base: str, key: str, model: str, dataset: str, qa_text: str,
                 temperature: float, max_tokens: int, timeout: float, retries: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(dataset=dataset, qa_text=qa_text)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urlrequest.Request(
                base.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = (payload["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
            last = "empty completion"
        except Exception as exc:                # noqa: BLE001 - retry anything transient
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last or "unknown failure")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-dir", type=Path, default=Path("data/judge/all"))
    ap.add_argument("--out", type=Path, default=Path("data/judge/blind"))
    ap.add_argument("--model", default="plus")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    base = os.environ.get("LITELLM_BASE", "http://127.0.0.1:4000/v1")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    if not key:
        print("LITELLM_MASTER_KEY not set", file=sys.stderr)
        return 2

    wanted = {d for d in args.datasets.split(",") if d}
    files = sorted(p for p in args.all_dir.rglob("*.txt")
                   if not wanted or p.parent.name in wanted)
    if args.limit:
        files = files[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    done = {"n": 0, "fail": 0}

    def work(path: Path) -> dict | None:
        dataset = path.parent.name
        dest = args.out / dataset / (path.stem + ".json")
        if dest.exists():
            with lock:
                done["n"] += 1
            return json.loads(dest.read_text(encoding="utf-8"))
        bundle = parse_bundle(path)
        if bundle is None:
            return None
        # permutation + labels are a deterministic function of the question id
        order = list(bundle["methods"])
        random.Random(f"{SEED}:{bundle.get('QUESTION_ID', path.stem)}").shuffle(order)
        label_to_method = {}
        lines = ["GOLD\t" + bundle["gold"]]
        for label, (method, answer) in zip(LABELS, order):
            label_to_method[label] = method
            lines.append(f"{label}\t{answer}")
        lines.append("QUESTION\t" + bundle.get("QUESTION", ""))
        lines.append("QUESTION_ID\t" + bundle.get("QUESTION_ID", ""))
        lines.append("DATASET\t" + bundle.get("DATASET", dataset))
        try:
            text = call_gateway(base, key, args.model, dataset, "\n".join(lines),
                                args.temperature, args.max_tokens, args.timeout, args.retries)
        except Exception as exc:                # noqa: BLE001
            with lock:
                done["fail"] += 1
            print(f"FAIL {dataset}/{path.stem}: {exc}", file=sys.stderr)
            return None
        scores = parse_scores(text, label_to_method)
        row = {
            "dataset": dataset,
            "question_id": bundle.get("QUESTION_ID", ""),
            "label_to_method": label_to_method,
            "position": {m: i for i, (m, _) in enumerate(order)},
            "scores": scores,
            "raw": text,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        with lock:
            done["n"] += 1
            if done["n"] % 25 == 0:
                print(f"  {done['n']}/{len(files)} scored ({done['fail']} failed)", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, files))
    print(f"done: {done['n']}/{len(files)} scored, {done['fail']} failed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
