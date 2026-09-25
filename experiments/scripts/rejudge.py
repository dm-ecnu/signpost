#!/usr/bin/env python3
"""Re-run the MuSiQue LLM judge so every method is scored on every question.

Why: in the frozen 2026-06-09 run the judge output was truncated in 54 of the 100
files -- it emitted the seven signpost.* sections and stopped -- so the six
baselines ended up scored on only 24 questions while signpost.full was scored on
all 100. The published MuSiQue S_LLM column therefore compares different question
sets. This script re-runs the *same* joint prompt (identical SYSTEM/USER text,
temperature 0, thinking disabled) with a larger output budget, and retries any
file whose response is missing a method section.

Model: `plus` on the local litellm gateway == ECNU ecnu-plus == the original judge.
Local cache stays off (gateway default is default_off); a cache hit would replay a
previous verdict and ignore temperature.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import queue
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATEWAY = "http://127.0.0.1:4000/v1/chat/completions"

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

DIM = re.compile(r"准确性\s*([0-9.]+)\s*/\s*4[，,]\s*完整性\s*([0-9.]+)\s*/\s*3[，,]\s*简洁性\s*([0-9.]+)\s*/\s*3")


def master_key() -> str:
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    if not key:
        raise SystemExit("set LITELLM_MASTER_KEY (key of the OpenAI-compatible judge endpoint)")
    return key


def methods_of(qa_text: str) -> list[str]:
    """Method names in the judge input: every tab-separated line except GOLD/QUESTION."""
    out = []
    for line in qa_text.splitlines():
        if "\t" not in line:
            continue
        name = line.split("\t", 1)[0].strip()
        if name and name not in {"GOLD", "QUESTION", "QUESTION_ID", "DATASET"} and name not in out:
            out.append(name)
    return out


def parsed_sections(text: str) -> set[str]:
    out = set()
    for sec in re.split(r"^### 方法\s+", text, flags=re.M)[1:]:
        name, _, rest = sec.partition("\n")
        if DIM.search(rest) or re.search(r"准确性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*4", rest):
            out.add(name.strip())
    return out


def call(key: str, dataset: str, qa_text: str, model: str, max_tokens: int, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(dataset=dataset, qa_text=qa_text.strip())},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        GATEWAY,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(HERE / "musique_in"))
    ap.add_argument("--out-dir", default=str(HERE / "musique_ans"))
    ap.add_argument("--dataset", default="musique_q100")
    ap.add_argument("--model", default="plus")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    key = master_key()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.glob("*.txt") if p.name != "manifest.jsonl")

    todo: queue.Queue[Path] = queue.Queue()
    for p in files:
        dst = out_dir / p.name
        if dst.exists() and not args.overwrite:
            if parsed_sections(dst.read_text(encoding="utf-8")) >= set(methods_of(p.read_text(encoding="utf-8"))):
                continue
        todo.put(p)

    total = todo.qsize()
    print(f"{len(files)} inputs, {total} to (re)judge, model={args.model}", flush=True)
    lock = threading.Lock()
    done = [0]
    bad: list[str] = []

    def worker() -> None:
        while True:
            try:
                src = todo.get_nowait()
            except queue.Empty:
                return
            qa = src.read_text(encoding="utf-8")
            want = set(methods_of(qa))
            best, best_missing = "", want
            for attempt in range(1, args.attempts + 1):
                try:
                    text = call(key, args.dataset, qa, args.model, args.max_tokens, args.timeout)
                except Exception as exc:                      # noqa: BLE001
                    time.sleep(min(2 ** attempt, 15))
                    if attempt == args.attempts:
                        with lock:
                            bad.append(f"{src.name}: {type(exc).__name__}")
                    continue
                missing = want - parsed_sections(text)
                if len(missing) < len(best_missing):
                    best, best_missing = text, missing
                if not missing:
                    break
                time.sleep(1)
            if best:
                (out_dir / src.name).write_text(best, encoding="utf-8")
            with lock:
                done[0] += 1
                if best_missing:
                    bad.append(f"{src.name}: missing {sorted(best_missing)}")
                if done[0] % 10 == 0 or done[0] == total:
                    print(f"  {done[0]}/{total}", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"done. incomplete/failed: {len(bad)}")
    for line in bad[:20]:
        print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
