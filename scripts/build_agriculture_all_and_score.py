#!/usr/bin/env python3
"""Build all QA comparison files for one dataset and score them with ECNU chat."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from build_suffer_samples import (
    DEFAULT_ROOT,
    load_jsonl,
    load_predictions,
    normalize_one_line,
    ordered_methods,
    question_id,
    safe_filename,
    standard_answer,
    write_sample_file,
)


DEFAULT_DATASET = "agriculture"
DEFAULT_PROJECT_DIR = Path("/home/ruolinsu/signpost/signpost_re")

SYSTEM_PROMPT = (
    "你是一位严谨、客观且具备跨领域知识和算法评测经验的问答评测专家。"
    "你的任务是基于同一个问题的标准答案，对不同方法生成的回答进行逐一评估和打分。"
)

USER_PROMPT_TEMPLATE = """# 角色设定
你是一位严谨、客观且具备跨领域知识和算法评测经验的问答评测专家。你的任务是对一组针对同一问题的不同回答进行评估、打分，并给出具体反馈。

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create <root>/all/<dest-dataset>/*.txt for all QA pairs, then score each "
            "file with ECNU and write <root>/ans/<dest-dataset>/*.txt."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--questions-dataset",
        default=None,
        help="Dataset name under datasets/processed/. Default: --dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Dataset/version name under outputs/. Default: --dataset.",
    )
    parser.add_argument(
        "--dest-dataset",
        default=None,
        help="Dataset/version name under all/ and ans/. Default: --output-dataset or --dataset.",
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--model", default=None, help="ECNU chat model. Default: env ECNU_CHAT_MODEL or ecnu-plus.")
    parser.add_argument("--base-url", default=None, help="ECNU API base. Default: env ECNU_API_BASE.")
    parser.add_argument("--api-key", default=None, help="ECNU API key. Default: env ECNU_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--sleep", type=float, default=0.8, help="Seconds to wait between ECNU calls.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-raw-prediction", action="store_true")
    parser.add_argument("--strict-json", action="store_true")
    parser.add_argument("--clean-all", action="store_true", help="Remove old all/agriculture .txt files first.")
    parser.add_argument("--clean-ans", action="store_true", help="Remove old ans/agriculture .txt files first.")
    parser.add_argument("--build-only", action="store_true", help="Only build all/agriculture files; do not score.")
    parser.add_argument("--score-only", action="store_true", help="Only score existing all/agriculture files.")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many files.")
    parser.add_argument("--overwrite", action="store_true", help="Re-score files even when ans output exists.")
    return parser.parse_args()


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def resolve_ecnu_config(args: argparse.Namespace) -> tuple[str, str, str]:
    file_env: dict[str, str] = {}
    for name in (".env.local.ecnu", ".env"):
        file_env.update(load_env_file(args.project_dir / name))

    base_url = (
        args.base_url
        or os.environ.get("ECNU_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or file_env.get("ECNU_API_BASE")
        or file_env.get("OPENAI_API_BASE")
        or "https://chat.ecnu.edu.cn/open/api/v1"
    ).rstrip("/")
    api_key = (
        args.api_key
        or os.environ.get("ECNU_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or file_env.get("ECNU_API_KEY")
        or file_env.get("OPENAI_API_KEY")
        or ""
    )
    model = args.model or os.environ.get("ECNU_CHAT_MODEL") or file_env.get("ECNU_CHAT_MODEL") or "ecnu-plus"

    if not api_key:
        raise ValueError("Missing ECNU API key. Set ECNU_API_KEY or pass --api-key.")
    return base_url, api_key, model


def build_all_files(
    root: Path,
    dataset: str,
    questions_dataset: str,
    output_dataset: str,
    dest_dataset: str,
    keep_raw: bool,
    strict_json: bool,
    clean: bool,
) -> dict[str, Any]:
    questions_path = root / "datasets" / "processed" / questions_dataset / "questions.jsonl"
    questions = load_jsonl(questions_path)
    predictions = load_predictions(root / "outputs" / output_dataset, keep_raw, strict_json)
    methods = ordered_methods(set(predictions))

    output_dir = root / "all" / dest_dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for old_file in output_dir.glob("*.txt"):
            old_file.unlink()

    manifest_path = output_dir / "manifest.jsonl"
    total_missing: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(questions, start=1):
            qid = question_id(row)
            sample_path = output_dir / f"{index:03d}_{safe_filename(qid)}.txt"
            missing = write_sample_file(sample_path, dataset, row, methods, predictions)
            for method in missing:
                total_missing[method] = total_missing.get(method, 0) + 1
            manifest.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "questions_dataset": questions_dataset,
                        "output_dataset": output_dataset,
                        "dest_dataset": dest_dataset,
                        "index": index,
                        "question_id": qid,
                        "question": row.get("question", ""),
                        "file": str(sample_path),
                        "missing_methods": missing,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "dataset": dataset,
        "questions": len(questions),
        "methods": methods,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "missing_counts": total_missing,
    }


def normalize_judge_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"### 方法 ", text)
    if match:
        text = text[match.start() :]
    return text.strip() + "\n"


def call_ecnu(
    qa_text: str,
    dataset: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(dataset=dataset, qa_text=qa_text.strip()),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"ECNU call failed after {retries} attempts: {last_error}")


def score_all_files(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    dataset = args.dataset
    dest_dataset = args.dest_dataset or args.output_dataset or args.dataset
    all_dir = root / "all" / dest_dataset
    ans_dir = root / "ans" / dest_dataset
    if not all_dir.is_dir():
        raise FileNotFoundError(f"Missing all QA dir: {all_dir}")
    ans_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_ans:
        for old_file in ans_dir.glob("*.txt"):
            old_file.unlink()

    base_url, api_key, model = resolve_ecnu_config(args)
    files = sorted(path for path in all_dir.glob("*.txt") if path.name != "manifest.jsonl")
    if args.limit is not None:
        files = files[: args.limit]

    scored = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for index, qa_path in enumerate(files, start=1):
        ans_path = ans_dir / qa_path.name
        if ans_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            qa_text = qa_path.read_text(encoding="utf-8")
            result = call_ecnu(
                qa_text=qa_text,
                dataset=dataset,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            ans_path.write_text(normalize_judge_output(result), encoding="utf-8")
            scored += 1
            print(f"[{index}/{len(files)}] scored {qa_path.name}", file=sys.stderr, flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:  # keep batch resumable
            failures.append({"file": str(qa_path), "error": str(exc)})
            print(f"ERROR: failed {qa_path}: {exc}", file=sys.stderr, flush=True)

    return {
        "dataset": dataset,
        "dest_dataset": dest_dataset,
        "all_dir": str(all_dir),
        "ans_dir": str(ans_dir),
        "model": model,
        "target_files": len(files),
        "scored": scored,
        "skipped_existing": skipped,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.questions_dataset = args.questions_dataset or args.dataset
    args.output_dataset = args.output_dataset or args.dataset
    args.dest_dataset = args.dest_dataset or args.output_dataset

    summaries: dict[str, Any] = {}
    if not args.score_only:
        summaries["build"] = build_all_files(
            root=args.root,
            dataset=args.dataset,
            questions_dataset=args.questions_dataset,
            output_dataset=args.output_dataset,
            dest_dataset=args.dest_dataset,
            keep_raw=args.keep_raw_prediction,
            strict_json=args.strict_json,
            clean=args.clean_all,
        )
    if not args.build_only:
        summaries["score"] = score_all_files(args)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
