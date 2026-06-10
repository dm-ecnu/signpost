from __future__ import annotations

"""Optional LLM-as-Judge adapter for F16.

The implementation keeps the judge prompt and parser local so experiments can
run with the same ECNU/OpenAI-compatible client used by the rest of signpost_re.
"""

import argparse
import json
import re
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.evaluation.metrics import extract_answer_from_prediction
from signpost.llm.client import OpenAICompatibleClient
from signpost.parsing.io import read_jsonl, write_jsonl


DIMENSIONS = {
    "answer_correctness": "预测答案是否与标准答案语义一致，是否直接回答问题。",
    "factuality": "回答是否基于可验证事实，是否避免编造。",
    "completeness": "回答是否覆盖问题所需的关键方面。",
}


def build_judge_prompt(row: dict[str, Any], dimension: str) -> str:
    return f"""你是一个严格的问答评估员。请只根据问题、标准答案和预测答案评分。

评分维度：{dimension}
评分目标：{DIMENSIONS[dimension]}
分数范围：0 到 10 的整数，10 表示最好。

问题：
{row.get("question", "")}

标准答案：
{row.get("answer", "")}

预测答案：
{extract_answer_from_prediction(str(row.get("prediction", "")))}

请按如下格式输出：
<explanation>简短说明</explanation>
<score>整数分数</score>
"""


def parse_score(raw: str) -> dict[str, Any]:
    match = re.search(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", raw, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"judge response does not contain <score>: {raw[:200]}")
    score = max(0.0, min(10.0, float(match.group(1))))
    return {"score": score / 10.0, "raw_score": score, "raw_response": raw}


def judge_file(input_path: str, output_path: str, *, dimensions: list[str] | None = None) -> int:
    selected = dimensions or list(DIMENSIONS)
    client = OpenAICompatibleClient()
    rows = []
    for row in read_jsonl(resolve_project_path(input_path)):
        metrics = {}
        for dimension in selected:
            raw = client.chat([{"role": "user", "content": build_judge_prompt(row, dimension)}])
            metrics[dimension] = parse_score(raw)
        rows.append({**row, "llm_judge": metrics})
    return write_jsonl(resolve_project_path(output_path), rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="F16 optional LLM-as-Judge evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", action="append", choices=sorted(DIMENSIONS))
    args = parser.parse_args()

    count = judge_file(args.input, args.output, dimensions=args.dimension)
    print(json.dumps({"output": str(resolve_project_path(args.output)), "count": count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
