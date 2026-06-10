from __future__ import annotations

"""Vanilla LLM baseline with no retrieval."""

import argparse
from typing import Any

from signpost.baselines.common import BaselineResult, build_paths, chat_once, question_text, run_baseline_batch
from signpost.config.context import resolve_project_path
from signpost.llm.client import OpenAICompatibleClient


METHOD = "vanilla_llm"


def answer_question(row: dict[str, Any], *, llm: OpenAICompatibleClient) -> BaselineResult:
    question = question_text(row)
    messages = [
        {
            "role": "system",
            "content": "Answer the question directly. Do not cite documents because no retrieval context is provided.",
        },
        {"role": "user", "content": question},
    ]
    answer, input_tokens, output_tokens, llm_latency = chat_once(llm, messages, input_text=question)
    return BaselineResult(
        answer=answer,
        rationale="No retrieval context was used.",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=1.0,
        tool_calls=0.0,
        trace=[
            {
                "event_type": "llm_call",
                "stage": "vanilla_llm_answer",
                "latency_seconds": llm_latency,
                "input_tokens_estimate": input_tokens,
                "output_tokens_estimate": output_tokens,
            }
        ],
    )


def run_vanilla_llm(
    *,
    dataset: str,
    namespace: str | None = None,
    questions_path: str | None = None,
    output_path: str | None = None,
    query_log_path: str | None = None,
    limit: int | None = None,
    workers: int | None = None,
) -> int:
    paths = build_paths(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method=METHOD,
    )
    llm = OpenAICompatibleClient()
    return run_baseline_batch(
        method=METHOD,
        paths=paths,
        answer_fn=lambda row: answer_question(row, llm=llm),
        limit=limit,
        workers=workers,
        metadata={"retrieval": "none"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vanilla LLM baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    count = run_vanilla_llm(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        output_path=args.output,
        query_log_path=args.query_log,
        limit=args.limit,
        workers=args.workers,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
