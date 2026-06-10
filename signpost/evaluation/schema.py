from __future__ import annotations

"""F16 prediction JSONL schema helpers.

Evaluation scripts in the reference project expect a compact record containing
question, gold answer/rationale, prediction, and method/dataset metadata.  These
helpers normalize outputs from F15 agent runs or other retrieval baselines into
that stable format.
"""

from dataclasses import dataclass
from typing import Any


REQUIRED_FIELDS = ("question_id", "question", "answer", "prediction", "metadata")
OPTIONAL_PASSTHROUGH_FIELDS = (
    "citations",
    "trace_id",
    "trace",
    "retrieved_chunks",
    "latency_seconds",
    "retrieval_latency_seconds",
    "ppr_latency_seconds",
    "read_file_latency_seconds",
    "agent_reasoning_latency_seconds",
    "online_llm_calls",
    "llm_calls",
    "tool_calls",
    "knowledge_search_calls",
    "read_file_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "graph_ppr_calls",
    "max_context_tokens",
)


@dataclass(frozen=True)
class ValidationIssue:
    line_no: int
    field: str
    message: str


def normalize_prediction_record(row: dict[str, Any], *, default_method: str = "signpost", default_dataset: str | None = None) -> dict[str, Any]:
    question_id = str(row.get("question_id") or row.get("id") or row.get("qid") or "")
    question = str(row.get("question") or row.get("query") or row.get("input") or "")
    gold_answer = row.get("gold_answer", row.get("answer", ""))
    rationale = row.get("rationale", row.get("gold_rationale", ""))
    prediction = row.get("prediction")
    if prediction is None:
        generated_answer = str(row.get("generated_answer") or row.get("model_answer") or row.get("answer") or "")
        generated_rationale = str(row.get("generated_rationale") or row.get("reasoning") or "")
        prediction = build_prediction_text(answer=generated_answer, rationale=generated_rationale)
    metadata = _metadata(row, default_method=default_method, default_dataset=default_dataset)
    normalized = {
        "question_id": question_id,
        "question": question,
        "answer": gold_answer,
        "rationale": rationale,
        "prediction": str(prediction),
        "metadata": metadata,
    }
    for key in OPTIONAL_PASSTHROUGH_FIELDS:
        if key in row:
            normalized[key] = row[key]
    return normalized


def build_prediction_text(*, answer: str, rationale: str = "") -> str:
    answer = answer.strip()
    rationale = rationale.strip()
    if rationale:
        return f"<think>\n{rationale}\n</think>\n<answer>\n{answer}\n</answer>"
    return f"<answer>\n{answer}\n</answer>"


def validate_prediction_record(row: dict[str, Any], line_no: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(ValidationIssue(line_no, field, "missing required field"))
    for field in ("question_id", "question", "prediction"):
        if field in row and not str(row.get(field, "")).strip():
            issues.append(ValidationIssue(line_no, field, "must be a non-empty string"))
    if "metadata" in row and not isinstance(row["metadata"], dict):
        issues.append(ValidationIssue(line_no, "metadata", "must be an object"))
    else:
        metadata = row.get("metadata", {})
        for key in ("method", "dataset"):
            if not str(metadata.get(key, "")).strip():
                issues.append(ValidationIssue(line_no, f"metadata.{key}", "must be present"))
    return issues


def _metadata(row: dict[str, Any], *, default_method: str, default_dataset: str | None) -> dict[str, Any]:
    source = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    metadata = dict(source)
    metadata["method"] = row.get("method") or metadata.get("method") or default_method
    metadata["dataset"] = row.get("dataset") or metadata.get("dataset") or default_dataset or ""
    return metadata
