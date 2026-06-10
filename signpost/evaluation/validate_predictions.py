from __future__ import annotations

"""Validate and optionally normalize F16 prediction JSONL files."""

import argparse
import json
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.evaluation.schema import normalize_prediction_record, validate_prediction_record
from signpost.parsing.io import read_jsonl, write_jsonl


def validate_predictions(
    input_path: str | Path,
    *,
    normalize: bool = False,
    output_path: str | Path | None = None,
    default_method: str = "signpost",
    default_dataset: str | None = None,
) -> dict:
    rows = []
    issues = []
    for line_no, row in enumerate(read_jsonl(resolve_project_path(input_path)), start=1):
        candidate = normalize_prediction_record(row, default_method=default_method, default_dataset=default_dataset) if normalize else row
        rows.append(candidate)
        issues.extend(validate_prediction_record(candidate, line_no))
    if output_path:
        write_jsonl(resolve_project_path(output_path), rows)
    return {
        "input": str(resolve_project_path(input_path)),
        "output": str(resolve_project_path(output_path)) if output_path else None,
        "num_rows": len(rows),
        "valid": not issues,
        "issues": [issue.__dict__ for issue in issues],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="F16 validate prediction JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--normalize", action="store_true", help="Normalize agent/retrieval rows into F16 schema before validating.")
    parser.add_argument("--default-method", default="signpost")
    parser.add_argument("--default-dataset")
    args = parser.parse_args()

    result = validate_predictions(
        args.input,
        normalize=args.normalize,
        output_path=args.output,
        default_method=args.default_method,
        default_dataset=args.default_dataset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
