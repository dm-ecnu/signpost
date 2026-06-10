from __future__ import annotations

"""Run basic EM/precision/recall/F1 metrics on F16 prediction files."""

import argparse
import json

from signpost.config.context import resolve_project_path
from signpost.evaluation.metrics import evaluate_rows
from signpost.evaluation.schema import normalize_prediction_record
from signpost.parsing.io import read_jsonl


def evaluate_prediction_file(input_path: str, *, normalize: bool = False) -> dict:
    rows = []
    for row in read_jsonl(resolve_project_path(input_path)):
        rows.append(normalize_prediction_record(row) if normalize else row)
    result = evaluate_rows(rows)
    result["input"] = str(resolve_project_path(input_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="F16 basic prediction evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    result = evaluate_prediction_file(args.input, normalize=args.normalize)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} f1={result['metrics']['f1']:.4f}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
