from __future__ import annotations

"""Convert agent/retrieval outputs to the F16 prediction JSONL schema."""

import argparse

from signpost.evaluation.validate_predictions import validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="F16 convert prediction JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="signpost")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    result = validate_predictions(
        args.input,
        normalize=True,
        output_path=args.output,
        default_method=args.method,
        default_dataset=args.dataset,
    )
    print(f"output={result['output']} rows={result['num_rows']} valid={result['valid']}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
