from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


"""Validator for F3.5 `documents.jsonl` artifacts."""


def validate_documents(input_path: Path) -> dict[str, int]:
    """Validate required fields and basic line-number monotonicity."""

    documents = 0
    lines = 0
    placeholders = 0
    seen_doc_ids: set[str] = set()
    for line_no, row in enumerate(read_jsonl(input_path), start=1):
        _require(input_path, line_no, row, "doc_id")
        _require(input_path, line_no, row, "file_name")
        _require(input_path, line_no, row, "text")
        _require(input_path, line_no, row, "lines")
        if row["doc_id"] in seen_doc_ids:
            raise ValueError(f"{input_path}:{line_no} duplicate doc_id={row['doc_id']}")
        seen_doc_ids.add(row["doc_id"])
        if not isinstance(row["lines"], list) or not row["lines"]:
            raise ValueError(f"{input_path}:{line_no} lines must be a non-empty list")
        previous_line_no = 0
        for item in row["lines"]:
            if not isinstance(item, dict) or "line_no" not in item or "text" not in item:
                raise ValueError(f"{input_path}:{line_no} invalid line item")
            if item["line_no"] <= previous_line_no:
                raise ValueError(f"{input_path}:{line_no} line_no must be increasing")
            previous_line_no = item["line_no"]
        documents += 1
        lines += len(row["lines"])
        placeholders += len(row.get("placeholders") or [])
    return {"documents": documents, "lines": lines, "placeholders": placeholders}


def _require(path: Path, line_no: int, row: dict[str, Any], key: str) -> None:
    if key not in row or row[key] in ("", None, []):
        raise ValueError(f"{path}:{line_no} missing {key}")


def main() -> int:
    """Command-line validation entry point."""

    parser = argparse.ArgumentParser(description="Validate documents.jsonl")
    parser.add_argument("--input", required=True, help="Path to documents.jsonl")
    args = parser.parse_args()

    result = validate_documents(resolve_project_path(args.input))
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
