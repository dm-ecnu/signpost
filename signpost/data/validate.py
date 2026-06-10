from __future__ import annotations

"""F3 validator for processed datasets.

This command verifies the contract required by F3.5: stable document identifiers,
document text or source path, source format, and question identifiers.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def validate_dataset(dataset: str) -> dict[str, Any]:
    base = resolve_project_path(Path("datasets/processed") / dataset)
    raw_corpus = base / "raw_corpus.jsonl"
    questions = base / "questions.jsonl"
    if not raw_corpus.exists():
        raise FileNotFoundError(raw_corpus)
    if not questions.exists():
        raise FileNotFoundError(questions)

    doc_ids: set[str] = set()
    doc_count = 0
    for line_no, row in enumerate(read_jsonl(raw_corpus), start=1):
        for key in ("doc_id", "file_name", "source_format", "metadata"):
            if key not in row or row[key] in ("", None):
                raise ValueError(f"{raw_corpus}:{line_no} missing {key}")
        if not row.get("text") and not row.get("source_path"):
            raise ValueError(f"{raw_corpus}:{line_no} needs text or source_path")
        if row["doc_id"] in doc_ids:
            raise ValueError(f"{raw_corpus}:{line_no} duplicate doc_id={row['doc_id']}")
        doc_ids.add(row["doc_id"])
        doc_count += 1

    question_count = 0
    linked_questions = 0
    for line_no, row in enumerate(read_jsonl(questions), start=1):
        for key in ("question_id", "question", "metadata"):
            if key not in row or row[key] in ("", None):
                raise ValueError(f"{questions}:{line_no} missing {key}")
        question_count += 1
        linked_questions += bool(row.get("doc_ids"))

    return {
        "dataset": dataset,
        "documents": doc_count,
        "questions": question_count,
        "questions_with_doc_ids": linked_questions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate F3 processed dataset files")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

