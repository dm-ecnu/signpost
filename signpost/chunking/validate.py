from __future__ import annotations

"""F4 validator for chunks.jsonl."""

import argparse
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def validate_chunks(path: Path) -> dict[str, int]:
    chunks = 0
    linked = 0
    seen: set[str] = set()
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        for key in ("chunk_id", "doc_id", "file_name", "content", "start_line", "end_line", "section_path", "metadata"):
            if key not in row or row[key] in ("", None, []):
                raise ValueError(f"{path}:{line_no} missing {key}")
        if row["chunk_id"] in seen:
            raise ValueError(f"{path}:{line_no} duplicate chunk_id={row['chunk_id']}")
        if int(row["start_line"]) > int(row["end_line"]):
            raise ValueError(f"{path}:{line_no} invalid line range")
        seen.add(row["chunk_id"])
        by_doc.setdefault(row["doc_id"], []).append(row)
        chunks += 1
    for doc_chunks in by_doc.values():
        for idx, chunk in enumerate(doc_chunks):
            if idx > 0 and chunk.get("prev_chunk_id") == doc_chunks[idx - 1]["chunk_id"]:
                linked += 1
            if idx + 1 < len(doc_chunks) and chunk.get("next_chunk_id") != doc_chunks[idx + 1]["chunk_id"]:
                raise ValueError(f"{path}: invalid next link for {chunk['chunk_id']}")
    return {"chunks": chunks, "documents": len(by_doc), "linked_prev": linked}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate F4 chunks.jsonl")
    parser.add_argument("--chunks", required=True)
    args = parser.parse_args()
    result = validate_chunks(resolve_project_path(args.chunks))
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

