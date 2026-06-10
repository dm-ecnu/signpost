from __future__ import annotations

"""F14 ReadFile / provenance reader.

The research pipeline reads normalized `documents.jsonl` artifacts instead of
old product storage.  A caller can pass either file + line range or a locate
string like `file.txt:L10-L35`.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def read_file_window(
    *,
    dataset: str | None = None,
    documents_path: Path | None = None,
    file_name: str,
    start_line: int,
    end_line: int,
    before: int = 0,
    after: int = 0,
) -> dict[str, Any]:
    if start_line <= 0 or end_line <= 0:
        raise ValueError("start_line and end_line must be positive")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    target_path = documents_path or documents_path_for_dataset(dataset or "")
    document = _find_document(target_path, file_name)
    line_map = {int(row["line_no"]): str(row["text"]) for row in document.get("lines", [])}
    read_start = max(1, start_line - max(0, before))
    read_end = end_line + max(0, after)
    rows = [{"line_no": line_no, "text": line_map[line_no]} for line_no in range(read_start, read_end + 1) if line_no in line_map]
    return {
        "tool": "read_file",
        "dataset": dataset,
        "documents_path": str(target_path),
        "doc_id": document.get("doc_id"),
        "file_name": document.get("file_name"),
        "requested": {"start_line": start_line, "end_line": end_line, "before": before, "after": after},
        "resolved": {"start_line": read_start, "end_line": read_end},
        "lines": rows,
        "file_content_view": format_file_view(str(document.get("file_name")), rows),
    }


def read_locate(
    locate: str,
    *,
    dataset: str | None = None,
    documents_path: Path | None = None,
    before: int = 0,
    after: int = 0,
) -> dict[str, Any]:
    file_name, start_line, end_line = parse_locate(locate)
    return read_file_window(dataset=dataset, documents_path=documents_path, file_name=file_name, start_line=start_line, end_line=end_line, before=before, after=after)


def parse_locate(locate: str) -> tuple[str, int, int]:
    match = re.match(r"^(?P<file>.+):L(?P<start>\d+)-L(?P<end>\d+)$", locate)
    if not match:
        raise ValueError(f"invalid locate format: {locate}; expected file.txt:L10-L35")
    return match.group("file"), int(match.group("start")), int(match.group("end"))


def documents_path_for_dataset(dataset: str) -> Path:
    if not dataset:
        raise ValueError("dataset is required when documents_path is not provided")
    candidates = [
        resolve_project_path(f"datasets/processed/{dataset}/documents.jsonl"),
        resolve_project_path(f"outputs/{dataset}/documents.jsonl"),
        resolve_project_path(f"samples/{dataset}/documents.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"documents.jsonl not found for dataset={dataset}; checked: {', '.join(str(path) for path in candidates)}")


def format_file_view(file_name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"=== {file_name} (0 lines) ==="
    start = rows[0]["line_no"]
    end = rows[-1]["line_no"]
    lines = [f"=== {file_name}:L{start}-L{end} ==="]
    lines.extend(f"{int(row['line_no']):>6} | {row['text']}" for row in rows)
    return "\n".join(lines)


def _find_document(documents_path: Path, file_name: str) -> dict[str, Any]:
    matches = []
    for document in read_jsonl(documents_path):
        if document.get("file_name") == file_name or document.get("doc_id") == file_name:
            matches.append(document)
    if not matches:
        raise FileNotFoundError(f"file_name={file_name} not found in {documents_path}")
    if len(matches) > 1:
        raise ValueError(f"file_name={file_name} matched {len(matches)} documents in {documents_path}; use a unique doc_id/file_name")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="F14 read source file lines by provenance coordinates")
    parser.add_argument("--dataset")
    parser.add_argument("--documents")
    parser.add_argument("--file")
    parser.add_argument("--start-line", type=int)
    parser.add_argument("--end-line", type=int)
    parser.add_argument("--locate")
    parser.add_argument("--before", type=int, default=0)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Optional path to write the JSON result.")
    args = parser.parse_args()

    documents_path = resolve_project_path(args.documents) if args.documents else None
    if args.locate:
        result = read_locate(args.locate, dataset=args.dataset, documents_path=documents_path, before=args.before, after=args.after)
    else:
        if not args.file or args.start_line is None or args.end_line is None:
            parser.error("provide either --locate or --file with --start-line and --end-line")
        result = read_file_window(
            dataset=args.dataset,
            documents_path=documents_path,
            file_name=args.file,
            start_line=args.start_line,
            end_line=args.end_line,
            before=args.before,
            after=args.after,
        )
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={output} lines={len(result.get('lines', []))}")
    elif args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["file_content_view"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
