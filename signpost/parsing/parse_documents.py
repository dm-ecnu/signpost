from __future__ import annotations

import argparse
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl, write_jsonl
from signpost.parsing.parser import parse_raw_corpus_row


"""CLI for F3.5 document parsing.

This is the boundary between F3 standardized corpus metadata and F4 chunking.
It intentionally does not infer chapters or chunks; it only normalizes documents
and preserves source line locations.
"""


def parse_documents(input_path: Path, output_path: Path) -> int:
    """Parse all raw corpus rows and write `documents.jsonl`."""

    return write_jsonl(output_path, (parse_raw_corpus_row(row).payload for row in read_jsonl(input_path)))


def main() -> int:
    """Command-line entry point used by smoke tests and batch parsing."""

    parser = argparse.ArgumentParser(description="Parse raw_corpus.jsonl into documents.jsonl")
    parser.add_argument("--input", required=True, help="Path to raw_corpus.jsonl")
    parser.add_argument("--output", required=True, help="Path to documents.jsonl")
    args = parser.parse_args()

    count = parse_documents(resolve_project_path(args.input), resolve_project_path(args.output))
    print(f"parsed_documents={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
