from __future__ import annotations

"""F4 CLI: documents.jsonl -> chunks.jsonl and document_trees.jsonl."""

import argparse
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.chunking.chunker import chunk_document
from signpost.parsing.io import read_jsonl, write_jsonl


def run_chunking(input_path: Path, output_path: Path, tree_output_path: Path | None = None, *, max_tokens: int = 1200, overlap_tokens: int = 100, use_llm: bool = False) -> int:
    all_chunks = []
    trees = []
    for document in read_jsonl(input_path):
        chunks, tree = chunk_document(document, max_tokens=max_tokens, overlap_tokens=overlap_tokens, use_llm=use_llm)
        all_chunks.extend(chunk.to_dict() for chunk in chunks)
        trees.append(tree)
    count = write_jsonl(output_path, all_chunks)
    if tree_output_path is not None:
        write_jsonl(tree_output_path, trees)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run F4 document-tree chunking")
    parser.add_argument("--input", required=True, help="documents.jsonl")
    parser.add_argument("--output", required=True, help="chunks.jsonl")
    parser.add_argument("--tree-output", help="document_trees.jsonl")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--use-llm", action="store_true", help="Use thesis LLM dual-path chapter recognition")
    args = parser.parse_args()

    count = run_chunking(
        resolve_project_path(args.input),
        resolve_project_path(args.output),
        resolve_project_path(args.tree_output) if args.tree_output else None,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        use_llm=args.use_llm,
    )
    print(f"chunks={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

