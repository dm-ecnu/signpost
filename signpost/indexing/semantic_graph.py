from __future__ import annotations

"""F6 CLI: chunks.jsonl -> graph.semantic.json."""

import argparse
import json
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.graph.semantic import build_semantic_graph
from signpost.indexing.semantic_extractor import ExtractionResult, create_semantic_extractor, extraction_result_from_dict, extraction_result_to_dict
from signpost.parsing.io import read_jsonl, write_jsonl


def build_semantic_graph_file(
    chunks_path: Path,
    output_path: Path,
    *,
    namespace: str,
    extractor_name: str = "llm",
    gleaning_rounds: int = 2,
    max_chunks: int | None = None,
    synthesize_descriptions: bool = False,
    llm_retries: int = 3,
    retry_sleep: float = 2.0,
    llm_timeout: int = 120,
    progress_every: int = 0,
    progress_file: Path | None = None,
    extractions_cache: Path | None = None,
) -> dict[str, int | str]:
    chunks = list(read_jsonl(chunks_path))
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    extractor = create_semantic_extractor(extractor_name, gleaning_rounds=gleaning_rounds, retries=llm_retries, retry_sleep=retry_sleep, timeout=llm_timeout)
    cached = _load_extraction_cache(extractions_cache)
    missing_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in cached]
    if extractions_cache is not None and missing_chunks:
        _extract_missing_chunks(missing_chunks, extractor, cache_path=extractions_cache, progress_every=progress_every, progress_file=progress_file)
        cached = _load_extraction_cache(extractions_cache)
    graph = build_semantic_graph(
        chunks,
        extractor,
        namespace=namespace,
        synthesize_descriptions=synthesize_descriptions,
        progress_every=progress_every,
        progress_file=progress_file,
        extraction_results=cached if extractions_cache is not None else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "chunks": graph["metadata"]["chunks"],
        "entities": graph["metadata"]["entities"],
        "relations": graph["metadata"]["relations"],
        "source_edges": graph["metadata"]["source_edges"],
    }


def _extract_missing_chunks(
    chunks: list[dict[str, Any]],
    extractor,
    *,
    cache_path: Path,
    progress_every: int,
    progress_file: Path | None,
) -> None:
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if progress_every:
            print(f"semantic_extract_cache extracting={idx}/{total} chunk_id={chunk['chunk_id']} tokens={chunk.get('metadata', {}).get('token_count')}", flush=True)
        _append_progress(progress_file, {"event": "cache_extracting", "index": idx, "total": total, "chunk_id": chunk["chunk_id"], "tokens": chunk.get("metadata", {}).get("token_count")})
        result = extractor.extract(chunk)
        _append_extraction_cache(
            cache_path,
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk.get("doc_id"),
                    "file_name": chunk.get("file_name"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "extraction": extraction_result_to_dict(result),
                }
            ],
        )
        if progress_every and idx % progress_every == 0:
            print(f"semantic_extract_cache processed={idx}/{total}", flush=True)
        _append_progress(progress_file, {"event": "cache_processed", "index": idx, "total": total, "chunk_id": chunk["chunk_id"]})


def _load_extraction_cache(path: Path | None) -> dict[str, ExtractionResult]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, ExtractionResult] = {}
    for row in read_jsonl(path):
        chunk_id = row.get("chunk_id")
        extraction = row.get("extraction")
        if isinstance(chunk_id, str) and isinstance(extraction, dict):
            cache[chunk_id] = extraction_result_from_dict(extraction)
    return cache


def _append_extraction_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _append_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build F6 semantic entity-relation graph")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extractor", choices=["llm", "deterministic"], default="llm")
    parser.add_argument("--gleaning-rounds", type=int, default=2)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--progress-file", help="Write JSONL progress events to this file.")
    parser.add_argument("--extractions-cache", help="JSONL cache for per-chunk extraction results; enables resume.")
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--synthesize-descriptions", action="store_true", help="Use LLM to synthesize multi-source entity/relation descriptions")
    args = parser.parse_args()

    output_path = resolve_project_path(args.output)
    extractions_cache = resolve_project_path(args.extractions_cache) if args.extractions_cache else output_path.with_suffix(".extractions.jsonl") if args.extractor == "llm" else None
    result = build_semantic_graph_file(
        resolve_project_path(args.chunks),
        output_path,
        namespace=args.namespace,
        extractor_name=args.extractor,
        gleaning_rounds=args.gleaning_rounds,
        max_chunks=args.max_chunks,
        synthesize_descriptions=args.synthesize_descriptions,
        llm_retries=args.llm_retries,
        retry_sleep=args.retry_sleep,
        llm_timeout=args.llm_timeout,
        progress_every=args.progress_every,
        progress_file=resolve_project_path(args.progress_file) if args.progress_file else None,
        extractions_cache=extractions_cache,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
