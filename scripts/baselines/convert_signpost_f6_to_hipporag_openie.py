from __future__ import annotations

"""Convert Signpost F6 semantic extractions to HippoRAG2 OpenIE JSON.

The converter does not run a new extractor. It only rewrites the fixed
``chunks.jsonl`` and ``semantic_llm.extractions.jsonl`` artifacts into the
``{"docs": [...]}`` format expected by the GraphRAG-R1 HippoRAG2 server.
"""

import argparse
import json
from pathlib import Path
from typing import Any


METHOD = "graphrag_r1_hipporag2"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Signpost F6 extractions to HippoRAG2 OpenIE JSON.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chunks")
    parser.add_argument("--extractions")
    parser.add_argument("--output")
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    chunks_path = Path(args.chunks or f"datasets/processed/{args.dataset}/chunks.jsonl")
    extractions_path = Path(args.extractions or f"datasets/processed/{args.dataset}/semantic_llm.extractions.jsonl")
    output_path = Path(
        args.output or f"outputs/{args.dataset}/baselines/{METHOD}/server/openie_results_ner_signpost_f6.json"
    )
    manifest_path = Path(args.manifest_output) if args.manifest_output else output_path.with_suffix(".manifest.json")

    chunks = list(_read_jsonl(chunks_path))
    extraction_by_chunk = _load_extractions(extractions_path)
    docs = []
    entity_chars = 0
    entity_words = 0
    entity_count = 0
    triple_count = 0
    chunks_without_extraction = 0

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        passage = _chunk_content(chunk)
        extraction = extraction_by_chunk.get(chunk_id, {})
        if not extraction:
            chunks_without_extraction += 1
        entities = _entities(extraction)
        triples = _triples(extraction)
        for triple in triples:
            for endpoint in (triple[0], triple[2]):
                if endpoint and endpoint not in entities:
                    entities.append(endpoint)
        entity_count += len(entities)
        triple_count += len(triples)
        entity_chars += sum(len(item) for item in entities)
        entity_words += sum(len(item.split()) for item in entities)
        docs.append(
            {
                "idx": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "file_name": chunk.get("file_name"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "passage": passage,
                "extracted_entities": entities,
                "extracted_triples": triples,
            }
        )

    payload = {
        "docs": docs,
        "source": "signpost_f6_semantic_llm_extractions",
        "dataset": args.dataset,
        "chunks_path": str(chunks_path),
        "extractions_path": str(extractions_path),
        "num_docs": len(docs),
        "num_entities": entity_count,
        "num_triples": triple_count,
        "chunks_without_extraction": chunks_without_extraction,
        "avg_ent_chars": round(entity_chars / entity_count, 4) if entity_count else 0.0,
        "avg_ent_words": round(entity_words / entity_count, 4) if entity_count else 0.0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset": args.dataset,
        "method": METHOD,
        "output": str(output_path),
        "chunks": str(chunks_path),
        "extractions": str(extractions_path),
        "num_docs": len(docs),
        "num_entities": entity_count,
        "num_triples": triple_count,
        "chunks_without_extraction": chunks_without_extraction,
        "note": "Converted from fixed Signpost F6 annotations; no rechunking or entity/relation re-extraction was run.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_extractions(path: Path) -> dict[str, dict[str, Any]]:
    by_chunk: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        chunk_id = str(row.get("chunk_id") or "")
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
        if chunk_id:
            by_chunk[chunk_id] = extraction
    return by_chunk


def _chunk_content(chunk: dict[str, Any]) -> str:
    for key in ("content", "text", "passage"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entities(extraction: dict[str, Any]) -> list[str]:
    entities = extraction.get("entities") if isinstance(extraction.get("entities"), list) else []
    names = []
    seen = set()
    for item in entities:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name") or "").split())
        key = name.lower()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def _triples(extraction: dict[str, Any]) -> list[list[str]]:
    relations = extraction.get("relations") if isinstance(extraction.get("relations"), list) else []
    triples = []
    seen = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        source = " ".join(str(rel.get("source") or "").split())
        target = " ".join(str(rel.get("target") or "").split())
        predicate = _relation_label(rel)
        if not source or not target:
            continue
        key = (source.lower(), predicate.lower(), target.lower())
        if key not in seen:
            triples.append([source, predicate, target])
            seen.add(key)
    return triples


def _relation_label(rel: dict[str, Any]) -> str:
    keywords = rel.get("keywords")
    if isinstance(keywords, list) and keywords:
        label = ", ".join(str(item).strip() for item in keywords[:3] if str(item).strip())
        if label:
            return label
    for key in ("relation", "predicate", "description"):
        value = str(rel.get(key) or "").strip()
        if value:
            return value
    return "related_to"


if __name__ == "__main__":
    raise SystemExit(main())
