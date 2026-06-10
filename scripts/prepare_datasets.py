#!/usr/bin/env python3
"""Download and normalize datasets before document parsing.

This script prepares the pre-parsing data layer:

    datasets/raw/<dataset>/...
    datasets/processed/<dataset>/raw_corpus.jsonl
    datasets/processed/<dataset>/questions.jsonl

It does not perform document parsing, chunking, embedding, or indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"


ULTRADOMAIN_FILES = {
    "agriculture": "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/agriculture.jsonl",
    "legal": "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/legal.jsonl",
    "mix": "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/mix.jsonl",
}

GRAPHRAG_BENCH_FILES = {
    "medical_corpus": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/medical.json",
    "medical_questions": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/medical_questions.json",
    "novel_corpus": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/novel.json",
    "novel_questions": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/novel_questions.json",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log(f"[skip] {dest}")
        return
    log(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        last_size = -1
        for _ in range(40):
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "45",
                    "-C",
                    "-",
                    "-o",
                    str(tmp),
                    url,
                ],
                check=False,
            )
            size = tmp.stat().st_size if tmp.exists() else 0
            if size == last_size:
                break
            last_size = size
            # Try parsing small JSON files as an early completion check.
            if dest.suffix == ".json" and size > 0:
                try:
                    read_json(tmp)
                    break
                except Exception:
                    pass
            if dest.suffix == ".jsonl" and size > 0:
                # JSONL files are large; curl returns success when complete.
                pass
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise subprocess.CalledProcessError(1, "curl")
        if dest.suffix == ".json":
            read_json(tmp)
    except (FileNotFoundError, subprocess.CalledProcessError):
        with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    tmp.replace(dest)
    log(f"[saved] {dest} ({dest.stat().st_size} bytes)")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    # Some UltraDomain cells are JSON-encoded strings such as "\"\n# title...".
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
        except Exception:
            pass
    return text.replace("\r\n", "\n").replace("\r", "\n")


def stable_hash(text: str, prefix: str = "") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [clean_text(v) for v in parsed]
            except Exception:
                pass
        return [clean_text(s)]
    return [clean_text(value)]


def prepare_ultradomain(dataset: str, force_download: bool = False) -> dict[str, Any]:
    raw_path = RAW_DIR / "ultradomain" / f"{dataset}.jsonl"
    url = ULTRADOMAIN_FILES.get(dataset)
    if url:
        download(url, raw_path, force=force_download)
    elif not raw_path.exists():
        raise FileNotFoundError(f"No UltraDomain raw file for dataset={dataset}: {raw_path}")

    processed_dir = PROCESSED_DIR / dataset
    raw_corpus_path = processed_dir / "raw_corpus.jsonl"
    questions_path = processed_dir / "questions.jsonl"

    docs_by_id: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    row_count = 0

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_count += 1
            row = json.loads(line)
            label = str(row.get("label") or row.get("dataset") or dataset).lower()
            context = clean_text(row.get("context"))
            context_id = clean_text(row.get("context_id")) or stable_hash(context, prefix=f"{dataset}_doc_")
            title = ""
            authors = ""
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            title = clean_text(meta.get("title")) if meta else ""
            authors = clean_text(meta.get("authors")) if meta else ""
            file_title = title or context_id
            safe_title = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in file_title)[:80].strip("_")
            file_name = f"{safe_title or context_id}.txt"

            if context_id not in docs_by_id:
                docs_by_id[context_id] = {
                    "doc_id": context_id,
                    "file_name": file_name,
                    "source_path": str(raw_path.relative_to(ROOT)),
                    "source_format": "jsonl_context",
                    "text": context,
                    "metadata": {
                        "dataset": dataset,
                        "source_dataset": "TommyChien/UltraDomain",
                        "label": label,
                        "title": title,
                        "authors": authors,
                        "length": row.get("length"),
                    },
                }

            answers = normalize_answers(row.get("answers"))
            qid = clean_text(row.get("_id")) or stable_hash(clean_text(row.get("input")) + context_id, prefix=f"{dataset}_q_")
            questions.append(
                {
                    "question_id": qid,
                    "question": clean_text(row.get("input")),
                    "answer": answers[0] if answers else "",
                    "answers": answers,
                    "rationale": "",
                    "doc_ids": [context_id],
                    "metadata": {
                        "dataset": dataset,
                        "source_dataset": "TommyChien/UltraDomain",
                        "label": label,
                        "context_id": context_id,
                        "raw_id": row.get("_id"),
                        "title": title,
                    },
                }
            )

    docs = list(docs_by_id.values())
    write_jsonl(raw_corpus_path, docs)
    write_jsonl(questions_path, questions)
    return {
        "dataset": dataset,
        "source": "TommyChien/UltraDomain",
        "raw_file": str(raw_path.relative_to(ROOT)),
        "rows": row_count,
        "documents": len(docs),
        "questions": len(questions),
        "raw_corpus": str(raw_corpus_path.relative_to(ROOT)),
        "questions_file": str(questions_path.relative_to(ROOT)),
    }


def extract_graphrag_corpus_items(data: Any) -> list[tuple[str, str, dict[str, Any]]]:
    items: list[tuple[str, str, dict[str, Any]]] = []

    def visit(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            text_keys = ["text", "content", "context", "document", "passage"]
            text = ""
            for key in text_keys:
                if isinstance(obj.get(key), str) and obj.get(key, "").strip():
                    text = obj[key]
                    break
            if text:
                doc_id = clean_text(
                    obj.get("id")
                    or obj.get("doc_id")
                    or obj.get("source")
                    or obj.get("corpus_name")
                    or path
                    or stable_hash(text, "gbench_doc_")
                )
                items.append((doc_id, clean_text(text), obj))
                return
            for key, value in obj.items():
                visit(value, f"{path}/{key}" if path else str(key))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                visit(value, f"{path}/{i}" if path else str(i))

    visit(data)
    # Fallback: GraphRAG-Bench corpus may be a mapping id -> raw text.
    if not items and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                items.append((clean_text(key), clean_text(value), {"id": key}))
    return items


def extract_graphrag_questions(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("data", "questions", "rows"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    else:
        rows = []

    questions: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        q = clean_text(row.get("question") or row.get("query") or row.get("input"))
        if not q:
            continue
        qid = clean_text(row.get("id") or row.get("_id") or row.get("question_id")) or f"gbench_q_{i:05d}"
        evidence = row.get("evidence") or row.get("contexts") or []
        if isinstance(evidence, str):
            evidence_list = [evidence]
        elif isinstance(evidence, list):
            evidence_list = [clean_text(v) for v in evidence]
        else:
            evidence_list = [clean_text(evidence)] if evidence else []
        source = clean_text(row.get("source"))
        questions.append(
            {
                "question_id": qid,
                "question": q,
                "answer": clean_text(row.get("answer")),
                "answers": [clean_text(row.get("answer"))] if row.get("answer") is not None else [],
                "rationale": "\n".join(evidence_list),
                "doc_ids": [source] if source else [],
                "metadata": {
                    "source_dataset": "GraphRAG-Bench/GraphRAG-Bench",
                    "source": source,
                    "question_type": row.get("question_type"),
                    "evidence": evidence,
                    "raw": row,
                },
            }
        )
    return questions


def prepare_graphrag_bench(config: str, force_download: bool = False) -> dict[str, Any]:
    if config not in {"medical", "novel"}:
        raise ValueError("GraphRAG-Bench config must be 'medical' or 'novel'")
    raw_subdir = RAW_DIR / "graphrag-bench"
    corpus_key = f"{config}_corpus"
    questions_key = f"{config}_questions"
    corpus_raw = raw_subdir / f"{config}_corpus.json"
    questions_raw = raw_subdir / f"{config}_questions.json"
    download(GRAPHRAG_BENCH_FILES[corpus_key], corpus_raw, force=force_download)
    download(GRAPHRAG_BENCH_FILES[questions_key], questions_raw, force=force_download)

    dataset = f"graphrag-bench-{config}"
    processed_dir = PROCESSED_DIR / dataset
    raw_corpus_path = processed_dir / "raw_corpus.jsonl"
    questions_path = processed_dir / "questions.jsonl"

    corpus_data = read_json(corpus_raw)
    corpus_items = extract_graphrag_corpus_items(corpus_data)
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc_id, text, raw in corpus_items:
        if not text:
            continue
        if doc_id in seen:
            doc_id = f"{doc_id}_{stable_hash(text)}"
        seen.add(doc_id)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in doc_id)[:80].strip("_")
        docs.append(
            {
                "doc_id": doc_id,
                "file_name": f"{safe_name or stable_hash(text)}.txt",
                "source_path": str(corpus_raw.relative_to(ROOT)),
                "source_format": "json",
                "text": text,
                "metadata": {
                    "dataset": dataset,
                    "source_dataset": "GraphRAG-Bench/GraphRAG-Bench",
                    "config": config,
                    "raw": raw,
                },
            }
        )

    question_data = read_json(questions_raw)
    questions = extract_graphrag_questions(question_data)
    for q in questions:
        q["metadata"]["dataset"] = dataset
        q["metadata"]["config"] = config

    write_jsonl(raw_corpus_path, docs)
    write_jsonl(questions_path, questions)
    return {
        "dataset": dataset,
        "source": "GraphRAG-Bench/GraphRAG-Bench",
        "raw_files": [str(corpus_raw.relative_to(ROOT)), str(questions_raw.relative_to(ROOT))],
        "documents": len(docs),
        "questions": len(questions),
        "raw_corpus": str(raw_corpus_path.relative_to(ROOT)),
        "questions_file": str(questions_path.relative_to(ROOT)),
    }


def validate_processed(dataset: str) -> dict[str, Any]:
    processed_dir = PROCESSED_DIR / dataset
    raw_corpus = processed_dir / "raw_corpus.jsonl"
    questions_file = processed_dir / "questions.jsonl"
    if not raw_corpus.exists():
        raise FileNotFoundError(raw_corpus)
    if not questions_file.exists():
        raise FileNotFoundError(questions_file)
    doc_ids: set[str] = set()
    docs = 0
    with raw_corpus.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            for key in ("doc_id", "file_name", "source_format", "metadata"):
                if key not in row:
                    raise ValueError(f"{raw_corpus}:{line_no} missing {key}")
            if not row.get("source_path") and not row.get("text"):
                raise ValueError(f"{raw_corpus}:{line_no} needs source_path or text")
            doc_ids.add(row["doc_id"])
            docs += 1
    questions = 0
    with questions_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            for key in ("question_id", "question", "metadata"):
                if key not in row:
                    raise ValueError(f"{questions_file}:{line_no} missing {key}")
            questions += 1
    return {"dataset": dataset, "documents": docs, "questions": questions, "known_doc_ids": len(doc_ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare datasets before document parsing")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["agriculture", "legal", "mix", "graphrag-bench-medical", "graphrag-bench-novel"],
        help="Datasets to prepare",
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download raw files")
    parser.add_argument("--validate-only", action="store_true", help="Only validate processed files")
    args = parser.parse_args()

    ensure_dirs()
    summaries: list[dict[str, Any]] = []

    for dataset in args.datasets:
        if args.validate_only:
            summaries.append(validate_processed(dataset))
            continue
        if dataset in ULTRADOMAIN_FILES or (RAW_DIR / "ultradomain" / f"{dataset}.jsonl").exists():
            summaries.append(prepare_ultradomain(dataset, force_download=args.force_download))
        elif dataset == "graphrag-bench-medical":
            summaries.append(prepare_graphrag_bench("medical", force_download=args.force_download))
        elif dataset == "graphrag-bench-novel":
            summaries.append(prepare_graphrag_bench("novel", force_download=args.force_download))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    manifest = {
        "description": "Pre-parsing dataset preparation outputs.",
        "datasets": summaries,
        "notes": [
            "UltraDomain rows are converted by deduplicating context_id into documents.",
            "GraphRAG-Bench public repository currently exposes medical and novel configs.",
            "The thesis G-Bench computer-science subset is not present in the provided HuggingFace splits; add it later if the raw source is provided.",
        ],
    }
    manifest_path = DATASETS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
