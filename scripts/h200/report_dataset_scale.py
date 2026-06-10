from __future__ import annotations

"""Report processed dataset document/question/chunk scale for H200 runs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_len(row: dict[str, Any]) -> int:
    return len(str(row.get("text") or row.get("content") or ""))


def q_text(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or row.get("input") or "")


def report_dataset(root: Path, dataset: str, *, top_docs: int) -> None:
    processed = root / "datasets" / "processed" / dataset
    out_dir = root / "outputs" / dataset / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = read_jsonl(processed / "documents.jsonl")
    chunks = read_jsonl(processed / "chunks.jsonl")
    questions = read_jsonl(processed / "questions.jsonl")

    doc_stats: dict[str, dict[str, Any]] = {}
    for row in documents:
        doc_id = str(row.get("doc_id") or row.get("id") or row.get("file_name") or "")
        if not doc_id:
            continue
        stat = doc_stats.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "files": set(),
                "lines": 0,
                "document_rows": 0,
                "chars": 0,
                "approx_words": 0,
                "chunks": 0,
            },
        )
        stat["document_rows"] += 1
        stat["lines"] += 1
        stat["chars"] += text_len(row)
        stat["approx_words"] += len(str(row.get("text") or row.get("content") or "").split())
        if row.get("file_name"):
            stat["files"].add(str(row.get("file_name")))

    chunks_by_doc = Counter(str(row.get("doc_id") or "") for row in chunks)
    for doc_id, count in chunks_by_doc.items():
        if not doc_id:
            continue
        stat = doc_stats.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "files": set(),
                "lines": 0,
                "document_rows": 0,
                "chars": 0,
                "approx_words": 0,
                "chunks": 0,
            },
        )
        stat["chunks"] = count

    q_lengths = [len(q_text(row)) for row in questions]
    q_doc_refs = defaultdict(int)
    for row in questions:
        for doc_id in row.get("doc_ids") or []:
            q_doc_refs[str(doc_id)] += 1

    csv_path = out_dir / "dataset_doc_size_report.csv"
    rows = sorted(doc_stats.values(), key=lambda item: (int(item["chars"]), str(item["doc_id"])), reverse=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "files", "lines", "document_rows", "chars", "approx_words", "chunks", "question_refs"],
        )
        writer.writeheader()
        for stat in rows:
            writer.writerow(
                {
                    **{key: stat[key] for key in ("doc_id", "lines", "document_rows", "chars", "approx_words", "chunks")},
                    "files": "|".join(sorted(stat["files"])),
                    "question_refs": q_doc_refs.get(str(stat["doc_id"]), 0),
                }
            )

    summary = {
        "dataset": dataset,
        "documents": len(doc_stats),
        "document_rows": len(documents),
        "chunks": len(chunks),
        "questions": len(questions),
        "total_chars": sum(int(stat["chars"]) for stat in doc_stats.values()),
        "max_doc_chars": max((int(stat["chars"]) for stat in doc_stats.values()), default=0),
        "question_length_chars": {
            "min": min(q_lengths) if q_lengths else 0,
            "max": max(q_lengths) if q_lengths else 0,
            "avg": round(sum(q_lengths) / len(q_lengths), 2) if q_lengths else 0,
        },
        "doc_size_csv": str(csv_path),
    }
    summary_path = out_dir / "dataset_scale_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"===== {dataset} =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if top_docs < 0:
        shown_rows = rows
        print_docs_label = "all_docs_by_size_desc"
    else:
        shown_rows = rows[:top_docs]
        print_docs_label = f"largest_docs_top_{top_docs}"
    print(f"{print_docs_label}:")
    for stat in shown_rows:
        print(
            f"  doc_id={stat['doc_id']} chars={stat['chars']} words~={stat['approx_words']} "
            f"lines={stat['lines']} chunks={stat['chunks']} question_refs={q_doc_refs.get(str(stat['doc_id']), 0)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--top-docs", type=int, default=-1, help="Number of largest docs to print; use -1 to print all docs. CSV output is always full.")
    args = parser.parse_args()

    root = args.root.resolve()
    for dataset in args.dataset:
        report_dataset(root, dataset, top_docs=args.top_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
