from __future__ import annotations

"""Create raw UltraDomain document-complete subsets before F3 preparation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path


def create_ultradomain_subset(
    *,
    source_dataset: str,
    target_dataset: str,
    doc_ids: list[str],
    raw_root: Path | None = None,
    output_path: Path | None = None,
    max_questions_per_doc: int | None = None,
) -> dict[str, Any]:
    raw_dir = raw_root or resolve_project_path("datasets/raw/ultradomain")
    source_path = raw_dir / f"{source_dataset}.jsonl"
    target_path = output_path or raw_dir / f"{target_dataset}.jsonl"
    selected = set(doc_ids)
    if not selected:
        raise ValueError("doc_ids must not be empty")
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    kept_rows = []
    seen_per_doc: Counter[str] = Counter()
    source_rows = 0
    for line in source_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        source_rows += 1
        row = json.loads(line)
        context = clean_text(row.get("context"))
        doc_id = clean_text(row.get("context_id")) or stable_hash(context, prefix=f"{source_dataset}_doc_")
        if doc_id not in selected:
            continue
        if max_questions_per_doc is not None and seen_per_doc[doc_id] >= max_questions_per_doc:
            continue
        kept_rows.append(rewrite_row(row, source_dataset=source_dataset, target_dataset=target_dataset, doc_id=doc_id))
        seen_per_doc[doc_id] += 1

    missing = sorted(selected - set(seen_per_doc))
    if missing:
        raise ValueError(f"Selected doc_ids were not found in {source_path}: {missing}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "source_rows": source_rows,
        "rows": len(kept_rows),
        "documents": len(seen_per_doc),
        "questions": len(kept_rows),
        "max_questions_per_doc": max_questions_per_doc,
        "doc_ids": doc_ids,
        "questions_per_doc": dict(seen_per_doc),
    }
    target_path.with_suffix(".selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def rewrite_row(row: dict[str, Any], *, source_dataset: str, target_dataset: str, doc_id: str) -> dict[str, Any]:
    updated = dict(row)
    updated["context_id"] = doc_id
    updated["dataset"] = target_dataset
    updated["label"] = row.get("label") or source_dataset
    meta = dict(updated.get("meta", {}) if isinstance(updated.get("meta"), dict) else {})
    meta.setdefault("source_dataset", f"UltraDomain/{source_dataset}")
    meta["parent_dataset"] = source_dataset
    meta["subset_dataset"] = target_dataset
    updated["meta"] = meta
    return updated


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Create raw UltraDomain document-complete subsets before F3 preparation.")
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--doc-id", action="append", required=True, help="Document id to include. Repeat this option.")
    parser.add_argument("--raw-root")
    parser.add_argument("--output")
    parser.add_argument("--max-questions-per-doc", type=int)
    args = parser.parse_args()

    summary = create_ultradomain_subset(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        doc_ids=args.doc_id,
        raw_root=resolve_project_path(args.raw_root) if args.raw_root else None,
        output_path=resolve_project_path(args.output) if args.output else None,
        max_questions_per_doc=args.max_questions_per_doc,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
