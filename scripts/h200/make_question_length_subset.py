from __future__ import annotations

"""Create a processed dataset view whose questions are sampled by question length."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


QUESTION_FILES = {
    "questions.jsonl",
    "llm_targets_silver.jsonl",
    "llm_target_units.jsonl",
    "llm_silver_chunks.jsonl",
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def question_id(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or row.get("id") or row.get("qid") or "")


def question_text(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or row.get("input") or "")


def choose_by_length(rows: list[dict[str, Any]], sample_size: int) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda pair: (len(question_text(pair[1])), pair[0]))
    n = len(indexed)
    if sample_size >= n:
        return indexed
    if sample_size <= 0:
        return []
    if sample_size == 1:
        return [indexed[n // 2]]

    selected_positions: list[int] = []
    used: set[int] = set()
    for i in range(sample_size):
        pos = round(i * (n - 1) / (sample_size - 1))
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos not in used:
            used.add(pos)
            selected_positions.append(pos)
    return [indexed[pos] for pos in selected_positions]


def link_or_copy_static_files(source_dir: Path, output_dir: Path, *, copy_static: bool) -> None:
    for path in source_dir.iterdir():
        if path.name in QUESTION_FILES:
            continue
        target = output_dir / path.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if copy_static:
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        else:
            target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def filter_sidecar(source_dir: Path, output_dir: Path, filename: str, selected_ids: set[str]) -> None:
    source = source_dir / filename
    if not source.exists():
        return
    rows = [row for row in read_jsonl(source) if question_id(row) in selected_ids]
    write_jsonl(output_dir / filename, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-static", action="store_true", help="Copy non-question files instead of symlinking them.")
    args = parser.parse_args()

    root = args.root.resolve()
    source_dir = root / "datasets" / "processed" / args.source_dataset
    output_dir = root / "datasets" / "processed" / args.output_dataset
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if output_dir.exists() or output_dir.is_symlink():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = read_jsonl(source_dir / "questions.jsonl")
    selected = choose_by_length(questions, args.sample_size)
    selected_rows = [row for _, row in selected]
    selected_ids = {question_id(row) for row in selected_rows if question_id(row)}
    write_jsonl(output_dir / "questions.jsonl", selected_rows)
    link_or_copy_static_files(source_dir, output_dir, copy_static=args.copy_static)
    for filename in ("llm_targets_silver.jsonl", "llm_target_units.jsonl", "llm_silver_chunks.jsonl"):
        filter_sidecar(source_dir, output_dir, filename, selected_ids)

    manifest = {
        "source_dataset": args.source_dataset,
        "output_dataset": args.output_dataset,
        "sample_size_requested": args.sample_size,
        "source_questions": len(questions),
        "selected_questions": len(selected_rows),
        "sampling": "sort by question character length, then evenly sample across the sorted list",
        "copy_static": args.copy_static,
        "selected": [
            {
                "source_index": original_index,
                "question_id": question_id(row),
                "question_chars": len(question_text(row)),
                "question": question_text(row),
            }
            for original_index, row in selected
        ],
    }
    (output_dir / "question_length_subset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lengths = [item["question_chars"] for item in manifest["selected"]]
    print(json.dumps({k: manifest[k] for k in ("source_dataset", "output_dataset", "source_questions", "selected_questions", "sampling")}, ensure_ascii=False, indent=2))
    if lengths:
        print(f"question_chars min={min(lengths)} median={lengths[len(lengths)//2]} max={max(lengths)}")
    print(f"manifest={output_dir / 'question_length_subset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
