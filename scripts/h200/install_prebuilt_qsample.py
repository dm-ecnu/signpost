from __future__ import annotations

"""Install locally prebuilt qsample files into an H200 processed dataset view."""

import argparse
import shutil
from pathlib import Path


QSAMPLE_FILES = {
    "questions.jsonl",
    "question_length_subset_manifest.json",
    "llm_targets_silver.jsonl",
    "llm_target_units.jsonl",
    "llm_silver_chunks.jsonl",
}


def replace_path(target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()


def install_static(source_dir: Path, output_dir: Path, *, copy_static: bool) -> None:
    for path in source_dir.iterdir():
        if path.name in QSAMPLE_FILES:
            continue
        target = output_dir / path.name
        replace_path(target)
        if copy_static:
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        else:
            target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--prebuilt-dir", type=Path, required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--copy-static", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-target-silver", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source_dir = root / "datasets" / "processed" / args.source_dataset
    prebuilt_dir = args.prebuilt_dir.resolve()
    output_dir = root / "datasets" / "processed" / args.output_dataset

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not prebuilt_dir.exists():
        raise FileNotFoundError(prebuilt_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")

    replace_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_static(source_dir, output_dir, copy_static=args.copy_static)

    for name in QSAMPLE_FILES:
        src = prebuilt_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    q_lines = count_lines(output_dir / "questions.jsonl")
    if q_lines != args.expected_questions:
        raise SystemExit(f"questions.jsonl has {q_lines} rows; expected {args.expected_questions}")
    if args.require_target_silver:
        for name in ("llm_targets_silver.jsonl", "llm_target_units.jsonl", "llm_silver_chunks.jsonl"):
            lines = count_lines(output_dir / name)
            if lines != args.expected_questions:
                raise SystemExit(f"{name} has {lines} rows; expected {args.expected_questions}")

    print(f"installed output_dataset={args.output_dataset} from prebuilt={prebuilt_dir}")
    print(f"static_source={source_dir} copy_static={args.copy_static}")
    print(f"questions={q_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
