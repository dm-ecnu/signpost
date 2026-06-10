#!/usr/bin/env python3
"""Build random QA answer samples for side-by-side manual inspection."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525")

SIGNPOST_METHODS = [
    "signpost.full",
    "signpost.full_rerank_v1",
    "signpost.no_semantic_cues",
    "signpost.no_provenance_cues",
    "signpost.no_vertical_cues",
    "signpost.no_horizontal_cues",
    "signpost.no_online",
    "signpost.no_offline",
    "signpost.full_rerank_v1.no_semantic_cues",
    "signpost.full_rerank_v1.no_provenance_cues",
    "signpost.full_rerank_v1.no_vertical_cues",
    "signpost.full_rerank_v1.no_horizontal_cues",
    "signpost.full_rerank_v1.no_online",
    "signpost.full_rerank_v1.no_offline",
]

BASELINE_METHODS = [
    "linearrag",
    "hiprag",
    "cluerag",
    "cluerag_prompt_normalized",
    "hybrid_rag",
    "graphrag_r1",
    "agrag",
    "vanilla_llm",
    "vanilla_rag",
]

METHOD_ORDER = SIGNPOST_METHODS + BASELINE_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample QA pairs and write one side-by-side answer file per QA "
            "under <root>/suffer/<dataset>/."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Backup root containing datasets/ and outputs/. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to sample. Default: all datasets under datasets/processed.",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=10,
        help="Number of QA pairs to sample per dataset. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    parser.add_argument(
        "--require-method",
        default="signpost.full",
        help=(
            "Only sample questions that have this prediction method. Use an empty "
            "string to sample from all questions. Default: signpost.full."
        ),
    )
    parser.add_argument(
        "--keep-raw-prediction",
        action="store_true",
        help="Keep full prediction text instead of extracting <answer>...</answer>.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old .txt sample files in each target dataset suffer folder first.",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Fail on invalid JSONL rows. Default: skip invalid prediction rows with a warning.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON in {path}:{line_no}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                print(f"WARNING: skipping {message}", file=sys.stderr)
    return rows


def question_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "id", "qid"):
        value = row.get(key)
        if value:
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("raw_id", "question_id", "id"):
            value = metadata.get(key)
            if value:
                return str(value)
    raise KeyError(f"Could not find question id in row keys: {sorted(row)}")


def standard_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer:
        return normalize_one_line(str(answer))
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return normalize_one_line(str(answers[0]))
    return ""


def extract_answer(prediction: Any, keep_raw: bool) -> str:
    if prediction is None:
        return ""
    text = str(prediction)
    if not keep_raw:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
    return normalize_one_line(text)


def normalize_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_predictions(
    dataset_output_dir: Path,
    keep_raw: bool,
    strict_json: bool,
) -> dict[str, dict[str, str]]:
    predictions_dir = dataset_output_dir / "predictions"
    by_method: dict[str, dict[str, str]] = {}
    if not predictions_dir.is_dir():
        return by_method

    for path in sorted(predictions_dir.glob("*.jsonl")):
        method = path.stem
        method_predictions: dict[str, str] = {}
        for row in load_jsonl(path, strict=strict_json):
            qid = question_id(row)
            prediction = row.get("prediction", row.get("answer", row.get("response", "")))
            method_predictions[qid] = extract_answer(prediction, keep_raw)
        by_method[method] = method_predictions
    return by_method


def ordered_methods(available_methods: set[str]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in available_methods]
    extras = sorted(available_methods - set(METHOD_ORDER))
    return ordered + extras


def safe_filename(text: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe[:max_len] or "qa"


def write_sample_file(
    path: Path,
    dataset: str,
    row: dict[str, Any],
    methods: list[str],
    predictions: dict[str, dict[str, str]],
) -> list[str]:
    qid = question_id(row)
    missing: list[str] = []

    lines = [f"GOLD\t{standard_answer(row)}"]
    for method in methods:
        answer = predictions.get(method, {}).get(qid)
        if answer is None:
            missing.append(method)
            continue
        lines.append(f"{method}\t{answer}")

    lines.extend(
        [
            "",
            f"QUESTION\t{normalize_one_line(str(row.get('question', '')))}",
            f"QUESTION_ID\t{qid}",
            f"DATASET\t{dataset}",
        ]
    )
    if missing:
        lines.append(f"MISSING_METHODS\t{', '.join(missing)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return missing


def sample_dataset(
    root: Path,
    dataset: str,
    sample_size: int,
    rng: random.Random,
    require_method: str,
    keep_raw: bool,
    strict_json: bool,
    clean: bool,
) -> dict[str, Any]:
    questions_path = root / "datasets" / "processed" / dataset / "questions.jsonl"
    if not questions_path.is_file():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")

    questions = load_jsonl(questions_path)
    predictions = load_predictions(root / "outputs" / dataset, keep_raw, strict_json)
    methods = ordered_methods(set(predictions))

    if require_method:
        required_qids = set(predictions.get(require_method, {}))
        candidates = [row for row in questions if question_id(row) in required_qids]
    else:
        candidates = questions

    if not candidates:
        return {
            "dataset": dataset,
            "available_questions": len(questions),
            "sampled": 0,
            "methods": methods,
            "warning": f"No candidates after require_method={require_method!r}",
        }

    sample_count = min(sample_size, len(candidates))
    sampled = rng.sample(candidates, sample_count)

    output_dir = root / "suffer" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for old_file in output_dir.glob("*.txt"):
            old_file.unlink()

    manifest_path = output_dir / "manifest.jsonl"
    total_missing: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(sampled, start=1):
            qid = question_id(row)
            filename = f"{index:03d}_{safe_filename(qid)}.txt"
            sample_path = output_dir / filename
            missing = write_sample_file(sample_path, dataset, row, methods, predictions)
            for method in missing:
                total_missing[method] = total_missing.get(method, 0) + 1
            manifest.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "index": index,
                        "question_id": qid,
                        "question": row.get("question", ""),
                        "file": str(sample_path),
                        "missing_methods": missing,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "dataset": dataset,
        "available_questions": len(questions),
        "candidate_questions": len(candidates),
        "sampled": sample_count,
        "methods": methods,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "missing_counts": total_missing,
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    processed_dir = root / "datasets" / "processed"
    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Missing processed datasets dir: {processed_dir}")

    datasets = args.datasets or sorted(path.name for path in processed_dir.iterdir() if path.is_dir())
    rng = random.Random(args.seed)
    summaries = [
        sample_dataset(
            root=root,
            dataset=dataset,
            sample_size=args.sample_size,
            rng=rng,
            require_method=args.require_method,
            keep_raw=args.keep_raw_prediction,
            strict_json=args.strict_json,
            clean=args.clean,
        )
        for dataset in datasets
    ]

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
