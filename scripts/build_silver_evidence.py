"""Build silver-evidence targets for a dataset (in-repo, resumable).

Produces ``llm_target_units.jsonl`` and ``llm_silver_chunks.jsonl`` in
``--out-dir`` — the files ``scripts/h200_final_eval_v2.py`` reads. For
``signpost.benchmark.final_metrics`` copy/rename ``llm_silver_chunks.jsonl``
to ``silver_evidence_chunks.jsonl`` in the targets dir.

Usage::

    export ECNU_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
    export ECNU_API_KEY=...   # never commit
    python3 scripts/build_silver_evidence.py \
        --questions-jsonl data/processed/questions.jsonl \
        --chunks-jsonl data/processed/chunks.jsonl \
        --out-dir data/processed \
        --model ecnu-max        # use a model != the evaluated backbone

Questions rows need question_id / question / answer (or answers/gold_answer);
chunk rows need chunk_id / file_name / start_line / end_line / content.
Already-built question_ids are skipped, so interrupted runs just re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpost.evaluation.silver_builder import build_for_question  # noqa: E402
from signpost.llm.client import OpenAICompatibleClient  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("question_id")) for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-jsonl", required=True, type=Path)
    parser.add_argument("--chunks-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default=None, help="chat model override (default: ECNU_CHAT_MODEL)")
    parser.add_argument("--top-k", type=int, default=20, help="lexical candidate pool size per question")
    parser.add_argument("--limit", type=int, default=0, help="stop after N questions (0 = all)")
    args = parser.parse_args()

    questions = read_jsonl(args.questions_jsonl)
    chunks = read_jsonl(args.chunks_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.out_dir / "llm_target_units.jsonl"
    silver_path = args.out_dir / "llm_silver_chunks.jsonl"
    done = existing_ids(units_path) & existing_ids(silver_path)

    client = OpenAICompatibleClient()
    chat = (lambda messages: client.chat(messages, model=args.model)) if args.model else client.chat

    built = skipped = failed = 0
    with units_path.open("a", encoding="utf-8") as units_out, silver_path.open("a", encoding="utf-8") as silver_out:
        for row in questions:
            if args.limit and built >= args.limit:
                break
            qid = str(row.get("question_id"))
            if qid in done:
                skipped += 1
                continue
            try:
                units_row, silver_row = build_for_question(chat, row, chunks, top_k=args.top_k)
            except Exception as exc:  # keep going; failed ids re-run on resume
                failed += 1
                print(f"[fail] {qid}: {exc}", file=sys.stderr)
                continue
            units_out.write(json.dumps(units_row, ensure_ascii=False) + "\n")
            silver_out.write(json.dumps(silver_row, ensure_ascii=False) + "\n")
            units_out.flush()
            silver_out.flush()
            built += 1
            print(f"[ok] {qid}: {len(units_row['target_units'])} units, {len(silver_row['silver_chunks'])} silver chunks")

    print(f"done: built={built} skipped={skipped} failed={failed} -> {units_path}, {silver_path}")


if __name__ == "__main__":
    main()
