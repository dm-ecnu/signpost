from __future__ import annotations

"""Basic non-LLM evaluation metrics for F16 prediction files."""

from collections import Counter
import re
import string
from typing import Any


def normalize_answer(answer: str) -> str:
    text = answer.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer_from_prediction(prediction: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", prediction, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return re.sub(r"<think>.*?</think>\s*", "", prediction, flags=re.DOTALL | re.IGNORECASE).strip()


def exact_match(gold_answers: list[str], predicted: str) -> float:
    predicted_norm = normalize_answer(predicted)
    return 1.0 if any(normalize_answer(gold) == predicted_norm for gold in gold_answers) else 0.0


def prf(gold: str, predicted: str) -> dict[str, float]:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    if not gold_tokens and not predicted_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_tokens or not predicted_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(gold_tokens) & Counter(predicted_tokens)
    same = sum(common.values())
    if same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = same / len(predicted_tokens)
    recall = same / len(gold_tokens)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_example = []
    skipped = 0
    totals = {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for row in rows:
        prediction = str(row.get("prediction", ""))
        if not prediction.strip():
            skipped += 1
            continue
        predicted = extract_answer_from_prediction(prediction)
        gold_answers = _gold_answers(row.get("answer", ""))
        best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for gold in gold_answers:
            score = prf(gold, predicted)
            if score["f1"] > best["f1"]:
                best = score
        em = exact_match(gold_answers, predicted)
        item = {"question_id": row.get("question_id"), "exact_match": em, **best}
        per_example.append(item)
        totals["exact_match"] += em
        totals["precision"] += best["precision"]
        totals["recall"] += best["recall"]
        totals["f1"] += best["f1"]
    scored = len(per_example)
    averages = {key: (value / scored if scored else 0.0) for key, value in totals.items()}
    return {
        "num_samples": len(rows),
        "num_scored": scored,
        "num_skipped": skipped,
        "metrics": averages,
        "per_example": per_example,
    }


def _gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
