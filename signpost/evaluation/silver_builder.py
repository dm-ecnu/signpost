"""In-repo silver-evidence construction.

Builds the two target files the evaluation reads:

- ``llm_target_units.jsonl``  — per question, the minimal answer-bearing facts
  (*target units*): ``{"question_id", "target_units": [{"unit_id", "text"}]}``
- ``llm_silver_chunks.jsonl`` — per question, the corpus chunks that contain
  evidence for those units: ``{"question_id", "silver_chunks": [{"chunk_id",
  "file_name", "start_line", "end_line", "supports": [unit_id, ...]}]}``

These schemas match what ``signpost.benchmark.final_metrics`` and
``scripts/h200_final_eval_v2.py`` consume (``silver_evidence_chunks.jsonl`` is
the same rows under the targets-dir name).

Construction is two LLM steps over a lexical candidate pool:

1. *Decompose*: given (question, gold answer), the LLM lists the minimal facts
   an answer must establish (target units).
2. *Ground*: given the units and the top-K lexically matching chunks, the LLM
   selects, per chunk, which units that chunk's text actually supports. Chunks
   supporting no unit are dropped.

Provenance note: the numbers in the current paper draft were produced by an
external script on the experiment host (``extract_llm_targets_silver.py``);
this module is the reproducible in-repo reference implementation for re-runs.
To limit self-reference bias, build silver targets with a model that is NOT
the backbone being evaluated (e.g. ``ECNU_REASONING_MODEL``) and keep the
constructed files frozen across all compared systems.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# A "chat function" takes OpenAI-style messages and returns the assistant text.
# scripts/build_silver_evidence.py passes OpenAICompatibleClient.chat; tests
# pass a stub. Keeping the dependency this thin makes the logic fully testable
# offline.
ChatFn = Callable[[list[dict[str, str]]], str]

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "what", "which", "who", "when", "where", "how",
    "why", "does", "do", "did", "that", "this", "it", "its", "with", "by",
    "from", "as", "after", "before",
}


def _tokens(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(text.lower()) if tok not in _STOPWORDS}


def lexical_candidates(
    query_text: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Rank chunks by stopword-filtered token overlap with *query_text*.

    Pure-stdlib candidate generation: recall-oriented, the LLM grounding step
    does the precision work. Returns up to *top_k* chunks with a positive
    overlap, original dicts untouched.
    """
    query_tokens = _tokens(query_text)
    if not query_tokens:
        return []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        content_tokens = _tokens(str(chunk.get("content", "")))
        if not content_tokens:
            continue
        overlap = len(query_tokens & content_tokens)
        if not overlap:
            continue
        # Jaccard-style normalization so huge chunks don't win on length alone.
        score = overlap / (len(query_tokens | content_tokens) ** 0.5)
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:top_k]]


def _extract_json(text: str) -> Any:
    """Parse the first JSON value in an LLM reply (tolerates fences/prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON value found in LLM reply: {text[:200]!r}")


_DECOMPOSE_SYSTEM = (
    "You decompose question-answer pairs into target units: the minimal facts a "
    "correct answer must establish. Each unit is one short, self-contained "
    "statement copied or tightly paraphrased from the answer. Reply with a JSON "
    "array of strings only — no commentary."
)

_GROUND_SYSTEM = (
    "You verify which text chunks contain evidence for which target units. A "
    "chunk supports a unit only if the chunk text itself states or directly "
    "entails the unit — topical similarity is not enough. Reply with a JSON "
    'object mapping chunk_id to the list of supported unit_ids, e.g. '
    '{"c1": ["q1-u0"], "c2": []}. Include every chunk_id given. No commentary.'
)


def build_target_units(
    chat: ChatFn,
    question_id: str,
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    """LLM step 1: decompose the gold answer into target units."""
    reply = chat(
        [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nGold answer: {answer}\n\nList the target units."},
        ]
    )
    parsed = _extract_json(reply)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array of units, got: {type(parsed).__name__}")
    units: list[dict[str, str]] = []
    for text in parsed:
        text = str(text).strip()
        if text:
            units.append({"unit_id": f"{question_id}-u{len(units)}", "text": text})
    return units


def _chunk_excerpt(chunk: dict[str, Any], max_chars: int = 1200) -> str:
    content = str(chunk.get("content", ""))
    return content if len(content) <= max_chars else content[:max_chars] + " …"


def build_silver_chunks(
    chat: ChatFn,
    units: list[dict[str, str]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """LLM step 2: ground units in candidate chunks → silver rows with supports."""
    if not units or not candidates:
        return []
    unit_lines = "\n".join(f"- {unit['unit_id']}: {unit['text']}" for unit in units)
    chunk_lines = "\n\n".join(
        f"chunk_id: {chunk.get('chunk_id')}\n{_chunk_excerpt(chunk)}" for chunk in candidates
    )
    reply = chat(
        [
            {"role": "system", "content": _GROUND_SYSTEM},
            {"role": "user", "content": f"Target units:\n{unit_lines}\n\nChunks:\n\n{chunk_lines}"},
        ]
    )
    parsed = _extract_json(reply)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object chunk_id->unit_ids, got: {type(parsed).__name__}")
    valid_unit_ids = {unit["unit_id"] for unit in units}
    by_id = {str(chunk.get("chunk_id")): chunk for chunk in candidates}
    silver: list[dict[str, Any]] = []
    for chunk_id, supported in parsed.items():
        chunk = by_id.get(str(chunk_id))
        if chunk is None:
            continue
        supports = [str(uid) for uid in (supported if isinstance(supported, list) else []) if str(uid) in valid_unit_ids]
        if not supports:
            continue
        silver.append(
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "file_name": str(chunk.get("file_name", "")),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "supports": supports,
            }
        )
    return silver


def build_for_question(
    chat: ChatFn,
    row: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the (units_row, silver_row) pair for one question row.

    *row* needs ``question_id``, ``question``, and a gold answer under
    ``answer`` (or ``answers``/``gold_answer``).
    """
    question_id = str(row.get("question_id"))
    question = str(row.get("question", ""))
    answer = row.get("answer") or row.get("gold_answer") or row.get("answers") or ""
    if isinstance(answer, list):
        answer = "; ".join(str(item) for item in answer)
    units = build_target_units(chat, question_id, question, str(answer))
    candidates = lexical_candidates(f"{question} {answer} " + " ".join(u["text"] for u in units), chunks, top_k=top_k)
    silver = build_silver_chunks(chat, units, candidates)
    return (
        {"question_id": question_id, "target_units": units},
        {"question_id": question_id, "silver_chunks": silver},
    )
