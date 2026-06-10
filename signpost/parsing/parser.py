from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signpost.chunking.tokenizer import count_tokens
from signpost.config.context import resolve_project_path
from signpost.parsing.normalizer import normalize_line, normalize_text


"""Core F3.5 parser.

Input rows come from F3 `raw_corpus.jsonl`.  The parser chooses `row["text"]`
when present, otherwise reads `row["source_path"]`.  It emits the canonical
`documents.jsonl` object consumed by F4 chunking and later source-location tools.
"""


_TEXT_KEYS = ("text", "content", "context", "document", "passage")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", flags=re.IGNORECASE | re.DOTALL)
_DEFAULT_MAX_SOURCE_LINE_TOKENS = 384


@dataclass(frozen=True)
class ParsedDocument:
    """Container for one normalized document row."""

    payload: dict[str, Any]


def parse_raw_corpus_row(row: dict[str, Any]) -> ParsedDocument:
    """Convert one F3 raw corpus row into one F3.5 parsed document."""

    _require(row, "doc_id")
    _require(row, "file_name")
    _require(row, "source_format")

    raw_text = _load_text(row)
    normalized = normalize_text(raw_text)

    max_source_line_tokens = _max_source_line_tokens()
    source_rows: list[tuple[int, str, list[str]]] = []
    split_original_lines = 0
    synthetic_line_count = 0
    for original_line_no, raw_line in enumerate(normalized.split("\n"), start=1):
        line = normalize_line(raw_line)
        if not line:
            continue
        segments = _split_source_line(line, max_tokens=max_source_line_tokens)
        if len(segments) > 1:
            split_original_lines += 1
            synthetic_line_count += len(segments)
        source_rows.append((original_line_no, line, segments))

    use_synthetic_line_numbers = split_original_lines > 0
    lines: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    next_line_no = 1
    for original_line_no, line, segments in source_rows:
        if use_synthetic_line_numbers:
            for segment_index, segment in enumerate(segments, start=1):
                line_no = next_line_no
                next_line_no += 1
                item: dict[str, Any] = {"line_no": line_no, "text": segment}
                if len(segments) > 1:
                    item["metadata"] = {
                        "original_line_no": original_line_no,
                        "segment_index": segment_index,
                        "segment_count": len(segments),
                        "source_line_split": True,
                    }
                lines.append(item)
                placeholders.extend(_scan_placeholders(segment, line_no))
        else:
            # Keep the original line number even when empty lines are filtered out.
            # Later Signpost source citations depend on this mapping.
            lines.append({"line_no": original_line_no, "text": line})
            placeholders.extend(_scan_placeholders(line, original_line_no))

    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("source_format", row["source_format"])
    if split_original_lines:
        metadata["source_line_splitting"] = {
            "policy": "token_budget_synthetic_lines",
            "max_tokens": max_source_line_tokens,
            "original_lines_split": split_original_lines,
            "synthetic_lines_from_split": synthetic_line_count,
        }

    payload = {
        "doc_id": row["doc_id"],
        "file_name": row["file_name"],
        "source_path": row.get("source_path"),
        "text": "\n".join(item["text"] for item in lines),
        "lines": lines,
        "placeholders": placeholders,
        "metadata": metadata,
    }
    return ParsedDocument(payload)


def _max_source_line_tokens() -> int:
    value = os.environ.get("SIGNPOST_PARSE_MAX_LINE_TOKENS")
    if value is None or not value.strip():
        return _DEFAULT_MAX_SOURCE_LINE_TOKENS
    try:
        return int(value)
    except ValueError:
        return _DEFAULT_MAX_SOURCE_LINE_TOKENS


def _split_source_line(line: str, *, max_tokens: int) -> list[str]:
    if max_tokens <= 0 or count_tokens(line) <= max_tokens:
        return [line]
    words = line.split()
    if not words:
        return [line]
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = max(1, count_tokens(word))
        if current and current_tokens + word_tokens > max_tokens:
            segments.append(" ".join(current))
            current = []
            current_tokens = 0
        if word_tokens > max_tokens:
            if current:
                segments.append(" ".join(current))
                current = []
                current_tokens = 0
            segments.extend(_split_oversized_word(word, max_tokens=max_tokens))
            continue
        current.append(word)
        current_tokens += word_tokens
    if current:
        segments.append(" ".join(current))
    return segments or [line]


def _split_oversized_word(word: str, *, max_tokens: int) -> list[str]:
    if max_tokens <= 0:
        return [word]
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for char in word:
        char_tokens = max(1, count_tokens(char))
        if current and current_tokens + char_tokens > max_tokens:
            segments.append("".join(current))
            current = []
            current_tokens = 0
        current.append(char)
        current_tokens += char_tokens
    if current:
        segments.append("".join(current))
    return segments


def _require(row: dict[str, Any], key: str) -> None:
    if not row.get(key):
        raise ValueError(f"raw_corpus row missing required field: {key}")


def _load_text(row: dict[str, Any]) -> str:
    """Load document text from the inline `text` field or registered path."""

    text = row.get("text")
    if isinstance(text, str) and text:
        return text
    source_path = row.get("source_path")
    if not source_path:
        raise ValueError(f"{row.get('doc_id', '<unknown>')} needs text or source_path")
    path = resolve_project_path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    source_format = str(row.get("source_format") or path.suffix.lstrip(".")).lower()
    if source_format in {"txt", "md", "markdown"}:
        return path.read_text(encoding="utf-8")
    if source_format == "json":
        return _extract_text_from_json(json.loads(path.read_text(encoding="utf-8")))
    if source_format in {"jsonl", "jsonl_context"}:
        return _extract_text_from_jsonl(path)
    raise ValueError(f"unsupported source_format={source_format!r} for {row.get('doc_id')}")


def _extract_text_from_json(data: Any) -> str:
    """Best-effort text extraction for JSON corpus records."""

    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in _TEXT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def _extract_text_from_jsonl(path: Path) -> str:
    """Join text-bearing JSONL rows into one document string."""

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(_extract_text_from_json(obj))
    return "\n\n".join(texts)


def _scan_placeholders(line: str, line_no: int) -> list[dict[str, Any]]:
    """Record non-plain-text markers without trying to interpret them yet."""

    found: list[dict[str, Any]] = []
    for pattern, item_type in ((_HTML_TABLE, "table"), (_MARKDOWN_IMAGE, "image"), (_MARKDOWN_LINK, "link")):
        for idx, match in enumerate(pattern.finditer(line), start=1):
            found.append(
                {
                    "placeholder": f"[{item_type}_{line_no}_{idx}]",
                    "type": item_type,
                    "line_no": line_no,
                    "raw": match.group(0),
                }
            )
    return found
