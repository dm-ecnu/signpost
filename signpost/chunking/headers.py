from __future__ import annotations

"""F4 chapter recognition.

The paper describes a dual-path design: short documents can be converted to
Markdown end-to-end, while long documents use iterative section extraction.
This module exposes both LLM paths and a deterministic recognizer.  The
deterministic recognizer is always available for tests and for datasets where a
model key should not be consumed.
"""

import json
import re
from typing import Any

from signpost.chunking.models import Header
from signpost.chunking.tokenizer import count_tokens
from signpost.llm.client import OpenAICompatibleClient


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CHINESE_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万\d]+[章节篇部卷编]\s*[、:：.\-]?\s*(.+)?$")
_CHINESE_SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千万\d]+[条款项节]\s*[、:：.\-]?\s*(.+)?$")
_EN_ARTICLE_RE = re.compile(r"^(ARTICLE|CHAPTER|PART)\s+([IVXLCDM\d]+)\b[.\s:-]*(.*)$", re.IGNORECASE)
_EN_SECTION_RE = re.compile(r"^(Section|SECTION)\s+\d+(\.\d+)*\b[.\s:-]*(.*)$")
_NUMBERED_RE = re.compile(r"^\d+(\.\d+){1,5}\s+.{2,120}$")


def recognize_headers(
    document: dict[str, Any],
    *,
    use_llm: bool = False,
    short_token_threshold: int = 64000,
    window_tokens: int = 4096,
    compression_interval: int = 8,
    client: OpenAICompatibleClient | None = None,
) -> list[Header]:
    """Recognize chapter headers using thesis dual-path logic.

    When `use_llm` is false, this function uses deterministic recognition.  When
    true, short documents call Markdown conversion and long documents call
    iterative extraction.  If an LLM call fails, callers get the exception rather
    than a silent fallback, because benchmark runs must be reproducible.
    """

    token_count = count_tokens(document.get("text", ""))
    if use_llm:
        llm_client = client or OpenAICompatibleClient()
        if token_count <= short_token_threshold:
            return recognize_headers_short_markdown(document, llm_client)
        return recognize_headers_long_iterative(document, llm_client, window_tokens=window_tokens, compression_interval=compression_interval)
    return recognize_headers_deterministic(document)


def recognize_headers_deterministic(document: dict[str, Any]) -> list[Header]:
    """Recognize common Markdown, Chinese, legal, and numbered section titles."""

    headers: list[Header] = []
    lines = document.get("lines") or []
    for item in lines:
        text = str(item.get("text", "")).strip()
        line_no = int(item.get("line_no", 0))
        if not text or line_no < 1:
            continue
        parsed = _parse_heading_line(text)
        if parsed is None:
            continue
        level, title = parsed
        headers.append(Header(title=title, level=level, line_start=line_no, line_end=line_no))
    return _dedupe_headers(headers)


def recognize_headers_short_markdown(document: dict[str, Any], client: OpenAICompatibleClient) -> list[Header]:
    """Short-document path: convert the whole document to Markdown, then parse headings."""

    prompt = (
        "Convert the document into clean Markdown headings and body text. "
        "Preserve all factual text and placeholders. Return only Markdown.\n\n"
        f"Document:\n{document.get('text', '')}"
    )
    markdown = client.chat(
        [
            {"role": "system", "content": "You are a document structure parser."},
            {"role": "user", "content": prompt},
        ]
    )
    headers: list[Header] = []
    for idx, line in enumerate(markdown.splitlines(), start=1):
        match = _MD_HEADING_RE.match(line.strip())
        if match:
            headers.append(Header(title=match.group(2).strip(), level=len(match.group(1)), line_start=idx, line_end=idx))
    return _dedupe_headers(headers)


def recognize_headers_long_iterative(
    document: dict[str, Any],
    client: OpenAICompatibleClient,
    *,
    window_tokens: int,
    compression_interval: int,
) -> list[Header]:
    """Long-document path: scan windows, periodically compress history, merge JSON headers."""

    windows = _line_windows(document.get("lines") or [], max_tokens=window_tokens)
    history_summary = ""
    headers: list[Header] = []
    for idx, window in enumerate(windows, start=1):
        text = "\n".join(f"{line['line_no']}: {line['text']}" for line in window)
        prompt = (
            "Extract section headings from this numbered document fragment. "
            "Return JSON array objects with title, level, line_start, line_end. "
            "Use line numbers from the fragment. Do not invent headings.\n\n"
            f"Previous summary:\n{history_summary}\n\nFragment:\n{text}"
        )
        response = client.chat(
            [
                {"role": "system", "content": "You extract document chapter hierarchy as strict JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        headers.extend(_headers_from_json(response))
        if idx % compression_interval == 0:
            history_summary = client.chat(
                [
                    {"role": "system", "content": "Summarize extracted document structure compactly."},
                    {"role": "user", "content": json.dumps([h.__dict__ for h in headers], ensure_ascii=False)},
                ]
            )
    return _dedupe_headers(headers)


def _parse_heading_line(text: str) -> tuple[int, str] | None:
    md = _MD_HEADING_RE.match(text)
    if md:
        return len(md.group(1)), md.group(2).strip()
    if _CHINESE_CHAPTER_RE.match(text):
        return 1, text
    if _CHINESE_SECTION_RE.match(text):
        marker = text[: min(len(text), 12)]
        if "条" in marker:
            return 3, text
        return 2, text
    en_article = _EN_ARTICLE_RE.match(text)
    if en_article:
        return 1, text
    if _EN_SECTION_RE.match(text):
        return 2, text
    if _NUMBERED_RE.match(text):
        depth = text.split()[0].count(".") + 1
        return min(depth, 6), text
    return None


def _headers_from_json(text: str) -> list[Header]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return []
    data = json.loads(text[start : end + 1])
    headers: list[Header] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        headers.append(
            Header(
                title=title,
                level=max(1, int(row.get("level", 1))),
                line_start=max(1, int(row.get("line_start", row.get("line", 1)))),
                line_end=max(1, int(row.get("line_end", row.get("line_start", 1)))),
            )
        )
    return headers


def _line_windows(lines: list[dict[str, Any]], *, max_tokens: int) -> list[list[dict[str, Any]]]:
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for line in lines:
        line_tokens = max(1, count_tokens(str(line.get("text", ""))))
        if current and current_tokens + line_tokens > max_tokens:
            windows.append(current)
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        windows.append(current)
    return windows


def _dedupe_headers(headers: list[Header]) -> list[Header]:
    seen: set[tuple[int, str]] = set()
    result: list[Header] = []
    for header in sorted(headers, key=lambda item: (item.line_start, item.level)):
        key = (header.line_start, header.title)
        if key in seen:
            continue
        seen.add(key)
        result.append(header)
    return result

