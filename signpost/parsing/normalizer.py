from __future__ import annotations

import re
import unicodedata


"""Text normalization for F3.5 document parsing.

The goal is not aggressive cleaning.  We normalize Unicode and common typography
so matching is stable, but we keep original line numbers by normalizing one line
at a time after splitting.
"""


_SPECIAL_SPACES = re.compile(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]")
_INNER_SPACES = re.compile(r"[ \t]{2,}")

_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)


def normalize_text(text: str) -> str:
    """Normalize whole-document text before line splitting."""

    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPECIAL_SPACES.sub(" ", text)
    text = text.translate(_PUNCT_TRANSLATION)
    return text


def normalize_line(line: str) -> str:
    """Normalize a single line while preserving its original line number."""

    line = normalize_text(line)
    line = _INNER_SPACES.sub(" ", line)
    return line.strip()
