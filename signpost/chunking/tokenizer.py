from __future__ import annotations

"""Lightweight token counting for F4 chunk budgets.

The thesis reports token budgets, but F4 should not force a heavyweight
transformers download.  This counter is deterministic and conservative enough
for splitting; later indexing can replace it with provider-specific tokenizers.
"""

import re


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u4e00-\u9fff]")


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))

