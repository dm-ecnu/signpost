from __future__ import annotations

"""Summarizers used by F7 structure view and RAPTOR nodes."""

from typing import Protocol

from signpost.llm.client import OpenAICompatibleClient


class Summarizer(Protocol):
    def summarize(self, title: str, texts: list[str], *, max_tokens: int = 512) -> tuple[str, str]:
        ...


class DeterministicSummarizer:
    """Local summarizer for tests and dry-run graph construction."""

    def summarize(self, title: str, texts: list[str], *, max_tokens: int = 512) -> tuple[str, str]:
        joined = " ".join(text.strip().replace("\n", " ") for text in texts if text.strip())
        words = joined.split()
        if len(words) > max_tokens:
            joined = " ".join(words[:max_tokens])
        summary_title = title or (joined[:60] if joined else "Summary")
        content = joined or summary_title
        return summary_title[:120], content


class LLMSummarizer:
    """ECNU/OpenAI-compatible summarizer for production F7 runs."""

    def __init__(self, client: OpenAICompatibleClient | None = None):
        self.client = client or OpenAICompatibleClient()

    def summarize(self, title: str, texts: list[str], *, max_tokens: int = 512) -> tuple[str, str]:
        prompt = (
            "Summarize the following document-tree node for retrieval. "
            "Return strict JSON with fields title and content. "
            "The content should preserve key facts and cite no unsupported details.\n\n"
            f"Node title: {title}\n\nTexts:\n" + "\n\n---\n\n".join(texts)
        )
        response = self.client.chat(
            [
                {"role": "system", "content": "You write concise hierarchical retrieval summaries as JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        import json

        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start : end + 1])
                return str(data.get("title") or title), str(data.get("content") or "")
            except json.JSONDecodeError:
                pass
        return title, response.strip()


def create_summarizer(name: str) -> Summarizer:
    if name == "deterministic":
        return DeterministicSummarizer()
    if name == "llm":
        return LLMSummarizer()
    raise ValueError(f"Unknown summarizer: {name}")

