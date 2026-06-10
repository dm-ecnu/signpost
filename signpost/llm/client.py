from __future__ import annotations

"""F1 ECNU/OpenAI-compatible model client.

The implementation keeps chat, embedding, and rerank as separate methods because
later benchmarks need to time and count them independently.  It uses only the
standard library HTTP stack in this first pass so the research code does not
depend on old project provider abstractions.
"""

import json
import os
import time
import urllib.error
import urllib.request
import socket
from dataclasses import dataclass
from typing import Any

from signpost.config.settings import load_settings


@dataclass(frozen=True)
class LLMConfig:
    api_base: str
    embedding_api_base: str
    api_key: str
    embedding_api_key: str
    chat_model: str
    reasoning_model: str
    embedding_model: str
    rerank_model: str


def load_llm_config() -> LLMConfig:
    settings = load_settings()
    env = settings.env
    return LLMConfig(
        api_base=os.environ.get("ECNU_API_BASE") or os.environ.get("OPENAI_API_BASE") or env.get("ECNU_API_BASE") or env.get("OPENAI_API_BASE", ""),
        embedding_api_base=os.environ.get("ECNU_EMBEDDING_API_BASE")
        or os.environ.get("OPENAI_EMBEDDING_API_BASE")
        or env.get("ECNU_EMBEDDING_API_BASE")
        or env.get("OPENAI_EMBEDDING_API_BASE", ""),
        api_key=os.environ.get("ECNU_API_KEY") or os.environ.get("OPENAI_API_KEY") or env.get("ECNU_API_KEY") or env.get("OPENAI_API_KEY", ""),
        embedding_api_key=os.environ.get("ECNU_EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_EMBEDDING_API_KEY")
        or env.get("ECNU_EMBEDDING_API_KEY")
        or env.get("OPENAI_EMBEDDING_API_KEY", ""),
        chat_model=os.environ.get("ECNU_CHAT_MODEL") or env.get("ECNU_CHAT_MODEL", "ecnu-plus"),
        reasoning_model=os.environ.get("ECNU_REASONING_MODEL") or env.get("ECNU_REASONING_MODEL", "ecnu-max"),
        embedding_model=os.environ.get("ECNU_EMBEDDING_MODEL") or env.get("ECNU_EMBEDDING_MODEL", "ecnu-embedding-small"),
        rerank_model=os.environ.get("ECNU_RERANK_MODEL") or env.get("ECNU_RERANK_MODEL", "ecnu-rerank"),
    )


class OpenAICompatibleClient:
    """Small client for OpenAI-compatible chat, embedding, and rerank APIs."""

    def __init__(self, config: LLMConfig | None = None, timeout: int = 60):
        self.config = config or load_llm_config()
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, thinking: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": model or self.config.chat_model,
            "messages": messages,
        }
        if thinking:
            payload["thinking"] = {"type": "enabled"}
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def embedding(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        data = self._post(
            "/embeddings",
            {"model": model or self.config.embedding_model, "input": texts},
            api_base=self.config.embedding_api_base or self.config.api_base,
            api_key=self.config.embedding_api_key or self.config.api_key,
        )
        return [item["embedding"] for item in data.get("data", [])]

    def rerank(self, query: str, documents: list[str], *, model: str | None = None) -> list[float]:
        data = self._post("/rerank", {"model": model or self.config.rerank_model, "query": query, "documents": documents})
        results = data.get("results") or data.get("data") or []
        return [float(item.get("relevance_score", item.get("score", 0.0))) for item in results]

    def _post(self, suffix: str, payload: dict[str, Any], *, api_base: str | None = None, api_key: str | None = None) -> dict[str, Any]:
        base = api_base or self.config.api_base
        key = api_key or self.config.api_key
        if not base:
            raise ValueError("ECNU_API_BASE or OPENAI_API_BASE is required")
        if not key:
            raise ValueError("ECNU_API_KEY or OPENAI_API_KEY is required")
        url = _join_openai_url(base, suffix)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        retries = max(1, int(os.environ.get("LLM_RETRIES", "3") or 3))
        retry_sleep = max(0.0, float(os.environ.get("RETRY_SLEEP", "5") or 5))
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {error_text}") from exc
                last_error = RuntimeError(f"LLM request failed: HTTP {exc.code}: {error_text}")
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                last_error = exc
            if attempt >= retries:
                break
            print(f"[llm-client] retry={attempt}/{retries} url={url} error={last_error}", flush=True)
            time.sleep(retry_sleep)
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last_error}") from last_error


def _join_openai_url(api_base: str, suffix: str) -> str:
    base = api_base.rstrip("/")
    normalized_suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    if base.endswith(normalized_suffix):
        return base
    return base + normalized_suffix
