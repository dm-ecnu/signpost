from __future__ import annotations

"""Embedding providers for F5 chunk indexing.

`ECNUEmbeddingProvider` is the production path described by the thesis setup.
`HashEmbeddingProvider` is deterministic and local, used for smoke tests and for
benchmark plumbing before paid/network model calls are enabled.
"""

import hashlib
import math
from typing import Protocol

from signpost.llm.client import OpenAICompatibleClient


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ECNUEmbeddingProvider:
    def __init__(self, client: OpenAICompatibleClient | None = None, dimensions: int = 0):
        self.client = client or OpenAICompatibleClient()
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.client.embedding(texts)
        if vectors and not self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors


class HashEmbeddingProvider:
    """Deterministic local embedding for tests and offline smoke runs."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[idx] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


def create_embedding_provider(name: str, *, dimensions: int = 128) -> EmbeddingProvider:
    if name == "ecnu":
        return ECNUEmbeddingProvider()
    if name == "hash":
        return HashEmbeddingProvider(dimensions=dimensions)
    raise ValueError(f"Unknown embedding provider: {name}")

