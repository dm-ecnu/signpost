from __future__ import annotations

"""MemGraphRAG adapter over Signpost shared preprocessing artifacts.

This adapter reuses only the shared public corpus preprocessing boundary:
chunks, entities, entity types, and relation observations from
``semantic_llm.extractions.jsonl``. MemGraphRAG-owned schema/fact/passage
memory, fact-to-passage links, dense stores, and PPR retrieval graph are built
inside this baseline.
"""

import argparse
import ast
import copy
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from signpost.baselines.common import (
    BaselineResult,
    build_paths,
    chat_once,
    join_context,
    load_jsonl_list,
    locate_from_chunk,
    question_text,
    run_baseline_batch,
)
from signpost.config.context import resolve_project_path
from signpost.indexing.embedding import create_embedding_provider
from signpost.llm.client import OpenAICompatibleClient


METHOD = "memgraphrag"
SYSTEM_PROMPT = """As an advanced reading comprehension assistant, answer the question in English strictly based on the retrieved passages selected by MemGraphRAG. Your response starts after "Thought: ", where you briefly analyze the question and identify the relevant evidence. Conclude with "Answer: " to present the final response.

Follow these rules:
- Include all necessary details supported by the retrieved passages.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the retrieved passages are insufficient, write exactly: "Insufficient evidence." after "Answer: "."""


@dataclass
class MemSchemaNode:
    idx: int
    content: tuple[str, str, str]
    frequency: int = 0
    fact_indices: list[int] = field(default_factory=list)


@dataclass
class MemFactNode:
    idx: int
    content: tuple[str, str, str]
    schema_idx: int
    passage_indices: list[int] = field(default_factory=list)
    raw_samples: list[list[str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return str(tuple(self.content))


@dataclass
class MemPassageNode:
    idx: int
    chunk_id: str
    content: str
    fact_indices: list[int] = field(default_factory=list)


class ThreeLayerMemory:
    def __init__(self) -> None:
        self.schema_layer: list[MemSchemaNode] = []
        self.fact_layer: list[MemFactNode] = []
        self.passage_layer: list[MemPassageNode] = []
        self._schema_to_idx: dict[tuple[str, str, str], int] = {}
        self._fact_to_idx: dict[tuple[str, str, str], int] = {}
        self._chunk_id_to_idx: dict[str, int] = {}

    def build_from_openie_results(self, data: dict[str, Any]) -> None:
        for doc in data.get("docs", []):
            if not isinstance(doc, dict):
                continue
            chunk_id = str(doc.get("idx") or "")
            passage_text = str(doc.get("passage") or "")
            triple_ont_map = doc.get("extracted_triple_ontology")
            if not isinstance(triple_ont_map, dict) or not triple_ont_map:
                continue
            passage_idx = self._get_or_create_passage(chunk_id, passage_text)
            raw_map = doc.get("raw_triple_map") if isinstance(doc.get("raw_triple_map"), dict) else {}
            for triple_key, ontology in triple_ont_map.items():
                triple = _literal_triple(triple_key)
                if triple is None or not _valid_ontology(ontology):
                    continue
                schema_idx = self._get_or_create_schema(tuple(ontology))
                fact_idx = self._get_or_create_fact(triple, schema_idx)
                if fact_idx not in self.schema_layer[schema_idx].fact_indices:
                    self.schema_layer[schema_idx].fact_indices.append(fact_idx)
                if passage_idx not in self.fact_layer[fact_idx].passage_indices:
                    self.fact_layer[fact_idx].passage_indices.append(passage_idx)
                if fact_idx not in self.passage_layer[passage_idx].fact_indices:
                    self.passage_layer[passage_idx].fact_indices.append(fact_idx)
                raw_sample = raw_map.get(triple_key)
                if (
                    isinstance(raw_sample, list)
                    and len(raw_sample) == 3
                    and raw_sample not in self.fact_layer[fact_idx].raw_samples
                    and len(self.fact_layer[fact_idx].raw_samples) < 5
                ):
                    self.fact_layer[fact_idx].raw_samples.append([str(item) for item in raw_sample])
        for schema in self.schema_layer:
            schema.frequency = len(schema.fact_indices)

    def _get_or_create_schema(self, ontology: tuple[str, str, str]) -> int:
        if ontology in self._schema_to_idx:
            return self._schema_to_idx[ontology]
        idx = len(self.schema_layer)
        self._schema_to_idx[ontology] = idx
        self.schema_layer.append(MemSchemaNode(idx=idx, content=ontology))
        return idx

    def _get_or_create_fact(self, triple: tuple[str, str, str], schema_idx: int) -> int:
        if triple in self._fact_to_idx:
            return self._fact_to_idx[triple]
        idx = len(self.fact_layer)
        self._fact_to_idx[triple] = idx
        self.fact_layer.append(MemFactNode(idx=idx, content=triple, schema_idx=schema_idx))
        return idx

    def _get_or_create_passage(self, chunk_id: str, passage_text: str) -> int:
        if chunk_id in self._chunk_id_to_idx:
            return self._chunk_id_to_idx[chunk_id]
        idx = len(self.passage_layer)
        self._chunk_id_to_idx[chunk_id] = idx
        self.passage_layer.append(MemPassageNode(idx=idx, chunk_id=chunk_id, content=passage_text))
        return idx

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "frequency": node.frequency,
                    "embedding": None,
                    "fact_indices": node.fact_indices,
                }
                for node in self.schema_layer
            ],
            "fact_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "embedding": None,
                    "schema_idx": node.schema_idx,
                    "passage_indices": node.passage_indices,
                    "raw_samples": node.raw_samples,
                }
                for node in self.fact_layer
            ],
            "passage_layer": [
                {
                    "idx": node.idx,
                    "chunk_id": node.chunk_id,
                    "content": node.content,
                    "embedding": None,
                    "fact_indices": node.fact_indices,
                }
                for node in self.passage_layer
            ],
            "stats": {
                "num_schemas": len(self.schema_layer),
                "num_facts": len(self.fact_layer),
                "num_passages": len(self.passage_layer),
            },
        }


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
    score_source: str


class MemGraphRAGIndex:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        extractions: list[dict[str, Any]],
        embedding_provider: str,
        artifact_dir: Path,
        embedding_batch_size: int,
        schema_min_count: int,
        synonymy_edges: bool,
        synonymy_edge_sim_threshold: float,
        synonymy_edge_max_neighbors: int,
    ):
        self.chunks_by_id = {str(item.get("chunk_id")): item for item in chunks if item.get("chunk_id")}
        self.all_chunks = [item for item in chunks if item.get("chunk_id")]
        self.embedding_provider_name = embedding_provider
        self.embedding_provider = create_embedding_provider(embedding_provider)
        self.artifact_dir = artifact_dir
        self.embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self.schema_min_count = max(1, int(schema_min_count or 1))
        self.synonymy_edges = bool(synonymy_edges)
        self.synonymy_edge_sim_threshold = float(synonymy_edge_sim_threshold)
        self.synonymy_edge_max_neighbors = max(1, int(synonymy_edge_max_neighbors or 1))
        self.memory = ThreeLayerMemory()
        self.openie_data: dict[str, Any] = {}
        self.filtered_openie_data: dict[str, Any] = {}
        self.filter_stats: dict[str, Any] = {}
        self.entity_text_by_id: dict[str, str] = {}
        self.entity_id_by_text: dict[str, str] = {}
        self.entity_vectors: list[list[float]] = []
        self.fact_vectors: list[list[float]] = []
        self.passage_vectors: list[list[float]] = []
        self.passage_node_ids: list[str] = []
        self.fact_node_ids: list[str] = []
        self.entity_node_ids: list[str] = []
        self.ent_node_to_num_chunk: dict[str, int] = {}
        self.adj: dict[str, dict[str, float]] = defaultdict(dict)
        self.fact_occurrences: dict[int, int] = defaultdict(int)
        self.offline_embedding_calls = 0
        self.offline_embedding_retries = 0
        self.offline_embedding_failures = 0
        self.offline_embedding_wall_time_seconds = 0.0
        self.synonymy_edges_added = 0
        self._build(extractions)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["embedding_provider"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.embedding_provider = create_embedding_provider(self.embedding_provider_name)

    @classmethod
    def load_cache(cls, artifact_dir: Path, *, embedding_provider: str) -> "MemGraphRAGIndex":
        cache_path = artifact_dir / "index.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"MemGraphRAG index cache not found: {cache_path}")
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError(f"MemGraphRAG index cache has wrong type: {type(index)!r}")
        if index.embedding_provider_name != embedding_provider:
            raise ValueError(
                f"MemGraphRAG index cache embedding_provider={index.embedding_provider_name!r} "
                f"does not match requested {embedding_provider!r}"
            )
        index.artifact_dir = artifact_dir
        index.embedding_provider = create_embedding_provider(index.embedding_provider_name)
        return index

    def save_cache(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with (self.artifact_dir / "index.pkl").open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build(self, extractions: list[dict[str, Any]]) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.openie_data = _convert_public_extractions_to_openie(self.all_chunks, extractions)
        _write_json(self.artifact_dir / "openie_observations.json", self.openie_data)
        self.filtered_openie_data, self.filter_stats = _filter_low_frequency_ontology(
            self.openie_data,
            min_count=self.schema_min_count,
        )
        _write_json(self.artifact_dir / "filtered_openie.json", self.filtered_openie_data)
        self.memory.build_from_openie_results(self.filtered_openie_data)
        self._build_graph()
        self.write_artifacts(status="memory_built")
        self._embed_memory()
        if self.synonymy_edges:
            self._add_synonymy_edges()
        self.write_artifacts(status="ready")

    def _build_graph(self) -> None:
        self.passage_node_ids = [str(item.get("chunk_id")) for item in self.all_chunks if item.get("chunk_id")]
        for chunk_id in self.passage_node_ids:
            self.adj.setdefault(chunk_id, {})

        entity_chunks: dict[str, set[str]] = defaultdict(set)
        for doc in self.filtered_openie_data.get("docs", []):
            if not isinstance(doc, dict):
                continue
            chunk_id = str(doc.get("idx") or "")
            triple_ont_map = doc.get("extracted_triple_ontology")
            if not isinstance(triple_ont_map, dict):
                continue
            entities_in_chunk: set[str] = set()
            for triple_key in triple_ont_map:
                triple = _literal_triple(triple_key)
                if triple is None:
                    continue
                head, _relation, tail = triple
                head_id = _node_id(head, prefix="entity-")
                tail_id = _node_id(tail, prefix="entity-")
                self.entity_text_by_id.setdefault(head_id, head)
                self.entity_text_by_id.setdefault(tail_id, tail)
                self.entity_id_by_text.setdefault(head, head_id)
                self.entity_id_by_text.setdefault(tail, tail_id)
                self._add_edge(head_id, tail_id, 1.0)
                entities_in_chunk.add(head_id)
                entities_in_chunk.add(tail_id)
                fact_idx = self.memory._fact_to_idx.get(triple)
                if fact_idx is not None:
                    self.fact_occurrences[fact_idx] += 1
            for entity_id in entities_in_chunk:
                entity_chunks[entity_id].add(chunk_id)
                self._add_edge(chunk_id, entity_id, 1.0)
        self.ent_node_to_num_chunk = {entity_id: len(chunks) for entity_id, chunks in entity_chunks.items()}
        self.entity_node_ids = sorted(self.entity_text_by_id)
        self.fact_node_ids = [_node_id(fact.text, prefix="fact-") for fact in self.memory.fact_layer]

    def _embed_memory(self) -> None:
        started = time.time()
        entity_texts = [self.entity_text_by_id[node_id] for node_id in self.entity_node_ids]
        fact_texts = [fact.text for fact in self.memory.fact_layer]
        passage_texts = [str(self.chunks_by_id[chunk_id].get("content") or "") for chunk_id in self.passage_node_ids if chunk_id in self.chunks_by_id]
        self.entity_vectors = self._embed_texts(entity_texts, label="entity")
        self.fact_vectors = self._embed_texts(fact_texts, label="fact")
        self.passage_vectors = self._embed_texts(passage_texts, label="passage")
        self.offline_embedding_wall_time_seconds = time.time() - started

    def _embed_texts(self, texts: list[str], *, label: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        total_batches = math.ceil(len(texts) / self.embedding_batch_size)
        started = time.time()
        for batch_index, start in enumerate(range(0, len(texts), self.embedding_batch_size), start=1):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend(self._embed_batch_with_retry(batch, label=label))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = time.time() - started
                print(
                    f"[memgraphrag] embedded {label} batch {batch_index}/{total_batches} "
                    f"vectors={len(vectors)}/{len(texts)} elapsed_seconds={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                self.write_artifacts(status=f"embedding_{label}")
        return vectors

    def _embed_batch_with_retry(self, batch: list[str], *, label: str) -> list[list[float]]:
        retries = max(1, int(os.environ.get("MEMGRAPHRAG_EMBED_RETRIES") or os.environ.get("BASELINE_EMBED_RETRIES", "3")))
        retry_sleep = max(
            0.0,
            float(os.environ.get("MEMGRAPHRAG_EMBED_RETRY_SLEEP") or os.environ.get("BASELINE_EMBED_RETRY_SLEEP", "5")),
        )
        for attempt in range(1, retries + 1):
            try:
                vectors = self.embedding_provider.embed(batch)
                self.offline_embedding_calls += 1
                return vectors
            except Exception as exc:
                self.offline_embedding_failures += 1
                if attempt < retries:
                    self.offline_embedding_retries += 1
                    print(
                        f"[memgraphrag] embedding {label} batch_size={len(batch)} failed "
                        f"attempt={attempt}/{retries}: {exc}; retrying after {retry_sleep:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if retry_sleep:
                        time.sleep(retry_sleep)
                    continue
                if len(batch) == 1:
                    raise
                midpoint = max(1, len(batch) // 2)
                print(
                    f"[memgraphrag] embedding {label} batch_size={len(batch)} failed after {retries} attempts; "
                    f"splitting into {midpoint}+{len(batch) - midpoint}",
                    file=sys.stderr,
                    flush=True,
                )
                return self._embed_batch_with_retry(batch[:midpoint], label=label) + self._embed_batch_with_retry(batch[midpoint:], label=label)
        raise RuntimeError("unreachable embedding retry state")

    def embed_query(self, query: str) -> list[float]:
        return self.embedding_provider.embed([query])[0]

    def _add_edge(self, left: str, right: str, weight: float) -> None:
        if not left or not right or left == right:
            return
        self.adj[left][right] = self.adj[left].get(right, 0.0) + weight
        self.adj[right][left] = self.adj[right].get(left, 0.0) + weight

    def _add_synonymy_edges(self) -> None:
        if len(self.entity_vectors) < 2:
            return
        try:
            import numpy as np
        except Exception as exc:
            print(f"[memgraphrag] numpy unavailable; skipping synonymy edges: {exc}", file=sys.stderr, flush=True)
            return

        vectors = np.asarray(self.entity_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        threshold = self.synonymy_edge_sim_threshold
        added = 0
        for row_idx in range(vectors.shape[0]):
            sims = vectors @ vectors[row_idx]
            candidates = [
                (idx, float(score))
                for idx, score in enumerate(sims.tolist())
                if idx != row_idx and score >= threshold
            ]
            candidates = heapq.nlargest(self.synonymy_edge_max_neighbors, candidates, key=lambda item: item[1])
            if len(candidates) > 100:
                continue
            left = self.entity_node_ids[row_idx]
            for idx, score in candidates:
                right = self.entity_node_ids[idx]
                before = self.adj[left].get(right, 0.0)
                self._add_edge(left, right, score)
                if before == 0.0:
                    added += 1
        self.synonymy_edges_added = added

    def dense_passage_scores(self, query_vector: list[float]) -> list[tuple[int, float]]:
        scored = [(idx, _dot(query_vector, vector)) for idx, vector in enumerate(self.passage_vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def fact_scores(self, query_vector: list[float]) -> list[tuple[int, float]]:
        scored = [(idx, _dot(query_vector, vector)) for idx, vector in enumerate(self.fact_vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def ppr_rank_passages(
        self,
        *,
        top_fact_scores: list[tuple[int, float]],
        dense_scores: list[tuple[int, float]],
        linking_top_k: int,
        passage_node_weight: float,
        damping: float,
        iterations: int,
    ) -> list[RetrievalHit]:
        if not self.passage_node_ids:
            return []
        dense_score_by_passage = {idx: score for idx, score in dense_scores}
        normalized_dense = _min_max_by_index(dense_score_by_passage)
        reset: dict[str, float] = defaultdict(float)
        for rank, (fact_idx, score) in enumerate(top_fact_scores[:linking_top_k]):
            if fact_idx >= len(self.memory.fact_layer):
                continue
            fact = self.memory.fact_layer[fact_idx]
            fact_score = max(score, 0.0)
            for phrase in (fact.content[0], fact.content[2]):
                entity_id = self.entity_id_by_text.get(phrase)
                if not entity_id:
                    continue
                denom = max(1, self.ent_node_to_num_chunk.get(entity_id, 1))
                reset[entity_id] += fact_score / denom
        for passage_idx, score in normalized_dense.items():
            if passage_idx < len(self.passage_node_ids):
                reset[self.passage_node_ids[passage_idx]] += score * passage_node_weight
        if not reset or sum(reset.values()) <= 0.0:
            return [
                RetrievalHit(chunk_id=self.passage_node_ids[idx], score=float(score), score_source="memgraphrag_dense_fallback")
                for idx, score in dense_scores
                if idx < len(self.passage_node_ids)
            ]
        return self._personalized_pagerank(reset, damping=damping, iterations=iterations)

    def _personalized_pagerank(self, reset: dict[str, float], *, damping: float, iterations: int) -> list[RetrievalHit]:
        nodes = set(self.adj)
        nodes.update(reset)
        for chunk_id in self.passage_node_ids:
            nodes.add(chunk_id)
        node_list = sorted(nodes)
        total_reset = sum(max(0.0, value) for value in reset.values()) or 1.0
        reset_prob = {node: max(0.0, reset.get(node, 0.0)) / total_reset for node in node_list}
        scores = dict(reset_prob)
        damping = max(0.0, min(0.99, float(damping)))
        iterations = max(1, int(iterations or 1))
        for _ in range(iterations):
            next_scores = {node: (1.0 - damping) * reset_prob.get(node, 0.0) for node in node_list}
            dangling_mass = 0.0
            for node in node_list:
                neighbors = self.adj.get(node, {})
                mass = scores.get(node, 0.0) * damping
                total_weight = sum(weight for weight in neighbors.values() if weight > 0)
                if total_weight <= 0.0:
                    dangling_mass += mass
                    continue
                for neighbor, weight in neighbors.items():
                    if weight > 0:
                        next_scores[neighbor] = next_scores.get(neighbor, 0.0) + mass * (weight / total_weight)
            if dangling_mass:
                for node in node_list:
                    next_scores[node] = next_scores.get(node, 0.0) + dangling_mass * reset_prob.get(node, 0.0)
            scores = next_scores
        hits = [
            RetrievalHit(chunk_id=chunk_id, score=float(scores.get(chunk_id, 0.0)), score_source="memgraphrag_ppr")
            for chunk_id in self.passage_node_ids
        ]
        return sorted(hits, key=lambda item: (-item.score, item.chunk_id))

    def write_artifacts(self, *, status: str = "ready") -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "status": status,
            "method": METHOD,
            "input_boundary": {
                "shared_public_inputs": ["chunk", "entity", "type", "relation"],
                "baseline_owned_outputs": ["schema_memory", "fact_memory", "passage_memory", "fact_to_passage_links", "retrieval_graph"],
                "uses_signpost_fact_or_provenance": False,
                "uses_silver_evidence": False,
                "uses_signpost_online_graph": False,
            },
            "schema_min_count": self.schema_min_count,
            "filter_stats": self.filter_stats,
            "num_phrase_nodes": len(self.entity_node_ids),
            "num_passage_nodes": len(self.passage_node_ids),
            "num_total_nodes": len(set(self.adj) | set(self.passage_node_ids)),
            "num_extracted_triples": len(self.memory.fact_layer),
            "num_schema_nodes": len(self.memory.schema_layer),
            "num_memory_passage_nodes": len(self.memory.passage_layer),
            "num_total_edges": sum(len(neighbors) for neighbors in self.adj.values()) // 2,
            "num_synonymy_triples": self.synonymy_edges_added,
            "synonymy_edges_enabled": self.synonymy_edges,
            "synonymy_edge_sim_threshold": self.synonymy_edge_sim_threshold,
            "embedding_provider": self.embedding_provider_name,
            "embedding_batch_size": self.embedding_batch_size,
            "offline_embedding_calls": self.offline_embedding_calls,
            "offline_embedding_retries": self.offline_embedding_retries,
            "offline_embedding_failures": self.offline_embedding_failures,
            "offline_embedding_wall_time_seconds": self.offline_embedding_wall_time_seconds,
            "embedded_entities": len(self.entity_vectors),
            "embedded_facts": len(self.fact_vectors),
            "embedded_passages": len(self.passage_vectors),
        }
        _write_json(self.artifact_dir / "graph.json", graph)
        _write_json(self.artifact_dir / "memory.json", self.memory.to_dict())
        _write_jsonl(
            self.artifact_dir / "schemas.jsonl",
            [
                {
                    "idx": schema.idx,
                    "content": list(schema.content),
                    "frequency": schema.frequency,
                    "fact_indices": schema.fact_indices,
                }
                for schema in self.memory.schema_layer
            ],
        )
        _write_jsonl(
            self.artifact_dir / "facts.jsonl",
            [
                {
                    "idx": fact.idx,
                    "fact_id": self.fact_node_ids[fact.idx] if fact.idx < len(self.fact_node_ids) else _node_id(fact.text, prefix="fact-"),
                    "content": list(fact.content),
                    "schema_idx": fact.schema_idx,
                    "passage_indices": fact.passage_indices,
                    "passage_chunk_ids": [self.memory.passage_layer[idx].chunk_id for idx in fact.passage_indices if idx < len(self.memory.passage_layer)],
                    "raw_samples": fact.raw_samples,
                    "occurrences": self.fact_occurrences.get(fact.idx, 0),
                }
                for fact in self.memory.fact_layer
            ],
        )
        _write_jsonl(
            self.artifact_dir / "passages.jsonl",
            [
                {
                    "idx": passage.idx,
                    "chunk_id": passage.chunk_id,
                    "fact_indices": passage.fact_indices,
                }
                for passage in self.memory.passage_layer
            ],
        )
        return graph


class MemGraphRAGRunner:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        extractions_path: Path,
        artifact_dir: Path,
        retrieval_top_k: int,
        qa_top_k: int,
        linking_top_k: int,
        ppr_damping: float,
        ppr_iterations: int,
        passage_node_weight: float,
        max_context_tokens: int,
        embedding_provider: str,
        embedding_batch_size: int,
        schema_min_count: int,
        synonymy_edges: bool,
        synonymy_edge_sim_threshold: float,
        synonymy_edge_max_neighbors: int,
        reuse_index: bool = False,
        reuse_index_dir: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.retrieval_top_k = max(1, int(retrieval_top_k or 1))
        self.qa_top_k = max(1, int(qa_top_k or 1))
        self.linking_top_k = max(1, int(linking_top_k or 1))
        self.ppr_damping = float(ppr_damping)
        self.ppr_iterations = max(1, int(ppr_iterations or 1))
        self.passage_node_weight = float(passage_node_weight)
        self.max_context_tokens = max_context_tokens
        self.embedding_provider = embedding_provider
        self.embedding_batch_size = embedding_batch_size
        self.llm = OpenAICompatibleClient()
        build_started = time.time()
        if reuse_index:
            self.index = MemGraphRAGIndex.load_cache(reuse_index_dir or artifact_dir, embedding_provider=embedding_provider)
            self.index.artifact_dir = artifact_dir
            self.offline_wall_time_seconds = 0.0
            self.index_metrics = self.index.write_artifacts()
            self.index_metrics["offline_reused"] = True
            if reuse_index_dir:
                self.index_metrics["offline_reuse_source_dir"] = str(reuse_index_dir)
        else:
            self.index = MemGraphRAGIndex(
                chunks=load_jsonl_list(chunks_path),
                extractions=load_jsonl_list(extractions_path),
                embedding_provider=embedding_provider,
                artifact_dir=artifact_dir,
                embedding_batch_size=embedding_batch_size,
                schema_min_count=schema_min_count,
                synonymy_edges=synonymy_edges,
                synonymy_edge_sim_threshold=synonymy_edge_sim_threshold,
                synonymy_edge_max_neighbors=synonymy_edge_max_neighbors,
            )
            self.index.save_cache()
            self.offline_wall_time_seconds = time.time() - build_started
            self.index_metrics = self.index.write_artifacts()
            self.index_metrics["offline_reused"] = False
        self.index_metrics["offline_wall_time_seconds"] = 0.0 if reuse_index else self.offline_wall_time_seconds
        self.index_metrics["cached_offline_wall_time_seconds"] = self.index.offline_embedding_wall_time_seconds if reuse_index else self.offline_wall_time_seconds
        self.index_metrics["offline_embedding_calls"] = self.index.offline_embedding_calls
        self.index_metrics["offline_embedding_wall_time_seconds"] = self.index.offline_embedding_wall_time_seconds
        _write_json(artifact_dir / "graph.json", self.index_metrics)

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        retrieval_started = time.time()
        fact_query_vector = self.index.embed_query(question)
        passage_query_vector = self.index.embed_query(question)
        fact_scores = self.index.fact_scores(fact_query_vector)
        dense_scores = self.index.dense_passage_scores(passage_query_vector)
        top_fact_scores = fact_scores[: self.linking_top_k]
        ppr_started = time.time()
        if top_fact_scores:
            ppr_hits = self.index.ppr_rank_passages(
                top_fact_scores=top_fact_scores,
                dense_scores=dense_scores,
                linking_top_k=self.linking_top_k,
                passage_node_weight=self.passage_node_weight,
                damping=self.ppr_damping,
                iterations=self.ppr_iterations,
            )
        else:
            ppr_hits = [
                RetrievalHit(chunk_id=self.index.passage_node_ids[idx], score=float(score), score_source="memgraphrag_dense_fallback")
                for idx, score in dense_scores
                if idx < len(self.index.passage_node_ids)
            ]
        ppr_latency = time.time() - ppr_started
        retrieved = self._chunks_from_hits(ppr_hits[: self.retrieval_top_k])
        answer_chunks = retrieved[: self.qa_top_k]
        retrieval_latency = time.time() - retrieval_started
        context, used_chunks = join_context(answer_chunks, max_context_tokens=self.max_context_tokens)
        prompt = (
            "Retrieved passages:\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Thought: "
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        raw_answer, input_tokens, output_tokens, llm_latency = chat_once(self.llm, messages, input_text=SYSTEM_PROMPT + "\n" + prompt)
        rationale, answer = _parse_thought_answer(raw_answer)
        citations = [
            {"file_name": item.get("file_name"), "start_line": item.get("start_line"), "end_line": item.get("end_line"), "locate": locate_from_chunk(item)}
            for item in used_chunks
            if locate_from_chunk(item)
        ]
        retrieved_chunks = [
            {
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
            }
            for item in retrieved[: max(self.qa_top_k, 10)]
            if item.get("chunk_id")
        ]
        top_facts = [
            {
                "fact_idx": fact_idx,
                "score": score,
                "content": list(self.index.memory.fact_layer[fact_idx].content),
                "raw_samples": self.index.memory.fact_layer[fact_idx].raw_samples[:1],
            }
            for fact_idx, score in top_fact_scores
            if fact_idx < len(self.index.memory.fact_layer)
        ]
        return BaselineResult(
            answer=answer,
            rationale=rationale,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            evidence_chunks=retrieved_chunks[: self.qa_top_k],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=1.0,
            tool_calls=1.0,
            embedding_calls=2.0,
            graph_ppr_calls=1.0 if top_fact_scores else 0.0,
            ppr_latency_seconds=ppr_latency,
            retrieval_latency_seconds=retrieval_latency,
            trace=[
                {
                    "event_type": "tool_call",
                    "tool": "memgraphrag_fact_dense_ppr_retrieve",
                    "latency_seconds": retrieval_latency,
                    "output_summary": {
                        "top_facts": top_facts,
                        "retrieved_chunks": len(retrieved_chunks),
                        "linking_top_k": self.linking_top_k,
                        "retrieval_top_k": self.retrieval_top_k,
                        "qa_top_k": self.qa_top_k,
                        "graph_ppr_calls": 1 if top_fact_scores else 0,
                        "ppr_latency_seconds": ppr_latency,
                    },
                },
                {
                    "event_type": "llm_call",
                    "stage": "memgraphrag_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": input_tokens,
                    "output_tokens_estimate": output_tokens,
                },
            ],
        )

    def _chunks_from_hits(self, hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        result = []
        seen = set()
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            chunk = self.index.chunks_by_id.get(hit.chunk_id)
            if not chunk:
                continue
            result.append({**chunk, "score": hit.score, "score_source": hit.score_source})
            seen.add(hit.chunk_id)
        return result


def run_memgraphrag(
    *,
    dataset: str,
    namespace: str | None = None,
    questions_path: str | None = None,
    chunks_path: str | None = None,
    extractions_path: str | None = None,
    output_path: str | None = None,
    query_log_path: str | None = None,
    artifact_dir: str | None = None,
    limit: int | None = None,
    retrieval_top_k: int = 200,
    qa_top_k: int = 5,
    linking_top_k: int = 5,
    ppr_damping: float = 0.5,
    ppr_iterations: int = 20,
    passage_node_weight: float = 0.05,
    max_context_tokens: int = 3500,
    embedding_provider: str = "ecnu",
    embedding_batch_size: int | None = None,
    schema_min_count: int = 2,
    synonymy_edges: bool = True,
    synonymy_edge_sim_threshold: float = 0.8,
    synonymy_edge_max_neighbors: int = 100,
    reuse_index: bool = False,
    workers: int | None = None,
    reuse_index_dir: str | None = None,
) -> int:
    paths = build_paths(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method=METHOD,
    )
    runner = MemGraphRAGRunner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        extractions_path=resolve_project_path(extractions_path or f"datasets/processed/{dataset}/semantic_llm.extractions.jsonl"),
        artifact_dir=resolve_project_path(artifact_dir or f"outputs/{dataset}/baselines/{METHOD}"),
        retrieval_top_k=retrieval_top_k,
        qa_top_k=qa_top_k,
        linking_top_k=linking_top_k,
        ppr_damping=ppr_damping,
        ppr_iterations=ppr_iterations,
        passage_node_weight=passage_node_weight,
        max_context_tokens=max_context_tokens,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size or int(os.environ.get("MEMGRAPHRAG_EMBED_BATCH_SIZE") or os.environ.get("BASELINE_EMBED_BATCH_SIZE", "32")),
        schema_min_count=schema_min_count,
        synonymy_edges=synonymy_edges,
        synonymy_edge_sim_threshold=synonymy_edge_sim_threshold,
        synonymy_edge_max_neighbors=synonymy_edge_max_neighbors,
        reuse_index=reuse_index,
        reuse_index_dir=resolve_project_path(reuse_index_dir) if reuse_index_dir else None,
    )
    return run_baseline_batch(
        method=METHOD,
        paths=paths,
        answer_fn=runner.answer,
        limit=limit,
        workers=workers,
        metadata={
            "retrieval": "memgraphrag_fact_dense_ppr",
            "shared_public_inputs": ["chunk", "entity", "type", "relation"],
            "baseline_owned_outputs": ["schema", "fact", "passage_memory", "fact_to_passage_links"],
            "schema_min_count": schema_min_count,
            "retrieval_top_k": retrieval_top_k,
            "qa_top_k": qa_top_k,
            "linking_top_k": linking_top_k,
            "ppr_damping": ppr_damping,
            "ppr_iterations": ppr_iterations,
            "passage_node_weight": passage_node_weight,
            "embedding_provider": embedding_provider,
            "embedding_batch_size": runner.embedding_batch_size,
            "max_context_tokens": max_context_tokens,
            "index_metrics": runner.index_metrics,
            "prompt_style": "memgraphrag_retrieved_passages_thought_answer",
            "offline_reused": reuse_index,
        },
    )


def _convert_public_extractions_to_openie(chunks: list[dict[str, Any]], extractions: list[dict[str, Any]]) -> dict[str, Any]:
    docs_by_chunk: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        docs_by_chunk[chunk_id] = {
            "idx": chunk_id,
            "passage": str(chunk.get("content") or ""),
            "extracted_entities": [],
            "extracted_triples": [],
            "extracted_triple_ontology": {},
            "raw_triple_map": {},
            "chunk_metadata": {
                "doc_id": chunk.get("doc_id"),
                "file_name": chunk.get("file_name"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
            },
        }
    global_type_by_entity: dict[str, str] = {}
    for row in extractions:
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
        entities = extraction.get("entities") if isinstance(extraction.get("entities"), list) else []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = _text_processing(entity.get("name"))
            entity_type = _entity_type(entity.get("entity_type"))
            if name:
                global_type_by_entity.setdefault(name, entity_type)

    for row in extractions:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        doc = docs_by_chunk.setdefault(
            chunk_id,
            {
                "idx": chunk_id,
                "passage": "",
                "extracted_entities": [],
                "extracted_triples": [],
                "extracted_triple_ontology": {},
                "raw_triple_map": {},
                "chunk_metadata": {
                    "doc_id": row.get("doc_id"),
                    "file_name": row.get("file_name"),
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                },
            },
        )
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
        entities = extraction.get("entities") if isinstance(extraction.get("entities"), list) else []
        type_by_entity: dict[str, str] = {}
        extracted_entities = set(doc.get("extracted_entities", []))
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = _text_processing(entity.get("name"))
            if not name:
                continue
            entity_type = _entity_type(entity.get("entity_type"))
            type_by_entity[name] = entity_type
            global_type_by_entity.setdefault(name, entity_type)
            extracted_entities.add(name)
        relations = extraction.get("relations") if isinstance(extraction.get("relations"), list) else []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            raw_head = str(rel.get("source") or "").strip()
            raw_tail = str(rel.get("target") or "").strip()
            raw_relation = _relation_label(rel)
            head = _text_processing(raw_head)
            tail = _text_processing(raw_tail)
            relation = _text_processing(raw_relation) or "related to"
            if not head or not tail or head == tail:
                continue
            triple = (head, relation, tail)
            triple_key = str(triple)
            head_type = type_by_entity.get(head) or global_type_by_entity.get(head) or "OTHER"
            tail_type = type_by_entity.get(tail) or global_type_by_entity.get(tail) or "OTHER"
            ontology = [head_type, relation, tail_type]
            if triple_key not in doc["extracted_triple_ontology"]:
                doc["extracted_triples"].append([head, relation, tail])
            doc["extracted_triple_ontology"][triple_key] = ontology
            doc["raw_triple_map"].setdefault(triple_key, [raw_head, raw_relation, raw_tail])
            extracted_entities.add(head)
            extracted_entities.add(tail)
        doc["extracted_entities"] = sorted(extracted_entities)

    docs = [docs_by_chunk[str(chunk.get("chunk_id"))] for chunk in chunks if str(chunk.get("chunk_id") or "") in docs_by_chunk]
    num_entities = sum(len(doc.get("extracted_entities", [])) for doc in docs)
    entity_char_sum = sum(len(entity) for doc in docs for entity in doc.get("extracted_entities", []))
    entity_word_sum = sum(len(str(entity).split()) for doc in docs for entity in doc.get("extracted_entities", []))
    return {
        "docs": docs,
        "avg_ent_chars": round(entity_char_sum / max(1, num_entities), 4),
        "avg_ent_words": round(entity_word_sum / max(1, num_entities), 4),
        "source": "signpost_shared_public_chunk_entity_type_relation",
        "public_boundary": {
            "chunk": True,
            "entity": True,
            "type": True,
            "relation": True,
            "fact": False,
            "provenance": False,
        },
    }


def _filter_low_frequency_ontology(data: dict[str, Any], *, min_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    filtered = copy.deepcopy(data)
    docs = filtered.get("docs", [])
    if not isinstance(docs, list):
        return filtered, {}
    ontology_counter: Counter[tuple[str, str, str]] = Counter()
    triple_latest_ontology: dict[str, tuple[str, str, str]] = {}
    triple_occurrences: dict[str, list[int]] = defaultdict(list)
    for doc_idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        triple_ont_map = doc.get("extracted_triple_ontology")
        if not isinstance(triple_ont_map, dict):
            continue
        for triple_key, ontology in triple_ont_map.items():
            if not _valid_ontology(ontology):
                continue
            ontology_tuple = tuple(str(item) for item in ontology)
            triple_latest_ontology[str(triple_key)] = ontology_tuple
            triple_occurrences[str(triple_key)].append(doc_idx)
    for triple_key, ontology in triple_latest_ontology.items():
        for _doc_idx in triple_occurrences.get(triple_key, []):
            ontology_counter[ontology] += 1
    to_remove = {ontology for ontology, count in ontology_counter.items() if count < min_count}
    removed_triples = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        triple_ont_map = doc.get("extracted_triple_ontology")
        if not isinstance(triple_ont_map, dict):
            continue
        for triple_key in list(triple_ont_map):
            latest = triple_latest_ontology.get(str(triple_key))
            if latest is None:
                continue
            if latest in to_remove:
                del triple_ont_map[triple_key]
                removed_triples += 1
            else:
                triple_ont_map[triple_key] = list(latest)
        doc["extracted_triple_ontology"] = triple_ont_map
        kept_keys = set(triple_ont_map)
        kept_triples = []
        for triple in doc.get("extracted_triples", []):
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            if str(tuple(str(item) for item in triple)) in kept_keys:
                kept_triples.append(triple)
        doc["extracted_triples"] = kept_triples
        _add_chunk_extra_fields(doc)
    stats = {
        "min_count": min_count,
        "unique_ontology_original": len(ontology_counter),
        "unique_ontology_removed": len(to_remove),
        "unique_ontology_remaining": len(ontology_counter) - len(to_remove),
        "triple_instances_original": sum(ontology_counter.values()),
        "triple_instances_removed": removed_triples,
        "triple_instances_remaining": max(0, sum(ontology_counter.values()) - removed_triples),
    }
    filtered["filter_stats"] = stats
    return filtered, stats


def _add_chunk_extra_fields(doc: dict[str, Any]) -> None:
    unique_ontologies = set()
    entity_mapping: list[dict[str, str]] = []
    triple_ont_map = doc.get("extracted_triple_ontology")
    if not isinstance(triple_ont_map, dict):
        doc["unique_ontologies"] = []
        doc["entity_mapping"] = []
        return
    for triple_key, ontology in triple_ont_map.items():
        triple = _literal_triple(triple_key)
        if triple is None or not _valid_ontology(ontology):
            continue
        head_type, _relation, tail_type = [str(item) for item in ontology]
        unique_ontologies.add(tuple(str(item) for item in ontology))
        entity_mapping.append({"type": head_type, "entity": triple[0]})
        entity_mapping.append({"type": tail_type, "entity": triple[2]})
    doc["unique_ontologies"] = [list(item) for item in sorted(unique_ontologies)]
    doc["entity_mapping"] = entity_mapping


def _relation_label(rel: dict[str, Any]) -> str:
    value = rel.get("relation") or rel.get("predicate")
    if isinstance(value, str) and value.strip():
        return value.strip()
    keywords = rel.get("keywords")
    if isinstance(keywords, list):
        for item in keywords:
            text = str(item).strip()
            if text:
                return text
    return "related to"


def _entity_type(value: Any) -> str:
    text = str(value or "OTHER").strip().upper()
    return text or "OTHER"


def _text_processing(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    cleaned = re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()
    return " ".join(cleaned.split())


def _node_id(content: str, *, prefix: str) -> str:
    return prefix + hashlib.md5(str(content).encode("utf-8")).hexdigest()


def _literal_triple(triple_key: Any) -> tuple[str, str, str] | None:
    try:
        value = ast.literal_eval(str(triple_key))
    except Exception:
        return None
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    return tuple(str(item) for item in value)


def _valid_ontology(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(item, str) and item for item in value)


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def _min_max_by_index(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {idx: 1.0 for idx in scores}
    return {idx: (score - lo) / (hi - lo) for idx, score in scores.items()}


def _parse_thought_answer(raw: str) -> tuple[str, str]:
    text = raw.strip()
    answer_marker = "Answer:"
    thought_marker = "Thought:"
    if answer_marker not in text:
        return "", text
    before, answer = text.split(answer_marker, 1)
    rationale = before.split(thought_marker, 1)[-1].strip() if thought_marker in before else before.strip()
    return rationale, answer.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MemGraphRAG baseline adapter.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--extractions")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-top-k", type=int, default=200)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--linking-top-k", type=int, default=5)
    parser.add_argument("--ppr-damping", type=float, default=0.5)
    parser.add_argument("--ppr-iterations", type=int, default=20)
    parser.add_argument("--passage-node-weight", type=float, default=0.05)
    parser.add_argument("--max-context-tokens", type=int, default=3500)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--schema-min-count", type=int, default=2)
    synonymy_group = parser.add_mutually_exclusive_group()
    synonymy_group.add_argument("--synonymy-edges", dest="synonymy_edges", action="store_true")
    synonymy_group.add_argument("--no-synonymy-edges", dest="synonymy_edges", action="store_false")
    parser.set_defaults(synonymy_edges=True)
    parser.add_argument("--synonymy-edge-sim-threshold", type=float, default=0.8)
    parser.add_argument("--synonymy-edge-max-neighbors", type=int, default=100)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reuse-index-dir")
    args = parser.parse_args()
    count = run_memgraphrag(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        output_path=args.output,
        query_log_path=args.query_log,
        artifact_dir=args.artifact_dir,
        limit=args.limit,
        retrieval_top_k=args.retrieval_top_k,
        qa_top_k=args.qa_top_k,
        linking_top_k=args.linking_top_k,
        ppr_damping=args.ppr_damping,
        ppr_iterations=args.ppr_iterations,
        passage_node_weight=args.passage_node_weight,
        max_context_tokens=args.max_context_tokens,
        embedding_provider=args.embedding_provider,
        embedding_batch_size=args.embedding_batch_size,
        schema_min_count=args.schema_min_count,
        synonymy_edges=args.synonymy_edges,
        synonymy_edge_sim_threshold=args.synonymy_edge_sim_threshold,
        synonymy_edge_max_neighbors=args.synonymy_edge_max_neighbors,
        reuse_index=args.reuse_index,
        workers=args.workers,
        reuse_index_dir=args.reuse_index_dir,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
