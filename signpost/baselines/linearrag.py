from __future__ import annotations

"""LinearRAG adapter over Signpost shared artifacts.

The adapter preserves LinearRAG's relation-free retrieval shape: entity and
passage nodes, sentence-mediated entity expansion, adjacent-passage links, and
personalized PageRank over a linear graph. It does not rechunk documents or run
a new NER model; entity mentions come from ``semantic_llm.extractions.jsonl``.
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
from signpost.retrieval.chunk_search import search_chunks


METHOD = "linearrag"
SYSTEM_PROMPT = """As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence and LinearRAG relation-free reasoning graph. Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence and reasoning graph. Conclude with "Answer: " to present a complete, well-formed final response.

Follow these rules:
- Include all necessary context and details supported by the evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".

Example Input:
LinearRAG relation-free reasoning graph:
Seed entities: Greensgrow Farm
Activated bridge entities: hydroponic growing; aquaponics; composting; biodiesel production; community engagement and education
Top passages are selected by entity/sentence bridging and personalized PageRank.

Evidence:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.

Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?
Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence and LinearRAG reasoning graph identify hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education."""


@dataclass(frozen=True)
class LinearRagConfig:
    retrieval_top_k: int
    hybrid_top_k: int
    seed_top_k: int
    top_k_sentence: int
    max_iterations: int
    iteration_threshold: float
    passage_ratio: float
    passage_node_weight: float
    damping: float
    max_context_tokens: int


class LinearRagIndex:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        extractions: list[dict[str, Any]],
        embedding_provider: str,
        artifact_dir: Path,
        embedding_batch_size: int,
    ):
        self.chunks = chunks
        self.chunks_by_id = {str(item.get("chunk_id")): item for item in chunks if item.get("chunk_id")}
        self.chunk_ids = [str(item.get("chunk_id")) for item in chunks if item.get("chunk_id")]
        self.embedding_provider_name = embedding_provider
        self.embedding_provider = create_embedding_provider(embedding_provider)
        self.embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self.artifact_dir = artifact_dir

        self.entity_names: dict[str, str] = {}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.entity_sentences: dict[str, set[str]] = defaultdict(set)
        self.sentence_entities: dict[str, set[str]] = defaultdict(set)
        self.sentence_texts: dict[str, str] = {}
        self.sentence_chunks: dict[str, str] = {}
        self.adj: dict[str, dict[str, float]] = defaultdict(dict)
        self.passage_vectors: list[list[float]] = []
        self.entity_vectors: list[list[float]] = []
        self.sentence_vectors: list[list[float]] = []
        self.entity_ids: list[str] = []
        self.sentence_ids: list[str] = []
        self.offline_embedding_calls = 0
        self.offline_embedding_retries = 0
        self.offline_embedding_failures = 0
        self.offline_embedding_wall_time_seconds = 0.0
        self._build(extractions)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["embedding_provider"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.embedding_provider = create_embedding_provider(self.embedding_provider_name)

    @classmethod
    def load_cache(cls, artifact_dir: Path, *, embedding_provider: str) -> "LinearRagIndex":
        cache_path = artifact_dir / "index.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"LinearRAG index cache not found: {cache_path}")
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError(f"LinearRAG index cache has wrong type: {type(index)!r}")
        if index.embedding_provider_name != embedding_provider:
            raise ValueError(
                f"LinearRAG index cache embedding_provider={index.embedding_provider_name!r} "
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
        chunk_entities = self._chunk_entities(extractions)
        for chunk in self.chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            names = chunk_entities.get(chunk_id, [])
            entity_keys = []
            for name in names:
                key = _entity_key(name)
                if not key:
                    continue
                self.entity_names.setdefault(key, name)
                self.entity_chunks[key].add(chunk_id)
                entity_keys.append(key)
            self._add_entity_passage_edges(chunk_id, entity_keys, str(chunk.get("content") or ""))
            self._add_sentence_bridges(chunk_id, entity_keys, str(chunk.get("content") or ""))
        self._add_adjacent_passage_edges()
        self.entity_ids = sorted(self.entity_names)
        self.sentence_ids = sorted(self.sentence_texts)
        self.write_artifacts(status="graph_built")
        print(
            f"[linearrag] graph built entities={len(self.entity_ids)} sentences={len(self.sentence_ids)} "
            f"passages={len(self.chunk_ids)} edges={sum(len(v) for v in self.adj.values()) // 2}",
            file=sys.stderr,
            flush=True,
        )
        started = time.time()
        self.passage_vectors = self._embed([str(item.get("content") or "") for item in self.chunks if item.get("chunk_id")], "passage")
        self.entity_vectors = self._embed([self.entity_names[key] for key in self.entity_ids], "entity")
        self.sentence_vectors = self._embed([self.sentence_texts[key] for key in self.sentence_ids], "sentence")
        self.offline_embedding_wall_time_seconds = time.time() - started
        self.write_artifacts(status="embedded")

    def _chunk_entities(self, extractions: list[dict[str, Any]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        seen: dict[str, set[str]] = defaultdict(set)
        for row in extractions:
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                continue
            extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
            entities = extraction.get("entities") if isinstance(extraction.get("entities"), list) else []
            relations = extraction.get("relations") if isinstance(extraction.get("relations"), list) else []
            names = [str(item.get("name") or "").strip() for item in entities if isinstance(item, dict)]
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                names.extend([str(rel.get("source") or "").strip(), str(rel.get("target") or "").strip()])
            for name in names:
                key = _entity_key(name)
                if key and key not in seen[chunk_id]:
                    seen[chunk_id].add(key)
                    result[chunk_id].append(name)
        return result

    def _add_entity_passage_edges(self, chunk_id: str, entity_keys: list[str], content: str) -> None:
        counts = []
        content_lower = content.lower()
        for key in entity_keys:
            count = max(1, content_lower.count(self.entity_names[key].lower()))
            counts.append((key, count))
        total = sum(count for _key, count in counts) or 1
        for key, count in counts:
            self._add_edge(key, f"chunk:{chunk_id}", max(0.01, count / total))

    def _add_sentence_bridges(self, chunk_id: str, entity_keys: list[str], content: str) -> None:
        if not entity_keys:
            return
        for sent_index, sentence in enumerate(_split_sentences(content)):
            sentence_lower = sentence.lower()
            linked = [key for key in entity_keys if self.entity_names[key].lower() in sentence_lower]
            if not linked:
                continue
            sentence_id = f"sent:{chunk_id}:{sent_index}"
            self.sentence_texts[sentence_id] = sentence
            self.sentence_chunks[sentence_id] = chunk_id
            for key in linked:
                self.entity_sentences[key].add(sentence_id)
                self.sentence_entities[sentence_id].add(key)

    def _add_adjacent_passage_edges(self) -> None:
        previous_by_doc: dict[str, str] = {}
        for chunk in self.chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            doc_id = str(chunk.get("doc_id") or "")
            if not chunk_id or not doc_id:
                continue
            previous = previous_by_doc.get(doc_id)
            if previous:
                self._add_edge(f"chunk:{previous}", f"chunk:{chunk_id}", 1.0)
            previous_by_doc[doc_id] = chunk_id

    def _add_edge(self, left: str, right: str, weight: float) -> None:
        if not left or not right or left == right:
            return
        self.adj[left][right] = max(float(weight), self.adj[left].get(right, 0.0))
        self.adj[right][left] = max(float(weight), self.adj[right].get(left, 0.0))

    def _embed(self, texts: list[str], label: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        if not texts:
            return vectors
        total_batches = math.ceil(len(texts) / self.embedding_batch_size)
        started = time.time()
        for batch_index, start in enumerate(range(0, len(texts), self.embedding_batch_size), start=1):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend(self._embed_batch_with_retry(batch, label=label))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = time.time() - started
                print(
                    f"[linearrag] embedded {label} batch {batch_index}/{total_batches} "
                    f"vectors={len(vectors)}/{len(texts)} elapsed_seconds={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                self.write_artifacts(status=f"embedding_{label}")
        return vectors

    def _embed_batch_with_retry(self, batch: list[str], *, label: str) -> list[list[float]]:
        retries = max(1, int(os.environ.get("LINEARRAG_EMBED_RETRIES", "3")))
        retry_sleep = max(0.0, float(os.environ.get("LINEARRAG_EMBED_RETRY_SLEEP", "5")))
        for attempt in range(1, retries + 1):
            try:
                vectors = self.embedding_provider.embed(batch)
                self.offline_embedding_calls += 1
                return _normalize_vectors(vectors)
            except Exception as exc:
                self.offline_embedding_failures += 1
                if _is_connection_refused(exc):
                    if attempt < retries:
                        self.offline_embedding_retries += 1
                        print(
                            f"[linearrag] embedding service connection refused for {label} batch_size={len(batch)} "
                            f"attempt={attempt}/{retries}; retrying after {retry_sleep:.1f}s. "
                            "Check that the H200 embedding service is listening before rerunning.",
                            file=sys.stderr,
                            flush=True,
                        )
                        if retry_sleep:
                            time.sleep(retry_sleep)
                        continue
                    raise RuntimeError(
                        "LinearRAG embedding service is not reachable after retries. "
                        "This is a service availability/configuration failure, not a batch-size failure. "
                        "Check ECNU_EMBEDDING_API_BASE or OPENAI_EMBEDDING_API_BASE and ensure the H200 "
                        "embedding server is listening on the configured host/port."
                    ) from exc
                if attempt < retries:
                    self.offline_embedding_retries += 1
                    print(
                        f"[linearrag] embedding {label} batch_size={len(batch)} failed "
                        f"attempt={attempt}/{retries}: {exc}; retrying after {retry_sleep:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if retry_sleep:
                        time.sleep(retry_sleep)
                    continue
                if len(batch) == 1:
                    print(
                        f"[linearrag] embedding {label} single item failed after {retries} attempts",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                midpoint = max(1, len(batch) // 2)
                print(
                    f"[linearrag] embedding {label} batch_size={len(batch)} failed after {retries} attempts; "
                    f"splitting into {midpoint}+{len(batch) - midpoint}",
                    file=sys.stderr,
                    flush=True,
                )
                return self._embed_batch_with_retry(batch[:midpoint], label=label) + self._embed_batch_with_retry(batch[midpoint:], label=label)
        raise RuntimeError("unreachable embedding retry state")

    def write_artifacts(self, *, status: str = "ready") -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        graph_path = self.artifact_dir / "graph.json"
        entities_path = self.artifact_dir / "entities.jsonl"
        sentences_path = self.artifact_dir / "sentences.jsonl"
        passage_links_path = self.artifact_dir / "passage_links.jsonl"
        graph = {
            "status": status,
            "method": METHOD,
            "graph_type": "relation_free_entity_sentence_passage_ppr",
            "embedding_provider": self.embedding_provider_name,
            "embedding_batch_size": self.embedding_batch_size,
            "passage_nodes": len(self.chunk_ids),
            "entity_nodes": len(self.entity_names),
            "sentence_bridge_nodes": len(self.sentence_texts),
            "nodes": len(self.adj),
            "edges": sum(len(v) for v in self.adj.values()) // 2,
            "entity_sentence_links": sum(len(v) for v in self.entity_sentences.values()),
            "offline_embedding_calls": self.offline_embedding_calls,
            "offline_embedding_retries": self.offline_embedding_retries,
            "offline_embedding_failures": self.offline_embedding_failures,
            "offline_embedding_wall_time_seconds": self.offline_embedding_wall_time_seconds,
            "embedded_passages": len(self.passage_vectors),
            "embedded_entities": len(self.entity_vectors),
            "embedded_sentences": len(self.sentence_vectors),
        }
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        with entities_path.open("w", encoding="utf-8") as f:
            for key in sorted(self.entity_names):
                f.write(json.dumps({"entity_id": key, "name": self.entity_names[key], "chunks": sorted(self.entity_chunks[key])}, ensure_ascii=False, separators=(",", ":")) + "\n")
        with sentences_path.open("w", encoding="utf-8") as f:
            for key in sorted(self.sentence_texts):
                f.write(json.dumps({"sentence_id": key, "chunk_id": self.sentence_chunks.get(key), "text": self.sentence_texts[key], "entities": sorted(self.sentence_entities[key])}, ensure_ascii=False, separators=(",", ":")) + "\n")
        with passage_links_path.open("w", encoding="utf-8") as f:
            for left in sorted(self.adj):
                for right, weight in sorted(self.adj[left].items()):
                    if left < right:
                        f.write(json.dumps({"source": left, "target": right, "weight": weight}, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {**graph, "graph_path": str(graph_path), "entities_path": str(entities_path), "sentences_path": str(sentences_path), "passage_links_path": str(passage_links_path)}


class LinearRagRunner:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        extractions_path: Path,
        artifact_dir: Path,
        use_es: bool,
        mode: str,
        config: LinearRagConfig,
        embedding_provider: str,
        embedding_batch_size: int,
        reuse_index: bool = False,
        reuse_index_dir: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.use_es = use_es
        self.mode = mode
        self.config = config
        self.embedding_provider = embedding_provider
        self.llm = OpenAICompatibleClient()
        self.local_chunks = load_jsonl_list(chunks_path)
        build_started = time.time()
        if reuse_index:
            self.index = LinearRagIndex.load_cache(reuse_index_dir or artifact_dir, embedding_provider=embedding_provider)
            self.index.artifact_dir = artifact_dir
            self.offline_wall_time_seconds = 0.0
            self.index_metrics = self.index.write_artifacts()
            self.index_metrics["offline_reused"] = True
            if reuse_index_dir:
                self.index_metrics["offline_reuse_source_dir"] = str(reuse_index_dir)
        else:
            self.index = LinearRagIndex(
                chunks=self.local_chunks,
                extractions=load_jsonl_list(extractions_path),
                embedding_provider=embedding_provider,
                artifact_dir=artifact_dir,
                embedding_batch_size=embedding_batch_size,
            )
            self.index.save_cache()
            self.offline_wall_time_seconds = time.time() - build_started
            self.index_metrics = self.index.write_artifacts()
            self.index_metrics["offline_reused"] = False
        self.index_metrics["offline_wall_time_seconds"] = 0.0 if reuse_index else self.offline_wall_time_seconds
        self.index_metrics["cached_offline_embedding_wall_time_seconds"] = self.index.offline_embedding_wall_time_seconds
        (artifact_dir / "graph.json").write_text(json.dumps(self.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        retrieval_started = time.time()
        query_vector = _normalize_vectors(self.index.embedding_provider.embed([question]))[0]
        seed_entities = self._seed_entities(question, query_vector)
        active_entities = self._activate_entities(query_vector, seed_entities)
        passage_weights = self._passage_weights(query_vector, active_entities)
        ppr_started = time.time()
        ppr = self._ppr({**active_entities, **passage_weights})
        ppr_latency = time.time() - ppr_started
        graph_chunks = self._chunks_from_ppr(ppr)
        hybrid_chunks = self._hybrid_retrieve(question)
        retrieved = _dedupe_chunks(graph_chunks + hybrid_chunks, self.config.retrieval_top_k + self.config.hybrid_top_k)
        retrieval_latency = time.time() - retrieval_started
        context, used_chunks = join_context(retrieved, max_context_tokens=self.config.max_context_tokens)
        graph_summary = self._graph_summary(seed_entities, active_entities)
        prompt = (
            "LinearRAG relation-free reasoning graph:\n"
            f"{graph_summary}\n\n"
            "Evidence:\n"
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
            {"chunk_id": str(item.get("chunk_id") or ""), "doc_id": item.get("doc_id"), "score": item.get("score"), "score_source": item.get("score_source")}
            for item in retrieved
            if item.get("chunk_id")
        ]
        return BaselineResult(
            answer=answer,
            rationale=rationale,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=1.0,
            tool_calls=2.0,
            embedding_calls=2.0 if self.mode != "bm25" else 1.0,
            graph_ppr_calls=1.0,
            ppr_latency_seconds=ppr_latency,
            retrieval_latency_seconds=retrieval_latency,
            trace=[
                {
                    "event_type": "tool_call",
                    "tool": "linearrag_relation_free_ppr_search",
                    "latency_seconds": retrieval_latency,
                    "output_summary": {
                        "seed_entities": len(seed_entities),
                        "active_entities": len(active_entities),
                        "retrieved_chunks": len(retrieved_chunks),
                        "graph_ppr_calls": 1,
                        "embedding_calls": 2 if self.mode != "bm25" else 1,
                        "ppr_latency_seconds": ppr_latency,
                    },
                },
                {
                    "event_type": "llm_call",
                    "stage": "linearrag_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": input_tokens,
                    "output_tokens_estimate": output_tokens,
                },
            ],
        )

    def _seed_entities(self, question: str, query_vector: list[float]) -> dict[str, float]:
        question_terms = set(_terms(question))
        scored = []
        for key, vector in zip(self.index.entity_ids, self.index.entity_vectors, strict=False):
            name_terms = set(_terms(self.index.entity_names[key]))
            lexical = len(question_terms & name_terms) / max(1, len(name_terms))
            score = _dot(query_vector, vector) + lexical
            scored.append((key, max(0.0, score)))
        scored.sort(key=lambda item: (-item[1], self.index.entity_names[item[0]]))
        return {key: score for key, score in scored[: self.config.seed_top_k] if score > 0}

    def _activate_entities(self, query_vector: list[float], seed_entities: dict[str, float]) -> dict[str, float]:
        active = dict(seed_entities)
        current = dict(seed_entities)
        sentence_index = {key: idx for idx, key in enumerate(self.index.sentence_ids)}
        for iteration in range(1, self.config.max_iterations):
            next_entities: dict[str, float] = {}
            used_sentences: set[str] = set()
            for entity_key, entity_score in current.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_ids = [sid for sid in self.index.entity_sentences.get(entity_key, set()) if sid not in used_sentences]
                sentence_scores = []
                for sentence_id in sentence_ids:
                    idx = sentence_index.get(sentence_id)
                    if idx is None:
                        continue
                    sentence_scores.append((sentence_id, _dot(query_vector, self.index.sentence_vectors[idx])))
                sentence_scores.sort(key=lambda item: item[1], reverse=True)
                for sentence_id, sentence_score in sentence_scores[: self.config.top_k_sentence]:
                    used_sentences.add(sentence_id)
                    for next_entity in self.index.sentence_entities.get(sentence_id, set()):
                        next_score = entity_score * max(0.0, sentence_score)
                        if next_score < self.config.iteration_threshold:
                            continue
                        if next_score > next_entities.get(next_entity, 0.0):
                            next_entities[next_entity] = next_score
                        if next_score > active.get(next_entity, 0.0):
                            active[next_entity] = next_score
            current = next_entities
            if not current:
                break
        return active

    def _passage_weights(self, query_vector: list[float], active_entities: dict[str, float]) -> dict[str, float]:
        dense_scores = [_dot(query_vector, vector) for vector in self.index.passage_vectors]
        dense_scores = _minmax(dense_scores)
        weights: dict[str, float] = {}
        for index, chunk_id in enumerate(self.index.chunk_ids):
            content = str(self.index.chunks_by_id.get(chunk_id, {}).get("content") or "").lower()
            entity_bonus = 0.0
            for entity_key, entity_score in active_entities.items():
                occurrences = content.count(self.index.entity_names.get(entity_key, entity_key).lower())
                if occurrences > 0:
                    entity_bonus += entity_score * math.log(1 + occurrences)
            passage_score = self.config.passage_ratio * dense_scores[index] + math.log(1 + entity_bonus)
            weights[f"chunk:{chunk_id}"] = passage_score * self.config.passage_node_weight
        return weights

    def _ppr(self, node_weights: dict[str, float]) -> dict[str, float]:
        seeds = {node: max(0.0, score) for node, score in node_weights.items() if score > 0 and node in self.index.adj}
        if not seeds:
            return {}
        total = sum(seeds.values()) or 1.0
        personalization = {node: value / total for node, value in seeds.items()}
        nodes = list(self.index.adj)
        scores = dict(personalization)
        for _ in range(20):
            next_scores = {node: (1.0 - self.config.damping) * personalization.get(node, 0.0) for node in nodes}
            for node in nodes:
                neighbors = self.index.adj.get(node, {})
                total_weight = sum(neighbors.values()) or 1.0
                mass = scores.get(node, 0.0) * self.config.damping
                for neighbor, weight in neighbors.items():
                    next_scores[neighbor] = next_scores.get(neighbor, 0.0) + mass * (weight / total_weight)
            scores = next_scores
        return scores

    def _chunks_from_ppr(self, ppr: dict[str, float]) -> list[dict[str, Any]]:
        ranked = []
        for node, score in ppr.items():
            if not node.startswith("chunk:"):
                continue
            chunk_id = node.removeprefix("chunk:")
            chunk = self.index.chunks_by_id.get(chunk_id)
            if chunk:
                ranked.append(({**chunk, "score": score, "score_source": "linearrag_ppr"}, score))
        ranked.sort(key=lambda item: (-item[1], str(item[0].get("chunk_id", ""))))
        return [item for item, _score in ranked[: self.config.retrieval_top_k]]

    def _hybrid_retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.use_es:
            return search_chunks(
                namespace=self.namespace,
                query=question,
                mode=self.mode,
                top_k=self.config.hybrid_top_k,
                embedding_provider_name=self.embedding_provider,
            ).get("items", [])
        return _local_keyword_search(self.local_chunks, question, self.config.hybrid_top_k)

    def _graph_summary(self, seed_entities: dict[str, float], active_entities: dict[str, float]) -> str:
        seeds = [self.index.entity_names.get(key, key) for key in sorted(seed_entities, key=seed_entities.get, reverse=True)]
        activated = [self.index.entity_names.get(key, key) for key in sorted(active_entities, key=active_entities.get, reverse=True)[:12]]
        return (
            f"Seed entities: {'; '.join(seeds) if seeds else 'None'}\n"
            f"Activated bridge entities: {'; '.join(activated) if activated else 'None'}\n"
            "Top passages are selected by entity/sentence bridging and personalized PageRank."
        )


def run_linearrag(
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
    use_es: bool = False,
    mode: str = "hybrid",
    retrieval_top_k: int = 5,
    hybrid_top_k: int = 5,
    seed_top_k: int = 8,
    top_k_sentence: int = 1,
    max_iterations: int = 3,
    iteration_threshold: float = 0.5,
    passage_ratio: float = 1.5,
    passage_node_weight: float = 0.05,
    damping: float = 0.5,
    max_context_tokens: int = 3500,
    embedding_provider: str = "ecnu",
    embedding_batch_size: int | None = None,
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
    config = LinearRagConfig(
        retrieval_top_k=retrieval_top_k,
        hybrid_top_k=hybrid_top_k,
        seed_top_k=seed_top_k,
        top_k_sentence=top_k_sentence,
        max_iterations=max_iterations,
        iteration_threshold=iteration_threshold,
        passage_ratio=passage_ratio,
        passage_node_weight=passage_node_weight,
        damping=damping,
        max_context_tokens=max_context_tokens,
    )
    runner = LinearRagRunner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        extractions_path=resolve_project_path(extractions_path or f"datasets/processed/{dataset}/semantic_llm.extractions.jsonl"),
        artifact_dir=resolve_project_path(artifact_dir or f"outputs/{dataset}/baselines/{METHOD}"),
        use_es=use_es,
        mode=mode,
        config=config,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size or int(os.environ.get("LINEARRAG_EMBED_BATCH_SIZE", "32")),
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
            "retrieval": "linearrag_relation_free_entity_sentence_passage_ppr",
            "use_es": use_es,
            "mode": mode,
            "retrieval_top_k": retrieval_top_k,
            "hybrid_top_k": hybrid_top_k,
            "seed_top_k": seed_top_k,
            "top_k_sentence": top_k_sentence,
            "max_iterations": max_iterations,
            "iteration_threshold": iteration_threshold,
            "passage_ratio": passage_ratio,
            "passage_node_weight": passage_node_weight,
            "damping": damping,
            "embedding_provider": embedding_provider,
            "embedding_batch_size": runner.index.embedding_batch_size,
            "max_context_tokens": max_context_tokens,
            "index_metrics": runner.index_metrics,
            "prompt_style": "signpost_thought_answer_evidence_grounded",
            "offline_reused": reuse_index,
        },
    )


def _entity_key(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(term) > 1]


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized = []
    for vector in vectors:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
        normalized.append([float(value) / norm for value in vector])
    return normalized


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _is_connection_refused(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "connection refused" in text or "errno 111" in text


def _local_keyword_search(chunks: list[dict[str, Any]], question: str, top_k: int) -> list[dict[str, Any]]:
    terms = _terms(question)
    scored = []
    for item in chunks:
        content = str(item.get("content") or "")
        score = sum(content.lower().count(term) for term in terms)
        if score > 0:
            scored.append({**item, "score": float(score), "score_source": "local_keyword"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def _dedupe_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        result.append(chunk)
        seen.add(chunk_id)
        if len(result) >= limit:
            break
    return result


def _parse_thought_answer(raw: str) -> tuple[str, str]:
    text = raw.strip()
    answer_marker = "Answer:"
    thought_marker = "Thought:"
    if answer_marker not in text:
        return "", text
    before, answer = text.split(answer_marker, 1)
    rationale = before.split(thought_marker, 1)[-1].strip() if thought_marker in before else before.strip()
    return rationale, answer.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LinearRAG baseline adapter.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--extractions")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--hybrid-top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=8)
    parser.add_argument("--top-k-sentence", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--iteration-threshold", type=float, default=0.5)
    parser.add_argument("--passage-ratio", type=float, default=1.5)
    parser.add_argument("--passage-node-weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--max-context-tokens", type=int, default=3500)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reuse-index-dir")
    args = parser.parse_args()
    count = run_linearrag(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        output_path=args.output,
        query_log_path=args.query_log,
        artifact_dir=args.artifact_dir,
        limit=args.limit,
        use_es=args.use_es,
        mode=args.mode,
        retrieval_top_k=args.retrieval_top_k,
        hybrid_top_k=args.hybrid_top_k,
        seed_top_k=args.seed_top_k,
        top_k_sentence=args.top_k_sentence,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        passage_node_weight=args.passage_node_weight,
        damping=args.damping,
        max_context_tokens=args.max_context_tokens,
        embedding_provider=args.embedding_provider,
        embedding_batch_size=args.embedding_batch_size,
        reuse_index=args.reuse_index,
        workers=args.workers,
        reuse_index_dir=args.reuse_index_dir,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
