from __future__ import annotations

"""AGRAG adapter over Signpost shared artifacts.

The adapter keeps the AGRAG retrieval ingredients that are compatible with the
control-variable setup: an entity/relation/passage graph, query-to-triple
linking, PPR influence scoring, MCMI-style greedy subgraph expansion, and
hybrid chunk retrieval. It does not rechunk or re-extract entities; graph input
comes from ``semantic_llm.extractions.jsonl``.
"""

import argparse
import heapq
import json
import math
import os
import pickle
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


METHOD = "agrag"
SYSTEM_PROMPT = """As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence and AGRAG reasoning subgraph. Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence and reasoning subgraph. Conclude with "Answer: " to present a complete, well-formed final response.

Follow these rules:
- Include all necessary context and details supported by the evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".

Example Input:
AGRAG MCMI reasoning subgraph:
(Greensgrow Farm [SEP] uses [SEP] hydroponic growing)
(Greensgrow Farm [SEP] uses [SEP] aquaponics)
(Greensgrow Farm [SEP] promotes [SEP] community engagement and education)

Evidence:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.

Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?
Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence and AGRAG reasoning subgraph identify hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education."""


@dataclass(frozen=True)
class AgragTriple:
    source: str
    target: str
    relation: str
    description: str
    chunk_id: str

    @property
    def text(self) -> str:
        detail = self.description or self.relation
        return f"{self.source} [SEP] {detail} [SEP] {self.target}"


class AgragIndex:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        extractions: list[dict[str, Any]],
        embedding_provider: str,
        artifact_dir: Path,
        embedding_batch_size: int,
    ):
        self.chunks_by_id = {str(item.get("chunk_id")): item for item in chunks if item.get("chunk_id")}
        self.embedding_provider_name = embedding_provider
        self.embedding_provider = create_embedding_provider(embedding_provider)
        self.artifact_dir = artifact_dir
        self.embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self.entity_names: dict[str, str] = {}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.adj: dict[str, dict[str, float]] = defaultdict(dict)
        self.edge_labels: dict[tuple[str, str], str] = {}
        self.triples: list[AgragTriple] = []
        self.triple_vectors: list[list[float]] = []
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
    def load_cache(cls, artifact_dir: Path, *, embedding_provider: str) -> "AgragIndex":
        cache_path = artifact_dir / "index.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"AGRAG index cache not found: {cache_path}")
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError(f"AGRAG index cache has wrong type: {type(index)!r}")
        if index.embedding_provider_name != embedding_provider:
            raise ValueError(
                f"AGRAG index cache embedding_provider={index.embedding_provider_name!r} "
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
        seen_triples: set[tuple[str, str, str, str]] = set()
        for row in extractions:
            chunk_id = str(row.get("chunk_id") or "")
            extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
            entities = extraction.get("entities") if isinstance(extraction.get("entities"), list) else []
            relations = extraction.get("relations") if isinstance(extraction.get("relations"), list) else []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name") or "").strip()
                key = _entity_key(name)
                if not key:
                    continue
                self.entity_names.setdefault(key, name)
                if chunk_id:
                    self.entity_chunks[key].add(chunk_id)
                    self._add_edge(key, f"chunk:{chunk_id}", 0.5, "passage_link")
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                source = str(rel.get("source") or "").strip()
                target = str(rel.get("target") or "").strip()
                source_key = _entity_key(source)
                target_key = _entity_key(target)
                if not source_key or not target_key:
                    continue
                self.entity_names.setdefault(source_key, source)
                self.entity_names.setdefault(target_key, target)
                label = _relation_label(rel)
                weight = _positive_float(rel.get("weight"), 1.0)
                self._add_edge(source_key, target_key, max(0.05, weight), label)
                triple_key = (source_key, target_key, label, chunk_id)
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    self.triples.append(
                        AgragTriple(
                            source=source,
                            target=target,
                            relation=label,
                            description=str(rel.get("description") or ""),
                            chunk_id=chunk_id,
                        )
                    )
        self.write_artifacts(status="triples_built")
        print(
            f"[agrag] graph built triples={len(self.triples)} nodes={len(self.adj)} "
            f"edges={sum(len(v) for v in self.adj.values()) // 2} embedding_batch_size={self.embedding_batch_size}",
            file=sys.stderr,
            flush=True,
        )
        if self.triples:
            self.triple_vectors = []
            texts = [triple.text for triple in self.triples]
            started = time.time()
            total_batches = math.ceil(len(texts) / self.embedding_batch_size)
            for batch_index, start in enumerate(range(0, len(texts), self.embedding_batch_size), start=1):
                batch = texts[start : start + self.embedding_batch_size]
                self.triple_vectors.extend(self._embed_batch_with_retry(batch, label="triple"))
                if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                    elapsed = time.time() - started
                    print(
                        f"[agrag] embedded triple batch {batch_index}/{total_batches} "
                        f"vectors={len(self.triple_vectors)}/{len(texts)} elapsed_seconds={elapsed:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.write_artifacts(status="embedding")
            self.offline_embedding_wall_time_seconds = time.time() - started
            self.write_artifacts(status="embedded")

    def _embed_batch_with_retry(self, batch: list[str], *, label: str) -> list[list[float]]:
        retries = max(1, int(os.environ.get("AGRAG_EMBED_RETRIES") or os.environ.get("BASELINE_EMBED_RETRIES", "3")))
        retry_sleep = max(0.0, float(os.environ.get("AGRAG_EMBED_RETRY_SLEEP") or os.environ.get("BASELINE_EMBED_RETRY_SLEEP", "5")))
        for attempt in range(1, retries + 1):
            try:
                vectors = self.embedding_provider.embed(batch)
                self.offline_embedding_calls += 1
                return vectors
            except Exception as exc:
                self.offline_embedding_failures += 1
                if _is_connection_refused(exc):
                    if attempt < retries:
                        self.offline_embedding_retries += 1
                        print(
                            f"[agrag] embedding service connection refused for {label} batch_size={len(batch)} "
                            f"attempt={attempt}/{retries}; retrying after {retry_sleep:.1f}s. "
                            "Check that the H200 embedding service is listening before rerunning.",
                            file=sys.stderr,
                            flush=True,
                        )
                        if retry_sleep:
                            time.sleep(retry_sleep)
                        continue
                    raise RuntimeError(
                        "AGRAG embedding service is not reachable after retries. "
                        "This is a service availability/configuration failure, not a batch-size failure. "
                        "Check ECNU_EMBEDDING_API_BASE or OPENAI_EMBEDDING_API_BASE and ensure the H200 "
                        "embedding server is listening on the configured host/port."
                    ) from exc
                if attempt < retries:
                    self.offline_embedding_retries += 1
                    print(
                        f"[agrag] embedding {label} batch_size={len(batch)} failed "
                        f"attempt={attempt}/{retries}: {exc}; retrying after {retry_sleep:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if retry_sleep:
                        time.sleep(retry_sleep)
                    continue
                if len(batch) == 1:
                    print(f"[agrag] embedding {label} single item failed after {retries} attempts", file=sys.stderr, flush=True)
                    raise
                midpoint = max(1, len(batch) // 2)
                print(
                    f"[agrag] embedding {label} batch_size={len(batch)} failed after {retries} attempts; "
                    f"splitting into {midpoint}+{len(batch) - midpoint}",
                    file=sys.stderr,
                    flush=True,
                )
                return self._embed_batch_with_retry(batch[:midpoint], label=label) + self._embed_batch_with_retry(batch[midpoint:], label=label)
        raise RuntimeError("unreachable embedding retry state")

    def _add_edge(self, left: str, right: str, weight: float, label: str) -> None:
        if not left or not right or left == right:
            return
        current = self.adj[left].get(right, 0.0)
        self.adj[left][right] = max(current, weight)
        self.adj[right][left] = max(self.adj[right].get(left, 0.0), weight)
        self.edge_labels[(left, right)] = label
        self.edge_labels[(right, left)] = label

    def write_artifacts(self, *, status: str = "ready") -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        graph_path = self.artifact_dir / "graph.json"
        triples_path = self.artifact_dir / "triples.jsonl"
        graph = {
            "nodes": len(self.adj),
            "entity_nodes": len(self.entity_names),
            "chunk_nodes": sum(1 for key in self.adj if key.startswith("chunk:")),
            "edges": sum(len(v) for v in self.adj.values()) // 2,
            "triples": len(self.triples),
            "embedding_provider": self.embedding_provider_name,
            "embedding_batch_size": self.embedding_batch_size,
            "offline_embedding_calls": self.offline_embedding_calls,
            "offline_embedding_retries": self.offline_embedding_retries,
            "offline_embedding_failures": self.offline_embedding_failures,
            "offline_embedding_wall_time_seconds": self.offline_embedding_wall_time_seconds,
            "status": status,
            "embedded_triples": len(self.triple_vectors),
        }
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        with triples_path.open("w", encoding="utf-8") as f:
            for triple in self.triples:
                f.write(json.dumps(triple.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {**graph, "graph_path": str(graph_path), "triples_path": str(triples_path)}


class AgragRunner:
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
        top_k: int,
        graph_top_k: int,
        link_top_k: int,
        ppr_alpha: float,
        mcmi_steps: int,
        max_context_tokens: int,
        embedding_provider: str,
        embedding_batch_size: int,
        reuse_index: bool = False,
        reuse_index_dir: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.use_es = use_es
        self.mode = mode
        self.top_k = top_k
        self.graph_top_k = graph_top_k
        self.link_top_k = link_top_k
        self.ppr_alpha = ppr_alpha
        self.mcmi_steps = mcmi_steps
        self.max_context_tokens = max_context_tokens
        self.embedding_provider = embedding_provider
        self.embedding_batch_size = embedding_batch_size
        self.llm = OpenAICompatibleClient()
        self.local_chunks = load_jsonl_list(chunks_path)
        build_started = time.time()
        if reuse_index:
            self.index = AgragIndex.load_cache(reuse_index_dir or artifact_dir, embedding_provider=embedding_provider)
            self.index.artifact_dir = artifact_dir
            self.offline_wall_time_seconds = 0.0
            self.index_metrics = self.index.write_artifacts()
            self.index_metrics["offline_reused"] = True
            if reuse_index_dir:
                self.index_metrics["offline_reuse_source_dir"] = str(reuse_index_dir)
        else:
            self.index = AgragIndex(
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
        self.index_metrics["cached_offline_wall_time_seconds"] = self.index.offline_embedding_wall_time_seconds if reuse_index else self.offline_wall_time_seconds
        self.index_metrics["offline_embedding_calls"] = self.index.offline_embedding_calls
        self.index_metrics["offline_embedding_wall_time_seconds"] = self.index.offline_embedding_wall_time_seconds
        (artifact_dir / "graph.json").write_text(json.dumps(self.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        retrieval_started = time.time()
        query_vector = self.index.embedding_provider.embed([question])[0]
        anchor_triples = self._select_anchor_triples(query_vector)
        ppr_started = time.time()
        ppr = self._ppr(anchor_triples)
        ppr_latency = time.time() - ppr_started
        subgraph_nodes = self._mcmi_subgraph(anchor_triples, ppr, query_vector)
        graph_chunks = self._chunks_from_subgraph(subgraph_nodes)
        hybrid_chunks = self._hybrid_retrieve(question)
        retrieved = _dedupe_chunks(graph_chunks + hybrid_chunks, self.top_k + self.graph_top_k)
        retrieval_latency = time.time() - retrieval_started
        context, used_chunks = join_context(retrieved, max_context_tokens=self.max_context_tokens)
        reasoning_paths = self._reasoning_paths(subgraph_nodes, ppr)
        prompt = (
            "AGRAG MCMI reasoning subgraph:\n"
            f"{reasoning_paths or 'No graph reasoning path selected.'}\n\n"
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
                    "tool": "agrag_ppr_mcmi_search",
                    "latency_seconds": retrieval_latency,
                    "output_summary": {
                        "anchor_triples": len(anchor_triples),
                        "subgraph_nodes": len(subgraph_nodes),
                        "retrieved_chunks": len(retrieved_chunks),
                        "graph_ppr_calls": 1,
                        "embedding_calls": 2 if self.mode != "bm25" else 1,
                        "ppr_latency_seconds": ppr_latency,
                    },
                },
                {
                    "event_type": "llm_call",
                    "stage": "agrag_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": input_tokens,
                    "output_tokens_estimate": output_tokens,
                },
            ],
        )

    def _select_anchor_triples(self, query_vector: list[float]) -> list[tuple[AgragTriple, float]]:
        scored = []
        for triple, vector in zip(self.index.triples, self.index.triple_vectors, strict=False):
            scored.append((triple, _dot(query_vector, vector)))
        return heapq.nlargest(self.link_top_k, scored, key=lambda item: item[1])

    def _ppr(self, anchors: list[tuple[AgragTriple, float]]) -> dict[str, float]:
        seeds: dict[str, float] = defaultdict(float)
        for triple, score in anchors:
            weight = max(score, 0.0) + 0.01
            seeds[_entity_key(triple.source)] += weight
            seeds[_entity_key(triple.target)] += weight
        if not seeds:
            return {}
        seed_total = sum(seeds.values()) or 1.0
        personalization = {node: value / seed_total for node, value in seeds.items()}
        scores = dict(personalization)
        nodes = list(self.index.adj)
        for _ in range(20):
            next_scores = {node: (1.0 - self.ppr_alpha) * personalization.get(node, 0.0) for node in nodes}
            for node in nodes:
                neighbors = self.index.adj.get(node, {})
                total_weight = sum(neighbors.values()) or 1.0
                mass = scores.get(node, 0.0) * self.ppr_alpha
                for neighbor, weight in neighbors.items():
                    next_scores[neighbor] = next_scores.get(neighbor, 0.0) + mass * (weight / total_weight)
            scores = next_scores
        return scores

    def _edge_cost(self, u: str, v: str, query_vector: list[float]) -> float:
        """Query-aware edge cost C_E = (1 - MS(q, f_edge)) / 2 (AGRAG Eq. 14).

        The edge feature f_edge is the relation triple connecting u and v; we reuse
        the triple embedding already computed at index-build time (no extra online
        embedding call), so the online cost accounting is unchanged. Cost is in
        [0,1]: edges whose relation is semantically close to the query are cheap.
        Falls back to the structural-weight form (1 - w) when no triple vector is
        available for the pair.
        """
        vec = self._edge_feature_vector(u, v)
        if vec is not None:
            ms = max(0.0, min(1.0, _dot(query_vector, vec)))
            return (1.0 - ms) / 2.0
        weight = self.index.adj.get(u, {}).get(v, 0.0)
        return max(1e-6, 1.0 - min(weight, 0.99))

    def _edge_feature_vector(self, u: str, v: str) -> list[float] | None:
        """Map an undirected edge (u,v) to the embedding of its relation triple,
        reusing index.triple_vectors. Built once and cached on the runner."""
        emap = getattr(self, "_edge_feature_map", None)
        if emap is None:
            emap = {}
            triples = getattr(self.index, "triples", []) or []
            tvecs = getattr(self.index, "triple_vectors", []) or []
            for triple, vec in zip(triples, tvecs, strict=False):
                a, b = _entity_key(triple.source), _entity_key(triple.target)
                if a and b:
                    key = (a, b) if a <= b else (b, a)
                    emap.setdefault(key, vec)  # first triple per endpoint pair
            self._edge_feature_map = emap
        key = (u, v) if u <= v else (v, u)
        return emap.get(key)

    def _mcst_init(self, terminals: list[str], query_vector: list[float]) -> set[str]:
        """Connect the anchor terminals into one subgraph via cheapest query-aware
        paths, approximating a Minimum-Cost Steiner Tree (AGRAG Algorithm 3,
        Mehlhorn-style: grow from the first terminal, attach each remaining
        terminal through the lowest query-aware-cost path found by Dijkstra over
        C_E edges). Guarantees terminal connectivity (Definition 1)."""
        import heapq as _hq

        terminals = [t for t in terminals if t in self.index.adj]
        if not terminals:
            return set()
        tset = set(terminals)
        tree: set[str] = {terminals[0]}
        for target in terminals[1:]:
            if target in tree:
                continue
            # Dijkstra from the current tree to `target` over query-aware edge costs.
            dist: dict[str, float] = {n: 0.0 for n in tree}
            prev: dict[str, str] = {}
            pq: list[tuple[float, str]] = [(0.0, n) for n in tree]
            _hq.heapify(pq)
            reached = False
            visited: set[str] = set()
            while pq:
                d, node = _hq.heappop(pq)
                if node in visited:
                    continue
                visited.add(node)
                if node == target:
                    reached = True
                    break
                for neighbor in self.index.adj.get(node, {}):
                    nd = d + self._edge_cost(node, neighbor, query_vector)
                    if nd < dist.get(neighbor, float("inf")):
                        dist[neighbor] = nd
                        prev[neighbor] = node
                        _hq.heappush(pq, (nd, neighbor))
            if reached:
                cur = target
                while cur in prev and cur not in tree:
                    tree.add(cur)
                    cur = prev[cur]
                tree.add(target)
            else:
                tree.add(target)  # disconnected component: still keep the terminal
        return tree | tset

    def _mcmi_subgraph(
        self,
        anchors: list[tuple[AgragTriple, float]],
        ppr: dict[str, float],
        query_vector: list[float],
    ) -> set[str]:
        """Maximum-Coverage / Minimum-Importance subgraph selection (AGRAG Alg. 3,
        Eq. 15). Initialize with a connected MCST over the anchor terminals, then
        greedily add the neighbor maximizing the cost-score ratio score(v)/C_E(u,v),
        i.e. gain per query-aware edge cost, rather than the degree-weight heuristic."""
        terminals = [
            key
            for triple, _ in anchors
            for key in (_entity_key(triple.source), _entity_key(triple.target))
            if key
        ]
        selected = self._mcst_init(terminals, query_vector)
        if not selected:
            return set()
        for _ in range(self.mcmi_steps):
            best_node = ""
            best_ratio = 0.0
            for node in list(selected):
                for neighbor in self.index.adj.get(node, {}):
                    if neighbor in selected:
                        continue
                    cost = self._edge_cost(node, neighbor, query_vector)
                    ratio = ppr.get(neighbor, 0.0) / max(1e-6, cost)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_node = neighbor
            if not best_node or best_ratio <= 0:
                break
            selected.add(best_node)
        return selected

    def _chunks_from_subgraph(self, nodes: set[str]) -> list[dict[str, Any]]:
        scored: dict[str, float] = defaultdict(float)
        for node in nodes:
            if node.startswith("chunk:"):
                scored[node.removeprefix("chunk:")] += 1.0
                continue
            for chunk_id in self.index.entity_chunks.get(node, set()):
                scored[chunk_id] += 1.0
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[: self.graph_top_k]
        return [{**self.index.chunks_by_id[chunk_id], "score": score, "score_source": "agrag_graph"} for chunk_id, score in ranked if chunk_id in self.index.chunks_by_id]

    def _hybrid_retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.use_es:
            return search_chunks(
                namespace=self.namespace,
                query=question,
                mode=self.mode,
                top_k=self.top_k,
                embedding_provider_name=self.embedding_provider,
            ).get("items", [])
        return _local_keyword_search(self.local_chunks, question, self.top_k)

    def _reasoning_paths(self, nodes: set[str], ppr: dict[str, float]) -> str:
        lines = []
        for left in sorted(nodes, key=lambda node: ppr.get(node, 0.0), reverse=True):
            if left.startswith("chunk:"):
                continue
            for right, _weight in sorted(self.index.adj.get(left, {}).items(), key=lambda item: ppr.get(item[0], 0.0), reverse=True):
                if right not in nodes or right.startswith("chunk:") or left > right:
                    continue
                label = self.index.edge_labels.get((left, right), "related_to")
                lines.append(f"({self.index.entity_names.get(left, left)} [SEP] {label} [SEP] {self.index.entity_names.get(right, right)})")
                if len(lines) >= self.graph_top_k:
                    return "\n".join(lines)
        return "\n".join(lines)


def run_agrag(
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
    top_k: int = 5,
    graph_top_k: int = 5,
    link_top_k: int = 8,
    ppr_alpha: float = 0.85,
    mcmi_steps: int = 20,
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
    runner = AgragRunner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        extractions_path=resolve_project_path(extractions_path or f"datasets/processed/{dataset}/semantic_llm.extractions.jsonl"),
        artifact_dir=resolve_project_path(artifact_dir or f"outputs/{dataset}/baselines/{METHOD}"),
        use_es=use_es,
        mode=mode,
        top_k=top_k,
        graph_top_k=graph_top_k,
        link_top_k=link_top_k,
        ppr_alpha=ppr_alpha,
        mcmi_steps=mcmi_steps,
        max_context_tokens=max_context_tokens,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size or int(os.environ.get("AGRAG_EMBED_BATCH_SIZE") or os.environ.get("BASELINE_EMBED_BATCH_SIZE", "32")),
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
            "retrieval": "agrag_ppr_mcmi_hybrid",
            "use_es": use_es,
            "mode": mode,
            "top_k": top_k,
            "graph_top_k": graph_top_k,
            "link_top_k": link_top_k,
            "ppr_alpha": ppr_alpha,
            "mcmi_steps": mcmi_steps,
            "embedding_provider": embedding_provider,
            "embedding_batch_size": runner.embedding_batch_size,
            "max_context_tokens": max_context_tokens,
            "index_metrics": runner.index_metrics,
            "prompt_style": "signpost_thought_answer_evidence_grounded",
            "offline_reused": reuse_index,
        },
    )


def _entity_key(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _relation_label(rel: dict[str, Any]) -> str:
    keywords = rel.get("keywords")
    if isinstance(keywords, list) and keywords:
        return ", ".join(str(item) for item in keywords[:3])
    return str(rel.get("relation") or rel.get("predicate") or "related_to")


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def _is_connection_refused(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "connection refused" in text or "errno 111" in text


def _local_keyword_search(chunks: list[dict[str, Any]], question: str, top_k: int) -> list[dict[str, Any]]:
    terms = [term for term in "".join(ch.lower() if ch.isalnum() else " " for ch in question).split() if len(term) > 1]
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
    parser = argparse.ArgumentParser(description="Run AGRAG baseline adapter.")
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--link-top-k", type=int, default=8)
    parser.add_argument("--ppr-alpha", type=float, default=0.85)
    parser.add_argument("--mcmi-steps", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=3500)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reuse-index-dir")
    args = parser.parse_args()
    count = run_agrag(
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
        top_k=args.top_k,
        graph_top_k=args.graph_top_k,
        link_top_k=args.link_top_k,
        ppr_alpha=args.ppr_alpha,
        mcmi_steps=args.mcmi_steps,
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
