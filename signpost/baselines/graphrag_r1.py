from __future__ import annotations

"""GraphRAG-R1 adapter over Signpost shared artifacts.

The adapter preserves the GraphRAG-R1 inference contract: an agent emits
``<think>...</think><answer>...</answer>`` and requests graph retrieval with
``<|begin_of_query|>...<|end_of_query|>``. The graph/index is baseline-owned
and built only from Signpost shared chunks and F6 semantic extractions.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signpost.baselines.agrag import AgragIndex, _dedupe_chunks, _dot, _entity_key
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
from signpost.llm.client import OpenAICompatibleClient, LLMConfig, load_llm_config


def _graphrag_r1_client() -> tuple["OpenAICompatibleClient", str]:
    """Build the chat client for the GraphRAG-R1 baseline.

    GraphRAG-R1's published contribution is an RL-trained policy released as a
    Qwen2.5-7B GRPO checkpoint (huggingface.co/yuchuanyue/GraphRAG-R1). To run the
    *trained* system rather than an untrained loop on the shared backbone, the
    baseline can be pointed at a dedicated endpoint serving that checkpoint via
    GRAPHRAG_R1_API_BASE / GRAPHRAG_R1_API_KEY / GRAPHRAG_R1_CHAT_MODEL, without
    disturbing the shared backbone used by every other method. If those are unset
    it falls back to the shared client (documented as an untrained-procedure run).
    Retrieval stays over the shared corpus either way. Returns (client, model_name)
    so the served model is recorded in the run metrics for honest reporting.
    """
    base = load_llm_config()
    api_base = os.environ.get("GRAPHRAG_R1_API_BASE")
    chat_model = os.environ.get("GRAPHRAG_R1_CHAT_MODEL")
    timeout = int(
        os.environ.get("GRAPHRAG_R1_TIMEOUT")
        or os.environ.get("BASELINE_LLM_TIMEOUT")
        or os.environ.get("LLM_TIMEOUT")
        or "300"
    )
    if not api_base and not chat_model:
        client = OpenAICompatibleClient(timeout=timeout)
        return client, client.config.chat_model
    cfg = LLMConfig(
        api_base=api_base or base.api_base,
        embedding_api_base=base.embedding_api_base,
        api_key=os.environ.get("GRAPHRAG_R1_API_KEY") or base.api_key,
        embedding_api_key=base.embedding_api_key,
        chat_model=chat_model or base.chat_model,
        reasoning_model=base.reasoning_model,
        embedding_model=base.embedding_model,
        rerank_model=base.rerank_model,
    )
    return OpenAICompatibleClient(config=cfg, timeout=timeout), cfg.chat_model
from signpost.retrieval.chunk_search import search_chunks


METHOD = "graphrag_r1"

SYSTEM_PROMPT = """You are a GraphRAG-R1-style agentic graph retrieval assistant. Answer the question in English strictly based on retrieved evidence from the private corpus. Preserve the GraphRAG-R1 output contract: use a single <think> block for reasoning and retrieval, request graph retrieval by writing <|begin_of_query|>query<|end_of_query|>, read returned evidence inside <|begin_of_documents|>...<|end_of_documents|>, and place the final answer in <answer> after </think>.

Follow these evidence-grounded answer rules:
- Use graph retrieval when evidence is needed.
- Include all necessary context and details supported by retrieved evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers in <answer>.
- Do not include conversational filler.
- If the retrieved evidence is insufficient, write exactly: "Insufficient evidence." inside <answer>.

Graph retrieval contract:
- To retrieve evidence, output exactly one query span using <|begin_of_query|> and <|end_of_query|>.
- The system will append graph facts and documents inside <|begin_of_documents|>...</|end_of_documents|>.
- After enough evidence is available, close </think> and write <answer>...</answer>.

Example:
<think>
The question asks what practices Greensgrow Farm uses for sustainable urban farming. I need graph-linked evidence about Greensgrow Farm practices.
<|begin_of_query|>Greensgrow Farm sustainable urban farming practices<|end_of_query|>
<|begin_of_documents|>
Graph facts:
(Greensgrow Farm [SEP] uses [SEP] hydroponic growing)
(Greensgrow Farm [SEP] uses [SEP] aquaponics)

Documents:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.
<|end_of_documents|>
The evidence identifies hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
</think>
<answer>Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.</answer>"""


@dataclass(frozen=True)
class GraphRAGR1Config:
    mode: str
    graph_top_k: int
    chunk_top_k: int
    link_top_k: int
    max_steps: int
    max_context_tokens: int
    ppr_alpha: float
    ppr_iterations: int
    embedding_provider: str
    embedding_batch_size: int
    use_es: bool


class GraphRAGR1Runner:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        extractions_path: Path,
        artifact_dir: Path,
        config: GraphRAGR1Config,
        reuse_index: bool = False,
        reuse_index_dir: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.artifact_dir = artifact_dir
        self.config = config
        self.llm, self.chat_model_used = _graphrag_r1_client()
        self.chunks = load_jsonl_list(chunks_path)
        started = time.time()
        if reuse_index:
            self.index = AgragIndex.load_cache(reuse_index_dir or artifact_dir, embedding_provider=config.embedding_provider)
            self.index.artifact_dir = artifact_dir
            self.offline_wall_time_seconds = 0.0
            self.index_metrics = self._write_graph_metrics(status="ready_reused")
            self.index_metrics["offline_reused"] = True
            if reuse_index_dir:
                self.index_metrics["offline_reuse_source_dir"] = str(reuse_index_dir)
            (self.artifact_dir / "graph.json").write_text(json.dumps(self.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            self.index = AgragIndex(
                chunks=self.chunks,
                extractions=load_jsonl_list(extractions_path),
                embedding_provider=config.embedding_provider,
                artifact_dir=artifact_dir,
                embedding_batch_size=config.embedding_batch_size,
            )
            self.index.save_cache()
            self.offline_wall_time_seconds = time.time() - started
            self.index_metrics = self._write_graph_metrics(status="ready")
            self.index_metrics["offline_reused"] = False
            (self.artifact_dir / "graph.json").write_text(json.dumps(self.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_graph_metrics(self, *, status: str) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        base = self.index.write_artifacts(status=status)
        metrics = {
            **base,
            "method": METHOD,
            "index_type": "graphrag_r1_local_agentic_graph_retrieval",
            "source_artifacts": ["chunks.jsonl", "semantic_llm.extractions.jsonl", "questions.jsonl"],
            "uses_signpost_graph_or_navigation_index": False,
            "uses_shared_signpost_chunk_es_index": bool(self.config.use_es),
            "chat_model_used": self.chat_model_used,
            "graphrag_r1_run_mode": (
                "released_trained_policy"
                if (os.environ.get("GRAPHRAG_R1_API_BASE") or os.environ.get("GRAPHRAG_R1_CHAT_MODEL"))
                else "untrained_procedure_shared_backbone"
            ),
            "dataset": self.dataset,
            "namespace": self.namespace,
            "offline_wall_time_seconds": self.offline_wall_time_seconds,
            "offline_embedding_calls": self.index.offline_embedding_calls,
            "offline_embedding_retries": self.index.offline_embedding_retries,
            "offline_embedding_failures": self.index.offline_embedding_failures,
            "offline_embedding_wall_time_seconds": self.index.offline_embedding_wall_time_seconds,
            "mode": self.config.mode,
            "graph_top_k": self.config.graph_top_k,
            "chunk_top_k": self.config.chunk_top_k,
            "link_top_k": self.config.link_top_k,
            "max_steps": self.config.max_steps,
            "ppr_alpha": self.config.ppr_alpha,
            "ppr_iterations": self.config.ppr_iterations,
        }
        (self.artifact_dir / "graph.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        transcript = f"Question: {question}\n<think>\n"
        all_retrieved: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        input_tokens = 0.0
        output_tokens = 0.0
        llm_calls = 0.0
        tool_calls = 0.0
        embedding_calls = 0.0
        graph_ppr_calls = 0.0
        retrieval_latency = 0.0
        ppr_latency = 0.0
        final_raw = ""
        evidence_chunks: list[dict[str, Any]] = []
        rejected_answer_before_evidence = False

        for step_index in range(1, self.config.max_steps + 1):
            step_prompt = (
                f"{transcript}\n"
                "Continue the GraphRAG-R1 reasoning trace. This is private-corpus QA: before giving a final "
                "answer, you should have retrieved graph/doc evidence. If no retrieved evidence is available yet, "
                "issue exactly one <|begin_of_query|>query<|end_of_query|> span. If the accumulated evidence is "
                "sufficient, close </think> and provide the final <answer>."
            )
            raw, in_tok, out_tok, llm_latency = chat_once(
                self.llm,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": step_prompt}],
                input_text=SYSTEM_PROMPT + "\n" + step_prompt,
            )
            input_tokens += in_tok
            output_tokens += out_tok
            llm_calls += 1.0
            trace.append(
                {
                    "event_type": "llm_call",
                    "stage": "graphrag_r1_agent_step",
                    "step": step_index,
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )
            transcript += _strip_duplicate_think_prefix(raw).strip() + "\n"
            final_raw = transcript
            graph_query = _extract_last_graph_query(raw)
            if _extract_answer(transcript) and all_retrieved and not graph_query:
                break
            if _extract_answer(transcript) and not all_retrieved and not graph_query:
                rejected_answer_before_evidence = True
                transcript = _remove_answer_tags(transcript)
                graph_query = question
                transcript += "No graph/doc evidence has been retrieved yet, so an initial graph query is required.\n"
                trace.append(
                    {
                        "event_type": "control",
                        "stage": "graphrag_r1_forced_initial_query",
                        "step": step_index,
                        "reason": "model_answered_before_retrieving_evidence",
                    }
                )
            if not graph_query:
                transcript += "No graph retrieval query was issued in this step.\n"
                continue

            search_started = time.time()
            search = self._graph_search(graph_query)
            search_latency = time.time() - search_started
            retrieval_latency += search_latency
            ppr_latency += search["ppr_latency_seconds"]
            tool_calls += 1.0
            graph_ppr_calls += 1.0
            embedding_calls += 1.0
            if self.config.use_es and self.config.mode in {"dense", "hybrid"}:
                embedding_calls += 1.0
            all_retrieved.extend(search["chunks"])
            context, used = join_context(search["chunks"], max_context_tokens=self.config.max_context_tokens)
            evidence_chunks.extend(_evidence_chunk_rows(used, round_index=step_index, query=graph_query, source="retrieval_context"))
            documents = (
                "<|begin_of_documents|>\n"
                "Graph facts:\n"
                f"{search['facts'] or 'No graph facts selected.'}\n\n"
                "Documents:\n"
                f"{context}\n"
                "<|end_of_documents|>\n"
            )
            transcript += documents
            trace.append(
                {
                    "event_type": "tool_call",
                    "tool": "graphrag_r1_graph_search",
                    "step": step_index,
                    "query": graph_query,
                    "latency_seconds": search_latency,
                    "output_summary": {
                        "anchor_triples": search["anchor_triples"],
                        "graph_nodes": search["graph_nodes"],
                        "retrieved_chunks": len(used),
                        "ppr_latency_seconds": search["ppr_latency_seconds"],
                        "mode": self.config.mode,
                        "use_es": self.config.use_es,
                    },
                }
            )

        answer = _extract_answer(final_raw)
        if answer and rejected_answer_before_evidence:
            answer = ""
        if not answer:
            final_prompt = (
                f"{transcript}\n"
                "Now close the GraphRAG-R1 reasoning trace and write the final <answer>. Use only accumulated "
                "retrieved graph facts and documents. If they are insufficient, write exactly "
                "<answer>Insufficient evidence.</answer>."
            )
            raw, in_tok, out_tok, llm_latency = chat_once(
                self.llm,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": final_prompt}],
                input_text=SYSTEM_PROMPT + "\n" + final_prompt,
            )
            input_tokens += in_tok
            output_tokens += out_tok
            llm_calls += 1.0
            final_raw = transcript + raw
            answer = _extract_answer(final_raw) or raw.strip()
            trace.append(
                {
                    "event_type": "llm_call",
                    "stage": "graphrag_r1_final_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )

        if not evidence_chunks:
            answer = "Insufficient evidence."

        retrieved = _dedupe_chunks(all_retrieved, self.config.chunk_top_k * max(1, self.config.max_steps))
        citations = [
            {"file_name": item.get("file_name"), "start_line": item.get("start_line"), "end_line": item.get("end_line"), "locate": locate_from_chunk(item)}
            for item in retrieved
            if locate_from_chunk(item)
        ]
        retrieved_chunks = [
            {"chunk_id": str(item.get("chunk_id") or ""), "doc_id": item.get("doc_id"), "score": item.get("score"), "score_source": item.get("score_source")}
            for item in retrieved
            if item.get("chunk_id")
        ]
        return BaselineResult(
            answer=answer.strip(),
            rationale=_extract_think(final_raw),
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            evidence_chunks=evidence_chunks,
            trace=trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            embedding_calls=embedding_calls,
            graph_ppr_calls=graph_ppr_calls,
            ppr_latency_seconds=ppr_latency,
            retrieval_latency_seconds=retrieval_latency,
        )

    def _graph_search(self, query: str) -> dict[str, Any]:
        query_vector = self.index.embedding_provider.embed([query])[0]
        anchors = self._select_anchor_triples(query_vector)
        ppr_started = time.time()
        ppr = self._ppr(anchors)
        ppr_latency = time.time() - ppr_started
        nodes = self._expand_nodes(anchors, ppr)
        graph_chunks = self._chunks_from_nodes(nodes)
        chunk_hits = self._chunk_retrieve(query)
        chunks = _dedupe_chunks(graph_chunks + chunk_hits, self.config.graph_top_k + self.config.chunk_top_k)
        return {
            "anchor_triples": len(anchors),
            "graph_nodes": len(nodes),
            "ppr_latency_seconds": ppr_latency,
            "facts": self._facts(nodes, ppr),
            "chunks": chunks,
        }

    def _select_anchor_triples(self, query_vector: list[float]) -> list[tuple[Any, float]]:
        scored = [(triple, _dot(query_vector, vector)) for triple, vector in zip(self.index.triples, self.index.triple_vectors, strict=False)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[: self.config.link_top_k]

    def _ppr(self, anchors: list[tuple[Any, float]]) -> dict[str, float]:
        seeds: dict[str, float] = defaultdict(float)
        for triple, score in anchors:
            weight = max(float(score), 0.0) + 0.01
            seeds[_entity_key(triple.source)] += weight
            seeds[_entity_key(triple.target)] += weight
        if not seeds:
            return {}
        seed_total = sum(seeds.values()) or 1.0
        personalization = {node: value / seed_total for node, value in seeds.items()}
        scores = dict(personalization)
        nodes = list(self.index.adj)
        for _ in range(self.config.ppr_iterations):
            next_scores = {node: (1.0 - self.config.ppr_alpha) * personalization.get(node, 0.0) for node in nodes}
            for node in nodes:
                neighbors = self.index.adj.get(node, {})
                total_weight = sum(neighbors.values()) or 1.0
                mass = scores.get(node, 0.0) * self.config.ppr_alpha
                for neighbor, weight in neighbors.items():
                    next_scores[neighbor] = next_scores.get(neighbor, 0.0) + mass * (weight / total_weight)
            scores = next_scores
        return scores

    def _expand_nodes(self, anchors: list[tuple[Any, float]], ppr: dict[str, float]) -> set[str]:
        selected = {_entity_key(triple.source) for triple, _ in anchors} | {_entity_key(triple.target) for triple, _ in anchors}
        selected = {node for node in selected if node}
        ranked = [node for node, _score in sorted(ppr.items(), key=lambda item: item[1], reverse=True)]
        for node in ranked:
            if len(selected) >= self.config.graph_top_k * 2:
                break
            if node and not node.startswith("chunk:"):
                selected.add(node)
        return selected

    def _chunks_from_nodes(self, nodes: set[str]) -> list[dict[str, Any]]:
        scored: dict[str, float] = defaultdict(float)
        for node in nodes:
            for chunk_id in self.index.entity_chunks.get(node, set()):
                scored[chunk_id] += 1.0
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[: self.config.graph_top_k]
        return [
            {**self.index.chunks_by_id[chunk_id], "score": score, "score_source": "graphrag_r1_graph"}
            for chunk_id, score in ranked
            if chunk_id in self.index.chunks_by_id
        ]

    def _chunk_retrieve(self, query: str) -> list[dict[str, Any]]:
        if self.config.use_es:
            return search_chunks(
                namespace=self.namespace,
                query=query,
                mode=self.config.mode,
                top_k=self.config.chunk_top_k,
                embedding_provider_name=self.config.embedding_provider,
            ).get("items", [])
        return _local_keyword_search(self.chunks, query, self.config.chunk_top_k)

    def _facts(self, nodes: set[str], ppr: dict[str, float]) -> str:
        lines = []
        for left in sorted(nodes, key=lambda node: ppr.get(node, 0.0), reverse=True):
            if left.startswith("chunk:"):
                continue
            for right, _weight in sorted(self.index.adj.get(left, {}).items(), key=lambda item: ppr.get(item[0], 0.0), reverse=True):
                if right not in nodes or right.startswith("chunk:") or left > right:
                    continue
                label = self.index.edge_labels.get((left, right), "related_to")
                lines.append(f"({self.index.entity_names.get(left, left)} [SEP] {label} [SEP] {self.index.entity_names.get(right, right)})")
                if len(lines) >= self.config.graph_top_k:
                    return "\n".join(lines)
        return "\n".join(lines)


def run_graphrag_r1(
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
    graph_top_k: int = 5,
    chunk_top_k: int = 5,
    link_top_k: int = 8,
    max_steps: int = 4,
    max_context_tokens: int = 2500,
    ppr_alpha: float = 0.85,
    ppr_iterations: int = 20,
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
    config = GraphRAGR1Config(
        mode=mode,
        graph_top_k=graph_top_k,
        chunk_top_k=chunk_top_k,
        link_top_k=link_top_k,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        ppr_alpha=ppr_alpha,
        ppr_iterations=ppr_iterations,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size
        or int(os.environ.get("GRAPHRAG_R1_EMBED_BATCH_SIZE") or os.environ.get("BASELINE_EMBED_BATCH_SIZE", "32")),
        use_es=use_es,
    )
    runner = GraphRAGR1Runner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        extractions_path=resolve_project_path(extractions_path or f"datasets/processed/{dataset}/semantic_llm.extractions.jsonl"),
        artifact_dir=resolve_project_path(artifact_dir or f"outputs/{dataset}/baselines/{METHOD}"),
        config=config,
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
            "retrieval": "graphrag_r1_agentic_graph_search",
            "use_es": use_es,
            "mode": mode,
            "graph_top_k": graph_top_k,
            "chunk_top_k": chunk_top_k,
            "link_top_k": link_top_k,
            "max_steps": max_steps,
            "embedding_provider": embedding_provider,
            "embedding_batch_size": config.embedding_batch_size,
            "max_context_tokens": max_context_tokens,
            "index_metrics": runner.index_metrics,
            "prompt_style": "graphrag_r1_xml_query_tags_signpost_evidence_grounded",
            "offline_reused": reuse_index,
        },
    )


def _extract_last_graph_query(text: str) -> str:
    patterns = [
        r"<\|\s*begin_of_query\s*\|>(.*?)<\|\s*end_of_query\s*\|>",
        r"<\s*begin_of_query\s*>(.*?)<\s*/\s*end_of_query\s*>",
        r"(?:^|\n)\s*Graph\s+query\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)\s*Query\s*:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return _clean_text(matches[-1])
    return ""


def _extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return _clean_text(matches[-1]) if matches else ""


def _extract_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return _clean_text(match.group(1)) if match else ""


def _strip_duplicate_think_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith("<think>"):
        return re.sub(r"^\s*<think>\s*", "", stripped, flags=re.IGNORECASE)
    return stripped


def _remove_answer_tags(text: str) -> str:
    text = re.sub(r"</think>\s*<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _evidence_chunk_rows(chunks: list[dict[str, Any]], *, round_index: int, query: str, source: str) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(chunks, start=1):
        rows.append(
            {
                "rank": rank,
                "round": round_index,
                "source": source,
                "query": query,
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "file_name": item.get("file_name"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
                "content_preview": str(item.get("content") or "")[:240],
            }
        )
    return rows


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(term) > 1]


def _local_keyword_search(chunks: list[dict[str, Any]], question: str, top_k: int) -> list[dict[str, Any]]:
    terms = _terms(question)
    scored = []
    for item in chunks:
        content = str(item.get("content") or "")
        score = sum(content.lower().count(term) for term in terms)
        if score > 0:
            scored.append({**item, "score": float(score), "score_source": "graphrag_r1_local_keyword"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GraphRAG-R1 baseline adapter.")
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
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--link-top-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--ppr-alpha", type=float, default=0.85)
    parser.add_argument("--ppr-iterations", type=int, default=20)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reuse-index-dir")
    args = parser.parse_args()
    count = run_graphrag_r1(
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
        graph_top_k=args.graph_top_k,
        chunk_top_k=args.chunk_top_k,
        link_top_k=args.link_top_k,
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        ppr_alpha=args.ppr_alpha,
        ppr_iterations=args.ppr_iterations,
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
