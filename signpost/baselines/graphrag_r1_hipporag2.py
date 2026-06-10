from __future__ import annotations

"""GraphRAG-R1 adapter backed by the official HippoRAG2 retrieval server."""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from signpost.baselines.agrag import _dedupe_chunks
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
from signpost.baselines.graphrag_r1 import (
    SYSTEM_PROMPT,
    _evidence_chunk_rows,
    _extract_answer,
    _extract_last_graph_query,
    _extract_think,
    _graphrag_r1_client,
    _remove_answer_tags,
    _strip_duplicate_think_prefix,
)
from signpost.config.context import resolve_project_path


METHOD = "graphrag_r1_hipporag2"


@dataclass(frozen=True)
class GraphRAGR1HippoRAG2Config:
    hipporag_url: str
    retrieval_num: int
    max_steps: int
    max_context_tokens: int
    server_timeout_seconds: float
    skip_server_health_check: bool


class GraphRAGR1HippoRAG2Runner:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        artifact_dir: Path,
        config: GraphRAGR1HippoRAG2Config,
        extractions_path: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.artifact_dir = artifact_dir
        self.config = config
        self.llm, self.chat_model_used = _graphrag_r1_client()
        self.chunks = load_jsonl_list(chunks_path)
        self.chunks_by_passage = {_chunk_content(item): item for item in self.chunks if _chunk_content(item)}
        self.chunks_by_normalized_passage = {
            _normalize_passage(_chunk_content(item)): item for item in self.chunks if _chunk_content(item)
        }
        self.openie_path = os.environ.get("GRAPHRAG_R1_HIPPORAG2_OPENIE_PATH") or os.environ.get("DATA_PATH") or ""
        self.save_dir = os.environ.get("GRAPHRAG_R1_HIPPORAG2_SAVE_DIR") or os.environ.get("SAVE_DIR") or ""
        self.openie_stats = _load_openie_stats(self.openie_path)
        self.index_metrics = self._write_graph_metrics(
            chunks_path=chunks_path,
            extractions_path=extractions_path,
            server_status={} if config.skip_server_health_check else self._server_get("/health"),
            server_stats={} if config.skip_server_health_check else self._server_get("/stats"),
        )

    def _write_graph_metrics(
        self,
        *,
        chunks_path: Path,
        extractions_path: Path | None,
        server_status: dict[str, Any],
        server_stats: dict[str, Any],
    ) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        estimated_embedding_calls = _estimated_offline_embedding_calls(self.openie_stats)
        metrics = {
            "method": METHOD,
            "dataset": self.dataset,
            "namespace": self.namespace,
            "index_type": "official_hipporag2_retrieval_server_over_signpost_f6_openie",
            "retrieval_backend": "GraphRAG-R1 server/HippoRAG2 /query",
            "source_artifacts": ["chunks.jsonl", "semantic_llm.extractions.jsonl", "questions.jsonl"],
            "chunks_path": str(chunks_path),
            "extractions_path": str(extractions_path) if extractions_path else None,
            "uses_signpost_graph_or_navigation_index": False,
            "uses_shared_signpost_chunk_es_index": False,
            "openie_source": "converted_signpost_f6_semantic_llm_extractions",
            "openie_path": self.openie_path,
            "hipporag_save_dir": self.save_dir,
            "hipporag_server_url": self.config.hipporag_url,
            "hipporag_server_status": server_status,
            "hipporag_server_stats": server_stats,
            "chat_model_used": self.chat_model_used,
            "graphrag_r1_run_mode": (
                "released_trained_policy"
                if (os.environ.get("GRAPHRAG_R1_API_BASE") or os.environ.get("GRAPHRAG_R1_CHAT_MODEL"))
                else "untrained_procedure_shared_backbone"
            ),
            "retrieval_num": self.config.retrieval_num,
            "max_steps": self.config.max_steps,
            "max_context_tokens": self.config.max_context_tokens,
            "offline_wall_time_seconds": _float_env("GRAPHRAG_R1_HIPPORAG2_OFFLINE_WALL_SECONDS", 0.0),
            "offline_embedding_calls": _float_env(
                "GRAPHRAG_R1_HIPPORAG2_OFFLINE_EMBEDDING_CALLS", estimated_embedding_calls
            ),
            "offline_embedding_calls_estimated": estimated_embedding_calls,
            "offline_reused": True,
            **self.openie_stats,
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
                    "stage": "graphrag_r1_hipporag2_agent_step",
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
                        "stage": "graphrag_r1_hipporag2_forced_initial_query",
                        "step": step_index,
                        "reason": "model_answered_before_retrieving_evidence",
                    }
                )
            if not graph_query:
                transcript += "No graph retrieval query was issued in this step.\n"
                continue

            search_started = time.time()
            search = self._hipporag_search(graph_query)
            search_latency = time.time() - search_started
            retrieval_latency += search_latency
            tool_calls += 1.0
            embedding_calls += 1.0
            graph_ppr_calls += 1.0
            all_retrieved.extend(search["chunks"])
            context, used = join_context(search["chunks"], max_context_tokens=self.config.max_context_tokens)
            evidence_chunks.extend(
                _evidence_chunk_rows(used, round_index=step_index, query=graph_query, source="hipporag2_retrieval_context")
            )
            documents = (
                "<|begin_of_documents|>\n"
                "Graph facts:\n"
                f"{search['facts_text'] or 'No graph facts selected.'}\n\n"
                "Documents:\n"
                f"{context}\n"
                "<|end_of_documents|>\n"
            )
            transcript += documents
            trace.append(
                {
                    "event_type": "tool_call",
                    "tool": "graphrag_r1_hipporag2_query",
                    "step": step_index,
                    "query": graph_query,
                    "latency_seconds": search_latency,
                    "output_summary": {
                        "facts": len(search["facts"]),
                        "retrieved_chunks": len(used),
                        "retrieval_num": self.config.retrieval_num,
                        "hipporag_server_url": self.config.hipporag_url,
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
                    "stage": "graphrag_r1_hipporag2_final_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )

        if not evidence_chunks:
            answer = "Insufficient evidence."

        retrieved = _dedupe_chunks(all_retrieved, self.config.retrieval_num * max(1, self.config.max_steps))
        citations = [
            {
                "file_name": item.get("file_name"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "locate": locate_from_chunk(item),
            }
            for item in retrieved
            if locate_from_chunk(item)
        ]
        retrieved_chunks = [
            {
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
            }
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

    def _hipporag_search(self, query: str) -> dict[str, Any]:
        payload = {"query": [query], "retrieval_num": self.config.retrieval_num}
        response = self._server_post("/query", payload)
        result_rows = response.get("result") if isinstance(response, dict) else None
        first = result_rows[0] if isinstance(result_rows, list) and result_rows else {}
        docs = first.get("docs") if isinstance(first.get("docs"), list) else []
        facts = first.get("facts") if isinstance(first.get("facts"), list) else []
        chunks = [self._chunk_from_doc(doc, rank=rank) for rank, doc in enumerate(docs, start=1)]
        return {
            "facts": facts,
            "facts_text": _format_facts(facts),
            "chunks": chunks,
        }

    def _chunk_from_doc(self, doc: Any, *, rank: int) -> dict[str, Any]:
        passage = str(doc or "").strip()
        chunk = self.chunks_by_passage.get(passage) or self.chunks_by_normalized_passage.get(_normalize_passage(passage))
        if chunk is not None:
            return {**chunk, "score": float(max(0, self.config.retrieval_num - rank + 1)), "score_source": "hipporag2_server"}
        return {
            "chunk_id": "",
            "doc_id": None,
            "file_name": None,
            "start_line": None,
            "end_line": None,
            "content": passage,
            "score": float(max(0, self.config.retrieval_num - rank + 1)),
            "score_source": "hipporag2_server_unmapped_doc",
        }

    def _server_get(self, path: str) -> dict[str, Any]:
        url = self.config.hipporag_url.rstrip("/") + path
        try:
            with request.urlopen(url, timeout=self.config.server_timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"HippoRAG2 server check failed url={url}: {exc}") from exc

    def _server_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.hipporag_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=self.config.server_timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"HippoRAG2 server query failed url={url}: {exc}") from exc


def run_graphrag_r1_hipporag2(
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
    hipporag_url: str | None = None,
    retrieval_num: int = 5,
    max_steps: int = 4,
    max_context_tokens: int = 2500,
    server_timeout_seconds: float = 300.0,
    skip_server_health_check: bool = False,
    workers: int | None = None,
) -> int:
    paths = build_paths(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method=METHOD,
    )
    config = GraphRAGR1HippoRAG2Config(
        hipporag_url=hipporag_url or os.environ.get("GRAPHRAG_R1_HIPPORAG2_URL", "http://127.0.0.1:8090"),
        retrieval_num=retrieval_num,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        server_timeout_seconds=server_timeout_seconds,
        skip_server_health_check=skip_server_health_check,
    )
    runner = GraphRAGR1HippoRAG2Runner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        extractions_path=resolve_project_path(extractions_path) if extractions_path else None,
        artifact_dir=resolve_project_path(artifact_dir or f"outputs/{dataset}/baselines/{METHOD}"),
        config=config,
    )
    return run_baseline_batch(
        method=METHOD,
        paths=paths,
        answer_fn=runner.answer,
        limit=limit,
        workers=workers,
        metadata={
            "retrieval": "graphrag_r1_official_hipporag2_server",
            "hipporag_url": config.hipporag_url,
            "retrieval_num": config.retrieval_num,
            "max_steps": config.max_steps,
            "max_context_tokens": config.max_context_tokens,
            "index_metrics": runner.index_metrics,
            "prompt_style": "graphrag_r1_xml_query_tags_signpost_evidence_grounded",
            "offline_reused": True,
        },
    )


def _chunk_content(chunk: dict[str, Any]) -> str:
    for key in ("content", "text", "passage"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_passage(text: str) -> str:
    return " ".join(str(text or "").split())


def _format_facts(facts: list[Any]) -> str:
    lines = []
    for fact in facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 3:
            lines.append(f"({fact[0]} [SEP] {fact[1]} [SEP] {fact[2]})")
        elif fact:
            lines.append(str(fact))
    return "\n".join(lines)


def _load_openie_stats(path: str) -> dict[str, Any]:
    if not path:
        return {}
    openie_path = Path(path)
    if not openie_path.exists():
        return {"openie_stats_error": f"missing openie file: {path}"}
    payload = json.loads(openie_path.read_text(encoding="utf-8"))
    docs = payload.get("docs") if isinstance(payload.get("docs"), list) else []
    unique_entities = set()
    unique_triples = set()
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for entity in doc.get("extracted_entities") or []:
            unique_entities.add(str(entity))
        for triple in doc.get("extracted_triples") or []:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                unique_triples.add(tuple(str(item) for item in triple[:3]))
    return {
        "hipporag_openie_docs": len(docs),
        "hipporag_openie_unique_entities": len(unique_entities),
        "hipporag_openie_unique_triples": len(unique_triples),
        "hipporag_openie_total_entities": payload.get("num_entities", 0),
        "hipporag_openie_total_triples": payload.get("num_triples", 0),
    }


def _estimated_offline_embedding_calls(stats: dict[str, Any]) -> float:
    batch_size = max(1, int(os.environ.get("GRAPHRAG_R1_HIPPORAG2_EMBED_BATCH_SIZE", "32") or 32))
    docs = int(stats.get("hipporag_openie_docs", 0) or 0)
    entities = int(stats.get("hipporag_openie_unique_entities", 0) or 0)
    triples = int(stats.get("hipporag_openie_unique_triples", 0) or 0)
    return float(math.ceil(docs / batch_size) + math.ceil(entities / batch_size) + math.ceil(triples / batch_size))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GraphRAG-R1 with official HippoRAG2 retrieval server.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--extractions")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hipporag-url", default=None)
    parser.add_argument("--retrieval-num", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--server-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--skip-server-health-check", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    count = run_graphrag_r1_hipporag2(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        output_path=args.output,
        query_log_path=args.query_log,
        artifact_dir=args.artifact_dir,
        limit=args.limit,
        hipporag_url=args.hipporag_url,
        retrieval_num=args.retrieval_num,
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        server_timeout_seconds=args.server_timeout_seconds,
        skip_server_health_check=args.skip_server_health_check,
        workers=args.workers,
    )
    print(f"wrote {count} {METHOD} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
