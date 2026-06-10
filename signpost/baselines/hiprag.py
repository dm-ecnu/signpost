from __future__ import annotations

"""HiPRAG adapter over Signpost shared chunks.

The adapter preserves HiPRAG's inference-time contract: an agent alternates
XML-formatted reasoning steps with a search tool and emits a final ``<answer>``.
The search tool is backed by a baseline-owned local chunk retrieval index built
from ``chunks.jsonl``; it does not read Signpost graph/index artifacts.
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
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
from signpost.llm.client import OpenAICompatibleClient, LLMConfig, load_llm_config


def _hiprag_client() -> tuple["OpenAICompatibleClient", str]:
    """Build the chat client for the HiPRAG baseline.

    HiPRAG's published contribution is an RL-trained policy released as Qwen2.5-7B
    / Llama-3.2-3B checkpoints (github.com/qualidea1217/HiPRAG). To run the trained
    system, point the baseline at a dedicated endpoint serving the released
    HiPRAG-7B checkpoint via HIPRAG_API_BASE / HIPRAG_API_KEY / HIPRAG_CHAT_MODEL,
    without disturbing the shared backbone used by other methods. HiPRAG uses flat
    passage retrieval (Search-R1 style); with USE_ES=1 its search engine is the
    shared Elasticsearch chunk index. If the overrides are unset, falls back to the
    shared client (documented as an untrained-procedure run). Returns
    (client, model_name) so the served model is recorded in run metrics.
    """
    base = load_llm_config()
    api_base = os.environ.get("HIPRAG_API_BASE")
    chat_model = os.environ.get("HIPRAG_CHAT_MODEL")
    timeout = int(
        os.environ.get("HIPRAG_TIMEOUT")
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
        api_key=os.environ.get("HIPRAG_API_KEY") or base.api_key,
        embedding_api_key=base.embedding_api_key,
        chat_model=chat_model or base.chat_model,
        reasoning_model=base.reasoning_model,
        embedding_model=base.embedding_model,
        rerank_model=base.rerank_model,
    )
    return OpenAICompatibleClient(config=cfg, timeout=timeout), cfg.chat_model
from signpost.retrieval.chunk_search import search_chunks


METHOD = "hiprag"

SYSTEM_PROMPT = """You are a HiPRAG-style agentic retrieval assistant. Answer the question in English strictly based on retrieved evidence from the private corpus. You must preserve HiPRAG's XML output contract: reasoning and tool-use steps go inside a single <think> block, each step uses <step>, <reasoning>, optional <search>, optional <context>, and <conclusion>, and the final response is placed in <answer> after </think>.

Follow these evidence-grounded answer rules:
- Use the search tool to inspect the private corpus when evidence is needed.
- Include all necessary context and details supported by retrieved evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers in <answer>.
- Do not include conversational filler.
- If the retrieved evidence is insufficient, write exactly: "Insufficient evidence." inside <answer>.

Search tool contract:
- To search the corpus, output a single <search>query</search> tag inside the current <step>.
- The system will append retrieved evidence inside <context>...</context>.
- After enough evidence is available, close </think> and write <answer>...</answer>.

Example:
<think>
<step>
    <reasoning>The question asks what practices Greensgrow Farm uses for sustainable urban farming. I need corpus evidence listing those practices.</reasoning>
    <search>Greensgrow Farm sustainable urban farming practices</search>
    <context>Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.</context>
    <conclusion>The evidence identifies hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.</conclusion>
</step>
</think>
<answer>Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.</answer>"""


@dataclass(frozen=True)
class HiPRAGConfig:
    mode: str
    search_top_k: int
    max_steps: int
    max_context_tokens: int
    embedding_provider: str
    embedding_batch_size: int
    use_es: bool


class HiPRAGSearchIndex:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks: list[dict[str, Any]],
        artifact_dir: Path,
        config: HiPRAGConfig,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.chunks = chunks
        self.artifact_dir = artifact_dir
        self.config = config
        self.embedding_provider = None
        self.chunk_vectors: list[list[float]] = []
        self.offline_embedding_calls = 0
        self.offline_embedding_retries = 0
        self.offline_embedding_failures = 0
        self.offline_embedding_wall_time_seconds = 0.0
        self.offline_wall_time_seconds = 0.0
        self.index_metrics: dict[str, Any] = {}
        self._build()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["embedding_provider"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.embedding_provider = (
            create_embedding_provider(self.config.embedding_provider)
            if not self.config.use_es and self.config.mode in {"dense", "hybrid"}
            else None
        )

    @classmethod
    def load_cache(cls, artifact_dir: Path, *, config: HiPRAGConfig) -> "HiPRAGSearchIndex":
        cache_path = artifact_dir / "index.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"HiPRAG index cache not found: {cache_path}")
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError(f"HiPRAG index cache has wrong type: {type(index)!r}")
        if index.config.embedding_provider != config.embedding_provider or index.config.mode != config.mode or index.config.use_es != config.use_es:
            raise ValueError("HiPRAG index cache config does not match requested mode/use_es/embedding_provider")
        index.artifact_dir = artifact_dir
        index.config = config
        index.embedding_provider = (
            create_embedding_provider(config.embedding_provider)
            if not config.use_es and config.mode in {"dense", "hybrid"}
            else None
        )
        return index

    def save_cache(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with (self.artifact_dir / "index.pkl").open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build(self) -> None:
        started = time.time()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.use_es and self.config.mode in {"dense", "hybrid"}:
            self.embedding_provider = create_embedding_provider(self.config.embedding_provider)
            texts = [str(item.get("content") or "") for item in self.chunks]
            vectors: list[list[float]] = []
            batch_size = max(1, self.config.embedding_batch_size)
            total_batches = math.ceil(len(texts) / batch_size)
            embed_started = time.time()
            for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
                batch = texts[start : start + batch_size]
                vectors.extend(self._embed_batch_with_retry(batch, label="chunk"))
                if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                    elapsed = time.time() - embed_started
                    print(
                        f"[hiprag] embedded chunk batch {batch_index}/{total_batches} "
                        f"vectors={len(vectors)}/{len(texts)} elapsed_seconds={elapsed:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.chunk_vectors = vectors
                    self.offline_embedding_wall_time_seconds = elapsed
                    self.write_artifacts(status="embedding_chunks")
            self.chunk_vectors = vectors
            self.offline_embedding_wall_time_seconds = time.time() - embed_started
        self.offline_wall_time_seconds = time.time() - started
        self.index_metrics = self.write_artifacts(status="ready")

    def _embed_batch_with_retry(self, batch: list[str], *, label: str) -> list[list[float]]:
        retries = max(1, int(os.environ.get("HIPRAG_EMBED_RETRIES", "3")))
        retry_sleep = max(0.0, float(os.environ.get("HIPRAG_EMBED_RETRY_SLEEP", "5")))
        for attempt in range(1, retries + 1):
            try:
                vectors = self.embedding_provider.embed(batch)  # type: ignore[union-attr]
                self.offline_embedding_calls += 1
                return _normalize_vectors(vectors)
            except Exception as exc:
                self.offline_embedding_failures += 1
                if _is_connection_refused(exc):
                    if attempt < retries:
                        self.offline_embedding_retries += 1
                        print(
                            f"[hiprag] embedding service connection refused for {label} batch_size={len(batch)} "
                            f"attempt={attempt}/{retries}; retrying after {retry_sleep:.1f}s. "
                            "Check that the H200 embedding service is listening before rerunning.",
                            file=sys.stderr,
                            flush=True,
                        )
                        if retry_sleep:
                            time.sleep(retry_sleep)
                        continue
                    raise RuntimeError(
                        "HiPRAG embedding service is not reachable after retries. "
                        "This is a service availability/configuration failure, not a batch-size failure. "
                        "Check ECNU_EMBEDDING_API_BASE or OPENAI_EMBEDDING_API_BASE and ensure the H200 "
                        "embedding server is listening on the configured host/port."
                    ) from exc
                if attempt < retries:
                    self.offline_embedding_retries += 1
                    print(
                        f"[hiprag] embedding {label} batch_size={len(batch)} failed "
                        f"attempt={attempt}/{retries}: {exc}; retrying after {retry_sleep:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if retry_sleep:
                        time.sleep(retry_sleep)
                    continue
                if len(batch) == 1:
                    print(
                        f"[hiprag] embedding {label} single item failed after {retries} attempts",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                midpoint = max(1, len(batch) // 2)
                print(
                    f"[hiprag] embedding {label} batch_size={len(batch)} failed after {retries} attempts; "
                    f"splitting into {midpoint}+{len(batch) - midpoint}",
                    file=sys.stderr,
                    flush=True,
                )
                return self._embed_batch_with_retry(batch[:midpoint], label=label) + self._embed_batch_with_retry(batch[midpoint:], label=label)
        raise RuntimeError("unreachable embedding retry state")

    def write_artifacts(self, *, status: str) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "status": status,
            "method": METHOD,
            "index_type": "hiprag_local_agentic_chunk_retrieval",
            "source_artifacts": ["chunks.jsonl", "questions.jsonl"],
            "uses_signpost_graph_or_navigation_index": False,
            "uses_shared_signpost_chunk_es_index": bool(self.config.use_es),
            "dataset": self.dataset,
            "namespace": self.namespace,
            "mode": self.config.mode,
            "chunk_documents": len(self.chunks),
            "embedded_chunks": len(self.chunk_vectors),
            "offline_embedding_calls": self.offline_embedding_calls,
            "offline_embedding_retries": self.offline_embedding_retries,
            "offline_embedding_failures": self.offline_embedding_failures,
            "offline_embedding_wall_time_seconds": self.offline_embedding_wall_time_seconds,
            "offline_wall_time_seconds": self.offline_wall_time_seconds,
            "embedding_provider": self.config.embedding_provider,
            "embedding_batch_size": self.config.embedding_batch_size,
            "search_top_k": self.config.search_top_k,
            "max_steps": self.config.max_steps,
        }
        (self.artifact_dir / "retrieval_index.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.config.use_es:
            return search_chunks(
                namespace=self.namespace,
                query=query,
                mode=self.config.mode,
                top_k=self.config.search_top_k,
                embedding_provider_name=self.config.embedding_provider,
            ).get("items", [])
        if self.config.mode == "bm25" or not self.chunk_vectors:
            return _local_keyword_search(self.chunks, query, self.config.search_top_k)
        query_vector = _normalize_vectors(self.embedding_provider.embed([query]))[0]  # type: ignore[union-attr]
        dense_items = _local_dense_search(self.chunks, self.chunk_vectors, query_vector, self.config.search_top_k * 2)
        if self.config.mode == "dense":
            return dense_items[: self.config.search_top_k]
        keyword_items = _local_keyword_search(self.chunks, query, self.config.search_top_k * 2)
        return _rrf_fuse(keyword_items, dense_items, self.config.search_top_k)


class HiPRAGRunner:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        artifact_dir: Path,
        config: HiPRAGConfig,
        reuse_index: bool = False,
        reuse_index_dir: Path | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.config = config
        self.llm, self.chat_model_used = _hiprag_client()
        if reuse_index:
            self.index = HiPRAGSearchIndex.load_cache(reuse_index_dir or artifact_dir, config=config)
            self.index.artifact_dir = artifact_dir
            self.index.offline_wall_time_seconds = 0.0
            self.index.index_metrics = self.index.write_artifacts(status="ready_reused")
            self.index.index_metrics["offline_reused"] = True
            if reuse_index_dir:
                self.index.index_metrics["offline_reuse_source_dir"] = str(reuse_index_dir)
            self._stamp_model_metrics()
            (artifact_dir / "retrieval_index.json").write_text(json.dumps(self.index.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            self.index = HiPRAGSearchIndex(
                dataset=dataset,
                namespace=namespace,
                chunks=load_jsonl_list(chunks_path),
                artifact_dir=artifact_dir,
                config=config,
            )
            self.index.save_cache()
            self.index.index_metrics["offline_reused"] = False
            self._stamp_model_metrics()
            (artifact_dir / "retrieval_index.json").write_text(json.dumps(self.index.index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def _stamp_model_metrics(self) -> None:
        """Record the served chat model and whether the released trained policy is
        in use, for honest reporting in retrieval_index.json."""
        self.index.index_metrics["chat_model_used"] = self.chat_model_used
        self.index.index_metrics["hiprag_run_mode"] = (
            "released_trained_policy"
            if (os.environ.get("HIPRAG_API_BASE") or os.environ.get("HIPRAG_CHAT_MODEL"))
            else "untrained_procedure_shared_backbone"
        )

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        transcript = f"User Question: {question}\n<think>\n"
        all_retrieved: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        input_tokens = 0.0
        output_tokens = 0.0
        llm_calls = 0.0
        tool_calls = 0.0
        embedding_calls = 0.0
        retrieval_latency = 0.0
        final_raw = ""
        evidence_chunks: list[dict[str, Any]] = []
        rejected_answer_before_evidence = False

        for step_index in range(1, self.config.max_steps + 1):
            step_prompt = (
                f"{transcript}\n"
                "Continue with the next HiPRAG XML step. This is private-corpus QA: before giving a final "
                "answer, you should have retrieved corpus evidence. If no retrieved evidence is available yet, "
                "include exactly one <search>query</search>. If the accumulated evidence is sufficient, close "
                "</think> and provide the final <answer>."
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
                    "stage": "hiprag_agent_step",
                    "step": step_index,
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )
            transcript += _strip_duplicate_think_prefix(raw).strip() + "\n"
            final_raw = transcript
            search_query = _extract_last_search(raw)
            if _extract_answer(transcript) and all_retrieved and not search_query:
                break
            if _extract_answer(transcript) and not all_retrieved and not search_query:
                rejected_answer_before_evidence = True
                transcript = _remove_answer_tags(transcript)
                search_query = question
                transcript += "<conclusion>No corpus evidence has been retrieved yet, so an initial corpus search is required.</conclusion>\n"
                trace.append(
                    {
                        "event_type": "control",
                        "stage": "hiprag_forced_initial_search",
                        "step": step_index,
                        "reason": "model_answered_before_retrieving_evidence",
                    }
                )
            if not search_query:
                transcript += "<conclusion>No search query was issued in this step.</conclusion>\n"
                continue
            search_started = time.time()
            results = self.index.search(search_query)
            search_latency = time.time() - search_started
            retrieval_latency += search_latency
            tool_calls += 1.0
            if not self.config.use_es and self.config.mode in {"dense", "hybrid"}:
                embedding_calls += 1.0
            all_retrieved.extend(results)
            context, used = join_context(results, max_context_tokens=self.config.max_context_tokens)
            evidence_chunks.extend(_evidence_chunk_rows(used, round_index=step_index, query=search_query, source="retrieval_context"))
            transcript += f"<context>{context}</context>\n<conclusion>The search returned {len(used)} evidence chunks.</conclusion>\n"
            trace.append(
                {
                    "event_type": "tool_call",
                    "tool": "hiprag_private_chunk_search",
                    "step": step_index,
                    "query": search_query,
                    "latency_seconds": search_latency,
                    "output_summary": {"retrieved_chunks": len(used), "mode": self.config.mode, "use_es": self.config.use_es},
                }
            )

        answer = _extract_answer(final_raw)
        if answer and rejected_answer_before_evidence:
            answer = ""
        if not answer:
            final_prompt = (
                f"{transcript}\n"
                "Now close the HiPRAG reasoning trace and write the final <answer>. Use only the accumulated "
                "retrieved evidence. If it is insufficient, write exactly <answer>Insufficient evidence.</answer>."
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
                    "stage": "hiprag_final_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )

        if not evidence_chunks:
            answer = "Insufficient evidence."

        retrieved = _dedupe_chunks(all_retrieved, self.config.search_top_k * max(1, self.config.max_steps))
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
        rationale = _extract_think(final_raw)
        return BaselineResult(
            answer=answer.strip(),
            rationale=rationale,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            evidence_chunks=evidence_chunks,
            trace=trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            embedding_calls=embedding_calls,
            retrieval_latency_seconds=retrieval_latency,
        )


def run_hiprag(
    *,
    dataset: str,
    namespace: str | None = None,
    questions_path: str | None = None,
    chunks_path: str | None = None,
    output_path: str | None = None,
    query_log_path: str | None = None,
    artifact_dir: str | None = None,
    limit: int | None = None,
    use_es: bool = False,
    mode: str = "hybrid",
    search_top_k: int = 3,
    max_steps: int = 4,
    max_context_tokens: int = 2500,
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
    config = HiPRAGConfig(
        mode=mode,
        search_top_k=search_top_k,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size or int(os.environ.get("HIPRAG_EMBED_BATCH_SIZE", "32")),
        use_es=use_es,
    )
    runner = HiPRAGRunner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
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
            "retrieval": "hiprag_agentic_private_chunk_search",
            "use_es": use_es,
            "mode": mode,
            "search_top_k": search_top_k,
            "max_steps": max_steps,
            "embedding_provider": embedding_provider,
            "embedding_batch_size": config.embedding_batch_size,
            "max_context_tokens": max_context_tokens,
            "index_metrics": runner.index.index_metrics,
            "prompt_style": "hiprag_xml_signpost_evidence_grounded",
            "offline_reused": reuse_index,
        },
    )


def _extract_last_search(text: str) -> str:
    patterns = [
        r"<\s*search\s*>(.*?)<\s*/\s*search\s*>",
        r"<\s*search\s+query\s*=\s*['\"]([^'\"]+)['\"]\s*/?\s*>",
        r"(?:^|\n)\s*Search\s*:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return _clean_xml_text(matches[-1])
    return ""


def _extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return _clean_xml_text(matches[-1]) if matches else ""


def _extract_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return _clean_xml_text(match.group(1)) if match else ""


def _clean_xml_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(term) > 1]


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized = []
    for vector in vectors:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
        normalized.append([float(value) / norm for value in vector])
    return normalized


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


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
            scored.append({**item, "score": float(score), "score_source": "hiprag_local_keyword"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def _local_dense_search(chunks: list[dict[str, Any]], vectors: list[list[float]], query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for item, vector in zip(chunks, vectors, strict=True):
        scored.append({**item, "score": _dot(query_vector, vector), "score_source": "hiprag_local_dense"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def _rrf_fuse(left: list[dict[str, Any]], right: list[dict[str, Any]], top_k: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(left, start=1):
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "hiprag_local_hybrid"}
    for rank, item in enumerate(right, start=1):
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "hiprag_local_hybrid"}
    return [{**docs[chunk_id], "score": score} for chunk_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HiPRAG baseline adapter.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reuse-index-dir")
    args = parser.parse_args()
    count = run_hiprag(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        output_path=args.output,
        query_log_path=args.query_log,
        artifact_dir=args.artifact_dir,
        limit=args.limit,
        use_es=args.use_es,
        mode=args.mode,
        search_top_k=args.search_top_k,
        max_steps=args.max_steps,
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
