from __future__ import annotations

"""Clue-RAG external baseline adapter.

The official Clue-RAG repository is kept under ``baselines/ClueRAG``.  This
module owns only Signpost-side data conversion, optional invocation, and output
normalization so the external code stays isolated from the main package.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Any

from signpost.baselines.common import BaselineResult, append_jsonl, baseline_cost, load_jsonl_list, question_id_from_row, question_text
from signpost.chunking.tokenizer import count_tokens
from signpost.config.context import resolve_project_path
from signpost.evaluation.schema import build_prediction_text
from signpost.indexing.embedding import create_embedding_provider
from signpost.llm.client import load_llm_config
from signpost.parsing.io import read_jsonl, write_jsonl
from signpost.storage.elasticsearch import ElasticsearchClient


METHOD = "cluerag"
PROMPT_STYLE_DEFAULT = "adapter"
PROMPT_STYLE_SIGNPOST_FEWSHOT = "signpost_fewshot"


class SharedClueRAGIndex:
    def __init__(
        self,
        *,
        chunks: dict[str, dict[str, Any]],
        knowledge_units: dict[str, dict[str, Any]],
        entities: dict[str, dict[str, Any]],
        chunk_entities: dict[str, list[str]],
        ku2chunkids: dict[str, list[str]],
        ku2entities: dict[str, list[str]],
        entity2kus: dict[str, list[str]],
        es_index: str = "",
        offline_metrics: dict[str, Any] | None = None,
    ):
        self.chunks = chunks
        self.knowledge_units = knowledge_units
        self.chunk_entities = chunk_entities
        self.entities = entities
        self.ku2chunkids = ku2chunkids
        self.ku2entities = ku2entities
        self.entity2kus = entity2kus
        self.es_index = es_index
        self.offline_metrics = offline_metrics or {}


def prepare_cluerag_inputs(
    *,
    dataset: str,
    repo_path: str | Path = "baselines/ClueRAG",
    documents_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    limit: int | None = None,
    cluerag_dataset: str | None = None,
) -> dict[str, Any]:
    repo = resolve_project_path(repo_path)
    data_dir = repo / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target_name = cluerag_dataset or _cluerag_dataset_name(dataset)
    docs = _load_documents(resolve_project_path(documents_path or f"datasets/processed/{dataset}/documents.jsonl"))
    questions = _load_questions_for_cluerag(resolve_project_path(questions_path or f"datasets/processed/{dataset}/questions.jsonl"), limit=limit)
    corpus_path = data_dir / f"{target_name}_corpus.json"
    questions_out = data_dir / f"{target_name}.json"
    corpus_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    questions_out.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "method": METHOD,
        "dataset": dataset,
        "cluerag_dataset": target_name,
        "repo_path": str(repo),
        "corpus_path": str(corpus_path),
        "questions_path": str(questions_out),
        "documents": len(docs),
        "questions": len(questions),
    }
    manifest_path = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_cluerag_official(
    *,
    dataset: str,
    repo_path: str | Path = "baselines/ClueRAG",
    cluerag_dataset: str | None = None,
    alpha: float = 1.0,
    select_metric: str = "COSINE",
    save_dir: str | Path | None = None,
    llm_processes: int = 1,
    num_processes: int = 1,
    embedding_batch_size: int = 64,
    rerank_url: str | None = None,
) -> dict[str, Any]:
    repo = resolve_project_path(repo_path)
    target_name = cluerag_dataset or _cluerag_dataset_name(dataset)
    started = time.time()
    cwd = Path.cwd()
    sys_path_added = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        sys_path_added = True
    os.chdir(repo)
    try:
        from dataset.dataclass import Dataset
        from generation.generation import Generation
        from index.construction import MultiLayerGraph
        from index.hybrid_extraction import HybridExtraction
        from retrieval.retrieval import IterativeRetrieval
        from utils.config import BaseConfig

        llm = load_llm_config()
        config = BaseConfig(dataset_name=target_name)
        config.alpha = alpha
        config.select_metric = select_metric
        config.save_dir = str(resolve_project_path(save_dir or f"outputs/{dataset}/baselines/cluerag/official_outputs"))
        config.llm_base_url = llm.api_base
        config.llm_name = llm.chat_model
        config.api_key = llm.api_key or "EMPTY"
        config.embedding_model_url = _openai_embedding_base(llm.embedding_api_base or llm.api_base)
        config.embedding_model_name = llm.embedding_model
        config.llm_num_processes = llm_processes
        config.num_processes = num_processes
        config.embedding_batch_size = embedding_batch_size
        config.multiprocess = llm_processes > 1
        config.db_host = os.getenv("CLUERAG_DB_HOST", config.db_host)
        config.db_port = int(os.getenv("CLUERAG_DB_PORT", str(config.db_port)))
        config.db_user = os.getenv("CLUERAG_DB_USER", config.db_user)
        config.db_password = os.getenv("CLUERAG_DB_PASSWORD", config.db_password)
        config.db_default_database = os.getenv("CLUERAG_DB_NAME", config.db_default_database)
        config.rerank_model_name = os.getenv("CLUERAG_RERANK_MODEL", config.rerank_model_name)
        if rerank_url or os.getenv("CLUERAG_RERANK_URL"):
            config.rerank_url = rerank_url or os.environ["CLUERAG_RERANK_URL"]

        async def _run() -> dict[str, Any]:
            stage_timings: dict[str, float] = {}
            stage_started = time.perf_counter()
            loaded = Dataset(global_config=config)
            stage_timings["dataset_load_seconds"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            hybrid = HybridExtraction(global_config=config, dataset=loaded)
            await hybrid.process()
            stage_timings["hybrid_extraction_seconds"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            graph = MultiLayerGraph(config, hybrid, loaded)
            stage_timings["graph_construction_seconds"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            retriever = IterativeRetrieval(global_config=config, graph=graph, dataset=loaded)
            retrieval_results = await retriever.query_async()
            stage_timings["retrieval_seconds"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            generator = Generation(global_config=config, dataset=loaded)
            generation_results = await generator.augmentated_generation(retrieval_results=retrieval_results)
            stage_timings["generation_seconds"] = time.perf_counter() - stage_started
            return {
                "queries": len(generation_results or []),
                "official_output_dir": str(Path(config.save_dir) / f"{config.select_metric}_{config.alpha:.2f}"),
                "stage_timings": stage_timings,
                "hybrid_metadata": getattr(hybrid, "res_metadata", {}),
                "retrieval_metadata": getattr(retriever, "res_metadata", {}),
                "generation_metadata": getattr(generator, "res_metadata", {}),
            }

        result = asyncio.run(_run())
        result.update({"wall_time_seconds": time.time() - started, "cluerag_dataset": target_name})
        metrics = _cluerag_run_metrics(result)
        metrics_path = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/run_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["metrics_path"] = str(metrics_path)
        result["stage_metrics"] = metrics
        status_path = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/run_status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        os.chdir(cwd)
        if sys_path_added:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass


def run_cluerag_shared(
    *,
    dataset: str,
    namespace: str | None = None,
    chunks_path: str | Path | None = None,
    semantic_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    query_log_path: str | Path | None = None,
    limit: int | None = None,
    use_es: bool = True,
    mode: str = "hybrid",
    embedding_provider: str = "ecnu",
    direct_top_k: int = 10,
    ku_top_k: int = 3,
    graph_top_k: int = 5,
    top_n: int = 5,
    depth: int = 3,
    rerank_url: str | None = None,
    rerank_model: str | None = None,
    official_output_dir: str | Path | None = None,
    recreate_graph: bool = True,
    prompt_style: str = PROMPT_STYLE_DEFAULT,
    method_name: str = METHOD,
) -> dict[str, Any]:
    """Run a ClueRAG-style adapter over shared Signpost chunks and F6 semantics.

    This is the default fair-comparison path for the paper experiments.  It
    keeps ClueRAG's chunk/KU/entity graph organization and final reranking, but
    uses Signpost's shared F4 chunks and F6 semantic extraction instead of
    re-chunking or re-extracting entities.
    """

    resolved_namespace = namespace or dataset
    started = time.time()
    stage_timings: dict[str, float] = {}

    graph_dir = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/shared_graph")

    stage_started = time.perf_counter()
    chunks = _load_shared_chunks(resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"))
    if recreate_graph:
        shared_index = _build_shared_cluerag_index(
            dataset=dataset,
            namespace=resolved_namespace,
            chunks=chunks,
            semantic_path=resolve_project_path(semantic_path or f"datasets/processed/{dataset}/semantic_llm.extractions.jsonl"),
            graph_dir=graph_dir,
            embedding_provider_name=embedding_provider,
            use_es=use_es,
            recreate_es=True,
        )
    else:
        shared_index = _load_shared_cluerag_index(
            dataset=dataset,
            namespace=resolved_namespace,
            graph_dir=graph_dir,
            use_es=use_es,
        )
    questions = _load_question_rows(resolve_project_path(questions_path or f"datasets/processed/{dataset}/questions.jsonl"), limit=limit)
    stage_timings["graph_construction_seconds"] = time.perf_counter() - stage_started

    result_dir = resolve_project_path(official_output_dir or f"outputs/{dataset}/baselines/cluerag/shared_outputs/COSINE_1.00")
    result_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm_config()
    retrieval_rows = []
    generation_rows = []
    generation_prompt_tokens = 0.0
    generation_completion_tokens = 0.0
    rerank_calls = 0.0
    query_ner_calls = 0.0
    query_ner_prompt_tokens = 0.0
    query_ner_completion_tokens = 0.0
    embedding_calls = 0.0
    retrieval_started = time.perf_counter()
    generation_seconds = 0.0

    from signpost.llm.client import OpenAICompatibleClient

    client = OpenAICompatibleClient(timeout=int(os.getenv("LLM_TIMEOUT", "600") or 600))
    for index, question_row in enumerate(questions, start=1):
        qid = question_id_from_row(question_row, index)
        question = question_text(question_row)
        answer = str(question_row.get("answer", ""))
        retrieved = _retrieve_shared_cluerag(
            namespace=resolved_namespace,
            question=question,
            shared_index=shared_index,
            use_es=use_es,
            mode=mode,
            embedding_provider=embedding_provider,
            direct_top_k=direct_top_k,
            ku_top_k=ku_top_k,
            graph_top_k=graph_top_k,
            top_n=top_n,
            depth=depth,
            rerank_url=rerank_url or os.getenv("CLUERAG_RERANK_URL", ""),
            rerank_model=rerank_model or os.getenv("CLUERAG_RERANK_MODEL", llm.rerank_model),
            llm_client=client,
        )
        rerank_calls += float(retrieved.get("rerank_calls", 0.0) or 0.0)
        query_ner_calls += float(retrieved.get("query_ner_calls", 0.0) or 0.0)
        query_ner_prompt_tokens += float(retrieved.get("query_ner_prompt_tokens", 0.0) or 0.0)
        query_ner_completion_tokens += float(retrieved.get("query_ner_completion_tokens", 0.0) or 0.0)
        embedding_calls += float(retrieved.get("embedding_calls", 0.0) or 0.0)
        retrieval_row = {
            "qid": qid,
            "question": question,
            "chunks": [item["chunk_id"] for item in retrieved["chunks"]],
            "paths": retrieved.get("paths", []),
            "ground_truth_chunks": question_row.get("metadata", {}).get("gold_chunk_ids", []),
            "answer": answer,
            "recall@2": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
        }
        retrieval_rows.append(retrieval_row)

        generation_started = time.perf_counter()
        prompt = _cluerag_generation_prompt(
            question=question,
            chunks=retrieved["chunks"],
            paths=retrieved.get("paths", []),
            prompt_style=prompt_style,
        )
        generated = client.chat(
            [
                {"role": "system", "content": _cluerag_system_prompt(prompt_style)},
                {"role": "user", "content": prompt},
            ]
        )
        generation_seconds += time.perf_counter() - generation_started
        current_generation_prompt_tokens = float(count_tokens(prompt))
        current_generation_completion_tokens = float(count_tokens(generated))
        generation_prompt_tokens += current_generation_prompt_tokens
        generation_completion_tokens += current_generation_completion_tokens
        generation_rows.append(
            {
                **retrieval_row,
                "generation": generated,
                "metadata": {
                    "query_ner_calls": float(retrieved.get("query_ner_calls", 0.0) or 0.0),
                    "rerank_calls": float(retrieved.get("rerank_calls", 0.0) or 0.0),
                    "embedding_calls": float(retrieved.get("embedding_calls", 0.0) or 0.0),
                    "input_tokens": float(retrieved.get("query_ner_prompt_tokens", 0.0) or 0.0)
                    + current_generation_prompt_tokens,
                    "output_tokens": float(retrieved.get("query_ner_completion_tokens", 0.0) or 0.0)
                    + current_generation_completion_tokens,
                    "llm_calls": float(retrieved.get("query_ner_calls", 0.0) or 0.0) + 1.0,
                    "tool_calls": 1.0,
                    "knowledge_search_calls": 1.0,
                },
            }
        )

    retrieval_seconds = max(0.0, time.perf_counter() - retrieval_started - generation_seconds)
    stage_timings["retrieval_seconds"] = retrieval_seconds
    stage_timings["generation_seconds"] = generation_seconds

    retrieval_meta = {
        "prompt_tokens": query_ner_prompt_tokens,
        "completion_tokens": query_ner_completion_tokens,
        "total_tokens": query_ner_prompt_tokens + query_ner_completion_tokens,
        "num_requests": query_ner_calls,
        "query_ner_calls": query_ner_calls,
        "rerank_calls": rerank_calls,
        "embedding_calls": embedding_calls,
        "model_calls": query_ner_calls + rerank_calls + embedding_calls,
        "backend": "shared_es" if use_es else "shared_local",
        "direct_top_k": direct_top_k,
        "ku_top_k": ku_top_k,
        "graph_top_k": graph_top_k,
        "top_n": top_n,
        "depth": depth,
        "shared_chunks": len(shared_index.chunks),
        "shared_knowledge_units": len(shared_index.knowledge_units),
        "shared_entities": len(shared_index.entities),
        "cluerag_graph_index": shared_index.es_index,
    }
    generation_meta = {
        "prompt_tokens": generation_prompt_tokens,
        "completion_tokens": generation_completion_tokens,
        "total_tokens": generation_prompt_tokens + generation_completion_tokens,
        "num_requests": float(len(generation_rows)),
        "model": llm.chat_model,
        "prompt_style": prompt_style,
    }
    retrieval_payload = {"retrieval_results": retrieval_rows, "metadata": retrieval_meta}
    generation_payload = {"generation_results": generation_rows, "metadata": generation_meta}
    (result_dir / "retrieval_results.json").write_text(json.dumps(retrieval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "generation_results.json").write_text(json.dumps(generation_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "queries": len(generation_rows),
        "official_output_dir": str(result_dir),
        "wall_time_seconds": time.time() - started,
        "cluerag_dataset": _cluerag_dataset_name(dataset),
        "backend": "shared_es" if use_es else "shared_local",
        "stage_timings": stage_timings,
        "hybrid_metadata": {},
        "offline_metadata": shared_index.offline_metrics,
        "graph_artifact_dir": str(graph_dir),
        "retrieval_metadata": retrieval_meta,
        "generation_metadata": generation_meta,
    }
    metrics = _cluerag_run_metrics(result)
    metrics_path = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/run_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["metrics_path"] = str(metrics_path)
    result["stage_metrics"] = metrics
    converted = convert_cluerag_outputs(
        dataset=dataset,
        namespace=resolved_namespace,
        official_output_dir=result_dir,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method_name=method_name,
    )
    result["converted_predictions"] = converted
    status_path = resolve_project_path(f"outputs/{dataset}/baselines/cluerag/run_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def regenerate_cluerag_generation_from_retrieval(
    *,
    dataset: str,
    namespace: str | None = None,
    chunks_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    source_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    query_log_path: str | Path | None = None,
    prompt_style: str = PROMPT_STYLE_SIGNPOST_FEWSHOT,
    method_name: str = "cluerag_prompt_normalized",
) -> dict[str, Any]:
    """Reuse ClueRAG retrieval rows and rerun only final answer generation."""

    resolved_namespace = namespace or dataset
    started = time.time()
    source_dir = resolve_project_path(source_output_dir or f"outputs/{dataset}/baselines/cluerag/shared_outputs/COSINE_1.00")
    retrieval_path = source_dir / "retrieval_results.json"
    if not retrieval_path.exists():
        raise FileNotFoundError(retrieval_path)
    source_generation_path = source_dir / "generation_results.json"
    result_dir = resolve_project_path(output_dir or f"outputs/{dataset}/baselines/{method_name}/shared_outputs/COSINE_1.00")
    result_dir.mkdir(parents=True, exist_ok=True)

    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval_rows = retrieval_payload.get("retrieval_results", [])
    if not isinstance(retrieval_rows, list):
        raise ValueError(f"{retrieval_path} does not contain a retrieval_results list")
    retrieval_meta = retrieval_payload.get("metadata", {}) if isinstance(retrieval_payload, dict) else {}
    prior_generation_rows: dict[str, dict[str, Any]] = {}
    if source_generation_path.exists():
        source_generation_payload = json.loads(source_generation_path.read_text(encoding="utf-8"))
        for row in source_generation_payload.get("generation_results", []):
            if isinstance(row, dict):
                prior_generation_rows[str(row.get("qid") or "")] = row

    chunks = _load_shared_chunks(resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"))
    questions = _question_lookup(resolve_project_path(questions_path or f"datasets/processed/{dataset}/questions.jsonl"))

    from signpost.llm.client import OpenAICompatibleClient

    client = OpenAICompatibleClient(timeout=int(os.getenv("LLM_TIMEOUT", "600") or 600))
    llm = load_llm_config()
    generation_rows = []
    generation_prompt_tokens = 0.0
    generation_completion_tokens = 0.0
    generation_seconds = 0.0
    query_count = max(1, len(retrieval_rows))
    retrieval_prompt_per_query = float(retrieval_meta.get("prompt_tokens", 0.0) or 0.0) / query_count
    retrieval_completion_per_query = float(retrieval_meta.get("completion_tokens", 0.0) or 0.0) / query_count
    retrieval_llm_per_query = float(retrieval_meta.get("num_requests", 0.0) or 0.0) / query_count

    for index, retrieval_row in enumerate(retrieval_rows, start=1):
        qid = str(retrieval_row.get("qid") or index)
        question = str(retrieval_row.get("question") or questions.get(qid, {}).get("question") or "")
        answer = str(retrieval_row.get("answer") or questions.get(qid, {}).get("answer") or "")
        chunk_ids = [str(chunk_id) for chunk_id in retrieval_row.get("chunks", [])]
        retrieved_chunks = [chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks]
        paths = retrieval_row.get("paths", []) if isinstance(retrieval_row.get("paths"), list) else []
        prompt = _cluerag_generation_prompt(question=question, chunks=retrieved_chunks, paths=paths, prompt_style=prompt_style)
        generation_started = time.perf_counter()
        generated = client.chat(
            [
                {"role": "system", "content": _cluerag_system_prompt(prompt_style)},
                {"role": "user", "content": prompt},
            ]
        )
        generation_seconds += time.perf_counter() - generation_started
        current_generation_prompt_tokens = float(count_tokens(prompt))
        current_generation_completion_tokens = float(count_tokens(generated))
        generation_prompt_tokens += current_generation_prompt_tokens
        generation_completion_tokens += current_generation_completion_tokens
        prior_meta = _row_metadata(prior_generation_rows.get(qid, {}))
        query_ner_calls = float(prior_meta.get("query_ner_calls", retrieval_llm_per_query) or 0.0)
        generation_rows.append(
            {
                **retrieval_row,
                "qid": qid,
                "question": question,
                "answer": answer,
                "generation": generated,
                "metadata": {
                    "query_ner_calls": query_ner_calls,
                    "rerank_calls": float(prior_meta.get("rerank_calls", 0.0) or 0.0),
                    "embedding_calls": float(prior_meta.get("embedding_calls", 0.0) or 0.0),
                    "input_tokens": retrieval_prompt_per_query + current_generation_prompt_tokens,
                    "output_tokens": retrieval_completion_per_query + current_generation_completion_tokens,
                    "llm_calls": query_ner_calls + 1.0,
                    "tool_calls": 1.0,
                    "knowledge_search_calls": 1.0,
                    "prompt_style": prompt_style,
                    "source_output_dir": str(source_dir),
                },
            }
        )

    generation_meta = {
        "prompt_tokens": generation_prompt_tokens,
        "completion_tokens": generation_completion_tokens,
        "total_tokens": generation_prompt_tokens + generation_completion_tokens,
        "num_requests": float(len(generation_rows)),
        "model": llm.chat_model,
        "prompt_style": prompt_style,
        "source_output_dir": str(source_dir),
    }
    (result_dir / "retrieval_results.json").write_text(json.dumps(retrieval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "generation_results.json").write_text(
        json.dumps({"generation_results": generation_rows, "metadata": generation_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = {
        "queries": len(generation_rows),
        "official_output_dir": str(result_dir),
        "wall_time_seconds": time.time() - started,
        "cluerag_dataset": _cluerag_dataset_name(dataset),
        "backend": "shared_generation_only",
        "stage_timings": {"generation_seconds": generation_seconds},
        "hybrid_metadata": {},
        "offline_metadata": {},
        "graph_artifact_dir": "",
        "retrieval_metadata": retrieval_meta,
        "generation_metadata": generation_meta,
        "source_output_dir": str(source_dir),
    }
    metrics = _cluerag_run_metrics(result)
    metrics_path = resolve_project_path(f"outputs/{dataset}/baselines/{method_name}/run_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["metrics_path"] = str(metrics_path)
    result["stage_metrics"] = metrics
    converted = convert_cluerag_outputs(
        dataset=dataset,
        namespace=resolved_namespace,
        official_output_dir=result_dir,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method_name=method_name,
    )
    result["converted_predictions"] = converted
    status_path = resolve_project_path(f"outputs/{dataset}/baselines/{method_name}/run_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def convert_cluerag_outputs(
    *,
    dataset: str,
    namespace: str | None = None,
    repo_path: str | Path = "baselines/ClueRAG",
    cluerag_dataset: str | None = None,
    official_output_dir: str | Path | None = None,
    questions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    query_log_path: str | Path | None = None,
    method_name: str = METHOD,
) -> int:
    target_name = cluerag_dataset or _cluerag_dataset_name(dataset)
    result_dir = resolve_project_path(
        official_output_dir or f"outputs/{dataset}/baselines/cluerag/official_outputs/COSINE_1.00"
    )
    generation_path = result_dir / "generation_results.json"
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)

    output = resolve_project_path(output_path or f"outputs/{dataset}/predictions/{METHOD}.jsonl")
    query_log = resolve_project_path(query_log_path or f"outputs/{dataset}/logs/{METHOD}.query.jsonl")
    question_rows = _question_lookup(resolve_project_path(questions_path or f"datasets/processed/{dataset}/questions.jsonl"))
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    retrieval_path = result_dir / "retrieval_results.json"
    retrieval_meta = {}
    if retrieval_path.exists():
        retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        retrieval_meta = retrieval_payload.get("metadata", {}) if isinstance(retrieval_payload, dict) else {}
    rows = payload.get("generation_results", [])
    generation_meta = payload.get("metadata", {})
    online_meta = _combine_online_metadata(retrieval_meta, generation_meta, query_count=max(1, len(rows)))
    predictions = []
    query_log.parent.mkdir(parents=True, exist_ok=True)
    query_log.write_text("", encoding="utf-8")
    for index, row in enumerate(rows, start=1):
        qid = str(row.get("qid") or index)
        original = question_rows.get(qid, {})
        question = str(row.get("question") or original.get("question") or "")
        gold_answer = str(row.get("answer") or original.get("answer") or "")
        parsed = _parse_generation(row.get("generation", ""))
        result = {**_result_cost(row, parsed, online_meta), **_row_extra_model_cost(row)}
        prediction = {
            "question_id": qid,
            "question": question,
            "answer": gold_answer,
            "rationale": original.get("rationale", ""),
            "prediction": build_prediction_text(answer=parsed["answer"], rationale=parsed["rationale"]),
            "citations": [],
            "trace": [
                {
                    "event_type": "external_baseline",
                    "tool": METHOD,
                    "output_summary": {
                        "cluerag_dataset": target_name,
                        "chunks": len(row.get("chunks", [])),
                        "recall@2": row.get("recall@2"),
                        "recall@5": row.get("recall@5"),
                        "recall@10": row.get("recall@10"),
                    },
                }
            ],
            "retrieved_chunks": [
                {"chunk_id": str(chunk_id), "score_source": "cluerag_q_iter"}
                for chunk_id in row.get("chunks", [])
            ],
            **result,
            "metadata": {
                **original.get("metadata", {}),
                "method": method_name,
                "dataset": dataset,
                "namespace": namespace or dataset,
                "cluerag_dataset": target_name,
                "official_output_dir": str(result_dir),
                **_row_metadata(row),
            },
        }
        predictions.append(prediction)
        append_jsonl(
            query_log,
            {
                "dataset": dataset,
                "namespace": namespace or dataset,
                "method": method_name,
                "question_id": qid,
                "question": question,
                "retrieved_chunks": prediction["retrieved_chunks"],
                **result,
            },
        )
    return write_jsonl(output, predictions)


def _load_documents(path: Path) -> list[dict[str, Any]]:
    docs = []
    for index, row in enumerate(read_jsonl(path)):
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        title = str(row.get("title") or row.get("file_name") or row.get("doc_id") or f"doc_{index}")
        docs.append({"idx": index, "title": title, "text": text})
    if not docs:
        raise ValueError(f"{path} contains no non-empty documents")
    return docs


def _load_questions_for_cluerag(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_jsonl(path), start=1):
        if limit is not None and index > limit:
            break
        qid = question_id_from_row(row, index)
        rows.append({"_id": qid, "id": qid, "question": question_text(row), "answer": str(row.get("answer", ""))})
    if not rows:
        raise ValueError(f"{path} contains no questions")
    return rows


def _load_shared_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        chunk_id = str(row.get("chunk_id") or row.get("id") or index)
        content = str(row.get("content") or row.get("text") or "").strip()
        if not content:
            continue
        chunks[chunk_id] = {
            **row,
            "chunk_id": chunk_id,
            "content": content,
        }
    if not chunks:
        raise ValueError(f"{path} contains no non-empty chunks")
    return chunks


def _load_question_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_jsonl(path), start=1):
        if limit is not None and index > limit:
            break
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no questions")
    return rows


def _build_shared_cluerag_index(
    *,
    dataset: str,
    namespace: str,
    chunks: dict[str, dict[str, Any]],
    semantic_path: Path,
    graph_dir: Path,
    embedding_provider_name: str,
    use_es: bool,
    recreate_es: bool,
) -> SharedClueRAGIndex:
    """Build ClueRAG's own chunk/KU/entity multilayer graph from shared inputs.

    Shared Signpost artifacts stop at chunks and semantic extractions.  This
    function deliberately does not read Signpost unified/navigation graphs; it
    organizes the shared facts into the ClueRAG graph shape:
    chunks <-> knowledge units <-> entities.
    """

    knowledge_units: dict[str, dict[str, Any]] = {}
    entities: dict[str, dict[str, Any]] = {}
    chunk_entities: dict[str, set[str]] = defaultdict(set)
    ku2chunkids: dict[str, set[str]] = defaultdict(set)
    ku2entities: dict[str, set[str]] = defaultdict(set)
    entity2kus: dict[str, set[str]] = defaultdict(set)

    def ensure_entity(raw_name: Any, *, entity_type: str = "", description: str = "", chunk_id: str | None = None) -> str:
        name = _cluerag_normalize_name(str(raw_name or ""))
        if not name:
            return ""
        entity = entities.setdefault(
            name,
            {
                "id": _mdhash_id(name),
                "entity_name": name,
                "entity_type": entity_type,
                "description": description,
                "chunks": set(),
                "ku_ids": set(),
            },
        )
        if entity_type and not entity.get("entity_type"):
            entity["entity_type"] = entity_type
        if description and not entity.get("description"):
            entity["description"] = description
        if chunk_id:
            entity["chunks"].add(chunk_id)
            chunk_entities[chunk_id].add(name)
        return name

    def add_ku(ku_text: Any, *, chunk_id: str, entity_names: list[str], source: str, source_id: str = "") -> str:
        text = _cluerag_clean_str(str(ku_text or ""))
        if not text:
            return ""
        clean_entities = sorted({name for name in entity_names if name})
        if not clean_entities:
            return ""
        ku_id = _mdhash_id(text)
        item = knowledge_units.setdefault(
            ku_id,
            {
                "id": ku_id,
                "text": text,
                "entity_list": set(),
                "chunk_ids": set(),
                "source": source,
                "source_ids": set(),
            },
        )
        item["entity_list"].update(clean_entities)
        item["chunk_ids"].add(chunk_id)
        if source_id:
            item["source_ids"].add(source_id)
        ku2chunkids[text].add(chunk_id)
        ku2entities[text].update(clean_entities)
        for entity_name in clean_entities:
            entity = entities.setdefault(
                entity_name,
                {
                    "id": _mdhash_id(entity_name),
                    "entity_name": entity_name,
                    "entity_type": "",
                    "description": "",
                    "chunks": set(),
                    "ku_ids": set(),
                },
            )
            entity["chunks"].add(chunk_id)
            entity["ku_ids"].add(ku_id)
            entity2kus[entity_name].add(text)
            chunk_entities[chunk_id].add(entity_name)
        return ku_id

    for row in read_jsonl(semantic_path):
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id not in chunks:
            continue
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
        entity_names_by_raw: dict[str, str] = {}
        for item in extraction.get("entities", []) or []:
            if not isinstance(item, dict):
                continue
            entity_type = str(item.get("entity_type") or "")
            if entity_type.upper() == "DATE":
                continue
            name = ensure_entity(
                item.get("name"),
                entity_type=entity_type,
                description=str(item.get("description") or ""),
                chunk_id=chunk_id,
            )
            if not name:
                continue
            entity_names_by_raw[_entity_key(item.get("name"))] = name
            description = str(item.get("description") or "").strip()
            if description:
                add_ku(f"{name}: {description}", chunk_id=chunk_id, entity_names=[name], source="entity_description", source_id=name)

        for item in extraction.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            source_name = entity_names_by_raw.get(_entity_key(item.get("source"))) or ensure_entity(item.get("source"), chunk_id=chunk_id)
            target_name = entity_names_by_raw.get(_entity_key(item.get("target"))) or ensure_entity(item.get("target"), chunk_id=chunk_id)
            if not source_name or not target_name:
                continue
            description = str(item.get("description") or "").strip()
            if description:
                add_ku(
                    description,
                    chunk_id=chunk_id,
                    entity_names=[source_name, target_name],
                    source="relation_description",
                    source_id=f"{source_name}->{target_name}",
                )

    materialized_kus = {
        ku_id: {
            **value,
            "entity_list": sorted(value.get("entity_list", set())),
            "chunk_ids": sorted(value.get("chunk_ids", set())),
            "source_ids": sorted(value.get("source_ids", set())),
        }
        for ku_id, value in knowledge_units.items()
    }
    materialized_entities = {
        name: {
            **value,
            "chunks": sorted(value.get("chunks", set())),
            "ku_ids": sorted(value.get("ku_ids", set())),
        }
        for name, value in entities.items()
    }
    materialized_chunk_entities = {chunk_id: sorted(values) for chunk_id, values in chunk_entities.items()}
    materialized_ku2chunkids = {text: sorted(values) for text, values in ku2chunkids.items()}
    materialized_ku2entities = {text: sorted(values) for text, values in ku2entities.items()}
    materialized_entity2kus = {name: sorted(values) for name, values in entity2kus.items()}
    offline_metrics: dict[str, Any] = {
        "shared_chunks": float(len(chunks)),
        "shared_knowledge_units": float(len(materialized_kus)),
        "shared_entities": float(len(materialized_entities)),
        "cluerag_graph_nodes": float(len(chunks) + len(materialized_kus) + len(materialized_entities)),
        "cluerag_graph_chunk_nodes": float(len(chunks)),
        "cluerag_graph_knowledge_unit_nodes": float(len(materialized_kus)),
        "cluerag_graph_entity_nodes": float(len(materialized_entities)),
        "cluerag_graph_edges": float(
            sum(len(values) for values in materialized_ku2chunkids.values())
            + sum(len(values) for values in materialized_ku2entities.values())
        ),
    }

    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_cluerag_graph_artifacts(
        graph_dir=graph_dir,
        dataset=dataset,
        namespace=namespace,
        chunks=chunks,
        knowledge_units=materialized_kus,
        entities=materialized_entities,
        chunk_entities=materialized_chunk_entities,
        ku2chunkids=materialized_ku2chunkids,
        ku2entities=materialized_ku2entities,
    )

    es_index = ""
    if use_es:
        es_index, es_metrics = _index_cluerag_graph_es(
            namespace=namespace,
            dataset=dataset,
            chunks=chunks,
            knowledge_units=materialized_kus,
            entities=materialized_entities,
            embedding_provider_name=embedding_provider_name,
            recreate=recreate_es,
        )
        offline_metrics.update(es_metrics)

    return SharedClueRAGIndex(
        chunks=chunks,
        knowledge_units=materialized_kus,
        entities=materialized_entities,
        chunk_entities=materialized_chunk_entities,
        ku2chunkids=materialized_ku2chunkids,
        ku2entities=materialized_ku2entities,
        entity2kus=materialized_entity2kus,
        es_index=es_index,
        offline_metrics=offline_metrics,
    )


def _load_shared_cluerag_index(
    *,
    dataset: str,
    namespace: str,
    graph_dir: Path,
    use_es: bool,
) -> SharedClueRAGIndex:
    required = [
        graph_dir / "manifest.json",
        graph_dir / "graph_cache.json",
        graph_dir / "chunks.jsonl",
        graph_dir / "knowledge_units.jsonl",
        graph_dir / "entities.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ClueRAG shared graph artifacts are missing; cannot reuse graph: {missing}")

    manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((graph_dir / "graph_cache.json").read_text(encoding="utf-8"))
    chunks = {str(row.get("chunk_id")): row for row in read_jsonl(graph_dir / "chunks.jsonl") if row.get("chunk_id")}
    knowledge_units = {str(row.get("id")): row for row in read_jsonl(graph_dir / "knowledge_units.jsonl") if row.get("id")}
    entities = {str(row.get("entity_name")): row for row in read_jsonl(graph_dir / "entities.jsonl") if row.get("entity_name")}
    chunk_entities = {chunk_id: list(row.get("entity_list") or []) for chunk_id, row in chunks.items()}
    ku2chunkids = {str(key): list(value or []) for key, value in (cache.get("ku2chunkids") or {}).items()}
    ku2entities = {str(key): list(value or []) for key, value in (cache.get("ku2entities") or {}).items()}
    entity2kus: dict[str, list[str]] = defaultdict(list)
    for ku_text, entity_names in ku2entities.items():
        for entity_name in entity_names:
            entity2kus[str(entity_name)].append(ku_text)

    es_index = _cluerag_es_index_name(namespace) if use_es else ""
    if use_es:
        client = ElasticsearchClient(timeout=int(os.getenv("CLUERAG_ES_TIMEOUT", "120") or 120))
        if not client.exists_index(es_index):
            raise FileNotFoundError(f"ClueRAG ES index is missing; cannot reuse graph: {es_index}")

    offline_metrics = {
        "shared_chunks": float(manifest.get("source_chunks", len(chunks)) or 0.0),
        "shared_knowledge_units": float(manifest.get("knowledge_units", len(knowledge_units)) or 0.0),
        "shared_entities": float(manifest.get("entities", len(entities)) or 0.0),
        "cluerag_graph_nodes": float(len(chunks) + len(knowledge_units) + len(entities)),
        "cluerag_graph_chunk_nodes": float(len(chunks)),
        "cluerag_graph_knowledge_unit_nodes": float(len(knowledge_units)),
        "cluerag_graph_entity_nodes": float(len(entities)),
        "cluerag_graph_edges": float(manifest.get("chunk_ku_edges", 0.0) or 0.0) + float(manifest.get("ku_entity_edges", 0.0) or 0.0),
        "cluerag_graph_reused": True,
        "offline_embedding_calls": 0.0,
        "offline_embedding_items": 0.0,
    }
    if es_index:
        offline_metrics["cluerag_graph_index"] = es_index
        offline_metrics["cluerag_es_reused"] = True
        offline_metrics["cluerag_es_documents_indexed"] = 0.0

    return SharedClueRAGIndex(
        chunks=chunks,
        knowledge_units=knowledge_units,
        entities=entities,
        chunk_entities=chunk_entities,
        ku2chunkids=ku2chunkids,
        ku2entities=ku2entities,
        entity2kus={key: sorted(set(values)) for key, values in entity2kus.items()},
        es_index=es_index,
        offline_metrics=offline_metrics,
    )


def _retrieve_shared_cluerag(
    *,
    namespace: str,
    question: str,
    shared_index: SharedClueRAGIndex,
    use_es: bool,
    mode: str,
    embedding_provider: str,
    direct_top_k: int,
    ku_top_k: int,
    graph_top_k: int,
    top_n: int,
    depth: int,
    rerank_url: str,
    rerank_model: str,
    llm_client: Any,
) -> dict[str, Any]:
    embedding_calls = 0.0
    query_vector = None
    if shared_index.es_index:
        provider = create_embedding_provider(embedding_provider)
        query_vector = provider.embed([question])[0]
        embedding_calls += 1.0

    ner_entities, ner_usage = _cluerag_query_ner(question=question, llm_client=llm_client)
    linked_entities, entity_embedding_calls = _cluerag_entity_linking(
        entities=ner_entities,
        shared_index=shared_index,
        embedding_provider=embedding_provider,
        query_vector=query_vector,
    )
    embedding_calls += entity_embedding_calls

    anchored_kus, anchoring_rerank_calls = _cluerag_knowledge_anchoring(
        question=question,
        shared_index=shared_index,
        query_vector=query_vector,
        top_m=graph_top_k,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
    )
    start_entities = set(linked_entities)
    for ku in anchored_kus:
        start_entities.update(shared_index.ku2entities.get(ku, []))
    if not start_entities:
        start_entities.update(_seed_entities(question=question, shared_index=shared_index, direct_items=[], limit=graph_top_k))

    graph_items, paths, path_rerank_calls = _expand_cluerag_multilayer_graph(
        question=question,
        shared_index=shared_index,
        seed_entities=sorted(start_entities),
        query_vector=query_vector,
        ku_top_k=ku_top_k,
        top_m=graph_top_k,
        max_depth=depth,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
    )

    direct_items = _cluerag_direct_chunk_search(
        namespace=namespace,
        question=question,
        shared_index=shared_index,
        use_es=use_es,
        mode=mode,
        embedding_provider=embedding_provider,
        query_vector=query_vector,
        top_k=direct_top_k,
    )
    if not direct_items:
        direct_items = _local_chunk_search(question=question, shared_index=shared_index, top_k=direct_top_k)

    candidates = _merge_chunk_candidates(direct_items, graph_items, shared_index=shared_index)
    if not candidates:
        candidates = _local_chunk_search(question=question, shared_index=shared_index, top_k=max(top_n, direct_top_k))
    chunks, rerank_calls = _rerank_chunks(
        question=question,
        candidates=candidates,
        top_n=top_n,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
    )
    total_rerank_calls = rerank_calls + anchoring_rerank_calls + path_rerank_calls
    return {
        "chunks": chunks,
        "paths": paths[:50],
        "rerank_calls": total_rerank_calls,
        "query_ner_calls": ner_usage["calls"],
        "query_ner_prompt_tokens": ner_usage["prompt_tokens"],
        "query_ner_completion_tokens": ner_usage["completion_tokens"],
        "embedding_calls": embedding_calls,
        "seed_entities": sorted(start_entities),
    }


def _write_cluerag_graph_artifacts(
    *,
    graph_dir: Path,
    dataset: str,
    namespace: str,
    chunks: dict[str, dict[str, Any]],
    knowledge_units: dict[str, dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    chunk_entities: dict[str, list[str]],
    ku2chunkids: dict[str, list[str]],
    ku2entities: dict[str, list[str]],
) -> None:
    write_jsonl(graph_dir / "knowledge_units.jsonl", list(knowledge_units.values()))
    write_jsonl(graph_dir / "entities.jsonl", list(entities.values()))
    write_jsonl(
        graph_dir / "chunks.jsonl",
        [
            {
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "file_name": chunk.get("file_name", ""),
                "content": chunk.get("content", ""),
                "entity_list": chunk_entities.get(chunk_id, []),
                "token_count": (chunk.get("metadata") or {}).get("token_count", 0),
            }
            for chunk_id, chunk in chunks.items()
        ],
    )
    graph_cache = {
        "ku2chunkids": ku2chunkids,
        "kuid2ku_text": {ku_id: item["text"] for ku_id, item in knowledge_units.items()},
        "ku2entities": ku2entities,
    }
    (graph_dir / "graph_cache.json").write_text(json.dumps(graph_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "method": METHOD,
        "dataset": dataset,
        "namespace": namespace,
        "graph_organization": "cluerag_multilayer_chunk_knowledge_unit_entity",
        "source_chunks": len(chunks),
        "knowledge_units": len(knowledge_units),
        "entities": len(entities),
        "chunk_ku_edges": sum(len(values) for values in ku2chunkids.values()),
        "ku_entity_edges": sum(len(values) for values in ku2entities.values()),
        "notes": "Built from shared Signpost chunks and semantic extractions; does not read Signpost unified/navigation graph.",
    }
    (graph_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _index_cluerag_graph_es(
    *,
    namespace: str,
    dataset: str,
    chunks: dict[str, dict[str, Any]],
    knowledge_units: dict[str, dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    embedding_provider_name: str,
    recreate: bool,
) -> tuple[str, dict[str, Any]]:
    provider = create_embedding_provider(embedding_provider_name)
    client = ElasticsearchClient(timeout=int(os.getenv("CLUERAG_ES_TIMEOUT", "120") or 120))
    index_name = _cluerag_es_index_name(namespace)
    if not recreate and client.exists_index(index_name):
        return (
            index_name,
            {
                "cluerag_graph_index": index_name,
                "cluerag_es_reused": True,
                "cluerag_es_documents_indexed": 0.0,
                "offline_embedding_calls": 0.0,
                "offline_embedding_items": 0.0,
            },
        )

    metrics: dict[str, Any] = {
        "cluerag_graph_index": index_name,
        "cluerag_es_reused": False,
        "cluerag_es_documents_indexed": 0.0,
        "offline_embedding_calls": 0.0,
        "offline_embedding_items": 0.0,
    }
    sample_text = next(iter(knowledge_units.values()))["text"] if knowledge_units else next(iter(chunks.values()))["content"]
    sample_vector = provider.embed([sample_text])[0]
    metrics["offline_embedding_calls"] += 1.0
    metrics["offline_embedding_items"] += 1.0
    client.create_index(index_name, _cluerag_graph_mapping(len(sample_vector)), recreate=recreate)

    batch_size = int(os.getenv("CLUERAG_EMBED_BATCH_SIZE", "32") or 32)
    chunk_docs = [
        {
            "id": chunk_id,
            "namespace": namespace,
            "dataset_id": dataset,
            "node_type": "chunk",
            "text": str(chunk.get("content") or ""),
            "chunk_id_raw": chunk_id,
            "chunk_ids": [chunk_id],
            "entity_list": [],
            "metadata": {
                "doc_id": chunk.get("doc_id", ""),
                "file_name": chunk.get("file_name", ""),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
            },
        }
        for chunk_id, chunk in chunks.items()
    ]
    ku_docs = [
        {
            "id": ku_id,
            "namespace": namespace,
            "dataset_id": dataset,
            "node_type": "knowledge_unit",
            "text": item["text"],
            "chunk_ids": item.get("chunk_ids", []),
            "entity_list": item.get("entity_list", []),
            "metadata": {"source": item.get("source", ""), "source_ids": item.get("source_ids", [])},
        }
        for ku_id, item in knowledge_units.items()
    ]
    entity_docs = [
        {
            "id": item["id"],
            "namespace": namespace,
            "dataset_id": dataset,
            "node_type": "entity",
            "text": name,
            "entity_name": name,
            "chunk_ids": item.get("chunks", []),
            "ku_ids": item.get("ku_ids", []),
            "entity_list": [name],
            "metadata": {"entity_type": item.get("entity_type", ""), "description": item.get("description", "")},
        }
        for name, item in entities.items()
    ]
    for docs in (chunk_docs, ku_docs, entity_docs):
        batch_metrics = _bulk_embed_cluerag_docs(client, index_name, docs, provider=provider, batch_size=batch_size)
        for key, value in batch_metrics.items():
            metrics[key] = float(metrics.get(key, 0.0) or 0.0) + float(value or 0.0)
    client.refresh(index_name)
    return index_name, metrics


def _bulk_embed_cluerag_docs(
    client: ElasticsearchClient,
    index_name: str,
    docs: list[dict[str, Any]],
    *,
    provider: Any,
    batch_size: int,
) -> dict[str, float]:
    metrics = {"cluerag_es_documents_indexed": 0.0, "offline_embedding_calls": 0.0, "offline_embedding_items": 0.0}
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vectors = _embed_with_retry(provider, [doc["text"] for doc in batch])
        metrics["offline_embedding_calls"] += 1.0
        metrics["offline_embedding_items"] += float(len(batch))
        operations = []
        for doc, vector in zip(batch, vectors, strict=True):
            operations.append({"index": {"_index": index_name, "_id": f"{doc['node_type']}:{doc['id']}"}})
            operations.append({**doc, "content_vector": vector})
        client.bulk(operations)
        metrics["cluerag_es_documents_indexed"] += float(len(batch))
    return metrics


def _embed_with_retry(provider: Any, texts: list[str]) -> list[list[float]]:
    attempts = int(os.getenv("CLUERAG_EMBED_RETRIES", "3") or 3) + 1
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return provider.embed(texts)
        except Exception as exc:
            last_exc = exc
            time.sleep(float(os.getenv("CLUERAG_EMBED_RETRY_SLEEP", "2") or 2))
    if len(texts) > 1:
        midpoint = len(texts) // 2
        return _embed_with_retry(provider, texts[:midpoint]) + _embed_with_retry(provider, texts[midpoint:])
    raise RuntimeError("ClueRAG graph embedding failed") from last_exc


def _cluerag_direct_chunk_search(
    *,
    namespace: str,
    question: str,
    shared_index: SharedClueRAGIndex,
    use_es: bool,
    mode: str,
    embedding_provider: str,
    query_vector: list[float] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if not use_es or not shared_index.es_index:
        return []
    try:
        payload: list[dict[str, Any]] = []
        if mode in {"dense", "hybrid"}:
            vector = query_vector or create_embedding_provider(embedding_provider).embed([question])[0]
            payload.extend(
                {
                    **row,
                    "score_source": "cluerag_chunk_vector",
                }
                for row in _cluerag_es_vector_search(
                    index_name=shared_index.es_index,
                    node_type="chunk",
                    vector=vector,
                    top_k=top_k,
                    output_fields=["id", "chunk_id_raw", "text"],
                )
            )
        if mode in {"bm25", "hybrid"}:
            payload.extend(
                {
                    **row,
                    "score_source": "cluerag_chunk_bm25",
                }
                for row in _cluerag_es_text_search(
                    index_name=shared_index.es_index,
                    node_type="chunk",
                    query=question,
                    top_k=top_k,
                    output_fields=["id", "chunk_id_raw", "text"],
                )
            )
    except Exception as exc:
        if os.getenv("CLUERAG_ALLOW_ES_FALLBACK", "0") == "1":
            return []
        raise RuntimeError("ClueRAG could not query its own Elasticsearch graph index") from exc
    merged: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(payload, start=1):
        chunk_id = str(item.get("chunk_id_raw") or item.get("id") or "")
        chunk = shared_index.chunks.get(chunk_id)
        if not chunk:
            continue
        score = float(item.get("score", 0.0) or 0.0) + 1.0 / rank
        if chunk_id not in merged:
            merged[chunk_id] = {
                **chunk,
                "chunk_id": chunk_id,
                "content": str(chunk.get("content") or item.get("text") or ""),
                "rank": rank,
                "score": score,
                "score_source": str(item.get("score_source") or "cluerag_chunk_search"),
            }
        else:
            merged[chunk_id]["score"] = float(merged[chunk_id].get("score", 0.0) or 0.0) + score
            merged[chunk_id]["score_source"] = "cluerag_chunk_hybrid"
    return sorted(merged.values(), key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)[:top_k]


def _cluerag_es_index_name(namespace: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "-", namespace.lower()).strip("-")
    if not safe:
        raise ValueError("namespace must contain at least one index-safe character")
    return f"cluerag-{safe}-multilayer"


def _cluerag_graph_mapping(vector_dimensions: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"cluerag_text": {"type": "standard"}}},
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "namespace": {"type": "keyword"},
                "dataset_id": {"type": "keyword"},
                "node_type": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "cluerag_text"},
                "entity_name": {"type": "keyword"},
                "chunk_id_raw": {"type": "keyword"},
                "chunk_ids": {"type": "keyword"},
                "ku_ids": {"type": "keyword"},
                "entity_list": {"type": "keyword"},
                "content_vector": {"type": "dense_vector", "dims": vector_dimensions, "index": True, "similarity": "cosine"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }


def _cluerag_es_vector_search(
    *,
    index_name: str,
    node_type: str,
    vector: list[float],
    top_k: int,
    output_fields: list[str],
    filters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    query_filters = [{"term": {"node_type": node_type}}, *(filters or [])]
    body = {
        "size": top_k,
        "_source": output_fields,
        "query": {
            "script_score": {
                "query": {"bool": {"filter": query_filters}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": {"query_vector": vector},
                },
            }
        },
    }
    response = ElasticsearchClient(timeout=int(os.getenv("CLUERAG_ES_TIMEOUT", "120") or 120)).request("POST", f"{index_name}/_search", body)
    rows = []
    for hit in response.get("hits", {}).get("hits", []):
        source = dict(hit.get("_source") or {})
        source["score"] = hit.get("_score", 0.0)
        rows.append(source)
    return rows


def _cluerag_es_text_search(
    *,
    index_name: str,
    node_type: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    filters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "_source": output_fields,
        "query": {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [{"term": {"node_type": node_type}}, *(filters or [])],
            }
        },
    }
    response = ElasticsearchClient(timeout=int(os.getenv("CLUERAG_ES_TIMEOUT", "120") or 120)).request("POST", f"{index_name}/_search", body)
    rows = []
    for hit in response.get("hits", {}).get("hits", []):
        source = dict(hit.get("_source") or {})
        source["score"] = hit.get("_score", 0.0)
        rows.append(source)
    return rows


def _cluerag_query_ner(*, question: str, llm_client: Any) -> tuple[list[str], dict[str, float]]:
    prompt = (
        "Your task is to extract named entities from the given question. "
        "Respond only with a JSON object in the form {\"named_entities\": [\"...\"]}.\n\n"
        f"Question: {question}"
    )
    response = llm_client.chat(
        [
            {"role": "system", "content": "You extract named entities and return strict JSON only."},
            {"role": "user", "content": prompt},
        ]
    )
    entities = []
    try:
        parsed = json.loads(response)
        values = parsed.get("named_entities", []) if isinstance(parsed, dict) else []
        entities = [_cluerag_normalize_name(value) for value in values if _cluerag_normalize_name(value)]
    except json.JSONDecodeError:
        entities = [_cluerag_normalize_name(value) for value in re.split(r"[,;\n]", response) if _cluerag_normalize_name(value)]
    return entities, {"calls": 1.0, "prompt_tokens": float(count_tokens(prompt)), "completion_tokens": float(count_tokens(response))}


def _cluerag_entity_linking(
    *,
    entities: list[str],
    shared_index: SharedClueRAGIndex,
    embedding_provider: str,
    query_vector: list[float] | None,
) -> tuple[list[str], float]:
    linked: set[str] = set()
    for entity in entities:
        key = _closest_entity_name(entity, shared_index)
        if key:
            linked.add(key)
    embedding_calls = 0.0
    if entities and shared_index.es_index:
        provider = create_embedding_provider(embedding_provider)
        vectors = provider.embed(entities)
        embedding_calls += float(len(entities))
        for vector in vectors:
            for row in _cluerag_es_vector_search(
                index_name=shared_index.es_index,
                node_type="entity",
                vector=vector,
                top_k=3,
                output_fields=["entity_name"],
            ):
                name = _cluerag_normalize_name(row.get("entity_name", ""))
                if name in shared_index.entities:
                    linked.add(name)
    return sorted(linked), embedding_calls


def _closest_entity_name(value: str, shared_index: SharedClueRAGIndex) -> str:
    normalized = _cluerag_normalize_name(value)
    if normalized in shared_index.entities:
        return normalized
    lowered = normalized.lower()
    for name in shared_index.entities:
        if name.lower() == lowered:
            return name
    query_terms = set(_terms(normalized))
    best_name = ""
    best_score = 0.0
    for name in shared_index.entities:
        name_terms = set(_terms(name))
        if not name_terms:
            continue
        score = len(query_terms & name_terms) / max(1.0, float(len(query_terms | name_terms)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score > 0.5 else ""


def _cluerag_knowledge_anchoring(
    *,
    question: str,
    shared_index: SharedClueRAGIndex,
    query_vector: list[float] | None,
    top_m: int,
    rerank_url: str,
    rerank_model: str,
) -> tuple[list[str], float]:
    candidates: list[str] = []
    if shared_index.es_index and query_vector is not None:
        rows = _cluerag_es_vector_search(
            index_name=shared_index.es_index,
            node_type="knowledge_unit",
            vector=query_vector,
            top_k=max(1, top_m * 10),
            output_fields=["text", "entity_list"],
        )
        candidates = [str(row.get("text") or "") for row in rows if str(row.get("text") or "")]
    if not candidates:
        candidates = [item["text"] for item in _local_ku_search(question=question, shared_index=shared_index, top_k=max(1, top_m * 10))]
    ranked, calls = _rerank_texts(question=question, texts=candidates, top_n=top_m, rerank_url=rerank_url, rerank_model=rerank_model)
    return ranked, calls


def _expand_cluerag_multilayer_graph(
    *,
    question: str,
    shared_index: SharedClueRAGIndex,
    seed_entities: list[str],
    query_vector: list[float] | None,
    ku_top_k: int,
    top_m: int,
    max_depth: int,
    rerank_url: str,
    rerank_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    queue = [(entity, [], {entity}) for entity in seed_entities]
    final_paths: list[list[str]] = []
    rerank_calls = 0.0
    for depth in range(max(0, max_depth)):
        if not queue:
            break
        raw_candidates = []
        for entity, path, visited in queue:
            if shared_index.es_index and query_vector is not None:
                rows = _cluerag_es_vector_search(
                    index_name=shared_index.es_index,
                    node_type="knowledge_unit",
                    vector=query_vector,
                    top_k=ku_top_k,
                    output_fields=["text", "entity_list"],
                    filters=[{"term": {"entity_list": entity}}],
                )
                scored_kus = [str(row.get("text") or "") for row in rows if str(row.get("text") or "")]
            else:
                kus = shared_index.entity2kus.get(entity, [])
                scored_kus = _score_kus_for_question(question, kus)[:ku_top_k]
            for ku_text in scored_kus:
                if ku_text in path:
                    continue
                new_path = path + [ku_text]
                final_paths.append(new_path)
                for next_entity in shared_index.ku2entities.get(ku_text, []):
                    if next_entity in visited:
                        continue
                    raw_candidates.append((next_entity, new_path, visited | {next_entity}))
        if not raw_candidates:
            break
        unique_paths = [" ".join(path) for _, path, _ in raw_candidates]
        ranked_paths, calls = _rerank_texts(
            question=question,
            texts=unique_paths,
            top_n=max(top_m, 1),
            rerank_url=rerank_url,
            rerank_model=rerank_model,
        )
        rerank_calls += calls
        allowed = set(ranked_paths)
        next_queue = []
        seen_entities = set()
        for entity, path, visited in raw_candidates:
            if " ".join(path) not in allowed or entity in seen_entities:
                continue
            next_queue.append((entity, path, visited))
            seen_entities.add(entity)
            if len(next_queue) >= top_m:
                break
        queue = next_queue

    chunk_scores: dict[str, float] = defaultdict(float)
    path_rows = []
    for rank, path in enumerate(final_paths, start=1):
        path_rows.append({"knowledge_units": path, "depth": len(path), "path_text": " ".join(path)})
        for depth_rank, ku_text in enumerate(path, start=1):
            for chunk_id in shared_index.ku2chunkids.get(ku_text, []):
                chunk_scores[chunk_id] += 1.0 / (rank + depth_rank)
    items = []
    for rank, (chunk_id, score) in enumerate(sorted(chunk_scores.items(), key=lambda pair: pair[1], reverse=True), start=1):
        chunk = shared_index.chunks.get(chunk_id)
        if not chunk:
            continue
        items.append({**chunk, "chunk_id": chunk_id, "score": score, "score_source": "cluerag_ku_entity_iter", "rank": rank})
    return items, path_rows, rerank_calls


def _score_kus_for_question(question: str, kus: list[str]) -> list[str]:
    query_terms = set(_terms(question))
    scored = []
    for ku in kus:
        terms = set(_terms(ku))
        overlap = len(query_terms & terms)
        score = overlap / max(1.0, float(len(query_terms))) if query_terms else 0.0
        scored.append((score, ku))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [ku for _, ku in scored]


def _local_ku_search(*, question: str, shared_index: SharedClueRAGIndex, top_k: int) -> list[dict[str, Any]]:
    query_terms = set(_terms(question))
    scored = []
    for item in shared_index.knowledge_units.values():
        terms = set(_terms(str(item.get("text") or "")))
        overlap = len(query_terms & terms)
        if overlap:
            scored.append((overlap / max(1.0, float(len(query_terms))), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**item, "score": score} for score, item in scored[:top_k]]


def _local_chunk_search(*, question: str, shared_index: SharedClueRAGIndex, top_k: int) -> list[dict[str, Any]]:
    query_terms = set(_terms(question))
    scored = []
    for chunk_id, chunk in shared_index.chunks.items():
        content = str(chunk.get("content") or "")
        terms = set(_terms(content))
        overlap = len(query_terms & terms)
        if not overlap:
            continue
        score = float(overlap) / max(1.0, float(len(query_terms)))
        scored.append((score, chunk_id, chunk))
    scored.sort(key=lambda value: value[0], reverse=True)
    return [
        {
            **chunk,
            "chunk_id": chunk_id,
            "score": score,
            "score_source": "local_lexical",
            "rank": rank,
        }
        for rank, (score, chunk_id, chunk) in enumerate(scored[:top_k], start=1)
    ]


def _seed_entities(
    *,
    question: str,
    shared_index: SharedClueRAGIndex,
    direct_items: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    question_lower = question.lower()
    query_terms = set(_terms(question))
    scores: dict[str, float] = defaultdict(float)
    for key, entity in shared_index.entities.items():
        name = str(entity.get("name") or key)
        name_terms = set(_terms(name))
        if not name_terms:
            continue
        if name.lower() in question_lower:
            scores[key] += 5.0
        overlap = len(query_terms & name_terms)
        if overlap:
            scores[key] += 2.0 * overlap / max(1.0, float(len(name_terms)))
    for rank, item in enumerate(direct_items, start=1):
        chunk_id = str(item.get("chunk_id") or "")
        for key in shared_index.chunk_entities.get(chunk_id, []):
            scores[key] += 1.0 / rank
    if not scores and direct_items:
        for item in direct_items[:3]:
            for key in shared_index.chunk_entities.get(str(item.get("chunk_id") or ""), []):
                scores[key] += 0.1
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [key for key, score in ranked[:limit] if score > 0]


def _expand_shared_graph(
    *,
    shared_index: SharedClueRAGIndex,
    seed_entities: list[str],
    max_depth: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_scores: dict[str, float] = defaultdict(float)
    paths: list[dict[str, Any]] = []
    visited = set(seed_entities)
    frontier = list(seed_entities)
    for depth in range(max(0, max_depth) + 1):
        next_frontier = []
        for entity_key in frontier:
            entity = shared_index.entities.get(entity_key, {})
            for chunk_id in entity.get("chunks", []) or []:
                chunk_scores[chunk_id] += 1.0 / (depth + 1)
            if depth >= max_depth:
                continue
            for edge in shared_index.adjacency.get(entity_key, []):
                other = str(edge.get("other") or "")
                chunk_id = str(edge.get("chunk_id") or "")
                if chunk_id:
                    chunk_scores[chunk_id] += float(edge.get("weight", 1.0) or 1.0) / (depth + 1)
                paths.append(_path_from_edge(edge, shared_index=shared_index, depth=depth + 1))
                if other and other not in visited:
                    visited.add(other)
                    next_frontier.append(other)
        frontier = next_frontier
        if not frontier:
            break
    ranked = sorted(chunk_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    items = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = shared_index.chunks.get(chunk_id)
        if not chunk:
            continue
        items.append(
            {
                **chunk,
                "chunk_id": chunk_id,
                "score": score,
                "score_source": "shared_entity_graph",
                "rank": rank,
            }
        )
    return items, paths


def _path_from_edge(edge: dict[str, Any], *, shared_index: SharedClueRAGIndex, depth: int) -> dict[str, Any]:
    source = shared_index.entities.get(str(edge.get("source") or ""), {})
    target = shared_index.entities.get(str(edge.get("target") or ""), {})
    return {
        "source": source.get("name") or edge.get("source"),
        "target": target.get("name") or edge.get("target"),
        "description": edge.get("description", ""),
        "keywords": edge.get("keywords", []),
        "chunk_id": edge.get("chunk_id", ""),
        "depth": depth,
    }


def _merge_chunk_candidates(
    direct_items: list[dict[str, Any]],
    graph_items: list[dict[str, Any]],
    *,
    shared_index: SharedClueRAGIndex,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_weight, items in ((2.0, direct_items), (1.0, graph_items)):
        for rank, item in enumerate(items, start=1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunk = shared_index.chunks.get(chunk_id, item)
            score = float(item.get("score", 0.0) or 0.0) + source_weight / rank
            if chunk_id not in merged:
                merged[chunk_id] = {**chunk, **item, "chunk_id": chunk_id, "score": score}
            else:
                merged[chunk_id]["score"] = float(merged[chunk_id].get("score", 0.0) or 0.0) + score
                merged[chunk_id]["score_source"] = "shared_es_graph"
    return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)


def _rerank_chunks(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    top_n: int,
    rerank_url: str,
    rerank_model: str,
) -> tuple[list[dict[str, Any]], float]:
    if not candidates:
        return [], 0.0
    limit = top_n if top_n > 0 else len(candidates)
    if not rerank_url:
        return candidates[:limit], 0.0
    documents = [str(item.get("content") or "") for item in candidates]
    payload = {"model": rerank_model, "query": question, "documents": documents}
    request = urllib.request.Request(
        rerank_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("CLUERAG_RERANK_TIMEOUT", "600") or 600)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        if os.getenv("CLUERAG_ALLOW_RERANK_FALLBACK", "0") == "1":
            return candidates[:limit], 0.0
        raise RuntimeError("ClueRAG rerank request failed") from exc

    ranked_with_scores: list[tuple[float, dict[str, Any]]] = []
    used_indexes = set()
    for fallback_rank, result in enumerate(data.get("results") or data.get("data") or []):
        if isinstance(result, dict):
            index = int(result.get("index", result.get("document_index", fallback_rank)) or 0)
            score = float(result.get("relevance_score", result.get("score", 0.0)) or 0.0)
        else:
            index = fallback_rank
            score = float(result or 0.0)
        if index < 0 or index >= len(candidates):
            continue
        item = {**candidates[index], "score": score, "score_source": "nvidia_rerank"}
        ranked_with_scores.append((score, item))
        used_indexes.add(index)
    ranked = [item for _, item in sorted(ranked_with_scores, key=lambda pair: pair[0], reverse=True)]
    if len(ranked) < limit:
        for index, item in enumerate(candidates):
            if index not in used_indexes:
                ranked.append(item)
            if len(ranked) >= limit:
                break
    return ranked[:limit], 1.0


def _rerank_texts(
    *,
    question: str,
    texts: list[str],
    top_n: int,
    rerank_url: str,
    rerank_model: str,
) -> tuple[list[str], float]:
    clean_texts = [text for text in texts if str(text).strip()]
    if not clean_texts:
        return [], 0.0
    limit = top_n if top_n > 0 else len(clean_texts)
    if not rerank_url:
        return clean_texts[:limit], 0.0
    payload = {"model": rerank_model, "query": question, "documents": clean_texts}
    request = urllib.request.Request(
        rerank_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("CLUERAG_RERANK_TIMEOUT", "600") or 600)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        if os.getenv("CLUERAG_ALLOW_RERANK_FALLBACK", "0") == "1":
            return clean_texts[:limit], 0.0
        raise RuntimeError("ClueRAG rerank request failed") from exc
    ranked = []
    used_indexes = set()
    for fallback_rank, result in enumerate(data.get("results") or data.get("data") or []):
        index = int(result.get("index", fallback_rank) if isinstance(result, dict) else fallback_rank)
        score = float(result.get("relevance_score", result.get("score", 0.0)) if isinstance(result, dict) else result or 0.0)
        if 0 <= index < len(clean_texts):
            ranked.append((score, clean_texts[index]))
            used_indexes.add(index)
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    ordered = [text for _, text in ranked]
    for index, text in enumerate(clean_texts):
        if index not in used_indexes:
            ordered.append(text)
        if len(ordered) >= limit:
            break
    return ordered[:limit], 1.0


def _cluerag_generation_prompt(
    *,
    question: str,
    chunks: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    prompt_style: str = PROMPT_STYLE_DEFAULT,
) -> str:
    context = _cluerag_generation_context(chunks=chunks, paths=paths)
    if prompt_style == PROMPT_STYLE_SIGNPOST_FEWSHOT:
        return _cluerag_signpost_fewshot_prompt(question=question, context=context)
    if prompt_style != PROMPT_STYLE_DEFAULT:
        raise ValueError(f"unknown ClueRAG prompt_style={prompt_style}")
    return (
        "Question:\n"
        f"{question}\n\n"
        f"{context}\n\n"
        "Write a concise answer. Do not use outside knowledge."
    )


def _cluerag_system_prompt(prompt_style: str) -> str:
    return "You are a careful retrieval-augmented QA assistant. Answer using only the provided context."


def _cluerag_generation_context(*, chunks: list[dict[str, Any]], paths: list[dict[str, Any]]) -> str:
    path_lines = []
    for index, path in enumerate(paths[:20], start=1):
        if isinstance(path.get("knowledge_units"), list):
            path_lines.append(f"{index}. " + " -> ".join(str(value) for value in path["knowledge_units"]))
            continue
        relation = str(path.get("description") or path.get("path_text") or "").strip()
        if relation:
            path_lines.append(f"{index}. {relation}")
    chunk_lines = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or "")
        file_name = str(chunk.get("file_name") or chunk.get("doc_id") or "")
        content = str(chunk.get("content") or "")
        chunk_lines.append(f"[{index}] chunk_id={chunk_id} source={file_name}\n{content}")
    return (
        "Retrieved knowledge paths:\n"
        f"{chr(10).join(path_lines) if path_lines else 'None'}\n\n"
        "Retrieved chunks:\n"
        f"{chr(10).join(chunk_lines)}"
    )


def _cluerag_signpost_fewshot_prompt(*, question: str, context: str) -> str:
    return (
        "As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence. "
        'Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence. '
        'Conclude with "Answer: " to present a complete, well-formed final response.\n\n'
        "Follow these rules:\n"
        "- Include all necessary context and details supported by the evidence.\n"
        "- Do not use outside knowledge.\n"
        "- Do not include citations, file names, chunk IDs, or line numbers.\n"
        "- Do not include conversational filler.\n"
        '- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".\n\n'
        "Example Input:\n"
        "Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. "
        "It also emphasizes community engagement and education to promote sustainable food practices.\n\n"
        "Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?\n"
        "Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. "
        "The evidence lists hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.\n"
        "Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. "
        "It also promotes sustainable food practices through community engagement and education.\n\n"
        "Real Input:\n"
        f"{context}\n\n"
        f"Question: {question}"
    )


def _entity_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _mdhash_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _cluerag_clean_str(value: str) -> str:
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", str(value or "").strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _cluerag_normalize_name(value: str) -> str:
    name = str(value or "").strip().strip('"\'`“”‘’«»<> ')
    for suffix in ("'s", "s'", "’s", "’"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip().strip('"\'`“”‘’«»<> ')
    return name.replace("\n", " ").replace("\r", "").strip()


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2 and term not in _STOPWORDS]


def _question_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        lookup[question_id_from_row(row, index)] = row
    return lookup


def _parse_generation(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        answer = value.get("answer") or value.get("response") or value.get("result") or ""
        rationale = value.get("thought") or value.get("rationale") or ""
        return {"answer": str(answer), "rationale": str(rationale)}
    text = str(value or "")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _parse_generation(parsed)
    except json.JSONDecodeError:
        pass
    thought_match = re.search(r"(?is)\bThought\s*:\s*(.*?)(?:\bAnswer\s*:|$)", text)
    answer_match = re.search(r"(?is)\bAnswer\s*:\s*(.*)$", text)
    if answer_match:
        return {
            "answer": answer_match.group(1).strip(),
            "rationale": thought_match.group(1).strip() if thought_match else "",
        }
    return {"answer": text, "rationale": ""}


def _result_cost(row: dict[str, Any], parsed: dict[str, str], metadata: dict[str, Any]) -> dict[str, float]:
    output_tokens = float(count_tokens(parsed["answer"]) + count_tokens(parsed["rationale"]))
    input_tokens = 0.0
    llm_calls = 1.0
    row_metadata = _row_metadata(row)
    if any(key in row_metadata for key in ("input_tokens", "output_tokens", "llm_calls")):
        input_tokens = float(row_metadata.get("input_tokens", 0.0) or 0.0)
        output_tokens = float(row_metadata.get("output_tokens", output_tokens) or output_tokens)
        llm_calls = float(row_metadata.get("llm_calls", llm_calls) or llm_calls)
        tool_calls = float(row_metadata.get("tool_calls", 1.0) or 1.0)
        return baseline_cost(
            BaselineResult(
                answer=parsed["answer"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                embedding_calls=float(row_metadata.get("embedding_calls", 0.0) or 0.0),
                rerank_calls=float(row_metadata.get("rerank_calls", 0.0) or 0.0),
                retrieval_latency_seconds=0.0,
            ),
            started_at=0.0,
            finished_at=0.0,
        )
    if metadata:
        total_prompt = float(metadata.get("prompt_tokens", 0.0) or 0.0)
        total_completion = float(metadata.get("completion_tokens", 0.0) or 0.0)
        query_count = float(metadata.get("query_count", 0.0) or 0.0)
        num_requests = float(metadata.get("num_requests", 0.0) or 0.0)
        if query_count > 0:
            input_tokens = total_prompt / query_count
            output_tokens = total_completion / query_count
            llm_calls = max(1.0, num_requests / query_count) if num_requests > 0 else 1.0
    query_count = float(metadata.get("query_count", 1.0) or 1.0)
    return baseline_cost(
        BaselineResult(
            answer=parsed["answer"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=llm_calls,
            tool_calls=float(metadata.get("knowledge_search_calls", 1.0) or 1.0) / query_count,
            embedding_calls=float(metadata.get("embedding_calls", 0.0) or 0.0) / query_count,
            rerank_calls=float(metadata.get("rerank_calls", 0.0) or 0.0) / query_count,
            retrieval_latency_seconds=0.0,
        ),
        started_at=0.0,
        finished_at=0.0,
    )


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _row_extra_model_cost(row: dict[str, Any]) -> dict[str, float]:
    metadata = _row_metadata(row)
    fields = (
        "query_ner_calls",
        "rerank_calls",
        "embedding_calls",
        "knowledge_search_calls",
    )
    return {field: float(metadata.get(field, 0.0) or 0.0) for field in fields if field in metadata}


def _combine_online_metadata(retrieval_meta: dict[str, Any], generation_meta: dict[str, Any], *, query_count: int) -> dict[str, Any]:
    prompt_tokens = float(retrieval_meta.get("prompt_tokens", 0.0) or 0.0) + float(generation_meta.get("prompt_tokens", 0.0) or 0.0)
    completion_tokens = float(retrieval_meta.get("completion_tokens", 0.0) or 0.0) + float(generation_meta.get("completion_tokens", 0.0) or 0.0)
    total_tokens = float(retrieval_meta.get("total_tokens", 0.0) or 0.0) + float(generation_meta.get("total_tokens", 0.0) or 0.0)
    num_requests = float(retrieval_meta.get("num_requests", 0.0) or 0.0) + float(generation_meta.get("num_requests", 0.0) or 0.0)
    if not num_requests:
        num_requests = float(query_count)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "num_requests": max(1.0, num_requests),
        "query_count": max(1, query_count),
        "rerank_calls": float(retrieval_meta.get("rerank_calls", 0.0) or 0.0),
        "embedding_calls": float(retrieval_meta.get("embedding_calls", 0.0) or 0.0),
        "knowledge_search_calls": float(retrieval_meta.get("knowledge_search_calls", query_count) or query_count),
    }


def _cluerag_run_metrics(result: dict[str, Any]) -> dict[str, Any]:
    timings = result.get("stage_timings", {}) if isinstance(result.get("stage_timings"), dict) else {}
    official_output_dir = result.get("official_output_dir", "")
    graph_artifact_dir = result.get("graph_artifact_dir", "")
    hybrid = _usage(result.get("hybrid_metadata"))
    retrieval = _usage(result.get("retrieval_metadata"))
    generation = _usage(result.get("generation_metadata"))
    online = _sum_usage(retrieval, generation)
    total = _sum_usage(hybrid, online)
    retrieval_metadata = result.get("retrieval_metadata", {}) if isinstance(result.get("retrieval_metadata"), dict) else {}
    offline_metadata = result.get("offline_metadata", {}) if isinstance(result.get("offline_metadata"), dict) else {}
    rerank_calls = float(retrieval_metadata.get("rerank_calls", 0.0) or 0.0)
    online_embedding_calls = float(retrieval_metadata.get("embedding_calls", 0.0) or 0.0)
    offline_embedding_calls = float(offline_metadata.get("offline_embedding_calls", 0.0) or 0.0)
    offline_disk_path = Path(str(graph_artifact_dir or official_output_dir)) if (graph_artifact_dir or official_output_dir) else None
    return {
        "offline_wall_time_seconds": float(timings.get("hybrid_extraction_seconds", 0.0) or 0.0)
        + float(timings.get("graph_construction_seconds", 0.0) or 0.0),
        "online_wall_time_seconds": float(timings.get("retrieval_seconds", 0.0) or 0.0)
        + float(timings.get("generation_seconds", 0.0) or 0.0),
        "total_wall_time_seconds": float(result.get("wall_time_seconds", 0.0) or 0.0),
        "offline_llm_calls": hybrid["llm_calls"],
        "offline_input_tokens": hybrid["input_tokens"],
        "offline_output_tokens": hybrid["output_tokens"],
        "offline_total_tokens": hybrid["total_tokens"],
        "online_llm_calls": online["llm_calls"],
        "online_input_tokens": online["input_tokens"],
        "online_output_tokens": online["output_tokens"],
        "online_total_tokens": online["total_tokens"],
        "online_rerank_calls": rerank_calls,
        "online_embedding_calls": online_embedding_calls,
        "offline_embedding_calls": offline_embedding_calls,
        "offline_embedding_items": float(offline_metadata.get("offline_embedding_items", 0.0) or 0.0),
        "offline_disk_bytes": _path_size(offline_disk_path) if offline_disk_path else 0.0,
        "online_disk_bytes": _path_size(Path(str(official_output_dir))) if official_output_dir else 0.0,
        "llm_calls": total["llm_calls"],
        "rerank_calls": rerank_calls,
        "embedding_calls": offline_embedding_calls + online_embedding_calls,
        "input_tokens": total["input_tokens"],
        "output_tokens": total["output_tokens"],
        "total_tokens": total["total_tokens"],
        "queries": float(result.get("queries", 0.0) or 0.0),
        "official_output_dir": str(official_output_dir),
        "graph_artifact_dir": str(graph_artifact_dir),
        "stage_timings": timings,
        "hybrid_metadata": result.get("hybrid_metadata", {}),
        "offline_metadata": result.get("offline_metadata", {}),
        "retrieval_metadata": result.get("retrieval_metadata", {}),
        "generation_metadata": result.get("generation_metadata", {}),
    }


def _usage(value: Any) -> dict[str, float]:
    metadata = value if isinstance(value, dict) else {}
    input_tokens = float(metadata.get("prompt_tokens", 0.0) or 0.0)
    output_tokens = float(metadata.get("completion_tokens", 0.0) or 0.0)
    total_tokens = float(metadata.get("total_tokens", input_tokens + output_tokens) or 0.0)
    return {
        "llm_calls": float(metadata.get("num_requests", 0.0) or 0.0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or input_tokens + output_tokens,
    }


def _sum_usage(*items: dict[str, float]) -> dict[str, float]:
    return {
        "llm_calls": sum(float(item.get("llm_calls", 0.0) or 0.0) for item in items),
        "input_tokens": sum(float(item.get("input_tokens", 0.0) or 0.0) for item in items),
        "output_tokens": sum(float(item.get("output_tokens", 0.0) or 0.0) for item in items),
        "total_tokens": sum(float(item.get("total_tokens", 0.0) or 0.0) for item in items),
    }


def _path_size(path: Path) -> float:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return 0.0
    if resolved.is_file():
        return float(resolved.stat().st_size)
    return float(sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file()))


def _cluerag_dataset_name(dataset: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in dataset).strip("_")
    return f"signpost_{safe}"


def _openai_embedding_base(value: str) -> str:
    base = value.rstrip("/")
    suffix = "/embeddings"
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, or convert Clue-RAG baseline outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--repo-path", default="baselines/ClueRAG")
    parser.add_argument("--cluerag-dataset")
    parser.add_argument("--documents")
    parser.add_argument("--chunks")
    parser.add_argument("--semantic-extractions")
    parser.add_argument("--questions")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--select-metric", default="COSINE")
    parser.add_argument("--official-output-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--llm-processes", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--rerank-url")
    parser.add_argument("--rerank-model")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--direct-top-k", type=int, default=10)
    parser.add_argument("--ku-top-k", type=int, default=3)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prompt-style", choices=[PROMPT_STYLE_DEFAULT, PROMPT_STYLE_SIGNPOST_FEWSHOT], default=PROMPT_STYLE_DEFAULT)
    parser.add_argument("--method-name", default=METHOD)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--source-output-dir")
    parser.add_argument("--generation-output-dir")
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--reuse-graph", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-shared", action="store_true")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()

    if not args.convert_only:
        manifest = prepare_cluerag_inputs(
            dataset=args.dataset,
            repo_path=args.repo_path,
            documents_path=args.documents,
            questions_path=args.questions,
            limit=args.limit,
            cluerag_dataset=args.cluerag_dataset,
        )
        print(f"prepared={manifest['cluerag_dataset']} documents={manifest['documents']} questions={manifest['questions']}")
    if args.prepare_only:
        return 0
    if args.generation_only:
        status = regenerate_cluerag_generation_from_retrieval(
            dataset=args.dataset,
            namespace=args.namespace,
            chunks_path=args.chunks,
            questions_path=args.questions,
            source_output_dir=args.source_output_dir or args.official_output_dir,
            output_dir=args.generation_output_dir,
            output_path=args.output,
            query_log_path=args.query_log,
            prompt_style=args.prompt_style,
            method_name=args.method_name,
        )
        output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{args.method_name}.jsonl")
        print(f"generation_output_dir={status.get('official_output_dir')}")
        print(f"output={output} count={status.get('converted_predictions', 0)}")
        return 0
    if args.run_shared:
        status = run_cluerag_shared(
            dataset=args.dataset,
            namespace=args.namespace,
            chunks_path=args.chunks,
            semantic_path=args.semantic_extractions,
            questions_path=args.questions,
            output_path=args.output,
            query_log_path=args.query_log,
            limit=args.limit,
            use_es=args.use_es,
            mode=args.mode,
            embedding_provider=args.embedding_provider,
            direct_top_k=args.direct_top_k,
            ku_top_k=args.ku_top_k,
            graph_top_k=args.graph_top_k,
            top_n=args.top_n,
            depth=args.depth,
            rerank_url=args.rerank_url,
            rerank_model=args.rerank_model,
            official_output_dir=args.official_output_dir,
            recreate_graph=not args.reuse_graph,
            prompt_style=args.prompt_style,
            method_name=args.method_name,
        )
        output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
        print(f"shared_output_dir={status.get('official_output_dir')}")
        print(f"output={output} count={status.get('converted_predictions', 0)}")
        return 0
    if args.run_official:
        status = run_cluerag_official(
            dataset=args.dataset,
            repo_path=args.repo_path,
            cluerag_dataset=args.cluerag_dataset,
            alpha=args.alpha,
            select_metric=args.select_metric,
            save_dir=args.save_dir,
            llm_processes=args.llm_processes,
            num_processes=args.num_processes,
            embedding_batch_size=args.embedding_batch_size,
            rerank_url=args.rerank_url,
        )
        args.official_output_dir = status.get("official_output_dir") or args.official_output_dir
        print(f"official_output_dir={args.official_output_dir}")
    count = convert_cluerag_outputs(
        dataset=args.dataset,
        namespace=args.namespace,
        repo_path=args.repo_path,
        cluerag_dataset=args.cluerag_dataset,
        official_output_dir=args.official_output_dir,
        questions_path=args.questions,
        output_path=args.output,
        query_log_path=args.query_log,
        method_name=args.method_name,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
