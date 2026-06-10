from __future__ import annotations

"""F15 research tools used by the Supervisor-Researcher agent.

The old project wrapped these tools in product-specific task, tenant, and SSE
machinery.  The refactored version keeps only the research functions: search
the Signpost retrieval space and read source evidence by provenance location.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl
from signpost.retrieval.read_file import read_locate
from signpost.retrieval.run import build_grouped_retrieval_result, run_retrieval
from signpost.retrieval.signpost_variants import FULL


@dataclass(frozen=True)
class KnowledgeSearchConfig:
    namespace: str
    graph_path: Path
    chunks_path: Path | None = None
    mode: str = "hybrid"
    chunk_top_k: int = 5
    summary_top_k: int = 5
    graph_top_k: int = 5
    ppr_top_k: int = 5
    embedding_provider_name: str = "ecnu"
    use_es: bool = False
    signpost_variant: str = FULL


class KnowledgeSearchTool:
    """Search text, summaries, and graph objects, then attach Signpost cues."""

    name = "knowledge_search"

    def __init__(self, config: KnowledgeSearchConfig):
        self.config = config
        self.graph = json.loads(config.graph_path.read_text(encoding="utf-8"))

    def run(self, query: str) -> dict[str, Any]:
        if self.config.use_es:
            return run_retrieval(
                namespace=self.config.namespace,
                query=query,
                graph=self.graph,
                mode=self.config.mode,
                chunk_top_k=self.config.chunk_top_k,
                summary_top_k=self.config.summary_top_k,
                graph_top_k=self.config.graph_top_k,
                ppr_top_k=self.config.ppr_top_k,
                embedding_provider_name=self.config.embedding_provider_name,
                signpost_variant=self.config.signpost_variant,
            )
        return build_grouped_retrieval_result(
            query=query,
            graph=self.graph,
            chunk_items=_local_chunk_search(self.config.chunks_path, query, self.config.chunk_top_k),
            summary_items=_local_graph_search(self.graph, query, ["summary"], self.config.summary_top_k),
            graph_items=_local_graph_search(self.graph, query, ["entity", "relation"], self.config.graph_top_k),
            ppr_top_k=self.config.ppr_top_k,
            signpost_variant=self.config.signpost_variant,
        )


@dataclass(frozen=True)
class ReadFileConfig:
    dataset: str
    documents_path: Path | None = None
    before: int = 1
    after: int = 1


class ReadFileTool:
    """Read precise source snippets from F3.5 `documents.jsonl` artifacts."""

    name = "read_file"

    def __init__(self, config: ReadFileConfig):
        self.config = config

    def run(self, locate: str) -> dict[str, Any]:
        return read_locate(
            locate,
            dataset=self.config.dataset,
            documents_path=self.config.documents_path,
            before=self.config.before,
            after=self.config.after,
        )


def default_search_config(
    namespace: str,
    *,
    dataset: str | None = None,
    use_es: bool = False,
    embedding_provider_name: str = "ecnu",
    signpost_variant: str = FULL,
) -> KnowledgeSearchConfig:
    artifact_dataset = dataset or namespace
    graph_path = _first_existing_path(
        [
            f"datasets/processed/{artifact_dataset}/graph.unified.json",
            f"outputs/{namespace}/graph.unified.json",
        ]
    )
    chunks_path = _first_existing_path(
        [
            f"datasets/processed/{artifact_dataset}/chunks.jsonl",
            f"outputs/{namespace}/chunks.jsonl",
        ]
    )
    return KnowledgeSearchConfig(
        namespace=namespace,
        graph_path=graph_path,
        chunks_path=chunks_path if chunks_path.exists() else None,
        embedding_provider_name=embedding_provider_name,
        use_es=use_es,
        signpost_variant=signpost_variant,
    )


def _first_existing_path(paths: list[str]) -> Path:
    resolved = [resolve_project_path(path) for path in paths]
    for path in resolved:
        if path.exists():
            return path
    return resolved[0]


def _local_chunk_search(chunks_path: Path | None, query: str, top_k: int) -> list[dict[str, Any]]:
    if not chunks_path or not chunks_path.exists():
        return []
    scored = []
    for chunk in read_jsonl(chunks_path):
        score = _keyword_score(query, chunk.get("content", ""))
        if score > 0:
            scored.append({**chunk, "score": score, "score_source": "local_keyword"})
    return sorted(scored, key=lambda item: (-float(item["score"]), item.get("chunk_id", "")))[:top_k]


def _local_graph_search(graph: dict[str, Any], query: str, object_types: list[str], top_k: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if "summary" in object_types:
        for node in graph.get("nodes", []):
            if node.get("node_type") != "summary":
                continue
            content = "\n".join(str(node.get(key, "")) for key in ("title", "summary", "content"))
            score = _keyword_score(query, content)
            if score > 0:
                items.append(
                    {
                        "id": node.get("node_id"),
                        "node_id": node.get("node_id"),
                        "object_type": "summary",
                        "title": node.get("title"),
                        "content": node.get("summary") or node.get("content"),
                        "score": score,
                        "score_source": "local_keyword",
                    }
                )
    if "entity" in object_types:
        for node in graph.get("nodes", []):
            if node.get("node_type") != "entity":
                continue
            content = "\n".join(str(node.get(key, "")) for key in ("name", "description", "entity_type"))
            score = _keyword_score(query, content)
            if score > 0:
                items.append(
                    {
                        "id": node.get("node_id"),
                        "node_id": node.get("node_id"),
                        "object_type": "entity",
                        "name": node.get("name"),
                        "content": node.get("description") or node.get("name"),
                        "source_chunk_ids": node.get("source_chunk_ids") or [],
                        "source_locates": node.get("source_locates") or [],
                        "score": score,
                        "score_source": "local_keyword",
                    }
                )
    if "relation" in object_types:
        node_names = {node.get("node_id"): node.get("name") for node in graph.get("nodes", []) if node.get("node_id")}
        for edge in graph.get("edges", []):
            if edge.get("edge_type") != "semantic":
                continue
            content = " ".join(
                str(part)
                for part in [
                    node_names.get(edge.get("source"), ""),
                    node_names.get(edge.get("target"), ""),
                    " ".join(edge.get("relation_types") or []),
                    edge.get("description", ""),
                ]
            )
            score = _keyword_score(query, content)
            if score > 0:
                edge_id = f"{edge.get('source')}->{edge.get('target')}"
                items.append(
                    {
                        "id": edge_id,
                        "edge_id": edge_id,
                        "object_type": "relation",
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "content": content,
                        "source_chunk_ids": edge.get("source_chunk_ids") or [],
                        "source_locates": edge.get("source_locates") or [],
                        "score": score,
                        "score_source": "local_keyword",
                    }
                )
    return sorted(items, key=lambda item: (-float(item["score"]), item.get("id", "")))[:top_k]


def _keyword_score(query: str, text: str) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    lowered = text.lower()
    return float(sum(lowered.count(term) for term in query_terms))


def _terms(text: str) -> list[str]:
    ascii_terms = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    terms = ascii_terms + cjk_terms
    return [term for term in terms if len(term) >= 2]
