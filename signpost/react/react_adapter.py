from __future__ import annotations

"""Adapter: signpost_re retrieval result -> signpost-main ReAct KGSearchResult.

The ported ReAct stack (signpost/react/deepresearch) expects its
``KnowledgeSearchTool`` to call ``config.kg_retrievaler.process(query, ...)``
and get back a ``KGSearchResult`` whose items carry an ``InstanceSignpost``
(per-result navigation cues) and whose groups carry a ``GroupSignpost``
(group-level PPR recommendations).

This module provides a drop-in ``kg_retrievaler`` whose ``.process()`` runs
signpost_re's own Elasticsearch retrieval (``build_grouped_retrieval_result``:
chunk_search + graph_search over I_C/I_G, then attach_offline_signposts +
compute_online_signpost) and converts the result into the ReAct
``KGSearchResult`` dataclass.

Field map (signpost_re offline_signpost/online_signpost  ->  ReAct InstanceSignpost):
  semantic.neighboring_entities[].name      -> neighboring_entities
  provenance.source_chunk_ids               -> source_chunk_ids
  vertical.parent_summary.title             -> parent_node_title
  vertical.nearest_parent_summary.title     -> parent_node_title (chunk case)
  vertical.child_summaries[].title          -> child_node_titles
  provenance.source_locates / locate        -> source_locates
  provenance.file_name/start_line/end_line  -> file_name/start_line/end_line
  online_signpost.recommended_entities[].name -> GroupSignpost.related_entities
"""

from typing import Any

from signpost.react.graphrag.retrieval.kg_retrieval import (
    GraphRetrievalItem,
    GroupSignpost,
    InstanceSignpost,
    KGSearchResult,
    RetrievalGroup,
)
from signpost.retrieval.run import build_grouped_retrieval_result


# signpost_re retrieval_type -> ReAct GraphRetrievalItem.type
_TYPE_MAP = {
    "chunk": "original_chunk",
    "summary": "raptor_node",
    "entity": "graphrag_entity",
    "relation": "graphrag_edge",
}


def _names(refs: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name") or ref.get("title") or ref.get("node_id")
        if name:
            out.append(str(name))
    return out


def _instance_signpost(item: dict[str, Any]) -> InstanceSignpost:
    sp = item.get("offline_signpost") or {}
    vertical = sp.get("vertical") or {}
    horizontal = sp.get("horizontal") or {}
    semantic = sp.get("semantic") or {}
    provenance = sp.get("provenance") or {}

    # parent title: summary has 'parent_summary', chunk has 'nearest_parent_summary'
    parent = vertical.get("parent_summary") or vertical.get("nearest_parent_summary") or {}
    parent_title = parent.get("title") if isinstance(parent, dict) else None
    parent_id = parent.get("node_id") if isinstance(parent, dict) else None

    child_summaries = vertical.get("child_summaries") or []
    child_titles = [c.get("title") for c in child_summaries if isinstance(c, dict) and c.get("title")]
    child_ids = [c.get("node_id") for c in child_summaries if isinstance(c, dict) and c.get("node_id")]

    # neighboring entities: semantic.neighboring_entities (+ source/target for relations)
    neighbors = list(semantic.get("neighboring_entities") or [])
    for key in ("source_entity", "target_entity"):
        ref = semantic.get(key)
        if isinstance(ref, dict):
            neighbors.append(ref)

    # source locates: provenance.source_locates plus a chunk's single 'locate'
    locates = list(provenance.get("source_locates") or [])
    single = provenance.get("locate")
    if single:
        locates = [single] + locates
    # horizontal prev/next chunk locates also count as readable source-order context
    for key in ("previous_chunk", "next_chunk"):
        ref = horizontal.get(key)
        if isinstance(ref, dict) and ref.get("locate"):
            locates.append(ref["locate"])
    # dedupe preserving order
    seen: set[str] = set()
    locates = [x for x in locates if x and not (x in seen or seen.add(x))]

    return InstanceSignpost(
        neighboring_entities=_names(neighbors),
        source_chunk_ids=list(provenance.get("source_chunk_ids") or item.get("source_chunk_ids") or []),
        parent_node_id=parent_id,
        parent_node_title=parent_title,
        child_node_ids=child_ids,
        child_node_titles=child_titles,
        source_locates=locates,
        file_name=provenance.get("file_name"),
        start_line=provenance.get("start_line"),
        end_line=provenance.get("end_line"),
    )


def _graph_item(item: dict[str, Any]) -> GraphRetrievalItem:
    rtype = item.get("retrieval_type", "chunk")
    return GraphRetrievalItem(
        type=_TYPE_MAP.get(rtype, rtype),
        title=str(item.get("title") or item.get("name") or ""),
        content=str(item.get("content") or item.get("description") or ""),
        similarity=float(item.get("score") or item.get("similarity") or 0.0),
        signpost=_instance_signpost(item),
        kb_id=item.get("kb_id"),
    )


def _group(group: dict[str, Any]) -> RetrievalGroup:
    items = [_graph_item(it) for it in (group.get("items") or [])]
    online = group.get("online_signpost") or {}
    related = _names(online.get("recommended_entities"))
    return RetrievalGroup(items=items, group_signpost=GroupSignpost(related_entities=related))


def to_kg_search_result(retrieval: dict[str, Any]) -> KGSearchResult:
    """Convert a signpost_re grouped retrieval dict into a ReAct KGSearchResult."""
    return KGSearchResult(
        text_group=_group(retrieval.get("text_group") or {}),
        graph_group=_group(retrieval.get("graph_group") or {}),
    )


class SignpostReRetrievaler:
    """Drop-in ``kg_retrievaler`` backed by signpost_re's ES retrieval.

    Wire it as ``config.kg_retrievaler = SignpostReRetrievaler(namespace=..., graph=...)``
    before constructing the ReAct Researcher. ``process()`` matches the signature
    the ported ``KnowledgeSearchTool`` calls.
    """

    def __init__(
        self,
        *,
        namespace: str,
        graph: dict[str, Any],
        mode: str = "hybrid",
        top_k: int = 5,
        ppr_top_k: int = 5,
        embedding_provider_name: str = "ecnu",
        hash_dimensions: int = 128,
        signpost_variant: str = "full",
    ) -> None:
        self.namespace = namespace
        self.graph = graph
        self.mode = mode
        self.top_k = top_k
        self.ppr_top_k = ppr_top_k
        self.embedding_provider_name = embedding_provider_name
        self.hash_dimensions = hash_dimensions
        self.signpost_variant = signpost_variant

    def process(
        self,
        *,
        query: str,
        tenant_id: str | None = None,
        kb_ids: list[str] | None = None,
        similarity_threshold: float = 0.2,
        **_: Any,
    ) -> KGSearchResult:
        from signpost.retrieval.chunk_search import search_chunks
        from signpost.retrieval.graph_search import search_graph

        chunk_items = search_chunks(
            namespace=self.namespace, query=query, mode=self.mode, top_k=self.top_k,
            embedding_provider_name=self.embedding_provider_name, hash_dimensions=self.hash_dimensions,
        ).get("items", [])
        graph_items = search_graph(
            namespace=self.namespace, query=query, mode=self.mode, top_k=self.top_k,
            embedding_provider_name=self.embedding_provider_name, hash_dimensions=self.hash_dimensions,
        ).get("items", [])
        # summaries are produced by search_chunks/graph in signpost_re; keep summary list empty
        # unless the backend separates them (build_grouped_retrieval_result tolerates empty).
        retrieval = build_grouped_retrieval_result(
            query=query,
            graph=self.graph,
            chunk_items=chunk_items,
            summary_items=[],
            graph_items=graph_items,
            ppr_top_k=self.ppr_top_k,
            signpost_variant=self.signpost_variant,
        )
        return to_kg_search_result(retrieval)
