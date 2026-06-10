import json
from pathlib import Path
from typing import Any

from signpost.indexing.embedding import HashEmbeddingProvider
from signpost.indexing.graph_es_sync import sync_graph_to_es
from signpost.indexing.graph_schema import chunk_parent_updates, graph_index_mapping, graph_index_name, graph_to_index_documents


class FakeElasticsearch:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, Any], bool]] = []
        self.bulk_operations: list[dict[str, Any]] = []
        self.refreshed: list[str] = []
        self.updates: list[tuple[str, str, dict[str, Any]]] = []

    def create_index(self, index_name: str, mapping: dict[str, Any], *, recreate: bool = False) -> None:
        self.created.append((index_name, mapping, recreate))

    def bulk(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        self.bulk_operations.extend(operations)
        return {"errors": False}

    def refresh(self, index_name: str) -> None:
        self.refreshed.append(index_name)

    def update_doc(self, index_name: str, doc_id: str, partial_doc: dict[str, Any]) -> dict[str, Any]:
        self.updates.append((index_name, doc_id, partial_doc))
        return {"result": "updated"}


def _mini_unified_graph() -> dict[str, Any]:
    return {
        "metadata": {"namespace": "mini", "graph_type": "unified"},
        "nodes": [
            {"node_id": "chunk:c1", "node_type": "chunk", "chunk_id": "c1"},
            {"node_id": "summary:s1", "node_type": "summary", "title": "章", "content": "摘要", "level": 1, "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
            {"node_id": "entity:e1", "node_type": "entity", "name": "实体", "entity_type": "CONCEPT", "description": "实体描述", "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
            {"node_id": "entity:e2", "node_type": "entity", "name": "关系对象", "entity_type": "CONCEPT", "description": "对象描述", "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
        ],
        "edges": [
            {"source": "summary:s1", "target": "chunk:c1", "edge_type": "structure"},
            {"source": "entity:e1", "target": "entity:e2", "edge_type": "semantic", "description": "相关", "relation_types": ["related_to"], "source_chunk_ids": ["c1"]},
        ],
    }


def test_graph_schema_builds_entity_relation_and_summary_docs() -> None:
    docs = graph_to_index_documents(_mini_unified_graph(), namespace="mini")
    assert [doc["object_type"] for doc in docs] == ["summary", "entity", "entity", "relation"]
    assert all(doc["type"] == "graph" for doc in docs)
    assert chunk_parent_updates(_mini_unified_graph()) == {"c1": ["summary:s1"]}
    assert graph_index_name("Mini Run") == "signpost-mini-run-graph"
    assert graph_index_mapping(8)["mappings"]["properties"]["content_vector"]["dims"] == 8


def test_graph_mapping_does_not_expand_freeform_metadata() -> None:
    mapping = graph_index_mapping(8)["mappings"]

    assert mapping["dynamic"] is False
    assert mapping["properties"]["metadata"] == {"type": "object", "enabled": False}


def test_sync_graph_to_es_with_hash_embeddings_and_parent_updates(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.unified.json"
    graph_path.write_text(json.dumps(_mini_unified_graph(), ensure_ascii=False), encoding="utf-8")
    fake_es = FakeElasticsearch()
    result = sync_graph_to_es(
        graph_path,
        namespace="mini",
        embedding_provider=HashEmbeddingProvider(dimensions=8),
        index_name="test-graph",
        recreate=True,
        update_chunk_parents=True,
        chunk_index="test-chunks",
        es=fake_es,  # type: ignore[arg-type]
    )

    assert result["indexed"] == 4
    assert result["dimensions"] == 8
    assert result["chunk_parent_updates"] == 1
    assert fake_es.created[0][0] == "test-graph"
    assert len(fake_es.bulk_operations) == 8
    assert fake_es.updates == [("test-chunks", "c1", {"parent_summary_ids": ["summary:s1"], "parent_summary_id": "summary:s1"})]
