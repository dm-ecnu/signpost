from signpost.retrieval.run import build_grouped_retrieval_result


def _graph() -> dict:
    return {
        "nodes": [
            {"node_id": "chunk:c1", "node_type": "chunk", "chunk_id": "c1", "file_name": "a.txt", "start_line": 1, "end_line": 2, "section_path": ["A"]},
            {"node_id": "summary:s1", "node_type": "summary", "title": "A", "level": 1, "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
            {"node_id": "entity:e1", "node_type": "entity", "name": "X", "entity_type": "CONCEPT", "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
            {"node_id": "entity:e2", "node_type": "entity", "name": "Y", "entity_type": "CONCEPT", "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
        ],
        "edges": [
            {"source": "summary:s1", "target": "chunk:c1", "edge_type": "structure"},
            {"source": "entity:e1", "target": "chunk:c1", "edge_type": "source"},
            {"source": "entity:e2", "target": "chunk:c1", "edge_type": "source"},
            {"source": "entity:e1", "target": "entity:e2", "edge_type": "semantic", "relation_types": ["related_to"], "weight": 2.0, "source_chunk_ids": ["c1"]},
        ],
    }


def test_build_grouped_retrieval_result_attaches_offline_and_online_signposts() -> None:
    result = build_grouped_retrieval_result(
        query="x",
        graph=_graph(),
        chunk_items=[{"chunk_id": "c1", "content": "chunk"}],
        summary_items=[{"id": "summary:s1", "node_id": "summary:s1", "object_type": "summary", "content": "summary"}],
        graph_items=[{"id": "entity:e1", "node_id": "entity:e1", "object_type": "entity", "content": "entity"}],
        ppr_top_k=2,
    )

    assert result["text_group"]["items"][0]["offline_signpost"]["result_type"] == "chunk"
    assert result["text_group"]["items"][1]["offline_signpost"]["result_type"] == "summary"
    assert result["graph_group"]["items"][0]["offline_signpost"]["result_type"] == "entity"
    assert result["text_group"]["online_signpost"]["recommended_entities"]
    assert result["graph_group"]["online_signpost"]["recommended_entities"][0]["name"] == "Y"
    assert result["metadata"] == {"text_items": 2, "graph_items": 1, "ppr_top_k": 2, "signpost_variant": "full"}


def test_build_grouped_retrieval_result_applies_ablation_variant() -> None:
    result = build_grouped_retrieval_result(
        query="x",
        graph=_graph(),
        chunk_items=[{"chunk_id": "c1", "content": "chunk"}],
        summary_items=[],
        graph_items=[{"id": "entity:e1", "node_id": "entity:e1", "object_type": "entity", "content": "entity"}],
        ppr_top_k=2,
        signpost_variant="no_semantic_cues",
    )

    assert result["metadata"]["signpost_variant"] == "no_semantic_cues"
    assert "semantic" not in result["text_group"]["items"][0]["offline_signpost"]
    assert result["text_group"]["online_signpost"]["recommended_entities"] == []
    assert result["graph_group"]["online_signpost"]["recommended_entities"] == []
