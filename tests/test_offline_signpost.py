from signpost.retrieval.offline_signpost import build_offline_signpost


def _graph() -> dict:
    return {
        "metadata": {"graph_type": "unified"},
        "nodes": [
            {"node_id": "chunk:c1", "node_type": "chunk", "chunk_id": "c1", "file_name": "a.txt", "start_line": 1, "end_line": 2, "section_path": ["A"]},
            {"node_id": "chunk:c2", "node_type": "chunk", "chunk_id": "c2", "file_name": "a.txt", "start_line": 3, "end_line": 4, "section_path": ["A"]},
            {"node_id": "summary:s1", "node_type": "summary", "title": "A", "level": 1, "source_chunk_ids": ["c1", "c2"], "source_locates": ["a.txt:L1-L2", "a.txt:L3-L4"]},
            {"node_id": "entity:e1", "node_type": "entity", "name": "X", "entity_type": "CONCEPT", "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
            {"node_id": "entity:e2", "node_type": "entity", "name": "Y", "entity_type": "CONCEPT", "source_chunk_ids": ["c2"], "source_locates": ["a.txt:L3-L4"]},
        ],
        "edges": [
            {"source": "summary:s1", "target": "chunk:c1", "edge_type": "structure"},
            {"source": "summary:s1", "target": "chunk:c2", "edge_type": "structure"},
            {"source": "chunk:c1", "target": "chunk:c2", "edge_type": "sequence", "direction": "next"},
            {"source": "chunk:c2", "target": "chunk:c1", "edge_type": "sequence", "direction": "prev"},
            {"source": "entity:e1", "target": "chunk:c1", "edge_type": "source"},
            {"source": "entity:e1", "target": "entity:e2", "edge_type": "semantic", "relation_types": ["related_to"], "source_chunk_ids": ["c1"], "source_locates": ["a.txt:L1-L2"]},
        ],
    }


def test_chunk_offline_signpost_has_vertical_horizontal_and_provenance() -> None:
    signpost = build_offline_signpost(_graph(), {"chunk_id": "c1"})
    assert signpost["result_type"] == "chunk"
    assert signpost["vertical"]["nearest_parent_summary"]["title"] == "A"
    assert signpost["horizontal"]["next_chunk"]["chunk_id"] == "c2"
    assert signpost["provenance"]["locate"] == "a.txt:L1-L2"


def test_summary_offline_signpost_merges_source_locates() -> None:
    signpost = build_offline_signpost(_graph(), "summary:s1")
    assert signpost["result_type"] == "summary"
    assert signpost["vertical"]["child_chunks"][0]["chunk_id"] == "c1"
    assert signpost["provenance"]["source_locates"] == ["a.txt:L1-L4"]


def test_entity_and_relation_offline_signposts_have_neighbors() -> None:
    entity = build_offline_signpost(_graph(), "entity:e1")
    assert entity["semantic"]["neighboring_entities"][0]["name"] == "Y"
    relation = build_offline_signpost(_graph(), {"source": "entity:e1", "target": "entity:e2"})
    assert relation["result_type"] == "relation"
    assert relation["semantic"]["source_entity"]["name"] == "X"
