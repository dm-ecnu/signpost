from signpost.retrieval.online_signpost import compute_online_signpost


def _graph() -> dict:
    return {
        "nodes": [
            {"node_id": "chunk:c1", "node_type": "chunk", "chunk_id": "c1"},
            {"node_id": "chunk:c2", "node_type": "chunk", "chunk_id": "c2"},
            {"node_id": "summary:s1", "node_type": "summary", "title": "A"},
            {"node_id": "entity:e1", "node_type": "entity", "name": "Seed", "entity_type": "CONCEPT"},
            {"node_id": "entity:e2", "node_type": "entity", "name": "Target", "entity_type": "CONCEPT"},
            {"node_id": "entity:e3", "node_type": "entity", "name": "Other", "entity_type": "CONCEPT"},
        ],
        "edges": [
            {"source": "summary:s1", "target": "chunk:c1", "edge_type": "structure"},
            {"source": "entity:e1", "target": "chunk:c1", "edge_type": "source"},
            {"source": "entity:e2", "target": "chunk:c1", "edge_type": "source"},
            {"source": "entity:e1", "target": "entity:e2", "edge_type": "semantic", "weight": 3.0},
            {"source": "entity:e1", "target": "entity:e3", "edge_type": "semantic", "weight": 1.0},
        ],
    }


def test_text_scene_recommends_entities_from_chunk_seed() -> None:
    result = compute_online_signpost(_graph(), ["chunk:c1"], scene="text", top_k=2)
    assert result["scene"] == "text"
    assert result["seeds"] == ["chunk:c1"]
    assert result["recommended_entities"]
    assert {item["name"] for item in result["recommended_entities"]} >= {"Seed", "Target"}


def test_graph_scene_excludes_seed_entity_and_uses_semantic_weight() -> None:
    result = compute_online_signpost(_graph(), ["entity:e1"], scene="graph", top_k=2)
    assert result["scene"] == "graph"
    names = [item["name"] for item in result["recommended_entities"]]
    assert "Seed" not in names
    assert names[0] == "Target"


def test_auto_scene_uses_graph_for_entity_only_seeds() -> None:
    result = compute_online_signpost(_graph(), [{"node_id": "entity:e1"}], scene="auto", top_k=1)
    assert result["scene"] == "graph"
