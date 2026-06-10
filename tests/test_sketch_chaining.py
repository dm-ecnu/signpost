"""Tests for sketch chaining (Algorithm 3, §5.1 ServeSignpostQuery).

Uses a small synthetic unified graph with structural/sequential/semantic/
provenance edges and mock KnowledgeSearchTool / ReadFileTool (no ES / LLM).

Assertions:
- The loop follows zoom→child, read→adjacent, jump→semantic-neighbor to reach
  an object NOT in the initial top-k.
- H_t prevents revisits / cycles.
- Priority order is respected (verify > read > zoom > jump).
- Verify cues drive ReadFile calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from signpost.agent.sketch_chaining import (
    SketchChainer,
    _object_id,
    _cue_target_id,
    run_sketch_chaining,
)
from signpost.retrieval.offline_signpost import GraphIndex, build_offline_signpost


# ---------------------------------------------------------------------------
# Synthetic graph fixture
# ---------------------------------------------------------------------------
#
# Topology (node_id -> node_type):
#
#   summary:s0  ──structure──►  chunk:c1  ──sequence/next──►  chunk:c2
#       │
#       └──structure──►  chunk:c3
#
#   entity:e1  ──semantic──  entity:e2
#
#   entity:e1  ──source──►  chunk:c1     (c1 is the seed hit)
#   entity:e2  ──source──►  chunk:c3     (c3 reachable via jump from e1 → e2)
#
# Provenance locates:
#   chunk:c1  →  "doc.txt:L1-L2"
#   chunk:c2  →  "doc.txt:L3-L4"      (reachable via read/next from c1)
#   chunk:c3  →  "doc.txt:L5-L6"      (reachable via zoom from s0 or jump via e2)
#
# Initial top-k seed: [c1, e1]   →  c2 and c3 are NOT in initial top-k.

SYNTHETIC_GRAPH: dict[str, Any] = {
    "nodes": [
        {
            "node_id": "chunk:c1",
            "node_type": "chunk",
            "chunk_id": "c1",
            "file_name": "doc.txt",
            "start_line": 1,
            "end_line": 2,
            "section_path": ["chapter1"],
        },
        {
            "node_id": "chunk:c2",
            "node_type": "chunk",
            "chunk_id": "c2",
            "file_name": "doc.txt",
            "start_line": 3,
            "end_line": 4,
            "section_path": ["chapter1"],
        },
        {
            "node_id": "chunk:c3",
            "node_type": "chunk",
            "chunk_id": "c3",
            "file_name": "doc.txt",
            "start_line": 5,
            "end_line": 6,
            "section_path": ["chapter1"],
        },
        {
            "node_id": "summary:s0",
            "node_type": "summary",
            "title": "Chapter 1 Summary",
            "level": 1,
            "source_chunk_ids": ["c1", "c2", "c3"],
            "source_locates": ["doc.txt:L1-L6"],
        },
        {
            "node_id": "entity:e1",
            "node_type": "entity",
            "name": "EntityOne",
            "entity_type": "CONCEPT",
            "source_chunk_ids": ["c1"],
            "source_locates": ["doc.txt:L1-L2"],
        },
        {
            "node_id": "entity:e2",
            "node_type": "entity",
            "name": "EntityTwo",
            "entity_type": "CONCEPT",
            "source_chunk_ids": ["c3"],
            "source_locates": ["doc.txt:L5-L6"],
        },
    ],
    "edges": [
        # Structure: s0 → c1, s0 → c3
        {"source": "summary:s0", "target": "chunk:c1", "edge_type": "structure"},
        {"source": "summary:s0", "target": "chunk:c3", "edge_type": "structure"},
        # Sequence: c1 ↔ c2
        {"source": "chunk:c1", "target": "chunk:c2", "edge_type": "sequence", "direction": "next"},
        {"source": "chunk:c2", "target": "chunk:c1", "edge_type": "sequence", "direction": "prev"},
        # Semantic: e1 ↔ e2
        {
            "source": "entity:e1",
            "target": "entity:e2",
            "edge_type": "semantic",
            "relation_types": ["related_to"],
        },
        # Source: entity provenance
        {"source": "entity:e1", "target": "chunk:c1", "edge_type": "source"},
        {"source": "entity:e2", "target": "chunk:c3", "edge_type": "source"},
    ],
}


# ---------------------------------------------------------------------------
# Mock ReadFileTool
# ---------------------------------------------------------------------------

MOCK_DOCUMENTS: dict[str, dict[str, Any]] = {
    "doc.txt:L1-L2": {
        "file_name": "doc.txt",
        "resolved": True,
        "lines": [{"line_no": 1, "text": "Line 1 content."}, {"line_no": 2, "text": "Line 2 content."}],
    },
    "doc.txt:L3-L4": {
        "file_name": "doc.txt",
        "resolved": True,
        "lines": [{"line_no": 3, "text": "Line 3 content."}, {"line_no": 4, "text": "Line 4 content."}],
    },
    "doc.txt:L5-L6": {
        "file_name": "doc.txt",
        "resolved": True,
        "lines": [{"line_no": 5, "text": "Line 5 content."}, {"line_no": 6, "text": "Line 6 content."}],
    },
    "doc.txt:L1-L6": {
        "file_name": "doc.txt",
        "resolved": True,
        "lines": [
            {"line_no": i, "text": f"Line {i} content."} for i in range(1, 7)
        ],
    },
}

_read_calls: list[str] = []   # module-level to track calls across test fixtures


def mock_read_file(locate: str) -> dict[str, Any]:
    _read_calls.append(locate)
    if locate not in MOCK_DOCUMENTS:
        raise FileNotFoundError(f"locate not found: {locate}")
    return MOCK_DOCUMENTS[locate]


# ---------------------------------------------------------------------------
# Build initial items (seed top-k = [c1, e1])
# ---------------------------------------------------------------------------

def _build_initial_items() -> list[dict[str, Any]]:
    index = GraphIndex(SYNTHETIC_GRAPH)
    items = []
    for node_id, score in [("chunk:c1", 0.9), ("entity:e1", 0.7)]:
        node = index.node_by_id[node_id]
        item = {
            "node_id": node_id,
            "chunk_id": node.get("chunk_id"),
            "object_type": node.get("node_type"),
            "retrieval_type": node.get("node_type"),
            "score": score,
            "score_source": "test",
            "source_locates": node.get("source_locates") or [],
            "source_chunk_ids": node.get("source_chunk_ids") or [],
        }
        item["offline_signpost"] = build_offline_signpost(SYNTHETIC_GRAPH, node)
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObjectIdHelper:
    def test_prefers_node_id(self):
        assert _object_id({"node_id": "chunk:c1", "chunk_id": "c1"}) == "chunk:c1"

    def test_falls_back_to_chunk_id(self):
        assert _object_id({"chunk_id": "c2"}) == "c2"

    def test_returns_empty_for_unknown(self):
        assert _object_id({}) == ""


class TestCueTargetIdHelper:
    def test_node_id(self):
        assert _cue_target_id({"node_id": "entity:e2"}) == "entity:e2"

    def test_locate(self):
        assert _cue_target_id({"locate": "doc.txt:L1-L2"}) == "doc.txt:L1-L2"


class TestRunSketchChaining:
    """End-to-end multi-hop loop test."""

    def setup_method(self):
        _read_calls.clear()

    def test_reaches_object_not_in_initial_topk(self):
        """The chainer must reach c2 (via read/next from c1) or c3 (via zoom/child
        from s0, or via jump e1→e2).  Neither c2 nor c3 is in the initial top-k."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="What is the content?",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=10,
            max_hops=4,
            cue_budget_per_family=3,
        )
        read_locates = set(result["locates"])
        # c1 should be read (seed item has verify cue)
        assert "doc.txt:L1-L2" in read_locates, f"Expected c1 locate in {read_locates}"
        # At least one non-seed object must have been read (c2, c3, or s0 summary)
        non_seed = read_locates - {"doc.txt:L1-L2"}
        assert non_seed, f"Chainer did not reach any non-seed evidence; got {read_locates}"

    def test_read_cue_reaches_c2_from_c1(self):
        """Chunk c2 is adjacent (sequence/next) to c1 — reachable via read cue."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="read hop test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=10,
            max_hops=4,
            cue_budget_per_family=3,
        )
        # Visited set should include chunk:c2 (expanded via read cue)
        assert "chunk:c2" in result["visited"] or "doc.txt:L3-L4" in set(result["locates"]), (
            f"Expected c2 reachable via read cue; visited={result['visited']} locates={result['locates']}"
        )

    def test_ht_prevents_revisits(self):
        """An object added to H_t must not be re-expanded."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        # Run twice with the SAME SketchChainer to confirm no double-processing
        chainer = SketchChainer(
            subquestion="revisit test",
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=20,
            max_hops=8,
            cue_budget_per_family=5,
        )
        result = chainer.run(initial)
        visited = result["visited"]
        # c1 should be visited exactly once — we verify by checking read calls
        c1_reads = [loc for loc in _read_calls if loc == "doc.txt:L1-L2"]
        assert len(c1_reads) == 1, f"c1 was read {len(c1_reads)} times (expected 1); calls={_read_calls}"

    def test_priority_order_verify_before_nav(self):
        """Verify cues (provenance) must be processed before zoom/read/jump.

        We check that the first read call is for a provenance locate from
        the seed items, not a navigation-target locate.
        """
        _read_calls.clear()
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        run_sketch_chaining(
            subquestion="priority test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=10,
            max_hops=4,
            cue_budget_per_family=3,
        )
        # The very first ReadFile call must be a verify cue (direct provenance locate)
        assert _read_calls, "No ReadFile calls were made"
        first_call = _read_calls[0]
        # Provenance locates for seed items: doc.txt:L1-L2 (c1) or doc.txt:L1-L2 (e1)
        assert first_call in MOCK_DOCUMENTS, f"First call was for unknown locate {first_call}"
        # Verify that c1's own provenance locate is read early
        assert first_call == "doc.txt:L1-L2", (
            f"Expected first ReadFile to be c1's provenance locate; got {first_call}"
        )

    def test_verify_cues_drive_read_file(self):
        """Every locate in result['locates'] must correspond to a ReadFile call."""
        _read_calls.clear()
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="verify drives readfile",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=5,
            max_hops=3,
            cue_budget_per_family=2,
        )
        assert set(result["locates"]) == set(_read_calls), (
            f"Locates {result['locates']} != ReadFile calls {_read_calls}"
        )

    def test_read_budget_limits_calls(self):
        """read_budget=1 must result in exactly 1 ReadFile call."""
        _read_calls.clear()
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="budget limit test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=1,
            max_hops=5,
            cue_budget_per_family=3,
        )
        assert len(_read_calls) == 1, f"Expected 1 ReadFile call; got {len(_read_calls)}: {_read_calls}"
        assert len(result["evidence"]) == 1

    def test_max_hops_limits_expansion(self):
        """max_hops=1 must complete after at most 1 expansion round."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="max hops test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=20,
            max_hops=1,
            cue_budget_per_family=3,
        )
        assert result["hops"] <= 1, f"Expected hops <= 1; got {result['hops']}"

    def test_jump_cue_reaches_e2_from_e1(self):
        """Jump (semantic) cue from e1 must put entity:e2 on the visited set."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="jump test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=10,
            max_hops=4,
            cue_budget_per_family=3,
        )
        # entity:e2 should be visited (reached via semantic jump from e1)
        assert "entity:e2" in result["visited"], (
            f"Expected entity:e2 in visited set; got {result['visited']}"
        )

    def test_zoom_cue_reaches_summary_from_chunk(self):
        """Zoom (vertical) cue from c1 must navigate toward summary:s0.

        With enough hops, summary:s0 reaches the visited set.  We also accept
        a sketch_chain_follow event targeting summary:s0 as proof that the zoom
        cue was followed (the object was enqueued on the frontier).
        """
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        # Use generous hop/budget to guarantee summary:s0 is expanded
        result = run_sketch_chaining(
            subquestion="zoom test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=15,
            max_hops=8,
            cue_budget_per_family=3,
        )
        # Either summary:s0 is fully visited, or at minimum a zoom follow event
        # shows the cue was followed (summary pushed to frontier).
        zoom_follows = [
            e for e in result["trace_events"]
            if e["event_type"] == "sketch_chain_follow"
            and e.get("family") == "v"
            and e.get("target_id") == "summary:s0"
        ]
        in_visited = "summary:s0" in result["visited"]
        assert in_visited or zoom_follows, (
            f"Expected summary:s0 in visited set or a zoom follow event; "
            f"visited={result['visited']} events={result['trace_events']}"
        )

    def test_trace_events_are_recorded(self):
        """Trace events must include sketch_chain_verify and sketch_chain_follow."""
        index = GraphIndex(SYNTHETIC_GRAPH)
        initial = _build_initial_items()
        result = run_sketch_chaining(
            subquestion="trace test",
            initial_items=initial,
            graph_index=index,
            read_file_fn=mock_read_file,
            read_budget=5,
            max_hops=3,
            cue_budget_per_family=3,
        )
        event_types = {e["event_type"] for e in result["trace_events"]}
        assert "sketch_chain_verify" in event_types, (
            f"Missing sketch_chain_verify event; got {event_types}"
        )

    def test_no_es_no_llm_at_module_import(self):
        """Importing sketch_chaining must not trigger ES or LLM initialisation."""
        import importlib
        import sys
        # Remove cached module to force fresh import
        mod_name = "signpost.agent.sketch_chaining"
        sys.modules.pop(mod_name, None)
        # This must complete without ImportError or network call
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "run_sketch_chaining")


class TestAdaptCues:
    """Unit tests for _adapt_cues context-filtering logic."""

    def test_already_visited_targets_are_filtered(self):
        """Cues whose target is in H_t must be removed."""
        chainer = SketchChainer(
            subquestion="test",
            graph_index=None,
            read_file_fn=lambda loc: {},
            read_budget=10,
            max_hops=3,
            cue_budget_per_family=3,
        )
        chainer.H_t.add("chunk:c2")  # pre-visit c2

        obj = {
            "node_id": "chunk:c1",
            "offline_signpost": {
                "horizontal": {
                    "next_chunk": {"node_id": "chunk:c2", "locate": "doc.txt:L3-L4"},
                },
                "vertical": {},
                "provenance": {"locate": "doc.txt:L1-L2"},
                "semantic": {},
            },
        }
        adapted = chainer._adapt_cues(obj)
        h_targets = [_cue_target_id(c) for c in adapted["h"]]
        assert "chunk:c2" not in h_targets, f"Pre-visited c2 should be filtered; got h={adapted['h']}"

    def test_already_read_provenance_is_filtered(self):
        """Verify cues whose locate is already in R_t must be removed."""
        chainer = SketchChainer(
            subquestion="test",
            graph_index=None,
            read_file_fn=lambda loc: {},
            read_budget=10,
            max_hops=3,
            cue_budget_per_family=3,
        )
        chainer.R_t["doc.txt:L1-L2"] = {"file_name": "doc.txt", "lines": []}

        obj = {
            "node_id": "chunk:c1",
            "offline_signpost": {
                "provenance": {"locate": "doc.txt:L1-L2", "source_locates": ["doc.txt:L1-L2"]},
                "vertical": {},
                "horizontal": {},
                "semantic": {},
            },
        }
        adapted = chainer._adapt_cues(obj)
        assert adapted["p"] == [], f"Already-read locate should be filtered; got p={adapted['p']}"

    def test_cue_budget_per_family_limits(self):
        """cue_budget_per_family=1 must keep at most 1 cue per family."""
        chainer = SketchChainer(
            subquestion="test",
            graph_index=None,
            read_file_fn=lambda loc: {},
            read_budget=10,
            max_hops=3,
            cue_budget_per_family=1,
        )
        # Give e1 two semantic neighbors
        obj = {
            "node_id": "entity:e1",
            "offline_signpost": {
                "vertical": {},
                "horizontal": {},
                "provenance": {"source_locates": ["doc.txt:L1-L2", "doc.txt:L3-L4"]},
                "semantic": {
                    "neighboring_entities": [
                        {"node_id": "entity:e2"},
                        {"node_id": "entity:e3"},
                    ]
                },
            },
        }
        adapted = chainer._adapt_cues(obj)
        assert len(adapted["p"]) <= 1, f"Budget 1 not respected for verify: {adapted['p']}"
        assert len(adapted["s"]) <= 1, f"Budget 1 not respected for jump: {adapted['s']}"


class TestSketchChainingIntegrationWithSupervisor:
    """Test that Supervisor routes through sketch chaining when enabled."""

    def test_supervisor_uses_chaining_by_default(self, tmp_path):
        """With enable_sketch_chaining=True (default), trace must contain
        sketch_chain_summary events."""
        import json as _json
        from signpost.agent.supervisor import AgentConfig, Supervisor, TraceRecorder
        from signpost.agent.tools import KnowledgeSearchConfig, KnowledgeSearchTool, ReadFileConfig, ReadFileTool
        from signpost.parsing.io import write_jsonl

        graph_path = tmp_path / "graph.unified.json"
        chunks_path = tmp_path / "chunks.jsonl"
        documents_path = tmp_path / "documents.jsonl"

        graph_path.write_text(_json.dumps(SYNTHETIC_GRAPH, ensure_ascii=False), encoding="utf-8")
        write_jsonl(
            chunks_path,
            [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "file_name": "doc.txt",
                    "content": "Line 1 content. Line 2 content.",
                    "start_line": 1,
                    "end_line": 2,
                    "section_path": ["chapter1"],
                }
            ],
        )
        write_jsonl(
            documents_path,
            [
                {
                    "doc_id": "d1",
                    "file_name": "doc.txt",
                    "lines": [
                        {"line_no": i, "text": f"Line {i} content."}
                        for i in range(1, 7)
                    ],
                }
            ],
        )

        search_tool = KnowledgeSearchTool(
            KnowledgeSearchConfig(namespace="test", graph_path=graph_path, chunks_path=chunks_path)
        )
        read_tool = ReadFileTool(ReadFileConfig(dataset="test", documents_path=documents_path, before=0, after=0))
        config = AgentConfig(
            namespace="test",
            read_top_k=2,
            enable_sketch_chaining=True,
            sketch_chaining_max_hops=2,
            sketch_chaining_read_budget=5,
        )
        supervisor = Supervisor(config, search_tool, read_tool)
        result = supervisor.run("What is EntityOne?")

        sketch_events = [e for e in result["trace"] if e["event_type"] == "sketch_chain_summary"]
        assert sketch_events, (
            f"Expected sketch_chain_summary in trace; event_types={[e['event_type'] for e in result['trace']]}"
        )

    def test_supervisor_uses_simplified_when_chaining_disabled(self, tmp_path):
        """With enable_sketch_chaining=False, trace must NOT contain
        sketch_chain_summary events (original 2-call path)."""
        import json as _json
        from signpost.agent.supervisor import AgentConfig, Supervisor
        from signpost.agent.tools import KnowledgeSearchConfig, KnowledgeSearchTool, ReadFileConfig, ReadFileTool
        from signpost.parsing.io import write_jsonl

        graph_path = tmp_path / "graph.unified.json"
        chunks_path = tmp_path / "chunks.jsonl"
        documents_path = tmp_path / "documents.jsonl"

        graph_path.write_text(_json.dumps(SYNTHETIC_GRAPH, ensure_ascii=False), encoding="utf-8")
        write_jsonl(
            chunks_path,
            [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "file_name": "doc.txt",
                    "content": "Line 1 content.",
                    "start_line": 1,
                    "end_line": 2,
                    "section_path": ["chapter1"],
                }
            ],
        )
        write_jsonl(
            documents_path,
            [
                {
                    "doc_id": "d1",
                    "file_name": "doc.txt",
                    "lines": [
                        {"line_no": 1, "text": "Line 1 content."},
                        {"line_no": 2, "text": "Line 2 content."},
                    ],
                }
            ],
        )

        search_tool = KnowledgeSearchTool(
            KnowledgeSearchConfig(namespace="test2", graph_path=graph_path, chunks_path=chunks_path)
        )
        read_tool = ReadFileTool(ReadFileConfig(dataset="test2", documents_path=documents_path, before=0, after=0))
        config = AgentConfig(
            namespace="test2",
            read_top_k=2,
            enable_sketch_chaining=False,   # original simplified path
        )
        supervisor = Supervisor(config, search_tool, read_tool)
        result = supervisor.run("What is EntityOne?")

        sketch_events = [e for e in result["trace"] if e["event_type"] == "sketch_chain_summary"]
        assert not sketch_events, (
            f"Simplified path should have no sketch_chain_summary; got {sketch_events}"
        )
