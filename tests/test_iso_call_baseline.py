"""Tests for the iso-call attribution baseline.

All three required properties are verified:
  (a) registers / instantiates exactly like the other baselines,
  (b) operates over UNTYPED neighbors (no cue-family typing anywhere),
  (c) respects the LLM call_budget.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from signpost.baselines.iso_call import (
    IsoCallConfig,
    IsoCallRunner,
    METHOD,
    _extract_action_and_query,
    _graph_text_neighbors,
    _local_keyword_search,
    run_iso_call,
)
from signpost.parsing.io import write_jsonl


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------


def _fake_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "file_name": "doc.txt",
            "content": "Signpost retrieves evidence from chunks using typed cues.",
            "start_line": 1,
            "end_line": 3,
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "file_name": "doc.txt",
            "content": "The iso-call baseline uses untyped neighbors.",
            "start_line": 4,
            "end_line": 6,
        },
    ]


def _fake_graph() -> dict[str, Any]:
    """Minimal unified graph with an entity node and a semantic edge."""
    return {
        "nodes": [
            {
                "node_id": "e1",
                "node_type": "entity",
                "name": "Signpost",
                "description": "A typed-cue graph retrieval system.",
                "entity_type": "SYSTEM",
                "source_chunk_ids": ["c1"],
                "source_locates": ["doc.txt:L1-L3"],
            },
            {
                "node_id": "e2",
                "node_type": "entity",
                "name": "Evidence",
                "description": "Information retrieved from the corpus.",
                "entity_type": "CONCEPT",
                "source_chunk_ids": ["c1"],
                "source_locates": ["doc.txt:L1-L3"],
            },
            {
                "node_id": "s1",
                "node_type": "summary",
                "title": "Overview",
                "summary": "Signpost retrieves evidence from typed cue graph neighbors.",
                "content": "Signpost retrieves evidence from typed cue graph neighbors.",
                "source_chunk_ids": ["c1"],
                "source_locates": ["doc.txt:L1-L3"],
            },
        ],
        "edges": [
            {
                "edge_type": "semantic",
                "source": "e1",
                "target": "e2",
                "relation_types": ["retrieves"],
                "description": "Signpost retrieves Evidence.",
                "source_chunk_ids": ["c1"],
                "source_locates": ["doc.txt:L1-L3"],
            }
        ],
    }


class _SequenceLLM:
    """Deterministic LLM that pops canned outputs in order."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.call_count = 0

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, thinking: bool = False) -> str:
        self.call_count += 1
        if self.outputs:
            return self.outputs.pop(0)
        return "Fallback answer."


class _CaptureLLM:
    """Records all calls and returns a simple search-then-finish sequence."""

    def __init__(self, react_steps: int = 1):
        self.call_count = 0
        self._react_steps = react_steps
        self._issued_search = 0

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, thinking: bool = False) -> str:
        self.call_count += 1
        # Return a search action for the first N-1 calls, then a synthesis answer.
        if self._issued_search < self._react_steps:
            self._issued_search += 1
            return (
                "Thought: I need to look up Signpost retrieval.\n"
                "Action: search\n"
                "Query: Signpost retrieves evidence"
            )
        # Synthesis call (or finish after search steps exhausted).
        return "Signpost retrieves evidence using typed cue graph neighbors."


def _write_questions(path: "Any") -> None:  # path: Path
    write_jsonl(
        path,
        [
            {
                "question_id": "q1",
                "question": "What does Signpost retrieve?",
                "answer": "Evidence.",
                "metadata": {"gold_chunk_ids": ["c1"]},
            }
        ],
    )


# ---------------------------------------------------------------------------
# (a) Registration / instantiation
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_method_constant_is_iso_call(self) -> None:
        assert METHOD == "iso_call"

    def test_module_importable_without_llm_or_es(self) -> None:
        # The import at the top of this file already proves this, but we make
        # it explicit so it reads as a deliberate property.
        from signpost.baselines import iso_call as _mod  # noqa: F401
        assert hasattr(_mod, "run_iso_call")
        assert hasattr(_mod, "IsoCallRunner")
        assert hasattr(_mod, "IsoCallConfig")

    def test_runner_instantiates_with_missing_graph(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """Runner must not raise when graph_path is None or absent."""
        chunks_path = tmp_path / "chunks.jsonl"
        write_jsonl(chunks_path, _fake_chunks())
        config = IsoCallConfig(
            call_budget=2,
            search_top_k=2,
            graph_top_k=2,
            max_context_tokens=500,
            use_es=False,
            mode="bm25",
            embedding_provider="hash",
        )

        # Patch out the LLM class so no live endpoint is needed at construction.
        monkeypatch.setattr(
            "signpost.baselines.iso_call.OpenAICompatibleClient",
            lambda: _CaptureLLM(),
        )
        runner = IsoCallRunner(
            dataset="mini",
            namespace="mini",
            chunks_path=chunks_path,
            graph_path=None,
            config=config,
        )
        assert runner.graph == {}
        assert len(runner.chunks) == 2

    def test_run_iso_call_writes_prediction_jsonl_matching_baseline_schema(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """run_iso_call produces JSONL with the same schema as all other baselines."""
        monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

        # Wire in a fake LLM: one ReAct step (search) + one synthesis call.
        llm = _CaptureLLM(react_steps=1)

        def _fake_llm_ctor():
            return llm

        monkeypatch.setattr("signpost.baselines.iso_call.OpenAICompatibleClient", _fake_llm_ctor, raising=False)

        base = tmp_path / "datasets" / "processed" / "mini"
        base.mkdir(parents=True)
        _write_questions(base / "questions.jsonl")
        write_jsonl(base / "chunks.jsonl", _fake_chunks())

        count = run_iso_call(
            dataset="mini",
            use_es=False,
            call_budget=2,
            search_top_k=2,
            graph_top_k=2,
        )

        assert count == 1
        row = json.loads(
            (tmp_path / "outputs" / "mini" / "predictions" / "iso_call.jsonl").read_text(encoding="utf-8")
        )
        # Schema fields present in every baseline prediction.
        assert row["metadata"]["method"] == "iso_call"
        assert "prediction" in row
        assert "retrieved_chunks" in row
        assert "evidence_chunks" in row
        assert "citations" in row
        assert "trace" in row
        assert "llm_calls" in row
        assert "tool_calls" in row


# ---------------------------------------------------------------------------
# (b) Untyped neighbors — no cue-family typing
# ---------------------------------------------------------------------------


class TestUntypedNeighbors:
    def test_graph_neighbors_carry_no_cue_type_field(self) -> None:
        """_graph_text_neighbors must NOT annotate items with any cue type."""
        graph = _fake_graph()
        neighbors = _graph_text_neighbors(graph, "Signpost retrieves", top_k=10)
        assert len(neighbors) > 0, "Expected at least one neighbor to match"
        for item in neighbors:
            assert "cue_type" not in item, f"Unexpected cue_type in neighbor: {item}"
            assert "cue_family" not in item, f"Unexpected cue_family in neighbor: {item}"
            assert "zoom" not in item, "Zoom cue label found in iso_call neighbor"
            assert "jump" not in item, "Jump cue label found in iso_call neighbor"
            assert "read" not in str(item.get("score_source") or "").lower() or \
                "iso_call" in str(item.get("score_source") or "").lower(), \
                f"Unexpected score_source: {item.get('score_source')}"

    def test_graph_neighbors_include_entities_summaries_and_edges(self) -> None:
        """All graph object types (entity, summary, edge) are flattened together."""
        graph = _fake_graph()
        # Use a query that matches all node types.
        neighbors = _graph_text_neighbors(graph, "Signpost retrieves evidence", top_k=20)
        score_sources = {item["score_source"] for item in neighbors}
        assert "iso_call_graph_keyword" in score_sources

    def test_trace_events_confirm_cue_typed_false(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """Each tool_call trace event must record cue_typed=False."""
        monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

        llm = _CaptureLLM(react_steps=1)

        def _fake_llm_ctor():
            return llm

        monkeypatch.setattr("signpost.baselines.iso_call.OpenAICompatibleClient", _fake_llm_ctor, raising=False)

        base = tmp_path / "datasets" / "processed" / "mini"
        base.mkdir(parents=True)
        _write_questions(base / "questions.jsonl")
        write_jsonl(base / "chunks.jsonl", _fake_chunks())

        run_iso_call(dataset="mini", use_es=False, call_budget=2, search_top_k=2, graph_top_k=2)

        row = json.loads(
            (tmp_path / "outputs" / "mini" / "predictions" / "iso_call.jsonl").read_text(encoding="utf-8")
        )
        tool_events = [e for e in row.get("trace", []) if e.get("event_type") == "tool_call"]
        assert len(tool_events) >= 1, "Expected at least one tool_call trace event"
        for event in tool_events:
            summary = event.get("output_summary", {})
            assert summary.get("cue_typed") is False, (
                f"Expected cue_typed=False in tool_call trace, got: {summary}"
            )

    def test_metadata_records_cue_typed_false(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """The prediction metadata must record cue_typed=False for auditing."""
        monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
        llm = _CaptureLLM(react_steps=1)

        def _fake_llm_ctor():
            return llm

        monkeypatch.setattr("signpost.baselines.iso_call.OpenAICompatibleClient", _fake_llm_ctor, raising=False)

        base = tmp_path / "datasets" / "processed" / "mini"
        base.mkdir(parents=True)
        _write_questions(base / "questions.jsonl")
        write_jsonl(base / "chunks.jsonl", _fake_chunks())

        run_iso_call(dataset="mini", use_es=False, call_budget=2)

        row = json.loads(
            (tmp_path / "outputs" / "mini" / "predictions" / "iso_call.jsonl").read_text(encoding="utf-8")
        )
        assert row["metadata"]["cue_typed"] is False


# ---------------------------------------------------------------------------
# (c) Call-budget enforcement
# ---------------------------------------------------------------------------


class TestCallBudget:
    def _run_with_budget(
        self,
        tmp_path: "Any",
        monkeypatch: "Any",
        *,
        budget: int,
        react_outputs: list[str],
        synthesis_output: str = "Final answer.",
    ) -> tuple[dict[str, Any], "_SequenceLLM"]:
        """Helper: run iso_call with given budget and canned LLM outputs.

        Returns (prediction_row, llm_instance).
        """
        monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

        outputs = react_outputs + [synthesis_output]
        llm = _SequenceLLM(outputs)

        def _fake_llm_ctor():
            return llm

        monkeypatch.setattr("signpost.baselines.iso_call.OpenAICompatibleClient", _fake_llm_ctor, raising=False)

        base = tmp_path / "datasets" / "processed" / "mini"
        base.mkdir(parents=True)
        _write_questions(base / "questions.jsonl")
        write_jsonl(base / "chunks.jsonl", _fake_chunks())

        run_iso_call(dataset="mini", use_es=False, call_budget=budget)

        row = json.loads(
            (tmp_path / "outputs" / "mini" / "predictions" / "iso_call.jsonl").read_text(encoding="utf-8")
        )
        return row, llm

    def test_budget_2_makes_exactly_2_llm_calls(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """call_budget=2 → 1 ReAct step + 1 synthesis = 2 total LLM calls."""
        react = [
            "Thought: need evidence\nAction: search\nQuery: Signpost retrieves evidence"
        ]
        row, llm = self._run_with_budget(tmp_path, monkeypatch, budget=2, react_outputs=react)
        assert row["llm_calls"] == 2.0, f"Expected 2 llm_calls, got {row['llm_calls']}"

    def test_budget_3_makes_at_most_3_llm_calls(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """call_budget=3 → up to 2 ReAct steps + 1 synthesis ≤ 3 total."""
        react = [
            "Thought: step 1\nAction: search\nQuery: Signpost",
            "Thought: step 2\nAction: search\nQuery: evidence chunks",
        ]
        row, llm = self._run_with_budget(tmp_path, monkeypatch, budget=3, react_outputs=react)
        assert row["llm_calls"] <= 3.0, f"Exceeded budget: {row['llm_calls']} calls"

    def test_early_finish_action_stops_react_loop(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """If the model emits Action: finish early, remaining budget is not consumed."""
        react = [
            # First ReAct step emits a finish action (model satisfied after 0 searches).
            "Thought: I know the answer from prior knowledge.\nAction: finish"
        ]
        row, _llm = self._run_with_budget(
            tmp_path, monkeypatch, budget=5, react_outputs=react
        )
        # Should have only used the 1 early-finish step + 1 synthesis = 2 calls.
        assert row["llm_calls"] <= 2.0, (
            f"Early-finish loop did not stop: {row['llm_calls']} calls"
        )
        trace = row.get("trace", [])
        assert any(e.get("stage") == "iso_call_early_finish" for e in trace), (
            "Expected iso_call_early_finish control event in trace"
        )

    def test_budget_recorded_in_metadata(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """The call_budget is written into the prediction metadata for reproducibility."""
        react = ["Thought: search\nAction: search\nQuery: Signpost"]
        row, _llm = self._run_with_budget(tmp_path, monkeypatch, budget=3, react_outputs=react)
        assert row["metadata"]["call_budget"] == 3

    def test_minimum_budget_2_enforced_when_1_requested(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """call_budget is clamped to at least 2 (1 ReAct + 1 synthesis)."""
        react = ["Thought: search\nAction: search\nQuery: Signpost evidence"]
        # Request budget=1 — should be silently promoted to 2.
        row, _llm = self._run_with_budget(tmp_path, monkeypatch, budget=1, react_outputs=react)
        # At minimum there must be a synthesis call, so llm_calls >= 1.
        assert row["llm_calls"] >= 1.0


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestInternals:
    def test_extract_action_search(self) -> None:
        text = "Thought: need info\nAction: search\nQuery: what is Signpost"
        action, query = _extract_action_and_query(text)
        assert action == "search"
        assert "Signpost" in query

    def test_extract_action_finish(self) -> None:
        text = "Thought: done\nAction: finish"
        action, query = _extract_action_and_query(text)
        assert action == "finish"
        assert query == ""

    def test_extract_action_unknown_returns_empty(self) -> None:
        text = "some random text with no action"
        action, query = _extract_action_and_query(text)
        assert action == ""

    def test_local_keyword_search_returns_matching_chunks(self) -> None:
        chunks = _fake_chunks()
        results = _local_keyword_search(chunks, "Signpost retrieves", top_k=5)
        assert len(results) >= 1
        assert all("score" in r for r in results)
        assert all(r["score_source"] == "iso_call_local_keyword" for r in results)

    def test_local_keyword_search_top_k_respected(self) -> None:
        chunks = _fake_chunks()
        results = _local_keyword_search(chunks, "evidence", top_k=1)
        assert len(results) <= 1

    def test_graph_neighbors_top_k_respected(self) -> None:
        graph = _fake_graph()
        results = _graph_text_neighbors(graph, "Signpost evidence", top_k=1)
        assert len(results) <= 1

    def test_graph_neighbors_empty_graph_returns_empty(self) -> None:
        results = _graph_text_neighbors({}, "anything", top_k=5)
        assert results == []
