"""Offline tests for the in-repo silver-evidence builder (mocked LLM)."""

from __future__ import annotations

import json

import pytest

from signpost.benchmark.final_metrics import EvidenceItem, silver_hits_for_item
from signpost.evaluation.silver_builder import (
    _extract_json,
    build_for_question,
    build_silver_chunks,
    build_target_units,
    lexical_candidates,
)

CHUNKS = [
    {"chunk_id": "c1", "file_name": "manual.md", "start_line": 10, "end_line": 20,
     "content": "The recommended pesticide dose is 2 ml per liter of water."},
    {"chunk_id": "c2", "file_name": "manual.md", "start_line": 21, "end_line": 30,
     "content": "After application, the safe re-entry interval is 48 hours."},
    {"chunk_id": "c3", "file_name": "other.md", "start_line": 1, "end_line": 5,
     "content": "Completely unrelated text about database indexing."},
]


class TestLexicalCandidates:
    def test_ranks_overlapping_chunks_first(self):
        out = lexical_candidates("safe re-entry interval after pesticide dose", CHUNKS, top_k=2)
        ids = [chunk["chunk_id"] for chunk in out]
        assert "c2" in ids and "c3" not in ids

    def test_zero_overlap_excluded(self):
        out = lexical_candidates("quantum entanglement", CHUNKS)
        assert out == []

    def test_top_k_respected(self):
        out = lexical_candidates("pesticide re-entry interval dose application", CHUNKS, top_k=1)
        assert len(out) == 1


class TestExtractJson:
    def test_plain_array(self):
        assert _extract_json('["a", "b"]') == ["a", "b"]

    def test_fenced_object_with_prose(self):
        text = 'Sure! Here you go:\n```json\n{"c1": ["q-u0"]}\n```\nHope that helps.'
        assert _extract_json(text) == {"c1": ["q-u0"]}

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")


class TestBuildTargetUnits:
    def test_assigns_sequential_unit_ids(self):
        chat = lambda messages: json.dumps(["the interval is 48 hours", "dose is 2 ml/l"])
        units = build_target_units(chat, "q7", "what is the interval?", "48 hours at 2 ml/l")
        assert [unit["unit_id"] for unit in units] == ["q7-u0", "q7-u1"]
        assert all(unit["text"] for unit in units)

    def test_non_array_reply_raises(self):
        with pytest.raises(ValueError):
            build_target_units(lambda m: '{"not": "an array"}', "q1", "q", "a")


class TestBuildSilverChunks:
    UNITS = [{"unit_id": "q1-u0", "text": "re-entry interval is 48 hours"}]

    def test_keeps_supporting_chunks_only(self):
        chat = lambda messages: json.dumps({"c1": [], "c2": ["q1-u0"], "c3": []})
        silver = build_silver_chunks(chat, self.UNITS, CHUNKS)
        assert len(silver) == 1
        row = silver[0]
        assert row["chunk_id"] == "c2"
        assert row["file_name"] == "manual.md"
        assert (row["start_line"], row["end_line"]) == (21, 30)
        assert row["supports"] == ["q1-u0"]

    def test_hallucinated_ids_dropped(self):
        chat = lambda messages: json.dumps({"c2": ["q1-u0", "q1-u99"], "c99": ["q1-u0"]})
        silver = build_silver_chunks(chat, self.UNITS, CHUNKS)
        assert len(silver) == 1
        assert silver[0]["supports"] == ["q1-u0"]

    def test_empty_inputs(self):
        assert build_silver_chunks(lambda m: "{}", [], CHUNKS) == []
        assert build_silver_chunks(lambda m: "{}", self.UNITS, []) == []


class TestEndToEndSchema:
    def _chat(self, messages):
        system = messages[0]["content"]
        if "decompose" in system.lower():
            return json.dumps(["the safe re-entry interval is 48 hours"])
        return json.dumps({"c1": [], "c2": ["q1-u0"], "c3": []})

    def test_rows_match_final_metrics_consumers(self):
        row = {"question_id": "q1", "question": "what is the safe re-entry interval?", "answer": "48 hours"}
        units_row, silver_row = build_for_question(self._chat, row, CHUNKS, top_k=3)

        assert units_row["question_id"] == "q1"
        assert units_row["target_units"][0]["unit_id"] == "q1-u0"

        # chunk-id match path used by final_metrics
        hit = silver_hits_for_item(EvidenceItem(kind="chunk", chunk_id="c2"), silver_row)
        assert hit == {"c2"}
        # span-overlap match path (same file, overlapping lines)
        hit = silver_hits_for_item(
            EvidenceItem(kind="span", file_name="manual.md", start_line=25, end_line=27), silver_row
        )
        assert hit == {"c2"}
        # non-overlapping span misses
        hit = silver_hits_for_item(
            EvidenceItem(kind="span", file_name="manual.md", start_line=100, end_line=110), silver_row
        )
        assert hit == set()

    def test_list_answer_joined(self):
        row = {"question_id": "q1", "question": "interval?", "answers": ["48 hours", "two days"]}
        units_row, _ = build_for_question(self._chat, row, CHUNKS)
        assert units_row["target_units"]
