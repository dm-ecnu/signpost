from pathlib import Path

from signpost.chunking.run import run_chunking
from signpost.graph.sequence import expand_sequence_context
from signpost.graph.validate import validate_graph
from signpost.indexing.sequence_graph import build_sequence_graph_file
from signpost.parsing.parse_documents import parse_documents


def test_build_sequence_graph_and_expand_context(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    documents = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    trees = tmp_path / "document_trees.jsonl"
    output = tmp_path / "graph.sequence.json"

    parse_documents(root / "samples/mini/raw_corpus.jsonl", documents)
    run_chunking(documents, chunks, trees, max_tokens=40, overlap_tokens=5)
    result = build_sequence_graph_file(chunks, output, namespace="mini")
    summary = validate_graph(output)

    assert result["chunks"] == 2
    assert result["sequence_edges"] == 2
    assert summary["edges"] == 2

    import json

    graph = json.loads(output.read_text(encoding="utf-8"))
    rows = expand_sequence_context(graph, ["mini_doc_001_c00001"], before=1, after=1)
    assert [row["chunk_id"] for row in rows] == ["mini_doc_001_c00000", "mini_doc_001_c00001"]
    assert [row["hop_from_seed"] for row in rows] == [-1, 0]
