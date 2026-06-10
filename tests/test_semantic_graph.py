from pathlib import Path

from signpost.chunking.run import run_chunking
from signpost.graph.validate import validate_graph
from signpost.indexing.semantic_graph import build_semantic_graph_file
from signpost.parsing.parse_documents import parse_documents


def test_build_semantic_graph_from_mini_chunks(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    documents = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    trees = tmp_path / "trees.jsonl"
    output = tmp_path / "graph.semantic.json"

    parse_documents(root / "samples/mini/raw_corpus.jsonl", documents)
    run_chunking(documents, chunks, trees, max_tokens=40, overlap_tokens=5)
    result = build_semantic_graph_file(
        chunks,
        output,
        namespace="mini",
        extractor_name="deterministic",
    )
    summary = validate_graph(output)

    assert result["chunks"] == 2
    assert summary["entity_nodes"] > 0
    assert summary["edges"] > 0
