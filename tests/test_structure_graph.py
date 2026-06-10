from pathlib import Path

from signpost.chunking.run import run_chunking
from signpost.graph.validate import validate_graph
from signpost.indexing.structure_graph import build_structure_graph_file
from signpost.parsing.parse_documents import parse_documents


def test_build_structure_graph_from_mini(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    documents = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    trees = tmp_path / "document_trees.jsonl"
    output = tmp_path / "graph.structure.json"

    parse_documents(root / "samples/mini/raw_corpus.jsonl", documents)
    run_chunking(documents, chunks, trees, max_tokens=40, overlap_tokens=5)
    result = build_structure_graph_file(chunks, trees, output, namespace="mini", summarizer_name="deterministic")
    summary = validate_graph(output)

    assert result["chunks"] == 2
    assert result["raptor_nodes"] >= 1
    assert summary["edges"] >= 1

