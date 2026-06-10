from pathlib import Path

from signpost.chunking.run import run_chunking
from signpost.graph.merge import merge_graph_files
from signpost.graph.validate import validate_graph
from signpost.indexing.semantic_graph import build_semantic_graph_file
from signpost.indexing.sequence_graph import build_sequence_graph_file
from signpost.indexing.structure_graph import build_structure_graph_file
from signpost.parsing.parse_documents import parse_documents


def test_merge_unified_graph_from_three_views(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    documents = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    trees = tmp_path / "document_trees.jsonl"
    semantic = tmp_path / "graph.semantic.json"
    structure = tmp_path / "graph.structure.json"
    sequence = tmp_path / "graph.sequence.json"
    unified = tmp_path / "graph.unified.json"

    parse_documents(root / "samples/mini/raw_corpus.jsonl", documents)
    run_chunking(documents, chunks, trees, max_tokens=40, overlap_tokens=5)
    build_semantic_graph_file(chunks, semantic, namespace="mini", extractor_name="deterministic")
    build_structure_graph_file(chunks, trees, structure, namespace="mini", summarizer_name="deterministic")
    build_sequence_graph_file(chunks, sequence, namespace="mini")

    result = merge_graph_files(
        semantic_path=str(semantic),
        structure_path=str(structure),
        sequence_path=str(sequence),
        output_path=str(unified),
        namespace="mini",
    )
    summary = validate_graph(unified)

    assert result["chunk_nodes"] == 2
    assert result["summary_nodes"] >= 1
    assert result["entity_nodes"] >= 1
    assert result["structure_edges"] >= 1
    assert result["semantic_edges"] >= 1
    assert result["sequence_edges"] == 2
    assert result["source_edges"] >= 1
    assert summary["nodes"] == result["nodes"]
