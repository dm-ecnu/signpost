from pathlib import Path
import json

from signpost.parsing.parse_documents import parse_documents
from signpost.chunking.run import run_chunking
from signpost.chunking.validate import validate_chunks
from signpost.parsing.io import write_jsonl


def test_chunk_mini_documents(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    documents_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    trees_path = tmp_path / "document_trees.jsonl"

    parse_documents(root / "samples/mini/raw_corpus.jsonl", documents_path)
    count = run_chunking(documents_path, chunks_path, trees_path, max_tokens=40, overlap_tokens=5)
    summary = validate_chunks(chunks_path)

    assert count >= 1
    assert summary["documents"] == 1
    assert trees_path.exists()


def test_chunking_splits_oversized_single_line(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    trees_path = tmp_path / "document_trees.jsonl"
    long_text = " ".join(f"word{i}" for i in range(60))
    write_jsonl(
        documents_path,
        [
            {
                "doc_id": "doc1",
                "file_name": "long.txt",
                "text": f"第一章 长行\n{long_text}",
                "lines": [
                    {"line_no": 1, "text": "第一章 长行"},
                    {"line_no": 2, "text": long_text},
                ],
                "metadata": {"dataset": "test"},
            }
        ],
    )

    count = run_chunking(documents_path, chunks_path, trees_path, max_tokens=20, overlap_tokens=2)
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]

    assert count > 1
    assert all(chunk["metadata"]["token_count"] <= 35 for chunk in chunks)
    assert any(chunk["metadata"]["merge"] == "split_long_line" for chunk in chunks)
