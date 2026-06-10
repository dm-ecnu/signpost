from pathlib import Path

from signpost.parsing.parse_documents import parse_documents
from signpost.parsing.validate_documents import validate_documents


def test_parse_mini_documents(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "documents.jsonl"

    count = parse_documents(root / "samples/mini/raw_corpus.jsonl", output)
    summary = validate_documents(output)

    assert count == 1
    assert summary["documents"] == 1
    assert summary["lines"] == 5
    assert summary["placeholders"] == 1

