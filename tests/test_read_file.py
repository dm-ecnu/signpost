from pathlib import Path

from signpost.parsing.io import write_jsonl
from signpost.retrieval.read_file import format_file_view, parse_locate, read_file_window, read_locate


def _documents(path: Path) -> Path:
    write_jsonl(
        path,
        [
            {
                "doc_id": "doc1",
                "file_name": "a.txt",
                "lines": [
                    {"line_no": 1, "text": "one"},
                    {"line_no": 2, "text": "two"},
                    {"line_no": 4, "text": "four"},
                ],
            }
        ],
    )
    return path


def test_parse_locate() -> None:
    assert parse_locate("a.txt:L10-L12") == ("a.txt", 10, 12)


def test_read_file_window_with_expansion(tmp_path: Path) -> None:
    result = read_file_window(documents_path=_documents(tmp_path / "documents.jsonl"), file_name="a.txt", start_line=2, end_line=2, before=1, after=2)
    assert [row["line_no"] for row in result["lines"]] == [1, 2, 4]
    assert "a.txt:L1-L4" in result["file_content_view"]
    assert "     2 | two" in result["file_content_view"]


def test_read_locate(tmp_path: Path) -> None:
    result = read_locate("a.txt:L4-L4", documents_path=_documents(tmp_path / "documents.jsonl"))
    assert result["lines"] == [{"line_no": 4, "text": "four"}]


def test_format_empty_file_view() -> None:
    assert format_file_view("a.txt", []) == "=== a.txt (0 lines) ==="
