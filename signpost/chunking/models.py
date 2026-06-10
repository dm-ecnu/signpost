from __future__ import annotations

"""F4 data models for chapter recognition, document trees, and chunks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Header:
    """A detected section title with a source line location."""

    title: str
    level: int
    line_start: int
    line_end: int
    content_start: int = 0
    content_end: int = 0


@dataclass
class TreeNode:
    """One node in the document chapter tree."""

    title: str
    level: int
    start_line: int
    end_line: int
    section_path: list[str] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "section_path": self.section_path,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class Chunk:
    """A line-located chunk emitted by F4."""

    chunk_id: str
    doc_id: str
    file_name: str
    content: str
    start_line: int
    end_line: int
    section_path: list[str]
    prev_chunk_id: str | None
    next_chunk_id: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "section_path": self.section_path,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "metadata": self.metadata,
        }

