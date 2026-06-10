from __future__ import annotations

"""F4 tree-aware chunk generation.

The chunker follows the thesis two-stage scheme:
1. If a subtree fits the token budget, merge it into one chunk.
2. If a node is still oversized, split it on line boundaries with overlap.
"""

from typing import Any

from signpost.chunking.models import Chunk, TreeNode
from signpost.chunking.tokenizer import count_tokens
from signpost.chunking.tree import build_document_tree
from signpost.chunking.headers import recognize_headers


def chunk_document(
    document: dict[str, Any],
    *,
    max_tokens: int = 1200,
    overlap_tokens: int = 100,
    use_llm: bool = False,
) -> tuple[list[Chunk], dict[str, Any]]:
    headers = recognize_headers(document, use_llm=use_llm)
    tree = build_document_tree(document, headers)
    line_map = {int(item["line_no"]): str(item["text"]) for item in document.get("lines", [])}
    chunks = _chunks_from_tree(document, tree, line_map, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunks = _link_chunks(chunks)
    tree_payload = {
        "doc_id": document["doc_id"],
        "file_name": document.get("file_name"),
        "headers": _headers_from_tree(tree),
        "tree": tree.to_dict(),
    }
    return chunks, tree_payload


def _chunks_from_tree(
    document: dict[str, Any],
    root: TreeNode,
    line_map: dict[int, str],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for child in root.children:
        chunks.extend(_traverse_node(document, child, line_map, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    if not chunks and root.end_line >= root.start_line:
        chunks.extend(_split_range(document, root, line_map, root.section_path or [document.get("file_name", document["doc_id"])], max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return _assign_chunk_ids(document, chunks)


def _traverse_node(
    document: dict[str, Any],
    node: TreeNode,
    line_map: dict[int, str],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    subtree_text = _range_text(line_map, node.start_line, _subtree_end(node))
    if subtree_text and count_tokens(subtree_text) <= max_tokens:
        return [_make_chunk(document, node.section_path, node.start_line, _subtree_end(node), subtree_text, {"merge": "subtree"})]

    node_text = _range_text(line_map, node.start_line, node.end_line)
    chunks: list[Chunk] = []
    if node_text:
        if count_tokens(node_text) > max_tokens:
            chunks.extend(_split_range(document, node, line_map, node.section_path, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
        else:
            chunks.append(_make_chunk(document, node.section_path, node.start_line, node.end_line, node_text, {"merge": "node"}))
    for child in node.children:
        chunks.extend(_traverse_node(document, child, line_map, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return chunks


def _split_range(
    document: dict[str, Any],
    node: TreeNode,
    line_map: dict[int, str],
    section_path: list[str],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0
    for line_no in range(node.start_line, node.end_line + 1):
        if line_no not in line_map:
            continue
        text = line_map[line_no]
        line_tokens = max(1, count_tokens(text))
        if line_tokens > max_tokens:
            if current:
                chunks.append(_make_chunk(document, section_path, current[0][0], current[-1][0], "\n".join(line for _, line in current), {"merge": "split"}))
                current, current_tokens = _overlap_tail(current, overlap_tokens)
            for segment in _split_long_line(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
                chunks.append(_make_chunk(document, section_path, line_no, line_no, segment, {"merge": "split_long_line"}))
            current = []
            current_tokens = 0
            continue
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append(_make_chunk(document, section_path, current[0][0], current[-1][0], "\n".join(line for _, line in current), {"merge": "split"}))
            current, current_tokens = _overlap_tail(current, overlap_tokens)
        current.append((line_no, text))
        current_tokens += line_tokens
    if current:
        chunks.append(_make_chunk(document, section_path, current[0][0], current[-1][0], "\n".join(line for _, line in current), {"merge": "split"}))
    return chunks


def _overlap_tail(lines: list[tuple[int, str]], overlap_tokens: int) -> tuple[list[tuple[int, str]], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept: list[tuple[int, str]] = []
    total = 0
    for item in reversed(lines):
        line_tokens = max(1, count_tokens(item[1]))
        if total + line_tokens > overlap_tokens:
            break
        kept.insert(0, item)
        total += line_tokens
    return kept, total


def _split_long_line(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a single oversized logical line into token-budgeted text windows."""

    words = text.split()
    if not words:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = max(1, count_tokens(word))
        if current and current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = _word_overlap_tail(current, overlap_tokens)
        if word_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_oversized_word(word, max_tokens=max_tokens))
            continue
        current.append(word)
        current_tokens += word_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _word_overlap_tail(words: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept: list[str] = []
    total = 0
    for word in reversed(words):
        word_tokens = max(1, count_tokens(word))
        if kept and total + word_tokens > overlap_tokens:
            break
        kept.insert(0, word)
        total += word_tokens
    return kept, total


def _split_oversized_word(word: str, *, max_tokens: int) -> list[str]:
    # Rare fallback for no-whitespace text.  The local tokenizer counts CJK and
    # punctuation per character, so character windows are conservative here.
    size = max(1, max_tokens)
    return [word[start : start + size] for start in range(0, len(word), size)]


def _assign_chunk_ids(document: dict[str, Any], chunks: list[Chunk]) -> list[Chunk]:
    assigned: list[Chunk] = []
    for idx, chunk in enumerate(chunks):
        assigned.append(
            Chunk(
                chunk_id=f"{document['doc_id']}_c{idx:05d}",
                doc_id=chunk.doc_id,
                file_name=chunk.file_name,
                content=chunk.content,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                section_path=chunk.section_path,
                prev_chunk_id=None,
                next_chunk_id=None,
                metadata={**chunk.metadata, "chunk_index": idx, "token_count": count_tokens(chunk.content)},
            )
        )
    return assigned


def _link_chunks(chunks: list[Chunk]) -> list[Chunk]:
    linked: list[Chunk] = []
    for idx, chunk in enumerate(chunks):
        linked.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                file_name=chunk.file_name,
                content=chunk.content,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                section_path=chunk.section_path,
                prev_chunk_id=chunks[idx - 1].chunk_id if idx > 0 else None,
                next_chunk_id=chunks[idx + 1].chunk_id if idx + 1 < len(chunks) else None,
                metadata=chunk.metadata,
            )
        )
    return linked


def _make_chunk(document: dict[str, Any], section_path: list[str], start_line: int, end_line: int, content: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id="",
        doc_id=document["doc_id"],
        file_name=document.get("file_name", ""),
        content=_content_with_path(section_path, content),
        start_line=start_line,
        end_line=end_line,
        section_path=section_path,
        prev_chunk_id=None,
        next_chunk_id=None,
        metadata=metadata,
    )


def _content_with_path(section_path: list[str], content: str) -> str:
    if not section_path:
        return content
    return " > ".join(section_path) + "\n\n[CONTENT]\n\n" + content


def _range_text(line_map: dict[int, str], start_line: int, end_line: int) -> str:
    return "\n".join(line_map[line_no] for line_no in range(start_line, end_line + 1) if line_no in line_map)


def _subtree_end(node: TreeNode) -> int:
    end = node.end_line
    for child in node.children:
        end = max(end, _subtree_end(child))
    return end


def _headers_from_tree(root: TreeNode) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    for node in root.children:
        headers.extend(_headers_from_node(node))
    return headers


def _headers_from_node(node: TreeNode) -> list[dict[str, Any]]:
    rows = [
        {
            "title": node.title,
            "level": node.level,
            "line_start": node.start_line,
            "line_end": node.start_line,
            "content_start": node.start_line,
            "content_end": node.end_line,
        }
    ]
    for child in node.children:
        rows.extend(_headers_from_node(child))
    return rows
