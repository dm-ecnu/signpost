from __future__ import annotations

"""F4 document tree construction.

This is the stack algorithm described in the thesis: the stack stores the
nearest possible parent chain; a new header pops nodes until it finds a shallower
parent, then becomes that parent's child.
"""

from typing import Any

from signpost.chunking.models import Header, TreeNode


def build_document_tree(document: dict[str, Any], headers: list[Header]) -> TreeNode:
    total_lines = _last_line_no(document)
    root = TreeNode(title="[ROOT]", level=0, start_line=1 if total_lines else 0, end_line=total_lines, section_path=[])
    if not headers:
        root.children.append(TreeNode(title=document.get("file_name") or document["doc_id"], level=1, start_line=1, end_line=total_lines, section_path=[document.get("file_name") or document["doc_id"]]))
        return root

    bounded_headers = _with_content_ranges(headers, total_lines)
    stack: list[TreeNode] = [root]
    for header in bounded_headers:
        node_path_parent = _path_for_parent(stack, header.level)
        node = TreeNode(
            title=header.title,
            level=header.level,
            start_line=header.content_start,
            end_line=header.content_end,
            section_path=node_path_parent + [header.title],
        )
        while len(stack) > 1 and stack[-1].level >= node.level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    return root


def tree_statistics(root: TreeNode) -> dict[str, int]:
    nodes = list(iter_nodes(root))
    return {
        "total_nodes": len(nodes),
        "max_level": max((node.level for node in nodes), default=0),
        "leaf_nodes": sum(1 for node in nodes if not node.children),
    }


def iter_nodes(root: TreeNode) -> list[TreeNode]:
    result = [root]
    for child in root.children:
        result.extend(iter_nodes(child))
    return result


def _with_content_ranges(headers: list[Header], total_lines: int) -> list[Header]:
    result: list[Header] = []
    ordered = sorted(headers, key=lambda item: (item.line_start, item.level))
    for idx, header in enumerate(ordered):
        end = total_lines
        for later in ordered[idx + 1 :]:
            if later.level <= header.level and later.line_start > header.line_start:
                end = later.line_start - 1
                break
        result.append(
            Header(
                title=header.title,
                level=header.level,
                line_start=header.line_start,
                line_end=header.line_end,
                content_start=header.line_start,
                content_end=max(header.line_start, end),
            )
        )
    return result


def _path_for_parent(stack: list[TreeNode], level: int) -> list[str]:
    path: list[str] = []
    for node in stack[1:]:
        if node.level < level:
            path.append(node.title)
    return path


def _last_line_no(document: dict[str, Any]) -> int:
    lines = document.get("lines") or []
    if not lines:
        return 0
    return max(int(line.get("line_no", 0)) for line in lines)

