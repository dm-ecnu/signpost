from __future__ import annotations

"""Expand retrieved chunks with F8 sequential context."""

import argparse
import json

from signpost.config.context import resolve_project_path
from signpost.graph.sequence import expand_sequence_context


def expand_sequence_context_file(
    graph_path: str,
    chunk_ids: list[str],
    *,
    before: int = 1,
    after: int = 1,
) -> list[dict[str, object]]:
    graph = json.loads(resolve_project_path(graph_path).read_text(encoding="utf-8"))
    return expand_sequence_context(graph, chunk_ids, before=before, after=after)


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand chunk hits with sequential context")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--chunk-id", action="append", required=True)
    parser.add_argument("--before", type=int, default=1)
    parser.add_argument("--after", type=int, default=1)
    args = parser.parse_args()

    rows = expand_sequence_context_file(args.graph, args.chunk_id, before=args.before, after=args.after)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
