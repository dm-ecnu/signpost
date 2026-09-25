#!/usr/bin/env python3
"""E4: incremental maintenance of the navigation view under corpus edits.

ZERO model calls, ZERO Elasticsearch. Measures what it costs to keep sigma(o)
fresh after a batch of document edits, against a full rebuild of the view.

Model (Section 4.3): an edit touching a set of graph vertices Delta-V dirties
    D = Delta-V  union  N(Delta-V),
where N is the adjacency that sigma(o) reads -- an object's view can only change
if the object itself changed or one of the neighbors it points at did. We take
Delta-V to be every node produced from the edited source documents, expand it
once through the same adjacency the offline builders traverse, and then
rematerialize sigma(o) for exactly the objects in D, timing it.

The reported ratio is cost(D) / cost(all objects) at identical code path: both
numbers are wall-clock over the same builders with one shared GraphIndex, so the
ratio is not sensitive to machine speed.

Usage (H200, project root on PYTHONPATH):
  python3 e4_incremental.py --graph .../graph.unified.json --dataset agriculture --out ./out
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from signpost.retrieval.offline_signpost import (
    GraphIndex,
    _chunk_signpost,
    _summary_signpost,
    _entity_signpost,
    _relation_signpost,
)

CHUNK_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.10]
SEED = 20260919


def doc_of(node: dict[str, Any]) -> str | None:
    for key in ("doc_id", "document_id", "file_name", "source_doc_id"):
        v = node.get(key)
        if v:
            return str(v)
    meta = node.get("metadata")
    if isinstance(meta, dict):
        for key in ("doc_id", "document_id", "file_name"):
            v = meta.get(key)
            if v:
                return str(v)
    return None


def node_id(node: dict[str, Any]) -> str | None:
    for key in ("node_id", "id", "chunk_id"):
        v = node.get(key)
        if v:
            return str(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    t0 = time.time()
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    index = GraphIndex(graph)
    t_index = time.time() - t0
    print(f"[{args.dataset}] graph loaded + indexed in {t_index:.1f}s", flush=True)

    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]

    # retrievable objects, exactly as the exposure/E6 scripts define them
    objects: list[tuple[str, dict[str, Any], str | None]] = []
    for n in nodes:
        if n.get("node_type") in {"chunk", "summary", "entity"}:
            objects.append((n["node_type"], n, node_id(n)))
    for e in edges:
        if e.get("edge_type") == "semantic":
            objects.append(("relation", e, node_id(e)))
    n_obj = len(objects)

    # undirected adjacency over ALL edges -- this is the N(.) of the dirty set
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        s, t = e.get("source") or e.get("src") or e.get("from"), e.get("target") or e.get("dst") or e.get("to")
        if s and t:
            adj[str(s)].add(str(t))
            adj[str(t)].add(str(s))

    # object id -> position, and which source document each object belongs to
    pos_of: dict[str, list[int]] = defaultdict(list)
    doc_objects: dict[str, list[int]] = defaultdict(list)
    obj_endpoints: list[set[str]] = []
    for i, (rtype, ref, oid) in enumerate(objects):
        if oid:
            pos_of[oid].append(i)
        d = doc_of(ref)
        if d:
            doc_objects[d].append(i)
        if rtype == "relation":
            eps = {str(x) for x in (ref.get("source") or ref.get("src"), ref.get("target") or ref.get("dst")) if x}
        else:
            eps = {oid} if oid else set()
        obj_endpoints.append(eps)

    # vertex -> objects whose view can change when that vertex changes
    touched_by: dict[str, set[int]] = defaultdict(set)
    for i, eps in enumerate(obj_endpoints):
        for v in eps:
            touched_by[v].add(i)

    def build(rtype: str, ref: dict[str, Any]) -> dict[str, Any]:
        if rtype == "chunk":
            return _chunk_signpost(index, ref)
        if rtype == "summary":
            return _summary_signpost(index, ref)
        if rtype == "entity":
            return _entity_signpost(index, ref)
        return _relation_signpost(index, ref)

    # ---- full rebuild of the view, for the denominator ----
    t0 = time.time()
    for rtype, ref, _ in objects:
        build(rtype, ref)
    t_full = time.time() - t0
    print(f"  full view rebuild: {n_obj} objects in {t_full:.1f}s "
          f"({t_full / n_obj * 1000:.3f} ms/object)", flush=True)

    docs = sorted(doc_objects)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "graph": args.graph,
        "objects": n_obj,
        "documents": len(docs),
        "index_seconds": round(t_index, 2),
        "full_rebuild_seconds": round(t_full, 2),
        "ms_per_object": round(t_full / n_obj * 1000, 4),
        "edits": [],
    }

    rng = random.Random(SEED)
    rng_ctrl = random.Random(SEED + 1)
    chunk_ids = [node_id(n) for n in nodes if n.get("node_type") == "chunk" and node_id(n)]
    report["chunks"] = len(chunk_ids)

    def run_edit(kind: str, frac: float, seeds: list[str]) -> None:
        """seeds = the vertices the edit touches directly (Delta-V).

        Also times a uniformly random subset of the SAME cardinality, built in the
        same (set-iteration) order. That control separates two explanations for a
        time share above the cardinality share: the dirty set being biased toward
        expensive high-degree objects, versus a memory-locality artifact of not
        walking the object list in order. Only the first would survive the control.
        """
        delta_v = set(seeds)
        dirty_vertices = set(delta_v)
        for v in delta_v:
            dirty_vertices |= adj.get(v, set())
        dirty_objects: set[int] = set()
        for v in dirty_vertices:
            dirty_objects |= touched_by.get(v, set())

        t0 = time.time()
        for i in dirty_objects:
            rtype, ref, _ = objects[i]
            build(rtype, ref)
        t_inc = time.time() - t0

        ctrl = set(rng_ctrl.sample(range(n_obj), len(dirty_objects)))
        t0 = time.time()
        for i in ctrl:
            rtype, ref, _ = objects[i]
            build(rtype, ref)
        t_ctrl = time.time() - t0

        row = {
            "granularity": kind,
            "edit_fraction": frac,
            "units_edited": len(seeds),
            "delta_v": len(delta_v),
            "dirty_vertices": len(dirty_vertices),
            "dirty_objects": len(dirty_objects),
            "dirty_object_share": round(len(dirty_objects) / n_obj, 4),
            "seconds": round(t_inc, 4),
            "share_of_full": round(t_inc / t_full, 4),
            "control_seconds": round(t_ctrl, 4),
            "control_share_of_full": round(t_ctrl / t_full, 4),
            "hub_bias": round(t_inc / t_ctrl, 2) if t_ctrl > 0 else None,
            "amplification": round((len(dirty_objects) / n_obj) / frac, 2) if frac else None,
        }
        report["edits"].append(row)
        print(f"  {kind:5s} {frac:7.3%} (k={len(seeds):6d}): dirty {len(dirty_objects):7d}"
              f" ({row['dirty_object_share']:6.2%} of objects), {t_inc:7.2f}s"
              f" = {row['share_of_full']:6.2%} of a full rebuild"
              f"  [rand ctrl {row['control_share_of_full']:6.2%}, hub-bias {row['hub_bias']}x]"
              f"  amp={row['amplification']}", flush=True)

    # primary: chunk-level edits -- Delta-V is the set of revised chunks
    for frac in CHUNK_FRACTIONS:
        k = max(1, int(round(len(chunk_ids) * frac)))
        run_edit("chunk", frac, rng.sample(chunk_ids, k))

    # secondary: whole-document edits, to show the amplification of a coarse edit
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(round(len(docs) * frac)))
        seeds: set[str] = set()
        for d in rng.sample(docs, k):
            for i in doc_objects[d]:
                seeds |= obj_endpoints[i]
        run_edit("doc", frac, sorted(seeds))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e4_{args.dataset}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / f'e4_{args.dataset}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
