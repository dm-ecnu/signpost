#!/usr/bin/env python3
"""E3: per-entry-class follow rate and unique evidence contribution.

ZERO model calls, ZERO Elasticsearch. Answers reviewer R2-W2 / R2-detailed:
"which entry classes are actually followed, and which contribute evidence no
other class could have supplied?"

The frozen action logs record every read_file locate but not the entry it came
from. We recover the attribution instead of re-running: sigma(o) is deterministic
given the graph (Proposition 1), so we rebuild it for the objects the query was
served and ask, for each locate the agent actually read, which classes exposed it.

Two attribution surfaces per class x and served object o:
  direct  -- the locator appears verbatim inside a class-x entry payload of sigma(o)
  onehop  -- the locator is reachable by following one class-x entry to its target
             object and taking that object's provenance locators
`onehop` is the honest model of "class x made this evidence reachable"; `direct`
is the stricter one. We report both, plus UNIQUE attribution: reads that only one
class could have supplied, which is the quantity the review actually asked for.

Usage (H200, project root on PYTHONPATH):
  python3 e3_follow_rate.py --dataset agriculture --root /data/srl/signpost_re_v2 --out ./out
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from signpost.retrieval.offline_signpost import (
    GraphIndex,
    _chunk_signpost,
    _summary_signpost,
    _entity_signpost,
    _relation_signpost,
    _edge_id,
)

LOC_RE = re.compile(r"^(?P<file>.+):L(?P<start>\d+)-L(?P<end>\d+)$")


def parse_loc(loc: str) -> tuple[str, int, int] | None:
    m = LOC_RE.match(str(loc))
    if not m:
        return None
    return m.group("file"), int(m.group("start")), int(m.group("end"))


class SpanSet:
    """Line spans per file, queried by overlap.

    The agent copies a locator out of an entry but the reader may widen or shift
    it by a line or two, so exact string equality undercounts genuine follows.
    Overlap on the same file is the faithful test of "this entry is the one the
    read came from".
    """

    def __init__(self) -> None:
        self.by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add_many(self, locs) -> None:
        for loc in locs:
            p = parse_loc(loc)
            if p:
                self.by_file[p[0]].append((p[1], p[2]))

    def hits(self, loc: str) -> bool:
        p = parse_loc(loc)
        if not p:
            return False
        f, s, e = p
        return any(s <= b and a <= e for a, b in self.by_file.get(f, ()))

    def __ior__(self, other: "SpanSet") -> "SpanSet":
        for f, spans in other.by_file.items():
            self.by_file[f].extend(spans)
        return self


CLASSES = ["zoom", "read", "jump", "verify"]  # C_v, C_h, C_s, C_p


def norm_id(raw: str) -> list[str]:
    """Candidate graph node ids for an id as it appears in retrieved_chunks."""
    raw = str(raw)
    if ":" in raw:
        return [raw]
    return [raw, f"chunk:{raw}"]


def locates_of_node(node: dict[str, Any] | None) -> set[str]:
    """Readable spans carried by an object itself."""
    if not isinstance(node, dict):
        return set()
    out: set[str] = set()
    for key in ("locate",):
        v = node.get(key)
        if isinstance(v, str) and v:
            out.add(v)
    for key in ("source_locates",):
        v = node.get(key)
        if isinstance(v, list):
            out |= {str(x) for x in v if x}
    prov = node.get("provenance")
    if isinstance(prov, dict):
        out |= locates_of_node(prov)
    fn, s, e = node.get("file_name"), node.get("start_line"), node.get("end_line")
    if fn and s is not None and e is not None:
        out.add(f"{fn}:L{s}-L{e}")
    return {x for x in out if x}


def entries_by_class(sk: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Split sigma(o) into the paper's four entry classes."""
    vert = sk.get("vertical") if isinstance(sk.get("vertical"), dict) else {}
    horiz = sk.get("horizontal") if isinstance(sk.get("horizontal"), dict) else {}
    sem = sk.get("semantic") if isinstance(sk.get("semantic"), dict) else {}
    prov = sk.get("provenance") if isinstance(sk.get("provenance"), dict) else {}

    def as_list(*vals) -> list[dict[str, Any]]:
        out = []
        for v in vals:
            if isinstance(v, dict):
                out.append(v)
            elif isinstance(v, list):
                out.extend(x for x in v if isinstance(x, dict))
        return out

    return {
        "zoom": as_list(vert.get("parent_summaries"), vert.get("nearest_parent_summary"),
                        vert.get("parent_summary"), vert.get("child_summaries"),
                        vert.get("child_chunks")),
        "read": as_list(horiz.get("previous_chunk"), horiz.get("next_chunk"),
                        horiz.get("neighbor_chunks")),
        "jump": as_list(sem.get("neighboring_entities"), sem.get("source_entity"),
                        sem.get("target_entity"), sem.get("neighboring_relations")),
        # verify is the object's own provenance: one entry, carrying its locators
        "verify": [prov] if prov else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="/data/srl/signpost_re_v2")
    ap.add_argument("--pred", default=None, help="override predictions jsonl")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    graph_path = root / "datasets/processed" / args.dataset / "graph.unified.json"
    pred_path = Path(args.pred) if args.pred else root / "outputs" / args.dataset / "predictions/signpost.full.jsonl"

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    index = GraphIndex(graph)

    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        t = n.get("node_type")
        if t not in {"chunk", "summary", "entity"}:
            continue
        for key in ("node_id", "id", "chunk_id"):
            v = n.get(key)
            if v:
                by_id[str(v)] = (t, n)
    for e in graph.get("edges", []):
        if isinstance(e, dict) and e.get("edge_type") == "semantic":
            by_id[_edge_id(e)] = ("relation", e)
            for key in ("node_id", "id", "edge_id"):
                v = e.get(key)
                if v:
                    by_id[str(v)] = ("relation", e)

    builders = {"chunk": _chunk_signpost, "summary": _summary_signpost,
                "entity": _entity_signpost, "relation": _relation_signpost}
    sketch_cache: dict[str, dict[str, Any]] = {}

    def sketch(oid: str) -> dict[str, Any] | None:
        if oid in sketch_cache:
            return sketch_cache[oid]
        hit = by_id.get(oid)
        if not hit:
            return None
        t, ref = hit
        sk = builders[t](index, ref)
        sketch_cache[oid] = sk
        return sk

    def resolve_targets(entry: dict[str, Any]) -> set[str]:
        """Locators reachable by following one entry to its target object."""
        out: set[str] = set()
        ids: list[str] = []
        for key in ("node_id", "chunk_id", "id"):
            v = entry.get(key)
            if v:
                ids.append(str(v))
        for key in ("source_chunk_ids",):
            v = entry.get(key)
            if isinstance(v, list):
                ids.extend(str(x) for x in v)
        for raw in ids:
            for cand in norm_id(raw):
                hit = by_id.get(cand)
                if hit:
                    out |= locates_of_node(hit[1])
                    break
        return out

    # ---- per query ----
    stats = {c: dict(surfaced=0, followed_direct=0, followed_onehop=0) for c in CLASSES}
    reads_total = 0
    reads_unattributed = 0
    reads_by_class_direct = {c: 0 for c in CLASSES}
    reads_by_class_onehop = {c: 0 for c in CLASSES}
    unique_onehop = {c: 0 for c in CLASSES}
    queries = 0
    served_hits, served_miss = 0, 0

    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        queries += 1

        served: list[str] = []
        for item in rec.get("retrieved_chunks", []):
            if not isinstance(item, dict) or item.get("score") is None:
                continue  # seeds = objects knowledge_search actually scored
            raw = item.get("chunk_id")
            if not raw:
                continue
            for cand in norm_id(raw):
                if cand in by_id:
                    served.append(cand)
                    served_hits += 1
                    break
            else:
                served_miss += 1

        # The locator the agent ASKED for is the one it copied out of an entry;
        # evidence_chunks stores the span the file reader resolved it to, which is
        # a different string. Attribution must use the request.
        reads = []
        for ev in rec.get("trace", []):
            if ev.get("event_type") == "tool_call" and ev.get("tool") == "read_file":
                loc = (ev.get("input") or {}).get("locate")
                if loc:
                    reads.append(loc)
        reads = list(dict.fromkeys(reads))

        # class -> line spans exposed this query
        direct: dict[str, SpanSet] = {c: SpanSet() for c in CLASSES}
        onehop: dict[str, SpanSet] = {c: SpanSet() for c in CLASSES}
        n_entries: dict[str, int] = {c: 0 for c in CLASSES}
        for oid in dict.fromkeys(served):
            sk = sketch(oid)
            if not sk:
                continue
            for cls, entries in entries_by_class(sk).items():
                n_entries[cls] += len(entries)
                for ent in entries:
                    direct[cls].add_many(locates_of_node(ent))
                    onehop[cls].add_many(resolve_targets(ent))
        for cls in CLASSES:
            onehop[cls] |= direct[cls]
            stats[cls]["surfaced"] += n_entries[cls]

        for loc in reads:
            reads_total += 1
            hit_d = [c for c in CLASSES if direct[c].hits(loc)]
            hit_o = [c for c in CLASSES if onehop[c].hits(loc)]
            for c in hit_d:
                reads_by_class_direct[c] += 1
                stats[c]["followed_direct"] += 1
            for c in hit_o:
                reads_by_class_onehop[c] += 1
                stats[c]["followed_onehop"] += 1
            if len(hit_o) == 1:
                unique_onehop[hit_o[0]] += 1
            if not hit_o:
                reads_unattributed += 1

    report = {
        "dataset": args.dataset,
        "queries": queries,
        "served_objects_resolved": served_hits,
        "served_objects_unresolved": served_miss,
        "reads_total": reads_total,
        "reads_unattributed": reads_unattributed,
        "per_class": {
            c: {
                "entries_surfaced": stats[c]["surfaced"],
                "reads_direct": reads_by_class_direct[c],
                "reads_onehop": reads_by_class_onehop[c],
                "reads_unique_onehop": unique_onehop[c],
                "share_reads_onehop": round(reads_by_class_onehop[c] / reads_total, 4) if reads_total else 0,
                "share_reads_unique": round(unique_onehop[c] / reads_total, 4) if reads_total else 0,
            }
            for c in CLASSES
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e3_{args.dataset}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{args.dataset}] {queries} queries, {reads_total} distinct read_file locates, "
          f"{reads_unattributed} ({reads_unattributed/max(reads_total,1):.1%}) not attributable to any served entry")
    print(f"  served objects resolved {served_hits}, unresolved {served_miss}")
    print(f"  {'class':8s} {'entries':>9s} {'direct':>8s} {'onehop':>8s} {'share':>7s} {'unique':>8s} {'uniq%':>7s}")
    for c in CLASSES:
        d = report["per_class"][c]
        print(f"  {c:8s} {d['entries_surfaced']:9d} {d['reads_direct']:8d} {d['reads_onehop']:8d} "
              f"{d['share_reads_onehop']:6.1%} {d['reads_unique_onehop']:8d} {d['share_reads_unique']:6.1%}")
    print(f"wrote {out / f'e3_{args.dataset}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
