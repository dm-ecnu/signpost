from __future__ import annotations

"""Sketch-chaining: multi-hop navigation for the Signpost agent (Algorithm 3).

This module implements `ServeSignpostQuery` from the paper (§5.1, Alg. 3):

    s_t = <q_i, F_t, H_t, R_t, B_t>
      q_i  – current subquestion
      F_t  – frontier (objects with their sketches yet to be expanded)
      H_t  – visited objects AND visited cues (prevents revisits/cycles)
      R_t  – accumulated read-span evidence (locate → snippet dict)
      B_t  – remaining read budget (number of ReadFile calls)

Algorithm 3 loop mapping
─────────────────────────────────────────────────────────────────────────────
Line in Alg. 3                          Code location
─────────────────────────────────────────────────────────────────────────────
TopK(I_C ∪ I_G, q_i, k)                Caller (Researcher.research_with_chaining)
                                        injects the initial F_0 from
                                        KnowledgeSearchTool.run().

Select unvisited o ∈ F by score/type   SketchChainer._select_next_object()
                                        pulls the highest-priority unvisited
                                        object from the frontier queue.

C ← context-adapted cues σ(o)          SketchChainer._adapt_cues()
                                        filters H_t (visited targets), dedup
                                        vs R_t (already-read locates), and
                                        suppresses cues with zero admissible
                                        targets.

Read verify cues Cp → R via ReadFile   SketchChainer._follow_verify_cues()

Follow zoom/read/jump cues → succs.    SketchChainer._follow_nav_cues()
not in H_t; update F with sketches     successor objects are resolved via
                                        the graph index and added to frontier.

Add to H_t                              SketchChainer._mark_visited()

Priority order: verify>read>zoom>jump  _PRIORITY constant in _adapt_cues().
─────────────────────────────────────────────────────────────────────────────

Interface
─────────────────────────────────────────────────────────────────────────────
The main entry point is `run_sketch_chaining(...)`.  It is called by
`Researcher.research_with_chaining` (supervisor.py) and returns a dict with
  - evidence: list of read-snippet dicts (R_t at termination)
  - locates:  ordered list of locate strings that were read
  - hops:     number of expansion iterations completed
  - visited:  frozenset of object ids added to H_t
  - trace_events: list of dicts for TraceRecorder
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Callable


# Priority order from §5.1: verify > read > zoom > jump.
# Lower integer = higher priority when selecting which family to expand first.
_FAMILY_PRIORITY = {"p": 0, "h": 1, "v": 2, "s": 3}

# Human-readable family names (matching cue_record.py)
_FAMILY_NAME = {"v": "zoom", "h": "read", "s": "jump", "p": "verify"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sketch_chaining(
    *,
    subquestion: str,
    initial_items: list[dict[str, Any]],
    graph_index: Any,                       # GraphIndex instance (offline_signpost)
    read_file_fn: Callable[[str], dict[str, Any]],
    read_budget: int = 10,
    max_hops: int = 3,
    cue_budget_per_family: int = 3,
) -> dict[str, Any]:
    """Execute Algorithm 3 for one subquestion.

    Parameters
    ----------
    subquestion:
        The subquestion being researched (q_i).
    initial_items:
        Objects returned by KnowledgeSearchTool (with offline_signpost attached).
        These form the initial frontier F_0.
    graph_index:
        A `GraphIndex` (from offline_signpost.py) for resolving successor node IDs
        to full node dicts.  May be None when testing without a real graph; in
        that case successor resolution via node lookup is skipped.
    read_file_fn:
        Callable(locate: str) → snippet dict.  Wraps ReadFileTool.run().
    read_budget:
        Maximum number of ReadFile calls (B_t).  Each verify-cue read
        decrements this.
    max_hops:
        Maximum expansion rounds (outer while-loop iterations).
    cue_budget_per_family:
        How many cues to expand per family per object per hop (b_x in the
        bounded-mode guarantee).  0 = unlimited (complete mode).

    Returns
    -------
    dict with keys: evidence, locates, hops, visited, trace_events
    """
    chainer = SketchChainer(
        subquestion=subquestion,
        graph_index=graph_index,
        read_file_fn=read_file_fn,
        read_budget=read_budget,
        max_hops=max_hops,
        cue_budget_per_family=cue_budget_per_family,
    )
    return chainer.run(initial_items)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class SketchChainer:
    """Stateful executor of Algorithm 3 for one subquestion."""

    def __init__(
        self,
        *,
        subquestion: str,
        graph_index: Any,
        read_file_fn: Callable[[str], dict[str, Any]],
        read_budget: int,
        max_hops: int,
        cue_budget_per_family: int,
    ) -> None:
        self.subquestion = subquestion
        self.graph_index = graph_index
        self.read_file_fn = read_file_fn
        self.read_budget = read_budget          # B_t (mutable during run)
        self.max_hops = max_hops
        self.cue_budget_per_family = cue_budget_per_family

        # Navigation state s_t
        # H_t: visited object ids AND visited cue targets (locate strings / node ids)
        self.H_t: set[str] = set()
        # R_t: accumulated evidence, keyed by locate to deduplicate
        self.R_t: dict[str, dict[str, Any]] = {}
        # Frontier: list of (priority_score, item_dict), highest first
        self.frontier: list[tuple[float, dict[str, Any]]] = []
        # Trace events for TraceRecorder
        self.trace_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public run
    # ------------------------------------------------------------------

    def run(self, initial_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute the chaining loop and return the navigation result."""
        # Seed frontier F_0 from initial retrieval results
        for item in initial_items:
            self._push_frontier(item)

        hops = 0
        while self.frontier and self.read_budget > 0 and hops < self.max_hops:
            obj = self._select_next_object()
            if obj is None:
                break
            obj_id = _object_id(obj)
            if obj_id in self.H_t:
                continue

            # Context-adapt cues against current state
            adapted = self._adapt_cues(obj)

            # Priority order: verify (p) > read (h) > zoom (v) > jump (s)
            self._follow_verify_cues(obj_id, adapted.get("p", []))
            if self.read_budget <= 0:
                self._mark_visited(obj_id)
                break
            successors = self._follow_nav_cues(obj_id, adapted)
            for succ in successors:
                self._push_frontier(succ)

            self._mark_visited(obj_id)
            hops += 1

        return {
            "evidence": list(self.R_t.values()),
            "locates": list(self.R_t.keys()),
            "hops": hops,
            "visited": frozenset(self.H_t),
            "trace_events": self.trace_events,
        }

    # ------------------------------------------------------------------
    # Frontier management
    # ------------------------------------------------------------------

    def _push_frontier(self, item: dict[str, Any]) -> None:
        """Add an item to the frontier if its id is not already visited."""
        obj_id = _object_id(item)
        if not obj_id or obj_id in self.H_t:
            return
        # Avoid duplicate frontier entries for the same object
        existing_ids = {_object_id(it) for _, it in self.frontier}
        if obj_id in existing_ids:
            return
        score = _retrieval_score(item)
        self.frontier.append((score, item))

    def _select_next_object(self) -> dict[str, Any] | None:
        """Pop the highest-score unvisited object from the frontier.

        Implements: "Select an unvisited object o ∈ F by retrieval score and
        type priority" (Alg. 3 line 5).  Type priority: chunk/entity first
        (concrete evidence) over summary (navigational).
        """
        if not self.frontier:
            return None
        # Sort by (score desc, type priority asc) to pick best unvisited
        self.frontier.sort(key=lambda t: (-t[0], _type_priority(t[1])))
        for i, (score, obj) in enumerate(self.frontier):
            obj_id = _object_id(obj)
            if obj_id and obj_id not in self.H_t:
                self.frontier.pop(i)
                return obj
        return None

    # ------------------------------------------------------------------
    # Context adaptation (Alg. 3 line 6: C ← adapted cues σ(o))
    # ------------------------------------------------------------------

    def _adapt_cues(self, obj: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Filter σ(o) against current state, return {family: [cue, ...]}.

        Context adaptation rules (§5.1):
        1. Remove targets already in H_t.
        2. Deduplicate verify cues whose locate is already in R_t.
        3. Respect cue_budget_per_family per family (b_x).
        """
        signpost = obj.get("offline_signpost") or {}
        result: dict[str, list[dict[str, Any]]] = {"v": [], "h": [], "s": [], "p": []}

        # --- verify (p): provenance.source_locates + top-level source_locates ---
        prov = signpost.get("provenance") or {}
        p_locates = list(prov.get("source_locates") or [])
        # top-level field also carries provenance locates for entity/relation items
        p_locates.extend(obj.get("source_locates") or [])
        # single locate on a chunk's own provenance
        if prov.get("locate"):
            p_locates.insert(0, prov["locate"])
        seen_p: set[str] = set()
        for loc in p_locates:
            if loc and loc not in self.R_t and loc not in self.H_t and loc not in seen_p:
                result["p"].append({"locate": loc, "x": "p", "family": "verify"})
                seen_p.add(loc)
        result["p"] = self._apply_budget(result["p"])

        # --- zoom (v): vertical cues (parent_summaries, child_summaries, child_chunks) ---
        vert = signpost.get("vertical") or {}
        v_refs: list[dict[str, Any]] = []
        for key in ("parent_summaries", "child_summaries", "child_chunks"):
            v_refs.extend(vert.get(key) or [])
        if vert.get("nearest_parent_summary"):
            v_refs.insert(0, vert["nearest_parent_summary"])
        if vert.get("parent_summary"):
            v_refs.insert(0, vert["parent_summary"])
        result["v"] = self._filter_nav_cues(v_refs, family="v")

        # --- read (h): horizontal cues (previous_chunk, next_chunk) ---
        horiz = signpost.get("horizontal") or {}
        h_refs: list[dict[str, Any]] = []
        for key in ("previous_chunk", "next_chunk"):
            ref = horiz.get(key)
            if isinstance(ref, dict):
                h_refs.append(ref)
        result["h"] = self._filter_nav_cues(h_refs, family="h")

        # --- jump (s): semantic cues (neighboring_entities) ---
        sem = signpost.get("semantic") or {}
        s_refs: list[dict[str, Any]] = list(sem.get("neighboring_entities") or [])
        for key in ("source_entity", "target_entity"):
            ref = sem.get(key)
            if isinstance(ref, dict):
                s_refs.append(ref)
        result["s"] = self._filter_nav_cues(s_refs, family="s")

        return result

    def _filter_nav_cues(self, refs: list[dict[str, Any]], *, family: str) -> list[dict[str, Any]]:
        """Filter nav cues (zoom/read/jump) removing H_t members, apply budget."""
        filtered = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            target_id = _cue_target_id(ref)
            if not target_id:
                continue
            if target_id in self.H_t or target_id in seen:
                continue
            seen.add(target_id)
            filtered.append({**ref, "x": family, "family": _FAMILY_NAME.get(family, family)})
        return self._apply_budget(filtered)

    def _apply_budget(self, cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.cue_budget_per_family > 0:
            return cues[: self.cue_budget_per_family]
        return cues

    # ------------------------------------------------------------------
    # Verify cue execution (Alg. 3 line 7: Read Cp → R via ReadFile)
    # ------------------------------------------------------------------

    def _follow_verify_cues(self, obj_id: str, verify_cues: list[dict[str, Any]]) -> None:
        """Execute verify cues: call ReadFile and add to R_t."""
        for cue in verify_cues:
            if self.read_budget <= 0:
                break
            locate = cue.get("locate")
            if not locate or locate in self.R_t:
                continue
            try:
                import time as _time
                _t0 = _time.time()
                snippet = self.read_file_fn(locate)
                _latency = _time.time() - _t0
                self.R_t[locate] = snippet
                self.read_budget -= 1
                self.H_t.add(locate)    # mark locate as visited to prevent re-read
                # Emit a tool_call event so trace consumers that look for
                # event_type="tool_call" / tool="read_file" continue to work
                # (backward-compatible with the simplified-path trace shape).
                self.trace_events.append({
                    "event_type": "tool_call",
                    "tool": "read_file",
                    "input": {"locate": locate},
                    "latency_seconds": _latency,
                    "output_summary": {
                        "file_name": snippet.get("file_name"),
                        "line_count": len(snippet.get("lines", [])),
                        "resolved": snippet.get("resolved"),
                    },
                })
                self.trace_events.append({
                    "event_type": "sketch_chain_verify",
                    "from_object": obj_id,
                    "locate": locate,
                    "read_budget_remaining": self.read_budget,
                })
            except Exception as exc:
                self.trace_events.append({
                    "event_type": "sketch_chain_verify_error",
                    "from_object": obj_id,
                    "locate": locate,
                    "error": str(exc),
                })

    # ------------------------------------------------------------------
    # Navigation cue following (Alg. 3 line 8: follow zoom/read/jump)
    # ------------------------------------------------------------------

    def _follow_nav_cues(
        self,
        obj_id: str,
        adapted: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Follow zoom/read/jump cues and return successor item dicts.

        Alg. 3 line 8: "Follow admissible zoom/read/jump cues in C_v, C_h, C_s
        to successor objects not in H_t".

        Successor objects may already carry their offline_signpost if they were
        resolved from the graph_index.  If graph_index is absent (test stub),
        we create a minimal item dict from the cue itself so the frontier still
        receives a traversable object.
        """
        successors: list[dict[str, Any]] = []
        # Follow in priority order: h (read) > v (zoom) > s (jump)
        for family in ("h", "v", "s"):
            cues = adapted.get(family) or []
            for cue in cues:
                target_id = _cue_target_id(cue)
                if not target_id or target_id in self.H_t:
                    continue
                succ = self._resolve_successor(target_id, cue)
                if succ is not None:
                    successors.append(succ)
                    self.trace_events.append({
                        "event_type": "sketch_chain_follow",
                        "from_object": obj_id,
                        "family": family,
                        "target_id": target_id,
                    })
        return successors

    def _resolve_successor(self, target_id: str, cue: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve a cue target id to an item dict (with offline_signpost if available).

        Uses graph_index.node_by_id if available; falls back to building a
        minimal stub from the cue record.
        """
        # Try graph index first (full run with a real graph)
        if self.graph_index is not None:
            node_by_id = getattr(self.graph_index, "node_by_id", {})
            node = node_by_id.get(target_id)
            if node is not None:
                # Build a retrieval-style item and attach its offline signpost
                from signpost.retrieval.offline_signpost import build_offline_signpost
                item = {
                    "node_id": node.get("node_id"),
                    "chunk_id": node.get("chunk_id"),
                    "object_type": node.get("node_type"),
                    "retrieval_type": node.get("node_type"),
                    "score": 0.0,           # successor; no retrieval score
                    "score_source": "sketch_chain",
                }
                try:
                    item["offline_signpost"] = build_offline_signpost(
                        self.graph_index.graph, node
                    )
                except Exception:
                    item["offline_signpost"] = {}
                return item

        # Fallback: build minimal item from the cue record itself.
        # This ensures the frontier receives a traversable object even when no
        # live graph index is available (e.g. in unit tests with mock graphs).
        item: dict[str, Any] = {
            "node_id": target_id,
            "score": 0.0,
            "score_source": "sketch_chain",
        }
        # Carry over locate / provenance if cue contains it
        if cue.get("locate"):
            item["offline_signpost"] = {
                "provenance": {"locate": cue["locate"], "source_locates": [cue["locate"]]},
                "vertical": {},
                "horizontal": {},
            }
        else:
            item["offline_signpost"] = {"provenance": {}, "vertical": {}, "horizontal": {}}
        # Copy any graph-structural fields directly present on the cue ref
        for key in ("file_name", "start_line", "end_line", "chunk_id", "source_locates", "source_chunk_ids"):
            if cue.get(key):
                item[key] = cue[key]
        return item

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _mark_visited(self, obj_id: str) -> None:
        """Add obj_id to H_t (Alg. 3 line 9)."""
        if obj_id:
            self.H_t.add(obj_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _object_id(item: dict[str, Any]) -> str:
    """Canonical id for an item, used for H_t membership tests."""
    # Prefer the most specific stable identifier
    return (
        item.get("node_id")
        or item.get("chunk_id")
        or item.get("edge_id")
        or item.get("id")
        or ""
    )


def _cue_target_id(cue: dict[str, Any]) -> str:
    """Extract the navigable target id from a cue record."""
    return (
        cue.get("node_id")
        or cue.get("chunk_id")
        or cue.get("target")
        or cue.get("locate")
        or ""
    )


def _retrieval_score(item: dict[str, Any]) -> float:
    """Numeric score for frontier ordering (higher = expanded sooner)."""
    score = item.get("score")
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass
    return 0.0


def _type_priority(item: dict[str, Any]) -> int:
    """Secondary sort key for frontier; lower = processed earlier.

    Priority: chunk (direct evidence) < entity < summary (navigational).
    """
    rt = item.get("retrieval_type") or item.get("object_type") or ""
    if rt == "chunk":
        return 0
    if rt == "entity":
        return 1
    if rt == "relation":
        return 2
    if rt == "summary":
        return 3
    return 4
