from __future__ import annotations

"""Cue coverage selection (paper Section 4.2 / Algorithm 2 MaterializeSignposts).

Implements the budgeted maximum-coverage cue selection the paper specifies:

- Each candidate cue ``c`` in a family ``x`` of object ``o`` resolves, through the
  provenance layer, to a set of *source-evidence units* ``Phi(c)`` (source chunks
  with line ranges that following ``c`` makes reachable). A cue with no source
  path has ``Phi(c) = {}``.
- A family-specific relevance ``omega_x(o, u)`` weights each evidence unit:
    zoom   (vertical):   tree-distance decay   (1 + hop_Td)^-1
    read   (horizontal): source-line proximity (1 + |delta_line|)^-1
    jump   (semantic):   relation salience     w * log(1 + |S(.)|)
    verify (provenance): coverage fraction     cov(., o)
- Coverage value of a cue set S:  f_{o,x}(S) = sum_{u in union Phi(c)} omega_x(o,u)
  (a unit covered by two cues is counted once => monotone submodular).
- ``complete`` mode: keep all source-checkable cues, sorted by family score then
  stable id. ``budgeted`` mode: greedy marginal-gain prefix of size <= b_x,
  giving the (1 - 1/e) approximation.

This module is deterministic and depends only on the unified graph + the cue
records already built by offline_signpost.py; no LLM, no ES.
"""

import math
from typing import Any, Callable, Hashable


# An evidence unit is identified by a hashable key (e.g. a "file:Lstart-Lend"
# locate string or a source chunk id). Phi(c) is a set of such keys with a
# per-unit weight contributed under omega_x.
EvidenceUnit = Hashable


def _parse_line_span(locate: str | None) -> tuple[int, int] | None:
    if not locate or ":L" not in locate:
        return None
    try:
        _, span = locate.rsplit(":L", 1)
        start_s, end_s = span.split("-L")
        return int(start_s), int(end_s)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# omega_x: family-specific relevance of an evidence unit u to object o.
# Returns a positive weight; larger = more relevant.
# ---------------------------------------------------------------------------

def omega_vertical(hop: int) -> float:
    """zoom: tree-distance decay (1 + hop_Td)^-1."""
    return 1.0 / (1.0 + max(0, hop))


def omega_horizontal(delta_line: int) -> float:
    """read: source-line proximity (1 + |delta_line|)^-1."""
    return 1.0 / (1.0 + abs(delta_line))


def omega_semantic(weight: float, support_size: int) -> float:
    """jump: relation salience w * log(1 + |S(.)|)."""
    return max(weight, 1.0) * math.log(1.0 + max(0, support_size))


def omega_provenance(cov_fraction: float) -> float:
    """verify: coverage fraction cov(., o) in [0, 1]."""
    return max(0.0, min(1.0, cov_fraction))


# ---------------------------------------------------------------------------
# Candidate cue model
# ---------------------------------------------------------------------------

class CueCandidate:
    """One candidate cue with its evidence-unit weights and stable ordering key.

    phi: dict mapping evidence-unit key -> omega weight contributed by this cue.
    score: family relevance score used for complete-mode ordering.
    stable_id: deterministic tiebreak (e.g. target node id).
    payload: the actual cue record (id/label/locator/...) to store if selected.
    """

    __slots__ = ("phi", "score", "stable_id", "payload")

    def __init__(self, phi: dict[EvidenceUnit, float], score: float, stable_id: str, payload: dict[str, Any]):
        self.phi = phi
        self.score = score
        self.stable_id = stable_id
        self.payload = payload


def coverage_value(selected: list[CueCandidate]) -> float:
    """f_{o,x}(S): weighted union coverage (each unit counted once, max weight)."""
    best: dict[EvidenceUnit, float] = {}
    for c in selected:
        for unit, w in c.phi.items():
            if w > best.get(unit, 0.0):
                best[unit] = w
    return sum(best.values())


def _marginal_gain(cand: CueCandidate, covered: dict[EvidenceUnit, float]) -> float:
    """Delta_{o,x}(c | S): extra coverage c adds over the already-covered units."""
    gain = 0.0
    for unit, w in cand.phi.items():
        prev = covered.get(unit, 0.0)
        if w > prev:
            gain += w - prev
    return gain


def select_complete(candidates: list[CueCandidate]) -> list[CueCandidate]:
    """Complete mode: all source-checkable cues sorted by (score desc, stable id)."""
    return sorted(candidates, key=lambda c: (-c.score, c.stable_id))


def select_budgeted_greedy(candidates: list[CueCandidate], budget: int) -> list[CueCandidate]:
    """Budgeted mode: greedy marginal-coverage prefix of size <= budget.

    Repeatedly adds the cue of largest marginal gain among those still adding new
    evidence, matching Algorithm 2's inner while-loop. (1 - 1/e) approximation.
    """
    if budget <= 0:
        return []
    selected: list[CueCandidate] = []
    covered: dict[EvidenceUnit, float] = {}
    remaining = [c for c in candidates if c.phi]  # cues with no source path are never selected
    while len(selected) < budget and remaining:
        best = None
        best_gain = 0.0
        best_key = None
        for cand in remaining:
            gain = _marginal_gain(cand, covered)
            # tiebreak deterministically by (score desc, stable id)
            key = (gain, cand.score, _neg_str(cand.stable_id))
            if gain > 0 and (best is None or key > best_key):
                best, best_gain, best_key = cand, gain, key
        if best is None or best_gain <= 0.0:
            break
        selected.append(best)
        for unit, w in best.phi.items():
            if w > covered.get(unit, 0.0):
                covered[unit] = w
        remaining.remove(best)
    return selected


def _neg_str(s: str) -> tuple:
    """Make string comparison work with 'larger is better' tuple ordering."""
    # Negate lexicographic order: smaller stable_id should win the tiebreak, so
    # invert by mapping to a reversed comparison key.
    return tuple(-ord(ch) for ch in s)


def select_family(
    candidates: list[CueCandidate],
    *,
    mode: str = "complete",
    budget: int = 0,
) -> list[dict[str, Any]]:
    """Select cues for one family and return their stored payloads in order.

    mode='complete' -> sort by family score + stable id (no truncation).
    mode='budgeted' -> greedy marginal-coverage top-budget prefix.
    """
    if mode == "budgeted":
        chosen = select_budgeted_greedy(candidates, budget)
    else:
        chosen = select_complete(candidates)
    return [c.payload for c in chosen]


# ---------------------------------------------------------------------------
# Builders: turn an object's family neighborhood (from offline_signpost) into
# CueCandidate lists with Phi and omega scores. These read the graph index.
# ---------------------------------------------------------------------------

def build_vertical_candidates(parent_refs: list[dict[str, Any]], *, locate_of: Callable[[dict], str | None]) -> list[CueCandidate]:
    """zoom candidates: parents/children. hop = list position (nearest = hop 1)."""
    out: list[CueCandidate] = []
    for hop, ref in enumerate(parent_refs, start=1):
        loc = locate_of(ref)
        phi = {loc: omega_vertical(hop)} if loc else {}
        out.append(CueCandidate(phi=phi, score=omega_vertical(hop), stable_id=str(ref.get("node_id") or ""), payload=ref))
    return out


def build_horizontal_candidates(neighbor_refs: list[tuple[dict[str, Any], int]], *, locate_of: Callable[[dict], str | None]) -> list[CueCandidate]:
    """read candidates: (chunk_ref, delta_line) pairs."""
    out: list[CueCandidate] = []
    for ref, delta in neighbor_refs:
        loc = locate_of(ref)
        phi = {loc: omega_horizontal(delta)} if loc else {}
        out.append(CueCandidate(phi=phi, score=omega_horizontal(delta), stable_id=str(ref.get("node_id") or ""), payload=ref))
    return out


def build_semantic_candidates(neighbor_refs: list[dict[str, Any]], *, weight_of: Callable[[dict], float], support_of: Callable[[dict], list[str]]) -> list[CueCandidate]:
    """jump candidates: neighbor entities/relations. Phi = supporting chunk ids."""
    out: list[CueCandidate] = []
    for ref in neighbor_refs:
        support = support_of(ref) or []
        w = weight_of(ref)
        score = omega_semantic(w, len(support))
        phi = {chunk_id: score for chunk_id in support}
        out.append(CueCandidate(phi=phi, score=score, stable_id=str(ref.get("node_id") or ref.get("name") or ""), payload=ref))
    return out


def build_provenance_candidates(locates: list[str], *, total_units: int) -> list[CueCandidate]:
    """verify candidates: source locates. cov = 1/total (each locate covers itself)."""
    out: list[CueCandidate] = []
    cov = 1.0 / total_units if total_units > 0 else 1.0
    for loc in locates:
        out.append(CueCandidate(phi={loc: omega_provenance(cov)}, score=omega_provenance(cov), stable_id=str(loc), payload={"locate": loc}))
    return out
