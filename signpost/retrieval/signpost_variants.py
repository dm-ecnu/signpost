from __future__ import annotations

"""Signpost variant filtering for ablation experiments."""

import os
from copy import deepcopy
from typing import Any


FULL = "full"
NO_OFFLINE = "no_offline"
NO_ONLINE = "no_online"
NO_SEMANTIC_CUES = "no_semantic_cues"
NO_PROVENANCE_CUES = "no_provenance_cues"
NO_VERTICAL_CUES = "no_vertical_cues"
NO_HORIZONTAL_CUES = "no_horizontal_cues"

VALID_VARIANTS = {
    FULL,
    NO_OFFLINE,
    NO_ONLINE,
    NO_SEMANTIC_CUES,
    NO_PROVENANCE_CUES,
    NO_VERTICAL_CUES,
    NO_HORIZONTAL_CUES,
}


def normalize_variant(variant: str | None) -> str:
    normalized = (variant or FULL).strip().lower().replace("-", "_")
    if normalized not in VALID_VARIANTS:
        raise ValueError(f"unknown Signpost variant: {variant!r}; expected one of {sorted(VALID_VARIANTS)}")
    return normalized


# --- Top-b cue-budget truncation (ICDE Prop 2 realization) ---------------------
#
# Env-gated. When SIGNPOST_CUE_TOPB is unset/<=0, _cue_topb() returns None and
# apply_signpost_variant() takes its original code path byte-for-byte (the FULL
# branch still returns early without deepcopy). Only when SIGNPOST_CUE_TOPB is a
# positive int do we deep-copy the result and truncate each per-object cue family
# to its first b entries. Per-family overrides:
#   SIGNPOST_CUE_TOPB_V / _H / _S / _P  (vertical / horizontal / semantic / provenance)
# fall back to SIGNPOST_CUE_TOPB when unset. This truncates the materialized
# *exposure* only; it is NOT the source of token savings (those come from the
# fixed 2 LLM calls + bounded ReadFile).
#
# Cue list fields truncated, by sketch result_type (see offline_signpost.py):
#   chunk:    vertical.parent_summaries
#   summary:  vertical.child_summaries, vertical.child_chunks,
#             provenance.source_chunk_ids, provenance.source_locates
#   entity:   semantic.neighboring_entities,
#             provenance.source_chunk_ids, provenance.source_locates
#   relation: semantic.neighboring_entities  (P95=389 long tail; main target)
#             provenance.source_chunk_ids, provenance.source_locates
# horizontal (prev/next) and a chunk's single provenance.locate are size<=1; left intact.


def _cue_topb() -> dict[str, int] | None:
    base = _env_int("SIGNPOST_CUE_TOPB", 0)
    per = {
        "v": _env_int("SIGNPOST_CUE_TOPB_V", base),
        "h": _env_int("SIGNPOST_CUE_TOPB_H", base),
        "s": _env_int("SIGNPOST_CUE_TOPB_S", base),
        "p": _env_int("SIGNPOST_CUE_TOPB_P", base),
    }
    if all(value <= 0 for value in per.values()):
        return None
    return per


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _truncate_list(container: dict[str, Any], key: str, budget: int) -> None:
    if budget <= 0:
        return
    value = container.get(key)
    if isinstance(value, list) and len(value) > budget:
        container[key] = value[:budget]


def _cue_select_mode() -> str:
    """SIGNPOST_CUE_SELECT = 'greedy' (paper Sec 4.2 omega_x coverage) or
    'truncate' (default, value[:b] prefix of the offline-ordered list)."""
    mode = os.getenv("SIGNPOST_CUE_SELECT", "truncate")
    return "greedy" if mode.strip().lower() == "greedy" else "truncate"


def _greedy_select_list(container: dict[str, Any], key: str, budget: int, *, family: str) -> None:
    """Replace container[key] with the greedy maximum-coverage top-budget prefix.

    Operates on the ALREADY-MATERIALIZED cue list (online, per query) -- the
    offline materialization is untouched. Each cue is scored by the family
    relevance omega_x and its evidence units Phi (locate / chunk id), then the
    greedy marginal-coverage rule of Sec 4.2 (Algorithm 2 budgeted branch) keeps
    the top-b complementary cues. Falls back to truncation if Phi is unavailable.
    """
    if budget <= 0:
        return
    value = container.get(key)
    if not isinstance(value, list) or len(value) <= budget:
        return
    from signpost.retrieval.cue_coverage import CueCandidate, select_budgeted_greedy

    candidates: list[CueCandidate] = []
    for pos, entry in enumerate(value):
        if isinstance(entry, dict):
            locate = entry.get("locate") or entry.get("node_id") or entry.get("chunk_id") or str(pos)
            chunks = entry.get("source_chunk_ids") or []
            stable = str(entry.get("node_id") or entry.get("chunk_id") or pos)
        else:
            locate, chunks, stable = str(entry), [], str(entry)
        # evidence units: the cue's own locate plus any supporting chunk ids;
        # omega weight decays with offline rank (earlier = more relevant prior).
        weight = 1.0 / (1.0 + pos)
        phi = {locate: weight}
        for cid in chunks:
            phi[cid] = max(phi.get(cid, 0.0), weight)
        candidates.append(CueCandidate(phi=phi, score=weight, stable_id=stable, payload=entry))
    chosen = select_budgeted_greedy(candidates, budget)
    if chosen:
        container[key] = [c.payload for c in chosen]
    else:
        container[key] = value[:budget]


def _cap_list(container: dict[str, Any], key: str, budget: int, *, family: str, mode: str) -> None:
    if mode == "greedy":
        _greedy_select_list(container, key, budget, family=family)
    else:
        _truncate_list(container, key, budget)


def _apply_cue_topb_item(item: dict[str, Any], budgets: dict[str, int]) -> None:
    signpost = item.get("offline_signpost")
    if not isinstance(signpost, dict):
        return
    mode = _cue_select_mode()
    vertical = signpost.get("vertical")
    if isinstance(vertical, dict):
        _cap_list(vertical, "parent_summaries", budgets["v"], family="v", mode=mode)
        _cap_list(vertical, "child_summaries", budgets["v"], family="v", mode=mode)
        _cap_list(vertical, "child_chunks", budgets["v"], family="v", mode=mode)
    semantic = signpost.get("semantic")
    if isinstance(semantic, dict):
        _cap_list(semantic, "neighboring_entities", budgets["s"], family="s", mode=mode)
    provenance = signpost.get("provenance")
    if isinstance(provenance, dict):
        _cap_list(provenance, "source_chunk_ids", budgets["p"], family="p", mode=mode)
        _cap_list(provenance, "source_locates", budgets["p"], family="p", mode=mode)
    # top-level source_locates/source_chunk_ids (entity/relation items) mirror provenance
    _cap_list(item, "source_locates", budgets["p"], family="p", mode=mode)
    _cap_list(item, "source_chunk_ids", budgets["p"], family="p", mode=mode)


def apply_cue_topb(result: dict[str, Any]) -> dict[str, Any]:
    budgets = _cue_topb()
    if budgets is None:
        return result
    truncated = deepcopy(result)
    truncated.setdefault("metadata", {})["cue_topb"] = budgets
    for group_name in ("text_group", "graph_group"):
        group = truncated.get(group_name)
        if not isinstance(group, dict):
            continue
        for item in group.get("items") or []:
            if isinstance(item, dict):
                _apply_cue_topb_item(item, budgets)
    return truncated


def apply_signpost_variant(result: dict[str, Any], variant: str | None) -> dict[str, Any]:
    normalized = normalize_variant(variant)
    if normalized == FULL:
        result.setdefault("metadata", {})["signpost_variant"] = FULL
        return apply_cue_topb(result)

    filtered = deepcopy(result)
    for group_name in ("text_group", "graph_group"):
        group = filtered.get(group_name)
        if not isinstance(group, dict):
            continue
        if normalized in {NO_ONLINE, NO_SEMANTIC_CUES}:
            group["online_signpost"] = _empty_online_signpost(group.get("online_signpost"))
        for item in group.get("items") or []:
            _filter_item(item, normalized)
    filtered.setdefault("metadata", {})["signpost_variant"] = normalized
    return apply_cue_topb(filtered)


def _filter_item(item: dict[str, Any], variant: str) -> None:
    if variant == NO_OFFLINE:
        item["offline_signpost"] = {}
        return
    signpost = item.get("offline_signpost")
    if not isinstance(signpost, dict):
        return
    if variant == NO_SEMANTIC_CUES:
        signpost.pop("semantic", None)
    elif variant == NO_PROVENANCE_CUES:
        signpost.pop("provenance", None)
        item.pop("source_locates", None)
        item.pop("source_chunk_ids", None)
    elif variant == NO_VERTICAL_CUES:
        signpost.pop("vertical", None)
    elif variant == NO_HORIZONTAL_CUES:
        signpost.pop("horizontal", None)


def _empty_online_signpost(previous: Any) -> dict[str, Any]:
    scene = previous.get("scene") if isinstance(previous, dict) else None
    seeds = previous.get("seeds") if isinstance(previous, dict) else []
    return {
        "scene": scene,
        "seeds": seeds or [],
        "subgraph": {"nodes": 0, "edges": 0},
        "recommended_entities": [],
    }
