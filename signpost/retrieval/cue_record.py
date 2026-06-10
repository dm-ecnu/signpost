from __future__ import annotations

"""Online cue-record enrichment (paper Section 3, eq:cue-record).

The paper models a cue as a typed record
    c = <x, target, label, locator, score, cost>
where x is the action family (v/h/s/p = zoom/read/jump/verify), target is a
successor object or readable span, label is the short text shown to the agent,
locator is the source span when checkable, score is the materialization
relevance, and cost estimates the tokens added if the cue is exposed/read.

The offline materializer (offline_signpost.py) stores target/locator/ids only.
To avoid re-running the (expensive) offline materialization, the score/cost/
label/family fields are derived ONLINE, at retrieval/serialization time, from the
already-materialized cue plus its offline rank. This module provides that
enrichment; it is deterministic and depends only on the cue dict.
"""

from typing import Any


# family code -> human label
_FAMILY_LABEL = {"v": "zoom", "h": "read", "s": "jump", "p": "verify"}


def _estimate_tokens(text: str | None) -> int:
    """Rough token estimate: ~1 token per 4 chars (English) / per char (CJK)."""
    if not text:
        return 0
    n = len(text)
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    ascii_chars = n - cjk
    return cjk + (ascii_chars + 3) // 4


def _cue_label(cue: dict[str, Any], family: str) -> str:
    """Short text shown to the agent for this cue."""
    if family == "p":  # verify
        return str(cue.get("locate") or "")
    name = cue.get("title") or cue.get("name")
    if name:
        return str(name)
    return str(cue.get("node_id") or cue.get("chunk_id") or "")


def _cue_target(cue: dict[str, Any]) -> str | None:
    return cue.get("node_id") or cue.get("chunk_id") or cue.get("locate")


def _cue_text_for_cost(cue: dict[str, Any], family: str) -> str:
    """The text that would be serialized for this cue (basis for cost)."""
    parts = [_cue_label(cue, family)]
    loc = cue.get("locate")
    if loc:
        parts.append(str(loc))
    return " ".join(p for p in parts if p)


def enrich_cue(cue: dict[str, Any], *, family: str, rank: int) -> dict[str, Any]:
    """Add x/target/label/score/cost to one materialized cue (in place, returns it).

    rank: 0-based position in the offline-ordered family list (earlier = higher
    relevance prior, so score = 1/(1+rank)). Idempotent.
    """
    if not isinstance(cue, dict):
        return cue
    cue.setdefault("x", family)
    cue.setdefault("family", _FAMILY_LABEL.get(family, family))
    if "target" not in cue:
        target = _cue_target(cue)
        if target is not None:
            cue["target"] = target
    if "label" not in cue:
        cue["label"] = _cue_label(cue, family)
    if "score" not in cue:
        cue["score"] = round(1.0 / (1.0 + max(0, rank)), 4)
    if "cost" not in cue:
        cue["cost"] = _estimate_tokens(_cue_text_for_cost(cue, family))
    return cue


def enrich_offline_signpost(signpost: dict[str, Any]) -> dict[str, Any]:
    """Enrich every cue in an item's offline_signpost with eq:cue-record fields.

    Maps the offline cue lists to action families:
      vertical   -> v (zoom)   : parent_summaries / child_summaries / child_chunks / nearest_parent_summary
      horizontal -> h (read)   : previous_chunk / next_chunk
      semantic   -> s (jump)   : neighboring_entities / source_entity / target_entity
      provenance -> p (verify) : source_locates (as {locate} records)
    """
    if not isinstance(signpost, dict):
        return signpost

    vertical = signpost.get("vertical")
    if isinstance(vertical, dict):
        for key in ("parent_summaries", "child_summaries", "child_chunks"):
            lst = vertical.get(key)
            if isinstance(lst, list):
                for i, c in enumerate(lst):
                    enrich_cue(c, family="v", rank=i)
        if isinstance(vertical.get("nearest_parent_summary"), dict):
            enrich_cue(vertical["nearest_parent_summary"], family="v", rank=0)
        if isinstance(vertical.get("parent_summary"), dict):
            enrich_cue(vertical["parent_summary"], family="v", rank=0)

    horizontal = signpost.get("horizontal")
    if isinstance(horizontal, dict):
        for rank, key in enumerate(("previous_chunk", "next_chunk")):
            if isinstance(horizontal.get(key), dict):
                enrich_cue(horizontal[key], family="h", rank=rank)

    semantic = signpost.get("semantic")
    if isinstance(semantic, dict):
        lst = semantic.get("neighboring_entities")
        if isinstance(lst, list):
            for i, c in enumerate(lst):
                enrich_cue(c, family="s", rank=i)
        for key in ("source_entity", "target_entity"):
            if isinstance(semantic.get(key), dict):
                enrich_cue(semantic[key], family="s", rank=0)

    provenance = signpost.get("provenance")
    if isinstance(provenance, dict):
        locates = provenance.get("source_locates")
        if isinstance(locates, list):
            # provenance locates are bare strings; expose a typed record view
            provenance["verify_cues"] = [
                enrich_cue({"locate": loc}, family="p", rank=i)
                for i, loc in enumerate(locates)
            ]

    return signpost


def enrich_retrieval_result(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Enrich all items in a grouped retrieval result (in place, returns it)."""
    for group_name in ("text_group", "graph_group"):
        group = retrieval.get(group_name)
        if not isinstance(group, dict):
            continue
        for item in group.get("items") or []:
            if isinstance(item, dict):
                sp = item.get("offline_signpost")
                if isinstance(sp, dict):
                    enrich_offline_signpost(sp)
    return retrieval
