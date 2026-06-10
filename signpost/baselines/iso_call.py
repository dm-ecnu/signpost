from __future__ import annotations

"""Iso-Call baseline: attribution control for Signpost call-budget experiments.

Design rationale (paper §5.1)
------------------------------
Signpost's gains could come from two distinct sources:
  (a) its typed cue-structure (zoom / read / jump / verify cue families), or
  (b) simply having a higher LLM-call budget than flat-RAG baselines.

The iso-call baseline rules out (b) by running a ReAct-style agent that:
  * operates on the SAME retrieval backend and corpus as Signpost,
  * receives the SAME number of LLM calls as Signpost (controlled by
    ``call_budget``),
  * but treats ALL graph neighbors as a FLAT, UNTYPED list — no cue families,
    no zoom/jump/read/verify categorisation.

If Signpost still outperforms iso-call at equal call budget, the gains are
attributable to the typed cue structure, not the budget.

Implementation contract
------------------------
* Uses ``signpost.baselines.common`` helpers (``BaselineResult``,
  ``run_baseline_batch``, ``chat_once``, ``join_context``, …) so it produces
  identical prediction-JSONL schema to every other baseline.
* Uses the same LLM client (``OpenAICompatibleClient``) as all other baselines.
* For corpus search it uses the same local keyword search over chunks that
  HiPRAG / vanilla_rag use when ``use_es=False``, and ``search_chunks`` when
  ``use_es=True``.  Graph objects (entities, relations) from the unified graph
  are appended as plain-text neighbors with no type annotation.
* Imports of ES / LLM clients are deferred (inside runner methods) so this
  module can be imported at test time without live services.
* Registered under ``METHOD = "iso_call"``.

Call-budget semantics
---------------------
``call_budget`` counts every LLM ``chat`` round-trip. The agent performs:
  - up to ``call_budget - 1`` ReAct *think-then-act* steps (each = 1 LLM call,
    optionally followed by a retrieval tool call),
  - 1 final synthesis LLM call (always consumed last).

This mirrors Signpost's 2-call default (1 decompose + 1 synthesize) and can be
set to match measured per-question call counts from Signpost batch logs.
"""

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signpost.baselines.common import (
    BaselineResult,
    build_paths,
    chat_once,
    join_context,
    load_jsonl_list,
    locate_from_chunk,
    question_text,
    run_baseline_batch,
)
from signpost.config.context import resolve_project_path
from signpost.llm.client import OpenAICompatibleClient


METHOD = "iso_call"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a retrieval-augmented QA assistant operating under a strict LLM-call budget.
Answer the question strictly from private-corpus evidence. Do NOT use outside knowledge.

Output format for each reasoning step:
Thought: <brief reasoning about what you still need>
Action: search
Query: <search query string>

When you have enough evidence, instead output:
Thought: <brief final reasoning>
Action: finish

Rules:
- Every step must be either an Action: search step or Action: finish step.
- Do not invent information. If evidence is insufficient, write "Insufficient evidence." as the answer.
- Do not include file names, line numbers, or chunk IDs in your final answer."""

SYNTHESIS_SYSTEM_PROMPT = """Answer the question in English strictly based on the retrieved evidence passages below.
Provide a comprehensive answer. Do not use outside knowledge. Do not include citations, file names, chunk IDs, or line numbers.
If evidence is insufficient, write exactly: "Insufficient evidence." """


# ---------------------------------------------------------------------------
# Local search helpers (no import of ES / embedding at module level)
# ---------------------------------------------------------------------------


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9一-鿿]+", text.lower()) if len(t) >= 2]


def _local_keyword_search(chunks: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    terms = _terms(query)
    scored = []
    for item in chunks:
        content = str(item.get("content") or "")
        score = sum(content.lower().count(t) for t in terms)
        if score > 0:
            scored.append({**item, "score": float(score), "score_source": "iso_call_local_keyword"})
    return sorted(scored, key=lambda x: (-float(x["score"]), str(x.get("chunk_id", ""))))[:top_k]


def _graph_text_neighbors(graph: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    """Flatten ALL graph objects (chunks, summaries, entities, relations) into
    untyped text neighbors and keyword-rank them.  No cue-family labels are
    applied — this is the key difference from Signpost's typed cue structure.
    """
    terms = _terms(query)
    candidates: list[dict[str, Any]] = []

    # Summary nodes
    for node in graph.get("nodes", []):
        node_type = node.get("node_type") or ""
        if node_type == "summary":
            content = " ".join(
                str(node.get(k, "")) for k in ("title", "summary", "content") if node.get(k)
            )
        elif node_type == "entity":
            content = " ".join(
                str(node.get(k, "")) for k in ("name", "description", "entity_type") if node.get(k)
            )
        elif node_type == "chunk":
            content = str(node.get("content") or "")
        else:
            continue  # skip unrecognised node types

        score = float(sum(content.lower().count(t) for t in terms))
        if score <= 0:
            continue
        candidates.append(
            {
                "content": content,
                "score": score,
                "score_source": "iso_call_graph_keyword",
                "source_chunk_ids": node.get("source_chunk_ids") or [],
                "source_locates": node.get("source_locates") or [],
                # Deliberately NO type / cue-family annotation.
            }
        )

    # Relation edges
    node_names: dict[str, str] = {
        n.get("node_id", ""): str(n.get("name") or "")
        for n in graph.get("nodes", [])
        if n.get("node_id")
    }
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "semantic":
            continue
        content = " ".join(
            filter(
                None,
                [
                    node_names.get(str(edge.get("source") or ""), ""),
                    node_names.get(str(edge.get("target") or ""), ""),
                    " ".join(edge.get("relation_types") or []),
                    str(edge.get("description") or ""),
                ],
            )
        )
        score = float(sum(content.lower().count(t) for t in terms))
        if score <= 0:
            continue
        candidates.append(
            {
                "content": content,
                "score": score,
                "score_source": "iso_call_graph_keyword",
                "source_chunk_ids": edge.get("source_chunk_ids") or [],
                "source_locates": edge.get("source_locates") or [],
                # Deliberately NO type / cue-family annotation.
            }
        )

    return sorted(candidates, key=lambda x: (-x["score"], str(x.get("content", ""))[:40]))[:top_k]


def _evidence_chunk_rows(chunks: list[dict[str, Any]], *, round_index: int, query: str) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(chunks, start=1):
        rows.append(
            {
                "rank": rank,
                "round": round_index,
                "source": "iso_call_search",
                "query": query,
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "file_name": item.get("file_name"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
                "content_preview": str(item.get("content") or "")[:240],
            }
        )
    return rows


def _dedupe_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        result.append(chunk)
        seen.add(chunk_id)
        if len(result) >= limit:
            break
    return result


def _extract_action_and_query(text: str) -> tuple[str, str]:
    """Parse the ReAct step output.

    Returns (action, query_string).  action is one of "search", "finish", or
    "" (could not parse).  query_string is empty for finish/unrecognised.
    """
    action_match = re.search(r"^\s*Action\s*:\s*(\w+)", text, re.MULTILINE | re.IGNORECASE)
    if not action_match:
        return "", ""
    action = action_match.group(1).lower().strip()
    if action == "finish":
        return "finish", ""
    if action == "search":
        query_match = re.search(r"^\s*Query\s*:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else ""
        return "search", query
    return action, ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsoCallConfig:
    """Configuration for the iso-call agent."""

    call_budget: int
    """Total LLM chat() calls per question. Must be >= 2 (at least one ReAct
    step + one synthesis). Mirrors Signpost's measured per-question call count.
    The final call is always the synthesis; all remaining calls are ReAct steps.
    """

    search_top_k: int
    """Number of chunks to retrieve per search step."""

    graph_top_k: int
    """Number of untyped graph-text neighbors to mix in per search step."""

    max_context_tokens: int
    """Token budget for the context window passed to the synthesis LLM call."""

    use_es: bool
    """If True, use Elasticsearch for chunk retrieval (requires a live ES)."""

    mode: str
    """Retrieval mode for ES: 'bm25', 'dense', or 'hybrid'. Ignored when
    use_es=False (local keyword-only search is always used instead)."""

    embedding_provider: str
    """Embedding provider name for ES dense retrieval."""


class IsoCallRunner:
    """Per-dataset runner for the iso-call baseline.

    Constructed once per batch run (not per question).  Heavy state (loaded
    chunks, graph) is held here so per-question calls stay cheap.
    """

    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        graph_path: Path | None,
        config: IsoCallConfig,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.config = config
        # Load chunks for local keyword search (always loaded; also used as ES
        # fallback when ES is unavailable at test time).
        self.chunks: list[dict[str, Any]] = load_jsonl_list(chunks_path) if chunks_path.exists() else []
        # Load unified graph for untyped neighbor expansion.  Optional — if the
        # graph file does not exist (e.g. in tests), graph neighbors are skipped.
        self.graph: dict[str, Any] = {}
        if graph_path and graph_path.exists():
            import json as _json  # noqa: PLC0415
            self.graph = _json.loads(graph_path.read_text(encoding="utf-8"))

        # OpenAICompatibleClient is imported at module level (matching the other
        # baselines) — importing it does not require a live endpoint.  The
        # actual HTTP connection only happens when runner.answer() is called.
        self.llm = OpenAICompatibleClient()

    def _search(self, query: str, round_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (chunk_hits, evidence_rows).

        Retrieves from the chunk corpus (ES or local keyword), then merges
        untyped graph-text neighbors.  No cue families are assigned.
        """
        if self.config.use_es:
            from signpost.retrieval.chunk_search import search_chunks  # noqa: PLC0415
            items = search_chunks(
                namespace=self.namespace,
                query=query,
                mode=self.config.mode,
                top_k=self.config.search_top_k,
                embedding_provider_name=self.config.embedding_provider,
            ).get("items", [])
        else:
            items = _local_keyword_search(self.chunks, query, self.config.search_top_k)

        # Mix in untyped graph neighbors (text content only, no type labels).
        graph_neighbors = _graph_text_neighbors(self.graph, query, self.config.graph_top_k)
        # Graph neighbors carry source_chunk_ids; attempt to resolve them back
        # to full chunk dicts so locate/citation fields are populated.
        chunks_by_id: dict[str, dict[str, Any]] = {
            str(c.get("chunk_id") or ""): c for c in self.chunks if c.get("chunk_id")
        }
        for neighbor in graph_neighbors:
            for src_cid in neighbor.get("source_chunk_ids") or []:
                chunk = chunks_by_id.get(str(src_cid))
                if chunk and not any(str(i.get("chunk_id") or "") == str(src_cid) for i in items):
                    items.append(
                        {
                            **chunk,
                            "score": neighbor["score"],
                            "score_source": "iso_call_graph_neighbor",
                        }
                    )

        # Cap to search_top_k (graph expansion may push us over).
        items = items[: self.config.search_top_k + self.config.graph_top_k]
        evidence_rows = _evidence_chunk_rows(items, round_index=round_index, query=query)
        return items, evidence_rows

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        # The running transcript fed back into the LLM at each ReAct step.
        transcript = f"Question: {question}\n"
        all_retrieved: list[dict[str, Any]] = []
        evidence_chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        input_tokens = 0.0
        output_tokens = 0.0
        llm_calls = 0.0
        tool_calls = 0.0
        retrieval_latency = 0.0

        # call_budget - 1 ReAct steps, then 1 synthesis call.
        max_react_steps = max(1, self.config.call_budget - 1)

        for step_index in range(1, max_react_steps + 1):
            # Build the step prompt.
            context_hint = (
                f"  (You have {max_react_steps - step_index} retrieval steps remaining after this one, "
                f"then one synthesis call.)"
                if step_index < max_react_steps
                else "  (This is your last retrieval step. After this, you must synthesize.)"
            )
            step_prompt = (
                f"{transcript}\n"
                f"Continue the ReAct trace.{context_hint}\n"
                "Output exactly one Thought + Action line."
            )
            raw, in_tok, out_tok, llm_latency_step = chat_once(
                self.llm,
                [{"role": "system", "content": REACT_SYSTEM_PROMPT}, {"role": "user", "content": step_prompt}],
                input_text=REACT_SYSTEM_PROMPT + "\n" + step_prompt,
            )
            input_tokens += in_tok
            output_tokens += out_tok
            llm_calls += 1.0
            trace.append(
                {
                    "event_type": "llm_call",
                    "stage": "iso_call_react_step",
                    "step": step_index,
                    "latency_seconds": llm_latency_step,
                    "input_tokens_estimate": in_tok,
                    "output_tokens_estimate": out_tok,
                }
            )
            transcript += raw.strip() + "\n"

            action, query = _extract_action_and_query(raw)
            if action == "finish":
                # Model decided it has enough evidence; stop early.
                trace.append({"event_type": "control", "stage": "iso_call_early_finish", "step": step_index})
                break
            if action == "search" and query:
                search_started = time.time()
                items, evidence_rows = self._search(query, step_index)
                search_latency = time.time() - search_started
                retrieval_latency += search_latency
                tool_calls += 1.0
                all_retrieved.extend(items)
                evidence_chunks.extend(evidence_rows)
                # Append plain evidence text to transcript (no type labels).
                context_text, _used = join_context(items, max_context_tokens=self.config.max_context_tokens // max(1, max_react_steps))
                transcript += f"Observation:\n{context_text}\n"
                trace.append(
                    {
                        "event_type": "tool_call",
                        "tool": "iso_call_untyped_neighbor_search",
                        "step": step_index,
                        "query": query,
                        "latency_seconds": search_latency,
                        "output_summary": {
                            "retrieved_chunks": len(items),
                            "use_es": self.config.use_es,
                            # Key: confirms no cue-family typing was applied.
                            "cue_typed": False,
                        },
                    }
                )
            else:
                # No valid action parsed; keep the transcript and continue.
                transcript += "Observation: (no search issued)\n"

        # --------------- Final synthesis call (always the last LLM call) ----
        # Build accumulated context from all retrieved chunks.
        deduped = _dedupe_chunks(all_retrieved, self.config.search_top_k * max(1, max_react_steps))
        context, used_for_synthesis = join_context(deduped, max_context_tokens=self.config.max_context_tokens)

        if not context.strip():
            # No evidence at all — force a search on the original question so
            # the synthesis call has something to work with.
            fallback_started = time.time()
            fallback_items, fallback_rows = self._search(question, 0)
            fallback_latency = time.time() - fallback_started
            retrieval_latency += fallback_latency
            tool_calls += 1.0
            all_retrieved.extend(fallback_items)
            evidence_chunks.extend(fallback_rows)
            context, used_for_synthesis = join_context(fallback_items, max_context_tokens=self.config.max_context_tokens)
            trace.append(
                {
                    "event_type": "tool_call",
                    "tool": "iso_call_untyped_neighbor_search",
                    "step": 0,
                    "query": question,
                    "latency_seconds": fallback_latency,
                    "output_summary": {
                        "retrieved_chunks": len(fallback_items),
                        "use_es": self.config.use_es,
                        "cue_typed": False,
                        "reason": "fallback_no_prior_evidence",
                    },
                }
            )

        synthesis_prompt = (
            f"Evidence passages:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        answer, in_tok, out_tok, synth_latency = chat_once(
            self.llm,
            [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT}, {"role": "user", "content": synthesis_prompt}],
            input_text=SYNTHESIS_SYSTEM_PROMPT + "\n" + synthesis_prompt,
        )
        input_tokens += in_tok
        output_tokens += out_tok
        llm_calls += 1.0
        trace.append(
            {
                "event_type": "llm_call",
                "stage": "iso_call_synthesis",
                "latency_seconds": synth_latency,
                "input_tokens_estimate": in_tok,
                "output_tokens_estimate": out_tok,
            }
        )

        # Assemble final result fields.
        final_deduped = _dedupe_chunks(all_retrieved, self.config.search_top_k * max(1, max_react_steps))
        citations = [
            {
                "file_name": item.get("file_name"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "locate": locate_from_chunk(item),
            }
            for item in final_deduped
            if locate_from_chunk(item)
        ]
        retrieved_chunks = [
            {
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
            }
            for item in final_deduped
            if item.get("chunk_id")
        ]
        return BaselineResult(
            answer=answer.strip(),
            rationale=transcript,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            evidence_chunks=evidence_chunks,
            trace=trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            retrieval_latency_seconds=retrieval_latency,
        )


# ---------------------------------------------------------------------------
# Public run function (matches the pattern of every other baseline)
# ---------------------------------------------------------------------------


def run_iso_call(
    *,
    dataset: str,
    namespace: str | None = None,
    questions_path: str | None = None,
    chunks_path: str | None = None,
    graph_path: str | None = None,
    output_path: str | None = None,
    query_log_path: str | None = None,
    limit: int | None = None,
    call_budget: int = 2,
    search_top_k: int = 5,
    graph_top_k: int = 5,
    max_context_tokens: int = 3500,
    use_es: bool = False,
    mode: str = "hybrid",
    embedding_provider: str = "ecnu",
    workers: int | None = None,
) -> int:
    """Run the iso-call baseline and write predictions JSONL.

    Parameters
    ----------
    call_budget:
        Total number of LLM ``chat()`` calls per question.  Set this to match
        Signpost's measured per-question LLM-call count (from
        ``outputs/<dataset>/metrics/signpost.query_metrics.json`` →
        ``avg_llm_calls``).  Default 2 matches Signpost's decompose+synthesize
        minimum with ``use_llm=True``.
    """
    paths = build_paths(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method=METHOD,
    )
    resolved_chunks = resolve_project_path(
        chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"
    )
    # The unified graph provides graph-neighbor context; optional.
    resolved_graph: Path | None = None
    if graph_path:
        resolved_graph = resolve_project_path(graph_path)
    else:
        candidate = resolve_project_path(f"datasets/processed/{dataset}/graph.unified.json")
        if candidate.exists():
            resolved_graph = candidate

    config = IsoCallConfig(
        call_budget=max(2, call_budget),
        search_top_k=search_top_k,
        graph_top_k=graph_top_k,
        max_context_tokens=max_context_tokens,
        use_es=use_es,
        mode=mode,
        embedding_provider=embedding_provider,
    )
    runner = IsoCallRunner(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolved_chunks,
        graph_path=resolved_graph,
        config=config,
    )
    return run_baseline_batch(
        method=METHOD,
        paths=paths,
        answer_fn=runner.answer,
        limit=limit,
        workers=workers,
        metadata={
            "retrieval": "iso_call_react_untyped_neighbors",
            "call_budget": config.call_budget,
            "search_top_k": search_top_k,
            "graph_top_k": graph_top_k,
            "max_context_tokens": max_context_tokens,
            "use_es": use_es,
            "mode": mode,
            "embedding_provider": embedding_provider,
            "cue_typed": False,
            "prompt_style": "iso_call_react_thought_action_query",
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Iso-Call baseline: ReAct over untyped graph neighbors "
            "at the same LLM-call budget as Signpost."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--graph", dest="graph_path", help="Path to graph.unified.json (optional).")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument(
        "--call-budget",
        type=int,
        default=2,
        help=(
            "Total LLM chat() calls per question. "
            "Set to Signpost's measured avg_llm_calls to iso-call budget-match. "
            "Minimum 2 (1 ReAct step + 1 synthesis)."
        ),
    )
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=3500)
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    count = run_iso_call(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        graph_path=args.graph_path,
        output_path=args.output,
        query_log_path=args.query_log,
        limit=args.limit,
        call_budget=args.call_budget,
        search_top_k=args.search_top_k,
        graph_top_k=args.graph_top_k,
        max_context_tokens=args.max_context_tokens,
        use_es=args.use_es,
        mode=args.mode,
        embedding_provider=args.embedding_provider,
        workers=args.workers,
    )
    output = resolve_project_path(
        args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl"
    )
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
