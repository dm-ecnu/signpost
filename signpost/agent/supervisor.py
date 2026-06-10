from __future__ import annotations

"""Budgeted controller for ServeSignpostQuery (paper Section 5.1, Alg. 3).

The default minimal controller of the paper: the Supervisor decomposes q into at
most three subquestions, each Researcher issues one KnowledgeSearch and follows
admissible cues in family priority (verify > read > zoom > jump; see
agent/sketch_chaining.py), every Verify resolves through ReadFile, and final
synthesis sees only the read set R_t, returning a cited answer or exactly
"Insufficient evidence".

The per-query model-call cost is fixed at TWO LLM calls -- one decomposition and
one synthesis -- independent of hop depth. Sketch chaining adds ReadFile calls
and object lookups, not LLM calls (it is deterministic), which is what preserves
the paper's 2-LLM-call claim. See METHOD_MAP.md.
"""

from dataclasses import dataclass, field
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from signpost.agent.tools import KnowledgeSearchTool, ReadFileTool
from signpost.chunking.tokenizer import count_tokens
from signpost.llm.client import OpenAICompatibleClient


SYNTHESIS_SYSTEM_PROMPT = """Answer the question in English strictly based on the provided evidence.
You must format your output as a valid JSON object containing exactly two keys: "rationale" and "answer".

Follow these rules:
1. "rationale": Briefly analyze the core intent of the question and identify the relevant facts from the evidence. Keep your step-by-step thinking and analysis in this field.
2. "answer": Provide the final response text here.
   - Write complete, well-formed sentences that fully answer the question.
   - Include all necessary context and details supported by the evidence so that the answer is comprehensive and stands alone clearly.
   - DO NOT include citations (e.g., [file.txt:L1-L3]), file names, or line numbers. Source tracking is handled externally.
   - DO NOT include conversational filler (e.g., "Based on the provided text...", "According to the evidence...") or your reasoning process here.
   - If the evidence is insufficient to answer the question, output exactly: "Insufficient evidence."

Example Output:
```json
{
  "rationale": "The question asks about the specific innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence lists hydroponic growing, aquaponics, composting, and biodiesel production, alongside community engagement efforts.",
  "answer": "Greensgrow Farm employs innovative practices such as hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. They also focus on community engagement and education to promote sustainable food practices."
}
```"""


@dataclass(frozen=True)
class AgentConfig:
    namespace: str
    max_subquestions: int = 3
    read_top_k: int = 3
    use_llm: bool = False
    # --- Sketch chaining (Algorithm 3, §5.1) ---
    # Set to False to reproduce the original simplified 2-call behaviour.
    # NOTE: enabling this changes the per-query LLM-call count from the
    # "fixed 2 calls" baseline; experiments MUST be re-run when this is ON.
    enable_sketch_chaining: bool = True
    sketch_chaining_max_hops: int = 3
    sketch_chaining_read_budget: int = 10
    sketch_chaining_cue_budget_per_family: int = 3


class TraceRecorder:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.events: list[dict[str, Any]] = []

    def add(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "trace_id": self.trace_id,
            "event_type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        self.events.append(event)
        return event


class Researcher:
    def __init__(
        self,
        search_tool: KnowledgeSearchTool,
        read_file_tool: ReadFileTool,
        trace: TraceRecorder,
        *,
        read_top_k: int = 3,
        config: "AgentConfig | None" = None,
    ):
        self.search_tool = search_tool
        self.read_file_tool = read_file_tool
        self.trace = trace
        self.read_top_k = read_top_k
        self.config = config
        self.evidence_rerank = _env_bool("SIGNPOST_EVIDENCE_RERANK", False)
        self.candidate_locate_top_k = _env_int("SIGNPOST_CANDIDATE_LOCATE_TOP_K", 30)
        self.rerank_top_k = _env_int("SIGNPOST_RERANK_TOP_K", max(read_top_k, 1))
        self.evidence_max_tokens = _env_int("SIGNPOST_EVIDENCE_MAX_TOKENS", 0)

    def research(self, subquestion: str) -> dict[str, Any]:
        """Dispatch to sketch-chaining or simplified path based on config."""
        if self.config is not None and self.config.enable_sketch_chaining:
            return self.research_with_chaining(subquestion)
        return self._research_simplified(subquestion)

    def research_with_chaining(self, subquestion: str) -> dict[str, Any]:
        """Algorithm 3 (ServeSignpostQuery) for one subquestion.

        Steps:
        1. Issue one KnowledgeSearch to seed F_0.
        2. Run SketchChainer: multi-hop loop following verify>read>zoom>jump cues
           while read budget remains.  H_t prevents revisits.  Verify cues drive
           ReadFile calls; nav cues (zoom/read/jump) expand the frontier.
        3. Apply optional evidence rerank over the collected R_t snippets.
        4. Return the same dict shape as _research_simplified so the Supervisor
           can use both paths interchangeably.

        NOTE: This path calls ReadFile up to `sketch_chaining_read_budget` times
        per subquestion (not just read_top_k), and adds sketch_chain_follow /
        sketch_chain_verify events to the trace.  The per-query LLM-call count
        is unchanged (decompose + synthesize = 2 if use_llm=True), but the
        number of ReadFile tool calls increases with the hop budget.
        """
        from signpost.agent.sketch_chaining import run_sketch_chaining
        from signpost.retrieval.offline_signpost import GraphIndex

        cfg = self.config
        assert cfg is not None  # guaranteed by callee check

        self.trace.add("researcher_start", subquestion=subquestion, mode="sketch_chaining")

        # --- Step 1: seed retrieval (F_0) ---
        search_started = time.time()
        retrieval = self.search_tool.run(subquestion)
        search_finished = time.time()
        self.trace.add(
            "tool_call",
            tool=self.search_tool.name,
            input={"query": subquestion},
            latency_seconds=search_finished - search_started,
            output_summary={
                "text_items": retrieval.get("metadata", {}).get("text_items", 0),
                "graph_items": retrieval.get("metadata", {}).get("graph_items", 0),
                "online_signposts": count_online_signpost_recommendations(retrieval),
            },
        )

        initial_items: list[dict[str, Any]] = []
        for group_name in ("text_group", "graph_group"):
            initial_items.extend(retrieval.get(group_name, {}).get("items", []))

        # Resolve graph index for successor lookups (optional – missing in tests)
        graph_index: Any = None
        if hasattr(self.search_tool, "graph") and self.search_tool.graph:
            try:
                graph_index = GraphIndex(self.search_tool.graph)
            except Exception:
                graph_index = None

        # --- Step 2: sketch chaining (Algorithm 3) ---
        chain_started = time.time()
        chain_result = run_sketch_chaining(
            subquestion=subquestion,
            initial_items=initial_items,
            graph_index=graph_index,
            read_file_fn=self.read_file_tool.run,
            read_budget=cfg.sketch_chaining_read_budget,
            max_hops=cfg.sketch_chaining_max_hops,
            cue_budget_per_family=cfg.sketch_chaining_cue_budget_per_family,
        )
        chain_finished = time.time()

        # Emit all sketch-chaining sub-events into the trace
        for evt in chain_result.get("trace_events", []):
            self.trace.add(**evt)

        self.trace.add(
            "sketch_chain_summary",
            subquestion=subquestion,
            hops=chain_result["hops"],
            evidence_count=len(chain_result["evidence"]),
            locates_read=len(chain_result["locates"]),
            visited_objects=len(chain_result["visited"]),
            latency_seconds=chain_finished - chain_started,
        )

        evidence = chain_result["evidence"]
        locates = chain_result["locates"]

        # --- Step 3: optional rerank (same path as simplified) ---
        if self.evidence_rerank:
            evidence = self._rerank_evidence(subquestion, evidence)[: self.read_top_k]
            locates = [
                _first_locate(snippet)
                for snippet in evidence
                if _first_locate(snippet)
            ]

        self.trace.add(
            "researcher_finish",
            subquestion=subquestion,
            evidence_count=len(evidence),
            mode="sketch_chaining",
        )

        return {
            "subquestion": subquestion,
            "retrieval": retrieval,
            "evidence": evidence,
            "locates": locates,
            "sketch_chain": {
                "hops": chain_result["hops"],
                "visited": len(chain_result["visited"]),
            },
        }

    def _research_simplified(self, subquestion: str) -> dict[str, Any]:
        self.trace.add("researcher_start", subquestion=subquestion)
        search_started = time.time()
        retrieval = self.search_tool.run(subquestion)
        search_finished = time.time()
        self.trace.add(
            "tool_call",
            tool=self.search_tool.name,
            input={"query": subquestion},
            latency_seconds=search_finished - search_started,
            output_summary={
                "text_items": retrieval.get("metadata", {}).get("text_items", 0),
                "graph_items": retrieval.get("metadata", {}).get("graph_items", 0),
                "online_signposts": count_online_signpost_recommendations(retrieval),
            },
        )
        locates = collect_locates(retrieval)
        if self.evidence_rerank:
            locates = locates[: max(self.candidate_locate_top_k, self.read_top_k)]
        else:
            locates = locates[: self.read_top_k]
        evidence = []
        for locate in locates:
            try:
                read_started = time.time()
                snippet = self.read_file_tool.run(locate)
                read_finished = time.time()
            except Exception as exc:  # pragma: no cover - kept in trace for batch robustness.
                self.trace.add("tool_error", tool=self.read_file_tool.name, input={"locate": locate}, error=str(exc))
                continue
            evidence.append(snippet)
            self.trace.add(
                "tool_call",
                tool=self.read_file_tool.name,
                input={"locate": locate},
                latency_seconds=read_finished - read_started,
                output_summary={
                    "file_name": snippet.get("file_name"),
                    "line_count": len(snippet.get("lines", [])),
                    "resolved": snippet.get("resolved"),
                },
            )
        if self.evidence_rerank:
            evidence = self._rerank_evidence(subquestion, evidence)
        else:
            evidence = evidence[: self.read_top_k]
        result = {
            "subquestion": subquestion,
            "retrieval": retrieval,
            "evidence": evidence,
            "locates": locates,
        }
        self.trace.add("researcher_finish", subquestion=subquestion, evidence_count=len(evidence))
        return result

    def _rerank_evidence(self, subquestion: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = _dedupe_snippets(evidence)
        if not candidates:
            return []
        started = time.time()
        try:
            ranked = rerank_evidence_snippets(
                query=subquestion,
                snippets=candidates,
                top_k=max(self.rerank_top_k, self.read_top_k),
            )
        except Exception as exc:
            self.trace.add("rerank_error", stage="evidence_rerank", error=str(exc), candidates=len(candidates))
            if _env_bool("SIGNPOST_ALLOW_RERANK_FALLBACK", False):
                ranked = candidates
            else:
                raise
        selected = _dedupe_snippets(ranked)[: self.read_top_k]
        selected = _limit_snippets_by_tokens(selected, self.evidence_max_tokens)
        self.trace.add(
            "tool_call",
            tool="evidence_rerank",
            input={"query": subquestion},
            latency_seconds=time.time() - started,
            output_summary={
                "candidates": len(candidates),
                "selected": len(selected),
                "rerank_top_k": self.rerank_top_k,
                "max_tokens": self.evidence_max_tokens,
            },
        )
        return selected


class Supervisor:
    def __init__(
        self,
        config: AgentConfig,
        search_tool: KnowledgeSearchTool,
        read_file_tool: ReadFileTool,
        *,
        llm: OpenAICompatibleClient | None = None,
    ):
        self.config = config
        self.trace = TraceRecorder()
        self.search_tool = search_tool
        self.read_file_tool = read_file_tool
        self.llm = llm

    def run(self, question: str) -> dict[str, Any]:
        self.trace.add("supervisor_start", question=question, namespace=self.config.namespace)
        subquestions = self.decompose(question)
        self.trace.add("plan", subquestions=subquestions)
        researcher = Researcher(self.search_tool, self.read_file_tool, self.trace, read_top_k=self.config.read_top_k, config=self.config)
        research_results = [researcher.research(subquestion) for subquestion in subquestions]
        answer = self.synthesize(question, research_results)
        citations = collect_citations(research_results)
        self.trace.add("final_answer", answer=answer, citation_count=len(citations))
        return {
            "trace_id": self.trace.trace_id,
            "namespace": self.config.namespace,
            "question": question,
            "subquestions": subquestions,
            "answer": answer,
            "citations": citations,
            "research": research_results,
            "trace": self.trace.events,
        }

    def decompose(self, question: str) -> list[str]:
        if self.config.use_llm and self.llm is not None:
            started = time.time()
            try:
                content = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Decompose the user question into at most three independently searchable "
                                "English subquestions. Output only a valid JSON array of strings. "
                                "Do not translate the final answer here."
                            ),
                        },
                        {"role": "user", "content": question},
                    ]
                )
                self.trace.add(
                    "llm_call",
                    stage="decompose",
                    latency_seconds=time.time() - started,
                    input_tokens_estimate=count_tokens(question),
                    output_tokens_estimate=count_tokens(content),
                    input_chars=len(question),
                    output_chars=len(content),
                )
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    subquestions = [str(item).strip() for item in parsed if str(item).strip()]
                    if subquestions:
                        return subquestions[: self.config.max_subquestions]
            except Exception as exc:  # pragma: no cover - network/LLM failures fall back.
                self.trace.add("llm_fallback", stage="decompose", latency_seconds=time.time() - started, error=str(exc))
        return deterministic_decompose(question, self.config.max_subquestions)

    def synthesize(self, question: str, research_results: list[dict[str, Any]]) -> str:
        if self.config.use_llm and self.llm is not None:
            evidence_text = "\n\n".join(format_evidence_block(item) for item in research_results)
            prompt_text = f"Question:\n{question}\n\n{_answer_slot_instruction(question)}Evidence:\n{evidence_text}"
            started = time.time()
            try:
                content = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": SYNTHESIS_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt_text},
                    ]
                )
                self.trace.add(
                    "llm_call",
                    stage="synthesize",
                    latency_seconds=time.time() - started,
                    input_tokens_estimate=count_tokens(prompt_text),
                    output_tokens_estimate=count_tokens(content),
                    input_chars=len(prompt_text),
                    output_chars=len(content),
                )
                return content
            except Exception as exc:  # pragma: no cover - network/LLM failures fall back.
                self.trace.add("llm_fallback", stage="synthesize", latency_seconds=time.time() - started, error=str(exc))
        return deterministic_synthesize(question, research_results)


def deterministic_decompose(question: str, max_subquestions: int = 3) -> list[str]:
    parts = [part.strip() for part in re.split(r"[？?。；;\n]+", question) if part.strip()]
    if not parts:
        return [question.strip()]
    return parts[:max_subquestions]


def collect_locates(retrieval: dict[str, Any]) -> list[str]:
    locates: list[str] = []
    for group_name in ("text_group", "graph_group"):
        for item in retrieval.get(group_name, {}).get("items", []):
            provenance = item.get("offline_signpost", {}).get("provenance", {})
            locate = provenance.get("locate")
            if locate:
                locates.append(locate)
            locates.extend(provenance.get("source_locates") or [])
            locates.extend(item.get("source_locates") or [])
    seen = set()
    unique = []
    for locate in locates:
        if locate and locate not in seen:
            unique.append(locate)
            seen.add(locate)
    return unique


def count_online_signpost_recommendations(retrieval: dict[str, Any]) -> int:
    total = 0
    for group_name in ("text_group", "graph_group"):
        online = retrieval.get(group_name, {}).get("online_signpost", {})
        total += len(online.get("recommended_entities") or [])
    return total


def collect_citations(research_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    seen = set()
    for result in research_results:
        for snippet in result.get("evidence", []):
            lines = snippet.get("lines") or []
            if not lines:
                continue
            file_name = snippet.get("file_name")
            start = lines[0]["line_no"]
            end = lines[-1]["line_no"]
            locate = f"{file_name}:L{start}-L{end}"
            if locate in seen:
                continue
            citations.append({"file_name": file_name, "start_line": start, "end_line": end, "locate": locate})
            seen.add(locate)
    return citations


def deterministic_synthesize(question: str, research_results: list[dict[str, Any]]) -> str:
    lines = [f"问题：{question}", "", "基于检索到的证据，可得到以下结论："]
    evidence_blocks = []
    for result in research_results:
        for snippet in result.get("evidence", []):
            snippet_lines = snippet.get("lines") or []
            if not snippet_lines:
                continue
            file_name = snippet.get("file_name")
            start = snippet_lines[0]["line_no"]
            end = snippet_lines[-1]["line_no"]
            text = " ".join(str(row.get("text", "")).strip() for row in snippet_lines if str(row.get("text", "")).strip())
            evidence_blocks.append(f"- {text} [{file_name}:L{start}-L{end}]")
    if evidence_blocks:
        lines.extend(evidence_blocks)
    else:
        lines.append("- 未能从当前索引中找到可回读的源文档证据。")
    return "\n".join(lines)


def format_evidence_block(result: dict[str, Any]) -> str:
    blocks = [f"子问题：{result.get('subquestion')}"]
    for snippet in result.get("evidence", []):
        blocks.append(snippet.get("file_content_view", ""))
    return "\n".join(blocks)


def rerank_evidence_snippets(*, query: str, snippets: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rerank_url = os.getenv("SIGNPOST_RERANK_URL", "").strip()
    rerank_model = os.getenv("SIGNPOST_RERANK_MODEL") or os.getenv("ECNU_RERANK_MODEL") or ""
    if not rerank_url:
        raise ValueError("SIGNPOST_RERANK_URL is required when SIGNPOST_EVIDENCE_RERANK=1")
    documents = [_snippet_text(snippet) for snippet in snippets]
    payload = {"model": rerank_model, "query": query, "documents": documents}
    request = urllib.request.Request(
        rerank_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = _env_int("SIGNPOST_RERANK_TIMEOUT", 600)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"evidence rerank request failed: {exc}") from exc

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    used_indexes = set()
    for fallback_rank, result in enumerate(data.get("results") or data.get("data") or []):
        if isinstance(result, dict):
            index = int(result.get("index", result.get("document_index", fallback_rank)) or 0)
            score = float(result.get("relevance_score", result.get("score", 0.0)) or 0.0)
        else:
            index = fallback_rank
            score = float(result or 0.0)
        if 0 <= index < len(snippets) and index not in used_indexes:
            item = {**snippets[index], "rerank_score": score}
            ranked.append((score, fallback_rank, item))
            used_indexes.add(index)
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [item for _score, _rank, item in ranked[:top_k]]
    if len(selected) < top_k:
        for index, snippet in enumerate(snippets):
            if index in used_indexes:
                continue
            selected.append(snippet)
            if len(selected) >= top_k:
                break
    return selected


def _dedupe_snippets(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    ranges_by_file: dict[str, list[tuple[int, int]]] = {}
    for snippet in snippets:
        lines = snippet.get("lines") or []
        if not lines:
            continue
        file_name = str(snippet.get("file_name") or "")
        start = int(lines[0].get("line_no") or 0)
        end = int(lines[-1].get("line_no") or start)
        ranges = ranges_by_file.setdefault(file_name, [])
        if any(_ranges_overlap_or_touch(start, end, left, right) for left, right in ranges):
            continue
        ranges.append((start, end))
        deduped.append(snippet)
    return deduped


def _limit_snippets_by_tokens(snippets: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        return snippets
    selected = []
    total = 0
    for snippet in snippets:
        token_count = count_tokens(_snippet_text(snippet))
        if selected and total + token_count > max_tokens:
            break
        selected.append(snippet)
        total += token_count
    return selected


def _snippet_text(snippet: dict[str, Any]) -> str:
    lines = snippet.get("lines") or []
    return " ".join(str(row.get("text", "")).strip() for row in lines if str(row.get("text", "")).strip())


def _ranges_overlap_or_touch(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start <= other_end + 1 and other_start <= end + 1


def _first_locate(snippet: dict[str, Any]) -> str | None:
    """Return the first locate string for a read snippet, or None."""
    lines = snippet.get("lines") or []
    if not lines:
        return None
    file_name = snippet.get("file_name")
    start = lines[0].get("line_no")
    end = lines[-1].get("line_no")
    if file_name and start is not None and end is not None:
        return f"{file_name}:L{start}-L{end}"
    return None


def _answer_slot_instruction(question: str) -> str:
    if not (_env_bool("SIGNPOST_ANSWER_SLOT_CHECK", False) or _env_bool("SIGNPOST_EVIDENCE_RERANK", False)):
        return ""
    lowered = question.lower()
    checks = [
        "Before writing the JSON answer, identify the answer slots required by the question and verify that the evidence supports each slot.",
        "If a slot is unsupported by evidence, do not fill it with outside knowledge.",
    ]
    if re.search(r"\b(who|which)\b", lowered):
        checks.append("For who/which questions, prioritize the exact entity name and role.")
    if re.search(r"\b(two|three|four|some|list|what are)\b", lowered):
        checks.append("For list questions, cover the required list items and any stated count.")
    if "benefit" in lowered and "challenge" in lowered:
        checks.append("For benefits-and-challenges questions, cover both sides separately.")
    if re.search(r"\b(how|why)\b", lowered):
        checks.append("For how/why questions, cover the action, reason, and important constraints.")
    return "Answer-slot checklist:\n- " + "\n- ".join(checks) + "\n\n"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
