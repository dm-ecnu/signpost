"""Trace log analysis tool

Reads trace.jsonl files and computes statistics.

Usage:
    # Analyze single log file
    uv run python -m core.logging.analyzer logs/task_0001/xxx/trace.jsonl

    # Analyze directory (latest timestamp for each task)
    uv run python -m core.logging.analyzer logs/graphrag-bench/

    # Output CSV
    uv run python -m core.logging.analyzer logs/graphrag-bench/ --format csv -o stats.csv
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class TraceStatistics:
    """Statistics for a single trace"""

    # Basic info
    trace_id: str = ""
    task: str = ""
    status: str = ""  # success/error/completed
    log_path: str = ""

    # Directory structure info (for --latest mode)
    dataset_name: str = ""
    task_name: str = ""

    # Time metrics
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    # Agent metrics
    total_agents: int = 0
    supervisor_count: int = 0
    researcher_count: int = 0

    # Step metrics
    total_steps: int = 0
    steps_by_agent: Dict[str, int] = field(default_factory=dict)

    # LLM call metrics
    total_llm_calls: int = 0
    llm_calls_by_agent: Dict[str, int] = field(default_factory=dict)

    # Tool call metrics
    total_tool_calls: int = 0
    tool_calls_by_name: Dict[str, int] = field(default_factory=dict)
    tool_calls_by_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)
    tool_errors: int = 0

    # Compact metrics
    total_compacts: int = 0
    total_chunks_deleted: int = 0

    # Error metrics
    total_errors: int = 0

    # Final report
    final_report_length: int = 0

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dict (for CSV output)"""
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "trace_id": self.trace_id,
            "task": self.task[:100] if self.task else "",
            "status": self.status,
            "log_path": self.log_path,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_agents": self.total_agents,
            "supervisor_count": self.supervisor_count,
            "researcher_count": self.researcher_count,
            "total_steps": self.total_steps,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "tool_knowledge_search": self.tool_calls_by_name.get("knowledge_search", 0),
            "tool_read_file": self.tool_calls_by_name.get("read_file", 0),
            "tool_get_toc": self.tool_calls_by_name.get("get_toc", 0),
            "tool_research": self.tool_calls_by_name.get("research", 0),
            "tool_research_complete": self.tool_calls_by_name.get("research_complete", 0),
            "tool_errors": self.tool_errors,
            "total_compacts": self.total_compacts,
            "total_chunks_deleted": self.total_chunks_deleted,
            "total_errors": self.total_errors,
            "final_report_length": self.final_report_length,
        }


# =============================================================================
# Core analysis logic
# =============================================================================


class TraceAnalyzer:
    """Trace log analyzer

    Design principles:
    1. Use *_end events for counting to avoid double-counting
    2. Deduplicate by event_id (prevent duplicate log writes)
    3. Per-agent statistics with hierarchical aggregation
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = TraceStatistics()
        self._seen_event_ids: set = set()
        self._agent_types: Dict[str, str] = {}

    def analyze_file(self, file_path: Path) -> TraceStatistics:
        self.reset()
        self.stats.log_path = str(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        self._process_event(event)
                    except json.JSONDecodeError as e:
                        logger.warning("Invalid JSON at line %d: %s", line_num, e)
                        continue

        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
            self.stats.status = "file_not_found"
        except Exception as e:
            logger.error("Failed to analyze %s: %s", file_path, e)
            self.stats.status = "analysis_error"

        self._finalize_stats()

        return self.stats

    def _process_event(self, event: Dict[str, Any]) -> None:
        event_id = event.get("event_id")
        event_type = event.get("event_type")

        if event_id and event_id in self._seen_event_ids:
            logger.debug("Duplicate event_id: %s", event_id)
            return
        if event_id:
            self._seen_event_ids.add(event_id)

        agent_id = event.get("agent_id")
        agent_type = event.get("agent_type")
        if agent_id and agent_type:
            self._agent_types[agent_id] = agent_type

        handler = getattr(self, f"_handle_{event_type}", None)
        if handler:
            handler(event)

    # =========================================================================
    # Event handlers
    # =========================================================================

    def _handle_trace_start(self, event: Dict[str, Any]) -> None:
        self.stats.trace_id = event.get("trace_id", "")
        self.stats.task = event.get("task", "")
        self.stats.start_time = event.get("timestamp")

    def _handle_agent_start(self, event: Dict[str, Any]) -> None:
        if "system_prompt" not in event:
            return

        agent_type = event.get("agent_type", "")

        self.stats.total_agents += 1

        if agent_type == "Supervisor":
            self.stats.supervisor_count += 1
        elif agent_type == "Researcher":
            self.stats.researcher_count += 1

    def _handle_step_end(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent_id", "unknown")

        self.stats.total_steps += 1

        if agent_id not in self.stats.steps_by_agent:
            self.stats.steps_by_agent[agent_id] = 0
        self.stats.steps_by_agent[agent_id] += 1

    def _handle_llm_call_end(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent_id", "unknown")

        self.stats.total_llm_calls += 1

        if agent_id not in self.stats.llm_calls_by_agent:
            self.stats.llm_calls_by_agent[agent_id] = 0
        self.stats.llm_calls_by_agent[agent_id] += 1

    def _handle_tool_call_end(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent_id", "unknown")
        tool_name = event.get("tool_name", "unknown")
        status = event.get("status", "success")

        self.stats.total_tool_calls += 1

        if tool_name not in self.stats.tool_calls_by_name:
            self.stats.tool_calls_by_name[tool_name] = 0
        self.stats.tool_calls_by_name[tool_name] += 1

        if agent_id not in self.stats.tool_calls_by_agent:
            self.stats.tool_calls_by_agent[agent_id] = {}
        if tool_name not in self.stats.tool_calls_by_agent[agent_id]:
            self.stats.tool_calls_by_agent[agent_id][tool_name] = 0
        self.stats.tool_calls_by_agent[agent_id][tool_name] += 1

        if status == "error":
            self.stats.tool_errors += 1

    def _handle_compact(self, event: Dict[str, Any]) -> None:
        self.stats.total_compacts += 1

        deleted_chunks = event.get("deleted_chunks", [])
        if isinstance(deleted_chunks, list):
            self.stats.total_chunks_deleted += len(deleted_chunks)

    def _handle_error(self, event: Dict[str, Any]) -> None:
        self.stats.total_errors += 1

    def _handle_final(self, event: Dict[str, Any]) -> None:
        self.stats.status = event.get("status", "")
        self.stats.end_time = event.get("timestamp")

        final_report = event.get("final_report", "")
        if final_report:
            self.stats.final_report_length = len(final_report)

    # =========================================================================
    # Statistics finalization
    # =========================================================================

    def _finalize_stats(self) -> None:
        if self.stats.start_time and self.stats.end_time:
            try:
                start = datetime.fromisoformat(self.stats.start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(self.stats.end_time.replace("Z", "+00:00"))
                self.stats.duration_seconds = (end - start).total_seconds()
            except Exception as e:
                logger.warning("Failed to parse timestamps: %s", e)


# =============================================================================
# Batch analysis
# =============================================================================


def find_trace_files(root_path: Path, recursive: bool = False) -> Iterator[Path]:
    if root_path.is_file():
        if root_path.name == "trace.jsonl" or root_path.suffix == ".jsonl":
            yield root_path
        return

    pattern = "**/trace.jsonl" if recursive else "*/trace.jsonl"
    for path in root_path.glob(pattern):
        yield path


def find_trace_files_latest(root_path: Path) -> Iterator[tuple[str, str, Path]]:
    if root_path.is_file():
        yield ("", "", root_path)
        return

    has_task_dir = any(d.is_dir() and d.name.startswith("task_") for d in root_path.iterdir())
    has_dataset_level = not has_task_dir

    if has_dataset_level:
        for dataset_dir in sorted(root_path.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            for task_dir in sorted(dataset_dir.iterdir()):
                if not task_dir.is_dir():
                    continue

                task_name = task_dir.name
                if not task_name.startswith("task_"):
                    continue

                timestamp_dirs = []
                for ts_dir in task_dir.iterdir():
                    if not ts_dir.is_dir():
                        continue
                    if len(ts_dir.name) >= 15 and ts_dir.name[8] == "_":
                        trace_file = ts_dir / "trace.jsonl"
                        if trace_file.exists():
                            timestamp_dirs.append((ts_dir.name, trace_file))

                if not timestamp_dirs:
                    logger.debug("No trace.jsonl in %s", task_dir)
                    continue

                timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
                latest_ts, latest_trace = timestamp_dirs[0]

                logger.debug("Selected %s for %s/%s", latest_ts, dataset_name, task_name)
                yield (dataset_name, task_name, latest_trace)
    else:
        for task_dir in sorted(root_path.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            if not task_name.startswith("task_"):
                continue

            timestamp_dirs = []
            for ts_dir in task_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                if len(ts_dir.name) >= 15 and ts_dir.name[8] == "_":
                    trace_file = ts_dir / "trace.jsonl"
                    if trace_file.exists():
                        timestamp_dirs.append((ts_dir.name, trace_file))

            if not timestamp_dirs:
                logger.debug("No trace.jsonl in %s", task_dir)
                continue

            timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_ts, latest_trace = timestamp_dirs[0]

            logger.debug("Selected %s for %s", latest_ts, task_name)
            yield ("", task_name, latest_trace)


def analyze_batch(
    root_path: Path,
    recursive: bool = False,
) -> List[TraceStatistics]:
    results = []
    analyzer = TraceAnalyzer()

    trace_files = list(find_trace_files(root_path, recursive))
    total = len(trace_files)

    if total == 0:
        logger.warning("No trace.jsonl files found in %s", root_path)
        return results

    logger.info("Found %d trace files", total)

    for i, file_path in enumerate(trace_files, start=1):
        logger.info("[%d/%d] Analyzing %s", i, total, file_path)
        stats = analyzer.analyze_file(file_path)
        results.append(stats)

    return results


def analyze_batch_latest(root_path: Path) -> List[TraceStatistics]:
    results = []
    analyzer = TraceAnalyzer()

    trace_entries = list(find_trace_files_latest(root_path))
    total = len(trace_entries)

    if total == 0:
        logger.warning("No trace.jsonl files found in %s", root_path)
        return results

    logger.info("Found %d latest trace files", total)

    for i, (dataset_name, task_name, file_path) in enumerate(trace_entries, start=1):
        logger.info("[%d/%d] Analyzing %s/%s", i, total, dataset_name, task_name)
        stats = analyzer.analyze_file(file_path)
        stats.dataset_name = dataset_name
        stats.task_name = task_name
        results.append(stats)

    return results


def aggregate_stats(stats_list: List[TraceStatistics]) -> Dict[str, Any]:
    if not stats_list:
        return {}

    total_traces = len(stats_list)
    success_count = sum(1 for s in stats_list if s.status in ("success", "completed"))
    error_count = sum(1 for s in stats_list if s.status == "error")

    total_steps = sum(s.total_steps for s in stats_list)
    total_llm_calls = sum(s.total_llm_calls for s in stats_list)
    total_tool_calls = sum(s.total_tool_calls for s in stats_list)
    total_compacts = sum(s.total_compacts for s in stats_list)
    total_errors = sum(s.total_errors for s in stats_list)
    total_duration = sum(s.duration_seconds for s in stats_list)

    tool_calls_summary: Dict[str, int] = {}
    for s in stats_list:
        for tool_name, count in s.tool_calls_by_name.items():
            if tool_name not in tool_calls_summary:
                tool_calls_summary[tool_name] = 0
            tool_calls_summary[tool_name] += count

    avg_steps = total_steps / total_traces if total_traces > 0 else 0
    avg_llm_calls = total_llm_calls / total_traces if total_traces > 0 else 0
    avg_tool_calls = total_tool_calls / total_traces if total_traces > 0 else 0
    avg_duration = total_duration / total_traces if total_traces > 0 else 0

    avg_tool_calls_summary: Dict[str, float] = {}
    for tool_name, count in tool_calls_summary.items():
        avg_tool_calls_summary[tool_name] = round(count / total_traces, 2)

    return {
        "total_traces": total_traces,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / total_traces * 100 if total_traces > 0 else 0,
        "total_steps": total_steps,
        "total_llm_calls": total_llm_calls,
        "total_tool_calls": total_tool_calls,
        "total_compacts": total_compacts,
        "total_errors": total_errors,
        "total_duration_seconds": round(total_duration, 2),
        "avg_steps_per_trace": round(avg_steps, 2),
        "avg_llm_calls_per_trace": round(avg_llm_calls, 2),
        "avg_tool_calls_per_trace": round(avg_tool_calls, 2),
        "avg_duration_seconds": round(avg_duration, 2),
        "tool_calls_summary": tool_calls_summary,
        "avg_tool_calls_summary": avg_tool_calls_summary,
    }


# =============================================================================
# Output formatting
# =============================================================================


def print_stats(stats: TraceStatistics, verbose: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Trace: {stats.trace_id}")
    print(f"Task: {stats.task[:80]}..." if len(stats.task) > 80 else f"Task: {stats.task}")
    print(f"Status: {stats.status}")
    print(f"Duration: {stats.duration_seconds:.2f}s")
    print(f"{'='*60}")

    print(f"\nAgents: {stats.total_agents} (Supervisor: {stats.supervisor_count}, Researcher: {stats.researcher_count})")
    print(f"Steps: {stats.total_steps}")
    print(f"LLM Calls: {stats.total_llm_calls}")
    print(f"Tool Calls: {stats.total_tool_calls}")

    if stats.tool_calls_by_name:
        print("\nTool Calls Breakdown:")
        for tool_name, count in sorted(stats.tool_calls_by_name.items()):
            print(f"  - {tool_name}: {count}")

    if stats.total_compacts > 0:
        print(f"\nCompacts: {stats.total_compacts} (chunks deleted: {stats.total_chunks_deleted})")

    if stats.total_errors > 0:
        print(f"\nErrors: {stats.total_errors}")

    if verbose and stats.tool_calls_by_agent:
        print("\nTool Calls by Agent:")
        for agent_id, tools in stats.tool_calls_by_agent.items():
            print(f"  {agent_id}:")
            for tool_name, count in tools.items():
                print(f"    - {tool_name}: {count}")


def print_aggregate(agg: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")

    print(f"\nTotal Traces: {agg['total_traces']}")
    print(f"Success: {agg['success_count']} ({agg['success_rate']:.1f}%)")
    print(f"Errors: {agg['error_count']}")

    print(f"\nTotal Duration: {agg['total_duration_seconds']:.2f}s")
    print(f"Avg Duration: {agg['avg_duration_seconds']:.2f}s")

    print(f"\nTotal Steps: {agg['total_steps']} (avg: {agg['avg_steps_per_trace']:.2f})")
    print(f"Total LLM Calls: {agg['total_llm_calls']} (avg: {agg['avg_llm_calls_per_trace']:.2f})")
    print(f"Total Tool Calls: {agg['total_tool_calls']} (avg: {agg['avg_tool_calls_per_trace']:.2f})")

    if agg.get("tool_calls_summary"):
        print("\nTool Calls Summary:")
        avg_summary = agg.get("avg_tool_calls_summary", {})
        for tool_name, count in sorted(agg["tool_calls_summary"].items()):
            avg = avg_summary.get(tool_name, 0)
            print(f"  - {tool_name}: {count} (avg: {avg})")

    if agg["total_compacts"] > 0:
        print(f"\nTotal Compacts: {agg['total_compacts']}")

    if agg["total_errors"] > 0:
        print(f"\nTotal Errors: {agg['total_errors']}")


def write_csv(stats_list: List[TraceStatistics], output_path: Path) -> None:
    if not stats_list:
        logger.warning("No stats to write")
        return

    fieldnames = list(stats_list[0].to_flat_dict().keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(stats.to_flat_dict())

    logger.info("CSV written to %s", output_path)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trace.jsonl log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single file
    uv run python -m core.logging.analyzer logs/task_0001/xxx/trace.jsonl

    # Analyze directory (latest timestamp for each task, default mode)
    uv run python -m core.logging.analyzer logs/graphrag-bench/

    # Output CSV
    uv run python -m core.logging.analyzer logs/graphrag-bench/ --format csv -o stats.csv
        """,
    )

    parser.add_argument("path", type=Path, help="Path to trace.jsonl file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "csv", "json"], default="text", help="Output format")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (for csv/json)")
    parser.add_argument("--aggregate-only", action="store_true", help="Only show aggregate stats")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.path.is_file():
        stats_list = analyze_batch(args.path, recursive=False)
    else:
        stats_list = analyze_batch_latest(args.path)

    if not stats_list:
        print("No trace files found or analyzed.")
        sys.exit(1)

    if args.format == "csv":
        output_path = args.output or Path("trace_stats.csv")
        write_csv(stats_list, output_path)
        print(f"CSV written to {output_path}")

    elif args.format == "json":
        output_path = args.output or Path("trace_stats.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stats": [asdict(s) for s in stats_list],
                    "aggregate": aggregate_stats(stats_list),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"JSON written to {output_path}")

    else:  # text
        if not args.aggregate_only:
            for stats in stats_list:
                print_stats(stats, args.verbose)

        if len(stats_list) > 1 or args.aggregate_only:
            agg = aggregate_stats(stats_list)
            print_aggregate(agg)


if __name__ == "__main__":
    main()
