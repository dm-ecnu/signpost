"""单问题调试运行器

用于调试 deepresearch_v2 的输出，逐步观察每个事件。

使用方式:
    # 运行 legal 数据集的第 1 个问题（索引从 0 开始）
    uv run python -m deepresearch.debug_runner --dataset legal

    # 运行第 5 个问题
    uv run python -m deepresearch.debug_runner --dataset legal --index 4

    # 运行所有数据集的第 1 个问题
    uv run python -m deepresearch.debug_runner --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .batch_runner import DATASETS, DATASETS_ROOT, TENANT_ID, get_dataset_paths
from .configuration import Configuration
from .events import EventType
from .supervisor import Supervisor
from core.logging.trace import TraceSession

# 配置彩色输出
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def cprint(text: str, color: str = "", bold: bool = False) -> None:
    """彩色打印"""
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if color:
        prefix += color
    suffix = Colors.ENDC if (color or bold) else ""
    print(f"{prefix}{text}{suffix}")


def load_question(dataset_name: str, index: int = 0) -> tuple[Dict[str, Any], str, str]:
    """加载指定数据集的指定问题

    Returns:
        (item, kb_id, log_root)
    """
    input_path, _, kb_id, log_root = get_dataset_paths(dataset_name)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if index >= len(lines):
        raise ValueError(f"Index {index} out of range (total: {len(lines)})")

    item = json.loads(lines[index])
    return item, kb_id, log_root


def run_debug(
    dataset_name: str,
    index: int = 0,
    model_id: str = "qwen-plus-thinking",
    verbose: bool = False,
) -> Dict[str, Any]:
    """运行单个问题并详细输出调试信息"""

    # 初始化依赖
    from api.settings import init_settings, kg_retrievaler
    if kg_retrievaler is None:
        cprint("Initializing API settings...", Colors.DIM)
        init_settings()

    # 加载问题
    item, kb_id, log_root = load_question(dataset_name, index)
    question = item.get("question", "")
    gold_answer = item.get("answer", "")
    item_id = item.get("id", index)

    cprint("=" * 80, Colors.BLUE, bold=True)
    cprint(f"Dataset: {dataset_name} | Index: {index} | ID: {item_id}", Colors.BLUE, bold=True)
    cprint("=" * 80, Colors.BLUE, bold=True)
    print()

    cprint("QUESTION:", Colors.CYAN, bold=True)
    print(question)
    print()

    if gold_answer:
        cprint("GOLD ANSWER:", Colors.YELLOW, bold=True)
        print(gold_answer)
        print()

    # 创建配置 (defaults from configuration.py, prompt_language=en)
    config = Configuration(
        kb_id=kb_id,
        tenant_id=TENANT_ID,
        model_id=model_id,
        max_tokens=32000,
        max_researcher_iterations=32,
        max_react_tool_calls=16,
        max_context_length=131000,
        log_root_path=f"{log_root}/debug",
        prompt_language="en",
    )

    # 创建 trace session
    trace_session = TraceSession(
        task=question[:100],
        log_root=Path(f"{log_root}/debug/task_{index:04d}"),
    )

    cprint("-" * 80, Colors.DIM)
    cprint("EXECUTION LOG:", Colors.GREEN, bold=True)
    cprint("-" * 80, Colors.DIM)

    start_time = time.time()
    final_report = ""
    errors = []

    # 事件计数
    event_counts = {}

    try:
        supervisor = Supervisor(
            config=config,
            task=question,
            parent_trace_session=trace_session,
        )

        for event in supervisor(question):
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)

            # 统计事件
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # 根据事件类型输出
            if event.event_type == EventType.RESEARCH_COMPLETED:
                final_report = event.content
                cprint(f"\n[{event_type}]", Colors.GREEN, bold=True)

            elif event.event_type == EventType.ERROR_OCCURRED:
                error_msg = f"{event.error_type}: {event.error_message}"
                errors.append(error_msg)
                cprint(f"\n[{event_type}] {error_msg}", Colors.RED, bold=True)

            elif event.event_type == EventType.TOOL_EXECUTION_STARTED:
                tool_name = getattr(event, 'tool_name', 'unknown')
                cprint(f"\n[TOOL_START] {tool_name}", Colors.CYAN)

            elif event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
                tool_name = getattr(event, 'tool_name', 'unknown')
                tool_output = getattr(event, 'tool_output', '')
                output_preview = tool_output[:200] + "..." if len(tool_output) > 200 else tool_output
                cprint(f"[TOOL_END] {tool_name}", Colors.CYAN)
                if verbose:
                    cprint(f"  Output: {output_preview}", Colors.DIM)

            elif event.event_type == EventType.SUB_RESEARCH_STARTED:
                topic = getattr(event, 'topic', '')
                researcher_id = getattr(event, 'researcher_id', '')
                cprint(f"\n[SUB_RESEARCH] {researcher_id}: {topic}", Colors.YELLOW)

            elif event.event_type == EventType.LLM_CONTENT_DELTA:
                # 流式输出 LLM 响应
                chunk = getattr(event, 'delta', getattr(event, 'content', ''))
                if chunk:
                    print(chunk, end="", flush=True)

            elif event.event_type == EventType.AGENT_STEP_STARTED:
                step_num = getattr(event, 'step_number', 0)
                cprint(f"\n[STEP {step_num}]", Colors.BLUE, bold=True)

            elif event.event_type == EventType.AGENT_STEP_COMPLETED:
                pass  # 静默

            elif verbose:
                # 其他事件（仅 verbose 模式）
                cprint(f"[{event_type}]", Colors.DIM)

        # 写入 metadata
        if trace_session.is_available and supervisor.trace_emitter:
            supervisor.trace_emitter.write_metadata()
            supervisor.trace_emitter.write_final_output(final_report)

    except Exception as e:
        cprint(f"\n[EXCEPTION] {e}", Colors.RED, bold=True)
        errors.append(str(e))
        import traceback
        traceback.print_exc()

    finally:
        trace_session.close()

    duration = time.time() - start_time

    # 输出结果
    print()
    cprint("=" * 80, Colors.BLUE, bold=True)
    cprint("EXECUTION SUMMARY", Colors.BLUE, bold=True)
    cprint("=" * 80, Colors.BLUE, bold=True)

    cprint(f"Duration: {duration:.2f}s", Colors.GREEN)
    cprint(f"Status: {'ERROR' if errors else 'SUCCESS'}", Colors.RED if errors else Colors.GREEN, bold=True)

    if event_counts:
        cprint("\nEvent counts:", Colors.CYAN)
        for evt, cnt in sorted(event_counts.items()):
            print(f"  {evt}: {cnt}")

    if errors:
        cprint("\nErrors:", Colors.RED)
        for err in errors:
            print(f"  - {err}")

    print()
    cprint("=" * 80, Colors.BLUE, bold=True)
    cprint("FINAL REPORT", Colors.BLUE, bold=True)
    cprint("=" * 80, Colors.BLUE, bold=True)

    if final_report:
        print(final_report)
    else:
        cprint("(No report generated)", Colors.RED)

    # 对比 gold answer
    if gold_answer:
        print()
        cprint("=" * 80, Colors.YELLOW, bold=True)
        cprint("COMPARISON WITH GOLD ANSWER", Colors.YELLOW, bold=True)
        cprint("=" * 80, Colors.YELLOW, bold=True)
        cprint("Gold:", Colors.YELLOW)
        print(gold_answer)
        print()
        cprint("Prediction (first 500 chars):", Colors.GREEN)
        print(final_report[:500] if final_report else "(empty)")

    return {
        "dataset": dataset_name,
        "index": index,
        "id": item_id,
        "question": question,
        "gold_answer": gold_answer,
        "prediction": final_report,
        "status": "error" if errors else "success",
        "errors": errors,
        "duration_seconds": round(duration, 2),
        "event_counts": event_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Debug runner for deepresearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument("--dataset", "-d", choices=list(DATASETS.keys()), help="Dataset name")
    parser.add_argument("--index", "-i", type=int, default=0, help="Question index (0-based, default: 0)")
    parser.add_argument("--all", "-a", action="store_true", help="Run first question from all datasets")
    parser.add_argument("--model", default="qwen-plus-thinking", help="Model ID (default: qwen-plus-thinking)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # 减少日志噪音
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.all:
        # 运行所有数据集的第一个问题
        results = []
        for dataset in DATASETS.keys():
            cprint(f"\n\n{'#' * 80}", Colors.HEADER, bold=True)
            cprint(f"# DATASET: {dataset}", Colors.HEADER, bold=True)
            cprint(f"{'#' * 80}\n", Colors.HEADER, bold=True)

            result = run_debug(
                dataset_name=dataset,
                index=args.index,
                model_id=args.model,
                verbose=args.verbose,
            )
            results.append(result)

        # 汇总
        print()
        cprint("=" * 80, Colors.HEADER, bold=True)
        cprint("ALL DATASETS SUMMARY", Colors.HEADER, bold=True)
        cprint("=" * 80, Colors.HEADER, bold=True)

        for r in results:
            status_color = Colors.GREEN if r["status"] == "success" else Colors.RED
            cprint(f"{r['dataset']}: {r['status']} ({r['duration_seconds']:.2f}s)", status_color)

    elif args.dataset:
        run_debug(
            dataset_name=args.dataset,
            index=args.index,
            model_id=args.model,
            verbose=args.verbose,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
