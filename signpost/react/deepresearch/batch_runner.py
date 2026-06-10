"""批量研究任务执行器

简化设计：函数式接口，无需复杂的类封装。

使用方式:
    # 1. 命令行运行（指定数据集名称即可）
    uv run python -m deepresearch.batch_runner --dataset legal
    uv run python -m deepresearch.batch_runner --dataset mix --workers 4
    uv run python -m deepresearch.batch_runner -d agriculture -w 8

    # 可用数据集: graphrag-bench, agriculture, legal, mix

    # 2. 代码调用
    from deepresearch.batch_runner import run_batch, get_dataset_paths, TENANT_ID

    input_path, output_path, kb_id = get_dataset_paths("legal")
    for result in run_batch(
        questions=["问题1", "问题2"],
        kb_id=kb_id,
        tenant_id=TENANT_ID,
    ):
        print(result)
"""

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .configuration import Configuration
from .events import EventType
from .supervisor import Supervisor
from core.logging.trace import TraceSession
from .types import TraceStatus

logger = logging.getLogger(__name__)

# ==============================================================================
# 数据集配置
# ==============================================================================

DATASETS_ROOT = os.getenv("DEEPRESEARCH_DATASETS_ROOT", "datasets")
TENANT_ID = os.getenv("DEEPRESEARCH_TENANT_ID", "")

DATASETS = {
    "graphrag-bench": {
        "kb_id_env": "DEEPRESEARCH_KB_ID_GRAPHRAG_BENCH",
        "input_dir": "GraphRAG-Bench",
    },
    "agriculture": {
        "kb_id_env": "DEEPRESEARCH_KB_ID_AGRICULTURE",
        "input_dir": "agriculture",
    },
    "legal": {
        "kb_id_env": "DEEPRESEARCH_KB_ID_LEGAL",
        "input_dir": "legal",
    },
    "mix": {
        "kb_id_env": "DEEPRESEARCH_KB_ID_MIX",
        "input_dir": "mix",
    },
}


def get_dataset_paths(dataset_name: str) -> tuple[str, str, str, str]:
    """获取数据集路径和 kb_id

    Args:
        dataset_name: 数据集名称 (graphrag-bench, agriculture, legal, mix)

    Returns:
        (input_path, output_path, kb_id, log_root)
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset_name]
    datasets_root = os.getenv("DEEPRESEARCH_DATASETS_ROOT", DATASETS_ROOT)
    kb_id = os.getenv(cfg["kb_id_env"], "")
    if not kb_id:
        raise ValueError(f"Missing KB ID for dataset '{dataset_name}'. Set environment variable {cfg['kb_id_env']}.")

    input_path = f"{datasets_root}/{cfg['input_dir']}/Question.jsonl"
    output_path = f"eval/agentic_rag/{dataset_name}.jsonl"
    log_root = f"logs/{dataset_name}"
    return input_path, output_path, kb_id, log_root


def _init_global_dependencies() -> None:
    """初始化全局依赖（kg_retrievaler、docStoreConn）"""
    from api.settings import init_settings, kg_retrievaler, docStoreConn

    if kg_retrievaler is None or docStoreConn is None:
        logger.info("Initializing API settings...")
        init_settings()

    from api import settings

    if settings.kg_retrievaler is None:
        raise RuntimeError("kg_retrievaler is None after init_settings()")


def _run_single_task(
    index: int,
    question: str,
    original_input: Dict[str, Any],
    config: Configuration,
    log_root: Path,
    total_tasks: int = 0,
    print_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    """执行单个研究任务

    Returns:
        dict: 包含原始输入字段 + prediction + _status + _duration_seconds
    """
    # 打印启动信息
    if print_lock is not None:
        preview = question[:50].replace('\n', ' ') + ("..." if len(question) > 50 else "")
        with print_lock:
            print(f"[START] Task {index}/{total_tasks}: {preview}", flush=True)

    start_time = time.time()

    trace_session = TraceSession(
        task=question[:100],
        log_root=log_root / f"task_{index:04d}",
    )

    errors: List[str] = []

    try:
        if trace_session.is_available:
            trace_session.emit_trace_start(
                config={
                    "batch_index": index,
                    "kb_id": config.kb_id,
                    "model_id": config.model_id,
                    "question_preview": question[:200],
                }
            )

        supervisor = Supervisor(
            config=config,
            task=question,
            parent_trace_session=trace_session,
        )

        final_report = ""
        for event in supervisor(question):
            if event.event_type == EventType.RESEARCH_COMPLETED:
                final_report = event.content
            elif event.event_type == EventType.ERROR_OCCURRED:
                errors.append(f"{event.error_type}: {event.error_message}")
                logger.warning("Task %d error: %s", index, errors[-1])

        # 写入 metadata
        if trace_session.is_available and supervisor.trace_emitter:
            try:
                supervisor.trace_emitter.write_metadata()
                supervisor.trace_emitter.write_final_output(final_report)
            except Exception as e:
                logger.warning("Failed to write metadata for task %d: %s", index, e)

        # 判断成功/失败
        has_errors = len(errors) > 0
        has_empty_report = not final_report or not final_report.strip()
        status = "error" if (has_errors or has_empty_report) else "success"

        if trace_session.is_available:
            stats = trace_session.collect_statistics()
            trace_session.emit_final(
                status=TraceStatus.ERROR if status == "error" else TraceStatus.COMPLETED,
                final_report=final_report or f"Error: {'; '.join(errors) or 'Empty output'}",
                total_steps=stats.get("total_steps", 0),
                total_tool_calls=stats.get("total_tool_calls", 0),
                total_llm_calls=stats.get("total_llm_calls", 0),
            )

        duration = time.time() - start_time
        logger.info("Task %d %s in %.2fs", index, status, duration)

        # 构建输出：原始字段 + 新字段
        output = dict(original_input)
        output["prediction"] = final_report if final_report else ""
        output["_status"] = status
        output["_duration_seconds"] = round(duration, 2)
        if trace_session.is_available:
            output["_log_dir"] = str(trace_session.log_dir)

        return output

    except Exception as e:
        logger.error("Task %d failed: %s", index, e, exc_info=True)

        if trace_session.is_available:
            trace_session.emit_final(
                status=TraceStatus.ERROR,
                final_report=f"Error: {e}",
                total_steps=0,
                total_tool_calls=0,
                total_llm_calls=0,
            )

        duration = time.time() - start_time
        output = dict(original_input)
        output["prediction"] = ""
        output["_status"] = "error"
        output["_duration_seconds"] = round(duration, 2)
        return output

    finally:
        try:
            trace_session.close()
        except Exception:
            pass


def run_batch(
    questions: List[str],
    kb_id: str,
    tenant_id: str,
    original_inputs: Optional[List[Dict[str, Any]]] = None,
    # Model config (defaults from configuration.py)
    model_id: str = "qwen-plus-thinking",
    api_base: str = "",
    api_key: str = "",
    max_tokens: int = 32000,
    # Execution config
    max_workers: int = 8,
    log_root: str = "logs/batch",
    # Agent config (defaults from configuration.py)
    max_researcher_iterations: int = 32,
    max_react_tool_calls: int = 16,
    max_context_length: int = 131000,
    prompt_language: str = "en",
) -> Iterator[Dict[str, Any]]:
    """批量执行研究任务

    Args:
        questions: 问题列表
        kb_id: 知识库 ID
        tenant_id: 租户 ID
        original_inputs: 原始输入字典列表（用于保留额外字段）
        model_id: 模型 ID
        api_base: API 地址
        api_key: API Key
        max_tokens: 最大 token 数
        max_workers: 并发数
        log_root: 日志根目录
        max_researcher_iterations: Researcher 最大迭代次数
        max_react_tool_calls: ReAct 最大工具调用次数
        max_context_length: 最大上下文长度
        prompt_language: 提示词语言 (en/zh)

    Yields:
        dict: 每个任务的结果（包含原始字段 + prediction + _status）
    """
    if not questions:
        logger.warning("Empty question list")
        return

    # 初始化依赖
    _init_global_dependencies()

    # 准备 original_inputs
    if original_inputs is None:
        original_inputs = [{"question": q} for q in questions]
    elif len(original_inputs) != len(questions):
        raise ValueError(f"original_inputs length mismatch: {len(original_inputs)} != {len(questions)}")

    # 创建配置
    config = Configuration(
        kb_id=kb_id,
        tenant_id=tenant_id,
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        max_researcher_iterations=max_researcher_iterations,
        max_react_tool_calls=max_react_tool_calls,
        max_context_length=max_context_length,
        log_root_path=log_root,
        prompt_language=prompt_language,
    )

    log_path = Path(log_root)
    total_tasks = len(questions)
    print(f"\n{'='*60}")
    print(f"Starting batch: {total_tasks} tasks, {max_workers} workers")
    print(f"{'='*60}\n")

    # 统计
    stats = {"completed": 0, "failed": 0}
    stats_lock = threading.Lock()
    print_lock = threading.Lock()

    def print_progress(msg: str):
        """线程安全的打印"""
        with print_lock:
            print(msg, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, (q, orig) in enumerate(zip(questions, original_inputs)):
            future = executor.submit(_run_single_task, idx, q, orig, config, log_path, total_tasks, print_lock)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                with stats_lock:
                    if result.get("_status") == "success":
                        stats["completed"] += 1
                    else:
                        stats["failed"] += 1
                    done = stats["completed"] + stats["failed"]

                status = result.get("_status", "unknown")
                duration = result.get("_duration_seconds", 0)
                icon = "OK" if status == "success" else "ERR"
                print_progress(f"[{icon}] Task {idx} done in {duration:.1f}s | Progress: {done}/{total_tasks} ({stats['completed']} ok, {stats['failed']} err)")
                yield result
            except Exception as e:
                logger.error("Task %d exception: %s", idx, e, exc_info=True)
                with stats_lock:
                    stats["failed"] += 1
                    done = stats["completed"] + stats["failed"]
                print_progress(f"[ERR] Task {idx} exception | Progress: {done}/{total_tasks} ({stats['completed']} ok, {stats['failed']} err)")
                output = dict(original_inputs[idx])
                output["prediction"] = ""
                output["_status"] = "error"
                output["_duration_seconds"] = 0
                yield output

    print(f"\n{'='*60}")
    print(f"Batch completed: {stats['completed']} success, {stats['failed']} failed")
    print(f"{'='*60}\n")


def run_batch_from_jsonl(
    input_path: str,
    output_path: str,
    kb_id: str,
    tenant_id: str,
    limit: Optional[int] = None,
    **kwargs,
) -> None:
    """从 JSONL 文件运行批量任务

    输入格式 (每行一个 JSON):
        {"question": "问题1", "id": 1, ...}
        {"question": "问题2", "id": 2, ...}

    输出格式 (每行一个 JSON):
        {"question": "问题1", "id": 1, "prediction": "答案...", "_status": "success", ...}

    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        kb_id: 知识库 ID
        tenant_id: 租户 ID
        limit: 限制处理的问题数量（None = 处理全部）
        **kwargs: 传递给 run_batch 的其他参数
    """
    questions = []
    original_inputs = []

    # 读取输入
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    questions.append(item.get("question", ""))
                    original_inputs.append(item)
                else:
                    logger.warning("Line %d: expected dict, got %s", line_num, type(item).__name__)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

            # 应用限制
            if limit is not None and len(questions) >= limit:
                logger.info("Reached limit of %d questions, stopping read", limit)
                break

    if not questions:
        raise ValueError("No valid questions found")

    logger.info("Loaded %d questions from %s", len(questions), input_path)

    # 执行并收集结果
    results = list(run_batch(
        questions=questions,
        kb_id=kb_id,
        tenant_id=tenant_id,
        original_inputs=original_inputs,
        **kwargs,
    ))

    # 按原始顺序排序（通过 id 或 index）
    # 注意：results 顺序可能与输入不同，需要恢复
    id_to_result = {}
    for r in results:
        # 尝试用 id 字段，否则用 question 作为 key
        key = r.get("id", r.get("question", ""))
        id_to_result[key] = r

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 写入输出（保持输入顺序）
    with open(output_path, "w", encoding="utf-8") as f:
        for orig in original_inputs:
            key = orig.get("id", orig.get("question", ""))
            result = id_to_result.get(key, orig)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("Results saved to %s", output_path)


def main():
    """CLI 入口

    使用方式:
        uv run python -m deepresearch.batch_runner --dataset legal
        uv run python -m deepresearch.batch_runner --dataset mix --workers 4
    """
    parser = argparse.ArgumentParser(
        description="Batch research task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS.keys()), help="Dataset name")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--model", default="qwen-plus-thinking", help="Model ID (default: qwen-plus-thinking)")
    parser.add_argument("--language", default="en", choices=["en", "zh"], help="Prompt language (default: en)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions to process (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 获取数据集配置
    input_path, output_path, kb_id, log_root = get_dataset_paths(args.dataset)

    logger.info("Dataset: %s", args.dataset)
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Log root: %s", log_root)
    logger.info("KB ID: %s", kb_id)

    run_batch_from_jsonl(
        input_path=input_path,
        output_path=output_path,
        kb_id=kb_id,
        tenant_id=TENANT_ID,
        max_workers=args.workers,
        model_id=args.model,
        log_root=log_root,
        prompt_language=args.language,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
