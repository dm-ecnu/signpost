"""CLI入口模块 - 为命令行用户提供友好的交互体验"""

import sys
import logging
from pathlib import Path
from typing import Optional

from .configuration import Configuration
from .events import EventType
from .supervisor import Supervisor
from core.logging.trace import TraceSession
from .types import TraceStatus

logger = logging.getLogger(__name__)


def run_research_cli(
    task: str,
    config: Configuration,
    output_file: Optional[str] = None,
) -> str:
    """CLI entry: run research and print to terminal in real-time

    Args:
        task: Research task description
        config: Configuration object (kb_id, tenant_id, model config, etc.)
        output_file: Optional output file path

    Returns:
        str: Final research report

    Example:
        >>> config = Configuration(kb_id="test_kb", tenant_id="test_tenant")
        >>> report = run_research_cli(task="Analyze GraphRAG principles", config=config)
    """
    # 修复 HIGH-1: 确保 settings 已初始化
    try:
        from api.settings import init_settings

        init_settings()
        logger.info("API settings initialized successfully")
    except Exception as e:
        logger.warning("Failed to initialize API settings: %s. Continuing with default configuration.", str(e))

    try:
        # 创建 TraceSession（强制日志，无条件创建）
        # TraceSession 内部会处理初始化失败，不会抛出异常
        trace_session = TraceSession(task=task, log_root=Path(config.log_root_path))
        if trace_session.is_available:
            logger.info("Trace logging enabled: log_dir=%s", trace_session.log_dir)
            # 发出 trace_start 事件（记录完整配置）
            trace_session.emit_trace_start(
                config={
                    "kb_id": config.kb_id,
                    "tenant_id": config.tenant_id,
                    "model_id": config.model_id,
                    "language": config.prompt_language,
                    "max_researcher_iterations": config.max_researcher_iterations,
                    "max_react_tool_calls": config.max_react_tool_calls,
                    "max_parallel_tools": config.max_parallel_tools,
                    "max_context_length": config.max_context_length,
                    "enable_context_compress": config.enable_context_compress,
                    "context_check_threshold": config.context_check_threshold,
                }
            )
        else:
            logger.warning("Trace logging unavailable: initialization failed")

        # 创建Supervisor Agent
        supervisor = Supervisor(
            config=config,
            task=task,
            parent_trace_session=trace_session,
        )

        print("[系统] 初始化完成，开始研究任务...", file=sys.stderr)
        print(f"[系统] 知识库ID: {config.kb_id}", file=sys.stderr)
        print(f"[系统] 租户ID: {config.tenant_id}", file=sys.stderr)
        print(f"[系统] 模型: {config.model_id}", file=sys.stderr)
        if trace_session.is_available:
            print(f"[系统] 日志目录: {trace_session.log_dir}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        # 累积最终报告
        final_report = ""
        current_step = 0

        # 消费流式事件
        for event in supervisor(task):
            # 1. LLM内容增量 - 实时显示
            if event.event_type == EventType.LLM_CONTENT_DELTA:
                print(event.content, end="", flush=True)

            # 2. LLM内容完成 - 换行
            elif event.event_type == EventType.LLM_CONTENT_DONE:
                print()  # 换行

            # 3. 研究完成 - 保存最终报告
            elif event.event_type == EventType.RESEARCH_COMPLETED:
                final_report = event.content
                print("\n[系统] 研究完成！", file=sys.stderr)

            # 4. 详细信息
            elif event.event_type == EventType.AGENT_STEP_STARTED:
                current_step = event.step_number
                print(f"\n[Agent] 第 {current_step} 步开始...", file=sys.stderr)

            elif event.event_type == EventType.AGENT_STEP_COMPLETED:
                print(f"[Agent] 第 {event.step_number} 步完成", file=sys.stderr)

            elif event.event_type == EventType.TOOL_EXECUTION_STARTED:
                print(f"[工具] {event.tool_name} 执行中...", file=sys.stderr)

            elif event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
                output_preview = (event.tool_output or "")[:100]
                print(f"[工具] {event.tool_name} 完成（输出: {output_preview}...）", file=sys.stderr)

            elif event.event_type == EventType.LLM_TOOL_CALL_DONE:
                print(f"[工具调用] {event.tool_name}({event.tool_arguments[:50]}...)", file=sys.stderr)

            elif event.event_type == EventType.ERROR_OCCURRED:
                print(f"[错误] {event.error_type}: {event.error_message}", file=sys.stderr)

        # 写入 Supervisor metadata（兼容方法）
        if trace_session.is_available and supervisor.trace_emitter:
            try:
                supervisor.trace_emitter.write_metadata()
                supervisor.trace_emitter.write_final_output(final_report)
            except Exception as e:
                logger.warning("Failed to write supervisor metadata: %s", str(e))

        # 如果指定了输出文件，写入报告
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_report)
                print(f"\n[系统] 报告已保存到: {output_file}", file=sys.stderr)
            except Exception as e:
                logger.error("Failed to write output file: path=%s, error=%s", output_file, str(e), exc_info=True)
                print(f"\n[错误] 无法写入输出文件: {e}", file=sys.stderr)

        # 发出 final 事件（替代旧的 write_manifest）
        if trace_session.is_available:
            try:
                # 收集统计信息
                stats = trace_session.collect_statistics()
                trace_session.emit_final(
                    status=TraceStatus.COMPLETED,
                    final_report=final_report,
                    total_steps=stats.get("total_steps", 0),
                    total_tool_calls=stats.get("total_tool_calls", 0),
                    total_llm_calls=stats.get("total_llm_calls", 0),
                )
                print(f"\n[系统] 执行日志已保存到: {trace_session.log_dir}", file=sys.stderr)
            except Exception as e:
                logger.warning("Failed to emit final trace event: %s", str(e))

        return final_report

    except KeyboardInterrupt:
        print("\n\n[系统] 用户中断", file=sys.stderr)
        raise

    except Exception as e:
        logger.error("CLI execution failed: %s", str(e), exc_info=True)
        print(f"\n[错误] 运行失败: {e}", file=sys.stderr)
        raise
