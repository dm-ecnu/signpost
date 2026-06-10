"""ReActAgent基类和工具函数"""

import json
import logging
import queue
import threading
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterator, List, Optional, TYPE_CHECKING

from .entities import AgentMemory, Message, MessageRole, ToolCall, ToolCallFunction
from .events import (
    DeepResearchEvent,
    EventType,
    AgentStepStartedEvent,
    AgentStepCompletedEvent,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
)
from core.llm.core import LLMCore

from .model_client import stream_events
from .tools import Tool, ToolExecutionContext
from .types import ExitReason, ContextDecision

if TYPE_CHECKING:
    from .configuration import Configuration
    from core.logging.trace import TraceEmitter, TraceSession

logger = logging.getLogger(__name__)


# ===== 工具执行辅助数据结构 =====


@dataclass
class ToolExecutionResult:
    """工具执行结果（线程安全传递用）"""

    tool_call: ToolCall
    result: Optional[str]
    error: Optional[Exception]
    log_path: Optional[str]


@dataclass
class LLMResponseResult:
    """LLM 响应结果"""

    content: str
    tool_calls: Optional[List[ToolCall]]
    start_time: datetime
    end_time: datetime
    reasoning_content: Optional[str] = None  # DeepSeek Reasoner Thinking Mode


class ReActAgent(ABC):
    """ReAct Agent基类（极简版，基于简化的数据结构）

    设计原则：
    1. 完全独立于smolagents
    2. 使用OpenAI SDK + 自定义Agent Loop
    3. 只提供流式输出接口（无非流式run方法）
    4. 业务逻辑直接写在run_stream()中（无callback机制）
    5. 不使用ActionStep，直接管理messages
    6. 使用ToolExecutionContext进行显式依赖注入（消除hasattr）
    """

    def __init__(
        self,
        model: LLMCore,
        tools: List[Tool],
        system_prompt: str,
        trace_emitter: "TraceEmitter",
        config: "Configuration",
        parent_trace_session: Optional["TraceSession"] = None,
        max_iterations: int = 5,
        max_parallel_tools: int = 10,
    ):
        """Initialize Agent

        Args:
            model: LLMCore instance
            tools: List of tools
            system_prompt: System prompt
            trace_emitter: Trace logger (required)
            config: Configuration object (required)
            parent_trace_session: Parent trace session (optional, for nested agents)
            max_iterations: Max ReAct loop iterations
            max_parallel_tools: Max parallel tool calls (default 10)
        """
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = AgentMemory(system_prompt=system_prompt)
        self.max_iterations = max_iterations
        self.max_parallel_tools = max_parallel_tools
        self.trace_emitter = trace_emitter
        self.config = config
        self.parent_trace_session = parent_trace_session
        self._trace_id: Optional[str] = None
        self._should_terminate = False
        self._tool_log_files: Dict[str, str] = {}
        self._cancelled = threading.Event()  # 取消标志

        # Explicit assertion: config is required (fail fast principle)
        assert self.config is not None, "config is required for all ReActAgent instances"

        # Current iteration tracking (for supervisor step association)
        self._current_iteration: Optional[int] = None

        # 发射 agent_start 事件（记录初始配置）
        self.trace_emitter.emit_agent_start(
            system_prompt=self.memory.system_prompt,
            tools=[tool.to_openai_tool() for tool in self.tools.values()],
            initialization={"created_at": datetime.now(timezone.utc).isoformat()},
        )

    def cancel(self) -> None:
        """外部取消入口

        调用此方法后，Agent 会在下一次迭代检查点优雅退出。
        线程安全，可以从任意线程调用。
        """
        self._cancelled.set()
        self._should_terminate = True
        logger.info("Agent cancellation requested: trace_id=%s", self._trace_id)

    def __call__(self, task: str, trace_id: Optional[str] = None) -> Iterator[DeepResearchEvent]:
        """流式执行主循环（核心实现）

        Args:
            task: 任务描述
            trace_id: 可选的trace_id，如果不提供则自动生成

        Yields:
            DeepResearchEvent: 流式事件

        Example:
            >>> agent = Researcher(config, emitter)
            >>> for event in agent("研究任务"):
            ...     print(event)
        """
        # 初始化
        self._trace_id = trace_id or str(uuid.uuid4())
        self.memory.add_message(Message(role=MessageRole.USER, content=task))
        exit_reason = ExitReason.MAX_ITERATIONS

        for iteration in range(1, self.max_iterations + 1):
            self._current_iteration = iteration
            step_start_time = datetime.now(timezone.utc)

            # 检查取消标志
            if self._cancelled.is_set():
                logger.info("Agent cancelled: trace_id=%s, iteration=%d", self._trace_id, iteration)
                from .events import ErrorEvent

                yield ErrorEvent(error_message="Research cancelled by user", error_type="cancelled", trace_id=self._trace_id, timestamp=datetime.now(timezone.utc))
                exit_reason = ExitReason.CANCELLED
                break

            # 检查上下文限制
            context_decision = self._check_and_handle_context()
            if context_decision == ContextDecision.FORCE_FINISH:
                logger.warning("Context management decision: forcing final answer generation")
                yield from self._on_force_final_answer()
                return

            # P0-3 修复：在调用 LLM 之前就发送 trace 的 step_start 事件
            step_id = f"{self._trace_id[:8]}.{self.trace_emitter.agent_id}.step{iteration}"
            step_start_event_id = self.trace_emitter.emit(
                "step_start",
                step_id=step_id,
                step_index=iteration,
                iteration=iteration,
            )
            # 设置当前活动的 step_id（供工具调用使用）
            self.trace_emitter.set_current_step(step_id)

            # 发送前端实时事件（保持原有逻辑）
            yield AgentStepStartedEvent(step_number=iteration, trace_id=self._trace_id)

            # 调用 LLM 并处理响应
            messages = self.memory.to_messages()
            openai_tools = [tool.to_openai_tool() for tool in self.tools.values()]

            llm_result, llm_events = self._call_llm_streaming(messages, openai_tools)
            yield from llm_events

            # 保存 assistant 消息
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=llm_result.content or None,
                tool_calls=llm_result.tool_calls,
                reasoning_content=llm_result.reasoning_content,  # DeepSeek Reasoner
            )
            self.memory.add_message(assistant_msg)

            # 记录 LLM 调用日志
            self._log_llm_call(iteration, messages, openai_tools, llm_result)

            # 处理无工具调用的情况
            if not llm_result.tool_calls:
                # P0-3 修复：发送 step_end
                step_end_time = datetime.now(timezone.utc)
                self.trace_emitter.emit(
                    "step_end",
                    step_id=step_id,
                    step_index=iteration,
                    parent_event_id=step_start_event_id,
                    response_type="no_tool_calls",
                    llm_call_ids=[],
                    tool_call_ids=[],
                    duration_ms=int((step_end_time - step_start_time).total_seconds() * 1000),
                )
                self.trace_emitter.set_current_step(None)  # 清除当前 step
                yield from self._handle_no_tool_calls(iteration, step_start_time, llm_result.content)
                exit_reason = ExitReason.NO_TOOL_CALLS
                break

            # 执行工具（此时 step_start 已发送，tool_call_start 会有正确的 step_id）
            tool_results, should_stop = yield from self._execute_and_yield_tools(llm_result.tool_calls)
            if should_stop:
                return

            # P0-3 修复：发送 step_end（不再调用 _log_step_with_tools）
            step_end_time = datetime.now(timezone.utc)
            self.trace_emitter.emit(
                "step_end",
                step_id=step_id,
                step_index=iteration,
                parent_event_id=step_start_event_id,
                response_type="tool_calls",
                llm_call_ids=[],  # 可以从 trace_emitter 获取
                tool_call_ids=[tc.get("tool_call_id") for tc in tool_results],
                duration_ms=int((step_end_time - step_start_time).total_seconds() * 1000),
            )
            self.trace_emitter.set_current_step(None)  # 清除当前 step

            # 发送前端实时事件
            yield AgentStepCompletedEvent(step_number=iteration, trace_id=self._trace_id)

            # P0-3 修复：不再调用 _log_step_with_tools
            # self._log_step_with_tools(iteration, step_start_time, tool_results)

        # 循环结束，生成最终答案
        logger.info("Exiting loop: reason=%s, generating final answer", exit_reason.value)
        yield from self.generate_final_answer()

    def _call_llm_streaming(
        self,
        messages: List[Message],
        openai_tools: List[Dict[str, Any]],
    ) -> tuple[LLMResponseResult, List[DeepResearchEvent]]:
        """调用 LLM 并收集响应（流式）

        Args:
            messages: 对话历史
            openai_tools: 工具定义

        Returns:
            (LLMResponseResult, 事件列表)
        """
        start_time = datetime.now(timezone.utc)
        accumulated_content = ""
        accumulated_reasoning_content: Optional[str] = None  # DeepSeek Reasoner
        accumulated_tool_calls: Dict[str, ToolCall] = {}
        events: List[DeepResearchEvent] = []

        for event in stream_events(self.model, messages, tools=openai_tools, trace_id=self._trace_id):
            if event.event_type == EventType.LLM_CONTENT_DELTA:
                accumulated_content += event.content
                events.append(event)
            elif event.event_type == EventType.LLM_CONTENT_DONE:
                # 捕获 DeepSeek Reasoner 的 reasoning_content
                if hasattr(event, "reasoning_content") and event.reasoning_content:
                    accumulated_reasoning_content = event.reasoning_content
                events.append(event)
            elif event.event_type == EventType.LLM_TOOL_CALL_DONE:
                tool_call = ToolCall(
                    id=event.tool_call_id,
                    function=ToolCallFunction(name=event.tool_name, arguments=event.tool_arguments),
                )
                accumulated_tool_calls[event.tool_call_id] = tool_call
                events.append(event)

        end_time = datetime.now(timezone.utc)
        tool_calls_list = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None

        result = LLMResponseResult(
            content=accumulated_content,
            tool_calls=tool_calls_list,
            start_time=start_time,
            end_time=end_time,
            reasoning_content=accumulated_reasoning_content,  # DeepSeek Reasoner
        )
        return result, events

    def _log_llm_call(
        self,
        iteration: int,
        messages: List[Message],
        openai_tools: List[Dict[str, Any]],
        llm_result: LLMResponseResult,
    ) -> None:
        """记录 LLM 调用日志"""
        self.trace_emitter.log_llm_call_simple(
            step_number=iteration,
            messages=messages,
            openai_tools=openai_tools,
            response_content=llm_result.content,
            tool_calls=llm_result.tool_calls,
            start_time=llm_result.start_time,
            end_time=llm_result.end_time,
            model_id=self.config.model_id,
            max_tokens=self.config.max_tokens,
            description=f"step{iteration}_react",
        )

    def _handle_no_tool_calls(
        self,
        iteration: int,
        step_start_time: datetime,
        content: Optional[str],
    ) -> Iterator[DeepResearchEvent]:
        """处理无工具调用的情况

        Args:
            iteration: 当前迭代次数
            step_start_time: 步骤开始时间
            content: LLM 返回的文本内容

        Yields:
            ErrorEvent: 错误事件
        """
        from .events import ErrorEvent

        logger.warning(
            "No tool calls in response: trace_id=%s, iteration=%d, content_preview=%s",
            self._trace_id,
            iteration,
            content[:100] if content else "(empty)",
        )

        error_msg = f"LLM did not call any tools in step {iteration}"
        if content:
            error_msg += f". Response: {content[:200]}"

        yield ErrorEvent(
            error_message=error_msg,
            error_type="no_tool_calls",
            trace_id=self._trace_id,
            timestamp=datetime.now(timezone.utc),
        )

        # P0-3 修复：不再调用 yield AgentStepCompletedEvent 和 log_step_simple
        # 这些已经在主循环中处理

    def _execute_and_yield_tools(
        self,
        tool_calls_list: List[ToolCall],
    ) -> Generator[DeepResearchEvent, None, tuple[List[Dict[str, Any]], bool]]:
        """执行工具并收集结果

        Args:
            tool_calls_list: 工具调用列表

        Yields:
            DeepResearchEvent: 工具执行事件

        Returns:
            tuple[List[Dict], bool]: (工具结果列表, 是否应该停止)
            通过 Generator 的 return 语句返回，调用方使用 yield from 获取
        """
        tool_results = []

        for tool_event in self._execute_tools(tool_calls_list):
            yield tool_event

            if tool_event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
                log_file = self._tool_log_files.get(tool_event.tool_call_id, "unknown")
                tool_results.append(
                    {
                        "tool_call_id": tool_event.tool_call_id,
                        "tool_name": tool_event.tool_name,
                        "log_file": log_file,
                        "result": tool_event.tool_output,  # 完整结果
                    }
                )

        # 检查终止信号
        if self._should_terminate:
            return tool_results, True

        return tool_results, False

    def _log_step_with_tools(
        self,
        iteration: int,
        step_start_time: datetime,
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """记录包含工具调用的步骤"""
        step_end_time = datetime.now(timezone.utc)
        self.trace_emitter.log_step_simple(
            step_number=iteration,
            start_time=step_start_time,
            end_time=step_end_time,
            llm_call_counter=self.trace_emitter.llm_call_counter,
            iteration=iteration,
            response_type="tool_calls",
            tool_call_summaries=tool_results,  # 传递完整工具结果
            next_action="continue",
        )

    def _execute_tools(self, tool_calls_list: List[ToolCall]) -> Iterator[DeepResearchEvent]:
        """执行工具调用（统一并行执行，支持流式和普通工具）

        设计要点：
        1. 提前检查终止信号，避免浪费资源
        2. 工具线程消费生成器，实时推送事件到队列
        3. 主线程消费队列，实时 yield 事件
        4. 完整的异常处理（所有路径）
        5. 完整的日志记录（所有路径）
        6. 线程安全的结果收集

        Args:
            tool_calls_list: 工具调用列表

        Yields:
            DeepResearchEvent: 工具执行事件（实时）
        """
        # 1. 检查终止信号（提前检查，避免浪费资源）
        for tool_call in tool_calls_list:
            if self.is_final_answer_signal(tool_call.function.name):
                # 关键修复：在调用 generate_final_answer 之前，必须添加 tool 结果消息
                # 否则 memory 中的 assistant 消息有 tool_calls 但没有对应的 tool 结果，
                # 导致下一次 LLM 调用（generate_final_answer 中的调用）失败
                tool_msg = Message(
                    role=MessageRole.TOOL,
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content="Acknowledged. Generating final answer...",
                )
                self.memory.add_message(tool_msg)

                yield from self.generate_final_answer()
                self._should_terminate = True
                return

        # 2. 创建共享状态
        event_queue: queue.Queue = queue.Queue()
        results_lock = threading.Lock()
        tool_results: Dict[str, ToolExecutionResult] = {}

        # 3. 并行执行工具
        max_workers = min(self.max_parallel_tools, len(tool_calls_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有工具执行任务
            for tool_call in tool_calls_list:
                executor.submit(
                    self._execute_single_tool,
                    tool_call,
                    event_queue,
                    results_lock,
                    tool_results,
                )

            # 4. 消费事件队列（实时透传）
            yield from self._consume_event_queue(event_queue, len(tool_calls_list))

        # 5. 收集结果并处理
        self._collect_tool_results(tool_results)

    def _execute_single_tool(
        self,
        tool_call: ToolCall,
        event_queue: queue.Queue,
        results_lock: threading.Lock,
        tool_results: Dict[str, ToolExecutionResult],
    ) -> None:
        """在工作线程中执行单个工具

        Args:
            tool_call: 工具调用对象
            event_queue: 事件队列（线程安全）
            results_lock: 结果锁
            tool_results: 结果字典
        """
        tool_start_time = datetime.now(timezone.utc)
        tool_name = tool_call.function.name
        result: Optional[str] = None
        error: Optional[Exception] = None
        received_completion = False

        try:
            # 1. 验证工具存在
            if tool_name not in self.tools:
                error = Exception(f"Unknown tool: {tool_name}")
                result = f"ERROR: Unknown tool {tool_name}"
                self._emit_tool_error_events(event_queue, tool_call, tool_name, result)
                return

            tool = self.tools[tool_name]

            # 2. 解析参数
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                error = e
                result = f"ERROR: Argument parsing failed - {str(e)}"
                self._emit_tool_error_events(event_queue, tool_call, tool_name, result)
                return

            # 3. 构建执行上下文
            context = ToolExecutionContext(
                tool_call_id=tool_call.id,
                trace_id=self._trace_id,
                config=self.config,
                parent_trace_session=self.parent_trace_session,
                on_result_callback=self._on_tool_result,
                supervisor_step_number=self._current_iteration,
            )

            # 4. 执行工具并推送事件
            for event in tool.execute_stream(context=context, **arguments):
                event_queue.put(event)
                if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
                    result = event.tool_output
                    received_completion = True

            # 5. 兜底检查
            if not received_completion:
                logger.error("Tool contract violated: tool=%s, missing completion event", tool_name, exc_info=True)
                error = Exception("Tool did not produce completion event")
                result = result or "ERROR: Tool execution incomplete"
                event_queue.put(
                    ToolExecutionCompletedEvent(
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        tool_output=result,
                        trace_id=self._trace_id,
                    )
                )

        except Exception as e:
            logger.error("Tool execution failed: tool=%s, error=%s", tool_name, str(e), exc_info=True)
            error = e
            result = result or f"ERROR: {str(e)}"
            if not received_completion:
                event_queue.put(
                    ToolExecutionCompletedEvent(
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        tool_output=result,
                        trace_id=self._trace_id,
                    )
                )

        finally:
            # 记录日志
            tool_end_time = datetime.now(timezone.utc)
            log_file_path = self._log_tool_execution(tool_call, tool_name, result, error, tool_start_time, tool_end_time)

            # 保存结果（线程安全）
            with results_lock:
                tool_results[tool_call.id] = ToolExecutionResult(
                    tool_call=tool_call,
                    result=result,
                    error=error,
                    log_path=log_file_path,
                )

            # 标记线程完成
            event_queue.put(("_THREAD_DONE_", tool_call.id))

    def _emit_tool_error_events(
        self,
        event_queue: queue.Queue,
        tool_call: ToolCall,
        tool_name: str,
        error_msg: str,
    ) -> None:
        """发送工具错误事件到队列"""
        event_queue.put(
            ToolExecutionStartedEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                trace_id=self._trace_id,
            )
        )
        event_queue.put(
            ToolExecutionCompletedEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                tool_output=error_msg,
                trace_id=self._trace_id,
            )
        )

    def _log_tool_execution(
        self,
        tool_call: ToolCall,
        tool_name: str,
        result: Optional[str],
        error: Optional[Exception],
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[str]:
        """记录工具执行日志（捕获异常）"""
        try:
            return self.trace_emitter.log_tool_call_simple(
                tool_call=tool_call,
                tool_name=tool_name,
                result=result if not error else None,
                error=error,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as log_error:
            logger.error("Failed to log tool call: %s", str(log_error), exc_info=True)
            return None

    def _consume_event_queue(
        self,
        event_queue: queue.Queue,
        total_tools: int,
    ) -> Iterator[DeepResearchEvent]:
        """消费事件队列，实时透传事件

        Args:
            event_queue: 事件队列
            total_tools: 工具总数

        Yields:
            DeepResearchEvent: 工具执行事件
        """
        completed_count = 0
        while completed_count < total_tools:
            try:
                item = event_queue.get(timeout=0.1)

                # 检查线程完成标记
                if isinstance(item, tuple) and item[0] == "_THREAD_DONE_":
                    _, tool_call_id = item
                    completed_count += 1
                    logger.debug("Tool execution completed: tool_call_id=%s, progress=%d/%d", tool_call_id, completed_count, total_tools)
                    continue

                # 透传事件
                yield item

            except queue.Empty:
                continue

    def _collect_tool_results(self, tool_results: Dict[str, ToolExecutionResult]) -> None:
        """收集工具执行结果并添加到 memory

        遍历工具执行结果，保存日志路径，格式化结果并添加到 memory。

        Args:
            tool_results: 工具执行结果字典 {tool_call_id: ToolExecutionResult}
        """
        for tool_call_id, exec_result in tool_results.items():
            # 保存日志路径
            if exec_result.log_path:
                self._tool_log_files[tool_call_id] = exec_result.log_path

            # 格式化结果
            tool_call = exec_result.tool_call
            tool_name = tool_call.function.name

            if exec_result.error:
                formatted_result = f"ERROR: Tool execution failed - {str(exec_result.error)}"
            else:
                tool = self.tools.get(tool_name)
                formatted_result = tool.format_for_llm(exec_result.result) if tool else exec_result.result

            # 添加到 memory
            tool_msg = Message(
                role=MessageRole.TOOL,
                tool_call_id=tool_call.id,
                name=tool_name,
                content=formatted_result,
            )
            self.memory.add_message(tool_msg)

    def _on_tool_result(self, key: str, value: str) -> None:
        """钩子方法：工具结果回调（子类可覆盖）

        用于接收工具通过回调保存的结果。

        Args:
            key: 结果键
            value: 结果值

        默认实现：什么都不做
        Supervisor 覆盖：保存到 self.research_results
        """
        pass

    # ════════════════════════════════════════════════════════
    # 统一上下文管理方法（新架构）
    # ════════════════════════════════════════════════════════

    def _check_and_handle_context(self) -> ContextDecision:
        """检查上下文并决定处理策略（统一检查点）

        此方法在每次迭代开始前调用，负责：
        1. 估算当前 token 数
        2. 判断是否超过阈值
        3. 决定处理策略：继续 / 压缩 / 强制终止

        Returns:
            ContextDecision: CONTINUE - 可以继续迭代
                            FORCE_FINISH - 必须强制生成最终答案

        注意：
        - 配置错误（config缺失、max_context_length=0等）会直接抛出异常
        - tokenizer 失败会直接抛出异常
        - 这些异常会传播到上层（cli_runner）进行统一处理
        """
        # 1. 估算当前 token 数
        current_tokens = self._estimate_memory_tokens()

        # 2. 计算阈值
        threshold_tokens = int(self.config.max_context_length * self.config.context_check_threshold)

        # 3. 检查是否超限
        if current_tokens < threshold_tokens:
            # 未超限，继续执行
            return ContextDecision.CONTINUE

        # 超限了，需要处理
        usage_percent = current_tokens / self.config.max_context_length * 100
        logger.warning("Context approaching limit: current=%d, max=%d (%.1f%%), threshold=%d", current_tokens, self.config.max_context_length, usage_percent, threshold_tokens)

        # 4. 决定处理策略
        if self.config.enable_context_compress:
            # 策略A: 尝试压缩
            logger.info("Attempting context compaction...")
            compress_success = self._on_attempt_context_compress()

            if compress_success:
                # 压缩成功，检查新的 token 数
                after_tokens = self._estimate_memory_tokens()
                saved_tokens = current_tokens - after_tokens
                logger.info("Compaction successful: tokens %d->%d (freed %d, %.1f%%)", current_tokens, after_tokens, saved_tokens, saved_tokens / current_tokens * 100 if current_tokens > 0 else 0)
                return ContextDecision.CONTINUE  # 可以继续迭代
            else:
                # 压缩失败或效果不佳
                logger.warning("Compaction failed or insufficient, forcing final answer generation")
                return ContextDecision.FORCE_FINISH
        else:
            # 策略B: 未启用压缩，直接强制终止
            logger.warning("Context compaction not enabled, forcing final answer generation")
            return ContextDecision.FORCE_FINISH

    def _estimate_memory_tokens(self) -> int:
        """估算当前 memory 的 token 数（委托给 utils 实现）

        Returns:
            int: 当前 memory 的 token 总数

        注意：
        - tokenizer 失败会直接抛出 RuntimeError
        - 异常会传播到上层进行处理
        """
        from .utils import estimate_memory_tokens

        return estimate_memory_tokens(self.memory.system_prompt, self.memory.messages)

    def _on_attempt_context_compress(self) -> bool:
        """钩子方法：尝试压缩上下文（子类可选实现）

        子类应该实现此方法来执行具体的压缩策略，例如：
        - Researcher: 删除低分的检索 chunks
        - Supervisor: 删除早期的子研究结果

        Returns:
            bool: True - 压缩成功且释放了足够空间
                  False - 压缩失败或效果不佳

        默认实现：不压缩，直接返回 False
        """
        logger.debug("Agent does not implement compaction logic, skipping: agent_class=%s", self.__class__.__name__)
        return False

    def _on_force_final_answer(self) -> Iterator[DeepResearchEvent]:
        """钩子方法：强制进入最终答案生成流程（子类可覆盖）

        在上下文超限等情况下被调用，直接生成最终答案。

        Yields:
            DeepResearchEvent: 报告生成过程中的事件
        """
        logger.info("Forced final answer generation triggered")
        yield from self.generate_final_answer()

    @abstractmethod
    def is_final_answer_signal(self, tool_name: str) -> bool:
        """判断工具是否为终止信号（子类必须实现）

        Args:
            tool_name: 工具名称

        Returns:
            bool: 是否为终止信号工具

        示例：
            Researcher: 返回 tool_name == "research_complete"
            Supervisor: 返回 tool_name == "research_complete"
        """
        raise NotImplementedError

    @abstractmethod
    def generate_final_answer(self) -> Iterator[DeepResearchEvent]:
        """生成最终答案（流式，子类必须实现）

        Yields:
            StreamEvent: 最终答案生成过程中的事件

        注意：
        - 必须是流式输出（yield StreamEvent）
        - 应该调用LLM生成报告，并透传content事件
        - 最后必须yield一个final_answer事件
        """
        raise NotImplementedError
