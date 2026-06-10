"""Web后端入口模块 - 兼容agentic_reasoning的调用方式

V4 架构:
- SSEEventStore: 写入 Redis Stream 的事件存储器
- run_research_background: 后台进程执行研究任务（仅写 Redis）
- run_research_web: 原有的同步生成器接口（兼容旧调用方式）
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .configuration import Configuration
from .events import DeepResearchEvent, EventType
from .supervisor import Supervisor
from .task_registry import task_registry
from core.logging.trace import TraceSession
from .types import TraceStatus

logger = logging.getLogger(__name__)

# ===== SSE Redis 配置（从环境变量读取） =====
SSE_STREAM_PREFIX = "sse:stream:"
SSE_STATUS_PREFIX = "sse:status:"
SSE_META_PREFIX = "sse:meta:"
SSE_TTL = int(os.getenv("SSE_REDIS_TTL", "259200"))  # 72 小时
SSE_MAXLEN = int(os.getenv("SSE_REDIS_MAXLEN", "5000"))
SSE_HEARTBEAT_INTERVAL = int(os.getenv("SSE_HEARTBEAT_INTERVAL", "30"))


class SSEEventStore:
    """SSE 事件存储器（仅写入 Redis，不 yield）

    V4 设计：web_runner 只写 Redis，API 只读 Redis
    故障隔离：Redis 不可用时任务继续执行，只是失去断点续传能力
    """

    def __init__(
        self,
        trace_id: str,
        tenant_id: str,
        kb_id: str,
        task: str,
        log_dir: Optional[str] = None,
    ):
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self.kb_id = kb_id
        self.task = task
        self.log_dir = log_dir

        self._stream_key = f"{SSE_STREAM_PREFIX}{trace_id}"
        self._status_key = f"{SSE_STATUS_PREFIX}{trace_id}"
        self._meta_key = f"{SSE_META_PREFIX}{trace_id}"
        self._sequence = 0
        self._redis_available = self._check_redis()

        if self._redis_available:
            self._init_metadata()
        else:
            logger.warning("Redis unavailable for trace_id=%s, SSE resume disabled", trace_id)

    def _check_redis(self) -> bool:
        """检查 Redis 是否可用"""
        try:
            from core.storage.redis_conn import REDIS_CONN

            return REDIS_CONN is not None and REDIS_CONN.is_alive()
        except Exception:
            return False

    def _init_metadata(self):
        """初始化任务元数据和状态"""
        try:
            from core.storage.redis_conn import REDIS_CONN

            # 写入元数据
            metadata = {
                "trace_id": self.trace_id,
                "tenant_id": self.tenant_id,
                "kb_id": self.kb_id,
                "task": self.task,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "log_dir": self.log_dir,
            }
            REDIS_CONN.set_obj(self._meta_key, metadata, exp=SSE_TTL)

            # 初始化状态为 running
            REDIS_CONN.set(self._status_key, "running", exp=SSE_TTL)

            logger.info("SSE metadata initialized: trace_id=%s", self.trace_id)
        except Exception as e:
            logger.warning("Failed to init SSE metadata: %s", e)
            self._redis_available = False

    def store_event(self, web_data: Dict[str, Any]) -> bool:
        """存储 SSE 事件到 Redis Stream

        Args:
            web_data: ResearchStreamData 字典（会被修改，添加 _seq）

        Returns:
            bool: 是否成功写入
        """
        if not self._redis_available:
            return False

        try:
            from core.storage.redis_conn import REDIS_CONN

            self._sequence += 1

            # 在 web_data 中添加序列号（重要：在序列化前）
            web_data["_seq"] = self._sequence

            entry = {
                "seq": str(self._sequence),
                "ts": datetime.now(timezone.utc).isoformat(),
                "data": json.dumps(web_data, ensure_ascii=False, default=str),
            }

            REDIS_CONN.REDIS.xadd(
                self._stream_key,
                entry,
                maxlen=SSE_MAXLEN,
            )

            # 刷新所有 key 的 TTL（修复：避免 meta/status 过期但 stream 仍在）
            REDIS_CONN.REDIS.expire(self._stream_key, SSE_TTL)
            REDIS_CONN.REDIS.expire(self._status_key, SSE_TTL)
            REDIS_CONN.REDIS.expire(self._meta_key, SSE_TTL)

            return True
        except Exception as e:
            logger.warning("Failed to store SSE event: %s", e)
            return False

    def store_done_event(self, final_answer: str) -> bool:
        """存储任务完成事件

        Args:
            final_answer: 最终答案文本

        Returns:
            bool: 是否成功写入
        """
        if not self._redis_available:
            return False

        try:
            done_data = {
                "_type": "DONE",
                "answer": final_answer,
                "reference": {},
                "audio_binary": None,
            }
            return self.store_event(done_data)
        except Exception as e:
            logger.warning("Failed to store DONE event: %s", e)
            return False

    def update_status(self, status: str) -> bool:
        """更新任务状态

        Args:
            status: "running" | "completed" | "error" | "cancelled"

        Returns:
            bool: 是否成功更新
        """
        if not self._redis_available:
            return False

        try:
            from core.storage.redis_conn import REDIS_CONN

            REDIS_CONN.set(self._status_key, status, exp=SSE_TTL)
            logger.info("Task status updated: trace_id=%s, status=%s", self.trace_id, status)
            return True
        except Exception as e:
            logger.warning("Failed to update status: %s", e)
            return False

    def get_status(self) -> Optional[str]:
        """获取任务状态

        Returns:
            Optional[str]: 状态字符串，Redis 不可用或 key 不存在时返回 None
        """
        if not self._redis_available:
            return None

        try:
            from core.storage.redis_conn import REDIS_CONN

            return REDIS_CONN.get(self._status_key)
        except Exception as e:
            logger.warning("Failed to get status: %s", e)
            return None


def extract_agent_id(trace_id: Optional[str]) -> str:
    """从 trace_id 提取 agent_id

    Args:
        trace_id: 追踪ID（如 "abc123.researcher_a1b2"）

    Returns:
        str: agent_id（如 "supervisor", "researcher_a1b2"）

    Examples:
        >>> extract_agent_id("abc123")
        "supervisor"
        >>> extract_agent_id("abc123.researcher_a1b2")
        "researcher_a1b2"
        >>> extract_agent_id(None)
        "unknown"
    """
    if not trace_id:
        return "unknown"
    parts = trace_id.split(".")
    if len(parts) == 1:
        return "supervisor"
    return parts[-1]


def run_research_web(task: str, config: Configuration, stream: bool = True, chunk_info: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Web后端入口：运行研究并返回兼容agentic_reasoning格式的结果

    此函数为Web API设计，兼容agentic_reasoning的返回格式，可直接替换
    DeepResearcher.thinking()的调用。

    事件适配逻辑：
    - DeepResearchEvent（deepresearch_v2）→ dict（agentic_reasoning）
    - 流式模式：每次LLM增量都yield一次
    - 非流式模式：只在最后yield完整结果
    - 自动记录执行轨迹日志（强制开启）

    Args:
        task: 研究任务描述
        config: 配置对象（包含kb_id、tenant_id、模型配置等）
        stream: 是否流式输出（默认True）
        chunk_info: 可选的chunk信息字典，用于收集引用信息（兼容agentic_reasoning）
                   格式: {"chunks": [], "doc_aggs": []}

    Yields:
        dict: 兼容agentic_reasoning格式的结果字典
              {
                  "answer": str,           # 累积的答案文本
                  "reference": dict,       # 引用信息（TODO: 暂时为空字典）
                  "audio_binary": None     # 音频数据（deepresearch_v2不支持）
              }

    Example:
        >>> from deepresearch_v2 import Configuration, run_research_web
        >>> config = Configuration(
        ...     kb_id="test_kb",
        ...     tenant_id="test_tenant",
        ...     model_id="qwen"
        ... )
        >>> chunk_info = {"chunks": [], "doc_aggs": []}
        >>> for result in run_research_web(
        ...     task="分析GraphRAG技术原理",
        ...     config=config,
        ...     stream=True,
        ...     chunk_info=chunk_info
        ... ):
        ...     print(result["answer"])  # 实时输出

    Note:
        - 完全兼容agentic_reasoning的DeepResearcher.thinking()接口
        - 可在dialog_service.py中直接替换使用
        - chunk_info会在执行过程中被填充（TODO: 待实现引用收集）
        - 日志会自动保存到 logs/ 目录下
    """
    try:
        # 初始化chunk_info（如果提供）
        if chunk_info is not None:
            chunk_info.setdefault("chunks", [])
            chunk_info.setdefault("doc_aggs", [])

        # 创建 TraceSession（强制日志，无条件创建）
        trace_session = TraceSession(task=task, log_root=Path(config.log_root_path))
        if trace_session.is_available:
            logger.info("Trace logging enabled: log_dir=%s", trace_session.log_dir)
            # 发出 trace_start 事件
            trace_session.emit_trace_start(
                config={
                    "kb_id": config.kb_id,
                    "tenant_id": config.tenant_id,
                    "model_id": config.model_id,
                    "language": config.prompt_language,
                    "max_researcher_iterations": config.max_researcher_iterations,
                    "max_react_tool_calls": config.max_react_tool_calls,
                }
            )
        else:
            logger.warning("Trace logging unavailable: initialization failed")

        # 创建Supervisor Agent（传递 trace_session）
        supervisor = Supervisor(config, parent_trace_session=trace_session)

        # 注册任务（支持停止功能）
        task_registry.register(trace_session.trace_id, supervisor, config.tenant_id)
        logger.info("Registered task: trace_id=%s", trace_session.trace_id)

        try:
            # 维护多个 agent 的状态和内容
            agent_states = {}  # agent_id -> state dict
            agent_contents = {}  # agent_id -> accumulated content

            # 首条消息：返回 session 信息
            yield {"answer": "", "reference": {}, "audio_binary": None, "session": {"trace_id": trace_session.trace_id, "log_dir": str(trace_session.log_dir) if trace_session.log_dir else None}}

            # 消费流式事件，适配为多 agent 格式（传递 trace_id）
            for event in supervisor(task, trace_id=trace_session.trace_id):
                # 提取 agent_id（从 trace_id）
                agent_id = extract_agent_id(event.trace_id)

                # 初始化 agent 状态
                if agent_id not in agent_states:
                    agent_states[agent_id] = {"status": "initialized", "last_activity": event.timestamp.isoformat()}
                    agent_contents[agent_id] = ""

                # 更新最后活动时间
                agent_states[agent_id]["last_activity"] = event.timestamp.isoformat()

                # 处理不同事件类型
                if event.event_type == EventType.SUB_RESEARCH_STARTED:
                    # 子研究开始
                    agent_states[agent_id]["status"] = "running"
                    agent_states[agent_id]["topic"] = event.topic
                    agent_states[agent_id]["researcher_id"] = event.researcher_id
                    agent_states[agent_id]["current_step"] = 0

                elif event.event_type == EventType.AGENT_STEP_STARTED:
                    # Agent 步骤开始
                    agent_states[agent_id]["current_step"] = event.step_number
                    agent_states[agent_id]["status"] = "running"

                elif event.event_type == EventType.LLM_CONTENT_DELTA:
                    # 累加 LLM 内容（每个 agent 独立累加）
                    agent_contents[agent_id] += event.content
                    agent_states[agent_id]["content"] = agent_contents[agent_id]

                    # 实时返回（流式）
                    if stream:
                        yield {
                            "answer": agent_contents.get("supervisor", ""),  # 主答案是 Supervisor 的内容
                            "reference": {},
                            "audio_binary": None,
                            "agents": {k: v.copy() for k, v in agent_states.items()},  # 深拷贝避免引用问题
                        }

                elif event.event_type == EventType.RESEARCH_COMPLETED:
                    # 研究完成
                    agent_states[agent_id]["status"] = "completed"
                    agent_states[agent_id]["final_content"] = event.content

                    # 如果是 Supervisor 完成，更新主答案
                    if agent_id == "supervisor":
                        agent_contents["supervisor"] = event.content

                    if stream:
                        yield {
                            "answer": agent_contents.get("supervisor", ""),
                            "reference": {},
                            "audio_binary": None,
                            "agents": {k: v.copy() for k, v in agent_states.items()},
                        }

                elif event.event_type == EventType.ERROR_OCCURRED:
                    # 错误事件
                    agent_states[agent_id]["status"] = "error"
                    agent_states[agent_id]["error"] = event.error_message
                    logger.error("Agent encountered error: agent_id=%s, error=%s", agent_id, event.error_message)

            # 最终返回（确保非流式模式也有返回）
            final_answer = agent_contents.get("supervisor", "")
            yield {
                "answer": final_answer,
                "reference": _build_reference(chunk_info),
                "audio_binary": None,
                "agents": agent_states,
            }

            # 发出 final 事件
            if trace_session.is_available:
                try:
                    # 写入 Supervisor metadata（兼容方法）
                    if supervisor.trace_emitter:
                        supervisor.trace_emitter.write_metadata()
                        supervisor.trace_emitter.write_final_output(final_answer)

                    # 收集统计信息并发出 final 事件
                    stats = trace_session.collect_statistics()
                    trace_session.emit_final(
                        status=TraceStatus.COMPLETED,
                        final_report=final_answer,
                        total_steps=stats.get("total_steps", 0),
                        total_tool_calls=stats.get("total_tool_calls", 0),
                        total_llm_calls=stats.get("total_llm_calls", 0),
                    )
                    logger.info("Execution trace saved: log_dir=%s", trace_session.log_dir)
                except Exception as e:
                    logger.warning("Failed to emit final trace event: %s", str(e))

        finally:
            # 注销任务（无论成功、失败还是取消）
            task_registry.unregister(trace_session.trace_id)
            logger.info("Unregistered task: trace_id=%s", trace_session.trace_id)

    except Exception as e:
        logger.error("Web execution failed: %s", str(e), exc_info=True)
        # 确保注销任务
        if "trace_session" in locals():
            task_registry.unregister(trace_session.trace_id)
        # 返回错误信息（兼容agentic_reasoning）
        yield {"answer": f"<error>Research execution failed: {str(e)}</error>", "reference": {}, "audio_binary": None}


def _build_reference(chunk_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """构建引用信息字典（兼容agentic_reasoning格式）

    Args:
        chunk_info: chunk信息字典，新版使用 kg_result 结构

    Returns:
        dict: 引用信息字典（目前返回空字典，待后续实现）

    TODO: 实现引用信息的提取和格式化
    可能的格式：
    {
        "chunks": [...],      # chunk详情列表
        "doc_aggs": [...],    # 文档聚合信息
        "total_chunks": int,  # chunk总数
        "doc_count": int      # 文档数
    }
    """
    if chunk_info is None:
        return {}

    # 检查新版 kg_result 结构
    kg_result = chunk_info.get("kg_result")
    if not kg_result:
        return {}

    text_group = kg_result.get("text_group", {})
    graph_group = kg_result.get("graph_group", {})

    # 如果没有任何检索结果，返回空
    has_text = text_group.get("items")
    has_graph = graph_group.get("items")

    if not has_text and not has_graph:
        return {}

    # TODO: 格式化chunk_info为reference字典
    # 当前版本：直接返回空字典（不影响功能）
    return {}


def adapt_event_to_web_format(event_stream: Iterator[DeepResearchEvent], stream: bool = True) -> Iterator[Dict[str, Any]]:
    """通用事件适配器：将DeepResearchEvent流转换为Web API格式

    此函数可用于任何需要适配事件格式的场景。

    Args:
        event_stream: DeepResearchEvent流（来自Agent.__call__()）
        stream: 是否流式输出

    Yields:
        dict: Web API格式的结果字典

    Example:
        >>> supervisor = Supervisor(config)
        >>> event_stream = supervisor("研究任务")
        >>> for result in adapt_event_to_web_format(event_stream):
        ...     print(result["answer"])
    """
    accumulated_answer = ""

    for event in event_stream:
        if event.event_type == EventType.LLM_CONTENT_DELTA:
            accumulated_answer += event.content
            if stream:
                yield {"answer": accumulated_answer, "reference": {}, "audio_binary": None}

        elif event.event_type == EventType.RESEARCH_COMPLETED:
            final_answer = event.content or accumulated_answer
            yield {"answer": final_answer, "reference": {}, "audio_binary": None}

        elif event.event_type == EventType.ERROR_OCCURRED:
            logger.error("Event stream error: type=%s, message=%s", event.error_type, event.error_message)

        else:
            # 记录未处理的事件类型（调试用）
            logger.debug("Unhandled event type in adapter: event_type=%s", event.event_type)


# ===== V4: 后台进程执行模式 =====


def run_research_background(
    trace_id: str,
    task: str,
    config_dict: Dict[str, Any],
) -> None:
    """后台进程执行研究（只写 Redis，不 yield）

    V4 设计：此函数在独立进程中执行，仅写入 Redis Stream
    API 层通过 XREAD 消费事件并输出到 SSE

    Args:
        trace_id: 追踪ID（由 API 层生成）
        task: 研究任务描述
        config_dict: 配置字典（注意：多进程需要序列化，传字典而非对象）
    """
    try:
        # 修复 P0-2: 初始化全局 settings（必须在创建工具之前）
        from api.settings import init_settings
        init_settings()
        logger.info("API settings initialized in background process (trace_id=%s)", trace_id)

        from .configuration import Configuration

        # 重建 Configuration 对象（从字典）
        config = Configuration(
            kb_id=config_dict["kb_id"],
            tenant_id=config_dict["tenant_id"],
            model_id=config_dict["model_id"],
            api_base=config_dict.get("api_base", ""),
            api_key=config_dict.get("api_key", ""),
            prompt_language=config_dict.get("language", "zh"),  # 修复 P1: 改为 prompt_language
            max_researcher_iterations=config_dict.get("max_researcher_iterations", 3),
            max_react_tool_calls=config_dict.get("max_react_tool_calls", 5),
            log_root_path=config_dict.get("log_root_path", "logs"),
        )

        # 创建 TraceSession（使用传入的 trace_id）
        trace_session = TraceSession(
            task=task,
            log_root=Path(config.log_root_path),
            trace_id=trace_id,  # 使用预生成的 trace_id
        )

        if trace_session.is_available:
            logger.info("Trace logging enabled: log_dir=%s", trace_session.log_dir)
            trace_session.emit_trace_start(
                config={
                    "kb_id": config.kb_id,
                    "tenant_id": config.tenant_id,
                    "model_id": config.model_id,
                    "language": config.prompt_language,
                    "max_researcher_iterations": config.max_researcher_iterations,
                    "max_react_tool_calls": config.max_react_tool_calls,
                }
            )

        # 创建 SSE 事件存储器
        sse_store = SSEEventStore(
            trace_id=trace_id,
            tenant_id=config.tenant_id,
            kb_id=config.kb_id,
            task=task,
            log_dir=str(trace_session.log_dir) if trace_session.log_dir else None,
        )

        # 创建 Supervisor
        supervisor = Supervisor(
            config=config,
            task=task,
            parent_trace_session=trace_session,
        )

        # 注册任务（支持停止功能）
        task_registry.register(trace_id, supervisor, config.tenant_id)
        logger.info("Background task registered: trace_id=%s", trace_id)

        try:
            agent_states: Dict[str, Any] = {}
            agent_contents: Dict[str, str] = {}

            # 首条消息：session 信息
            first_msg = {
                "answer": "",
                "reference": {},
                "audio_binary": None,
                "session": {
                    "trace_id": trace_id,
                    "log_dir": str(trace_session.log_dir) if trace_session.log_dir else None,
                },
            }
            sse_store.store_event(first_msg)

            # 消费流式事件
            for event in supervisor(task, trace_id=trace_id):
                agent_id = extract_agent_id(event.trace_id)

                # 更新 agent 状态
                _update_agent_state(agent_states, agent_contents, agent_id, event)

                # 构建 Web 格式数据
                web_data = {
                    "answer": agent_contents.get("supervisor", ""),
                    "reference": {},
                    "audio_binary": None,
                    "agents": {k: v.copy() for k, v in agent_states.items()},
                    "event": event.to_dict(),
                }

                # 写入 Redis（失败时吞掉异常）
                sse_store.store_event(web_data)

                # 处理完成事件
                if event.event_type == EventType.RESEARCH_COMPLETED and agent_id == "supervisor":
                    agent_contents["supervisor"] = event.content

            # 写入最终完成事件
            final_answer = agent_contents.get("supervisor", "")
            sse_store.store_done_event(final_answer)
            sse_store.update_status("completed")

            # 发出 trace final 事件
            if trace_session.is_available:
                try:
                    if supervisor.trace_emitter:
                        supervisor.trace_emitter.write_metadata()
                        supervisor.trace_emitter.write_final_output(final_answer)

                    stats = trace_session.collect_statistics()
                    trace_session.emit_final(
                        status=TraceStatus.COMPLETED,
                        final_report=final_answer,
                        total_steps=stats.get("total_steps", 0),
                        total_tool_calls=stats.get("total_tool_calls", 0),
                        total_llm_calls=stats.get("total_llm_calls", 0),
                    )
                except Exception as e:
                    logger.warning("Failed to emit final trace event: %s", e)

            logger.info("Research completed: trace_id=%s", trace_id)

        except Exception as e:
            logger.error("Research execution failed: trace_id=%s, error=%s", trace_id, e, exc_info=True)
            try:
                sse_store.update_status("error")
                # 写入错误事件
                error_data = {
                    "answer": f"**ERROR**: {str(e)}",
                    "reference": {},
                    "audio_binary": None,
                    "_type": "ERROR",
                }
                sse_store.store_event(error_data)
            except Exception:
                pass

        finally:
            # 注销任务
            task_registry.unregister(trace_id)
            logger.info("Background task unregistered: trace_id=%s", trace_id)

    except Exception as e:
        # 顶层异常捕获（进程启动失败等）
        logger.error("Failed to start research: trace_id=%s, error=%s", trace_id, e, exc_info=True)


def _update_agent_state(
    agent_states: Dict[str, Any],
    agent_contents: Dict[str, str],
    agent_id: str,
    event: DeepResearchEvent,
) -> None:
    """更新 agent 状态（辅助函数）

    Args:
        agent_states: agent 状态字典（会被修改）
        agent_contents: agent 内容字典（会被修改）
        agent_id: agent ID
        event: DeepResearch 事件
    """
    # 初始化
    if agent_id not in agent_states:
        agent_states[agent_id] = {
            "status": "initialized",
            "last_activity": event.timestamp.isoformat(),
        }
        agent_contents[agent_id] = ""

    # 更新最后活动时间
    agent_states[agent_id]["last_activity"] = event.timestamp.isoformat()

    # 处理不同事件类型
    if event.event_type == EventType.SUB_RESEARCH_STARTED:
        agent_states[agent_id].update(
            {
                "status": "running",
                "topic": event.topic,
                "researcher_id": event.researcher_id,
                "current_step": 0,
            }
        )

    elif event.event_type == EventType.AGENT_STEP_STARTED:
        agent_states[agent_id]["current_step"] = event.step_number
        agent_states[agent_id]["status"] = "running"

    elif event.event_type == EventType.LLM_CONTENT_DELTA:
        agent_contents[agent_id] += event.content
        agent_states[agent_id]["content"] = agent_contents[agent_id]

    elif event.event_type == EventType.RESEARCH_COMPLETED:
        agent_states[agent_id]["status"] = "completed"
        agent_states[agent_id]["final_content"] = event.content
        agent_contents[agent_id] = event.content

    elif event.event_type == EventType.ERROR_OCCURRED:
        agent_states[agent_id]["status"] = "error"
        agent_states[agent_id]["error"] = event.error_message
