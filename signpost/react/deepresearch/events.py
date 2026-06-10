"""内部业务事件系统

定义清晰的事件类层次结构，用于Agent执行过程的实时输出。

设计原则：
1. 每种事件类型有独立的数据类（避免Optional字段冗余）
2. 所有事件包含timestamp和trace_id（便于追踪和调试）
3. 所有字段都是基础类型（str, int），易于序列化
4. 使用继承提供类型安全和代码复用
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Union


class EventType(str, Enum):
    """事件类型枚举（业务语义）"""

    # LLM交互事件
    LLM_CONTENT_DELTA = "llm_content_delta"  # LLM内容增量
    LLM_CONTENT_DONE = "llm_content_done"  # LLM内容完成
    LLM_TOOL_CALL_DONE = "llm_tool_call_done"  # LLM工具调用完成

    # 工具执行事件
    TOOL_EXECUTION_STARTED = "tool_execution_started"  # 工具开始执行
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"  # 工具执行完成

    # Agent步骤事件
    AGENT_STEP_STARTED = "agent_step_started"  # Agent步骤开始
    AGENT_STEP_COMPLETED = "agent_step_completed"  # Agent步骤完成

    # 子研究事件
    SUB_RESEARCH_STARTED = "sub_research_started"  # 子研究开始

    # 研究完成事件
    RESEARCH_COMPLETED = "research_completed"  # 研究完成（最终答案）

    # 错误事件
    ERROR_OCCURRED = "error_occurred"  # 错误发生


@dataclass
class BaseEvent:
    """内部业务事件基类

    所有事件都包含：
    - event_type: 事件类型（由子类固定）
    - timestamp: 事件发生时间（UTC）
    - trace_id: 追踪ID（用于关联同一请求的多个事件）
    """

    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于JSON输出）

        Returns:
            Dict包含event_type, timestamp, trace_id（如果有）
        """
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        return result


# ==================== LLM交互事件 ====================


@dataclass
class LLMContentDeltaEvent(BaseEvent):
    """LLM内容增量事件

    当LLM流式输出时，每个文本片段产生一个此事件。
    """

    event_type: EventType = field(default=EventType.LLM_CONTENT_DELTA, init=False)
    content: str = ""  # 增量内容片段

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content"] = self.content
        return result


@dataclass
class LLMContentDoneEvent(BaseEvent):
    """LLM内容完成事件

    当LLM完成内容输出时产生此事件（包含完整内容）。
    """

    event_type: EventType = field(default=EventType.LLM_CONTENT_DONE, init=False)
    content: str = ""  # 完整内容
    reasoning_content: Optional[str] = None  # DeepSeek Reasoner Thinking Mode 推理内容

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content"] = self.content
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result


@dataclass
class LLMToolCallDoneEvent(BaseEvent):
    """LLM工具调用完成事件

    当LLM决定调用工具时产生此事件。
    """

    event_type: EventType = field(default=EventType.LLM_TOOL_CALL_DONE, init=False)
    tool_call_id: str = ""  # 工具调用ID（OpenAI格式）
    tool_name: str = ""  # 工具名称
    tool_arguments: str = ""  # 工具参数（JSON字符串）

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "tool_call_id": self.tool_call_id,
                "tool_name": self.tool_name,
                "tool_arguments": self.tool_arguments,
            }
        )
        return result


# ==================== 工具执行事件 ====================


@dataclass
class ToolExecutionStartedEvent(BaseEvent):
    """工具开始执行事件

    当Agent开始执行工具时产生此事件。
    """

    event_type: EventType = field(default=EventType.TOOL_EXECUTION_STARTED, init=False)
    tool_call_id: str = ""  # 工具调用ID
    tool_name: str = ""  # 工具名称

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "tool_call_id": self.tool_call_id,
                "tool_name": self.tool_name,
            }
        )
        return result


@dataclass
class ToolExecutionCompletedEvent(BaseEvent):
    """工具执行完成事件

    当Agent完成工具执行时产生此事件（包含执行结果）。
    """

    event_type: EventType = field(default=EventType.TOOL_EXECUTION_COMPLETED, init=False)
    tool_call_id: str = ""  # 工具调用ID
    tool_name: str = ""  # 工具名称
    tool_output: str = ""  # 工具输出（字符串格式）

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "tool_call_id": self.tool_call_id,
                "tool_name": self.tool_name,
                "tool_output": self.tool_output,
            }
        )
        return result


# ==================== Agent步骤事件 ====================


@dataclass
class AgentStepStartedEvent(BaseEvent):
    """Agent步骤开始事件

    当Agent开始新的ReAct步骤时产生此事件。
    """

    event_type: EventType = field(default=EventType.AGENT_STEP_STARTED, init=False)
    step_number: int = 0  # 步骤编号（从1开始）

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["step_number"] = self.step_number
        return result


@dataclass
class AgentStepCompletedEvent(BaseEvent):
    """Agent步骤完成事件

    当Agent完成一个ReAct步骤时产生此事件。
    """

    event_type: EventType = field(default=EventType.AGENT_STEP_COMPLETED, init=False)
    step_number: int = 0  # 步骤编号

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["step_number"] = self.step_number
        return result


# ==================== 子研究事件 ====================


@dataclass
class SubResearchStartedEvent(BaseEvent):
    """子研究开始事件

    当 Supervisor 开始执行一个子研究时产生此事件。
    用于前端显示子研究的主题信息。
    """

    event_type: EventType = field(default=EventType.SUB_RESEARCH_STARTED, init=False)
    topic: str = ""  # 研究主题
    researcher_id: str = ""  # 研究者ID

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({"topic": self.topic, "researcher_id": self.researcher_id})
        return result


# ==================== 研究完成事件 ====================


@dataclass
class ResearchCompletedEvent(BaseEvent):
    """研究完成事件（最终答案）

    当Researcher或Supervisor完成研究并生成最终报告时产生此事件。
    """

    event_type: EventType = field(default=EventType.RESEARCH_COMPLETED, init=False)
    content: str = ""  # 最终研究报告

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content"] = self.content
        return result


# ==================== 错误事件 ====================


@dataclass
class ErrorEvent(BaseEvent):
    """错误事件

    当Agent执行过程中发生错误时产生此事件。
    """

    event_type: EventType = field(default=EventType.ERROR_OCCURRED, init=False)
    error_message: str = ""  # 错误信息
    error_type: Optional[str] = None  # 错误类型（可选）

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["error_message"] = self.error_message
        if self.error_type:
            result["error_type"] = self.error_type
        return result


# ==================== 类型别名（用于类型提示）====================

DeepResearchEvent = Union[
    LLMContentDeltaEvent,
    LLMContentDoneEvent,
    LLMToolCallDoneEvent,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
    AgentStepStartedEvent,
    AgentStepCompletedEvent,
    SubResearchStartedEvent,
    ResearchCompletedEvent,
    ErrorEvent,
]
