"""DeepResearch V2 - 基于OpenAI SDK的深度研究Agent系统

这是DeepResearch的重构版本，移除了smolagents依赖，完全基于OpenAI SDK实现。

核心特性：
- 双层Agent架构（Supervisor + Researcher）
- 流式输出优先（只提供run_stream接口）
- 简化的代码结构（无回调、无拦截）
- 内置Compact记忆压缩
- 完整的上下文长度管理

公开接口：
- Configuration: 配置管理
- Researcher: 研究者Agent
- Supervisor: 监督者Agent
- run_research_cli: CLI入口函数
- run_research_web: Web后端入口函数（兼容agentic_reasoning）
- 内部业务事件类: EventType, BaseEvent, LLMContentDeltaEvent, 等
"""

from .configuration import Configuration
from .events import (
    # 事件类型枚举
    EventType,
    # 事件基类
    BaseEvent,
    # 具体事件类
    LLMContentDeltaEvent,
    LLMContentDoneEvent,
    LLMToolCallDoneEvent,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
    AgentStepStartedEvent,
    AgentStepCompletedEvent,
    ResearchCompletedEvent,
    ErrorEvent,
    # 类型别名
    DeepResearchEvent,
)

# 入口函数
from .cli_runner import run_research_cli
from .web_runner import run_research_web, adapt_event_to_web_format

# Agent类（按需导入）
from .researcher import Researcher
from .supervisor import Supervisor

__all__ = [
    # 配置
    "Configuration",
    # 事件系统
    "EventType",
    "BaseEvent",
    "LLMContentDeltaEvent",
    "LLMContentDoneEvent",
    "LLMToolCallDoneEvent",
    "ToolExecutionStartedEvent",
    "ToolExecutionCompletedEvent",
    "AgentStepStartedEvent",
    "AgentStepCompletedEvent",
    "ResearchCompletedEvent",
    "ErrorEvent",
    "DeepResearchEvent",
    # Agent类
    "Researcher",
    "Supervisor",
    # 入口函数
    "run_research_cli",
    "run_research_web",
    "adapt_event_to_web_format",
]

__version__ = "2.0.0"
