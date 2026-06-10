"""核心数据结构定义

完全兼容OpenAI SDK的数据结构，用于Agent系统的消息和状态管理。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(str, Enum):
    """消息角色（OpenAI兼容）"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCallFunction:
    """工具调用的函数信息（OpenAI格式）"""

    name: str
    arguments: str  # JSON字符串

    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    """工具调用记录（OpenAI格式）"""

    id: str
    type: str = "function"
    function: Optional[ToolCallFunction] = None

    def __post_init__(self):
        if self.function is None:
            self.function = ToolCallFunction(name="", arguments="{}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {"id": self.id, "type": self.type, "function": self.function.to_dict()}


@dataclass
class Message:
    """统一消息格式（完全兼容OpenAI SDK）

    设计原则：
    1. 可以直接序列化为OpenAI API需要的dict
    2. 可以从OpenAI API返回值直接构造
    """

    role: MessageRole
    content: Optional[str | List[Dict[str, Any]]] = None

    # OpenAI专用字段
    tool_calls: Optional[List[ToolCall]] = None  # assistant消息专用
    tool_call_id: Optional[str] = None  # tool消息专用
    name: Optional[str] = None  # tool消息的工具名称

    # DeepSeek Reasoner (Thinking Mode) 专用字段
    # 当使用 qwen-plus-thinking 模型时，API 会返回 reasoning_content 字段
    # 多轮对话时必须将此字段传回，否则 API 返回 400 错误
    reasoning_content: Optional[str] = None

    def to_openai_dict(self) -> Dict[str, Any]:
        """转换为OpenAI API需要的格式"""
        msg = {"role": self.role.value}

        if self.content is not None:
            msg["content"] = self.content

        # DeepSeek Reasoner (Thinking Mode) 要求在多轮对话中传回 reasoning_content
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content

        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.name:
            msg["name"] = self.name

        return msg

    @classmethod
    def from_openai_message(cls, msg: Any) -> "Message":
        """从OpenAI ChatCompletion.choices[0].message构造

        Args:
            msg: OpenAI API返回的message对象

        Returns:
            Message: 构造的Message实例
        """
        # 处理tool_calls
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]

        # 处理 DeepSeek Reasoner (Thinking Mode) 的 reasoning_content
        reasoning_content = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content

        return cls(
            role=MessageRole(msg.role),
            content=msg.content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于缓存）"""
        data: Dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            data["content"] = self.content

        # DeepSeek Reasoner (Thinking Mode) 的 reasoning_content
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content

        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id

        if self.name:
            data["name"] = self.name

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典反序列化（用于缓存）

        Args:
            data: 序列化的字典数据

        Returns:
            Message: 构造的Message实例
        """
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
                for tc in data["tool_calls"]
            ]

        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning_content=data.get("reasoning_content"),
        )


@dataclass
class AgentMemory:
    """Agent的对话历史（极简版）

    设计原则：
    1. 直接管理messages列表（更接近OpenAI SDK）
    2. 支持Compact直接修改tool消息的content
    3. 消除不必要的"步骤"抽象
    """

    system_prompt: str
    messages: List[Message] = field(default_factory=list)

    def add_message(self, message: Message):
        """添加消息到历史"""
        self.messages.append(message)

    def add_messages(self, messages: List[Message]):
        """批量添加消息（用于工具调用场景）"""
        self.messages.extend(messages)

    def to_messages(self) -> List[Message]:
        """转换为OpenAI API需要的消息列表"""
        return [Message(role=MessageRole.SYSTEM, content=self.system_prompt)] + self.messages

    def find_tool_messages(self, tool_name: Optional[str] = None) -> List[Message]:
        """找到所有tool消息（用于Compact）

        Args:
            tool_name: 可选，只返回特定工具的消息

        Returns:
            List[Message]: tool消息列表
        """
        if tool_name:
            return [m for m in self.messages if m.role == MessageRole.TOOL and m.name == tool_name]
        return [m for m in self.messages if m.role == MessageRole.TOOL]

    def find_assistant_messages(self) -> List[Message]:
        """找到所有assistant消息"""
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]

    def clear(self):
        """清空消息历史（保留system_prompt）

        用于需要清空对话历史但保持 Agent 配置的场景。
        注意：当前设计中 Agent 对象是一次性的，通常不需要调用此方法。
        """
        self.messages.clear()
