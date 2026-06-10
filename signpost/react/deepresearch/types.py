"""深度研究系统数据类型定义

重构说明：
- 直接复用 graphrag.retrieval 的数据结构，避免重复定义和数据转换开销
- ToolResponse 直接承载 KGSearchResult，保留完整的导航信息
- 移除 RetrievalGroup（与上游冲突），改用 graphrag 的 RetrievalGroup
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

# 导入 graphrag 的数据结构
from graphrag.retrieval.kg_retrieval import (
    KGSearchResult,
)


class ExitReason(str, Enum):
    """Agent 退出原因枚举"""

    MAX_ITERATIONS = "max_iterations"
    NO_TOOL_CALLS = "no_tool_calls"
    CANCELLED = "cancelled"

    @property
    def description(self) -> str:
        """返回退出原因的描述信息"""
        descriptions = {
            ExitReason.MAX_ITERATIONS: "达到最大迭代次数",
            ExitReason.NO_TOOL_CALLS: "LLM 未返回工具调用",
            ExitReason.CANCELLED: "用户取消",
        }
        return descriptions.get(self, self.value)


class ContextDecision(str, Enum):
    """上下文管理决策枚举"""

    CONTINUE = "continue"  # 可以继续迭代
    FORCE_FINISH = "force_finish"  # 必须强制生成最终答案


class TraceStatus(str, Enum):
    """追踪日志状态枚举"""

    SUCCESS = "success"  # 执行成功
    ERROR = "error"  # 执行失败
    COMPLETED = "completed"  # 研究完成
    CANCELLED = "cancelled"  # 用户取消


ToolName = Literal["knowledge_search", "read_file"]


@dataclass
class ToolResponse:
    """工具响应数据类 - 直接复用 graphrag 的数据结构

    设计原则：
    - knowledge_search: 使用 kg_result (KGSearchResult)
    - read_file: 使用 file_content_view 等字段

    优点：
    - 零转换开销
    - 保留完整的 InstanceSignpost 导航信息
    - 类型一致性，自动继承上游更新

    Attributes:
        tool: 工具名称
        query: 查询字符串
        kg_result: KGSearchResult（knowledge_search 工具使用）
        file_content_view: 格式化后的文件内容视图（read_file 工具使用，已包含行号、截断指示）
        file_name: 文件名（read_file 工具使用）
        start_line: 起始行号（read_file 工具使用）
        end_line: 结束行号（read_file 工具使用）
        total_lines: 文件总行数（read_file 工具使用）
        is_truncated: 是否截断（read_file 工具使用）
        next_offset: 下次读取offset（read_file 工具使用）
    """

    tool: ToolName
    query: str

    # knowledge_search 专用字段
    kg_result: Optional[KGSearchResult] = None

    # read_file 专用字段
    file_content_view: Optional[str] = None
    file_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    total_lines: Optional[int] = None
    is_truncated: Optional[bool] = None
    next_offset: Optional[int] = None
