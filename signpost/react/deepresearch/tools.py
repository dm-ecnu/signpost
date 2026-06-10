"""工具基类和工具实现"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration import Configuration
    from .events import DeepResearchEvent
    from core.logging.trace import TraceSession

from core import config
from graphrag.retrieval import KGSearchResult
from core.utils.knowledge_retrieval import get_kb_summary_from_es

from .types import ToolResponse
from .utils import format_file_view

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tool Execution Context
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolExecutionContext:
    """Tool execution context (unified encapsulation of dependencies)

    This context object is passed to all tools during execution, providing
    access to configuration, logging, and callback mechanisms.

    Attributes:
        tool_call_id: Unique identifier for this tool call
        trace_id: Trace ID for distributed tracing
        config: Configuration object (required, all Agents need this)
        parent_trace_session: Optional parent TraceSession (required by Supervisor)
        on_result_callback: Optional callback for tool results (used by Supervisor)
        supervisor_step_number: Current supervisor step number (for tracking researcher associations)

    Example:
        >>> context = ToolExecutionContext(
        ...     tool_call_id="tc_123",
        ...     trace_id="tr_456",
        ...     config=config,
        ...     parent_trace_session=session
        ... )
        >>> tool.execute_stream(context=context, query="test")
    """

    tool_call_id: str
    trace_id: str
    config: "Configuration"  # Required field (all Agents need this)
    parent_trace_session: Optional["TraceSession"] = None
    on_result_callback: Optional[Callable[[str, str], None]] = None
    # Supervisor step tracking (for researcher associations)
    supervisor_step_number: Optional[int] = None

    def requires_parent_session(self) -> bool:
        """Check if parent_trace_session is provided

        Returns:
            True if parent_trace_session is not None
        """
        return self.parent_trace_session is not None

    def requires_config(self) -> bool:
        """Check if config is provided

        Returns:
            True if config is not None
        """
        return self.config is not None

    def has_result_callback(self) -> bool:
        """Check if result callback is provided

        Returns:
            True if on_result_callback is not None
        """
        return self.on_result_callback is not None


# ═══════════════════════════════════════════════════════════════════════════
# Tool Base Class
# ═══════════════════════════════════════════════════════════════════════════


class Tool(ABC):
    """工具基类（简化版，兼容OpenAI tool calling）

    设计原则：
    1. 子类只需实现execute()方法
    2. 自动生成OpenAI tools格式的JSON Schema
    3. 最小化设计，不依赖其他框架
    4. 提供默认的execute_stream()适配器，统一流式接口
    5. 使用ToolExecutionContext进行依赖注入（消除hasattr）
    """

    # 子类需要定义的类属性
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}  # JSON Schema格式

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果

        注意：
        - 参数名必须与parameters中定义的一致
        - 返回值必须是字符串（会作为tool消息的content）
        - 异常应该在execute内部处理，返回错误信息字符串
        """
        raise NotImplementedError

    def execute_stream(self, context: "ToolExecutionContext", **kwargs) -> "Iterator[DeepResearchEvent]":
        """流式执行接口（默认适配器，使用上下文对象模式）

        将同步工具包装为流式接口，统一调度层处理逻辑。
        子类可重写此方法以实现真正的流式输出（如ResearchTool）。

        Args:
            context: 工具执行上下文（包含所有可选依赖）
            **kwargs: 业务参数（传递给execute()）

        Yields:
            DeepResearchEvent: 工具执行事件流
        """
        from .events import ToolExecutionStartedEvent, ToolExecutionCompletedEvent

        # 发送 started 事件
        yield ToolExecutionStartedEvent(tool_call_id=context.tool_call_id, tool_name=self.name, trace_id=context.trace_id)

        # 执行工具（捕获异常）
        try:
            result = self.execute(**kwargs)
        except Exception as e:
            # 发送错误 completed 事件
            yield ToolExecutionCompletedEvent(tool_call_id=context.tool_call_id, tool_name=self.name, tool_output=f"ERROR: {str(e)}", trace_id=context.trace_id)
            return

        # 发送成功 completed 事件
        yield ToolExecutionCompletedEvent(tool_call_id=context.tool_call_id, tool_name=self.name, tool_output=result, trace_id=context.trace_id)

    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为OpenAI tools参数格式"""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def format_for_llm(self, result: str) -> str:
        """格式化工具结果为 LLM 友好格式（子类可覆盖）

        将工具的原始输出（通常是 JSON）转换为更适合 LLM 理解的格式。
        默认实现直接返回原始结果。

        Args:
            result: execute() 返回的原始结果字符串

        Returns:
            str: 格式化后的结果，将作为 tool message 的 content

        Example:
            >>> tool = KnowledgeSearchTool(...)
            >>> raw = tool.execute(query="test")  # JSON
            >>> formatted = tool.format_for_llm(raw)  # XML or other format
        """
        return result


# ===== 格式化函数 =====


def format_tool_response_for_llm(raw_json: str) -> str | None:
    """将工具返回的 JSON 转换为精简分隔符格式

    格式设计原则：
    1. read_file: 直接返回 file_content_view（已格式化的视图）
    2. knowledge_search: 处理 KGSearchResult 字典（零转换）
       - 分组显示：text_group 和 graph_group 分别展示
       - 使用分隔符 `--- [TN] type ---` (T=text, G=graph) 区分结果
       - 使用 `>` 前缀标记导航/溯源信息（直接读取 InstanceSignpost）
       - 每个组的探索建议紧随该组结果
       - 溯源信息可直接用于 ReadFileTool

    Args:
        raw_json: 工具返回的原始 JSON 字符串

    Returns:
        str | None: 成功返回分隔符格式，失败返回 None
    """
    try:
        data = json.loads(raw_json)

        # 1. 处理 read_file 工具（file_content_view 已包含完整格式）
        if isinstance(data, dict) and data.get("tool") == "read_file":
            file_content_view = data.get("file_content_view")
            if file_content_view:
                return file_content_view
            else:
                # 文件读取失败或无内容
                file_name = data.get("file_name", "unknown")
                return f"Error: Failed to read file '{file_name}' or file is empty."

        # 2. 处理 knowledge_search 工具（包含 kg_result）
        elif isinstance(data, dict) and "kg_result" in data:
            kg_result_dict = data.get("kg_result")

            if not kg_result_dict:
                return "=== RETRIEVAL (0 results) ==="

            # 解析分组
            text_group = kg_result_dict.get("text_group", {})
            graph_group = kg_result_dict.get("graph_group", {})

            text_items = text_group.get("items", [])
            graph_items = graph_group.get("items", [])
            total_count = len(text_items) + len(graph_items)

            if total_count == 0:
                return "=== RETRIEVAL (0 results) ==="

            result_lines = [f"=== RETRIEVAL ({total_count} results) ===", ""]

            # 格式化 text_group
            if text_items:
                text_lines = _format_kg_group(
                    items=text_items,
                    group_signpost=text_group.get("group_signpost", {}),
                    group_prefix="T",
                    group_title="Text-based Results",
                    group_description="Results based on document content similarity (chunks + RAPTOR summaries)",
                )
                result_lines.extend(text_lines)
                result_lines.append("")

            # 格式化 graph_group
            if graph_items:
                graph_lines = _format_kg_group(
                    items=graph_items,
                    group_signpost=graph_group.get("group_signpost", {}),
                    group_prefix="G",
                    group_title="Graph-based Results",
                    group_description="Results from knowledge graph structure (entities + relations)",
                )
                result_lines.extend(graph_lines)

            return "\n".join(result_lines).rstrip()

        # 处理其他简单格式
        elif isinstance(data, (str, int, float, bool)):
            return str(data)

        else:
            # 其他复杂结构，转为简化字符串
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    except Exception as e:
        logger.warning("Failed to format tool response: %s, raw_json[:200]=%s", str(e), raw_json[:200] if raw_json else "None")
        return None


def _format_kg_group(
    items: list[dict],
    group_signpost: dict,
    group_prefix: str,
    group_title: str,
    group_description: str,
) -> list[str]:
    """格式化单个检索结果组（直接处理 GraphRetrievalItem 字典）

    Args:
        items: GraphRetrievalItem 字典列表
        group_signpost: GroupSignpost 字典
        group_prefix: 组前缀 ("T" for text, "G" for graph)
        group_title: 组标题
        group_description: 组描述

    Returns:
        格式化后的文本行列表
    """
    lines = []

    # 组标题
    lines.append(f"## {group_title} ({len(items)}) ##")
    # lines.append(f"> {group_description}")
    lines.append("")

    # 格式化每个 item
    for seq, item_dict in enumerate(items, start=1):
        if not isinstance(item_dict, dict):
            continue

        item_type = item_dict.get("type", "unknown")
        chunk_id = f"{group_prefix}{seq}"

        # 根据类型分派（直接处理 GraphRetrievalItem 字典）
        if item_type == "original_chunk":
            chunk_str = _format_original_chunk_from_dict(item_dict, chunk_id)
        elif item_type == "graphrag_entity":
            chunk_str = _format_entity_from_dict(item_dict, chunk_id)
        elif item_type == "graphrag_edge":
            chunk_str = _format_edge_from_dict(item_dict, chunk_id)
        elif item_type == "raptor_node":
            chunk_str = _format_raptor_from_dict(item_dict, chunk_id)
        else:
            chunk_str = _format_unknown_from_dict(item_dict, item_type, chunk_id)

        lines.append(chunk_str)
        lines.append("")  # 空行分隔

    # 该组的探索建议（PPR 算法推荐的相关实体）
    related_entities = group_signpost.get("related_entities", [])
    if related_entities:
        lines.append("--- RELATED ENTITIES IN KNOWLEDGE BASE ---")
        lines.append(f"> The knowledge base also contains information about: {', '.join(related_entities)}")
        lines.append("")

    return lines


# ===== GraphRetrievalItem 字典格式化函数 =====


def _format_raptor_from_dict(item: dict, seq: str | int) -> str:
    """格式化 RAPTOR 节点（从 GraphRetrievalItem 字典）"""
    title = item.get("title", "")
    content = item.get("content", "")
    signpost = item.get("signpost", {})

    lines = [f"--- [{seq}] CHAPTER_SUMMARY ---"]
    if title:
        lines.append(f"Title: {title}")
    if content:
        lines.append(f"Content: {content}")

    # 层次导航（只显示有 title 的节点，不显示 node_id）
    parent_title = signpost.get("parent_node_title")
    child_titles = signpost.get("child_node_titles", [])

    if parent_title:
        lines.append(f"> Belongs to: {parent_title}")

    # 只显示有 title 的子章节
    display_titles = [t for t in child_titles if t]
    if display_titles:
        lines.append(f"> Sub-sections: {', '.join(display_titles)}")

    # 源定位
    source_locates = signpost.get("source_locates", [])
    if source_locates:
        lines.append(f"> sources: {', '.join(source_locates)}")

    return "\n".join(lines)


def _format_entity_from_dict(item: dict, seq: str | int) -> str:
    """格式化实体（从 GraphRetrievalItem 字典）"""
    title = item.get("title", "")
    content = item.get("content", "")

    lines = [f"--- [{seq}] ENTITY ---"]
    if title:
        lines.append(f"Title: {title}")
    if content:
        lines.append(f"Content: {content}")

    return "\n".join(lines)


def _format_edge_from_dict(item: dict, seq: str | int) -> str:
    """格式化关系边（从 GraphRetrievalItem 字典）"""
    title = item.get("title", "")
    content = item.get("content", "")

    lines = [f"--- [{seq}] EDGE ---"]
    if title:
        lines.append(f"Title: {title}")
    if content:
        lines.append(f"Content: {content}")

    return "\n".join(lines)


def _format_original_chunk_from_dict(item: dict, seq: str | int) -> str:
    """格式化原始 chunk（从 GraphRetrievalItem 字典）"""
    content = item.get("content", "")
    signpost = item.get("signpost", {})

    # 从 signpost 提取位置信息（需要转为 int，因为 JSON 中可能是字符串）
    file_name = signpost.get("file_name") or "unknown"
    start_line = int(signpost.get("start_line") or 1)
    end_line = int(signpost.get("end_line") or 0)
    total_lines = int(signpost.get("total_lines") or 0)

    # 使用 format_file_view 格式化（与 ReadFileTool 一致）
    view = format_file_view(
        file_name=file_name,
        raw_text=content,
        start_line=start_line if start_line > 0 else 1,
        end_line=end_line if end_line >= start_line else start_line - 1,
        total_lines=total_lines,
    )

    lines = [f"--- [{seq}] CHUNK ---"]
    lines.append(view["title"])
    if view["content"]:
        lines.append(view["content"])

    return "\n".join(lines)


def _format_unknown_from_dict(item: dict, item_type: str, seq: str | int) -> str:
    """格式化未知类型（从 GraphRetrievalItem 字典）"""
    content = item.get("content", "")
    title = item.get("title", "")

    lines = [f"--- [{seq}] {item_type} ---"]
    if title:
        lines.append(title)
    lines.append(content[:500] if len(content) > 500 else content)

    return "\n".join(lines)


# ===== 工具实现 =====


class KnowledgeSearchTool(Tool):
    """知识库向量检索工具"""

    name = "knowledge_search"
    # V2架构不支持社区检索，动态生成描述
    description = "Search the knowledge base using vector retrieval, supporting original chunks, GraphRAG entities/edges, and RAPTOR nodes"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query string"}},
        "required": ["query"],
    }

    def __init__(self, default_kb_ids: list[str] | None = None, tenant_id: str | None = None) -> None:
        super().__init__()
        self.default_kb_ids = default_kb_ids or []
        self.tenant_id = tenant_id  # 不做默认值处理，None 就让它崩溃

    def execute(self, query: str) -> str:
        """执行知识库向量检索，直接返回 KGSearchResult

        返回结果：直接包含 KGSearchResult，零转换开销

        Raises:
            RuntimeError: kg_retrievaler 未初始化
            ValueError: default_kb_ids 为空
            其他异常: 检索过程中的任何错误都会直接抛出
        """
        # Fail-Fast: 配置错误直接抛异常
        if not config.kg_retrievaler:
            raise RuntimeError("kg_retrievaler not initialized")

        if not self.default_kb_ids:
            raise ValueError("default_kb_ids is empty")

        # 执行向量检索（任何异常都会直接抛出）
        kg_result: KGSearchResult = config.kg_retrievaler.process(
            query=query,
            tenant_id=self.tenant_id,
            kb_ids=self.default_kb_ids,
            similarity_threshold=0.2,
        )

        # 直接返回，无需转换
        retrieval_result = ToolResponse(
            tool="knowledge_search",
            query=query,
            kg_result=kg_result,
        )

        return json.dumps(asdict(retrieval_result), ensure_ascii=False)

    def format_for_llm(self, result: str) -> str:
        """将 JSON 结果格式化为精简 XML 格式

        Args:
            result: execute() 返回的 JSON 字符串

        Returns:
            str: 精简的 XML 格式，减少 token 占用
        """
        formatted = format_tool_response_for_llm(result)
        if formatted is not None:
            return formatted
        return result


class ReadFileTool(Tool):
    """文件读取工具类。

    通过预配置的知识库读取文件内容。支持分页读取和cat -n格式的行号显示。

    Attributes:
        name: 工具名称，固定为"read_file"
        description: 工具描述信息
        parameters: 输入参数定义（JSON Schema格式）
    """

    # 格式化常量
    MAX_TOKENS = 2000  # 最大 token 限制
    MAX_LINE_CHARS = 2000  # 单行最大字符数限制
    MAX_LIMIT = 500  # 最大读取行数限制（防止读取过大文件）

    name = "read_file"
    description = "Reads and returns the content of a specified text file from knowledge base. Supports pagination and cat -n style line numbering."
    parameters = {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "File name to read",
            },
            "offset": {
                "type": "integer",
                "description": "Optional: The 0-based line number to start reading from. Use with 'limit' for pagination.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional: Maximum number of lines to read. Use with 'offset' for pagination. Default is 200.",
            },
        },
        "required": ["file_name"],
    }

    def __init__(self, kb_id: str, tenant_id: str) -> None:
        """初始化文件读取工具。

        Args:
            kb_id: 知识库ID（UUID格式）
            tenant_id: 租户ID（必填）
        """
        super().__init__()
        self.kb_id = kb_id
        self.tenant_id = tenant_id

    def execute(self, file_name: str, offset: int = 0, limit: int = 200) -> str:
        """读取文件内容，返回 ToolResponse 格式的 JSON。

        - 存储层读取与行窗口切片由 get_file_content_by_kb_and_filename 完成。
        - 格式化逻辑（带行号、token 限制、截断指示）在工具层处理。
        - 返回 ToolResponse，file_content_view 包含已格式化的视图。
        """
        # 参数验证
        if not file_name or not file_name.strip():
            error_result = ToolResponse(tool="read_file", query="")
            return json.dumps(asdict(error_result), ensure_ascii=False)
        if offset < 0:
            error_result = ToolResponse(tool="read_file", query="")
            return json.dumps(asdict(error_result), ensure_ascii=False)
        if limit is not None and limit <= 0:
            error_result = ToolResponse(tool="read_file", query="")
            return json.dumps(asdict(error_result), ensure_ascii=False)

        # 修复：添加 limit 上限检查（防止读取过大文件）
        if limit is not None and limit > self.MAX_LIMIT:
            logger.warning("Limit exceeded maximum: limit=%d, max=%d, truncated", limit, self.MAX_LIMIT)
            limit = self.MAX_LIMIT

        try:
            # 存储层：解码并按行窗口切片（使用 kb_id 版本）
            from core.utils.file_utils import get_file_content_by_kb_id_and_filename

            file_result = get_file_content_by_kb_id_and_filename(
                self.kb_id.strip(),
                file_name.strip(),
                tenant_id=self.tenant_id,
                offset=offset,
                limit=limit,
            )

            # 若失败，返回空结果
            if not file_result.success:
                error_result = ToolResponse(tool="read_file", query=file_name)
                return json.dumps(asdict(error_result), ensure_ascii=False)

            # 基本信息
            total_lines = file_result.total_lines or 0
            visible_start = file_result.start_line or 1
            visible_end_theoretical = file_result.end_line or (visible_start - 1)

            # 处理窗口内容
            raw_text = file_result.content or ""
            # 使用公共格式化（行号+截断）生成视图
            view = format_file_view(
                file_name=file_result.file_name,
                raw_text=raw_text,
                start_line=visible_start,
                end_line=visible_end_theoretical,
                total_lines=total_lines,
                max_tokens=self.MAX_TOKENS,
                max_line_chars=self.MAX_LINE_CHARS,
            )

            # 构建 ToolResponse（使用 read_file 专用字段）
            result = ToolResponse(
                tool="read_file",
                query=file_name,
                # read_file 专用字段（view["content"] 已包含行号、截断指示等完整格式）
                file_content_view=view["content"],
                file_name=file_result.file_name,
                start_line=view["visible_start"],
                end_line=view["visible_end"],
                total_lines=total_lines,
                is_truncated=view["is_truncated"],
                next_offset=view["next_offset"],
            )

            return json.dumps(asdict(result), ensure_ascii=False)

        except Exception as e:
            logger.error("ReadFileTool execution failed: file_name=%s, offset=%d, limit=%s, error=%s", file_name, offset, limit, str(e), exc_info=True)
            error_result = ToolResponse(tool="read_file", query=file_name or "")
            return json.dumps(asdict(error_result), ensure_ascii=False)

    def format_for_llm(self, result: str) -> str:
        """提取已格式化的文件内容视图

        Args:
            result: execute() 返回的 JSON 字符串

        Returns:
            str: 格式化后的文件内容（已包含行号、截断指示）

        Note:
            format_file_view() 已生成最终格式，此处只提取 file_content_view 字段
        """
        formatted = format_tool_response_for_llm(result)
        if formatted is not None:
            return formatted
        return result


class GetTOCTool(Tool):
    """获取文档目录结构的工具

    允许 Agent 快速了解文档的章节组织结构，无需读取全文。
    TOC 以 Markdown 格式返回，层级使用 # 符号表示。
    """

    name = "get_toc"
    description = (
        "Get the Table of Contents (TOC) of a document. "
        "Returns a Markdown-formatted hierarchical structure of chapters and sections. "
        "Use this to understand document organization before reading specific sections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": ("File name to get TOC for."),
            }
        },
        "required": ["file_name"],
    }

    def __init__(self, kb_id: str, tenant_id: str) -> None:
        """初始化 TOC 获取工具

        Args:
            kb_id: 知识库ID（UUID格式）
            tenant_id: 租户ID（必填）
        """
        super().__init__()
        self.kb_id = kb_id
        self.tenant_id = tenant_id

    def execute(self, file_name: str) -> str:
        """获取文档的 TOC

        Args:
            file_name: 文件名

        Returns:
            str: JSON 格式的结果，包含 toc 内容或错误信息
        """
        from core.utils.file_utils import get_toc_by_kb_id_and_filename

        if not file_name or not file_name.strip():
            return json.dumps({"error": "file_name cannot be empty", "toc": None}, ensure_ascii=False)

        result = get_toc_by_kb_id_and_filename(
            self.kb_id.strip(),
            file_name.strip(),
            tenant_id=self.tenant_id,
        )

        if not result.success:
            return json.dumps({"error": result.error_message, "toc": None}, ensure_ascii=False)

        toc_content = result.content or ""
        section_count = len([line for line in toc_content.splitlines() if line.strip()])

        return json.dumps(
            {
                "file_name": result.file_name,
                "total_sections": section_count,
                "toc": toc_content,
            },
            ensure_ascii=False,
        )

    def format_for_llm(self, result: str) -> str:
        """TOC 本身就是 Markdown，直接提取返回（清理文件名）

        Args:
            result: execute() 返回的 JSON 字符串

        Returns:
            str: 格式化后的 TOC 或错误信息
        """
        try:
            data = json.loads(result)
            if data.get("toc"):
                from .utils import clean_filename_for_llm

                # 清理技术标识符（_toc），提升 LLM 可读性
                file_name = clean_filename_for_llm(data.get("file_name", "unknown"))
                total_sections = data.get("total_sections", 0)
                return f"**Document TOC** ({file_name}, {total_sections} sections):\n\n{data['toc']}"
            elif data.get("error"):
                return f"Error: {data['error']}"
        except Exception:
            pass
        return result


class KnowledgeOverviewTool(Tool):
    """获取知识库概览的工具

    从 ES 获取 Lazy Merge 生成的 KB Summary（预生成的知识库级别摘要）。
    如果 KB Summary 不存在（尚未执行 Lazy Merge），返回空字符串。
    """

    name = "knowledge_overview"
    description = "Get knowledge base overview to understand its overall content structure"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, kb_id: str, tenant_id: str) -> None:
        super().__init__()
        self.kb_id = kb_id
        self.tenant_id = tenant_id  # 不做默认值处理

    def execute(self) -> str:
        """获取知识库概览（从 ES 查询预生成的 KB Summary）"""
        try:
            kb_summary = get_kb_summary_from_es(self.kb_id, self.tenant_id)
            return kb_summary or ""
        except Exception as e:
            logger.warning(f"Failed to get KB Summary for kb {self.kb_id}: {e}")
            return ""


class ResearchCompleteTool(Tool):
    """研究完成信号工具

    该工具仅作为完成信号，不会被实际执行。当 Agent 检测到此工具调用时，
    会直接触发 generate_final_answer() 生成最终报告，而非执行此工具。

    设计说明：
    - is_final_answer_signal() 返回 True 时，Agent 跳过工具执行
    - execute() 不应该被调用，如果被调用说明存在逻辑错误
    """

    name = "research_complete"
    description = "Call this tool to indicate that the research is complete."
    parameters = {"type": "object", "properties": {}}

    def execute(self) -> str:
        """此方法不应该被调用

        Raises:
            RuntimeError: 始终抛出异常，提示开发者检查代码逻辑
        """
        raise RuntimeError(
            "ResearchCompleteTool.execute() should never be called. "
            "This tool is a signal-only tool that triggers generate_final_answer() "
            "instead of being executed. If you see this error, check the Agent's "
            "is_final_answer_signal() implementation."
        )


class ResearchTool(Tool):
    """研究工具（SubAgent 工具）

    设计说明：
    - 实现 execute_stream() 返回生成器
    - 发送完整的事件流（started -> 中间事件 -> completed）
    - 通过回调保存研究结果
    - execute() 提供回退实现（返回友好错误）
    - 使用ToolExecutionContext获取必需依赖

    使用流程：
    1. ReActAgent._execute_tools() 调用 execute_stream()，传入上下文
    2. 在工具线程中执行，从上下文获取config和parent_trace_session
    3. 消费生成器，实时推送事件到队列
    4. 主线程消费队列，实时 yield 事件
    """

    name = "research"
    description = "Conduct deep research on a single topic, can be called multiple times in parallel for different topics"
    parameters = {
        "type": "object",
        "properties": {"topic": {"type": "string", "description": "The specific topic or query to research"}},
        "required": ["topic"],
    }

    def execute(self, topic: str) -> str:
        """回退实现（兼容性）

        如果此方法被调用，说明调用方没有使用 execute_stream()。
        返回友好的错误提示而非抛异常。
        """
        return json.dumps({"error": "ResearchTool requires execute_stream interface", "hint": "Use execute_stream() with ToolExecutionContext", "topic": topic}, ensure_ascii=False)

    def execute_stream(
        self,
        context: "ToolExecutionContext",
        topic: str,
    ) -> "Iterator[DeepResearchEvent]":
        """流式执行研究（生成器接口，使用上下文对象模式）

        Args:
            context: 工具执行上下文（必需包含 config 和 parent_trace_session）
            topic: 研究主题

        Yields:
            DeepResearchEvent: 事件流

        设计要点：
        1. 从context显式获取依赖（config、parent_trace_session、on_result）
        2. 发送完整的事件流（started -> 中间 -> completed）
        3. 捕获 RESEARCH_COMPLETED 事件，提取结果
        4. 通过回调保存结果
        5. 异常会被外层捕获（工具线程）

        注意：此方法在工具线程中执行
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            pass

        from .events import SubResearchStartedEvent, ToolExecutionStartedEvent, ToolExecutionCompletedEvent, EventType
        from .researcher import Researcher

        # 0. 检查必需依赖（显式且类型安全）
        if not context.requires_config():
            error_msg = "ResearchTool requires config in context"
            yield ToolExecutionStartedEvent(tool_call_id=context.tool_call_id, tool_name="research", trace_id=context.trace_id)
            yield ToolExecutionCompletedEvent(tool_call_id=context.tool_call_id, tool_name="research", tool_output=f"ERROR: {error_msg}", trace_id=context.trace_id)
            return

        if not context.requires_parent_session():
            error_msg = "ResearchTool requires parent_trace_session in context"
            yield ToolExecutionStartedEvent(tool_call_id=context.tool_call_id, tool_name="research", trace_id=context.trace_id)
            yield ToolExecutionCompletedEvent(tool_call_id=context.tool_call_id, tool_name="research", tool_output=f"ERROR: {error_msg}", trace_id=context.trace_id)
            return

        # 提取上下文依赖
        config = context.config
        parent_trace_session = context.parent_trace_session
        on_result = context.on_result_callback

        # 1. 发送 started 事件
        yield ToolExecutionStartedEvent(tool_call_id=context.tool_call_id, tool_name="research", trace_id=context.trace_id)

        # 2. 创建 Researcher
        topic = topic.strip()
        researcher_id = f"researcher_{abs(hash(topic)) % 10000:04x}"

        # 创建 emitter 时记录 parent_supervisor_step
        metadata = {"research_topic": topic}
        if context.supervisor_step_number is not None:
            metadata["parent_supervisor_step"] = context.supervisor_step_number

        researcher_emitter = parent_trace_session.create_emitter(agent_id=researcher_id, agent_type="Researcher", **metadata)

        researcher = Researcher(
            config=config,
            trace_emitter=researcher_emitter,
            researcher_id=researcher_id,
        )

        # 3. 构造子 trace_id
        child_trace_id = f"{context.trace_id}.{researcher_id}"

        # 4. 发送子研究开始事件
        yield SubResearchStartedEvent(topic=topic, researcher_id=researcher_id, trace_id=child_trace_id)

        # 5. 透传所有 Researcher 的事件（核心逻辑）
        accumulated_result = ""
        for event in researcher(topic, trace_id=child_trace_id):
            yield event  # 实时透传

            # 捕获最终结果
            if event.event_type == EventType.RESEARCH_COMPLETED:
                accumulated_result = event.content

        # 6. 写入 Researcher 的 metadata
        researcher_emitter.write_metadata()

        # 7. 保存研究结果（通过回调）
        if on_result and accumulated_result:
            on_result(topic, accumulated_result)  # 工具线程内调用

        # 8. 格式化最终结果
        formatted_result = f"""**Research Topic**: {topic}

**Findings**:
{accumulated_result}

**Status**: Completed"""

        # 9. 发送 completed 事件
        yield ToolExecutionCompletedEvent(tool_call_id=context.tool_call_id, tool_name="research", tool_output=formatted_result, trace_id=context.trace_id)
