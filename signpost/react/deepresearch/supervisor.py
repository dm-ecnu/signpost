"""Supervisor Agent实现"""

import logging
from typing import Iterator, Optional, TYPE_CHECKING

from .agent import ReActAgent
from .configuration import Configuration
from .entities import Message, MessageRole
from .events import DeepResearchEvent, EventType, ResearchCompletedEvent
from .model_client import create_model, stream_events
from .template_loader import (
    load_templates,
    render_template,
    get_today_str,
    get_default_variables,
)
from .tools import ResearchCompleteTool, ResearchTool, KnowledgeOverviewTool

if TYPE_CHECKING:
    from core.logging.trace import TraceSession

logger = logging.getLogger(__name__)


# ===== Supervisor Agent =====


class Supervisor(ReActAgent):
    """Research Supervisor Agent (coordinates multiple sub-researches)

    Responsibilities:
    - Decompose complex research tasks into multiple sub-topics
    - Dispatch Researcher to execute individual sub-research
    - Aggregate all research results to generate comprehensive report

    Design principles (v3 simplified architecture):
    - No need to override _execute_tools() (base class handles serial and parallel uniformly)
    - Provides callback to save research results (_on_tool_result)
    - Fully reuses base class unified execution logic
    - Streaming final comprehensive report generation
    """

    def __init__(
        self,
        config: Configuration,
        task: str,
        parent_trace_session: Optional["TraceSession"] = None,
    ):
        """Initialize Supervisor

        Args:
            config: Configuration object
            task: The research task/question
            parent_trace_session: Parent TraceSession (required for creating TraceEmitter)
        """
        # 创建自己的 TraceEmitter
        trace_emitter = parent_trace_session.create_emitter(agent_id="supervisor", agent_type="Supervisor")

        # 创建工具列表
        tools = [
            ResearchTool(),
            ResearchCompleteTool(),
        ]

        # 创建模型客户端
        model = create_model(config)

        # 从 YAML 模板加载系统提示词
        templates = load_templates(profile="supervisor", language=config.prompt_language)

        # 获取知识库概览
        try:
            overview_tool = KnowledgeOverviewTool(kb_id=config.kb_id, tenant_id=config.tenant_id)
            knowledge_overview = overview_tool.execute()
        except Exception as e:
            logger.warning("Failed to fetch knowledge base overview: %s, using default", str(e))
            knowledge_overview = f"Knowledge Base ID: {config.kb_id}"

        # Build template variables
        template_variables = {
            "knowledge_overview": knowledge_overview,
            "date": get_today_str(),
        }
        template_variables.update(get_default_variables())

        # Render system prompt (fill variables)
        system_prompt = render_template(
            templates["system_prompt"],
            variables=template_variables,
        )

        # 初始化基类
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            trace_emitter=trace_emitter,
            config=config,
            parent_trace_session=parent_trace_session,
            max_iterations=config.max_researcher_iterations,
            max_parallel_tools=config.max_parallel_tools,
        )

        # 保存模板和研究结果
        self.templates = templates
        self.research_results = {}  # topic -> result
        # 保存原始任务和模板变量供 generate_final_answer 使用
        self.task = task
        self._template_variables = template_variables

    def is_final_answer_signal(self, tool_name: str) -> bool:
        """判断是否为终止信号

        Args:
            tool_name: 工具名称

        Returns:
            bool: 是否为research_complete工具
        """
        return tool_name == "research_complete"

    def _on_tool_result(self, key: str, value: str) -> None:
        """接收工具结果并保存到 research_results（回调方法）

        此回调在工具线程内调用，用于接收 ResearchTool 保存的研究结果。

        Args:
            key: 研究主题（topic）
            value: 研究结果（accumulated_result）

        注意：
        - 此方法在工具线程内调用
        - research_results 的写入是线程安全的（dict 单键写入是原子操作）
        """
        self.research_results[key] = value
        logger.info("Research result saved: key=%s, length=%d chars", key, len(value))

    def generate_final_answer(self) -> Iterator[DeepResearchEvent]:
        """Generate comprehensive research report (streaming)

        This method will:
        1. Build prompt with research_brief (findings are already in conversation history)
        2. Call LLM to generate comprehensive report
        3. Pass through content events
        4. Finally send final_answer event

        Yields:
            DeepResearchEvent: Events during report generation
        """
        # Load report generation prompt from template (required field)
        report_template = self.templates["final_report_generation_prompt"].strip()

        # Get original task as research brief
        research_brief = self.task

        # Render template with Jinja (findings removed - already in conversation history)
        report_variables = {
            **self._template_variables,
            "research_brief": research_brief,
        }
        report_prompt = render_template(report_template, variables=report_variables)

        # 添加报告生成请求到memory
        self.memory.add_message(Message(role=MessageRole.USER, content=report_prompt))

        # 获取完整对话历史
        messages = self.memory.to_messages()

        # 调用LLM生成报告（流式）- 使用 llm_span 记录
        accumulated_report = ""

        with self.trace_emitter.llm_span(
            model_id=self.config.model_id,
            messages=[m.to_openai_dict() if hasattr(m, "to_openai_dict") else {"role": str(m.role), "content": m.content} for m in messages],
            tools=None,
            max_tokens=self.config.max_tokens,
            stream=True,
            step_id=f"{self.trace_emitter.trace_id[:8]}.{self.trace_emitter.agent_id}.final_report",
        ) as llm:
            for event in stream_events(self.model, messages, trace_id=self._trace_id):
                if event.event_type == EventType.LLM_CONTENT_DELTA:
                    accumulated_report += event.content
                    yield event  # 透传增量事件
                elif event.event_type == EventType.LLM_CONTENT_DONE:
                    yield event  # 透传完成事件

            # Set LLM span result
            llm.set_result(content=accumulated_report, finish_reason="stop")

        # 将最终报告保存到 memory（修复报告丢失问题）
        if accumulated_report:
            self.memory.add_message(Message(role=MessageRole.ASSISTANT, content=accumulated_report))

        # 发送最终答案事件
        yield ResearchCompletedEvent(content=accumulated_report or "Comprehensive research completed", trace_id=self._trace_id)

    # ===== 钩子方法实现：上下文管理（新架构） =====

    def _get_final_answer_tool_name(self) -> Optional[str]:
        """返回 Supervisor 的终止工具名称

        Returns:
            str: "research_complete"
        """
        return "research_complete"

    def _on_attempt_context_compress(self) -> bool:
        """钩子方法：Supervisor 的压缩实现（暂不实现，后续优化）

        当前策略：
        - 直接返回 False，表示不支持压缩
        - 超过阈值时会直接触发强制生成最终答案

        未来可以实现的压缩策略：
        - 删除最早的子研究结果
        - 压缩知识库概览
        - 摘要化 ResearchTool 的返回结果

        Returns:
            bool: 始终返回 False（不支持压缩）
        """
        logger.info("Supervisor does not support context compaction, forcing final answer generation")
        return False
