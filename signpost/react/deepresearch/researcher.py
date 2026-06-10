"""Researcher Agent实现"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

from .agent import ReActAgent
from .configuration import Configuration
from .entities import Message, MessageRole
from .events import DeepResearchEvent, EventType, ResearchCompletedEvent
from .model_client import create_model, stream_events, chat_sync
from .template_loader import (
    load_templates,
    render_template,
    get_today_str,
)
from .tools import GetTOCTool, KnowledgeSearchTool, ReadFileTool, ResearchCompleteTool

if TYPE_CHECKING:
    from core.logging.trace import TraceEmitter

logger = logging.getLogger(__name__)


# ===== Helper Functions =====


def _extract_chunks_from_tool_response(data: dict) -> list:
    """从工具响应中提取所有 items（直接处理 KGSearchResult）

    新版格式（kg_result）：
    {
        "kg_result": {
            "text_group": {"items": [...], ...},
            "graph_group": {"items": [...], ...}
        }
    }

    Args:
        data: 工具响应的 JSON 字典

    Returns:
        list: 所有 GraphRetrievalItem 字典的列表
    """
    items = []

    kg_result = data.get("kg_result")
    if not kg_result:
        return items

    text_group = kg_result.get("text_group", {})
    graph_group = kg_result.get("graph_group", {})

    items.extend(text_group.get("items", []))
    items.extend(graph_group.get("items", []))

    return items


def _update_chunks_in_tool_response(data: dict, new_items: list) -> None:
    """更新工具响应中的 items（保留分组结构）

    策略：按照原有的分组比例分配 new_items

    Args:
        data: 工具响应的 JSON 字典（会被原地修改）
        new_items: 新的 items 列表（已按分数排序）
    """
    kg_result = data.get("kg_result")
    if not kg_result:
        return

    text_group = kg_result.get("text_group", {})
    graph_group = kg_result.get("graph_group", {})

    # 计算原始分组的比例
    text_count = len(text_group.get("items", []))
    graph_count = len(graph_group.get("items", []))
    total_count = text_count + graph_count

    if total_count == 0:
        return

    # 按比例分配新的 items
    text_ratio = text_count / total_count
    text_new_count = int(len(new_items) * text_ratio)

    text_new_items = new_items[:text_new_count]
    graph_new_items = new_items[text_new_count:]

    # 更新分组
    if text_group:
        text_group["items"] = text_new_items
    if graph_group:
        graph_group["items"] = graph_new_items


# ===== Researcher Agent =====


class Researcher(ReActAgent):
    """研究者Agent - 纯粹的信息收集者

    职责：
    - 执行单个研究主题的深度研究
    - 调用知识检索和文件读取工具
    - 生成研究报告

    设计原则：
    - 继承ReActAgent的完整流式执行逻辑
    - 实现两个抽象方法：is_final_answer_signal 和 generate_final_answer
    - Researcher 不关心问题类型，只专注于信息收集
    - Supervisor 负责提供自包含的、清晰的 topic
    """

    def __init__(
        self,
        config: Configuration,
        trace_emitter: "TraceEmitter",
        researcher_id: str = "researcher",
    ):
        """Initialize Researcher

        Args:
            config: Configuration object
            trace_emitter: Trace logger (required)
            researcher_id: Researcher ID for logging
        """
        # 创建工具列表（传递必要的配置参数）
        tools = [
            KnowledgeSearchTool(
                default_kb_ids=config.get_kb_ids(),
                tenant_id=config.tenant_id,
            ),
            ReadFileTool(
                kb_id=config.kb_id,
                tenant_id=config.tenant_id,
            ),
            GetTOCTool(
                kb_id=config.kb_id,
                tenant_id=config.tenant_id,
            ),
            ResearchCompleteTool(),
        ]

        # 创建模型客户端
        model = create_model(config)

        # 从 YAML 模板加载系统提示词
        templates = load_templates(profile="researcher", language=config.prompt_language)

        # 构建模板变量（Researcher 不需要 benchmark 相关变量）
        template_variables = {
            "date": get_today_str(),
        }

        # 渲染系统提示词
        system_prompt = render_template(
            templates["system_prompt"],
            variables=template_variables,
        )

        # 初始化基类（传递 config 和 trace_emitter）
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            trace_emitter=trace_emitter,
            config=config,  # Pass to base class
            parent_trace_session=None,  # Researcher does not create nested Agents
            max_iterations=config.max_react_tool_calls,
            max_parallel_tools=config.max_parallel_tools,
        )

        self.researcher_id = researcher_id
        self.templates = templates  # 保存模板供后续使用
        # 保存模板变量供 generate_final_answer 使用
        self._template_variables = template_variables

        # Compact 相关状态
        self.compacted_message_ids: set[str] = set()  # 记录已压缩的消息ID（tool_call_id）

    def is_final_answer_signal(self, tool_name: str) -> bool:
        """判断是否为终止信号

        Args:
            tool_name: 工具名称

        Returns:
            bool: 是否为research_complete工具
        """
        return tool_name == "research_complete"

    def generate_final_answer(self) -> Iterator[DeepResearchEvent]:
        """Generate final research report (streaming)

        This method will:
        1. Call LLM to generate final report (based on memory history)
        2. Pass through content events
        3. Finally send final_answer event

        Yields:
            DeepResearchEvent: Events during report generation
        """
        # Load report generation prompt from template (required field)
        report_template = self.templates["final_report_generation_prompt"].strip()

        # Get context info for current research task
        research_brief = "Single topic research completed"
        findings = "Research findings available in conversation history above"

        date = datetime.now().strftime("%Y-%m-%d")

        # Render template with Jinja
        report_variables = {
            **self._template_variables,
            "research_brief": research_brief,
            "findings": findings,
            "date": date,
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
        yield ResearchCompletedEvent(content=accumulated_report or "Research completed", trace_id=self._trace_id)

    # ===== 钩子方法实现：上下文管理（新架构） =====

    def _get_final_answer_tool_name(self) -> Optional[str]:
        """返回 Researcher 的终止工具名称

        Returns:
            str: "research_complete"
        """
        return "research_complete"

    def _on_attempt_context_compress(self) -> bool:
        """Researcher 的压缩实现：删除低分 chunks

        流程：
        1. 记录压缩前的 token 数
        2. 调用具体的压缩逻辑（_perform_compress）
        3. 计算释放的 token 数
        4. 判断压缩效果是否达标

        Returns:
            bool: 压缩是否成功（释放了足够的空间）
        """
        try:
            # 1. 记录压缩前的 token 数
            before_tokens = self._estimate_memory_tokens()

            # 2. 执行压缩（调用具体逻辑）
            compress_performed = self._perform_compress()

            if not compress_performed:
                logger.info("Compaction skipped: no compressible content")
                return False

            # 3. 计算压缩效果
            after_tokens = self._estimate_memory_tokens()
            saved_tokens = before_tokens - after_tokens

            # 4. 判断压缩是否达标
            min_reduction = int(before_tokens * self.config.compress_min_reduction_ratio)

            if saved_tokens >= min_reduction:
                logger.info("Compaction successful: freed %d tokens (required >= %d)", saved_tokens, min_reduction)
                return True
            else:
                logger.warning("Compaction effect insufficient: freed only %d tokens (required >= %d)", saved_tokens, min_reduction)
                return False

        except Exception as e:
            logger.error("Compaction failed: %s", str(e), exc_info=True)
            return False

    def _perform_compress(self) -> bool:
        """执行具体的压缩逻辑（删除低分 chunks）

        这是 Researcher 特有的压缩策略：
        1. 收集所有检索消息
        2. 提取 chunks
        3. LLM 评分
        4. 删除低分 chunks
        5. 记录 Compact 日志

        Returns:
            bool: 是否执行了压缩（False 表示没有可压缩内容）
        """
        try:
            compact_start_time = datetime.now(timezone.utc)

            # 1. 收集检索消息
            retrieval_messages = self._collect_retrieval_messages()

            if not retrieval_messages:
                logger.info("Compaction skipped: no retrieval messages")
                return False

            # 2. 提取所有 chunks
            all_chunks = []
            message_chunk_mapping = {}  # tool_call_id -> list of chunk objects

            for msg in retrieval_messages:
                # 跳过已压缩的消息
                if msg.tool_call_id in self.compacted_message_ids:
                    continue

                try:
                    data = json.loads(msg.content)
                    chunks = _extract_chunks_from_tool_response(data)

                    if chunks:
                        message_chunk_mapping[msg.tool_call_id] = chunks
                        all_chunks.extend(chunks)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool message as JSON: tool_call_id=%s", msg.tool_call_id)
                    continue

            if not all_chunks:
                logger.info("Compaction skipped: no compressible chunks")
                return False

            # 记录压缩前状态
            current_tokens = self._estimate_memory_tokens()
            before_compact_state = {
                "total_messages": len(self.memory.messages),
                "tool_messages": len([m for m in self.memory.messages if m.role == MessageRole.TOOL]),
                "retrieval_messages": len(retrieval_messages),
                "total_chunks": len(all_chunks),
                "total_tokens": current_tokens,
            }

            # 3. LLM 评分
            scores = self._score_chunks_with_llm(all_chunks)

            if not scores:
                logger.warning("Compaction skipped: scoring failed")
                return False

            # 4. 根据评分删除低分 chunks（会收集 deleted_chunks）
            score_dict = {s["id"]: s["score"] for s in scores}
            deleted_chunks, message_compression_details = self._compact_tool_messages(message_chunk_mapping, score_dict)

            # 5. 记录已压缩的消息（修复：定期清理以避免内存泄漏）
            self.compacted_message_ids.update(message_chunk_mapping.keys())

            # 清理不再存在的消息ID（防止内存泄漏）
            current_tool_call_ids = {msg.tool_call_id for msg in self.memory.messages if msg.role == MessageRole.TOOL and msg.tool_call_id}
            self.compacted_message_ids &= current_tool_call_ids  # 保留交集

            after_tokens = self._estimate_memory_tokens()
            compact_end_time = datetime.now(timezone.utc)

            saved_tokens = current_tokens - after_tokens
            logger.info("Compaction completed: tokens %d->%d, freed %d, chunks_deleted=%d", current_tokens, after_tokens, saved_tokens, len(deleted_chunks))

            # 6. 写入 Compact 日志
            self._log_compact_event(
                compact_start_time=compact_start_time,
                compact_end_time=compact_end_time,
                current_tokens=current_tokens,
                after_tokens=after_tokens,
                before_compact_state=before_compact_state,
                chunk_scores=scores,
                deleted_chunks=deleted_chunks,
                message_compression_details=message_compression_details,
            )

            return True

        except Exception as e:
            logger.error("Compaction failed: %s", str(e), exc_info=True)
            return False

    # ===== Compact 辅助方法 =====

    def _collect_retrieval_messages(self) -> List[Message]:
        """收集所有检索工具的 tool 消息"""
        retrieval_messages = []

        for msg in self.memory.messages:
            if msg.role == MessageRole.TOOL and msg.name in ("knowledge_search", "read_file"):
                retrieval_messages.append(msg)

        return retrieval_messages

    def _score_chunks_with_llm(self, chunks: List[Dict]) -> List[Dict]:
        """使用 LLM 对 chunks 评分（修复：分批评分所有chunks）

        Args:
            chunks: chunk 对象列表

        Returns:
            List[Dict]: 评分结果，格式 [{"id": "xxx", "score": 5}, ...]
        """
        all_scores = []
        batch_size = 20  # 每批评分的 chunk 数量

        try:
            # 获取压缩评分提示模板（必需字段）
            compact_prompt_template = self.templates["compact_system_prompt"].strip()

            # 分批评分所有 chunks（修复：之前只评分前20个）
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                chunks_text = "\n\n".join([f"ID: {c.get('id', 'unknown')}\nContent: {c.get('content', '')[:200]}" for c in batch_chunks])

                prompt = f"""{compact_prompt_template}

User Query: Research task relevance evaluation

Chunks to evaluate:
{chunks_text}

Return JSON array: [{{"id": "chunk_id", "score": 5}}, ...]"""

                messages = [Message(role=MessageRole.USER, content=prompt)]

                # 调用 LLM（非流式）
                response = chat_sync(self.model, messages)

                # 解析响应
                content = response.content or ""

                # 尝试提取 JSON
                start = content.find("[")
                end = content.rfind("]") + 1

                if start >= 0 and end > start:
                    batch_scores = json.loads(content[start:end])
                    # 修复：确保score为int类型，防止字符串比较导致TypeError
                    for item in batch_scores:
                        if isinstance(item.get("score"), str):
                            try:
                                item["score"] = int(item["score"])
                            except (ValueError, TypeError):
                                item["score"] = 5  # 默认中等分数
                        elif not isinstance(item.get("score"), (int, float)):
                            item["score"] = 5  # 非数字分数设为默认值
                    all_scores.extend(batch_scores)
                else:
                    logger.warning("Failed to extract JSON scoring from LLM response: batch=%d", i // batch_size + 1)

            return all_scores

        except Exception as e:
            logger.error("LLM scoring failed: %s", str(e), exc_info=True)
            return []

    def _compact_tool_messages(self, message_chunk_mapping: Dict[str, List[Dict]], score_dict: Dict[str, int]) -> tuple[List[Dict], List[Dict]]:
        """根据评分删除低分 chunks，直接修改 memory 中的消息

        保留策略（修复版）：
        1. 先按分数降序排序所有 chunks
        2. 保留所有分数 >= keep_score_threshold 的 chunks
        3. 如果保留数量 < min_keep_k，从剩余 chunks 中按分数高低补充
        4. 确保最终保留的是分数最高的 chunks

        Args:
            message_chunk_mapping: tool_call_id -> chunks
            score_dict: chunk_id -> score

        Returns:
            tuple[List[Dict], List[Dict]]: (deleted_chunks, message_compression_details)
        """
        # 收集被删除的 chunks（完整保存）
        deleted_chunks = []
        message_compression_details = []

        for msg in self.memory.messages:
            if msg.role != MessageRole.TOOL or msg.tool_call_id not in message_chunk_mapping:
                continue

            try:
                # 解析当前消息内容
                data = json.loads(msg.content)
                original_chunks = _extract_chunks_from_tool_response(data)

                if not original_chunks:
                    continue

                # 步骤1: 为每个 chunk 附加分数，按分数降序排序
                scored_chunks = [
                    (chunk, score_dict.get(chunk.get("id"), 5))  # 默认中等分数 5
                    for chunk in original_chunks
                ]
                sorted_by_score = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

                # 步骤2: 选择要保留的 chunks
                kept_chunks = []
                kept_chunk_ids = set()

                for chunk, score in sorted_by_score:
                    chunk_id = chunk.get("id")

                    # 保留条件：分数达标，或者数量不足 min_keep_k
                    if score >= self.config.keep_score_threshold:
                        kept_chunks.append(chunk)
                        kept_chunk_ids.add(chunk_id)
                    elif len(kept_chunks) < self.config.min_keep_k:
                        # 补充到 min_keep_k（已按分数排序，所以补充的是剩余中最高分的）
                        kept_chunks.append(chunk)
                        kept_chunk_ids.add(chunk_id)

                # 步骤3: 收集被删除的 chunks
                for chunk, score in sorted_by_score:
                    chunk_id = chunk.get("id")
                    if chunk_id not in kept_chunk_ids:
                        deleted_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "score": score,
                                "original_message_id": msg.tool_call_id,
                                "tool_name": msg.name,
                                "chunk_data": chunk,  # 完整保存chunk内容
                            }
                        )

                # 记录消息压缩详情
                deleted_chunk_ids = [chunk.get("id") for chunk in original_chunks if chunk.get("id") not in kept_chunk_ids]
                message_compression_details.append(
                    {
                        "message_id": msg.tool_call_id,
                        "tool_name": msg.name,
                        "before_chunks": len(original_chunks),
                        "after_chunks": len(kept_chunks),
                        "chunks_deleted": len(deleted_chunk_ids),
                        "deleted_chunk_ids": deleted_chunk_ids,
                    }
                )

                # 更新消息内容（保留分组结构）
                _update_chunks_in_tool_response(data, kept_chunks)
                msg.content = json.dumps(data, ensure_ascii=False)

                logger.debug("Compacted message: tool_call_id=%s, chunks %d->%d", msg.tool_call_id, len(original_chunks), len(kept_chunks))

            except Exception as e:
                logger.error("Message compaction failed: tool_call_id=%s, error=%s", msg.tool_call_id, str(e), exc_info=True)

        return deleted_chunks, message_compression_details

    def _log_compact_event(
        self,
        compact_start_time: datetime,
        compact_end_time: datetime,
        current_tokens: int,
        after_tokens: int,
        before_compact_state: Dict,
        chunk_scores: List[Dict],
        deleted_chunks: List[Dict],
        message_compression_details: List[Dict],
    ):
        """记录 Compact 事件日志

        统一处理 Compact 事件的日志记录逻辑。
        """
        if not self.trace_emitter:
            return

        try:
            # 计算压缩后状态
            after_compact_state = {
                "total_messages": len(self.memory.messages),
                "tool_messages": len([m for m in self.memory.messages if m.role == MessageRole.TOOL]),
                "retrieval_messages": len(self._collect_retrieval_messages()),
                "total_chunks": before_compact_state["total_chunks"] - len(deleted_chunks),
                "total_tokens": after_tokens,
            }

            # 计算实际使用的阈值（与agent.py:573的计算一致）
            actual_threshold_tokens = int(self.config.max_context_length * self.config.context_check_threshold)

            compact_data = {
                "timing": {
                    "triggered_at": compact_start_time.isoformat(),
                    "completed_at": compact_end_time.isoformat(),
                    "duration_seconds": (compact_end_time - compact_start_time).total_seconds(),
                },
                "trigger": {
                    "reason": "token_threshold_exceeded",
                    "current_tokens": current_tokens,
                    "threshold_tokens": actual_threshold_tokens,
                    "threshold_ratio": current_tokens / actual_threshold_tokens if actual_threshold_tokens > 0 else 0,
                },
                "before_compact": before_compact_state,
                "scoring": {
                    "total_chunks_evaluated": len(chunk_scores),
                    "chunk_scores": chunk_scores,  # 所有chunk的评分
                },
                "deletion_strategy": {
                    "keep_score_threshold": self.config.keep_score_threshold,
                    "min_keep_per_message": self.config.min_keep_k,
                    "total_chunks_to_delete": len(deleted_chunks),
                },
                "deleted_chunks": deleted_chunks,  # 完整记录被删除的chunks
                "after_compact": {
                    **after_compact_state,
                    "message_compression": message_compression_details,
                },
                "compression_result": {
                    "status": "success",
                    "tokens_saved": current_tokens - after_tokens,
                    "tokens_saved_percentage": (current_tokens - after_tokens) / current_tokens * 100 if current_tokens > 0 else 0,
                    "chunks_deleted": len(deleted_chunks),
                    "chunks_deleted_percentage": len(deleted_chunks) / before_compact_state["total_chunks"] * 100 if before_compact_state["total_chunks"] > 0 else 0,
                },
            }

            self.trace_emitter.write_compact(compact_data)
            logger.info("Compaction event logged: chunks_deleted=%d", len(deleted_chunks))
        except Exception as e:
            logger.warning("Failed to write compaction log: %s", str(e))
