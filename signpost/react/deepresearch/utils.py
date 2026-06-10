"""深度研究代理工具函数"""

import json
from datetime import datetime
from typing import Any
import xxhash


def _count_tokens(text: str) -> int:
    """使用全局tokenizer计算token数

    Args:
        text: 输入文本

    Returns:
        int: token数量

    Raises:
        RuntimeError: tokenizer初始化或计算失败时抛出
    """
    from .configuration import get_global_tokenizer

    try:
        tokenizer = get_global_tokenizer()
        return len(tokenizer.encode(text))
    except Exception as e:
        raise RuntimeError(f"Token计算失败，无法继续执行: {e}") from e


def generate_short_id(hash_input: str) -> str:
    """
    使用xxHash32生成8位十六进制ID (原生短hash，推荐)

    优点：原生32位输出，性能极佳，分布均匀
    如果xxhash库不可用则回退到CRC32

    Args:
        hash_input: 输入字符串

    Returns:
        8位十六进制ID字符串
    """
    hash_value = xxhash.xxh32(hash_input.encode("utf-8")).intdigest()
    return f"{hash_value:08x}"


# ===== 公共格式化与 ID 生成工具（供 ReadFileTool 与 KnowledgeSearchTool 复用） =====


def generate_chunk_id(kb_name: str, file_name: str, start_line: int, end_line: int) -> str:
    """生成基于内容的8位短ID。

    基于知识库名、文件名和行号范围生成8位十六进制ID
    """
    safe_kb = kb_name or ""
    safe_fn = file_name or ""
    content = f"{safe_kb}:{safe_fn}:{start_line}:{end_line}"
    return generate_short_id(content)


def format_numbered_lines(
    lines: list[str],
    start_line: int,
    total_lines: int,
    *,
    max_tokens: int = 2000,
    max_line_chars: int = 2000,
) -> tuple[list[str], int, bool]:
    """生成带行号的内容行，应用单行截断与 token 限制。

    与 ReadFileTool._format_numbered_lines 保持一致的行为。
    """
    width = max(2, len(str(total_lines or 0)))
    formatted_lines: list[str] = []
    current_tokens = 0
    lines_were_truncated = False

    base = start_line if isinstance(start_line, int) and start_line >= 1 else 1

    for idx, line in enumerate(lines or []):
        abs_line_no = base + idx

        display_line = line
        if len(display_line) > max_line_chars:
            display_line = display_line[:max_line_chars] + "... [truncated]"
            lines_were_truncated = True

        numbered = f"{abs_line_no:>{width}d} | {display_line}"
        line_tokens = _count_tokens(numbered + "\n")
        if formatted_lines and current_tokens + line_tokens > max_tokens:
            break
        formatted_lines.append(numbered)
        current_tokens += line_tokens

    return formatted_lines, current_tokens, lines_were_truncated


def build_indicator(
    visible_start: int,
    visible_end: int,
    total_lines: int,
    *,
    next_offset: int | None = None,
    is_at_end: bool = False,
) -> str:
    """构建截断指示块，与 ReadFileTool._build_indicator 保持一致。"""
    if is_at_end:
        action_text = "Action: This is the end of the file. To read from the beginning or any other part of the file, use the 'offset' and 'limit' parameters in a subsequent 'read_file' call."
    else:
        action_text = (
            "Action: To read more of the file, you can use the 'offset' and 'limit' parameters "
            "in a subsequent 'read_file' call. For example, to read the next section of the file, "
            f"use offset: {next_offset}."
        )
    return (
        "IMPORTANT: The file content has been truncated.\n"
        f"Status: Showing lines {visible_start}-{visible_end} of {total_lines} total lines.\n"
        'Format: Each line displays as "LINE_NUMBER | ACTUAL_CONTENT"\n'
        f"{action_text}\n\n"
        "--- FILE CONTENT (truncated) ---"
    )


def clean_filename_for_llm(filename: str) -> str:
    """移除文件名中的技术标识符，保留语义信息

    清理规则：
    - 移除 _normalized、_tree、_toc 等技术后缀
    - 保留文件扩展名
    - 保留虚拟文件的序号后缀（如 _0, _1）

    Examples:
        memory_aliasing_guide_normalized.txt -> memory_aliasing_guide.txt
        memory_aliasing_guide_toc.md -> memory_aliasing_guide.md
        memory_aliasing_guide_tree.json -> memory_aliasing_guide.json
        Hippocratic_oath_0.txt -> Hippocratic_oath_0.txt (不变)

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    if not filename:
        return filename

    import os

    base, ext = os.path.splitext(filename)

    # 移除技术后缀（按优先级检查）
    for infix in ("_normalized", "_tree", "_toc"):
        if base.endswith(infix):
            base = base[: -len(infix)]
            break

    return base + ext


def build_title(file_name: str, visible_start: int, visible_end: int, total_lines: int) -> str:
    """构建标题行，与 ReadFileTool._build_title 保持一致。

    使用清理后的文件名（移除技术标识符），提升 LLM 可读性。
    """
    fn = clean_filename_for_llm(file_name or "unknown")

    if visible_end < visible_start:
        return f"{fn} (no content in requested range, total {total_lines} lines)"
    return f"{fn} (lines {visible_start}-{visible_end} of {total_lines})"


def format_file_view(
    *,
    file_name: str,
    raw_text: str,
    start_line: int | None,
    end_line: int | None,
    total_lines: int | None,
    max_tokens: int = 2000,
    max_line_chars: int = 2000,
) -> dict[str, Any]:
    """将原始文本 + 行号信息格式化为带行号/截断指示/标题的视图。

    返回：{
      content, title, visible_start, visible_end, is_truncated, next_offset, tokens
    }
    """
    safe_total = total_lines or 0
    base_start = start_line if isinstance(start_line, int) and start_line >= 1 else 1
    theoretical_end = end_line if isinstance(end_line, int) and end_line >= base_start else (base_start - 1)

    lines = (raw_text or "").splitlines()
    formatted_lines, actual_tokens, lines_were_truncated = format_numbered_lines(lines, base_start, safe_total, max_tokens=max_tokens, max_line_chars=max_line_chars)
    if formatted_lines:
        visible_end_actual = base_start + len(formatted_lines) - 1
    else:
        visible_end_actual = base_start - 1 if base_start > 1 else 0

    content_range_truncated = (base_start > 1) or (safe_total > 0 and visible_end_actual < safe_total)
    token_truncated = visible_end_actual < theoretical_end
    is_truncated = content_range_truncated or token_truncated or lines_were_truncated

    is_at_end = (safe_total > 0 and visible_end_actual >= safe_total) and not token_truncated
    next_offset = None
    if is_truncated and not is_at_end:
        next_offset = (base_start - 1) + max(0, len(formatted_lines))

    if is_truncated and formatted_lines:
        indicator_block = build_indicator(base_start, visible_end_actual, safe_total, next_offset=next_offset, is_at_end=is_at_end)
        content = indicator_block + "\n" + "\n".join(formatted_lines)
    else:
        content = "\n".join(formatted_lines) if formatted_lines else ""

    title = build_title(file_name or "unknown", base_start, visible_end_actual, safe_total)

    # 修复：空内容时，将 title 作为 content（避免 format_tool_response_for_llm 误判为错误）
    if content == "":
        content = title

    return {
        "content": content,
        "title": title,
        "visible_start": base_start,
        "visible_end": visible_end_actual,
        "is_truncated": is_truncated,
        "next_offset": next_offset,
        "tokens": _count_tokens(content),
    }


def get_today_str() -> str:
    """获取今天的日期字符串"""
    return datetime.now().strftime("%Y-%m-%d")


class ResearchError(Exception):
    """研究过程中的异常"""

    pass


# 保留到文件末尾，无额外内容


class ConfigurationError(ResearchError):
    """配置错误"""

    pass


class APIError(ResearchError):
    """API调用错误"""

    pass


class TokenLimitError(ResearchError):
    """Token限制错误"""

    pass


# ========== Token 估算工具（上下文长度管理） ==========


def estimate_memory_tokens(system_prompt: str, messages: list) -> int:
    """估算 AgentMemory 的 token 总数（system_prompt + messages）

    使用全局 tokenizer 进行精确计算。

    Args:
        system_prompt: 系统提示词
        messages: Message 列表（来自 AgentMemory.messages）

    Returns:
        int: 估算的 token 总数

    Raises:
        RuntimeError: tokenizer 初始化或计算失败时抛出
    """
    from .configuration import get_global_tokenizer

    tokenizer = get_global_tokenizer()  # 失败则直接抛出 RuntimeError
    total_tokens = 0

    # 1. 统计 system prompt
    if system_prompt:
        total_tokens += len(tokenizer.encode(system_prompt))

    # 2. 统计所有消息
    for msg in messages:
        # 统计 content
        if msg.content:
            total_tokens += len(tokenizer.encode(str(msg.content)))

        # 统计 reasoning_content (DeepSeek Reasoner Thinking Mode)
        if msg.reasoning_content:
            total_tokens += len(tokenizer.encode(msg.reasoning_content))

        # 统计 tool_calls
        if msg.tool_calls:
            tool_calls_str = json.dumps([tc.to_dict() for tc in msg.tool_calls])
            total_tokens += len(tokenizer.encode(tool_calls_str))

    return total_tokens


def estimate_messages_tokens(messages: list) -> int:
    """估算消息列表的 token 总数

    使用全局 tokenizer (transformers.AutoTokenizer) 进行精确计算

    Args:
        messages: ChatMessage列表（已经过write_memory_to_messages转换，包含XML格式）

    Returns:
        int: 估算的token总数

    Raises:
        RuntimeError: token计算失败时抛出
    """
    from .configuration import get_global_tokenizer

    try:
        tokenizer = get_global_tokenizer()
        total_tokens = 0

        for msg in messages:
            # 1. 角色开销（每条消息约6个token：<|im_start|>role<|im_sep|>等）
            total_tokens += 6

            # 2. 处理 content 字段
            content_text = _extract_text_from_content(msg.content)
            if content_text:
                total_tokens += len(tokenizer.encode(content_text))

            # 3. 处理 reasoning_content 字段 (DeepSeek Reasoner Thinking Mode)
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                total_tokens += len(tokenizer.encode(msg.reasoning_content))

            # 4. 处理 tool_calls 字段（工具调用也会占用token）
            if msg.tool_calls:
                tool_calls_str = str([tc.dict() for tc in msg.tool_calls])
                total_tokens += len(tokenizer.encode(tool_calls_str))

        return total_tokens

    except Exception as e:
        import logging

        error_msg = f"Token估算失败: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def _extract_text_from_content(content) -> str:
    """从ChatMessage.content中提取纯文本

    Args:
        content: str | list[dict] | None

    Returns:
        str: 提取的文本内容
    """
    if content is None:
        return ""

    if isinstance(content, str):
        # 旧版格式：直接是字符串
        return content

    if isinstance(content, list):
        # 新版格式：list[dict]，每个dict可能是text或image块
        texts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")

                if block_type == "text":
                    texts.append(block.get("text", ""))

                # TODO: 暂不支持多模态（image块），当前项目不使用
                # elif block_type == "image":
                #     # 保守估计图像为1000 tokens
                #     texts.append("[IMAGE_PLACEHOLDER]" * 100)

        return "".join(texts)

    # 其他类型，转为字符串
    return str(content)


# ========== Token 估算工具结束 ==========
