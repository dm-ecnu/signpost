"""DeepResearch LLM utilities

Converts LLMCore streaming output to DeepResearch event types.
The actual LLM calling is handled by rag.llm.core.LLMCore.
"""

from typing import Any, Dict, Iterator, List, Optional

from core.llm.core import LLMCore

from .entities import Message, MessageRole, ToolCall, ToolCallFunction
from .events import (
    DeepResearchEvent,
    LLMContentDeltaEvent,
    LLMContentDoneEvent,
    LLMToolCallDoneEvent,
)


def create_model(config) -> LLMCore:
    """Create an LLMCore instance from a deepresearch Configuration object.

    Args:
        config: deepresearch.configuration.Configuration instance
    """
    return LLMCore(
        model_name=config.model_id,
        api_key=config.api_key,
        base_url=config.api_base,
        source_module="deepresearch",
    )


def stream_events(
    core: LLMCore,
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[DeepResearchEvent]:
    """Stream LLMCore output as DeepResearchEvent objects.

    Converts Message objects to OpenAI dicts, calls LLMCore.chat_stream(),
    and yields DeepResearch-specific event types.

    Args:
        core: LLMCore instance
        messages: List of Message objects
        tools: Optional tool schemas (OpenAI format)
        trace_id: Trace ID for event correlation
    """
    openai_messages = [msg.to_openai_dict() for msg in messages]
    for chunk in core.chat_stream(openai_messages, tools=tools):
        if chunk.content_delta:
            yield LLMContentDeltaEvent(content=chunk.content_delta, trace_id=trace_id)
        if chunk.is_done and chunk.accumulated:
            yield LLMContentDoneEvent(
                content=chunk.accumulated.content,
                reasoning_content=chunk.accumulated.reasoning_content,
                trace_id=trace_id,
            )
            for tc in chunk.accumulated.tool_calls or []:
                yield LLMToolCallDoneEvent(
                    tool_call_id=tc["id"],
                    tool_name=tc["function"]["name"],
                    tool_arguments=tc["function"]["arguments"],
                    trace_id=trace_id,
                )


def chat_sync(
    core: LLMCore,
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Message:
    """Non-streaming chat that returns a Message object.

    Args:
        core: LLMCore instance
        messages: List of Message objects
        tools: Optional tool schemas

    Returns:
        Message object with the LLM response
    """
    openai_messages = [msg.to_openai_dict() for msg in messages]
    result = core.chat(openai_messages, tools=tools)

    # Convert tool_calls from LLMResult format to Message format
    tool_calls = None
    if result.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc["id"],
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in result.tool_calls
        ]

    return Message(
        role=MessageRole.ASSISTANT,
        content=result.content,
        reasoning_content=result.reasoning_content,
        tool_calls=tool_calls,
    )
