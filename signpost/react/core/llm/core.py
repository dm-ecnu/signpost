"""Unified LLM Core

Shared LLM calling interface used by graphrag, deepresearch_v2, and ultradomain_txt.
Backed by OpenAI SDK (supports any OpenAI-compatible API).
Caches to PostgreSQL via the LLMCache Peewee model.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import xxhash
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 2.0

# Error classification constants
ERROR_RATE_LIMIT = "RATE_LIMIT_EXCEEDED"
ERROR_SERVER = "SERVER_ERROR"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_PREFIX = "**ERROR**"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    """Unified result from any LLM call."""

    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    total_tokens: int = 0

    def to_cache_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_cache_dict(cls, d: Dict[str, Any]) -> "LLMResult":
        return cls(
            content=d.get("content", ""),
            reasoning_content=d.get("reasoning_content"),
            tool_calls=d.get("tool_calls"),
            finish_reason=d.get("finish_reason", "stop"),
            total_tokens=d.get("total_tokens", 0),
        )


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    content_delta: str = ""
    reasoning_delta: str = ""
    tool_call_deltas: Optional[List[Dict[str, Any]]] = None
    is_done: bool = False
    accumulated: Optional[LLMResult] = None  # set only when is_done=True


# ---------------------------------------------------------------------------
# Cache helpers (PostgreSQL via LLMCache Peewee model)
# ---------------------------------------------------------------------------

def _cache_key(model_name: str, messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    hasher = xxhash.xxh64()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if tools:
        hasher.update(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return hasher.hexdigest()


def _cache_get(key: str) -> Optional[LLMResult]:
    try:
        from core.db.models import DB, LLMCache as LLMCacheModel
        with DB.connection_context():
            record = LLMCacheModel.get(LLMCacheModel.cache_key == key)
            # Reconstruct LLMResult
            result = LLMResult(
                content=record.response_content or "",
                total_tokens=record.total_tokens or 0,
            )
            # Extended fields
            if hasattr(record, "reasoning_content") and record.reasoning_content:
                result.reasoning_content = record.reasoning_content
            if hasattr(record, "tool_calls_json") and record.tool_calls_json:
                result.tool_calls = record.tool_calls_json
            # Try metadata for tokens
            meta = record.response_metadata or {}
            if not result.total_tokens:
                result.total_tokens = meta.get("usage", {}).get("total_tokens", 0)
            return result
    except Exception:
        return None


def _cache_set(
    key: str,
    model_name: str,
    messages: List[Dict],
    gen_conf: Optional[Dict],
    result: LLMResult,
    source_module: str = "",
) -> None:
    try:
        from core.db.models import DB, LLMCache as LLMCacheModel
        from core import utils

        system_prompt = ""
        user_prompt = ""
        history = messages
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")

        metadata = {
            "usage": {"total_tokens": result.total_tokens},
            "created_at": time.time(),
        }

        raw_request = {
            "llm_name": model_name,
            "messages": messages,
            "generation_config": gen_conf or {},
            "timestamp": time.time(),
        }

        insert_data = {
            "id": utils.get_uuid(),
            "cache_key": key,
            "llm_name": model_name,
            "raw_request": raw_request,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "history_messages": history,
            "generation_config": gen_conf or {},
            "response_content": result.content,
            "response_metadata": metadata,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": result.total_tokens,
        }

        # Extended fields (added by migration)
        if result.reasoning_content:
            insert_data["reasoning_content"] = result.reasoning_content
        if result.tool_calls:
            insert_data["tool_calls_json"] = result.tool_calls
        if source_module:
            insert_data["source_module"] = source_module

        update_data = {
            LLMCacheModel.response_content: result.content,
            LLMCacheModel.response_metadata: metadata,
            LLMCacheModel.total_tokens: result.total_tokens,
        }

        with DB.connection_context():
            with DB.atomic():
                (
                    LLMCacheModel.insert(**insert_data)
                    .on_conflict(
                        conflict_target=[LLMCacheModel.cache_key],
                        update=update_data,
                    )
                    .execute()
                )
    except Exception as e:
        logger.warning("Failed to cache LLM response: %s", e)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _classify_error(error: Exception) -> str:
    error_str = str(error).lower()
    if any(kw in error_str for kw in ("rate limit", "429", "too many requests")):
        return ERROR_RATE_LIMIT
    if any(kw in error_str for kw in ("server", "502", "503", "504", "500", "unavailable")):
        return ERROR_SERVER
    if any(kw in error_str for kw in ("timeout", "timed out")):
        return ERROR_TIMEOUT
    return "GENERIC_ERROR"


def _should_retry(error: Exception) -> bool:
    code = _classify_error(error)
    return code in (ERROR_RATE_LIMIT, ERROR_SERVER, ERROR_TIMEOUT)


# ---------------------------------------------------------------------------
# LLMCore
# ---------------------------------------------------------------------------

class LLMCore:
    """Unified LLM calling core.

    All three modules (graphrag, deepresearch_v2, ultradomain_txt) use this
    class directly for chat completions. Embedding and rerank remain in their
    respective model classes.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        timeout: int = 600,
        enable_cache: bool = True,
        source_module: str = "",
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_cache = enable_cache
        self.source_module = source_module

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get("OPENAI_API_BASE", "")

        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        gen_conf: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResult:
        """Non-streaming chat completion with retry and cache."""
        # Cache lookup
        if self.enable_cache:
            key = _cache_key(self.model_name, messages, tools)
            cached = _cache_get(key)
            if cached is not None:
                logger.debug("Cache hit for %s: %s", self.model_name, key[:12])
                return cached

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._do_chat(messages, gen_conf, tools)
                # Cache on success (skip error responses)
                if self.enable_cache and not (result.content.startswith(ERROR_PREFIX)):
                    _cache_set(
                        _cache_key(self.model_name, messages, tools),
                        self.model_name,
                        messages,
                        gen_conf,
                        result,
                        self.source_module,
                    )
                return result
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries and _should_retry(e):
                    delay = self.retry_base_delay + random.uniform(0, 0.5)
                    logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, self.max_retries, delay, e)
                    time.sleep(delay)
                else:
                    break
        logger.error("LLM call failed after %d attempts: %s", self.max_retries, last_exc)
        return LLMResult(content=f"{ERROR_PREFIX}: {last_exc}", finish_reason="error")

    async def chat_async(
        self,
        messages: List[Dict[str, Any]],
        gen_conf: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResult:
        """Async non-streaming chat completion with retry and cache."""
        import asyncio

        # Cache lookup (sync DB is fine, runs in thread if needed)
        if self.enable_cache:
            key = _cache_key(self.model_name, messages, tools)
            cached = _cache_get(key)
            if cached is not None:
                logger.debug("Cache hit for %s: %s", self.model_name, key[:12])
                return cached

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._do_chat_async(messages, gen_conf, tools)
                if self.enable_cache and not (result.content.startswith(ERROR_PREFIX)):
                    _cache_set(
                        _cache_key(self.model_name, messages, tools),
                        self.model_name,
                        messages,
                        gen_conf,
                        result,
                        self.source_module,
                    )
                return result
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries and _should_retry(e):
                    delay = self.retry_base_delay + random.uniform(0, 0.5)
                    logger.warning("Async LLM call failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, self.max_retries, delay, e)
                    await asyncio.sleep(delay)
                else:
                    break
        logger.error("Async LLM call failed after %d attempts: %s", self.max_retries, last_exc)
        return LLMResult(content=f"{ERROR_PREFIX}: {last_exc}", finish_reason="error")

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        gen_conf: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[StreamChunk]:
        """Streaming chat completion. Yields StreamChunk objects.

        The final chunk has is_done=True and accumulated holds the full LLMResult.
        Also caches the accumulated result on completion.
        """
        # Check cache first - replay as single chunk if hit
        if self.enable_cache:
            key = _cache_key(self.model_name, messages, tools)
            cached = _cache_get(key)
            if cached is not None:
                logger.debug("Cache hit (stream replay) for %s", self.model_name)
                if cached.content:
                    yield StreamChunk(content_delta=cached.content)
                yield StreamChunk(is_done=True, accumulated=cached)
                return

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                accumulated = LLMResult()
                accumulated_content = ""
                accumulated_reasoning = ""
                accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

                kwargs = self._build_api_params(messages, gen_conf, tools, stream=True)
                response = self.client.chat.completions.create(**kwargs)

                for chunk in response:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    sc = StreamChunk()

                    if delta.content:
                        accumulated_content += delta.content
                        sc.content_delta = delta.content

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        accumulated_reasoning += delta.reasoning_content
                        sc.reasoning_delta = delta.reasoning_content

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                accumulated_tool_calls[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments

                    if choice.finish_reason:
                        accumulated.content = accumulated_content
                        accumulated.reasoning_content = accumulated_reasoning or None
                        accumulated.finish_reason = choice.finish_reason
                        if accumulated_tool_calls:
                            accumulated.tool_calls = [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                                }
                                for tc in accumulated_tool_calls.values()
                            ]
                        try:
                            accumulated.total_tokens = chunk.usage.total_tokens if chunk.usage else 0
                        except Exception:
                            accumulated.total_tokens = 0
                        sc.is_done = True
                        sc.accumulated = accumulated

                    yield sc

                # Cache the accumulated result
                if self.enable_cache and accumulated.content and not accumulated.content.startswith(ERROR_PREFIX):
                    _cache_set(
                        _cache_key(self.model_name, messages, tools),
                        self.model_name,
                        messages,
                        gen_conf,
                        accumulated,
                        self.source_module,
                    )
                return  # success

            except Exception as e:
                last_exc = e
                if attempt < self.max_retries and _should_retry(e):
                    delay = self.retry_base_delay + random.uniform(0, 0.5)
                    logger.warning("Stream failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, self.max_retries, delay, e)
                    time.sleep(delay)
                else:
                    break

        # All retries exhausted
        logger.error("Stream failed after %d attempts: %s", self.max_retries, last_exc)
        error_result = LLMResult(content=f"{ERROR_PREFIX}: {last_exc}", finish_reason="error")
        yield StreamChunk(is_done=True, accumulated=error_result)

    # ------------------------------------------------------------------
    # Internal: build API params and execute
    # ------------------------------------------------------------------

    def _build_api_params(
        self,
        messages: List[Dict],
        gen_conf: Optional[Dict],
        tools: Optional[List[Dict]],
        stream: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if gen_conf:
            # Standard OpenAI params
            for k in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "response_format"):
                if k in gen_conf:
                    kwargs[k] = gen_conf[k]
            # Extra body params (top_k, min_p, etc.)
            extra = {}
            for k in ("top_k", "min_p", "repetition_penalty", "enable_thinking"):
                if k in gen_conf:
                    extra[k] = gen_conf[k]
            if extra:
                kwargs["extra_body"] = extra
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _do_chat(
        self,
        messages: List[Dict],
        gen_conf: Optional[Dict],
        tools: Optional[List[Dict]],
    ) -> LLMResult:
        kwargs = self._build_api_params(messages, gen_conf, tools, stream=False)
        response = self.client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    async def _do_chat_async(
        self,
        messages: List[Dict],
        gen_conf: Optional[Dict],
        tools: Optional[List[Dict]],
    ) -> LLMResult:
        kwargs = self._build_api_params(messages, gen_conf, tools, stream=False)
        response = await self.async_client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResult:
        """Parse an OpenAI ChatCompletion response into LLMResult."""
        if not response.choices:
            return LLMResult(content="", finish_reason="empty")

        choice = response.choices[0]
        message = choice.message
        result = LLMResult()

        result.content = (message.content or "").strip()
        result.finish_reason = choice.finish_reason or "stop"

        # reasoning_content (DeepSeek Reasoner / thinking models)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            result.reasoning_content = message.reasoning_content

        # tool_calls
        if message.tool_calls:
            result.tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # token count
        try:
            result.total_tokens = response.usage.total_tokens if response.usage else 0
        except Exception:
            result.total_tokens = 0

        return result
