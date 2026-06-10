#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import asyncio
import json
import logging
import os
import random
import time
from abc import ABC
from copy import deepcopy
from typing import Any, Protocol

import json_repair
import openai
from openai import OpenAI, AsyncOpenAI

from core.nlp import is_chinese, is_english
from core.utils import num_tokens_from_string
from core.utils.llm_cache_utils import db_cache_llm_call, db_cache_llm_call_async

# Error message constants
ERROR_PREFIX = "**ERROR**"
ERROR_RATE_LIMIT = "RATE_LIMIT_EXCEEDED"
ERROR_AUTHENTICATION = "AUTH_ERROR"
ERROR_INVALID_REQUEST = "INVALID_REQUEST"
ERROR_SERVER = "SERVER_ERROR"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_CONNECTION = "CONNECTION_ERROR"
ERROR_MODEL = "MODEL_ERROR"
ERROR_CONTENT_FILTER = "CONTENT_FILTERED"
ERROR_QUOTA = "QUOTA_EXCEEDED"
ERROR_MAX_RETRIES = "MAX_RETRIES_EXCEEDED"
ERROR_GENERIC = "GENERIC_ERROR"

LENGTH_NOTIFICATION_CN = "······\n由于大模型的上下文窗口大小限制，回答已经被大模型截断。"
LENGTH_NOTIFICATION_EN = "...\nThe answer is truncated by your chosen LLM due to its limitation on context length."

# Qwen-Plus specific sampling parameters
# top_p is a standard OpenAI parameter, top_k and min_p must be passed via extra_body
QWEN_PLUS_SAMPLING_PARAMS = {
    "top_p": 0.8,
}
QWEN_PLUS_EXTRA_PARAMS = {
    "top_k": 20,
    "min_p": 0.0,
}

# Parameters that need to be passed via extra_body (not standard OpenAI params)
EXTRA_BODY_PARAMS = {"top_k", "min_p", "repetition_penalty"}


class ToolCallSession(Protocol):
    def tool_call(self, name: str, arguments: dict[str, Any]) -> str: ...


class Base(ABC):
    def __init__(self, key, model_name, base_url, **kwargs):
        timeout = int(os.environ.get("LM_TIMEOUT_SECONDS", 600))
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model_name = model_name
        # Configure retry parameters
        self.max_retries = kwargs.get("max_retries", int(os.environ.get("LLM_MAX_RETRIES", 10)))
        self.base_delay = kwargs.get("retry_interval", float(os.environ.get("LLM_BASE_DELAY", 2.0)))
        self.max_rounds = kwargs.get("max_rounds", 5)
        self.is_tools = False
        self.tools = []
        self.toolcall_sessions = {}

    def _get_delay(self):
        """Calculate retry delay time"""
        return self.base_delay + random.uniform(0, 0.5)

    def _classify_error(self, error):
        """Classify error based on error message content"""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str or "tpm limit" in error_str or "too many requests" in error_str or "requests per minute" in error_str:
            return ERROR_RATE_LIMIT
        elif "auth" in error_str or "key" in error_str or "apikey" in error_str or "401" in error_str or "forbidden" in error_str or "permission" in error_str:
            return ERROR_AUTHENTICATION
        elif "invalid" in error_str or "bad request" in error_str or "400" in error_str or "format" in error_str or "malformed" in error_str or "parameter" in error_str:
            return ERROR_INVALID_REQUEST
        elif "server" in error_str or "502" in error_str or "503" in error_str or "504" in error_str or "500" in error_str or "unavailable" in error_str:
            return ERROR_SERVER
        elif "timeout" in error_str or "timed out" in error_str:
            return ERROR_TIMEOUT
        elif "connect" in error_str or "network" in error_str or "unreachable" in error_str or "dns" in error_str:
            return ERROR_CONNECTION
        elif "quota" in error_str or "capacity" in error_str or "credit" in error_str or "billing" in error_str or "limit" in error_str and "rate" not in error_str:
            return ERROR_QUOTA
        elif "filter" in error_str or "content" in error_str or "policy" in error_str or "blocked" in error_str or "safety" in error_str or "inappropriate" in error_str:
            return ERROR_CONTENT_FILTER
        elif "model" in error_str or "not found" in error_str or "does not exist" in error_str or "not available" in error_str:
            return ERROR_MODEL
        else:
            return ERROR_GENERIC

    def _clean_conf(self, gen_conf):
        # Keep max_tokens for OpenAI-compatible APIs to control output length
        # Subclasses can override this method if they need different behavior
        # (e.g., Gemini deletes it, Ollama converts to num_predict)
        return gen_conf

    def _chat(self, history, gen_conf):
        # Extract extra_body params from gen_conf
        extra_body = {}
        cleaned_conf = {}
        for k, v in gen_conf.items():
            if k in EXTRA_BODY_PARAMS:
                extra_body[k] = v
            else:
                cleaned_conf[k] = v

        # Merge Qwen-Plus specific sampling parameters if model is qwen-plus
        if self.model_name == "qwen-plus":
            cleaned_conf = {**QWEN_PLUS_SAMPLING_PARAMS, **cleaned_conf}
            # Merge with QWEN_PLUS_EXTRA_PARAMS, user params take precedence
            extra_body = {**QWEN_PLUS_EXTRA_PARAMS, **extra_body}

        # Only pass extra_body if it's not empty
        extra_body = extra_body if extra_body else None

        response = self.client.chat.completions.create(model=self.model_name, messages=history, extra_body=extra_body, **cleaned_conf)

        if any([not response.choices, not response.choices[0].message, not response.choices[0].message.content]):
            return "", 0
        ans = response.choices[0].message.content.strip()
        if response.choices[0].finish_reason == "length":
            # Log warning for all models when output is truncated due to length limit
            logging.warning(
                f"[LLM OUTPUT TRUNCATED] Model '{self.model_name}' output truncated due to length limit. "
                f"Response length: {len(ans)} chars. This may indicate infinite output or excessive verbosity. "
                f"Last user message preview: {history[-1].get('content', '')[:150] if history else 'N/A'}..."
            )
            if is_chinese(ans):
                ans += LENGTH_NOTIFICATION_CN
            else:
                ans += LENGTH_NOTIFICATION_EN
        return ans, self.total_token_count(response)

    def _length_stop(self, ans):
        if is_chinese([ans]):
            return ans + LENGTH_NOTIFICATION_CN
        return ans + LENGTH_NOTIFICATION_EN

    def _exceptions(self, e, attempt):
        logging.exception("OpenAI cat_with_tools")
        # Classify the error
        error_code = self._classify_error(e)

        # Check if it's a rate limit error or server error and not the last attempt
        should_retry = (error_code == ERROR_RATE_LIMIT or error_code == ERROR_SERVER) and attempt < self.max_retries

        if should_retry:
            delay = self._get_delay()
            logging.warning(f"Error: {error_code}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{self.max_retries})")
            time.sleep(delay)
        else:
            # For non-rate limit errors or the last attempt, return an error message
            if attempt == self.max_retries:
                error_code = ERROR_MAX_RETRIES
            return f"{ERROR_PREFIX}: {error_code} - {str(e)}"

    def _verbose_tool_use(self, name, args, res):
        return "<tool_call>" + json.dumps({"name": name, "args": args, "result": res}, ensure_ascii=False, indent=2) + "</tool_call>"

    def _append_history(self, hist, tool_call, tool_res):
        hist.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": tool_call.index,
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                        "type": "function",
                    },
                ],
            }
        )
        try:
            if isinstance(tool_res, dict):
                tool_res = json.dumps(tool_res, ensure_ascii=False)
        finally:
            hist.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_res)})
        return hist

    def bind_tools(self, toolcall_session, tools):
        if not (toolcall_session and tools):
            return
        self.is_tools = True

        for tool in tools:
            self.toolcall_sessions[tool["function"]["name"]] = toolcall_session
            self.tools.append(tool)

    def chat_with_tools(self, system: str, history: list, gen_conf: dict):
        gen_conf = self._clean_conf(deepcopy(gen_conf))
        base_history = deepcopy(history)
        if system:
            base_history.insert(0, {"role": "system", "content": system})

        ans = ""
        tk_count = 0
        # Implement exponential backoff retry strategy
        for attempt in range(self.max_retries + 1):
            history_iter = deepcopy(base_history)
            try:
                for _ in range(self.max_rounds * 2):
                    response = self.client.chat.completions.create(model=self.model_name, messages=history_iter, tools=self.tools, **gen_conf)
                    tk_count += self.total_token_count(response)
                    if any([not response.choices, not response.choices[0].message]):
                        raise Exception(f"500 response structure error. Response: {response}")

                    if not hasattr(response.choices[0].message, "tool_calls") or not response.choices[0].message.tool_calls:
                        if hasattr(response.choices[0].message, "reasoning_content") and response.choices[0].message.reasoning_content:
                            ans += "<think>" + response.choices[0].message.reasoning_content + "</think>"

                        ans += response.choices[0].message.content
                        if response.choices[0].finish_reason == "length":
                            ans = self._length_stop(ans)

                        return ans, tk_count

                    for tool_call in response.choices[0].message.tool_calls:
                        name = tool_call.function.name
                        try:
                            args = json_repair.loads(tool_call.function.arguments)
                            session = self.toolcall_sessions.get(name)
                            if not session:
                                raise KeyError(f"Tool session not found for {name}")
                            tool_response = session.tool_call(name, args)
                            history_iter = self._append_history(history_iter, tool_call, tool_response)
                            ans += self._verbose_tool_use(name, args, tool_response)
                        except Exception as e:
                            logging.exception(msg=f"Wrong JSON argument format in LLM tool call response: {tool_call}")
                            history_iter.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Tool call error: \n{tool_call}\nException:\n" + str(e)})
                            ans += self._verbose_tool_use(name, {}, str(e))

            except Exception as e:
                e = self._exceptions(e, attempt)
                if e:
                    return e, tk_count
        assert False, "Shouldn't be here."

    @db_cache_llm_call
    def chat(self, system, history, gen_conf):
        base_history = deepcopy(history)
        if system:
            base_history.insert(0, {"role": "system", "content": system})
        gen_conf = self._clean_conf(deepcopy(gen_conf))

        # Implement exponential backoff retry strategy
        for attempt in range(self.max_retries + 1):
            try:
                return self._chat(deepcopy(base_history), gen_conf)
            except Exception as e:
                e = self._exceptions(e, attempt)
                if e:
                    return e, 0
        assert False, "Shouldn't be here."

    def _wrap_toolcall_message(self, stream):
        final_tool_calls = {}

        for chunk in stream:
            for tool_call in chunk.choices[0].delta.tool_calls or []:
                index = tool_call.index

                if index not in final_tool_calls:
                    final_tool_calls[index] = tool_call

                final_tool_calls[index].function.arguments += tool_call.function.arguments

        return final_tool_calls

    def chat_streamly_with_tools(self, system: str, history: list, gen_conf: dict):
        gen_conf = self._clean_conf(deepcopy(gen_conf))
        tools = self.tools
        base_history = deepcopy(history)
        if system:
            base_history.insert(0, {"role": "system", "content": system})

        total_tokens = 0
        ans = ""
        # Implement exponential backoff retry strategy
        for attempt in range(self.max_retries + 1):
            history_iter = deepcopy(base_history)
            try:
                for _ in range(self.max_rounds * 2):
                    reasoning_start = False
                    response = self.client.chat.completions.create(model=self.model_name, messages=history_iter, stream=True, tools=tools, **gen_conf)
                    final_tool_calls = {}
                    answer = ""
                    for resp in response:
                        if resp.choices[0].delta.tool_calls:
                            for tool_call in resp.choices[0].delta.tool_calls or []:
                                index = tool_call.index

                                if index not in final_tool_calls:
                                    if not tool_call.function.arguments:
                                        tool_call.function.arguments = ""
                                    final_tool_calls[index] = tool_call
                                else:
                                    final_tool_calls[index].function.arguments += tool_call.function.arguments if tool_call.function.arguments else ""
                            continue

                        if any([not resp.choices, not resp.choices[0].delta, not hasattr(resp.choices[0].delta, "content")]):
                            raise Exception("500 response structure error.")

                        if not resp.choices[0].delta.content:
                            resp.choices[0].delta.content = ""

                        if hasattr(resp.choices[0].delta, "reasoning_content") and resp.choices[0].delta.reasoning_content:
                            ans = ""
                            if not reasoning_start:
                                reasoning_start = True
                                ans = "<think>"
                            ans += resp.choices[0].delta.reasoning_content + "</think>"
                            yield ans
                        else:
                            reasoning_start = False
                            answer += resp.choices[0].delta.content
                            yield resp.choices[0].delta.content

                        tol = self.total_token_count(resp)
                        if not tol:
                            total_tokens += num_tokens_from_string(resp.choices[0].delta.content)
                        else:
                            total_tokens += tol

                        finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], "finish_reason") else ""
                        if finish_reason == "length":
                            yield self._length_stop("")

                    if answer:
                        yield total_tokens
                        return

                    for tool_call in final_tool_calls.values():
                        name = tool_call.function.name
                        try:
                            args = json_repair.loads(tool_call.function.arguments)
                            session = self.toolcall_sessions.get(name)
                            if not session:
                                raise KeyError(f"Tool session not found for {name}")
                            tool_response = session.tool_call(name, args)
                            history_iter = self._append_history(history_iter, tool_call, tool_response)
                            yield self._verbose_tool_use(name, args, tool_response)
                        except Exception as e:
                            logging.exception(msg=f"Wrong JSON argument format in LLM tool call response: {tool_call}")
                            history_iter.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Tool call error: \n{tool_call}\nException:\n" + str(e)})
                            yield self._verbose_tool_use(name, {}, str(e))

            except Exception as e:
                e = self._exceptions(e, attempt)
                if e:
                    yield total_tokens
                    return

        yield total_tokens

    def chat_streamly(self, system, history, gen_conf):
        stream_history = deepcopy(history)
        if system:
            stream_history.insert(0, {"role": "system", "content": system})
        gen_conf = self._clean_conf(deepcopy(gen_conf))
        ans = ""
        total_tokens = 0
        reasoning_start = False
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=stream_history, stream=True, **gen_conf)
            for resp in response:
                if not resp.choices:
                    continue
                if not resp.choices[0].delta.content:
                    resp.choices[0].delta.content = ""
                if hasattr(resp.choices[0].delta, "reasoning_content") and resp.choices[0].delta.reasoning_content:
                    ans = ""
                    if not reasoning_start:
                        reasoning_start = True
                        ans = "<think>"
                    ans += resp.choices[0].delta.reasoning_content + "</think>"
                else:
                    reasoning_start = False
                    ans = resp.choices[0].delta.content

                tol = self.total_token_count(resp)
                if not tol:
                    total_tokens += num_tokens_from_string(resp.choices[0].delta.content)
                else:
                    total_tokens += tol

                if resp.choices[0].finish_reason == "length":
                    if is_chinese(ans):
                        ans += LENGTH_NOTIFICATION_CN
                    else:
                        ans += LENGTH_NOTIFICATION_EN
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens

    def total_token_count(self, resp):
        try:
            return resp.usage.total_tokens
        except Exception:
            pass
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass
        return 0

    def _calculate_dynamic_ctx(self, history):
        """Calculate dynamic context window size"""

        def count_tokens(text):
            """Calculate token count for text"""
            total = 0
            for char in text:
                if ord(char) < 128:
                    total += 1
                else:
                    total += 2
            return total

        total_tokens = 0
        for message in history:
            content = message.get("content", "")
            content_tokens = count_tokens(content)
            role_tokens = 4
            total_tokens += content_tokens + role_tokens

        total_tokens_with_buffer = int(total_tokens * 1.2)

        if total_tokens_with_buffer <= 8192:
            ctx_size = 8192
        else:
            ctx_multiplier = (total_tokens_with_buffer // 8192) + 1
            ctx_size = ctx_multiplier * 8192

        return ctx_size

    async def _chat_async(self, history, gen_conf):
        # Extract extra_body params from gen_conf
        extra_body = {}
        cleaned_conf = {}
        for k, v in gen_conf.items():
            if k in EXTRA_BODY_PARAMS:
                extra_body[k] = v
            else:
                cleaned_conf[k] = v

        # Merge Qwen-Plus specific sampling parameters if model is qwen-plus
        if self.model_name == "qwen-plus":
            cleaned_conf = {**QWEN_PLUS_SAMPLING_PARAMS, **cleaned_conf}
            extra_body = {**QWEN_PLUS_EXTRA_PARAMS, **extra_body}

        extra_body = extra_body if extra_body else None

        response = await self.async_client.chat.completions.create(model=self.model_name, messages=history, extra_body=extra_body, **cleaned_conf)

        if any([not response.choices, not response.choices[0].message, not response.choices[0].message.content]):
            return "", 0
        ans = response.choices[0].message.content.strip()
        if response.choices[0].finish_reason == "length":
            logging.warning(
                f"[LLM OUTPUT TRUNCATED] Model '{self.model_name}' output truncated due to length limit. "
                f"Response length: {len(ans)} chars. This may indicate infinite output or excessive verbosity. "
                f"Last user message preview: {history[-1].get('content', '')[:150] if history else 'N/A'}..."
            )
            if is_chinese(ans):
                ans += LENGTH_NOTIFICATION_CN
            else:
                ans += LENGTH_NOTIFICATION_EN
        return ans, self.total_token_count(response)

    async def _exceptions_async(self, e, attempt):
        logging.exception("OpenAI chat_async")
        error_code = self._classify_error(e)

        should_retry = (error_code == ERROR_RATE_LIMIT or error_code == ERROR_SERVER) and attempt < self.max_retries

        if should_retry:
            delay = self._get_delay()
            logging.warning(f"Error: {error_code}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{self.max_retries})")
            await asyncio.sleep(delay)
        else:
            if attempt == self.max_retries:
                error_code = ERROR_MAX_RETRIES
            return f"{ERROR_PREFIX}: {error_code} - {str(e)}"

    @db_cache_llm_call_async
    async def chat_async(self, system, history, gen_conf):
        base_history = deepcopy(history)
        if system:
            base_history.insert(0, {"role": "system", "content": system})
        gen_conf = self._clean_conf(deepcopy(gen_conf))

        for attempt in range(self.max_retries + 1):
            try:
                return await self._chat_async(deepcopy(base_history), gen_conf)
            except Exception as e:
                e = await self._exceptions_async(e, attempt)
                if e:
                    return e, 0
        assert False, "Shouldn't be here."

    async def chat_streamly_async(self, system, history, gen_conf):
        stream_history = deepcopy(history)
        if system:
            stream_history.insert(0, {"role": "system", "content": system})
        gen_conf = self._clean_conf(deepcopy(gen_conf))
        ans = ""
        total_tokens = 0
        reasoning_start = False
        try:
            response = await self.async_client.chat.completions.create(model=self.model_name, messages=stream_history, stream=True, **gen_conf)
            async for resp in response:
                if not resp.choices:
                    continue
                if not resp.choices[0].delta.content:
                    resp.choices[0].delta.content = ""
                if hasattr(resp.choices[0].delta, "reasoning_content") and resp.choices[0].delta.reasoning_content:
                    ans = ""
                    if not reasoning_start:
                        reasoning_start = True
                        ans = "<think>"
                    ans += resp.choices[0].delta.reasoning_content + "</think>"
                else:
                    reasoning_start = False
                    ans = resp.choices[0].delta.content

                tol = self.total_token_count(resp)
                if not tol:
                    total_tokens += num_tokens_from_string(resp.choices[0].delta.content)
                else:
                    total_tokens += tol

                if resp.choices[0].finish_reason == "length":
                    if is_chinese(ans):
                        ans += LENGTH_NOTIFICATION_CN
                    else:
                        ans += LENGTH_NOTIFICATION_EN
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class GptTurbo(Base):
    def __init__(self, key, model_name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", **kwargs):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        super().__init__(key, model_name, base_url, **kwargs)


class GeminiChat(Base):
    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        from google.generativeai import GenerativeModel, client

        client.configure(api_key=key)
        _client = client.get_default_generative_client()
        self.model_name = "models/" + model_name
        self.model = GenerativeModel(model_name=self.model_name)
        self.model._client = _client

    def _clean_conf(self, gen_conf):
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        return gen_conf

    def _chat(self, history, gen_conf):
        from google.generativeai.types import content_types

        system = history[0]["content"] if history and history[0]["role"] == "system" else ""
        hist = []
        for item in history:
            if item["role"] == "system":
                continue
            hist.append(deepcopy(item))
            item = hist[-1]
            if "role" in item and item["role"] == "assistant":
                item["role"] = "model"
            if "role" in item and item["role"] == "system":
                item["role"] = "user"
            if "content" in item:
                item["parts"] = item.pop("content")

        if system:
            self.model._system_instruction = content_types.to_content(system)
        response = self.model.generate_content(hist, generation_config=gen_conf)
        ans = response.text
        return ans, response.usage_metadata.total_token_count

    def chat_streamly(self, system, history, gen_conf):
        from google.generativeai.types import content_types

        gen_conf = self._clean_conf(gen_conf)
        if system:
            self.model._system_instruction = content_types.to_content(system)
        for item in history:
            if "role" in item and item["role"] == "assistant":
                item["role"] = "model"
            if "content" in item:
                item["parts"] = item.pop("content")
        ans = ""
        try:
            response = self.model.generate_content(history, generation_config=gen_conf, stream=True)
            for resp in response:
                ans = resp.text
                yield ans

            yield response._chunks[-1].usage_metadata.total_token_count
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield 0


class OpenAI_APIChat(Base):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("url cannot be None")
        model_name = model_name.split("___")[0]
        super().__init__(key, model_name, base_url)


class AnthropicChat(Base):
    def __init__(self, key, model_name, base_url="https://api.anthropic.com/v1/", **kwargs):
        if not base_url:
            base_url = "https://api.anthropic.com/v1/"
        super().__init__(key, model_name, base_url=base_url, **kwargs)
