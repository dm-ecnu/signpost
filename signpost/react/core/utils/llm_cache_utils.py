#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import json
import xxhash
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from core.db.models import LLMCache, DB
from core import utils
from core.utils import num_tokens_from_string


def generate_cache_key(llm_name: str, system_prompt: str, history: List[Dict]) -> str:
    """
    Generate cache key for LLM calls.

    Key components:
    - llm_name: Model name (different models produce different outputs)
    - system_prompt: System prompt
    - history: Conversation history (the actual prompt content)

    Note: gen_conf is excluded because it's almost always the same (GRAPHRAG_DEFAULT_GEN_CONF).
    """
    hasher = xxhash.xxh64()
    hasher.update(llm_name.encode("utf-8"))
    hasher.update((system_prompt or "").encode("utf-8"))
    # Extract text content from history for stable hashing
    for msg in history:
        hasher.update(msg.get("role", "").encode("utf-8"))
        hasher.update(msg.get("content", "").encode("utf-8"))
    return hasher.hexdigest()


def extract_metadata_from_response(response_data: Any, is_stream: bool = False, processing_time_ms: Optional[int] = None, model_info: Optional[Dict] = None) -> Dict:
    """
    从 LLM 响应中提取元数据

    Args:
        response_data: LLM 响应数据
        is_stream: 是否为流式响应
        processing_time_ms: 处理时间(毫秒)
        model_info: 模型信息

    Returns:
        包含元数据的字典
    """
    metadata = {
        "stream": is_stream,
        "created_at": time.time(),
        "processing_time_ms": processing_time_ms,
    }

    if model_info:
        metadata["model_info"] = model_info

    # 如果是 OpenAI 格式的响应，尝试提取更多信息
    if hasattr(response_data, "usage"):
        metadata["usage"] = {
            "prompt_tokens": getattr(response_data.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response_data.usage, "completion_tokens", 0),
            "total_tokens": getattr(response_data.usage, "total_tokens", 0),
        }

    if hasattr(response_data, "model"):
        metadata["model_name"] = response_data.model

    if hasattr(response_data, "id"):
        metadata["request_id"] = response_data.id

    return metadata


def _ensure_connection():
    """Close stale connections and reopen if necessary."""
    try:
        # Peewee 的 PooledDatabase 在遇到 ConnectionError 时需要显式重连
        DB.close_stale(age=30)
        if DB.is_closed():
            DB.connect(reuse_if_open=True)
    except Exception as conn_err:
        logging.warning(f"Failed to refresh DB connection: {conn_err}")


def get_from_db_cache(cache_key: str) -> Optional[Tuple[str, Dict]]:
    """
    从数据库获取缓存

    Returns:
        (response_content, metadata) 或 None
    """
    try:
        with DB.connection_context():
            cache_record = LLMCache.get(LLMCache.cache_key == cache_key)
            return cache_record.response_content, cache_record.response_metadata
    except LLMCache.DoesNotExist:
        return None
    except Exception as e:
        logging.warning(f"Failed to get cache from database: {e}")
        _ensure_connection()
        return None


def save_to_db_cache(
    cache_key: str,
    llm_name: str,
    system_prompt: str,
    user_prompt: str,
    history: List[Dict],
    gen_conf: Dict,
    response_content: str,
    response_metadata: Dict,
    input_tokens: int = 0,
    output_tokens: int = 0,
    raw_request: Optional[Dict] = None,
) -> bool:
    """
    保存到数据库缓存

    Returns:
        是否保存成功
    """

    def normalize_message_content(message):
        """将 message.content 统一转换为字符串，避免非字符串导致的序列化问题"""
        content = message.get("content", "") if isinstance(message, dict) else message
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False, default=str)
        if content is None:
            return ""
        return str(content)

    try:
        # 计算 token 数量
        if not input_tokens:
            full_history = [{"role": "system", "content": system_prompt}] + history
            input_text = "\n".join([normalize_message_content(m) for m in full_history])
            input_tokens = num_tokens_from_string(input_text)

        if not output_tokens:
            if not isinstance(response_content, str):
                response_content = json.dumps(response_content, ensure_ascii=False, default=str)
            output_tokens = num_tokens_from_string(response_content)

        total_tokens = input_tokens + output_tokens

        # 准备原始请求数据
        if not raw_request:
            raw_request = {
                "llm_name": llm_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "history": history,
                "generation_config": gen_conf,
                "timestamp": time.time(),
            }

        # 确保JSON字段可序列化
        def ensure_json_serializable(data):
            """确保数据可以JSON序列化"""
            if data is None:
                return None
            try:
                json_str = json.dumps(data, ensure_ascii=False, default=str)
                return json.loads(json_str)
            except (TypeError, ValueError) as exc:
                logging.warning(f"Data not JSON serializable, converting: {exc}")
                return str(data)

        serialized_raw_request = ensure_json_serializable(raw_request)
        serialized_history = ensure_json_serializable(history)
        serialized_gen_conf = ensure_json_serializable(gen_conf)
        serialized_metadata = ensure_json_serializable(response_metadata)

        with DB.connection_context():
            with DB.atomic():
                (
                    LLMCache.insert(
                        id=utils.get_uuid(),
                        cache_key=cache_key,
                        llm_name=llm_name,
                        raw_request=serialized_raw_request,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        history_messages=serialized_history,
                        generation_config=serialized_gen_conf,
                        response_content=response_content,
                        response_metadata=serialized_metadata,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                    .on_conflict(
                        conflict_target=[LLMCache.cache_key],
                        update={
                            LLMCache.response_content: response_content,
                            LLMCache.response_metadata: serialized_metadata,
                            LLMCache.input_tokens: input_tokens,
                            LLMCache.output_tokens: output_tokens,
                            LLMCache.total_tokens: total_tokens,
                            LLMCache.llm_name: llm_name,
                            LLMCache.system_prompt: system_prompt,
                            LLMCache.user_prompt: user_prompt,
                            LLMCache.history_messages: serialized_history,
                            LLMCache.generation_config: serialized_gen_conf,
                            LLMCache.raw_request: serialized_raw_request,
                        },
                    )
                    .execute()
                )

        return True

    except Exception as e:
        logging.warning(f"Failed to cache LLM response: {e}", exc_info=True)
        return False


def db_cache_llm_call(llm_func):
    """
    数据库缓存装饰器，用于 LLM 调用

    Usage:
        @db_cache_llm_call
        def chat(self, system, history, gen_conf):
            # 原有的 LLM 调用逻辑
            return response, token_count
    """

    def wrapper(self, system, history, gen_conf):
        # 深拷贝原始参数以保留缓存键生成时的状态
        from copy import deepcopy

        original_history = deepcopy(history)
        original_system = system

        # 提取用户输入
        user_prompt = ""
        if original_history and isinstance(original_history, list) and len(original_history) > 0:
            user_prompt = original_history[-1].get("content", "") if isinstance(original_history[-1], dict) else str(original_history[-1])

        # 生成缓存键（不含 gen_conf，因为几乎都是固定配置）
        cache_key = generate_cache_key(self.model_name, original_system or "", original_history)

        # 尝试从数据库获取缓存
        cached_result = get_from_db_cache(cache_key)
        if cached_result:
            response_content, metadata = cached_result

            # 从元数据中提取 token 信息
            usage = metadata.get("usage", {})
            token_count = usage.get("total_tokens", 0)

            logging.debug(f"Cache hit for {self.model_name}: {cache_key[:8]}...")
            return response_content, token_count

        # 缓存未命中，调用原始函数
        start_time = time.time()
        try:
            response, token_count = llm_func(self, system, history, gen_conf)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # 准备元数据
            metadata = extract_metadata_from_response(
                None,  # 没有原始响应对象
                is_stream=False,
                processing_time_ms=processing_time_ms,
                model_info={"model": self.model_name},
            )

            # 从 token_count 推断 token 使用情况
            metadata["usage"] = {
                "total_tokens": token_count,
                "prompt_tokens": 0,  # 需要在实际使用中计算
                "completion_tokens": 0,
            }

            # 检查响应是否为错误，不缓存错误响应
            if not (isinstance(response, str) and "**ERROR**" in response):
                # 保存到数据库缓存（使用原始参数）
                save_to_db_cache(
                    cache_key=cache_key,
                    llm_name=self.model_name,
                    system_prompt=original_system or "",
                    user_prompt=user_prompt,
                    history=original_history,
                    gen_conf=gen_conf,
                    response_content=response,
                    response_metadata=metadata,
                    input_tokens=0,  # 会在 save_to_db_cache 中计算
                    output_tokens=0,  # 会在 save_to_db_cache 中计算
                )
            else:
                logging.debug(f"Skipping cache for error response: {response[:100]}")

            return response, token_count

        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise

    return wrapper


def db_cache_llm_call_async(llm_func):
    """
    异步数据库缓存装饰器，用于异步 LLM 调用
    """

    async def wrapper(self, system, history, gen_conf):
        # 深拷贝原始参数以保留缓存键生成时的状态
        from copy import deepcopy

        original_history = deepcopy(history)
        original_system = system

        # 提取用户输入
        user_prompt = ""
        if original_history and isinstance(original_history, list) and len(original_history) > 0:
            user_prompt = original_history[-1].get("content", "") if isinstance(original_history[-1], dict) else str(original_history[-1])

        # 生成缓存键（不含 gen_conf，因为几乎都是固定配置）
        cache_key = generate_cache_key(self.model_name, original_system or "", original_history)

        # 尝试从数据库获取缓存
        cached_result = get_from_db_cache(cache_key)
        if cached_result:
            response_content, metadata = cached_result

            # 从元数据中提取 token 信息
            usage = metadata.get("usage", {})
            token_count = usage.get("total_tokens", 0)

            logging.debug(f"Cache hit for {self.model_name}: {cache_key[:8]}...")
            return response_content, token_count

        # 缓存未命中，调用原始函数
        start_time = time.time()
        try:
            response, token_count = await llm_func(self, system, history, gen_conf)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # 准备元数据
            metadata = extract_metadata_from_response(
                None,  # 没有原始响应对象
                is_stream=False,
                processing_time_ms=processing_time_ms,
                model_info={"model": self.model_name},
            )

            # 从 token_count 推断 token 使用情况
            metadata["usage"] = {
                "total_tokens": token_count,
                "prompt_tokens": 0,  # 需要在实际使用中计算
                "completion_tokens": 0,
            }

            # 检查响应是否为错误，不缓存错误响应
            if not (isinstance(response, str) and "**ERROR**" in response):
                # 保存到数据库缓存（使用原始参数）
                save_to_db_cache(
                    cache_key=cache_key,
                    llm_name=self.model_name,
                    system_prompt=original_system or "",
                    user_prompt=user_prompt,
                    history=original_history,
                    gen_conf=gen_conf,
                    response_content=response,
                    response_metadata=metadata,
                    input_tokens=0,  # 会在 save_to_db_cache 中计算
                    output_tokens=0,  # 会在 save_to_db_cache 中计算
                )
            else:
                logging.debug(f"Skipping cache for error response: {response[:100]}")

            return response, token_count

        except Exception as e:
            logging.error(f"Async LLM call failed: {e}")
            raise

    return wrapper
