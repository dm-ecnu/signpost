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
import json
import logging
import xxhash
import numpy as np
from typing import List, Dict
from core.storage.redis_conn import REDIS_CONN

# 默认缓存过期时间：3天
DEFAULT_EMBEDDING_CACHE_TTL = 72 * 3600


def generate_embedding_cache_key(model_name: str, text: str) -> str:
    """
    为单个文本生成embedding缓存键

    Args:
        model_name: 模型名称
        text: 文本内容

    Returns:
        缓存键的十六进制hash
    """
    hasher = xxhash.xxh64()
    hasher.update(str(model_name).encode("utf-8"))
    hasher.update(str(text).encode("utf-8"))
    return hasher.hexdigest()


def get_embeddings_from_redis(model_name: str, texts: List[str]) -> Dict[int, np.ndarray]:
    """
    批量从Redis获取embedding缓存

    Args:
        model_name: 模型名称
        texts: 文本列表

    Returns:
        字典 {text_index: embedding向量}，只包含缓存命中的结果
    """
    if not REDIS_CONN.is_alive():
        return {}

    # 生成所有缓存键
    cache_keys = [generate_embedding_cache_key(model_name, text) for text in texts]

    try:
        # 批量读取
        pipeline = REDIS_CONN.REDIS.pipeline()
        for key in cache_keys:
            pipeline.get(key)
        cached_values = pipeline.execute()

        # 解析命中的结果
        results = {}
        hit_count = 0
        for idx, cached_bin in enumerate(cached_values):
            if cached_bin:
                try:
                    embedding = np.array(json.loads(cached_bin))
                    results[idx] = embedding
                    hit_count += 1
                except Exception as e:
                    logging.warning(f"Failed to deserialize embedding cache for text {idx}: {e}")

        if hit_count > 0:
            logging.debug(f"Embedding cache hit: {hit_count}/{len(texts)} for model {model_name}")

        return results

    except Exception as e:
        logging.warning(f"Failed to get embeddings from Redis: {e}")
        return {}


def save_embeddings_to_redis(model_name: str, texts: List[str], embeddings: np.ndarray, ttl: int = DEFAULT_EMBEDDING_CACHE_TTL) -> bool:
    """
    批量保存embedding结果到Redis

    Args:
        model_name: 模型名称
        texts: 文本列表
        embeddings: embedding向量数组
        ttl: 缓存过期时间（秒），默认3天

    Returns:
        是否保存成功
    """
    if not REDIS_CONN.is_alive():
        return False

    try:
        pipeline = REDIS_CONN.REDIS.pipeline()

        for idx, text in enumerate(texts):
            cache_key = generate_embedding_cache_key(model_name, text)
            embedding = embeddings[idx] if len(embeddings.shape) > 1 else embeddings

            # 序列化为JSON
            arr_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            pipeline.setex(cache_key, ttl, arr_json.encode("utf-8"))

        pipeline.execute()
        logging.debug(f"Saved {len(texts)} embeddings to Redis for model {model_name}")
        return True

    except Exception as e:
        logging.warning(f"Failed to save embeddings to Redis: {e}")
        return False


def redis_cache_embedding(encode_func):
    """
    同步embedding缓存装饰器

    适用于返回格式为 (np.ndarray, int) 的encode方法
    缓存命中时返回 token_count = 0

    Usage:
        @redis_cache_embedding
        def encode(self, texts: list):
            return embeddings_array, token_count
    """

    def wrapper(self, texts: List[str]):
        model_name = self.model_name

        # 1. 尝试从Redis批量读取
        cached_results = get_embeddings_from_redis(model_name, texts)

        # 2. 如果全部命中，直接返回
        if len(cached_results) == len(texts):
            embeddings = np.array([cached_results[i] for i in range(len(texts))])
            return embeddings, 0  # 缓存命中返回0 token

        # 3. 收集未命中的文本
        uncached_indices = [i for i in range(len(texts)) if i not in cached_results]
        uncached_texts = [texts[i] for i in uncached_indices]

        # 4. 调用原始函数获取未缓存的embeddings
        if uncached_texts:
            logging.debug(f"Embedding cache miss: {len(uncached_texts)}/{len(texts)} for model {model_name}")
            uncached_embeddings, total_tokens = encode_func(self, uncached_texts)

            # 5. 保存新结果到Redis
            save_embeddings_to_redis(model_name, uncached_texts, uncached_embeddings)

            # 6. 合并缓存结果和新结果，保持原始顺序
            all_embeddings = []
            uncached_idx = 0

            for i in range(len(texts)):
                if i in cached_results:
                    all_embeddings.append(cached_results[i])
                else:
                    all_embeddings.append(uncached_embeddings[uncached_idx])
                    uncached_idx += 1

            return np.array(all_embeddings), total_tokens
        else:
            # 全部命中缓存
            embeddings = np.array([cached_results[i] for i in range(len(texts))])
            return embeddings, 0

    return wrapper


def redis_cache_embedding_async(encode_func):
    """
    异步embedding缓存装饰器

    适用于返回格式为 (np.ndarray, int) 的async encode方法
    缓存命中时返回 token_count = 0

    Usage:
        @redis_cache_embedding_async
        async def async_encode(self, texts: list):
            return embeddings_array, token_count
    """

    async def wrapper(self, texts: List[str]):
        import asyncio

        model_name = self.model_name

        # 1. 在异步环境中调用同步的Redis操作（使用线程池）
        loop = asyncio.get_event_loop()
        cached_results = await loop.run_in_executor(None, get_embeddings_from_redis, model_name, texts)

        # 2. 如果全部命中，直接返回
        if len(cached_results) == len(texts):
            embeddings = np.array([cached_results[i] for i in range(len(texts))])
            return embeddings, 0  # 缓存命中返回0 token

        # 3. 收集未命中的文本
        uncached_indices = [i for i in range(len(texts)) if i not in cached_results]
        uncached_texts = [texts[i] for i in uncached_indices]

        # 4. 调用原始异步函数获取未缓存的embeddings
        if uncached_texts:
            logging.debug(f"Embedding cache miss: {len(uncached_texts)}/{len(texts)} for model {model_name}")
            uncached_embeddings, total_tokens = await encode_func(self, uncached_texts)

            # 5. 保存新结果到Redis（异步执行）
            await loop.run_in_executor(None, save_embeddings_to_redis, model_name, uncached_texts, uncached_embeddings)

            # 6. 合并缓存结果和新结果，保持原始顺序
            all_embeddings = []
            uncached_idx = 0

            for i in range(len(texts)):
                if i in cached_results:
                    all_embeddings.append(cached_results[i])
                else:
                    all_embeddings.append(uncached_embeddings[uncached_idx])
                    uncached_idx += 1

            return np.array(all_embeddings), total_tokens
        else:
            # 全部命中缓存
            embeddings = np.array([cached_results[i] for i in range(len(texts))])
            return embeddings, 0

    return wrapper
