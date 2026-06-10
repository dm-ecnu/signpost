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
import logging
import re
import threading
import os
from abc import ABC

from huggingface_hub import snapshot_download
from openai import OpenAI, AsyncOpenAI
import numpy as np
import asyncio
import json

from core import config
from core.utils.asyncio_compat import CapacityLimiter
from core.config import get_home_cache_dir
from core.logging.config import log_exception
from core.utils.embedding_cache_utils import redis_cache_embedding, redis_cache_embedding_async
from core.utils import num_tokens_from_string, truncate
import google.generativeai as genai


class Base(ABC):
    _embedding_limiter = CapacityLimiter(20)

    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")

    async def async_encode(self, texts: list):
        """Default async encode implementation with concurrency limit and thread pool"""
        async with self._embedding_limiter:
            return await asyncio.to_thread(self.encode, texts)

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

    @classmethod
    def set_embedding_concurrency_limit(cls, limit: int):
        cls._embedding_limiter = CapacityLimiter(limit)


class DefaultEmbedding(Base):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _model = None
    _model_name = ""
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        if not config.LIGHTEN:
            with DefaultEmbedding._model_lock:
                from FlagEmbedding import FlagModel
                import torch

                if not DefaultEmbedding._model or model_name != DefaultEmbedding._model_name:
                    try:
                        DefaultEmbedding._model = FlagModel(
                            os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)),
                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                            use_fp16=torch.cuda.is_available(),
                        )
                        DefaultEmbedding._model_name = model_name
                    except Exception:
                        model_dir = snapshot_download(
                            repo_id="BAAI/bge-large-zh-v1.5", local_dir=os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)), local_dir_use_symlinks=False
                        )
                        DefaultEmbedding._model = FlagModel(model_dir, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=torch.cuda.is_available())
        self._model = DefaultEmbedding._model
        self._model_name = DefaultEmbedding._model_name

    def encode(self, texts: list):
        batch_size = 16
        texts = [truncate(t, 2048) for t in texts]
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        ress = []
        for i in range(0, len(texts), batch_size):
            ress.extend(self._model.encode(texts[i : i + batch_size]).tolist())
        return np.array(ress), token_count

    def encode_queries(self, text: str):
        token_count = num_tokens_from_string(text)
        return self._model.encode_queries([text]).tolist()[0], token_count


class OpenAIEmbed(Base):
    def __init__(self, key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1"):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    @redis_cache_embedding
    def encode(self, texts: list):
        batch_size = 16
        texts = [truncate(t, 8191) for t in texts]
        ress = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(input=texts[i : i + batch_size], model=self.model_name)
            try:
                ress.extend([d.embedding for d in res.data])
                total_tokens += self.total_token_count(res)
            except Exception as _e:
                log_exception(_e, res)
        return np.array(ress), total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=[truncate(text, 8191)], model=self.model_name)
        return np.array(res.data[0].embedding), self.total_token_count(res)

    @redis_cache_embedding_async
    async def async_encode(self, texts: list):
        """Asynchronous version of encode method with concurrency limit"""
        async with self._embedding_limiter:
            batch_size = 16
            texts = [truncate(t, 8191) for t in texts]
            ress = []
            total_tokens = 0
            for i in range(0, len(texts), batch_size):
                res = await self.async_client.embeddings.create(input=texts[i : i + batch_size], model=self.model_name)
                try:
                    ress.extend([d.embedding for d in res.data])
                    total_tokens += self.total_token_count(res)
                except Exception as _e:
                    log_exception(_e, res)
            return np.array(ress), total_tokens


class GeminiEmbed(Base):
    def __init__(self, key, model_name="models/text-embedding-004", **kwargs):
        self.key = key
        self.model_name = "models/" + model_name

    def encode(self, texts: list):
        texts = [truncate(t, 2048) for t in texts]
        token_count = sum(num_tokens_from_string(text) for text in texts)
        genai.configure(api_key=self.key)
        batch_size = 16
        ress = []
        for i in range(0, len(texts), batch_size):
            result = genai.embed_content(model=self.model_name, content=texts[i : i + batch_size], task_type="retrieval_document", title="Embedding of single string")
            try:
                ress.extend(result["embedding"])
            except Exception as _e:
                log_exception(_e, result)
        return np.array(ress), token_count

    def encode_queries(self, text):
        genai.configure(api_key=self.key)
        result = genai.embed_content(model=self.model_name, content=truncate(text, 2048), task_type="retrieval_document", title="Embedding of single string")
        token_count = num_tokens_from_string(text)
        try:
            return np.array(result["embedding"]), token_count
        except Exception as _e:
            log_exception(_e, result)


class OpenAI_APIEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("url cannot be None")
        from urllib.parse import urljoin

        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name.split("___")[0]
