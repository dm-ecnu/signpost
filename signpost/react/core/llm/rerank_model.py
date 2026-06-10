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
import re
import threading
from collections.abc import Iterable
from urllib.parse import urljoin

import requests
import httpx
from huggingface_hub import snapshot_download
import os
from abc import ABC
import numpy as np

from core import config
from core.config import get_home_cache_dir
from core.logging.config import log_exception
from core.utils import num_tokens_from_string, truncate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("Please implement encode method!")

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


class DefaultRerank(Base):
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        if not config.LIGHTEN and not DefaultRerank._model:
            import torch
            from FlagEmbedding import FlagReranker

            with DefaultRerank._model_lock:
                if not DefaultRerank._model:
                    try:
                        DefaultRerank._model = FlagReranker(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)), use_fp16=torch.cuda.is_available())
                    except Exception:
                        model_dir = snapshot_download(repo_id=model_name, local_dir=os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)), local_dir_use_symlinks=False)
                        DefaultRerank._model = FlagReranker(model_dir, use_fp16=torch.cuda.is_available())
        self._model = DefaultRerank._model
        self._dynamic_batch_size = 8
        self._min_batch_size = 1

    def torch_empty_cache(self):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error emptying cache: {e}")

    def _process_batch(self, pairs, max_batch_size=None):
        """template method for subclass call"""
        old_dynamic_batch_size = self._dynamic_batch_size
        if max_batch_size is not None:
            self._dynamic_batch_size = max_batch_size
        res = []
        i = 0
        while i < len(pairs):
            current_batch = self._dynamic_batch_size
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    batch_scores = self._compute_batch_scores(pairs[i : i + current_batch])
                    res.extend(batch_scores)
                    i += current_batch
                    self._dynamic_batch_size = min(self._dynamic_batch_size * 2, 8)
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and current_batch > self._min_batch_size:
                        current_batch = max(current_batch // 2, self._min_batch_size)
                        self.torch_empty_cache()
                        retry_count += 1
                    else:
                        raise
            if retry_count >= max_retries:
                raise RuntimeError("max retry times, still cannot process batch, please check your GPU memory")
            self.torch_empty_cache()

        self._dynamic_batch_size = old_dynamic_batch_size
        return np.array(res)

    def _compute_batch_scores(self, batch_pairs, max_length=None):
        if max_length is None:
            scores = self._model.compute_score(batch_pairs)
        else:
            scores = self._model.compute_score(batch_pairs, max_length=max_length)
        scores = sigmoid(np.array(scores)).tolist()
        if not isinstance(scores, Iterable):
            scores = [scores]
        return scores

    def similarity(self, query: str, texts: list):
        pairs = [(query, truncate(t, 2048)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 4096
        res = self._process_batch(pairs, max_batch_size=batch_size)
        return np.array(res), token_count


class OpenAI_APIRerank(Base):
    def __init__(self, key, model_name, base_url):
        if base_url.find("/rerank") == -1:
            self.base_url = urljoin(base_url, "/rerank")
        else:
            self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name.split("___")[0]

    def similarity(self, query: str, texts: list):
        texts = [truncate(t, 500) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res["results"]:
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)

        # Normalize the rank values to the range 0 to 1
        min_rank = np.min(rank)
        max_rank = np.max(rank)

        # Avoid division by zero if all ranks are identical
        if max_rank - min_rank != 0:
            rank = (rank - min_rank) / (max_rank - min_rank)
        else:
            rank = np.zeros_like(rank)

        return rank, token_count
