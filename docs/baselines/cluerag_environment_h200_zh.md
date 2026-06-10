# Clue-RAG 环境配置与 H200 迁移说明

> 当前正式实验默认使用 `baselines/ClueRAG/signpost_adapter/H200_MIGRATION_RUNBOOK.zh.md`
> 中的 `shared_es` 路径：复用 Signpost 已有 `chunks.jsonl` 和
> `semantic_llm.extractions.jsonl`，直接使用主 `signpost-re` 环境、现有 ES 和本地
> NVIDIA rerank 服务。本文保留的 ClueRAG 官方环境、OceanBase、`RUN_OFFICIAL=1`
> 内容仅用于 `official_oceanbase` 可选诊断路径。

本文档只覆盖 Clue-RAG baseline 的环境、服务、计时和 token 统计。方法设计、输入输出 schema 和运行入口见 `docs/baselines/cluerag_baseline_zh.md`。

## 0. 当前状态

截至当前版本：

```text
已完成：
1. 官方仓库已放在 baselines/ClueRAG/。
2. Signpost 数据到 Clue-RAG 官方数据格式的转换已完成。
3. Clue-RAG 官方输出到 Signpost 统一 prediction schema 的转换已完成。
4. wrapper 可以可选调用官方 pipeline，并记录 stage timing 与 token metadata。

未完成：
1. 没有在本机安装 Clue-RAG 官方依赖。
2. 没有把 Clue-RAG 依赖装进 signpost-re conda 环境。
3. H200 上还需要单独配置 Clue-RAG Python 环境、OceanBase/MySQL 向量数据库、rerank 服务。
```

当前 H200 实际项目目录是：

```bash
PROJECT_DIR=/home/srl/signpost_re
```

不要使用旧文档中的 `/data/srl/signpost_re` 作为项目目录。`/data/srl` 仍然只用于模型文件，例如 `/data/srl/Llama-3.3-70B-FP8`。

Clue-RAG 不应在 Vanilla LLM / Hybrid RAG 正在跑的时候直接启动官方 full pipeline。它会额外占用 LLM、embedding、数据库和 rerank 服务。可以先做 prepare-only 数据转换；正式 `RUN_OFFICIAL=1` 等前两个 baseline 和 legal 离线任务结束后再跑。

另外，官方 `baselines/ClueRAG/requirements.txt` 中包含若干本机构建路径依赖，例如 `cupy @ file:///...`、`numpy @ file:///...`。在 H200 上直接 `pip install -r requirements.txt` 可能失败。建议先按第 2.2 节建立环境，再根据报错补装依赖；不要污染 `signpost-re` 环境。

不建议把 Clue-RAG 官方依赖直接装进 `signpost-re`。官方依赖包括 `torch`、`FlagEmbedding`、`spacy`、`en_core_web_trf`、`pymilvus`、`milvus-lite`、`pymysql` 等，和 Signpost 主环境耦合后很容易污染已有实验环境。推荐在 H200 上使用两个环境：

```text
signpost-re  # 跑 Signpost wrapper、数据转换、评估、method summary
cluerag      # 跑 Clue-RAG 官方代码依赖
```

目前 `scripts/baselines/run_cluerag_method.sh` 默认用当前 shell 的 `python`。因此如果要 `RUN_OFFICIAL=1`，请在 `cluerag` 环境中运行脚本；如果只做 prepare/convert/eval，则在 `signpost-re` 环境中运行即可。

## 1. Clue-RAG 官方依赖边界

官方仓库的关键模块：

```text
baselines/ClueRAG/
  dataset/dataclass.py
  index/hybrid_extraction.py
  index/construction.py
  index/embedding.py
  index/rerank.py
  index/ob_connection.py
  retrieval/retrieval.py
  generation/generation.py
  utils/config.py
```

重要依赖结论：

```text
1. LLM: OpenAI-compatible /chat/completions。
2. Embedding: OpenAI-compatible /embeddings。
3. Rerank: 默认 http://127.0.0.1:8033/v1/rerank。
4. DB: 官方 OceanBase augmented version 使用 pymysql 连接 OceanBase/MySQL，并创建 VECTOR(...) 表和 VECTOR INDEX。
```

第四点很关键。`MultiLayerGraph` 会创建如下类型的表：

```sql
CREATE TABLE ... (
  embedding VECTOR(<dimension>)
);
CREATE VECTOR INDEX ... WITH (distance=cosine, type=HNSW_SQ, ...);
```

普通 MySQL 不支持 `VECTOR` 和 `CREATE VECTOR INDEX`，所以严格复现官方代码需要 OceanBase 向量能力或兼容这些 SQL 的数据库。没有这块时，Clue-RAG 官方完整 pipeline 不能算跑通。

## 2. H200 推荐环境

假设项目位于：

```bash
cd /home/srl/signpost_re
```

### 2.1 保留 Signpost 主环境

用于数据准备、统一评估、结果汇总：

```bash
conda activate signpost-re
set -a
source .env.h200
set +a
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

H200 已确认的本地服务：

```text
Chat:
  URL:   http://localhost:8000/v1
  Model: /data/srl/Llama-3.3-70B-FP8

Embedding:
  URL:   http://localhost:8001/v1/embeddings
  Model: /data/srl/nemotron-8b
```

### 2.2 新建 Clue-RAG 环境

官方 README 写的是 Python 3.9，但本项目 `pyproject.toml` 要求 Python `>=3.11,<3.13`。
Clue-RAG wrapper 需要在同一环境中 import `signpost` 的统一 schema、benchmark 和 runner，
因此 H200 上实际必须使用 Python 3.11。不要使用 Python 3.9/3.10，否则
`python -m pip install -e .` 会报：

```text
Package 'signpost-re' requires a different Python: 3.10.xx not in '<3.13,>=3.11'
```

建环境：

```bash
conda create -n cluerag python=3.11 -y
conda activate cluerag
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
python -m pip install --upgrade pip
```

先不要盲目执行完整 `requirements.txt`。该文件包含本机构建路径 wheel，H200 上可能不可用。建议先安装最小运行依赖：

```bash
python -m pip install \
  aiohttp \
  beautifulsoup4 \
  datasets \
  fastapi \
  FlagEmbedding \
  ijson \
  inscriptis \
  networkx \
  nltk \
  numpy \
  openai \
  pandas \
  pymilvus \
  PyMySQL \
  pyyaml \
  requests \
  scikit-learn \
  scipy \
  spacy==3.8.7 \
  tiktoken \
  torch \
  tqdm \
  transformers \
  ujson
```

再安装 spaCy 模型：

```bash
python -m spacy download en_core_web_trf
```

如果 `spacy download` 受网络限制，改用离线包：

```bash
python -m pip install /home/srl/packages/en_core_web_trf-3.8.0.tar.gz
```

离线包可先在本地下载，再传到 H200。

如果你确认 H200 可以处理 `requirements.txt` 中的本地 wheel 路径问题，也可以尝试：

```bash
python -m pip install -r baselines/ClueRAG/requirements.txt
```

但失败后不要在 `signpost-re` 环境里继续试；只在 `cluerag` 环境内处理。

### 2.3 在 cluerag 环境里补 Signpost 可编辑安装

因为 wrapper 在 `signpost/baselines/cluerag.py`，`cluerag` 环境也需要能 import Signpost：

```bash
conda activate cluerag
cd /home/srl/signpost_re
python -m pip install -e .
```

验证：

```bash
python -m scripts.baselines.run_cluerag --help
python - <<'PY'
import sys
sys.path.insert(0, "baselines/ClueRAG")
from utils.config import BaseConfig
from dataset.dataclass import Dataset
print(BaseConfig(dataset_name="signpost_legal_test").dataset_name)
PY
```

## 3. H200 服务配置

### 3.1 LLM 和 embedding

wrapper 会把 Signpost `.env.h200` 映射到 Clue-RAG 官方 config：

```text
config.llm_base_url        <- ECNU_API_BASE
config.llm_name            <- ECNU_CHAT_MODEL
config.api_key             <- ECNU_API_KEY or EMPTY
config.embedding_model_url <- ECNU_EMBEDDING_API_BASE or ECNU_API_BASE
config.embedding_model_name<- ECNU_EMBEDDING_MODEL
```

注意：Signpost 自己的 embedding client 可以使用 `http://localhost:8001/v1/embeddings` 这种完整 endpoint；Clue-RAG 官方使用 OpenAI Python client，会自动追加 `/embeddings`。因此 wrapper 会在传给 Clue-RAG 时把末尾的 `/embeddings` 去掉，实际传入官方 config 的 base URL 是 `http://localhost:8001/v1`。

因此正式运行前必须在 shell 中加载：

```bash
set -a
source .env.h200
set +a
```

### 3.2 OceanBase / MySQL 向量数据库

Clue-RAG 官方默认配置：

```text
db_host=127.0.0.1
db_port=32881
db_user=yaodong@mysql
db_password=yd0987
db_default_database=clueragdb
```

这些默认值不应直接用于 H200。wrapper 支持用环境变量覆盖：

```bash
export CLUERAG_DB_HOST=127.0.0.1
export CLUERAG_DB_PORT=32881
export CLUERAG_DB_USER='<your_user>'
export CLUERAG_DB_PASSWORD='<your_password>'
export CLUERAG_DB_NAME=clueragdb
```

最低验证：

```bash
python - <<'PY'
import pymysql, os
conn = pymysql.connect(
    host=os.environ["CLUERAG_DB_HOST"],
    port=int(os.environ["CLUERAG_DB_PORT"]),
    user=os.environ["CLUERAG_DB_USER"],
    password=os.environ["CLUERAG_DB_PASSWORD"],
    charset="utf8mb4",
    autocommit=True,
)
cur = conn.cursor()
cur.execute("SELECT 1")
print(cur.fetchall())
conn.close()
PY
```

向量能力验证必须包含 `VECTOR`：

```bash
python - <<'PY'
import pymysql, os
conn = pymysql.connect(
    host=os.environ["CLUERAG_DB_HOST"],
    port=int(os.environ["CLUERAG_DB_PORT"]),
    user=os.environ["CLUERAG_DB_USER"],
    password=os.environ["CLUERAG_DB_PASSWORD"],
    charset="utf8mb4",
    autocommit=True,
)
cur = conn.cursor()
cur.execute("CREATE DATABASE IF NOT EXISTS clueragdb CHARACTER SET utf8mb4")
cur.execute("USE clueragdb")
cur.execute("DROP TABLE IF EXISTS cluerag_vector_smoke")
cur.execute("CREATE TABLE cluerag_vector_smoke (id VARCHAR(64) PRIMARY KEY, embedding VECTOR(4))")
cur.execute("DROP TABLE cluerag_vector_smoke")
print("vector db ok")
conn.close()
PY
```

如果这里失败，不要继续跑 `RUN_OFFICIAL=1`，因为图构建阶段一定会失败。

### 3.3 Rerank 服务

Clue-RAG 官方 `index/rerank.py` 期望的请求格式：

```json
{
  "model": "BGE-M3-RERANKER",
  "query": "question text",
  "documents": ["doc 1", "doc 2"],
  "return_documents": false
}
```

可接受的返回格式之一：

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 1, "relevance_score": 0.12}
  ]
}
```

推荐模型：

```text
首选：BAAI/bge-reranker-v2-m3
理由：与官方默认的 BGE-M3-RERANKER 语义最接近，多语言，资源开销相对可控。

备选：Qwen/Qwen3-Reranker-4B
理由：能力可能更强，但部署更重，API 兼容性更容易出问题。
```

正式实验建议使用 `BAAI/bge-reranker-v2-m3`，并在 H200 上本地服务化，不调用公网 API。是否必须用 vLLM 取决于服务器上的 vLLM 版本是否支持 rerank/score API。若 vLLM 不能直接暴露上述 `/v1/rerank` 格式，建议用 `FlagEmbedding` 写一个轻量 FastAPI rerank server，接口保持和官方 `index/rerank.py` 一致。

wrapper 支持：

```bash
export CLUERAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
export CLUERAG_RERANK_URL=http://127.0.0.1:8033/v1/rerank
```

或运行脚本时传：

```bash
RERANK_URL=http://127.0.0.1:8033/v1/rerank scripts/baselines/run_cluerag_method.sh legal_test legal_test
```

rerank smoke：

```bash
python - <<'PY'
import os, requests
url = os.environ.get("CLUERAG_RERANK_URL", "http://127.0.0.1:8033/v1/rerank")
model = os.environ.get("CLUERAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
payload = {
    "model": model,
    "query": "What is deep learning?",
    "documents": [
        "Deep learning is a subset of machine learning based on neural networks.",
        "Tomato soup is a food."
    ],
    "return_documents": False,
}
resp = requests.post(url, json=payload, timeout=60)
print(resp.status_code)
print(resp.text[:1000])
PY
```

如果 rerank 服务失败，官方代码会记录错误并返回 0 分。这样 pipeline 可能继续跑，但不是公平的 Clue-RAG 主表设置。

## 4. 时间与 token 如何测量

### 4.1 是否修改官方代码

当前设计不直接改官方 Clue-RAG 代码。原因：

```text
1. 保持 baseline 尽量接近官方实现，减少“改坏 baseline”的风险。
2. 用 Signpost wrapper import 官方模块，在模块调用边界外计时。
3. 读取官方模块本来就返回/缓存的 metadata，转换到统一实验 schema。
```

如果后面发现官方代码内部某些阶段无法准确区分，可以再做最小 patch，但默认不改。

### 4.2 外层 wall time

`scripts/baselines/run_cluerag_method.sh` 用 `signpost.benchmark.time_stage` 包住阶段：

```text
baseline_prepare_cluerag  # 数据格式转换
baseline_cluerag_full     # 官方 extraction + graph construction + retrieval + generation
baseline_convert_cluerag  # 已有官方输出转换
baseline_eval_cluerag     # basic eval
```

输出：

```text
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

`baseline_cluerag_full` 是论文里可用于 Clue-RAG 总开销的 wall-clock 统计。

### 4.3 内部分阶段时间

wrapper 在 `run_cluerag_official()` 内部额外记录：

```text
dataset_load_seconds
hybrid_extraction_seconds
graph_construction_seconds
retrieval_seconds
generation_seconds
```

输出：

```text
outputs/<dataset>/baselines/cluerag/run_status.json
```

其中：

```text
hybrid_extraction_seconds + graph_construction_seconds
  可视为 Clue-RAG offline indexing time 的主体。

retrieval_seconds + generation_seconds
  可视为 Clue-RAG online query time 的主体。
```

### 4.4 token 统计

Clue-RAG 官方 LLM wrapper 会返回：

```text
prompt_tokens
completion_tokens
total_tokens
num_requests
```

这些 metadata 来自 OpenAI-compatible chat completion response 的 `usage` 字段。wrapper 记录三类 metadata：

```text
hybrid_metadata      # offline hybrid extraction 的 LLM token
retrieval_metadata   # retrieval 阶段 query/entity extraction 的 LLM token
generation_metadata  # answer generation 的 LLM token
```

保存位置：

```text
outputs/<dataset>/baselines/cluerag/run_status.json
outputs/<dataset>/baselines/cluerag/official_outputs/COSINE_1.00/retrieval_results.json
outputs/<dataset>/baselines/cluerag/official_outputs/COSINE_1.00/generation_results.json
```

转换为 Signpost 统一 prediction 时：

```text
online prompt tokens      = retrieval_metadata.prompt_tokens + generation_metadata.prompt_tokens
online completion tokens  = retrieval_metadata.completion_tokens + generation_metadata.completion_tokens
online llm calls/query    = (retrieval_metadata.num_requests + generation_metadata.num_requests) / query_count
```

`hybrid_metadata` 不写入每条 query 的 online cost。它应作为 Clue-RAG 官方离线构建阶段的记录项保留，用于透明报告和附录分析。按当前 v10 口径，如果该阶段属于共享语义标注或复用 Signpost 的 F6 实体/关系产物，则默认不计入主表中的方法离线成本；如果 Clue-RAG 必须执行自己的 graph/index construction，则从该方法自己的图组织和索引构建阶段开始计入离线成本。

注意：

```text
1. Embedding 服务通常不返回 token usage；当前记录 wall time 和 embedding call 的阶段开销。
2. Rerank 服务通常不返回 token usage；当前记录在 retrieval wall time 中。
3. 如果需要论文中单独报告 embedding/rerank token，可后续用 tokenizer 离线估算，但主实验先以 LLM token、wall time、disk size 为主。
```

## 5. 本地如何验证能迁移到 H200

本地没有 H200 算力和 OceanBase 时，不应尝试完整跑官方 Clue-RAG。可做迁移前 smoke：

### 5.1 Signpost wrapper 单测

```bash
cd /home/ruolinsu/signpost/signpost_re
conda activate signpost-re
python -m pytest tests/test_baselines.py
```

### 5.2 数据转换 smoke

```bash
LIMIT=3 scripts/baselines/run_cluerag_method.sh legal_test legal_test
```

预期输出：

```text
[cluerag] prepared inputs only. Set RUN_OFFICIAL=1 ...
```

检查：

```bash
test -s baselines/ClueRAG/data/signpost_legal_test.json
test -s baselines/ClueRAG/data/signpost_legal_test_corpus.json
test -s outputs/legal_test/baselines/cluerag/manifest.json
```

### 5.3 官方模块 import smoke

如果本地已装 `cluerag` 环境：

```bash
conda activate cluerag
cd /home/ruolinsu/signpost/signpost_re
python - <<'PY'
import sys
sys.path.insert(0, "baselines/ClueRAG")
from utils.config import BaseConfig
from dataset.dataclass import Dataset
print(BaseConfig(dataset_name="signpost_legal_test"))
PY
```

这一步只验证 Python 依赖和路径，不代表官方完整 pipeline 能跑。

## 6. H200 最小闭环

建议在 `tmux` 里跑，避免 SSH 断开：

```bash
tmux new -s cluerag-smoke
```

加载环境：

```bash
cd /home/srl/signpost_re
conda activate cluerag
set -a
source .env.h200
set +a
export CLUERAG_DB_HOST=127.0.0.1
export CLUERAG_DB_PORT=32881
export CLUERAG_DB_USER='<your_user>'
export CLUERAG_DB_PASSWORD='<your_password>'
export CLUERAG_DB_NAME=clueragdb
export CLUERAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
export CLUERAG_RERANK_URL=http://127.0.0.1:8033/v1/rerank
```

先做三个 smoke：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
python -m scripts.baselines.run_cluerag --help
```

再做 Clue-RAG legal_test 最小闭环：

```bash
LIMIT=3 RUN_OFFICIAL=1 LLM_PROCESSES=1 NUM_PROCESSES=1 EMBEDDING_BATCH_SIZE=16 \
  scripts/baselines/run_cluerag_method.sh legal_test legal_test
```

成功后检查：

```bash
test -s outputs/legal_test/baselines/cluerag/run_status.json
test -s outputs/legal_test/baselines/cluerag/official_outputs/COSINE_1.00/retrieval_results.json
test -s outputs/legal_test/baselines/cluerag/official_outputs/COSINE_1.00/generation_results.json
test -s outputs/legal_test/predictions/cluerag.jsonl
test -s outputs/legal_test/metrics/cluerag.query_metrics.json
test -s outputs/legal_test/metrics/cost_quality.json
```

看关键统计：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/legal_test/baselines/cluerag/run_status.json")
data = json.loads(p.read_text())
print(json.dumps({
    "wall_time_seconds": data.get("wall_time_seconds"),
    "stage_timings": data.get("stage_timings"),
    "hybrid_metadata": data.get("hybrid_metadata"),
    "retrieval_metadata": data.get("retrieval_metadata"),
    "generation_metadata": data.get("generation_metadata"),
}, indent=2, ensure_ascii=False))
PY
```

## 7. 正式运行顺序

Clue-RAG 最小 smoke 通过后，按数据集逐个跑，不要和 Signpost full run 同时抢 H200：

```bash
RUN_OFFICIAL=1 LLM_PROCESSES=1 NUM_PROCESSES=1 EMBEDDING_BATCH_SIZE=64 \
  scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

Legal 如果使用 full：

```bash
RUN_OFFICIAL=1 LLM_PROCESSES=1 NUM_PROCESSES=1 EMBEDDING_BATCH_SIZE=64 \
  scripts/baselines/run_cluerag_method.sh legal legal
```

如果先跑 Legal 子集，应使用已经在 Signpost processed dataset 中固定好的子集名，不能临时在命令里随意截断，否则 baseline 和 Signpost 不可比。

## 8. 失败定位

常见错误和含义：

```text
ModuleNotFoundError: spacy / FlagEmbedding / pymilvus
  Clue-RAG 官方依赖没装完整；确认当前是 cluerag 环境。

OceanBase Connection Failed
  CLUERAG_DB_* 没配，或数据库服务没启动。

Database Operation Error near VECTOR
  当前数据库不是 OceanBase 向量版本，或不支持 VECTOR 类型。

Rerank API failed
  rerank 服务没启动或返回格式不兼容。不要把全 0 rerank 结果作为正式主表。

embedding endpoint 404
  wrapper 已经会把 ECNU_EMBEDDING_API_BASE 末尾的 /embeddings 去掉再传给 Clue-RAG。
  如果仍然 404，说明 H200 embedding 服务的 OpenAI-compatible base URL 不是 http://localhost:8001/v1，需要用 smoke 单独核对服务路由。
```
