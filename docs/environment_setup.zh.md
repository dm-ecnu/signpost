# Signpost 科研环境搭建

本指南说明如何为重构后的 `signpost_re` 项目搭建一个干净的科研环境。目标不是运行一个多用户产品后端，而是支持两条清晰的科研工作流：

1. Index 阶段：文档预处理、分块、嵌入、GraphRAG 抽取、RAPTOR/树摘要、图持久化、ES 同步。
2. Retrieval 阶段：混合检索、图检索、Signpost 离线/在线导航、Agent 检索，以及评估输出生成。

具体 benchmark 指标和测速字段在这里有意不固定。重构后的代码应该让 index 和 retrieval 两个阶段都能独立运行，之后可以随时添加或调整测量逻辑。

当前 `signpost-main` 代码仍然包含面向产品的残留内容，例如用户、租户、权限、会话、Canvas、API token 和前端支持。在 `signpost_re` 中，这些内容应被替换为单一实验命名空间，例如 `default` 或数据集名称。

## 0. 将要安装的内容

Python 使用 Conda，外部服务使用 Docker。

必需的本地服务：

- Elasticsearch 8.x：BM25、稠密向量检索、chunk/entity/edge/RAPTOR 文档。
- MinIO：原始文档和持久化的 NetworkX 图文件。
- Valkey/Redis：队列、锁、缓存、可选 SSE/事件流。
- MySQL 或 PostgreSQL：最小元数据和 LLM 缓存。对于科研版，PostgreSQL 更容易保持自包含，并且有更好的 JSON 支持。

模型提供方：

- 本项目默认：ECNU OpenAI-compatible API。
- 后备/可选：标准 OpenAI-compatible API。保留这条路径，因为之后协作者可能仍然会使用它。

`signpost_re` 推荐版本：

- Python：3.11
- 包安装器：`uv`
- 数据库：PostgreSQL 16
- Elasticsearch：8.12.x
- MinIO：最新稳定 Docker 镜像
- Valkey：7.x

## 1. 目录布局

从工作区根目录开始：

```bash
cd /home/ruolinsu/signpost
```

推荐工作布局：

```text
signpost/
  signpost-main/       # 旧版/参考代码库
  signpost_re/         # 重构后的科研代码库
    docs/
    docker/
    conf/
    datasets/
    logs/
    outputs/
```

创建运行时目录：

```bash
mkdir -p signpost_re/{conf,datasets,logs,outputs,docker}
```

## 2. 启动 Docker 服务

创建 `signpost_re/docker/docker-compose.yml`：

```yaml
services:
  postgres:
    image: postgres:16
    container_name: signpost-postgres
    environment:
      POSTGRES_USER: signpost
      POSTGRES_PASSWORD: signpost
      POSTGRES_DB: signpost
    ports:
      - "5432:5432"
    volumes:
      - signpost-postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U signpost -d signpost"]
      interval: 5s
      timeout: 5s
      retries: 30

  valkey:
    image: valkey/valkey:7.2
    container_name: signpost-valkey
    ports:
      - "6379:6379"
    volumes:
      - signpost-valkey:/data
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 30

  minio:
    image: minio/minio:latest
    container_name: signpost-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: signpost
      MINIO_ROOT_PASSWORD: signpost123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - signpost-minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 30

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.1
    container_name: signpost-es
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: "-Xms2g -Xmx2g"
    ports:
      - "9200:9200"
    volumes:
      - signpost-es:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:9200 >/dev/null || exit 1"]
      interval: 10s
      timeout: 10s
      retries: 30

volumes:
  signpost-postgres:
  signpost-valkey:
  signpost-minio:
  signpost-es:
```

启动服务：

```bash
cd /home/ruolinsu/signpost/signpost_re/docker
docker compose up -d
```

检查服务健康状态：

```bash
docker compose ps
curl http://localhost:9200
curl http://localhost:9000/minio/health/live
```

MinIO 控制台：

```text
http://localhost:9001
user: signpost
password: signpost123
```

## 3. 创建 Conda 环境

```bash
conda create -n signpost-re python=3.11 -y
conda activate signpost-re
python -m pip install -U pip uv
```

安装重构后的科研项目本体。建议使用 editable 安装，这样后续修改源码后不需要重复安装：

```bash
cd /home/ruolinsu/signpost/signpost_re
python -m pip install -e .
uv pip install pytest pytest-asyncio requests-toolbelt psycopg2-binary
```

旧项目 `signpost-main` 只作为参考代码库，不再作为 `PYTHONPATH` 或主包安装目标。

可选 GPU/本地 embedding 依赖：

```bash
uv pip install "torch>=2.5,<3" flagembedding==1.2.10
```

对于第一次 CPU-only 环境，可以跳过可选 GPU 这一行，改用 OpenAI-compatible embedding API。ECNU 提供 OpenAI-compatible 的 chat、embedding 和 rerank 模型，因此无需本地 GPU 依赖也可以使用。

## 4. 环境变量

创建 `signpost_re/.env`：

```bash
# Python/runtime
PYTHONPATH=/home/ruolinsu/signpost/signpost_re
RAG_PROJECT_BASE=/home/ruolinsu/signpost/signpost_re
DOC_ENGINE=elasticsearch
STORAGE_IMPL=MINIO
DB_TYPE=postgres

# Model provider
# Do not write real API keys into tracked files.
SIGNPOST_LLM_PROVIDER=ecnu

# ECNU OpenAI-compatible endpoint
ECNU_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
ECNU_API_KEY=replace_with_your_ecnu_api_key
ECNU_CHAT_MODEL=ecnu-plus
ECNU_REASONING_MODEL=ecnu-max
ECNU_EMBEDDING_MODEL=ecnu-embedding-small
ECNU_RERANK_MODEL=ecnu-rerank

# Compatibility variables for code paths that still use OpenAI naming.
# For ECNU, point these at the ECNU endpoint.
OPENAI_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
OPENAI_API_KEY=replace_with_your_ecnu_api_key

# DeepResearch batch evaluation
DEEPRESEARCH_TENANT_ID=default
DEEPRESEARCH_DATASETS_ROOT=/home/ruolinsu/signpost/signpost_re/datasets
DEEPRESEARCH_KB_ID_AGRICULTURE=agriculture
DEEPRESEARCH_KB_ID_LEGAL=legal
DEEPRESEARCH_KB_ID_MIX=mix
DEEPRESEARCH_KB_ID_GRAPHRAG_BENCH=graphrag-bench

# Optional OpenAI fallback. Fill these only if you want to switch providers.
REAL_OPENAI_API_BASE=https://api.openai.com/v1
REAL_OPENAI_API_KEY=

# LLM-as-Judge. Use an ECNU model by default unless you explicitly switch.
EVAL_OPENAI_MODEL=ecnu-plus
```

运行命令前加载它：

```bash
set -a
source /home/ruolinsu/signpost/signpost_re/.env
set +a
```

## 5. 配置文件

旧的 `signpost-main` 当前不包含 `conf/service_conf.yaml`、`conf/mapping.json` 或 `conf/llm_factories.json`。重构版应该随项目提供这些文件的模板。

创建 `signpost_re/conf/service_conf.yaml`：

```yaml
database_type: postgres

postgres:
  name: signpost
  user: signpost
  password: signpost
  host: 127.0.0.1
  port: 5432
  max_connections: 32
  stale_timeout: 300

es:
  hosts: http://127.0.0.1:9200
  max_result_window: 10000
  scroll_timeout: 2m
  scroll_page_size: 1000

redis:
  host: 127.0.0.1:6379
  db: 1
  password:

minio:
  host: 127.0.0.1:9000
  user: signpost
  password: signpost123

user_default_llm:
  # ECNU is OpenAI-compatible. During the refactor, the code should treat this
  # as a provider selected by configuration rather than hard-coding OpenAI.
  factory: ECNU
  base_url: https://chat.ecnu.edu.cn/open/api/v1
  api_key: replace_with_your_ecnu_api_key
  default_models:
    chat_model: ecnu-plus
    reasoning_model: ecnu-max
    embedding_model: ecnu-embedding-small
    rerank_model: ecnu-rerank
```

重要：当前 `core.config` 不会展开 YAML 内部的 `${...}`。对于旧代码，请在 YAML 中使用字面值。重构期间应优先从环境变量读取密钥，避免把 API key 提交进代码库。

创建 `signpost_re/conf/llm_factories.json`：

```json
{
  "factory_llm_infos": [
    {
      "name": "ECNU",
      "logo": "",
      "tags": "LLM,Text Embedding,Chat",
      "llm": [
        {
          "llm_name": "ecnu-plus",
          "model_type": "chat",
          "fid": "ECNU",
          "max_tokens": 256000,
          "tags": "LLM,Chat"
        },
        {
          "llm_name": "ecnu-max",
          "model_type": "chat",
          "fid": "ECNU",
          "max_tokens": 1000000,
          "tags": "LLM,Chat,Reasoning"
        },
        {
          "llm_name": "ecnu-embedding-small",
          "model_type": "embedding",
          "fid": "ECNU",
          "max_tokens": 8192,
          "tags": "Text Embedding"
        },
        {
          "llm_name": "ecnu-rerank",
          "model_type": "rerank",
          "fid": "ECNU",
          "max_tokens": 8192,
          "tags": "Rerank"
        }
      ]
    },
    {
      "name": "OpenAI",
      "logo": "",
      "tags": "LLM,Text Embedding,Chat",
      "llm": [
        {
          "llm_name": "replace_with_openai_chat_model",
          "model_type": "chat",
          "fid": "OpenAI",
          "max_tokens": 128000,
          "tags": "LLM,Chat"
        },
        {
          "llm_name": "replace_with_openai_embedding_model",
          "model_type": "embedding",
          "fid": "OpenAI",
          "max_tokens": 8192,
          "tags": "Text Embedding"
        }
      ]
    }
  ]
}
```

创建 `signpost_re/conf/mapping.json`。

重构期间，只复制实际会用到的 ES 字段：

- original chunks
- RAPTOR nodes
- GraphRAG entities
- GraphRAG edges
- Signpost source locations 的可选元数据字段

不要保留与实验无关的产品字段。

## 6. 第一次 Smoke Tests

在 Docker 服务和 Conda 环境准备好之后运行这些检查。

检查 Python import：

```bash
cd /home/ruolinsu/signpost/signpost-main
python - <<'PY'
import networkx
import elasticsearch
import minio
import valkey
import openai
print("imports ok")
PY
```

检查 PostgreSQL：

```bash
python - <<'PY'
import psycopg2
conn = psycopg2.connect(
    dbname="signpost",
    user="signpost",
    password="signpost",
    host="127.0.0.1",
    port=5432,
)
cur = conn.cursor()
cur.execute("select 1")
print(cur.fetchone())
conn.close()
PY
```

如果缺少 `psycopg2`：

```bash
uv pip install psycopg2-binary
```

检查 Elasticsearch：

```bash
python - <<'PY'
from elasticsearch import Elasticsearch
es = Elasticsearch("http://127.0.0.1:9200")
print(es.info()["version"]["number"])
PY
```

检查 MinIO：

```bash
python - <<'PY'
from minio import Minio
client = Minio("127.0.0.1:9000", access_key="signpost", secret_key="signpost123", secure=False)
bucket = "smoke"
if not client.bucket_exists(bucket):
    client.make_bucket(bucket)
print("minio ok")
PY
```

检查 Valkey：

```bash
python - <<'PY'
import valkey
r = valkey.StrictRedis(host="127.0.0.1", port=6379, db=1, decode_responses=True)
r.set("signpost:smoke", "ok")
print(r.get("signpost:smoke"))
PY
```

检查 ECNU/OpenAI-compatible endpoint，不发起真实请求：

```bash
python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
print("client ok:", bool(client.api_key))
PY
```

在模型名称和 endpoint 确认之前，不要运行真实 LLM 请求。

可选的真实 ECNU chat 请求：

```bash
python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
)
resp = client.chat.completions.create(
    model=os.getenv("ECNU_CHAT_MODEL", "ecnu-plus"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请用一句话介绍你自己。"},
    ],
)
print(resp.choices[0].message.content)
PY
```

ECNU 注意事项：

- Base URL：`https://chat.ecnu.edu.cn/open/api/v1`
- 对话模型：`ecnu-plus` 用于通用任务，`ecnu-max` 用于更复杂任务。
- Embedding 模型：`ecnu-embedding-small`，1024 维，8K 上下文。
- Rerank 模型：`ecnu-rerank`，8K 上下文。
- ECNU 建议避免高并发 API 调用。先从低并发开始，确认服务稳定后再增加。
- 支持通过请求字段 `thinking: {"type": "enabled"}` 开启思考模式。重构时把它作为 model-client 的可选功能保留。

## 7. 重构目标：单独运行 Index 阶段

重构后的代码应该暴露一个用于 indexing 的离线命令。这个命令只运行 index pipeline，不包含 API、前端、用户或权限逻辑：

```bash
python -m signpost.benchmark.index \
  --dataset legal \
  --input-dir datasets/legal/corpus \
  --namespace legal \
  --output outputs/index/legal.jsonl
```

这个命令后续应便于插入测量逻辑，但现在不要把最终 benchmark 指标硬编码进设计里。

避免在 index 路径中包含产品层面的事务：

- 用户权限检查
- 前端/API 序列化
- 登录/bootstrap
- 文件夹管理
- 缩略图生成

## 8. 重构目标：单独运行 Retrieval 阶段

重构后的代码应该暴露一个用于 retrieval 和评估输出生成的离线命令。这个命令只运行 retrieval，假设 index 已经存在：

```bash
python -m signpost.benchmark.retrieval \
  --dataset legal \
  --questions datasets/legal/Question.jsonl \
  --namespace legal \
  --method signpost \
  --output outputs/retrieval/signpost/legal.jsonl
```

这个命令后续应便于评估和测速。输出 JSONL 应保留兼容 `eval/` 的字段：

```json
{
  "question": "...",
  "answer": "...",
  "Rationale": "...",
  "prediction": "...",
  "_duration_seconds": 12.34,
  "_trace": {
    "llm_calls": 8,
    "tool_calls": 5
  }
}
```

## 9. 评估

生成 retrieval 输出后：

```bash
cd /home/ruolinsu/signpost/signpost-main
python -m eval.run_all_evaluations \
  --root /home/ruolinsu/signpost/signpost_re/outputs/retrieval \
  --output /home/ruolinsu/signpost/signpost_re/outputs/evaluation \
  --methods signpost \
  --datasets legal
```

低成本 dry run：

```bash
python -m eval.run_all_evaluations \
  --root /home/ruolinsu/signpost/signpost_re/outputs/retrieval \
  --output /home/ruolinsu/signpost/signpost_re/outputs/evaluation \
  --basic-only
```

## 10. 常见问题

Elasticsearch 立刻退出：

- 将 Docker 内存增加到至少 4 GB。
- 保持 `ES_JAVA_OPTS=-Xms2g -Xmx2g`。
- 在 Linux 上，Elasticsearch 可能需要：

```bash
sudo sysctl -w vm.max_map_count=262144
```

`core.config` import 失败：

- 检查 `RAG_PROJECT_BASE`。
- 确保该目录下存在 `conf/service_conf.yaml`。
- 在实现环境变量展开之前，确保 YAML 值是字面值，而不是 `${...}` 占位符。

`ESConnection` 因缺少 `mapping.json` 失败：

- 添加 `conf/mapping.json`。
- 在 `signpost_re` 中，把一个最小 mapping 模板作为项目的一部分保留下来。

Embedding model 尝试下载本地 BAAI 模型：

- 第一次搭建时使用 ECNU embedding API。
- 将 `user_default_llm.default_models.embedding_model` 设置为 `ecnu-embedding-small`。
- 在需要 GPU/本地 embedding 之前，避免安装 `flagembedding`。

ECNU 请求失败：

- 确认 `OPENAI_API_BASE=https://chat.ecnu.edu.cn/open/api/v1`。
- 确认 `OPENAI_API_KEY` 已在 shell 中设置，不要提交到文件中。
- 从串行请求开始。ECNU 文档建议避免并行 API 调用，以获得更好的稳定性。

DeepResearch batch runner 提示缺少 KB ID：

- 设置以下环境变量之一：

```bash
export DEEPRESEARCH_KB_ID_LEGAL=legal
export DEEPRESEARCH_KB_ID_AGRICULTURE=agriculture
export DEEPRESEARCH_KB_ID_MIX=mix
export DEEPRESEARCH_KB_ID_GRAPHRAG_BENCH=graphrag-bench
```

## 11. `signpost_re` 的重构原则

从 `signpost-main` 迁移代码时遵循这些规则：

- 用单一 `namespace` 或 `experiment_id` 替换 `tenant_id` 和 `user_id`。
- 保留 `kb_id` 作为数据集/index 标识符。
- 保留 storage、graph、retrieval、agent 和 eval 逻辑。
- 保留 ECNU 和 OpenAI-compatible 两套模型客户端。ECNU 作为你实验的默认方案；OpenAI-compatible fallback 应保留给协作者使用。
- 移除 auth、权限检查、前端特定对象、Canvas、MCP、conversations 和 API token 逻辑。
- 实验优先使用离线 CLI 命令。
- FastAPI 只作为可选 wrapper 添加，不作为主要 benchmark 路径。

## 12. 退出项目时结束环境

退出实验前，先停掉 Docker 服务，避免 PostgreSQL、Valkey、MinIO 和 Elasticsearch 一直占用端口和内存：

```bash
cd /home/ruolinsu/signpost/signpost_re
docker compose down
```

`docker compose down` 会停止并删除容器，但会保留 named volumes，因此 PostgreSQL 数据、MinIO 数据和 Elasticsearch 索引不会被删除。

如果只是临时暂停服务，也可以使用：

```bash
cd /home/ruolinsu/signpost/signpost_re
docker compose stop
```

退出 Conda 环境：

```bash
conda deactivate
```

不要随意使用下面这类命令，除非明确想清空数据库、对象存储和 ES 索引：

```bash
docker compose down -v
```

## 13. 下一次启动环境

重新进入项目时，先启动 Docker 服务：

```bash
cd /home/ruolinsu/signpost/signpost_re/docker
docker compose up -d
docker compose ps
```

确认四个服务都是 `healthy` 后，进入 Python 环境：

```bash
conda activate signpost-re
```

加载环境变量：

```bash
set -a
source /home/ruolinsu/signpost/signpost_re/.env
set +a
```

然后切到参考代码库运行 index、retrieval 或 evaluation 命令：

```bash
cd /home/ruolinsu/signpost/signpost-main
```

如果需要快速检查服务是否正常：

```bash
curl http://localhost:9200
curl -i http://localhost:9000/minio/health/live
```
