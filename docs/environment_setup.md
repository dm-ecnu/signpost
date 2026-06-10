# Signpost Research Environment Setup

This guide describes how to set up a clean research environment for the
refactored `signpost_re` project. The goal is not to run a multi-user product
backend. The goal is to support two clean research workflows:

1. Index stage: document preprocessing, chunking, embedding,
   GraphRAG extraction, RAPTOR/tree summaries, graph persistence, ES sync.
2. Retrieval stage: hybrid retrieval, graph retrieval, Signpost offline/online
   navigation, Agent retrieval, and evaluation output generation.

Exact benchmark metrics and timing fields are intentionally not fixed here.
The refactored code should make the index and retrieval stages easy to run
independently, so measurement logic can be added or changed later.

The current `signpost-main` code still contains product-oriented remnants such
as users, tenants, permissions, conversations, Canvas, API tokens, and frontend
support. In `signpost_re`, those should be replaced with a single experiment
namespace, for example `default` or a dataset name.

## 0. What Will Be Installed

Use Conda for Python and Docker for external services.

Required local services:

- Elasticsearch 8.x: BM25, dense vector retrieval, chunk/entity/edge/RAPTOR
  documents.
- MinIO: original documents and persisted NetworkX graph files.
- Valkey/Redis: queues, locks, caches, optional SSE/event streams.
- MySQL or PostgreSQL: minimal metadata and LLM cache. For the research version,
  PostgreSQL is easier to keep self-contained and has better JSON support.

Model provider:

- Default for this project: ECNU OpenAI-compatible API.
- Fallback/alternative: standard OpenAI-compatible API. Keep this path because
  collaborators may still use it later.

Recommended for `signpost_re`:

- Python: 3.11
- Package installer: `uv`
- Database: PostgreSQL 16
- Elasticsearch: 8.12.x
- MinIO: latest stable Docker image
- Valkey: 7.x

## 1. Directory Layout

From the workspace root:

```bash
cd /home/ruolinsu/signpost
```

Recommended working layout:

```text
signpost/
  signpost-main/       # old/reference codebase
  signpost_re/         # refactored research codebase
    docs/
    docker/
    conf/
    datasets/
    logs/
    outputs/
```

Create runtime directories:

```bash
mkdir -p signpost_re/{conf,datasets,logs,outputs,docker}
```

## 2. Start Docker Services

Create `signpost_re/docker/docker-compose.yml`:

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

Start services:

```bash
cd /home/ruolinsu/signpost/signpost_re/docker
docker compose up -d
```

Check service health:

```bash
docker compose ps
curl http://localhost:9200
curl http://localhost:9000/minio/health/live
```

MinIO console:

```text
http://localhost:9001
user: signpost
password: signpost123
```

## 3. Create Conda Environment

```bash
conda create -n signpost-re python=3.11 -y
conda activate signpost-re
python -m pip install -U pip uv
```

For now, install from the reference project until `signpost_re` has its own
`pyproject.toml`:

```bash
cd /home/ruolinsu/signpost/signpost-main
uv pip install -e ".[dev-enhanced]"
uv pip install pytest pytest-asyncio requests-toolbelt psycopg2-binary
```

Optional GPU/local embedding dependencies:

```bash
uv pip install "torch>=2.5,<3" flagembedding==1.2.10
```

For a first CPU-only environment, skip the optional GPU line and use an
OpenAI-compatible embedding API instead. ECNU provides OpenAI-compatible chat,
embedding, and rerank models, so it can be used without local GPU dependencies.

## 4. Environment Variables

Create `signpost_re/.env`:

```bash
# Python/runtime
PYTHONPATH=/home/ruolinsu/signpost/signpost-main
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

Load it before running commands:

```bash
set -a
source /home/ruolinsu/signpost/signpost_re/.env
set +a
```

## 5. Configuration Files

The old `signpost-main` currently does not include `conf/service_conf.yaml`,
`conf/mapping.json`, or `conf/llm_factories.json`. The refactored version should
ship templates for these files.

Create `signpost_re/conf/service_conf.yaml`:

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

Important: the current `core.config` does not expand `${...}` inside YAML. For
the old code, use literal values in YAML. During refactoring, prefer reading
secrets from environment variables so API keys are not committed.

Create `signpost_re/conf/llm_factories.json`:

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

Create `signpost_re/conf/mapping.json`.

During refactoring, copy only the ES fields that are actually used by:

- original chunks
- RAPTOR nodes
- GraphRAG entities
- GraphRAG edges
- optional metadata fields for Signpost source locations

Do not keep product fields unrelated to experiments.

## 6. First Smoke Tests

Run these after the Docker services and Conda environment are ready.

Check Python imports:

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

Check PostgreSQL:

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

If `psycopg2` is missing:

```bash
uv pip install psycopg2-binary
```

Check Elasticsearch:

```bash
python - <<'PY'
from elasticsearch import Elasticsearch
es = Elasticsearch("http://127.0.0.1:9200")
print(es.info()["version"]["number"])
PY
```

Check MinIO:

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

Check Valkey:

```bash
python - <<'PY'
import valkey
r = valkey.StrictRedis(host="127.0.0.1", port=6379, db=1, decode_responses=True)
r.set("signpost:smoke", "ok")
print(r.get("signpost:smoke"))
PY
```

Check ECNU/OpenAI-compatible endpoint without making a real request:

```bash
python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
print("client ok:", bool(client.api_key))
PY
```

Do not run a real LLM request until the model names and endpoint are confirmed.

Optional real ECNU chat request:

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

ECNU notes:

- Base URL: `https://chat.ecnu.edu.cn/open/api/v1`
- Chat models: `ecnu-plus` for general use, `ecnu-max` for more complex tasks.
- Embedding model: `ecnu-embedding-small`, 1024 dimensions, 8K context.
- Rerank model: `ecnu-rerank`, 8K context.
- ECNU recommends avoiding high parallel API calls. Start with low concurrency
  and increase only after confirming service stability.
- Thinking mode is supported through the request field
  `thinking: {"type": "enabled"}`. Keep this as an optional model-client
  feature in the refactor.

## 7. Refactor Target: Separate Index Stage

The refactored code should expose one offline command for indexing. This command
should run the index pipeline only, without API, frontend, user, or permission
logic:

```bash
python -m signpost.benchmark.index \
  --dataset legal \
  --input-dir datasets/legal/corpus \
  --namespace legal \
  --output outputs/index/legal.jsonl
```

The command should be easy to instrument later. Do not bake final benchmark
metrics into the design yet.

Avoid including product concerns in the index path:

- user permission checks
- frontend/API serialization
- login/bootstrap
- file folder management
- thumbnail generation

## 8. Refactor Target: Separate Retrieval Stage

The refactored code should expose one offline command for retrieval and
evaluation output generation. This command should run retrieval only, assuming
the index already exists:

```bash
python -m signpost.benchmark.retrieval \
  --dataset legal \
  --questions datasets/legal/Question.jsonl \
  --namespace legal \
  --method signpost \
  --output outputs/retrieval/signpost/legal.jsonl
```

The command should be easy to evaluate and time later. The output JSONL should
keep fields compatible with `eval/`:

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

## 9. Evaluation

After retrieval outputs are generated:

```bash
cd /home/ruolinsu/signpost/signpost-main
python -m eval.run_all_evaluations \
  --root /home/ruolinsu/signpost/signpost_re/outputs/retrieval \
  --output /home/ruolinsu/signpost/signpost_re/outputs/evaluation \
  --methods signpost \
  --datasets legal
```

For a cheap dry run:

```bash
python -m eval.run_all_evaluations \
  --root /home/ruolinsu/signpost/signpost_re/outputs/retrieval \
  --output /home/ruolinsu/signpost/signpost_re/outputs/evaluation \
  --basic-only
```

## 10. Common Problems

Elasticsearch exits immediately:

- Increase Docker memory to at least 4 GB.
- Keep `ES_JAVA_OPTS=-Xms2g -Xmx2g`.
- On Linux, Elasticsearch may require:

```bash
sudo sysctl -w vm.max_map_count=262144
```

`core.config` fails on import:

- Check `RAG_PROJECT_BASE`.
- Ensure `conf/service_conf.yaml` exists under that directory.
- Ensure YAML values are literal values, not `${...}` placeholders, until
  environment-variable expansion is implemented.

`ESConnection` fails with missing `mapping.json`:

- Add `conf/mapping.json`.
- In `signpost_re`, keep a minimal mapping template as part of the project.

Embedding model tries to download local BAAI model:

- Use the ECNU embedding API for the first setup.
- Set `user_default_llm.default_models.embedding_model` to
  `ecnu-embedding-small`.
- Avoid installing `flagembedding` until GPU/local embedding is needed.

ECNU request fails:

- Confirm `OPENAI_API_BASE=https://chat.ecnu.edu.cn/open/api/v1`.
- Confirm `OPENAI_API_KEY` is set in the shell, not committed into a file.
- Start with serial requests. ECNU documentation recommends avoiding parallel
  API calls for better stability.

DeepResearch batch runner says KB ID is missing:

- Set one of:

```bash
export DEEPRESEARCH_KB_ID_LEGAL=legal
export DEEPRESEARCH_KB_ID_AGRICULTURE=agriculture
export DEEPRESEARCH_KB_ID_MIX=mix
export DEEPRESEARCH_KB_ID_GRAPHRAG_BENCH=graphrag-bench
```

## 11. Refactoring Principle for `signpost_re`

Use these rules while moving code from `signpost-main`:

- Replace `tenant_id` and `user_id` with a single `namespace` or
  `experiment_id`.
- Keep `kb_id` as the dataset/index identifier.
- Keep storage, graph, retrieval, agent, and eval logic.
- Keep both ECNU and OpenAI-compatible model clients. ECNU should be the default
  for your experiments; OpenAI-compatible fallback should remain available for
  collaborators.
- Remove auth, permission checks, frontend-specific objects, Canvas, MCP,
  conversations, and API token logic.
- Prefer offline CLI commands for experiments.
- Add FastAPI only as an optional wrapper, not as the primary benchmark path.
