# H200 本地服务迁移与 Signpost 实验执行指南

本文档面向当前服务器条件：

```text
服务器工作目录：/data/srl/
Chat 服务：http://localhost:8000/v1
Chat model：/data/srl/Llama-3.3-70B-FP8
Embedding 服务：http://localhost:8001/v1/embeddings
Embedding model：/data/srl/nemotron-8b
服务器 Python：可以使用 Conda，但不要假设已有完整科研依赖。
网络约束：不跨公网调用大模型 API；正式实验必须使用本地服务。
```

本文件只写 H200 迁移和正式实验流程。Python/服务环境的基础原则以 `docs/environment_setup.zh.md` 为准：使用 Conda 创建 `signpost-re` 环境，editable 安装本项目，外部检索服务优先使用 Elasticsearch。

## 1. 当前状态判断

Signpost 侧已经可以迁移到 H200 做正式实验：

```text
索引构建：F3-F10 已有统一脚本。
在线查询：F15 agent batch 已有统一脚本。
消融实验：full/no_offline/no_online/no_semantic/no_provenance/no_vertical/no_horizontal 已有统一入口。
评估汇总：basic_eval/query_metrics/method_summary/cost_quality 已经串联。
```

还没有完成的是外部 baseline wrapper。这个不影响先把 Signpost 在 H200 上跑通；baseline 后续只要对齐同一套 `documents.jsonl/chunks.jsonl/questions.jsonl` 和统一 prediction schema。

## 2. 先同步代码

### 2.1 rsync 在哪边执行

`rsync` 在本地福建电脑执行，不是在 H200 上执行。方向是：

```text
本地 /home/ruolinsu/signpost/signpost_re/  ->  H200 /data/srl/signpost_re/
```

也就是说，先在本地终端确认能 SSH 到 H200：

```bash
ssh <user>@<h200>
```

能登录后，在本地终端执行：

```bash
rsync -av \
  --exclude outputs \
  --exclude datasets/processed \
  --exclude .pytest_cache \
  --exclude '__pycache__' \
  /home/ruolinsu/signpost/signpost_re/ \
  <user>@<h200>:/data/srl/signpost_re/
```

说明：

```text
--exclude outputs：避免覆盖服务器上已经跑出的实验结果。
--exclude datasets/processed：避免覆盖服务器上已经构建好的索引。
不排除 datasets/raw：小规模 raw 数据可以同步；如果 raw 数据很大，单独用 scp/rsync 同步 raw。
```

如果服务器上已经有代码目录，仍建议覆盖代码文件，但不要覆盖已经跑出的 `outputs/` 和 `datasets/processed/`。正式实验最好记录代码快照：

```bash
cd /data/srl/signpost_re
python -V > outputs_python_version.txt
find signpost scripts docs -type f | sort > outputs_code_file_list.txt
```

### 2.2 rsync 会不会被服务器阻拦

`rsync` 走 SSH。能否使用只取决于：

```text
1. 本地电脑能否 SSH 到 H200；
2. H200 是否安装 rsync；
3. 你的账号是否有 /data/srl/ 写权限。
```

如果 SSH 可以登录但服务器没有 `rsync`，用压缩包 fallback。本地执行：

```bash
cd /home/ruolinsu/signpost
tar \
  --exclude='signpost_re/outputs' \
  --exclude='signpost_re/datasets/processed' \
  --exclude='signpost_re/.pytest_cache' \
  --exclude='*/__pycache__' \
  -czf signpost_re_h200.tar.gz signpost_re

scp signpost_re_h200.tar.gz <user>@<h200>:/data/srl/
```

然后在 H200 上执行：

```bash
cd /data/srl
tar -xzf signpost_re_h200.tar.gz
```

如果校园网禁止从本地直接 SSH 到 H200，就使用学校允许的跳板机、VS Code Remote 的文件上传、SFTP 客户端或服务器管理员提供的传输方式。只要最终目录是 `/data/srl/signpost_re/`，后续命令不变。

如果有 git 仓库，用 commit 固定版本更好；当前项目目录看起来不是 git repo，所以先用 rsync 或 tar 包是可行的。

## 3. 服务器 Python 环境

服务器可以使用 Conda，因此按 `docs/environment_setup.zh.md` 的思路配置。进入 H200 后执行：

```bash
cd /data/srl/signpost_re
```

创建 Conda 环境：

```bash
conda create -n signpost-re python=3.11 -y
conda activate signpost-re
python -m pip install -U pip uv
```

安装项目本体：

```bash
cd /data/srl/signpost_re
python -m pip install -e .
uv pip install pytest pytest-asyncio requests-toolbelt psycopg2-binary
```

当前 `signpost_re` 主体依赖很轻，`pyproject.toml` 里只有 `pyyaml` 是硬依赖；`pytest` 用于迁移 smoke test。H200 上已经有 `requests` 和 `openai` 没问题，但 Signpost 的模型调用走标准库 `urllib`，不依赖 OpenAI SDK。

如果 H200 无法访问 PyPI，可以在本地或能联网的机器下载 wheelhouse，再上传到服务器安装：

本地执行：

```bash
mkdir -p /tmp/signpost_wheels
python -m pip download -d /tmp/signpost_wheels pyyaml pytest pytest-asyncio requests-toolbelt psycopg2-binary
tar -czf signpost_wheels.tar.gz -C /tmp signpost_wheels
scp signpost_wheels.tar.gz <user>@<h200>:/data/srl/
```

H200 执行：

```bash
cd /data/srl
tar -xzf signpost_wheels.tar.gz
conda activate signpost-re
python -m pip install --no-index --find-links /data/srl/signpost_wheels pyyaml pytest pytest-asyncio requests-toolbelt psycopg2-binary
cd /data/srl/signpost_re
python -m pip install -e . --no-deps
```

完成后跑测试：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
python -m pytest
```

## 4. H200 环境变量

在服务器创建 `/data/srl/signpost_re/.env.h200`：

```bash
PYTHONPATH=/data/srl/signpost_re

ECNU_API_BASE=http://localhost:8000/v1
ECNU_API_KEY=EMPTY
ECNU_CHAT_MODEL=/data/srl/Llama-3.3-70B-FP8
ECNU_REASONING_MODEL=/data/srl/Llama-3.3-70B-FP8

# 这里可以写完整 endpoint；代码会识别末尾的 /embeddings，不会重复拼接。
ECNU_EMBEDDING_API_BASE=http://localhost:8001/v1/embeddings
ECNU_EMBEDDING_API_KEY=EMPTY
ECNU_EMBEDDING_MODEL=/data/srl/nemotron-8b
ECNU_RERANK_MODEL=unused-local-rerank

LLM_TIMEOUT=600
LLM_RETRIES=6
RETRY_SLEEP=20
GLEANING_ROUNDS=0
SEMANTIC_EXTRACTOR=llm
EMBEDDING_PROVIDER=ecnu
```

每次运行前加载：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a
```

说明：代码里 provider 名称仍叫 `ecnu`，但这只是沿用现有接口名；实际请求已经由环境变量指向 H200 本地服务。

## 5. 模型服务 smoke test

先不要跑索引。先确认两个本地服务都通。

### 5.1 服务启动参考命令

H200 上当前使用两个独立 tmux session 托管本地模型服务：

```bash
tmux new -s llama
```

在 `llama` session 中启动 chat 服务：

```bash
cd /data/srl
conda activate /data/srl/.conda_envs/vllm

CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/data/data_c/usr/local/cuda-12.1 \
LD_LIBRARY_PATH=/data/data_c/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH \
VLLM_USE_DEEP_GEMM=0 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/Llama-3.3-70B-FP8 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

```bash
tmux new -s embed
```

在 `embed` session 中启动 embedding 服务：

```bash
cd /data/srl
conda activate /data/srl/.conda_envs/vllm

CUDA_VISIBLE_DEVICES=2 \
VLLM_USE_DEEP_GEMM=0 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/nemotron-8b \
  --runner pooling \
  --port 8001 \
  --trust-remote-code
```

如果 session 已存在，用 `tmux attach -t llama` 或 `tmux attach -t embed` 进入对应窗口，在 shell prompt 下重启服务。若端口仍被旧进程占用，先用 `ss -ltnp | grep -E '8000|8001'` 找到监听进程，确认是已崩溃残留或旧服务后再处理。

### 5.2 Smoke test

```bash
conda activate signpost-re
python -m signpost.llm.smoke
```

预期能看到：

```text
api_base: http://localhost:8000/v1
embedding_api_base: http://localhost:8001/v1/embeddings
chat_model: /data/srl/Llama-3.3-70B-FP8
embedding_model: /data/srl/nemotron-8b
rerank_model: unused-local-rerank
```

测试 chat：

```bash
conda activate signpost-re
python -m signpost.llm.smoke --chat
```

测试 embedding：

```bash
conda activate signpost-re
python -m signpost.llm.smoke --embedding
```

如果 chat 成功、embedding 失败，通常是 8001 服务路径问题。可以临时改：

```bash
export ECNU_EMBEDDING_API_BASE=http://localhost:8001/v1
python -m signpost.llm.smoke --embedding
```

如果两种写法只有一种成功，就把成功的写回 `.env.h200`。

## 6. Elasticsearch 问题

Signpost 正式索引和检索默认需要 ES：

```text
F5_chunk_index
F10_graph_es_sync
F13/F15 with USE_ES=1
```

如果 H200 已经有 ES，先确认：

```bash
curl http://localhost:9200
```

如果没有 ES，优先使用**用户态 Elasticsearch tar 包**。这不需要 Docker，也通常不需要管理员权限，只要 `/data/srl/` 可写即可。

### 6.1 无 Docker 用户态 Elasticsearch

在 H200 上如果能访问 Elastic 下载站，直接执行：

```bash
cd /data/srl
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.12.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.12.1-linux-x86_64.tar.gz
mv elasticsearch-8.12.1 elasticsearch-8.12.1-signpost
```

如果 H200 不能联网下载，就在本地电脑下载：

```bash
cd /tmp
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.12.1-linux-x86_64.tar.gz
```

然后用当前可用的上传方式放到 H200 的 `/data/srl/`，再在 H200 解压：

```bash
cd /data/srl
tar -xzf elasticsearch-8.12.1-linux-x86_64.tar.gz
mv elasticsearch-8.12.1 elasticsearch-8.12.1-signpost
```

写入最小配置：

```bash
cat > /data/srl/elasticsearch-8.12.1-signpost/config/elasticsearch.yml <<'EOF'
cluster.name: signpost-local
node.name: signpost-h200
path.data: /data/srl/esdata
path.logs: /data/srl/eslogs
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
EOF

mkdir -p /data/srl/esdata /data/srl/eslogs /data/srl/logs
```

启动 ES。建议用 `nohup` 放后台：

```bash
cd /data/srl/elasticsearch-8.12.1-signpost
ES_JAVA_OPTS="-Xms2g -Xmx2g" nohup ./bin/elasticsearch > /data/srl/logs/elasticsearch.out 2>&1 &
```

等待 20-60 秒后检查：

```bash
curl http://127.0.0.1:9200
curl http://127.0.0.1:9200/_cat/health?v
```

如果 `curl` 返回 JSON，且里面有 `cluster_name: signpost-local`，说明 ES 可用。Signpost 默认读取 `conf/service_conf.yaml` 中的：

```yaml
elasticsearch:
  hosts: http://127.0.0.1:9200
```

因此不需要额外改代码。

停止 ES：

```bash
pkill -f elasticsearch-8.12.1-signpost
```

如果启动失败，先看日志：

```bash
tail -n 100 /data/srl/logs/elasticsearch.out
tail -n 100 /data/srl/eslogs/signpost-local.log
```

常见问题：

```text
端口占用：说明已有 ES 或其他服务占用 9200，先 curl http://127.0.0.1:9200 看是否可用。
内存不足：把 ES_JAVA_OPTS 改成 -Xms1g -Xmx1g 先做 smoke；正式实验建议 2g 或更高。
文件权限：确认 /data/srl/esdata 和 /data/srl/eslogs 归当前用户可写。
不能下载：本地下载 tar.gz 后上传到 /data/srl/。
```

### 6.2 Docker 方案，仅在 Docker 可用时使用

如果服务器后来提供 Docker，再判断支持哪一种 Compose 命令：

```bash
cd /data/srl/signpost_re/docker
docker compose version
docker-compose version
```

只需要其中一个能运行。

如果 `docker compose version` 成功，使用 Compose v2：

```bash
cd /data/srl/signpost_re/docker
docker compose up -d elasticsearch
curl http://localhost:9200
```

如果 `docker-compose version` 成功，使用 Compose v1：

```bash
cd /data/srl/signpost_re/docker
docker-compose up -d elasticsearch
curl http://localhost:9200
```

如果两条 `version` 命令都失败，说明当前账号没有可用的 Docker Compose。不要继续试 `docker compose up -d`。这时回到 6.1 的用户态 tar 包方案。

如果 Docker 可用但 Compose 不可用，也可以用裸 `docker run` 启动 ES：

```bash
docker run -d \
  --name signpost-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.1

curl http://localhost:9200
```

如果镜像无法拉取，说明 H200 不能访问 Docker Hub/Elastic registry，需要管理员提前导入镜像，或直接使用 6.1 的 tar 包方案。

### 6.3 临时跳过 ES 的 smoke

正式 full 不建议跳过 ES。可以用 `USE_ES=0` 做本地文件检索 smoke，但技术说明正式数值建议使用 ES，因为 F5/F10 的时间和索引空间都是 Signpost 成本的一部分。

如果 ES 实在暂时启动不了，先继续：

```text
USE_ES=0
```

只验证模型、F6 抽取、agent 回答和输出 schema。ES 修好后再跑正式 full。

## 7. 先跑 legal\_test 闭环

`legal_test` 是迁移后第一步，只验证环境和闭环，不进技术说明表格。

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

scripts/run_signpost_dataset_pipeline.sh legal_test legal_test
```

完成后核对：

```bash
test -s datasets/processed/legal_test/graph.unified.json
test -s datasets/processed/legal_test/semantic_llm.extractions.jsonl
test -s outputs/legal_test/metrics/index_metrics.json
```

再跑 Signpost full 和消融：

```bash
LIMIT=3 USE_ES=1 USE_LLM=1 scripts/run_signpost_method.sh legal_test full legal_test
LIMIT=3 USE_ES=1 USE_LLM=1 scripts/run_signpost_ablation_suite.sh legal_test legal_test
```

如果 ES 还没配置好，用下面命令只做 agent 链路 smoke：

```bash
LIMIT=3 USE_ES=0 USE_LLM=1 scripts/run_signpost_ablation_suite.sh legal_test legal_test
```

注意：`USE_ES=0` 的结果只用于检查流程，不作为正式实验。

## 8. 正式 Signpost 运行顺序

先跑 Agriculture，再跑 Legal。不要一上来跑 Legal-full。

### 8.1 两条命令分别覆盖什么

每个数据集的 Signpost 实验由两类脚本组成：

```text
scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>
```

覆盖数据集级离线阶段：

```text
F3  data prepare
F3.5 parse documents
F4  chunk/tree
F5  chunk Elasticsearch index
F6  LLM semantic graph extraction
F7  structure/RAPTOR graph
F8  sequence graph
F9  unified graph
F10 graph Elasticsearch sync
index_metrics
```

生成：

```text
datasets/processed/<dataset>/*
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/metrics/index_metrics.json
```

```text
scripts/run_signpost_ablation_suite.sh <dataset> <namespace>
```

覆盖 Signpost 方法级在线阶段和全部 Signpost 消融：

```text
signpost.full
signpost.no_offline
signpost.no_online
signpost.no_semantic_cues
signpost.no_provenance_cues
signpost.no_vertical_cues
signpost.no_horizontal_cues
```

每个变体都会跑：

```text
F15 agent batch
F16 basic evaluation
query_metrics
method_summary
cost_quality
```

因此，对 Signpost 自身来说，这两条命令就是一个数据集上的完整 Signpost 实验闭环。`scripts/run_signpost_method.sh <dataset> full <namespace>` 只在需要单独重跑 full 或 smoke 时使用；正式消融表直接跑 `run_signpost_ablation_suite.sh`。

Baseline 对比不能“复用消融结果”作为 baseline 结果。Baseline 需要后续 wrapper 生成自己的：

```text
outputs/<dataset>/predictions/<baseline>.jsonl
outputs/<dataset>/metrics/<baseline>.query_metrics.json
```

但是 baseline 可以复用同一份输入数据：

```text
datasets/processed/<dataset>/documents.jsonl
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl
```

技术说明主对比表使用 `signpost.full` 对比外部 baselines；Signpost 消融表使用 `signpost.full` 对比 `signpost.no_*`。

### 8.2 用 tmux 跑正式实验

正式实验建议全部放在 `tmux` 里，避免 SSH 断开导致长任务中断。

创建会话：

```bash
tmux new -s signpost-agriculture
```

进入会话后加载环境并运行：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

scripts/run_signpost_dataset_pipeline.sh agriculture agriculture
scripts/run_signpost_ablation_suite.sh agriculture agriculture
```

从 tmux detach：

```text
Ctrl-b 然后按 d
```

重新进入：

```bash
tmux attach -t signpost-agriculture
```

查看已有会话：

```bash
tmux ls
```

Legal 建议单独开一个会话：

```bash
tmux new -s signpost-legal
```

然后运行：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

scripts/run_signpost_dataset_pipeline.sh legal legal
scripts/run_signpost_ablation_suite.sh legal legal
```

不要在同一个 GPU 模型服务上并发跑多个 F6 语义抽取任务，除非你已经确认本地 Llama 服务可以稳定承受并发。正式计时也应避免并发干扰。

### 8.3 推荐正式顺序

先跑 Agriculture：

```bash
scripts/run_signpost_dataset_pipeline.sh agriculture agriculture
scripts/run_signpost_ablation_suite.sh agriculture agriculture
```

再跑 Legal：

```bash
scripts/run_signpost_dataset_pipeline.sh legal legal
scripts/run_signpost_ablation_suite.sh legal legal
```

如果你的正式数据集名称是 `Agriculture-full` / `Legal-full`，就把命令中的 dataset 和 namespace 同步替换：

```bash
scripts/run_signpost_dataset_pipeline.sh Agriculture-full Agriculture-full
scripts/run_signpost_ablation_suite.sh Agriculture-full Agriculture-full
```

## 9. F6 断点续跑

F6 语义抽取有 cache：

```text
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/semantic_llm.progress.jsonl
```

中途失败后，直接重跑同一条 `scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>`。已写入 `semantic_llm.extractions.jsonl` 的 chunk 会跳过。

不要删除 cache，除非你明确要重跑全部语义抽取。

## 10. 输出文件怎么看

每个数据集最终应该有：

```text
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/metrics/index_metrics.json
outputs/<dataset>/predictions/signpost.full.jsonl
outputs/<dataset>/predictions/signpost.no_offline.jsonl
outputs/<dataset>/predictions/signpost.no_online.jsonl
outputs/<dataset>/predictions/signpost.no_semantic_cues.jsonl
outputs/<dataset>/predictions/signpost.no_provenance_cues.jsonl
outputs/<dataset>/predictions/signpost.no_vertical_cues.jsonl
outputs/<dataset>/predictions/signpost.no_horizontal_cues.jsonl
outputs/<dataset>/metrics/signpost.<variant>.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

快速核对：

```bash
wc -l outputs/<dataset>/predictions/signpost.full.jsonl
python -m json.tool outputs/<dataset>/metrics/index_metrics.json >/dev/null
python -m json.tool outputs/<dataset>/metrics/cost_quality.json >/dev/null
```

## 11. baseline 后续如何接

baseline 还没有完成，但不需要阻塞 Signpost 迁移。后续每个 baseline wrapper 只需统一：

输入：

```text
datasets/processed/<dataset>/documents.jsonl
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl
```

输出：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
```

然后复用现有评估：

```bash
python -m signpost.evaluation.evaluate_basic \
  --input outputs/<dataset>/predictions/<method>.jsonl \
  --output outputs/<dataset>/metrics/<method>.basic_eval.json \
  --normalize

python -m signpost.benchmark.query_metrics \
  --input outputs/<dataset>/predictions/<method>.jsonl \
  --output outputs/<dataset>/metrics/<method>.query_metrics.json \
  --normalize \
  --top-k 5 10
```

baseline 必须也走 H200 本地 chat/embedding 服务，不能混用 ECNU 或外部 API，否则时间、延迟、失败率和 token 成本不可比。

## 12. 当前最推荐的操作

现在可以迁移，但顺序必须保守：

```text
1. 在本地执行 rsync 或 tar/scp，同步代码到 /data/srl/signpost_re。
2. 在 H200 创建 conda 环境 signpost-re，并按 environment_setup.zh.md 思路安装项目。
3. 在 H200 创建并加载 .env.h200。
4. 跑 python -m pytest。
5. 跑 python -m signpost.llm.smoke --chat。
6. 跑 python -m signpost.llm.smoke --embedding。
7. 确认 ES。
8. 跑 legal_test dataset pipeline。
9. 跑 legal_test ablation suite，LIMIT=3。
10. 跑 agriculture full。
11. 跑 legal full。
12. 再开始 baseline wrapper。
```

只要第 5-6 步通过，ECNU 的问题就可以完全绕开。
