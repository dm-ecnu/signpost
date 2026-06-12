# H200 实验迁移与执行流程

本文档说明如何把 `signpost_re` 从本地开发环境迁移到组内 H200 服务器，并完成最终技术说明实验。核心原则是：

> 本地负责开发、调试和封装；H200 负责可复现实验运行、模型推理和指标产出。

不要在 H200 上临场写复杂代码。服务器不适合临场编辑复杂代码，因此所有实验逻辑都应在本地封装成脚本、配置和命令模板，再同步到服务器执行。

## 1. 总体策略

### 1.1 本地做什么

本地继续作为开发机：

- 修改 `signpost_re` 源码；
- 补端到端运行脚本；
- 补 benchmark 汇总脚本；
- 写配置模板；
- 在 `samples/mini`、`legal_test` 或小 limit 上跑 smoke test；
- 提交 git commit；
- 生成服务器运行命令。

### 1.2 H200 做什么

H200 只做运行机：

- 部署本地模型服务；
- 安装 Python 环境和依赖；
- 拉取固定 git commit；
- 跑 smoke test；
- 跑正式 index 构建；
- 跑 batch retrieval / agent；
- 跑 evaluation 和 cost summary；
- 保存 logs、predictions、metrics、artifacts。

### 1.3 不要做什么

不要在 H200 上：

- 临时手改核心 Python 文件；
- 在不同 commit 上混跑同一张表；
- 用外部 API 和本地模型混报主结果；
- 跑没有 `stage_timing.jsonl` 或 query log 的长任务；
- 直接在 full Legal 上试错。

## 2. 服务器目录建议

建议在 H200 上建立固定目录：

```text
/data/<user>/signpost/
  signpost_re/                 # git clone 的代码
  models/
    Llama-3.3-70B-Instruct/
    llama-embed-nemotron-8b/
  datasets/
    raw/
    processed/
  runs/
    smoke/
    agriculture/
    legal/
    baselines/
  services/
    vllm/
    elasticsearch/
```

如果服务器已有统一模型目录，可以只在 `.env.h200` 中指向组内模型路径。

## 3. 代码同步方式

推荐使用 git，而不是手动复制目录。

本地：

```bash
cd /home/ruolinsu/signpost/signpost_re
git status
git add .
git commit -m "prepare h200 experiment workflow"
git push
```

H200：

```bash
cd /data/<user>/signpost
git clone <repo-url> signpost_re
cd signpost_re
git checkout <commit-sha>
```

正式实验记录必须保存 commit：

```bash
git rev-parse HEAD > outputs/RUN_COMMIT.txt
git diff --stat > outputs/RUN_DIFF_STAT.txt
```

如果暂时没有远端 git 仓库，用 `rsync` 可以作为过渡：

```bash
rsync -av --exclude outputs --exclude datasets --exclude .pytest_cache \
  /home/ruolinsu/signpost/signpost_re/ \
  <h200>:/data/<user>/signpost/signpost_re/
```

但正式实验建议尽快切到 git commit。

## 4. H200 环境搭建

### 4.1 Conda 环境

```bash
conda create -n signpost-re python=3.11 -y
conda activate signpost-re
python -m pip install -U pip
cd /data/<user>/signpost/signpost_re
python -m pip install -e ".[test]"
```

当前项目 `pyproject.toml` 依赖很轻，主要依赖是 `pyyaml` 和 `pytest`。模型服务、ES、vLLM 等应作为服务器环境单独安装。

### 4.2 Elasticsearch

Signpost 正式检索建议使用 ES。若服务器允许 Docker：

```bash
cd docker
docker compose up -d elasticsearch
curl http://localhost:9200
```

如果服务器不允许 Docker，使用组内已有 ES 服务，并在配置中写入 host/port。

### 4.3 本地模型服务

主实验应使用本地模型，避免外部 API 网络延迟、排队、中断和版本漂移。

推荐服务形式：

- generation / extraction / agent reasoning: OpenAI-compatible `vLLM` endpoint serving `meta-llama/Llama-3.3-70B-Instruct`
- embedding: OpenAI-compatible endpoint serving `nvidia/llama-embed-nemotron-8b`

示例，仅作模板，具体 tensor parallel 和端口按服务器调整：

```bash
vllm serve /data/<user>/signpost/models/Llama-3.3-70B-Instruct \
  --served-model-name llama-3.3-70b-instruct \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000
```

embedding 模型如果不能直接用 vLLM serving，需要单独提供 OpenAI-compatible embedding 服务，或者补一个很薄的本地 HTTP wrapper。主实验里 chat 和 embedding 都应走本地 endpoint。

## 5. H200 环境变量

在服务器上创建不入库的 `.env.h200`：

```bash
PYTHONPATH=/data/<user>/signpost/signpost_re
RAG_PROJECT_BASE=/data/<user>/signpost/signpost_re

SIGNPOST_LLM_PROVIDER=local

OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
LOCAL_CHAT_MODEL=llama-3.3-70b-instruct

LOCAL_EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
LOCAL_EMBEDDING_API_KEY=EMPTY
LOCAL_EMBEDDING_MODEL=llama-embed-nemotron-8b

EVAL_OPENAI_MODEL=llama-3.3-70b-instruct
```

运行前：

```bash
set -a
source .env.h200
set +a
```

如果当前代码只接受 `ecnu` / `openai` provider 名称，可以先把本地服务伪装成 OpenAI-compatible endpoint，再把 provider 名称设为现有代码支持的 provider。不要为 H200 临时分叉一套模型调用逻辑。

## 6. 实验推进顺序

### Phase 0: 本地补齐端到端脚本

在本地先补两个脚本或 Makefile target：

```text
scripts/run_dataset_pipeline.sh
scripts/run_method_batch.sh
```

目标是让 H200 上只需要改 dataset/method/model/env 参数，不需要手敲十几条 Python 命令。

### Phase 1: H200 smoke test

先跑最小数据，不要直接跑 Legal-full。

建议顺序：

1. `pytest`
2. LLM chat smoke
3. embedding smoke
4. ES health smoke
5. `legal_test` F3-F9
6. `legal_test` F13 one-query retrieval
7. `legal_test` F15 batch `--limit 3`
8. F16 evaluation
9. benchmark summary

每一步都要有输出文件和日志。

### Phase 2: Agriculture-full

Agriculture 是第一个正式完整数据集：

- 规模相对可控；
- 能验证完整 F3-F16；
- 能暴露 H200 环境和模型服务问题；
- 失败代价低于 Legal-full。

Agriculture 跑通后，再开始 Legal。

### Phase 3: Legal-full

Legal 的主要瓶颈是 F6 semantic extraction。策略：

- 必须使用 cache/progress；
- 必须可断点续跑；
- 必须包 `time_stage.py`；
- 不要同时跑太多并发导致模型服务不稳定；
- 先用 `--limit` 或 Legal 5% slices 验证抽取格式，再跑 full。

如果 Legal-full 能完成，则主实验使用 Legal-full，不再单独做 Legal-Core 主表。

如果 Legal-full 不能完成，则构造结构保真的 Legal-Core：按完整文档或主题抽样，不随机抽 chunks。

### Phase 4: Baselines

baseline 不要一开始全铺开。推荐顺序：

1. Vanilla LLM
2. Vanilla RAG
3. LinearRAG
4. LightRAG 或 LeanRAG
5. A-RAG
6. Youtu-GraphRAG / fallback
7. Signpost ablations

每接入一个 baseline，先在 `legal_test` 或 Agriculture `--limit 5` 上验证输入输出格式，再上 full。

### Phase 5: 汇总指标

最终每个 dataset/method 至少需要：

```text
outputs/<dataset>/
  logs/
    stage_timing.jsonl
    <method>.query.jsonl
  predictions/
    <method>.jsonl
  metrics/
    <method>.query_metrics.json
    index_metrics.json
    method_summaries.json
    cost_quality.json
```

没有日志的实验不应进入技术说明主表。

## 7. 本地开发与服务器运行的闭环

推荐日常循环：

```text
本地改代码
  -> 本地 pytest + mini smoke
  -> git commit
  -> H200 git pull
  -> H200 跑 smoke / limit
  -> rsync 拉回 logs 和失败样例
  -> 本地分析和修复
```

拉回结果：

```bash
rsync -av <h200>:/data/<user>/signpost/signpost_re/outputs/ \
  /home/ruolinsu/signpost/signpost_re/outputs_h200/
```

服务器上建议使用 `tmux`：

```bash
tmux new -s signpost
```

长任务用日志重定向，不要只依赖终端输出。

## 8. 技术说明实验最小可行闭环

在所有 baseline 都没准备好之前，先完成 Signpost 自身闭环：

1. Agriculture-full Signpost F3-F16。
2. Legal-full 或 Legal-Core Signpost F3-F16。
3. Signpost ablations。
4. Vanilla RAG。
5. LinearRAG / LightRAG。
6. A-RAG。
7. 成本摊销图。

这个闭环完成后，技术说明至少可以支撑：

- Signpost 是否有效；
- Signpost 哪些组件有效；
- Signpost 开销有多大；
- 什么时候摊销后值得；
- 和基本 RAG / graph RAG / agentic RAG 的关系。

## 9. 当前需要补的工程缺口

迁移前建议在本地补齐：

1. 一个统一 dataset pipeline 脚本，串起 F3-F10。
2. 一个统一 method batch 脚本，串起 F13-F16。
3. 一个 H200 `.env` 模板。
4. 一个模型 provider 配置，支持本地 OpenAI-compatible chat/embedding。
5. 一个 baseline registry，记录每个 baseline 的命令、输入、输出、是否计入 offline cost。
6. 一个 run manifest，保存 dataset、method、commit、model、hardware、start/end time。

其中 1、2、3、6 优先级最高。没有它们，服务器实验很容易变成不可复现的手工操作。

## 10. 决策建议

现在不要直接搬到 H200 上边跑边改。正确顺序是：

1. 本地补实验编排脚本。
2. 本地用 mini 数据验证端到端。
3. H200 配模型和环境。
4. H200 跑 `legal_test` smoke。
5. H200 跑 Agriculture-full。
6. H200 跑 Legal-full F6 semantic extraction。
7. 再接 baseline 和 ablation。

这样服务器即使只执行固定脚本，也只需要执行固定脚本和同步日志。复杂代码修改仍然留在本地完成。
