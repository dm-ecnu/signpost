# Signpost 正式实验管线说明

本文档说明本仓库已经整理好的 Signpost 侧闭环实验入口。外部 baseline 仍按 `docs/v8_exact_experiment_plan.zh.md` 的名单单独接入；本文只覆盖 Signpost 自身，包括完整方法、消融、图构建成本、在线成本和保留记录的 cost-quality 产物。

## 1. 目标和边界

这一步解决的是 Signpost 自身实验输出混乱、功能点没有连贯串起来的问题。完成后，正式实验中 Signpost 的所有产物固定落在以下目录：

```text
datasets/processed/<dataset>/
outputs/<dataset>/logs/
outputs/<dataset>/predictions/
outputs/<dataset>/metrics/
```

其中 `datasets/processed/<dataset>/` 放共享预处理和 Signpost 图索引，`outputs/<dataset>/` 放运行日志、预测结果和可直接进入论文表格的统计文件。

外部 baseline 不写入 Signpost 的内部索引文件，但为了公平，建议都读取同一份原始文档和问题文件：

```text
datasets/processed/<dataset>/documents.jsonl
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl
```

## 2. 已整理的入口

新增三个脚本：

```text
scripts/run_signpost_dataset_pipeline.sh
scripts/run_signpost_method.sh
scripts/run_signpost_ablation_suite.sh
```

三个脚本均已设置为可执行。

### 2.1 数据集级索引构建

```bash
scripts/run_signpost_dataset_pipeline.sh <dataset> [namespace]
```

用途：串联 F3-F10，生成 Signpost 查询所需的全部离线索引，并记录离线时间、空间和 token 代理成本。

默认配置：

```bash
SEMANTIC_EXTRACTOR=llm
GLEANING_ROUNDS=0
EMBEDDING_PROVIDER=ecnu
MAX_TOKENS=512
OVERLAP_TOKENS=64
SUMMARIZER=deterministic
LLM_TIMEOUT=300
LLM_RETRIES=3
RETRY_SLEEP=5
```

输入：

```text
data.prepare 支持的 raw source
```

核心输出：

```text
datasets/processed/<dataset>/raw_corpus.jsonl
datasets/processed/<dataset>/questions.jsonl
datasets/processed/<dataset>/documents.jsonl
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/document_trees.jsonl
datasets/processed/<dataset>/graph.semantic.llm.json
datasets/processed/<dataset>/semantic_llm.progress.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/graph.structure.json
datasets/processed/<dataset>/graph.sequence.json
datasets/processed/<dataset>/graph.unified.json
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/metrics/index_metrics.json
```

这些产物对应论文中的图构建成本、图规模、语义抽取调用数、离线空间开销和 cost-quality 辅助分析。F6 语义抽取的时间、调用次数和 token 必须记录，但按当前 v10 口径作为共享语义标注阶段，不默认计入 Signpost 方法离线成本。

### 2.2 Signpost full 方法

```bash
scripts/run_signpost_method.sh <dataset> full [namespace]
```

用途：在已经构建好的索引上运行 Signpost 完整方法，串联 F15-F16 和查询成本汇总。

默认配置：

```bash
EMBEDDING_PROVIDER=ecnu
USE_ES=1
USE_LLM=1
OFFLINE_STAGES="F7_structure_graph F8_sequence_graph F9_unified_graph F10_graph_es_sync"
```

可用 `LIMIT=<n>` 做 smoke run，例如：

```bash
LIMIT=3 scripts/run_signpost_method.sh legal_test full legal_test
```

核心输出：

```text
outputs/<dataset>/predictions/signpost.full.jsonl
outputs/<dataset>/logs/signpost.full.query.jsonl
outputs/<dataset>/metrics/signpost.full.basic_eval.json
outputs/<dataset>/metrics/signpost.full.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

其中 `predictions/signpost.full.jsonl` 是统一预测 schema；`logs/signpost.full.query.jsonl` 是每个 query 的在线成本日志；`method_summaries.json` 默认把 F7-F10 作为 Signpost 专属图构建/图索引成本，F3/F3.5/F4 作为共享预处理不计入方法专属成本，F6 作为共享语义标注只记录不计入；`cost_quality.json` 用于保留离线成本、在线成本和质量的辅助汇总。

### 2.3 Signpost 消融

```bash
scripts/run_signpost_ablation_suite.sh <dataset> [namespace]
```

该脚本顺序运行以下 Signpost 变体：

```text
signpost.full
signpost.no_offline
signpost.no_online
signpost.no_semantic_cues
signpost.no_provenance_cues
signpost.no_vertical_cues
signpost.no_horizontal_cues
```

变体含义：

```text
full: 完整 Signpost。
no_offline: 移除所有离线 signpost cues。
no_online: 移除在线 PPR/navigation recommendations。
no_semantic_cues: 移除语义视图 cues，同时清空在线推荐，检验 LLM 语义视图的贡献。
no_provenance_cues: 移除 provenance/source locate cues，检验回读和引用定位对最终回答的影响。
no_vertical_cues: 移除层级/章节方向 cues。
no_horizontal_cues: 移除顺序邻接 cues。
```

每个变体都会产生同构产物：

```text
outputs/<dataset>/predictions/signpost.<variant>.jsonl
outputs/<dataset>/logs/signpost.<variant>.query.jsonl
outputs/<dataset>/metrics/signpost.<variant>.basic_eval.json
outputs/<dataset>/metrics/signpost.<variant>.query_metrics.json
```

同一个 `outputs/<dataset>/metrics/method_summaries.json` 会被逐步 upsert，`cost_quality.json` 会随每个变体更新。

## 3. legal_test 闭环步骤

`legal_test` 只用于 smoke，不进入论文表格。它的作用是确认 H200 环境、ES、embedding、LLM、索引构建、agent 检索和评估汇总全部连通。

推荐顺序：

```bash
cd /home/ruolinsu/signpost/signpost_re

SEMANTIC_EXTRACTOR=llm \
EMBEDDING_PROVIDER=ecnu \
scripts/run_signpost_dataset_pipeline.sh legal_test legal_test

LIMIT=3 \
EMBEDDING_PROVIDER=ecnu \
USE_ES=1 \
USE_LLM=1 \
scripts/run_signpost_method.sh legal_test full legal_test

LIMIT=3 \
EMBEDDING_PROVIDER=ecnu \
USE_ES=1 \
USE_LLM=1 \
scripts/run_signpost_ablation_suite.sh legal_test legal_test
```

核对命令：

```bash
test -s datasets/processed/legal_test/graph.unified.json
test -s outputs/legal_test/metrics/index_metrics.json
test -s outputs/legal_test/predictions/signpost.full.jsonl
test -s outputs/legal_test/metrics/signpost.full.query_metrics.json
test -s outputs/legal_test/metrics/cost_quality.json
```

`LIMIT=3` 只限制在线 query 阶段，不改变离线索引构建。smoke 通过后，正式实验去掉 `LIMIT`。

## 4. 正式数据集运行

按 v8 计划，主实验优先跑：

```text
Agriculture-full
Legal-full
```

如果 `Legal-full` 的 F6 语义抽取在实验窗口内完成，就不再使用 legal 子集进入主表。若 F6 无法完成，则用 `Legal-Core` 替代主实验，但这个替代必须在论文中说明。

推荐顺序：

```bash
scripts/run_signpost_dataset_pipeline.sh Agriculture-full Agriculture-full
scripts/run_signpost_ablation_suite.sh Agriculture-full Agriculture-full

scripts/run_signpost_dataset_pipeline.sh Legal-full Legal-full
scripts/run_signpost_ablation_suite.sh Legal-full Legal-full
```

如果需要先判断 `Legal-full` 成本，可先跑到 F6 并观察：

```text
datasets/processed/Legal-full/semantic_llm.progress.jsonl
datasets/processed/Legal-full/semantic_llm.extractions.jsonl
outputs/Legal-full/logs/stage_timing.jsonl
```

Legal 的主要慢点是 F6 per-chunk semantic extraction；如果已经完成 `Legal-full` 的离线抽取，后续 full 方法、消融和 query metrics 不再需要 legal 子集。

## 5. 每个实验指标使用哪些数据和阶段

主质量指标使用：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/metrics/<method>.basic_eval.json
outputs/<dataset>/metrics/<method>.query_metrics.json
```

对 Signpost full 和所有消融，质量指标在同一个数据集全集问题上计算。`legal_test` 不进入论文主表。

离线索引成本使用：

```text
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/metrics/index_metrics.json
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/graph.unified.json
```

在线成本使用：

```text
outputs/<dataset>/logs/signpost.<variant>.query.jsonl
outputs/<dataset>/metrics/signpost.<variant>.query_metrics.json
```

cost-quality 辅助汇总使用：

```text
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

论文中对应的核心说法是：Signpost 的方法成本主要来自共享语义标注之后的图组织、路标物化和图索引同步；F6 语义抽取成本必须记录，但在主表中作为共享语义标注阶段处理。消融用于证明这不是简单堆叠，而是不同导航线索在证据可达性、回读定位和在线探索成本上分别承担作用。

## 6. H200 本地模型配置

当前代码的 `ecnu` provider 实际读取 OpenAI-compatible 环境变量。H200 上如果部署本地 `Llama-3.3-70B-Instruct` 和 `nvidia/llama-embed-nemotron-8b`，只要服务暴露 OpenAI-compatible `/chat/completions` 和 `/embeddings`，即可继续使用：

```bash
EMBEDDING_PROVIDER=ecnu
USE_LLM=1
```

并在 `.env.h200` 或 shell 中设置：

```bash
ECNU_API_BASE=http://<h200-host>:<port>/v1
ECNU_API_KEY=<local-service-key-or-dummy-key>
ECNU_CHAT_MODEL=meta-llama/Llama-3.3-70B-Instruct
ECNU_REASONING_MODEL=meta-llama/Llama-3.3-70B-Instruct
ECNU_EMBEDDING_MODEL=nvidia/llama-embed-nemotron-8b
```

正式实验中 Signpost、Vanilla RAG、GraphRAG 和 agentic baselines 应尽量使用同一套本地 chat/embedding 服务。这样在线检索仍然是在线阶段：它发生在 query 到来之后；是否本地部署只影响可控性、延迟和计费口径，不改变 offline/online 的实验定义。

## 7. 与外部 baseline 的衔接

Signpost 完成后，外部 baseline 只需要对齐两个文件：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
```

然后复用：

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

python -m signpost.benchmark.method_summary \
  --method <method> \
  --dataset <dataset> \
  --query-metrics outputs/<dataset>/metrics/<method>.query_metrics.json \
  --stage-log outputs/<dataset>/logs/stage_timing.jsonl \
  --output outputs/<dataset>/metrics/method_summaries.json

python -m signpost.benchmark.cost_quality \
  --methods outputs/<dataset>/metrics/method_summaries.json \
  --output outputs/<dataset>/metrics/cost_quality.json
```

外部 baseline 的 wrapper 不需要复用 Signpost 的 `graph.unified.json`，除非该 baseline 明确要求同类输入。主表比较的是每个方法在同一 raw corpus/questions 下构建自身索引并回答问题后的质量和成本。
