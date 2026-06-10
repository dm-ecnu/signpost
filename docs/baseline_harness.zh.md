# Baseline Harness 设计与运行说明

本文档说明当前已经实现的 baseline 运行框架，以及 `Vanilla LLM`、`Hybrid RAG` 两个 in-house 控制组的设计、输入输出和运行命令。`Vanilla RAG` 旧入口保留兼容，但正式论文和正式实验使用 `Hybrid RAG`。更完整的控制组说明见 `docs/baselines/in_house_controls_zh.md`，最终外部 baseline 名单见 `docs/baselines/final_baseline_selection_zh.md`。每个 baseline 的独立操作手册放在 `docs/baselines/<method>/`。

## 1. 目标

Baseline harness 的目标是把所有 baseline 都约束到同一套实验接口：

```text
输入：datasets/processed/<dataset>/documents.jsonl
输入：datasets/processed/<dataset>/chunks.jsonl
输入：datasets/processed/<dataset>/questions.jsonl

输出：outputs/<dataset>/predictions/<method>.jsonl
输出：outputs/<dataset>/logs/<method>.query.jsonl
输出：outputs/<dataset>/metrics/<method>.basic_eval.json
输出：outputs/<dataset>/metrics/<method>.query_metrics.json
输出：outputs/<dataset>/metrics/method_summaries.json
输出：outputs/<dataset>/metrics/cost_quality.json
```

这样 baseline 结果可以和 `signpost.full` 放入同一张主对比表；Signpost 消融仍然单独成表。

## 1.1 实验成本与共享语义标注口径

当前正式实验采用以下统一口径：

```text
F3/F3.5/F4: 共享数据准备，不计入方法离线成本。
F5: chunk index，Hybrid RAG 在 ES 正式设置下计入其 flat retrieval index 成本；Signpost 主表默认记录但不计入图构建离线成本。
F6: chunk-level entity/relation extraction，作为共享语义标注阶段。
F7/F8/F9/F10: Signpost 图组织、统一图和索引同步，计入 Signpost 方法专属离线成本。
```

因此：

```text
1. F6 的时间、LLM 调用次数和 token 必须记录，方便后续透明报告或附录分析。
2. F6 默认不计入主表中的方法离线成本。
3. 需要实体/关系输入的 baseline 统一复用 F6 产物，再接入自己的图组织、索引或检索阶段。
4. 若某个官方 baseline 无法复用 F6，必须单独记录并在论文中说明不可比风险。
```

在线阶段统一记录：

```text
每次检索时间、总 latency、LLM calls、input/output/total tokens、tool calls。
```

与 agentic RAG 方法比较时，重点看 LLM 调用次数、tokens、tool calls 和总 latency；与非 agentic 单次检索方法比较时，重点看单次 retrieval latency 和 top-k 检索质量。

## 2. 文件结构

当前新增文件：

```text
signpost/baselines/__init__.py
signpost/baselines/common.py
signpost/baselines/vanilla_llm.py
signpost/baselines/vanilla_rag.py
signpost/baselines/hybrid_rag.py

scripts/__init__.py
scripts/baselines/__init__.py
scripts/baselines/run_vanilla_llm.py
scripts/baselines/run_vanilla_rag.py
scripts/baselines/run_hybrid_rag.py
scripts/baselines/run_baseline_method.sh

tests/test_baselines.py
docs/baseline_harness.zh.md
docs/baselines/README.zh.md
docs/baselines/vanilla_llm/runbook.zh.md
docs/baselines/hybrid_rag/runbook.zh.md
```

功能覆盖：

```text
common.py：统一 question 读取、prediction 写入、query log 写入、成本字段估计、context 拼接。
vanilla_llm.py：无检索 LLM baseline。
vanilla_rag.py：扁平 chunk RAG baseline，支持 ES 检索和本地 smoke 检索。
hybrid_rag.py：正式 Hybrid RAG alias，复用 flat RAG 实现但输出 `method=hybrid_rag`。
run_baseline_method.sh：运行 baseline 后自动接 basic_eval、query_metrics、method_summary、cost_quality。
tests/test_baselines.py：用 fake LLM 验证输出 schema，不访问真实模型服务。
```

## 3. 统一 Prediction Schema

每个 baseline 的 prediction JSONL 每行至少包含：

```text
question_id
question
answer
rationale
prediction
citations
trace
retrieved_chunks
latency_seconds
retrieval_latency_seconds
read_file_latency_seconds
agent_reasoning_latency_seconds
llm_calls
online_llm_calls
tool_calls
knowledge_search_calls
read_file_calls
input_tokens
output_tokens
total_tokens
metadata.method
metadata.dataset
metadata.namespace
```

该 schema 可直接被现有命令消费：

```bash
python -m signpost.evaluation.evaluate_basic --normalize
python -m signpost.benchmark.query_metrics --normalize
python -m signpost.benchmark.method_summary
python -m signpost.benchmark.cost_quality
```

## 4. Baseline 1: Vanilla LLM

### 4.1 设计

`Vanilla LLM` 不使用任何检索上下文，只把问题发给 H200 本地 chat model。

作用：

```text
无检索下界；
检验问题本身是否能被模型参数知识直接回答；
不产生 evidence recall，不产生 retrieved_chunks。
```

输入：

```text
datasets/processed/<dataset>/questions.jsonl
```

输出：

```text
outputs/<dataset>/predictions/vanilla_llm.jsonl
outputs/<dataset>/logs/vanilla_llm.query.jsonl
outputs/<dataset>/metrics/vanilla_llm.basic_eval.json
outputs/<dataset>/metrics/vanilla_llm.query_metrics.json
```

核心文件：

```text
signpost/baselines/vanilla_llm.py
scripts/baselines/run_vanilla_llm.py
```

### 4.2 运行命令

Smoke：

```bash
LIMIT=3 scripts/baselines/run_baseline_method.sh vanilla_llm legal_test legal_test
```

正式：

```bash
scripts/baselines/run_baseline_method.sh vanilla_llm agriculture agriculture
```

## 5. Baseline 2: Hybrid RAG

### 5.1 设计

`Hybrid RAG` 是扁平 chunk 检索增强生成：

```text
1. 对 question 检索 top-k chunks。
2. 将 chunks 拼接为 context。
3. 调用 H200 本地 chat model 生成答案。
4. 输出 retrieved_chunks 和基于 chunk line range 的 citations。
```

正式实验建议使用 ES：

```text
USE_ES=1
MODE=hybrid
EMBEDDING_PROVIDER=ecnu
```

这里的 `ecnu` 是历史 provider 名称。在 H200 上它通过 `.env.h200` 指向本地 `/data/srl/nemotron-8b` embedding 服务。

本地 smoke 可以不用 ES：

```text
USE_ES=0
MODE=bm25
```

输入：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl
```

正式 ES 模式还要求该 dataset 已经跑过：

```bash
scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>
```

因为 F5 会建立 chunk ES index。Hybrid RAG 复用同一份 chunk index；在成本汇总中 `run_baseline_method.sh` 会把 `F5_chunk_index` 作为 `hybrid_rag` 的离线索引成本。

输出：

```text
outputs/<dataset>/predictions/hybrid_rag.jsonl
outputs/<dataset>/logs/hybrid_rag.query.jsonl
outputs/<dataset>/metrics/hybrid_rag.basic_eval.json
outputs/<dataset>/metrics/hybrid_rag.query_metrics.json
```

核心文件：

```text
signpost/baselines/vanilla_rag.py
signpost/baselines/hybrid_rag.py
scripts/baselines/run_vanilla_rag.py
scripts/baselines/run_hybrid_rag.py
```

### 5.2 运行命令

Smoke without ES：

```bash
LIMIT=3 USE_ES=0 MODE=bm25 scripts/baselines/run_baseline_method.sh hybrid_rag legal_test legal_test
```

Smoke with ES：

```bash
LIMIT=3 USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu scripts/baselines/run_baseline_method.sh hybrid_rag legal_test legal_test
```

正式：

```bash
USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu scripts/baselines/run_baseline_method.sh hybrid_rag agriculture agriculture
```

可调参数：

```text
TOP_K=5
MAX_CONTEXT_TOKENS=3500
MODE=bm25|dense|hybrid
EMBEDDING_PROVIDER=ecnu|hash
LIMIT=<n>
USE_ES=0|1
```

## 6. H200 环境配置

运行前加载 `.env.h200`：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a
```

必要变量：

```bash
ECNU_API_BASE=http://localhost:8000/v1
ECNU_API_KEY=EMPTY
ECNU_CHAT_MODEL=/data/srl/Llama-3.3-70B-FP8
ECNU_REASONING_MODEL=/data/srl/Llama-3.3-70B-FP8
ECNU_EMBEDDING_API_BASE=http://localhost:8001/v1/embeddings
ECNU_EMBEDDING_API_KEY=EMPTY
ECNU_EMBEDDING_MODEL=/data/srl/nemotron-8b
ECNU_RERANK_MODEL=unused-local-rerank
```

先确认模型服务：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

## 7. 推荐接入顺序

当前已经实现：

```text
1. vanilla_llm
2. hybrid_rag
```

后续外部 baseline 以 `docs/baselines/final_baseline_selection_zh.md` 为准，当前推荐接入顺序：

```text
3. Clue-RAG
4. AGRAG
5. LinearRAG
6. HiPRAG
7. GraphRAG-R1
```

每接一个外部 baseline，都必须完成：

```text
1. 官方仓库下载/安装说明；
2. 输入格式转换；
3. index/build 阶段；
4. query 阶段；
5. 输出转换成统一 prediction schema；
6. legal_test LIMIT=3 smoke；
7. agriculture smoke；
8. H200 full run。
```

## 8. 与 Signpost 结果的关系

主对比表：

```text
vanilla_llm
hybrid_rag
Clue-RAG
AGRAG
LinearRAG
HiPRAG
GraphRAG-R1
signpost.full
```

Signpost 消融表：

```text
signpost.full
signpost.no_offline
signpost.no_online
signpost.no_semantic_cues
signpost.no_provenance_cues
signpost.no_vertical_cues
signpost.no_horizontal_cues
```

Baseline 不复用 Signpost 消融结果。Baseline 只复用同一份输入数据和同一套模型服务。
