# v8 精确定版实验计划与执行步骤

本文档比 `v8_experiment_runbook.zh.md` 更收敛，只保留 v8 论文中明确写到的 baseline，不再使用 “best GraphRAG”“best agentic” 这类模糊表述。

## 1. 最终 baseline 名单

主表默认方法如下：

| Category | Method | Source | 角色 |
|---|---|---|---|
| No retrieval | Vanilla LLM | in-house | 无检索下界 |
| Flat RAG | Vanilla RAG | in-house | opaque chunk retrieval |
| Multi-granularity / hierarchical index | LinearRAG | https://github.com/DEEP-PolyU/LinearRAG | 多粒度/层次图索引 baseline |
| GraphRAG | LeanRAG | https://github.com/RaZzzyz/LeanRAG | hierarchical KG-RAG baseline |
| GraphRAG | LightRAG | https://github.com/HKUDS/LightRAG | lightweight GraphRAG baseline |
| Agentic retrieval | A-RAG | https://github.com/Ayanami0730/arag | agentic retrieval/read tools |
| Agentic retrieval | Youtu-GraphRAG | https://github.com/TencentCloudADP/youtu-graphrag | agentic GraphRAG |
| Ours | Signpost | this repo | navigation-cue index |

Fallback 规则：

1. `LinearRAG` 失败时，用 `RAPTOR` 替代，且主表中只出现一个：`LinearRAG` 或 `RAPTOR`。
2. `Youtu-GraphRAG` 失败时，优先用 `DRAGIN` 替代；如果 `DRAGIN` 也无法复现，才用 `Vanilla RAG Agent`。
3. Agentic retrieval 最多两个方法：默认 `A-RAG` + `Youtu-GraphRAG`；fallback 后仍然保持两个。
4. 不再单独比较 BM25 RAG、Dense RAG、Hybrid RAG。它们只是 `Vanilla RAG` 的内部配置，调优后报告一个 `Vanilla RAG`。
5. 如果某个外部 baseline 没有在 `legal_test` 和 Agriculture smoke 上跑通，不进入正式主表。

## 2. baseline 是否在服务器上部署

最终指标必须在 H200 上运行，但 baseline 不应该直接在 H200 上临时开发。

操作原则：

1. 本地写 wrapper 和 output converter。
2. 本地用 `legal_test` 或 `--limit 3` 验证 baseline 输入输出。
3. commit 代码。
4. H200 拉固定 commit。
5. H200 clone / install baseline 官方仓库。
6. H200 跑正式数据。
7. 输出转换成统一 schema。

每个外部 baseline 需要一个目录：

```text
baselines/
  LinearRAG/
  LeanRAG/
  LightRAG/
  arag/
  youtu-graphrag/
```

每个 baseline 需要一个 wrapper：

```text
scripts/baselines/run_linearrag.py
scripts/baselines/run_leanrag.py
scripts/baselines/run_lightrag.py
scripts/baselines/run_arag.py
scripts/baselines/run_youtu_graphrag.py
```

wrapper 统一输入：

```text
--dataset <dataset>
--raw-docs datasets/processed/<dataset>/documents.jsonl
--chunks datasets/processed/<dataset>/chunks.jsonl
--questions datasets/processed/<dataset>/questions.jsonl
--output outputs/<dataset>/predictions/<method>.jsonl
--query-log outputs/<dataset>/logs/<method>.query.jsonl
```

wrapper 统一输出：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
```

prediction schema 必须至少包含：

```text
question_id
question
answer
prediction
citations
retrieved_chunks
latency_seconds
tool_calls
input_tokens
output_tokens
total_tokens
metadata.method
metadata.dataset
```

## 3. 数据集使用

主实验数据集：

1. `Agriculture-full`
2. `Legal-full`

如果 `Legal-full` 的 F6 semantic extraction 无法在实验窗口内完成，则使用 `Legal-Core` 替代主实验；如果 `Legal-full` 完成，就不使用 `Legal-Core` 进入主表。

可选 robustness：

1. `Mix-full`
2. `GraphRAG-Bench-full`

可选数据集不跑全矩阵，只跑：

```text
Vanilla RAG
LightRAG
A-RAG
Signpost
```

这张 robustness 表不是主结论。

## 4. legal_test Signpost 闭环

`legal_test` 只用于 smoke，不进入论文表格。目标是确认 F3-F17 全链路可运行，并且每一步产物可核对。

工作目录：

```bash
cd /home/ruolinsu/signpost/signpost_re
```

或 H200：

```bash
cd /data/<user>/signpost/signpost_re
set -a
source .env.h200
set +a
```

### Step 0: 输出目录

```bash
mkdir -p outputs/legal_test/{logs,logs/stage_metrics,predictions,metrics}
```

输出：

```text
outputs/legal_test/logs/
outputs/legal_test/predictions/
outputs/legal_test/metrics/
```

### Step 1: F3 数据准备

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_data_prepare \
  --method-scope shared_preprocess \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --output-path datasets/processed/legal_test/raw_corpus.jsonl \
  --disk-path datasets/processed/legal_test \
  --auto-metrics \
  -- \
  python -m signpost.data.prepare --datasets legal_test
```

输入：

```text
datasets/raw/... 或 data.prepare 支持的 raw source
```

输出：

```text
datasets/processed/legal_test/raw_corpus.jsonl
datasets/processed/legal_test/questions.jsonl
outputs/legal_test/logs/stage_timing.jsonl
```

核对：

```bash
test -s datasets/processed/legal_test/raw_corpus.jsonl
test -s datasets/processed/legal_test/questions.jsonl
tail -n 1 outputs/legal_test/logs/stage_timing.jsonl
```

### Step 2: F3.5 文档解析

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_5_parse_documents \
  --method-scope shared_preprocess \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/raw_corpus.jsonl \
  --output-path datasets/processed/legal_test/documents.jsonl \
  --disk-path datasets/processed/legal_test/documents.jsonl \
  --auto-metrics \
  -- \
  python -m signpost.parsing.parse_documents \
    --input datasets/processed/legal_test/raw_corpus.jsonl \
    --output datasets/processed/legal_test/documents.jsonl
```

输出：

```text
datasets/processed/legal_test/documents.jsonl
```

### Step 3: F4 chunk/tree

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F4_chunk_tree \
  --method-scope shared_preprocess \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path datasets/processed/legal_test/chunks.jsonl \
  --disk-path datasets/processed/legal_test \
  --auto-metrics \
  -- \
  python -m signpost.chunking.run \
    --input datasets/processed/legal_test/documents.jsonl \
    --output datasets/processed/legal_test/chunks.jsonl \
    --tree-output datasets/processed/legal_test/document_trees.jsonl \
    --max-tokens 512 \
    --overlap-tokens 64
```

输出：

```text
datasets/processed/legal_test/chunks.jsonl
datasets/processed/legal_test/document_trees.jsonl
```

核对：

```bash
wc -l datasets/processed/legal_test/chunks.jsonl
wc -l datasets/processed/legal_test/document_trees.jsonl
```

### Step 4: F5 chunk index

legal_test 可用 `hash` smoke；正式 H200 用本地 embedding provider。

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F5_chunk_index \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path outputs/legal_test/logs/F5_chunk_index.done \
  --auto-metrics \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace legal_test \
    --dataset-id legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --embedding-provider hash \
    --recreate
```

输出：

```text
Elasticsearch chunk index
```

核对：

```bash
curl http://localhost:9200/_cat/indices?v | grep legal_test
```

### Step 5: F6 semantic graph

先 deterministic smoke，再 LLM smoke。

Deterministic:

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph_det \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.json \
  --disk-path datasets/processed/legal_test/graph.semantic.json \
  --auto-metrics \
  -- \
  python -m signpost.indexing.semantic_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.json \
    --extractor deterministic
```

LLM:

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph_llm \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.llm.json \
  --disk-path datasets/processed/legal_test/graph.semantic.llm.json \
  --auto-metrics \
  -- \
  python -m signpost.indexing.semantic_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 0 \
    --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
    --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
    --llm-retries 3 \
    --retry-sleep 5
```

输出：

```text
graph.semantic.llm.json
semantic_llm.progress.jsonl
semantic_llm.extractions.jsonl
```

核对：

```bash
test -s datasets/processed/legal_test/graph.semantic.llm.json
wc -l datasets/processed/legal_test/semantic_llm.extractions.jsonl
tail -n 3 datasets/processed/legal_test/semantic_llm.progress.jsonl
```

### Step 6: F7 structure graph

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F7_structure_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.structure.json \
  --disk-path datasets/processed/legal_test/graph.structure.json \
  --auto-metrics \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --document-trees datasets/processed/legal_test/document_trees.jsonl \
    --output datasets/processed/legal_test/graph.structure.json \
    --summarizer deterministic
```

输出：

```text
graph.structure.json
```

### Step 7: F8 sequence graph

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F8_sequence_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.sequence.json \
  --disk-path datasets/processed/legal_test/graph.sequence.json \
  --auto-metrics \
  -- \
  python -m signpost.indexing.sequence_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.sequence.json
```

输出：

```text
graph.sequence.json
```

### Step 8: F9 unified graph

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F9_unified_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --output-path datasets/processed/legal_test/graph.unified.json \
  --disk-path datasets/processed/legal_test/graph.unified.json \
  --auto-metrics \
  -- \
  python -m signpost.graph.merge \
    --namespace legal_test \
    --semantic datasets/processed/legal_test/graph.semantic.llm.json \
    --structure datasets/processed/legal_test/graph.structure.json \
    --sequence datasets/processed/legal_test/graph.sequence.json \
    --output datasets/processed/legal_test/graph.unified.json
```

输出：

```text
graph.unified.json
```

### Step 9: F10 graph object index

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_es_sync \
  --method-scope method_offline_index \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/logs/F10_graph_es_sync.done \
  --auto-metrics \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider hash \
    --recreate \
    --update-chunk-parents
```

输出：

```text
Elasticsearch graph index
```

### Step 10: F13 single-query retrieval smoke

```bash
python -m signpost.retrieval.run \
  --namespace legal_test \
  --query "What is the main obligation discussed in the document?" \
  --graph datasets/processed/legal_test/graph.unified.json \
  --mode hybrid \
  --embedding-provider hash \
  --output outputs/legal_test/retrieval_result.json
```

必须核对：

```text
text_group.items[].offline_signpost
text_group.online_signpost
graph_group.items[].offline_signpost
graph_group.online_signpost
```

### Step 11: F15 agent batch smoke

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F15_agent_batch_signpost \
  --method-scope online_query \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path datasets/processed/legal_test/questions.jsonl \
  --output-path outputs/legal_test/predictions/signpost.jsonl \
  --disk-path outputs/legal_test/predictions/signpost.jsonl \
  --auto-metrics \
  -- \
  python -m signpost.agent.batch \
    --namespace legal_test \
    --dataset legal_test \
    --questions datasets/processed/legal_test/questions.jsonl \
    --output outputs/legal_test/predictions/signpost.jsonl \
    --embedding-provider hash \
    --use-es \
    --limit 3 \
    --query-log outputs/legal_test/logs/signpost.query.jsonl
```

输出：

```text
outputs/legal_test/predictions/signpost.jsonl
outputs/legal_test/logs/signpost.query.jsonl
```

核对：

```bash
wc -l outputs/legal_test/predictions/signpost.jsonl
wc -l outputs/legal_test/logs/signpost.query.jsonl
```

### Step 12: F16 evaluation

```bash
python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F16_basic_eval_signpost \
  --method-scope evaluation \
  --method signpost \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  --input-path outputs/legal_test/predictions/signpost.jsonl \
  --output-path outputs/legal_test/metrics/signpost.basic_eval.json \
  --auto-metrics \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input outputs/legal_test/predictions/signpost.jsonl \
    --output outputs/legal_test/metrics/signpost.basic_eval.json \
    --normalize
```

输出：

```text
outputs/legal_test/metrics/signpost.basic_eval.json
```

### Step 13: F17 metrics summary

```bash
python -m signpost.benchmark.index_metrics \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --graph datasets/processed/legal_test/graph.unified.json \
  --gleaning-rounds 0 \
  --output outputs/legal_test/metrics/index_metrics.json
```

```bash
python -m signpost.benchmark.query_metrics \
  --input outputs/legal_test/predictions/signpost.jsonl \
  --output outputs/legal_test/metrics/signpost.query_metrics.json \
  --normalize \
  --top-k 5 10
```

```bash
python -m signpost.benchmark.method_summary \
  --method signpost \
  --dataset legal_test \
  --query-metrics outputs/legal_test/metrics/signpost.query_metrics.json \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --offline-stage F6_semantic_graph_llm \
  --output outputs/legal_test/metrics/method_summaries.json
```

```bash
python -m signpost.benchmark.cost_quality \
  --methods outputs/legal_test/metrics/method_summaries.json \
  --output outputs/legal_test/metrics/cost_quality.json
```

输出：

```text
outputs/legal_test/metrics/index_metrics.json
outputs/legal_test/metrics/signpost.query_metrics.json
outputs/legal_test/metrics/method_summaries.json
outputs/legal_test/metrics/cost_quality.json
```

## 5. 正式实验矩阵

### RQ1: Answer quality

数据集：

```text
Agriculture-full
Legal-full, or Legal-Core fallback
```

方法：

```text
Vanilla LLM
Vanilla RAG
LinearRAG
LeanRAG
LightRAG
A-RAG
Youtu-GraphRAG
Signpost
```

如果 fallback：

```text
LinearRAG -> RAPTOR
Youtu-GraphRAG -> DRAGIN -> Vanilla RAG Agent
```

指标：

```text
Accuracy
answer quality score
LLM judge dimensions if needed
```

### RQ2: Evidence reachability and blind exploration

数据集：

```text
Agriculture-full
Legal-full, or Legal-Core fallback
```

方法：

```text
Vanilla RAG
LinearRAG
LeanRAG
LightRAG
A-RAG
Youtu-GraphRAG
Signpost
```

不包括 Vanilla LLM，因为它没有 retrieval trajectory。

指标：

```text
Evidence Recall@5/10
first supporting evidence step
grounding rate
repeated search count
query reformulation count
duplicate reads
failed actions
online tokens
```

### RQ3: Amortized cost-effectiveness

数据集：

```text
Agriculture-full
Legal-full
```

方法：

```text
Vanilla RAG
LinearRAG
LeanRAG
LightRAG
A-RAG
Youtu-GraphRAG
Signpost
```

Vanilla LLM 可列入 cost table，但不作为 ICER 的主要 baseline，因为无索引、无 evidence grounding。

指标：

```text
offline build time
offline extraction tokens
storage
online latency
online tokens
AC(N)
CPGA(N)
ICER(N)
break-even N
```

### RQ4: Signpost ablation

数据集：

```text
Agriculture-full
Legal-full, or Legal-Core fallback
```

方法：

```text
Signpost-full
Signpost-no-offline
Signpost-no-online
Signpost-no-semantic-cues
Signpost-no-provenance-cues
Signpost-no-vertical-cues
Signpost-no-horizontal-cues
```

指标：

```text
answer quality
evidence recall
repeated search count
online tokens
```

### RQ5: Diagnostics

数据集：

```text
Legal-full, or Legal-Core fallback
Mix-full optional
```

方法：

```text
k-hop expansion
PPR-only
Structure-Aware Coordinate Reader
Signpost
```

这些是 diagnostics，不是 baseline 主表。

指标：

```text
evidence recall
cross-doc coverage
blind-search count
online tokens
```

## 6. 结果整理

每个正式 dataset/method 必须有：

```text
outputs/<dataset>/
  logs/
    stage_timing.jsonl
    <method>.query.jsonl
  predictions/
    <method>.jsonl
  metrics/
    <method>.basic_eval.json
    <method>.query_metrics.json
```

每个 dataset 还需要：

```text
outputs/<dataset>/metrics/index_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

论文表格映射：

| 论文内容 | 文件来源 |
|---|---|
| Dataset statistics | `documents.jsonl`, `chunks.jsonl`, `questions.jsonl`, `graph.unified.json`, `index_metrics.json` |
| Main quality | `<method>.basic_eval.json`, optional LLM judge |
| Evidence reachability | `<method>.query_metrics.json`, prediction trace |
| Blind exploration | `<method>.query.jsonl`, prediction trace |
| Offline cost | `stage_timing.jsonl`, `index_metrics.json`, baseline build logs |
| Online cost | `<method>.query_metrics.json` |
| Amortization curves | `method_summaries.json` -> `cost_quality.json` |
| Ablation | `signpost-*.query_metrics.json` |
| Diagnostics | diagnostic predictions + query metrics |

## 7. 当前还必须补的代码

现有代码可跑 Signpost full chain，但还需要补：

1. `scripts/run_dataset_pipeline.sh`
2. `scripts/run_method_batch.sh`
3. `scripts/baselines/run_vanilla_llm.py`
4. `scripts/baselines/run_vanilla_rag.py`
5. `scripts/baselines/run_linearrag.py`
6. `scripts/baselines/run_leanrag.py`
7. `scripts/baselines/run_lightrag.py`
8. `scripts/baselines/run_arag.py`
9. `scripts/baselines/run_youtu_graphrag.py`
10. Signpost ablation variant 开关
11. diagnostic runners: k-hop, PPR-only, coordinate reader
12. run manifest writer

优先级：

```text
P0: run_dataset_pipeline.sh, run_method_batch.sh, Vanilla LLM, Vanilla RAG, Signpost ablation
P1: LinearRAG, LightRAG, A-RAG
P2: LeanRAG, Youtu-GraphRAG
P3: diagnostics
```

在 P0 没完成前，不要上 H200 跑正式实验。
