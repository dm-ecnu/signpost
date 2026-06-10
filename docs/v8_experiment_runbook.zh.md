# v8 实验执行 Runbook

本文档根据 `paper_drafts/v8` 的论文设计，明确需要做哪些实验、哪些实验先用 `legal_test` 闭环测试、哪些必须在 H200 上正式跑、每一步输入输出是什么，以及最终结果如何整理成论文表格。

## 1. v8 需要的实验总表

v8 论文不是只做一个 Signpost 主实验。必须包含以下实验组：

| RQ | 实验 | 主数据集 | 方法 | 产出 |
|---|---|---|---|---|
| RQ1 | Answer quality | Agriculture-full, Legal-full 或 Legal-Core fallback | Vanilla LLM, Vanilla RAG, LinearRAG/RAPTOR, LeanRAG, LightRAG, A-RAG, Youtu/agent fallback, Signpost | Table: main quality |
| RQ2 | Evidence reachability + blind exploration | Agriculture-full, Legal-full/fallback | Vanilla RAG, LinearRAG, best GraphRAG, A-RAG, best agentic, Signpost | Table: evidence/blind exploration |
| RQ3 | Amortized cost-effectiveness | Agriculture-full, Legal-full | key baselines + Signpost | Figure: AC/CPGA/ICER curves; Table: offline/online cost |
| RQ4 | Signpost ablation | Agriculture-full, Legal-full/fallback | Signpost variants only | Table: ablation |
| RQ5 | Diagnostics + scalability | Legal-full/fallback; Mix-full optional | k-hop, PPR-only, Structure-Aware Coordinate Reader, Signpost | Table: diagnostics; optional scaling figure |

注意：

- 不是所有实验都跑所有数据集。
- 不是所有 baseline 都必须跑 diagnostics 和 ablation。
- ablation 只针对 Signpost。
- Legal-full 如果完成，就不要在主表中混用 Legal-Core。

## 2. Baseline 是否直接在 H200 上部署和测试

最终结果必须在 H200 上跑，但 baseline 不应该在 H200 上临时开发。

正确方式：

1. 本地接入 baseline，写成固定 runner。
2. 本地用 `legal_test` 或 `--limit 3` 验证输入输出格式。
3. commit 代码。
4. H200 拉取固定 commit。
5. H200 跑 baseline full experiment。

外部 baseline 的原则：

- 给同一批 raw documents。
- 允许 baseline 使用自己的 preprocessing/chunk/index pipeline。
- 其 preprocessing/index/build time 计入该 baseline 的 offline cost。
- 不允许 baseline 使用 Signpost 的 signpost annotations。
- 若使用统一 chunks，需要在论文中写明 `unified-chunk setting`。

当前 `signpost_re` 已经比较完整支持 Signpost 主链路；Vanilla LLM、Vanilla RAG、ablation variants、diagnostics 和外部 baseline runner 还需要补齐或封装。

## 3. legal_test 闭环测试目标

`legal_test` 不是论文结果，只用于证明：

1. H200 环境可用。
2. 本地模型 endpoint 可用。
3. ES 可用。
4. F3-F16 全链路可以跑通。
5. 每一步有功能输出和测量输出。
6. query log、prediction、evaluation、metrics 能被汇总。

`legal_test` 成功后才能跑 Agriculture-full；Agriculture-full 成功后才能跑 Legal-full。

## 4. legal_test Signpost 闭环步骤

下面命令以项目根目录为工作目录：

```bash
cd /data/<user>/signpost/signpost_re
set -a
source .env.h200
set +a
```

本地开发机路径可替换为：

```bash
cd /home/ruolinsu/signpost/signpost_re
```

### Step 0: 创建输出目录

```bash
mkdir -p outputs/legal_test/{logs,logs/stage_metrics,predictions,metrics}
```

核对：

```text
outputs/legal_test/logs/
outputs/legal_test/predictions/
outputs/legal_test/metrics/
```

### Step 1: F3 数据准备

命令：

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

命令：

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

输入：

```text
datasets/processed/legal_test/raw_corpus.jsonl
```

输出：

```text
datasets/processed/legal_test/documents.jsonl
```

核对：

```bash
test -s datasets/processed/legal_test/documents.jsonl
```

### Step 3: F4 chunking + document tree

命令：

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

Signpost 正式检索建议使用 ES。smoke 可以用 `hash` embedding，正式 H200 应用本地 embedding provider。

命令：

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

正式 H200 改为：

```text
--embedding-provider ecnu
```

或者先把本地 OpenAI-compatible embedding endpoint 伪装到现有 provider 配置中。

输出：

```text
Elasticsearch chunk index
stage_timing.jsonl 中 F5_chunk_index 一行
```

核对：

```bash
curl http://localhost:9200/_cat/indices?v | grep legal_test
```

### Step 5: F6 semantic graph

`legal_test` 闭环可以先跑 deterministic，确认流程；随后必须跑 LLM extractor 小规模 smoke。

#### 5A deterministic smoke

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

#### 5B LLM smoke / final mode

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

输入：

```text
chunks.jsonl
本地 Llama chat endpoint
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
tail -n 3 datasets/processed/legal_test/semantic_llm.progress.jsonl
wc -l datasets/processed/legal_test/semantic_llm.extractions.jsonl
```

正式实验中，Legal-full 的最大风险就是这一步。必须保留 `extractions-cache`，否则中断后无法有效续跑。

### Step 6: F7 structure graph

命令：

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

正式实验如果使用 LLM summaries，需要把 summarizer 改为 `llm`，并把对应 token/time 算入 offline cost。

### Step 7: F8 sequence graph

命令：

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

如果前面使用 LLM semantic graph，F9 应使用 `graph.semantic.llm.json`。

命令：

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

核对：

```bash
test -s datasets/processed/legal_test/graph.unified.json
```

### Step 9: F10 graph object index

命令：

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

正式 H200 使用本地 embedding provider。

输出：

```text
Elasticsearch graph object index
chunk index parent fields optionally updated
```

### Step 10: F13 单问题检索 smoke

命令：

```bash
python -m signpost.retrieval.run \
  --namespace legal_test \
  --query "What is the main obligation discussed in the document?" \
  --graph datasets/processed/legal_test/graph.unified.json \
  --mode hybrid \
  --embedding-provider hash \
  --output outputs/legal_test/retrieval_result.json
```

输出：

```text
outputs/legal_test/retrieval_result.json
```

核对 JSON 中必须有：

```text
text_group.items[].offline_signpost
text_group.online_signpost
graph_group.items[].offline_signpost
graph_group.online_signpost
```

这一步验证 v8 论文里的核心对象：signpost-enriched observation。

### Step 11: F15 agent batch smoke

先 limit 3，不要直接全量。

命令：

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

正式 H200 改为本地 embedding provider，并去掉 `--limit`。

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

prediction 每行应包含：

```text
question_id
prediction
citations
trace
latency_seconds
tool_calls
knowledge_search_calls
read_file_calls
input_tokens
output_tokens
total_tokens
```

### Step 12: F16 basic evaluation

命令：

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

可选 LLM judge：

```bash
python -m signpost.evaluation.llm_judge \
  --input outputs/legal_test/predictions/signpost.jsonl \
  --output outputs/legal_test/metrics/signpost.llm_judge.jsonl \
  --dimension answer_correctness
```

### Step 13: F17 指标汇总

#### index metrics

```bash
python -m signpost.benchmark.index_metrics \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --graph datasets/processed/legal_test/graph.unified.json \
  --gleaning-rounds 0 \
  --output outputs/legal_test/metrics/index_metrics.json
```

#### query metrics

```bash
python -m signpost.benchmark.query_metrics \
  --input outputs/legal_test/predictions/signpost.jsonl \
  --output outputs/legal_test/metrics/signpost.query_metrics.json \
  --normalize \
  --top-k 5 10
```

#### method summary

```bash
python -m signpost.benchmark.method_summary \
  --method signpost \
  --dataset legal_test \
  --query-metrics outputs/legal_test/metrics/signpost.query_metrics.json \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --offline-stage F6_semantic_graph_llm \
  --output outputs/legal_test/metrics/method_summaries.json
```

#### cost quality

```bash
python -m signpost.benchmark.cost_quality \
  --methods outputs/legal_test/metrics/method_summaries.json \
  --output outputs/legal_test/metrics/cost_quality.json
```

输出：

```text
index_metrics.json
signpost.query_metrics.json
method_summaries.json
cost_quality.json
```

## 5. 正式实验与 legal_test 的区别

`legal_test` 只验证闭环。正式实验需要：

| 项 | legal_test | 正式实验 |
|---|---|---|
| 模型 | 可先 hash / deterministic，再 LLM smoke | H200 local Llama + Nemotron |
| 数据 | legal_test | Agriculture-full, Legal-full/fallback |
| F6 | deterministic + small LLM smoke | LLM semantic extraction with cache |
| F15 | `--limit 3` | full questions |
| baseline | 不需要全跑 | 主表 baseline 都要跑 |
| ablation | 可只测 1-2 个 | RQ4 全部 Signpost variants |
| cost | 检查字段存在 | 论文主指标 |

## 6. Signpost 消融实验怎么做

v8 需要的 ablation：

1. Full Signpost。
2. w/o offline signpost。
3. w/o online signpost。
4. w/o semantic cues。
5. w/o provenance cues。
6. w/o vertical cues。
7. w/o horizontal cues。

当前代码的主链路默认是 Full Signpost。正式跑 ablation 前，需要补一个统一开关，建议加在 `KnowledgeSearchConfig` 或 retrieval postprocessor 中：

```text
signpost_variant:
  full
  no_offline
  no_online
  no_semantic_cues
  no_provenance_cues
  no_vertical_cues
  no_horizontal_cues
```

实现建议：

- `no_offline`: 不调用或清空 `offline_signpost`。
- `no_online`: 不计算或清空 `online_signpost`。
- `no_semantic_cues`: 删除 `offline_signpost.semantic`，并可将 online recommendation 置空；如果只想测 object-level semantic cues，则不要清空 online，需在实验名中区分。
- `no_provenance_cues`: 删除 `offline_signpost.provenance` 和 recommendation 中的 source locates。
- `no_vertical_cues`: 删除 `offline_signpost.vertical`。
- `no_horizontal_cues`: 删除 `offline_signpost.horizontal`。

每个 variant 的输出路径：

```text
outputs/<dataset>/predictions/signpost.<variant>.jsonl
outputs/<dataset>/logs/signpost.<variant>.query.jsonl
outputs/<dataset>/metrics/signpost.<variant>.query_metrics.json
```

消融实验只跑 Signpost variants，不跑所有 baseline。

## 7. Diagnostics 怎么做

v8 需要 diagnostics，但它们不应该阻塞主实验。

### 7.1 k-hop expansion

目的：证明 Signpost 不等价于普通邻域扩展。

输入：

```text
graph.unified.json
same initial seeds as Signpost retrieval
```

输出：

```text
retrieved evidence candidates
prediction/evidence metrics
```

### 7.2 PPR-only

目的：证明在线 PPR 本身不等于完整 Signpost。

做法：

- 使用同样 seeds。
- 只返回 PPR recommended entities/source chunks。
- 不附着 object-level offline signposts。

### 7.3 Structure-Aware Coordinate Reader

目的：DeepRead-style diagnostic。

做法：

- 只提供 section/line locate 和 read section/read file。
- 不提供 semantic cues。
- 不提供 online group-level PPR recommendation。

Diagnostics 推荐只在 Legal-full/fallback 上跑；Mix-full 可选用于弱结构边界。

## 8. Baseline 实验怎么组织

### 8.1 In-house baseline

至少需要本地补齐：

1. `vanilla_llm` runner：读 questions，直接 LLM answer，不检索。
2. `vanilla_rag` runner：chunk retrieval -> generator answer，不附着 signposts。
3. `vanilla_rag_agent` fallback：agent 有 search/read，但 observation 不含 signposts。

它们可以复用：

```text
documents.jsonl
chunks.jsonl
chunk ES index
questions.jsonl
```

但输出必须和 Signpost prediction schema 对齐：

```text
predictions/<method>.jsonl
logs/<method>.query.jsonl
metrics/<method>.query_metrics.json
```

### 8.2 外部 baseline

LinearRAG、LeanRAG、LightRAG、A-RAG、Youtu-GraphRAG：

1. 本地先写 wrapper，规定 input/output。
2. H200 上 clone 官方代码或 vendor 到 `baselines/`。
3. 统一模型 endpoint。
4. 输出转换成 Signpost prediction schema。
5. 记录 offline build log 和 online query log。

如果某个外部 baseline 不能稳定运行，不要硬放进主表。用 fallback，并在论文中说明 reproducibility constraint。

## 9. 最终结果怎么整理

每个 dataset/method 最终必须有：

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
    method_summaries.json
    cost_quality.json
```

论文表格对应关系：

| 论文表/图 | 来源 |
|---|---|
| Dataset statistics | `chunks.jsonl`, `documents.jsonl`, `questions.jsonl`, `graph.unified.json`, `index_metrics.json` |
| Baseline matrix | 手工维护的 baseline registry + run manifest |
| Main quality | `*.basic_eval.json`, `*.llm_judge.jsonl`, `*.query_metrics.json` |
| Evidence/blind exploration | `*.query_metrics.json`, `*.query.jsonl`, prediction trace |
| Offline/online cost | `stage_timing.jsonl`, `index_metrics.json`, `*.query_metrics.json` |
| Ablation | `signpost.<variant>.query_metrics.json` |
| Diagnostics | diagnostic method predictions/query metrics |
| Amortization curves | `method_summaries.json` -> `cost_quality.json` |

## 10. 推荐执行顺序

### 本地

1. 跑现有 unit tests。
2. 用 `legal_test` 完成 F3-F16 Signpost 闭环。
3. 补 ablation 开关。
4. 补 vanilla LLM / vanilla RAG runner。
5. 补 baseline output converter。
6. commit。

### H200

1. 配环境和本地模型。
2. 跑 `legal_test` F3-F16。
3. 跑 Agriculture-full Signpost。
4. 跑 Agriculture-full Vanilla LLM / Vanilla RAG / key baseline。
5. 跑 Legal-full F6 semantic extraction。
6. 若 Legal-full 完成，继续 Legal-full 全主实验和消融。
7. 若 Legal-full 未完成，构造 Legal-Core fallback，同时保留 Legal-full 已完成部分做 cost/scaling。
8. 跑 cost_quality 汇总。

## 11. 当前最重要的工程缺口

在开始正式 H200 实验前，必须补：

1. Signpost ablation variant 开关。
2. Vanilla LLM runner。
3. Vanilla RAG runner。
4. 统一 run manifest。
5. baseline wrapper/output converter。
6. dataset pipeline shell script。
7. method batch shell script。

没有这些，H200 上会变成手工实验，后面很难整理成论文指标。
