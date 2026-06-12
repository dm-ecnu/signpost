# 项目实验指标与日志说明

本文档说明本次为 项目实验补充的指标代码。它只负责测评、记账和汇总，不实现检索剪枝，也不实现 baseline。

如果你只是想先弄懂“这些指标为什么要测、先看哪些数”，建议先看更口语化的入门文档：

```text
docs/experiment_metrics_plain_guide.zh.md
```

## 1. 新增代码位置

新增模块都在 `signpost/benchmark/`：

| 文件 | 作用 |
| --- | --- |
| `stats.py` | 通用统计函数：sum、mean、median、p90、p95、min、max。 |
| `query_metrics.py` | 读取 prediction/query log，计算答案质量、在线成本、弱证据命中。 |
| `index_metrics.py` | 读取 stage log、semantic extraction cache、graph JSON，计算离线索引和图结构指标。 |
| `cost_quality.py` | 根据各方法 summary 计算摊销成本、break-even、Pareto frontier、每多答对一个问题的额外成本。 |
| `time_stage.py` | 包装任意阶段命令，自动记录一行 `stage_timing.jsonl`。 |

新增测试：

| 文件 | 作用 |
| --- | --- |
| `tests/test_benchmark_metrics.py` | 覆盖 query 成本、证据召回、图结构、摊销成本、stage timing 日志。 |

## 2. 指标分层

实验指标分成四层，对应实验设计文档里的口径。

### 2.1 Shared preprocessing

对应阶段：

```text
F3_data_prepare
F3_5_parse_normalize
F4_chunk_tree
```

含义：

- 数据标准化、文档解析、文本规范化、chunking、章节路径、document tree。
- 所有基于 chunks 的方法共享。
- 单独报表，不归入某个方法的专属 offline index cost。

需要记录：

- `wall_time_seconds`
- `documents`
- `chunks`
- 输入输出路径
- `status`

### 2.2 Method-specific offline index

对应阶段：

```text
F5_chunk_index
F6_semantic_graph
F7_structure_graph
F8_sequence_graph
F9_unified_graph
F10_graph_es_sync
```

含义：

- 某个方法为了运行额外需要构建的索引。
- BM25/Dense/Hybrid 主要使用 F5。
- Static Graph、GraphSearch、Signpost 使用 F5-F10。

需要记录：

- `wall_time_seconds`
- `llm_calls`
- `input_tokens`
- `output_tokens`
- `disk_bytes`
- 图规模：nodes、edges、edge type ratio、degree、connected components。
- F6 语义抽取规模：chunks、gleaning rounds、estimated LLM calls、entities/relations per chunk。

### 2.3 Online query

对应阶段：

```text
F11_offline_signpost
F12_online_signpost
F13_retrieval_batch
F14_read_file
F15_agent_batch
```

每条 query 记录：

- `online_llm_calls`
- `tool_calls`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `latency_seconds`
- `retrieval_latency_seconds`
- `ppr_latency_seconds`
- `read_file_latency_seconds`
- `agent_reasoning_latency_seconds`
- `retrieved_chunks`
- `read_file_calls`
- `graph_ppr_calls`
- `max_context_tokens`

聚合方式：

- sum
- mean
- median
- p90
- p95
- min/max

其中 `p95 latency` 是尾延迟指标，技术说明 online efficiency 表里应该保留。

### 2.4 Evaluation only

对应阶段：

```text
F16_evaluation
```

指标：

- `exact_match`
- `precision`
- `recall`
- `f1`
- LLM Judge 指标仍由 `signpost.evaluation.llm_judge` 负责。

注意：

- F16 和 LLM-as-Judge 是评估成本，不算入被评测方法的 online retrieval latency。

## 3. Query 指标怎么使用

输入可以是 F16 prediction JSONL，也可以是更丰富的 query log JSONL。每行至少建议包含：

```json
{
  "question_id": "q1",
  "question": "...",
  "answer": "gold answer",
  "prediction": "<answer>...</answer>",
  "metadata": {"method": "signpost", "dataset": "agriculture"},
  "trace": [],
  "citations": [],
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_seconds": 0
}
```

运行：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

如果输入不是标准 F16 schema，可以加 `--normalize`：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.raw.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json \
  --normalize
```

输出核心结构：

```json
{
  "num_queries": 100,
  "quality": {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
  "cost": {
    "totals": {"total_tokens": 0.0, "tool_calls": 0.0},
    "means": {"latency_seconds": 0.0},
    "p95": {"latency_seconds": 0.0}
  },
  "retrieval": {
    "num_evidence_scored": 0,
    "recall_at_k": {},
    "mrr": 0.0
  }
}
```

### 3.1 弱证据指标

如果 prediction 行里包含 gold evidence 字段，脚本会自动计算：

- `recall@1`
- `recall@3`
- `recall@5`
- `recall@10`
- `mrr`

支持字段：

```text
gold_evidence
gold_evidence_ids
gold_chunk_ids
gold_doc_ids
supporting_facts
```

检索结果支持字段：

```text
retrieved_chunks
retrieved_chunk_ids
citations
evidence
contexts
```

这个指标是 weak evidence metric。它不能替代人工证据标注，但可以在数据库技术说明中补充非 LLM 的检索质量信号。

## 4. Stage timing 日志怎么记录

用 `time_stage.py` 包装任意阶段命令：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset agriculture \
  --stage F6_semantic_graph \
  --method-scope signpost_offline_index \
  --input-path datasets/processed/agriculture/chunks.jsonl \
  --output-path datasets/processed/agriculture/graph.semantic.llm.json \
  --log outputs/agriculture/logs/stage_timing.jsonl \
  -- \
  conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
    --namespace agriculture-llm \
    --chunks datasets/processed/agriculture/chunks.jsonl \
    --output datasets/processed/agriculture/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --progress-every 10 \
    --progress-file datasets/processed/agriculture/semantic_llm.progress.jsonl \
    --extractions-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
    --llm-retries 5 \
    --retry-sleep 3 \
    --llm-timeout 180
```

日志每行格式：

```json
{
  "dataset": "agriculture",
  "method": "",
  "stage": "F6_semantic_graph",
  "method_scope": "signpost_offline_index",
  "input_path": "...",
  "output_path": "...",
  "command": ["conda", "run", "..."],
  "started_at": 0.0,
  "finished_at": 0.0,
  "wall_time_seconds": 0.0,
  "status": "ok",
  "return_code": 0
}
```

如果命令失败，`status = "failed"`，`return_code` 保留原命令退出码。

## 5. Offline index 和图结构指标怎么使用

汇总 stage log、F6 extraction cache 和 graph JSON：

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/agriculture/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/agriculture/graph.semantic.llm.json \
  --graph datasets/processed/agriculture/graph.structure.json \
  --graph datasets/processed/agriculture/graph.sequence.json \
  --graph datasets/processed/agriculture/graph.unified.json \
  --output outputs/agriculture/metrics/index_metrics.json
```

输出包含：

- 每个 stage 的运行次数、状态、耗时、LLM calls、tokens、disk bytes。
- F6 每个 chunk 抽取出的实体数、关系数、估算 LLM calls。
- 每个 graph 的节点数、边数、节点类型、边类型比例、平均度、连通分量。

图结构指标的解释：

| 指标 | 含义 |
| --- | --- |
| `nodes` / `edges` | 图规模。 |
| `node_counts` | entity、chunk、summary 等节点数量。 |
| `edge_counts` | semantic、structure、sequence、source 等边数量。 |
| `edge_type_ratio` | 各类边占比，用于说明多视图构成。 |
| `degree.mean` / `degree.p95` | 节点连接密度和尾部 hub 情况。 |
| `connected_components.count` | 图是否碎片化。 |
| `connected_components.largest` | 最大连通块大小。 |

## 6. 成本-质量派生指标怎么使用

先为每个 method 准备一个 summary JSON 对象。可以直接使用 `query_metrics.py` 输出，再补 `method`、`dataset` 和 offline 字段：

```json
[
  {
    "method": "hybrid",
    "dataset": "agriculture",
    "num_queries": 100,
    "quality": {"exact_match": 0.42, "f1": 0.55},
    "cost": {"means": {"total_tokens": 5000, "latency_seconds": 5, "llm_calls": 1}},
    "offline": {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "wall_time_seconds": 600}
  },
  {
    "method": "signpost",
    "dataset": "agriculture",
    "num_queries": 100,
    "quality": {"exact_match": 0.55, "f1": 0.66},
    "cost": {"means": {"total_tokens": 12000, "latency_seconds": 35, "llm_calls": 4}},
    "offline": {"input_tokens": 1000000, "output_tokens": 200000, "llm_calls": 8642, "wall_time_seconds": 10800}
  }
]
```

运行：

```bash
conda run -n signpost-re python -m signpost.benchmark.cost_quality \
  --methods outputs/agriculture/metrics/method_summaries.json \
  --workload-sizes 10 50 100 500 1000 5000 10000 \
  --output outputs/agriculture/metrics/cost_quality.json
```

输出指标：

| 指标 | 含义 |
| --- | --- |
| `amortized_time_seconds` | `offline_wall_time_seconds / N + mean_online_latency`。 |
| `amortized_tokens` | `offline_tokens / N + mean_online_tokens`。 |
| `amortized_llm_calls` | `offline_llm_calls / N + mean_online_llm_calls`。 |
| `break_even_queries_tokens` | 以 token 为成本时，方法追平 baseline 所需 query 数。若不存在则为 `null`。 |
| `break_even_queries_time` | 以时间为成本时，方法追平 baseline 所需 query 数。若不存在则为 `null`。 |
| `cost_per_extra_correct_tokens` | 每多答对一个问题的额外 token 成本。 |
| `pareto` | 质量更高且在线 token 不更高的方法前沿。 |

注意：

- `break_even` 只有当 baseline 的在线单题成本高于当前方法时才存在。
- Signpost vs BM25/Dense/Hybrid 很可能没有纯成本 break-even，这不是 bug，而是实验结论。

## 7. 推荐输出目录

建议每个数据集都使用下面结构：

```text
outputs/{dataset}/
  logs/
    stage_timing.jsonl
    {method}.query.jsonl
  predictions/
    {method}.jsonl
  metrics/
    {method}.query_metrics.json
    index_metrics.json
    method_summaries.json
    cost_quality.json
```

真实数据工件仍保留在：

```text
datasets/processed/{dataset}/
```

## 8. 与现有 F16 的关系

原有 F16：

- `signpost.evaluation.evaluate_basic`
- `signpost.evaluation.llm_judge`
- `signpost.evaluation.validate_predictions`

仍然保留。

新增 benchmark 指标层不会替代 F16，而是扩展技术说明实验需要的成本、延迟、证据和图结构指标。推荐流程是：

1. 先用 F16 校验 prediction 格式和基础 EM/F1。
2. 再用 `query_metrics.py` 生成统一质量+成本 summary。
3. 用 `index_metrics.py` 生成离线索引和图结构 summary。
4. 用 `cost_quality.py` 生成技术说明成本分析图表所需数据。
