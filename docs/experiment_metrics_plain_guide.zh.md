# ICDE 实验指标人话版说明

这份文档不按代码结构讲，而是按“实验到底想证明什么”来讲。你可以先只看这份，等真正要查字段细节时，再看 `experiment_metrics_guide.zh.md`。

## 0. 先说结论：我们到底要测什么？

这篇论文不是只要测“答案对不对”。我们还要回答审稿人最可能问的几个问题：

1. **质量问题**：Signpost 答案是不是更准？
2. **在线成本问题**：Signpost 回答一个问题时，是不是用了很多 token、工具调用和时间？
3. **离线成本问题**：Signpost 建索引是不是很贵？贵在哪里？
4. **图有没有意义**：多视图图是不是真的形成了可导航结构？
5. **成本值不值**：多花的离线索引成本，换来了多少质量提升？如果问很多问题，成本能不能被摊薄？

所以指标看起来多，其实可以分成五组：

| 你想回答的问题 | 看哪类指标 |
| --- | --- |
| 答案准不准？ | 质量指标 |
| 每个问题跑得贵不贵？ | 在线成本指标 |
| 建索引贵不贵？ | 离线索引指标 |
| 图结构长什么样？ | 图结构指标 |
| 质量和成本怎么权衡？ | 成本-质量指标 |

你不用一开始全看。真正跑实验时，优先看：

```text
F1 / EM
latency_seconds.mean / latency_seconds.p95
total_tokens.mean
tool_calls.mean
offline wall_time_seconds
semantic estimated_llm_calls
nodes / edges / edge_type_ratio
amortized_tokens
break_even_queries
```

## 1. 第一组：质量指标

### 1.1 它回答什么问题？

质量指标回答：

> 这个方法答案到底答得怎么样？

例如比较：

```text
BM25 vs Dense vs Hybrid vs GraphSearch vs Signpost
```

如果 Signpost 质量不提升，那后面的成本分析就很难讲。

### 1.2 主要看哪些数？

| 指标 | 人话解释 | 越大越好？ |
| --- | --- | --- |
| `exact_match` / `EM` | 预测答案和标准答案是否完全一样 | 是 |
| `precision` | 预测答案里有多少内容是对的 | 是 |
| `recall` | 标准答案里的内容，预测覆盖了多少 | 是 |
| `f1` | precision 和 recall 的综合分 | 是 |
| `LLM Judge correctness` | 让 LLM 判断答案是否正确 | 是 |
| `groundedness` | 答案有没有被证据支持 | 是 |

### 1.3 什么时候看 EM，什么时候看 F1？

如果问题答案很短，比如：

```text
答案：水稻
```

EM 比较有意义。

如果答案是解释性文本，比如：

```text
答案：Signpost 通过语义视图、结构视图和顺序视图联合组织文档...
```

EM 会过于严格，这时更应该看 F1 或 LLM Judge。

### 1.4 怎么跑？

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

输出里看：

```json
{
  "quality": {
    "exact_match": 0.42,
    "precision": 0.58,
    "recall": 0.61,
    "f1": 0.59
  }
}
```

论文表里可以写：

| Method | EM | F1 |
| --- | ---: | ---: |
| BM25 | 0.32 | 0.46 |
| Hybrid | 0.39 | 0.52 |
| Signpost | 0.48 | 0.61 |

## 2. 第二组：在线成本指标

### 2.1 它回答什么问题？

在线成本指标回答：

> 一个问题来了以后，这个方法为了回答它花了多少资源？

这里的“在线”指的是查询时发生的事情，不包括提前建索引。

比如用户问：

```text
某个农业政策如何影响水稻种植？
```

从系统开始处理这个问题，到最终写出答案，这段过程就是 online query。

### 2.2 主要看哪些数？

| 指标 | 人话解释 | 越小越好？ |
| --- | --- | --- |
| `latency_seconds` | 回答一个问题花了几秒 | 是 |
| `latency_seconds.p95` | 最慢的 5% 问题大概有多慢 | 是 |
| `input_tokens` | 发给 LLM 的输入 token | 是 |
| `output_tokens` | LLM 生成的输出 token | 通常是 |
| `total_tokens` | 输入 + 输出 token | 是 |
| `llm_calls` | 一个问题调用了几次 LLM | 是 |
| `tool_calls` | agent 调了几次工具 | 是 |
| `retrieved_chunks` | 检索拿回了多少 chunk | 不一定，过多通常不好 |
| `read_file_calls` | 回读原文几次 | 不一定，过多通常说明搜索不聚焦 |
| `graph_ppr_calls` | 在线 PPR/路标推荐跑了几次 | 不一定 |

### 2.3 mean、median、p95 是什么？

假设 100 个问题的耗时如下：

```text
大部分问题 5 秒
少数问题 60 秒
```

只看平均值可能看不出尾部问题很慢。所以要同时看：

| 指标 | 人话解释 |
| --- | --- |
| `mean` | 平均每个问题多慢 |
| `median` | 中位数，典型问题多慢 |
| `p95` | 最慢 5% 的问题多慢 |

数据库/系统论文很重视 p95，因为真实系统里“少数特别慢的问题”也很重要。

### 2.4 怎么跑？

同样用：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

输出里看：

```json
{
  "cost": {
    "means": {
      "latency_seconds": 35.2,
      "total_tokens": 12000,
      "tool_calls": 8
    },
    "p95": {
      "latency_seconds": 80.5,
      "total_tokens": 26000,
      "tool_calls": 15
    }
  }
}
```

论文里可以写：

| Method | Tokens / Query | Tool Calls / Query | Mean Latency | P95 Latency |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 5k | 1 | 6s | 10s |
| GraphSearch | 20k | 15 | 70s | 150s |
| Signpost | 14k | 9 | 45s | 90s |

这张表的核心不是证明 Signpost 比 Hybrid 便宜，而是看它是否比没有路标的 GraphSearch 更少盲搜。

## 3. 第三组：离线索引指标

### 3.1 它回答什么问题？

离线索引指标回答：

> Signpost 建图和建索引到底花了多少成本？

这是 ICDE 论文必须说清楚的地方。因为 Signpost 的语义视图需要 LLM 抽取实体关系，不能假装这部分是免费的。

### 3.2 哪些阶段算离线？

| 阶段 | 做什么 | 是否 Signpost 专属 |
| --- | --- | --- |
| F3/F3.5/F4 | 数据整理、文档解析、chunking | 共享预处理 |
| F5 | chunk 写入 ES，生成 embedding | BM25/Dense/Hybrid/Signpost 都可能用 |
| F6 | LLM 抽实体关系，建语义图 | Signpost/GraphRAG 类方法用 |
| F7 | 结构图/RAPTOR summary | Signpost 用 |
| F8 | 顺序图 | Signpost 用 |
| F9 | 合并统一图 | Signpost 用 |
| F10 | 图对象写入 ES | Signpost/图检索用 |

### 3.3 主要看哪些数？

| 指标 | 人话解释 |
| --- | --- |
| `wall_time_seconds` | 这个阶段跑了多久 |
| `semantic estimated_llm_calls` | F6 大概调用了多少次 LLM |
| `entities_per_chunk` | 每个 chunk 平均抽出多少实体 |
| `relations_per_chunk` | 每个 chunk 平均抽出多少关系 |
| `input_tokens/output_tokens` | 如果记录了，就是离线 LLM token 成本 |
| `disk_bytes` | 索引/图文件占多少空间，如果记录了 |

### 3.4 F6 的 LLM calls 怎么理解？

F6 对每个 chunk 做实体关系抽取。

如果：

```text
chunks = 4321
gleaning_rounds = 1
```

那么每个 chunk 至少：

```text
第一次抽取 1 次
补充抽取 gleaning 1 次
```

所以估算：

```text
estimated_llm_calls = 4321 * (1 + 1) = 8642
```

这就是为什么 F6 很贵。

### 3.5 怎么记录阶段耗时？

以后跑重要阶段时，建议用 `time_stage.py` 包一下。

例子：

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
    --extractions-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl
```

看起来长，但逻辑很简单：

```text
前半段：告诉 logger 这是哪个 dataset、哪个 stage、日志写到哪里
-- 后半段：真正要运行的原命令
```

### 3.6 怎么汇总离线指标？

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/agriculture/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/agriculture/graph.unified.json \
  --output outputs/agriculture/metrics/index_metrics.json
```

输出里重点看：

```json
{
  "stage_logs": [
    {
      "stages": {
        "F6_semantic_graph": {
          "wall_time_seconds": {"sum": 10800}
        }
      }
    }
  ],
  "semantic_extractions": [
    {
      "unique_chunks": 4321,
      "estimated_llm_calls": 8642
    }
  ]
}
```

## 4. 第四组：图结构指标

### 4.1 它回答什么问题？

图结构指标回答：

> Signpost 建出来的图到底是什么样？是不是足够连通？不同视图各占多少？

这类指标不是直接评估答案，而是帮助解释实验结果。

比如：

- Legal 文档结构很强，结构边很多，所以结构视图可能贡献大。
- Novel 文档顺序很重要，sequence 边可能有用。
- 如果图碎成很多小连通块，PPR 跨文档传播能力会变弱。

### 4.2 主要看哪些数？

| 指标 | 人话解释 |
| --- | --- |
| `nodes` | 图里有多少节点 |
| `edges` | 图里有多少边 |
| `node_counts` | entity/chunk/summary 等节点分别多少 |
| `edge_counts` | semantic/structure/sequence/source 等边分别多少 |
| `edge_type_ratio` | 每种边占比 |
| `degree.mean` | 平均每个节点连多少边 |
| `degree.p95` | 连接特别多的节点大概有多大 |
| `connected_components.count` | 图被分成多少个连通块 |
| `connected_components.largest` | 最大连通块有多大 |

### 4.3 怎么看？

如果看到：

```json
"edge_type_ratio": {
  "semantic_relation": 0.45,
  "structure": 0.25,
  "sequence": 0.20,
  "source": 0.10
}
```

说明这个图确实是多视图图，不是只有语义边。

如果看到：

```json
"connected_components": {
  "count": 500,
  "largest": 30
}
```

说明图很碎。后面如果检索效果不好，可以解释为图连通性不足。

如果看到：

```json
"connected_components": {
  "count": 3,
  "largest": 10000
}
```

说明大部分节点在一个大图里，PPR/图扩展更容易发挥作用。

## 5. 第五组：成本-质量指标

### 5.1 它回答什么问题？

成本-质量指标回答：

> Signpost 质量提升，值不值得这些成本？

这部分不是单个实验原始数据，而是根据前面的质量和成本进一步算出来的。

### 5.2 摊销成本是什么？

Signpost 建索引贵，但索引可以重复使用。

假设：

```text
建索引花 100 万 tokens
每个问题在线花 10 万 tokens
```

如果只问 1 个问题：

```text
平均每题成本 = 100万 + 10万 = 110万
```

如果问 100 个问题：

```text
平均每题成本 = 100万 / 100 + 10万 = 11万
```

所以：

```text
amortized_tokens = offline_tokens / query_count + online_tokens_per_query
```

这个数用来画：

```text
x 轴：未来会问多少问题
y 轴：平均摊销成本
```

### 5.3 break-even 是什么？

break-even 指：

> Signpost 前面建索引很贵，但如果它每个问题比某个 baseline 便宜，那么问到多少个问题后，总成本能追平 baseline？

公式：

```text
N* = (I_signpost - I_baseline) / (O_baseline - O_signpost)
```

但是要注意：

只有当：

```text
O_baseline > O_signpost
```

也就是 baseline 每个问题更贵时，才存在 break-even。

所以：

- Signpost vs GraphSearch：可能有 break-even。
- Signpost-Pruned vs Signpost-Full：可能有 break-even。
- Signpost vs BM25：通常没有纯成本 break-even。
- Signpost vs Dense/Hybrid：通常也不强调纯成本 break-even。

如果结果里是：

```json
"break_even_queries_tokens": null
```

意思不是代码错了，而是：

> 当前方法在线并不比 baseline 省 token，所以不存在纯 token 成本追平点。

### 5.4 Pareto frontier 是什么？

Pareto frontier 用来回答：

> 有没有某个方法，在质量更高的同时，成本也不更高？

如果方法 A：

```text
质量 >= 方法 B
成本 <= 方法 B
```

并且至少一个更好，那 A 就支配 B。

最后没有被别人支配的方法，就在 Pareto frontier 上。

论文图里可以画：

```text
x 轴：online tokens/query
y 轴：answer quality
```

如果 Signpost 在右上方，意思是：

```text
它更贵，但质量也更高
```

如果 Signpost-Pruned 往左移动且质量差不多，意思是：

```text
剪枝降低成本，同时基本保住质量
```

注意：你现在说过暂时不做剪枝，所以这里只是指标代码先支持，后面再用。

### 5.5 每多答对一个问题的额外成本

这个指标用来比较 Signpost 和轻量 baseline。

假设：

```text
Hybrid 答对 40 个
Signpost 答对 50 个
Signpost 多花 100 万 tokens
```

那么：

```text
每多答对一个问题的额外成本 = 100万 / 10 = 10万 tokens
```

这个指标适合回答：

> Signpost 比 Hybrid 多花了多少成本，换来了多少额外正确答案？

## 6. 最小可执行流程

如果你现在觉得很乱，可以先只按这个最小流程做。

### Step 1：跑某个方法的 prediction

比如已经有：

```text
outputs/agriculture/predictions/signpost.jsonl
```

### Step 2：算这个方法的 query 指标

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

先只看：

```text
quality.f1
quality.exact_match
cost.means.total_tokens
cost.means.latency_seconds
cost.p95.latency_seconds
cost.means.tool_calls
```

### Step 3：算这个数据集的图和离线指标

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/agriculture/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/agriculture/graph.unified.json \
  --output outputs/agriculture/metrics/index_metrics.json
```

先只看：

```text
F6 wall_time_seconds
estimated_llm_calls
graph nodes
graph edges
edge_type_ratio
connected_components.count
```

### Step 4：多个方法都有 summary 后，再算成本-质量

先准备：

```text
outputs/agriculture/metrics/method_summaries.json
```

里面放每个方法一条 summary。

然后：

```bash
conda run -n signpost-re python -m signpost.benchmark.cost_quality \
  --methods outputs/agriculture/metrics/method_summaries.json \
  --output outputs/agriculture/metrics/cost_quality.json
```

先只看：

```text
amortized_tokens
break_even_queries_tokens
pareto
cost_per_extra_correct_tokens
```

## 7. 最推荐你现在先看的 10 个数

如果只能看 10 个数，就看这些：

| 顺序 | 指标 | 为什么看 |
| --- | --- | --- |
| 1 | `quality.f1` | 答案整体质量 |
| 2 | `quality.exact_match` | 严格答案正确率 |
| 3 | `cost.means.total_tokens` | 每题平均 token 成本 |
| 4 | `cost.means.latency_seconds` | 每题平均耗时 |
| 5 | `cost.p95.latency_seconds` | 尾延迟 |
| 6 | `cost.means.tool_calls` | agent 搜索是否盲目 |
| 7 | `F6 wall_time_seconds` | 语义图构建耗时 |
| 8 | `semantic estimated_llm_calls` | 离线 LLM 抽取成本 |
| 9 | `graph edge_type_ratio` | 多视图图是否真的建出来 |
| 10 | `amortized_tokens` | 索引成本摊到每题后是多少 |

这 10 个数基本能支撑论文的主线：

> Signpost 通过更贵的离线多视图图索引，换取更好的答案质量和更有指导性的在线 agent 检索；它的成本需要被量化，并随查询量摊销分析。

## 8. 不要一开始就纠结的指标

这些指标现在可以先放后面：

| 指标 | 为什么可以晚点看 |
| --- | --- |
| `p90` | 有 p95 就够先写系统尾延迟 |
| `degree.p95` | 只有分析图结构异常时再看 |
| `cost_per_extra_correct_tokens` | 需要多个方法结果齐了才有意义 |
| `Pareto frontier` | 需要多个方法和预算点齐了才有意义 |
| `break_even` | 主要用于 Signpost vs GraphSearch，不适合所有 baseline |
| `weak evidence MRR` | 需要 gold evidence 或 weak evidence 字段，否则没有结果 |

## 9. 一句话记忆版

如果你只想记住最核心逻辑：

```text
质量指标：证明 Signpost 答得更好。
在线成本：证明 Signpost 回答问题时花了多少 token、时间和工具调用。
离线成本：证明 Signpost 建索引有多贵，尤其 F6 语义抽取。
图结构指标：证明我们确实建了一个多视图可导航图。
成本-质量指标：证明质量提升和成本之间的 trade-off 怎么算。
```

