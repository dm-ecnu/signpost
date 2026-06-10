# Signpost 消融实验说明

本文档说明 `scripts/run_signpost_ablation_suite.sh` 实际比较哪些 Signpost variant、是否会重新执行离线建图、以及运行过程中应如何判断进度。

## 1. 当前脚本行为

入口：

```bash
scripts/run_signpost_ablation_suite.sh <dataset> [namespace]
```

例如：

```bash
scripts/run_signpost_ablation_suite.sh agriculture agriculture
```

该脚本不会重新执行 F3--F10 离线索引构建，而是顺序调用：

```bash
scripts/run_signpost_method.sh agriculture full agriculture
scripts/run_signpost_method.sh agriculture no_offline agriculture
scripts/run_signpost_method.sh agriculture no_online agriculture
scripts/run_signpost_method.sh agriculture no_semantic_cues agriculture
scripts/run_signpost_method.sh agriculture no_provenance_cues agriculture
scripts/run_signpost_method.sh agriculture no_vertical_cues agriculture
scripts/run_signpost_method.sh agriculture no_horizontal_cues agriculture
```

每个 variant 主要执行：

```text
F15_agent_batch_<variant>  # 在线 agent 检索与回答
F16_basic_eval_<variant>   # 基础答案评估
query_metrics              # query-level metric 汇总
method_summary             # 合并离线/在线成本
cost_quality               # 成本-质量表
```

因此，农业数据集上 `full` 跑完后出现：

```text
[signpost-method] dataset=agriculture namespace=agriculture variant=no_offline embedding=ecnu
```

表示正在跑第二个消融 `no_offline` 的在线 F15 阶段。

## 2. 是否会重跑实体抽取/语义图构建

不会。

`run_signpost_ablation_suite.sh` 只调用 `run_signpost_method.sh`。后者直接读取已经生成的 artifacts：

```text
datasets/processed/<dataset>/graph.unified.json
datasets/processed/<dataset>/chunks.jsonl
Elasticsearch indexes:
  signpost-<namespace>-chunks
  signpost-<namespace>-graph
```

它不会调用：

```text
F3_data_prepare
F3.5_parse_documents
F4_chunk_tree
F5_chunk_index
F6_semantic_graph_llm
F7_structure_graph
F8_sequence_graph
F9_unified_graph
F10_graph_es_sync
```

也就是说，已经跑过的 LLM 实体/关系抽取、结构图、序列图、统一图、ES 同步不会在消融实验中重新执行。消融实验复用同一份图索引，只在在线检索结果呈现给 agent 之前过滤不同类型的 signpost cues。

唯一的例外是：如果你手动重新运行 `scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>`，才会重新执行离线阶段。

## 3. 运行进度文件

每个 variant 都会生成独立文件：

```text
outputs/<dataset>/predictions/signpost.<variant>.jsonl
outputs/<dataset>/logs/signpost.<variant>.query.jsonl
outputs/<dataset>/metrics/signpost.<variant>.basic_eval.json
outputs/<dataset>/metrics/signpost.<variant>.query_metrics.json
```

例如：

```text
outputs/agriculture/logs/signpost.full.query.jsonl
outputs/agriculture/logs/signpost.no_offline.query.jsonl
```

`*.query.jsonl` 在该 variant 开始时会被清空/新建，然后每完成一个 question 追加一行。因此：

```text
文件出现
  说明该 variant 的 F15 已经启动。

行数增长
  说明 agent 正在正常处理 query。

长时间 0 行
  说明可能卡在第一个 query。
```

查看进度：

```bash
wc -l datasets/processed/agriculture/questions.jsonl
wc -l outputs/agriculture/logs/signpost.no_offline.query.jsonl
tail -n 1 outputs/agriculture/logs/signpost.no_offline.query.jsonl
```

持续观察：

```bash
watch -n 30 'date; wc -l outputs/agriculture/logs/signpost.no_offline.query.jsonl; tail -n 1 outputs/agriculture/logs/signpost.no_offline.query.jsonl'
```

## 4. Variant 定义

消融逻辑在：

```text
signpost/retrieval/signpost_variants.py
```

所有 variant 都先执行相同的底层检索：

```text
1. chunk search
2. summary search
3. entity/relation graph search
4. attach offline signposts
5. compute online signposts
```

然后在返回给 agent 之前，对结果中的 signpost 信息做过滤。因此这些消融比较的是“agent 能看到哪些导航线索”，不是重新训练或重新建图。

| Variant | 保留内容 | 移除内容 | 证明目的 |
|---|---|---|---|
| `full` | 完整 Signpost：offline signposts + online signposts | 无 | 主方法。验证完整多视图路标索引和在线 PPR 推荐的总体效果。 |
| `no_offline` | 底层检索结果、online signposts | 每个 item 的 `offline_signpost` 全部置空 | 验证离线路标元数据是否有价值。若性能下降，说明仅靠普通检索结果和在线推荐不足以提供可导航证据。 |
| `no_online` | 底层检索结果、offline signposts | group-level `online_signpost` 清空 | 验证在线 PPR/推荐线索是否有价值。若性能下降，说明离线路标仍需要按 query 动态组织。 |
| `no_semantic_cues` | provenance/vertical/horizontal offline cues | offline `semantic` cue 移除，同时 online signpost 清空 | 验证语义实体/关系线索的贡献。这里同时关闭 online，是因为 online PPR 主要依赖语义图连接；否则语义消融不干净。 |
| `no_provenance_cues` | semantic/vertical/horizontal cues、online signposts | offline `provenance` cue、`source_locates`、`source_chunk_ids` | 验证可追溯定位线索是否帮助 agent 精确 read source evidence。 |
| `no_vertical_cues` | semantic/provenance/horizontal cues、online signposts | offline `vertical` cue | 验证层级/父子摘要等纵向上下文导航是否有价值。 |
| `no_horizontal_cues` | semantic/provenance/vertical cues、online signposts | offline `horizontal` cue | 验证相邻 chunk、同层结构、顺序上下文等横向导航是否有价值。 |

## 5. 与论文故事的关系

这组消融用于回答三个问题：

```text
RQ-Ablation-1:
  Signpost 的收益是否来自离线索引中的路标元数据，而不是 agent 本身？
  对比 full vs. no_offline。

RQ-Ablation-2:
  在线 query-aware navigation 是否必要？
  对比 full vs. no_online。

RQ-Ablation-3:
  多视图 cues 是否只是堆砌？
  对比 full vs. no_semantic_cues / no_provenance_cues / no_vertical_cues / no_horizontal_cues。
```

其中 `no_semantic_cues` 预计影响较大，因为语义视图承担跨 chunk、跨文档、多跳连接；`no_provenance_cues` 预计主要影响 citation/read-file 精度；`no_vertical_cues` 和 `no_horizontal_cues` 用于区分层级导航和顺序/邻接导航的贡献。

## 6. 成本统计方式

`method_summary` 会把每个 variant 的在线阶段和同一套离线阶段合并：

```text
offline stages:
  F5_chunk_index
  F6_semantic_graph_llm
  F7_structure_graph
  F8_sequence_graph
  F9_unified_graph
  F10_graph_es_sync

online stage:
  F15_agent_batch_signpost.<variant>
```

因此每个 variant 的离线成本在 summary 中相同，在线成本不同。这是有意设计：

```text
离线成本：
  表示构建完整 Signpost index 的一次性投资。

在线成本：
  表示 agent 在不同 navigation cue 可见性下解决同一批问题的开销。
```

如果论文里要表达“去掉某类离线 cue 后能否节省索引构建成本”，那需要另做 construction-level ablation，例如真正不构建 semantic graph。当前 suite 不是 construction ablation，而是 navigation-cue visibility ablation。

## 7. 计时准确性与解释边界

当前 suite 的计时对“visibility ablation”是准确的，但不能解释为“去掉某类 cue 后系统真实能节省多少构建/检索时间”。

原因是所有 variant 都复用同一条检索流水线：

```text
chunk/search summary/search graph/search
attach offline signposts
compute online signposts
apply variant filter
agent read/search/generate
```

例如 `no_offline` 的实际含义是：

```text
仍然检索相同的 chunk / summary / graph candidates；
仍然在内部 attach offline signposts；
但在返回给 agent 之前把 item.offline_signpost 置空。
```

因此：

```text
可以据此回答：
  如果 agent 看不到离线路标元数据，答案质量和在线交互开销会怎样变化？

不能据此回答：
  如果系统从一开始就不构建离线路标索引，离线成本能省多少？
```

这种设计的优点是控制变量干净：底层检索候选、图索引、ES 索引、LLM、问题集合完全一致，只改变 agent 可见的 navigation cues。缺点是 `no_offline`、`no_online`、`no_semantic_cues` 等 variant 的在线 wall time 会包含一些随后被过滤掉的本地计算开销。

这个额外开销主要是本地图查询、字典查找和 PPR 计算，通常远小于 agent 的 LLM 调用和 read/search 循环。但如果论文要报告“去掉某类 cue 的真实性能节省”，不能使用当前 suite 的时间作为证据。

若需要严格的成本消融，应新增 construction-level ablation：

| 目标 | 需要新增的实验 | 解释 |
|---|---|---|
| 语义视图构建成本 | 真正跳过 F6 semantic graph LLM extraction，再重建 F9/F10 | 测 semantic view 对离线 token/time/disk 的贡献 |
| 在线 PPR 成本 | 在 `run_retrieval` 中提前跳过 `compute_online_signpost` | 测 query-aware online signpost 的纯在线开销 |
| provenance cue 成本 | 构建时不写 provenance/source_locates，检索时不 attach | 测 source tracing metadata 的空间/时间贡献 |

当前 ICDE 主文建议这样表述：

```text
We use visibility ablations to isolate the utility of each navigation cue under a fixed index and retrieval backend.
We report construction and amortization costs for the full Signpost index separately.
```

不要写成：

```text
no_offline saves offline construction cost
```

因为当前实现并没有真的省掉离线构建。

## 8. 正式运行建议

正式跑全量前建议先做小样本：

```bash
LIMIT=5 USE_ES=1 USE_LLM=1 scripts/run_signpost_ablation_suite.sh agriculture agriculture
```

小样本无异常后再跑全量：

```bash
USE_ES=1 USE_LLM=1 scripts/run_signpost_ablation_suite.sh agriculture agriculture
```

如果某个 variant 中断，可以只重跑该 variant：

```bash
USE_ES=1 USE_LLM=1 scripts/run_signpost_method.sh agriculture no_offline agriculture
```

已完成的其他 variant 不需要重跑。
