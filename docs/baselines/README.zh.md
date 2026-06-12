# Baseline 文档索引与统一实验口径

本文档是 baseline 文档入口。每个 baseline 后续都放在独立目录中，避免外部官方代码、适配器、H200 命令和结果说明混在一起。

## 1. 目录约定

```text
docs/baselines/
  README.zh.md
  final_baseline_selection_zh.md
  external_baseline_evaluation_audit_zh.md
  external_baseline_metrics_prompts_zh.md
  baseline_control_requirements_and_handoff.zh.md
  in_house_controls_zh.md
  vanilla_llm/
    runbook.zh.md
  hybrid_rag/
    runbook.zh.md
  cluerag_baseline_zh.md
  cluerag_environment_h200_zh.md
```

后续接入外部 baseline 时，建议使用同样结构：

```text
docs/baselines/<method>/
  runbook.zh.md
  adaptation_notes.zh.md
  environment_h200.zh.md
  troubleshooting.zh.md
```

当前 Clue-RAG 文档仍保留在旧文件名，后续可以迁移到 `docs/baselines/cluerag/`。

外部 baseline 官方数据格式、指标计算方式和实体抽取依赖见：

```text
docs/baselines/external_baseline_evaluation_audit_zh.md
```

外部 baseline 的技术说明/代码指标公式可复现性，以及各 prompt 的用途和输出格式见：

```text
docs/baselines/external_baseline_metrics_prompts_zh.md
```

更详细的中文指标解释和能确认的原始完整 prompt 见：

```text
docs/baselines/external_baseline_metrics_prompts_full_zh.md
```

后续新对话或新 baseline 接入的控制变量要求与交接说明见：

```text
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

## 2. 统一公平性口径

正式实验采用以下口径：

```text
1. F3/F3.5/F4 是共享数据准备，不作为方法离线成本。
2. F6 chunk-level entity/relation extraction 是共享语义标注阶段。
3. F6 的时间、token、调用次数必须记录，但默认不计入各方法离线成本。
4. 需要实体或关系输入的方法，统一复用 F6 产物。
5. 方法离线成本从该方法自己的图组织、索引构建、同步或物化阶段开始计算。
6. 在线阶段统一记录 latency、retrieval latency、LLM calls、input/output/total tokens、tool calls。
7. 所有 baseline 的 final generation 使用 Signpost 的 evidence-grounded 回答约束；若 baseline 有自己的输出格式，保留其输出格式，只迁移回答约束。
```

第 4 点是强约束。除非某个官方 baseline 无法接入共享实体/关系，否则不应让不同方法各自重新抽取实体后再比较。若必须使用官方抽取器，需要在该 baseline 的 method card 和技术说明限制中明确说明。

## 3. 当前优先闭环顺序

先完成两个 in-house 控制组：

```text
1. vanilla_llm
2. hybrid_rag
```

原因：

```text
vanilla_llm: 无检索下界，验证模型参数知识是否足够。
hybrid_rag: 强 flat retriever 控制组，验证 Signpost 不是只赢弱检索。
```

这两个 baseline 的代码简单、依赖少、输出 schema 已统一。它们闭环后，可以作为外部 baseline 的验收模板。

## 4. 每个 baseline 必须产出的文件

每个方法最终都必须生成：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
outputs/<dataset>/metrics/<method>.basic_eval.json
outputs/<dataset>/metrics/<method>.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

只生成官方结果文件不算闭环。必须能进入 Signpost 的统一评估与成本汇总。

## 5. 本地到 H200 的验收门槛

每个 baseline 搬到 H200 前，至少完成：

```text
1. 本地 fake/mock 单测通过。
2. 本地 legal_test LIMIT=3 smoke 通过。
3. 输出 prediction schema 能被 basic_eval 和 query_metrics 消费。
4. H200 所需环境变量、依赖、输入文件、运行命令已经写入 runbook。
5. 明确哪些阶段计入该 baseline 离线成本，哪些只记录不计入。
```

H200 上只做 smoke 和正式运行，不做复杂开发。
