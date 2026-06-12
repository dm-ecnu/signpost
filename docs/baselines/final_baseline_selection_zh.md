# Signpost final Baseline 选择与可实现性核查

本文档固化当前版本的 baseline 决策。目标不是简单追新，而是让每个 baseline 对应一个明确的反事实问题：

- 无检索时，模型参数知识是否已经足够？
- 强 flat retriever 是否已经足够？
- 多粒度、层次化、拓扑化或 PPR 图检索是否已经足够？
- 在 flat retrieval 或 graph retrieval 上加入 agentic search 是否已经足够？
- 如果上述方法仍不足，Signpost 的 navigation-cue index 是否带来额外收益，并且这种收益是否体现在证据导航、在线成本和答案质量上？

## 1. 选择标准

外部方法 baseline 需要同时满足：

1. 真实存在并已被 peer reviewed。
2. 2026 届 A 会或同等顶级会议接收的长文。
3. arXiv 时间不早于 2025 年 7 月。
4. 代码开源，理论上可复现。
5. 能接入私有文档 QA 设置，至少能通过 wrapper 转换成统一输入输出。

`Vanilla LLM` 和 `Hybrid RAG` 是控制组，不适用第 2-4 条技术说明筛选标准，但必须保留，否则无法判断 Signpost 是否只是超过了弱下界。

## 2. 最终主表 Baseline

| 类别 | 方法 | arXiv >= 2025.07 | 2026 顶会长文 | 开源可复现 | 技术说明链接 | GitHub / 代码 | 实验目的 | 当前可实现性判断 |
|---|---|---:|---:|---:|---|---|---|---|
| No retrieval control | Vanilla LLM / CoT optional | N/A | N/A | yes, in-house | N/A | 本项目 | 检验模型参数知识和直接生成能力是否足够。 | 已实现。CoT 可作为可选诊断，不建议主表单独多一行。 |
| Flat retrieval control | Hybrid RAG | N/A | N/A | yes, in-house | N/A | 本项目 | 控制 BM25+dense flat retrieval 能力，证明 Signpost 不是靠更强底层 retriever。 | 基本已实现；现有代码名是 `vanilla_rag`，正式实验应固定 `MODE=hybrid USE_ES=1`，并在技术说明中命名为 `Hybrid RAG`。 |
| Multi-granularity GraphRAG | Clue-RAG | yes, 2025.07 | yes, ICDE 2026 | yes | https://arxiv.org/abs/2507.08445 | https://github.com/Feesuu/ClueRAG | 检验 chunk / knowledge unit / entity 多粒度图索引是否已经足够。 | 推荐主 baseline。需要把 `documents.jsonl` 或统一 chunks 转成 Clue-RAG 官方输入，记录其 graph build token/time。 |
| Topology / hierarchy GraphRAG | LinearRAG | yes, 2025.10 | yes, ICLR 2026 main poster | yes | https://arxiv.org/abs/2510.10114 | https://github.com/DEEP-PolyU/LinearRAG | 检验更好的层次化或拓扑化组织是否已经足以解决多跳私有知识库检索。 | 推荐主 baseline。需要先做 legal_test smoke，确认官方 pipeline 能换成本地 Llama/Nemotron。 |
| PPR / associative GraphRAG | AGRAG | yes, 2025.11 | yes, ICDE 2026 | yes | https://arxiv.org/abs/2511.05549 | https://github.com/Wyb0627/AGRAG | 检验 PPR、MCMI 子图和高阶关联检索能否替代 Signpost 的导航机制。 | 推荐主 baseline。与 Signpost 关系近，必须比较；需要记录 TF-IDF/entity extraction、relation extraction、MCMI retrieval 的离线和在线成本。 |
| Agentic RAG | HiPRAG | yes, 2025.10 | yes, ICLR 2026 main poster | yes, with engineering risk | https://arxiv.org/abs/2510.07794 | https://github.com/qualidea1217/HiPRAG | 检验在 flat retrieval 接口上加入多步 agentic search 是否足以解决复杂查询执行。 | 学术资格合格，但默认环境是 Search-R1 / Wikipedia corpus / e5 retriever。必须适配到 Agriculture/Legal 私有语料检索器；如果无法稳定适配，不应硬放主表。 |
| Agentic GraphRAG | GraphRAG-R1 | yes, 2025.07 | yes, WWW 2026 | yes, with engineering risk | https://arxiv.org/abs/2507.23581 | https://github.com/ycygit/GraphRAG-R1 | 检验图上 agentic traversal / RL GraphRAG 是否已经解决 graph-based multi-step retrieval。 | 学术资格合格，有代码和权重，但需要验证私有 KG / 文档图接入。如果只能在官方 Wiki KG 上跑，不构成公平 baseline。 |
| Ours | Signpost | N/A | 本文 | yes, in-house | 本文 | 本项目 | 主方法：navigation-cue index + signpost-enriched agent observation + evidence navigation / online efficiency analysis。 | Signpost 主链路和消融已接通；继续用同一套 H200 本地模型跑正式结果。 |

## 3. 人工核验链接

| 方法 | 录用核验 | 代码/权重核验 |
|---|---|---|
| LinearRAG | https://openreview.net/forum?id=mCtfkypdm6 | https://github.com/DEEP-PolyU/LinearRAG |
| Clue-RAG | https://icde2026.github.io/program_details.html | https://github.com/Feesuu/ClueRAG |
| AGRAG | https://icde2026.github.io/program_details.html | https://github.com/Wyb0627/AGRAG |
| HiPRAG | https://openreview.net/forum?id=Gt4v9WBPzm | https://github.com/qualidea1217/HiPRAG |
| GraphRAG-R1 | https://doi.org/10.1145/3774904.3792589 | https://github.com/ycygit/GraphRAG-R1 and https://huggingface.co/yuchuanyue/GraphRAG-R1 |

核验时优先看官方会议页或 OpenReview/ACM DOI 页面，其次看 arXiv 和 GitHub。GitHub README 的会议标签只能作为辅助证据。

## 4. 不进入主表的方法

| 方法 | 处理方式 | 原因 |
|---|---|---|
| LightRAG | Related Work only | arXiv 2024.10，不满足 2025 下半年之后的硬条件。 |
| RAPTOR | Related Work only | 时间更早，且仅作为层次化检索经典参考。 |
| LeanRAG | 暂不进入主表 | 当前主表已有 Clue-RAG、LinearRAG、AGRAG 覆盖 GraphRAG 关键变量；LeanRAG 不再优先。 |
| A-RAG | 工程 fallback only | 缺少明确 2026 A 会长文录用证据。 |
| Youtu-GraphRAG | GraphRAG-R1 fallback | ICLR 2026 与开源已确认，但与 GraphRAG-R1 同属 Agentic GraphRAG slot。主表只保留一个，避免 baseline 失衡。 |
| Graph-R1 | GraphRAG-R1 fallback | 与 GraphRAG-R1 名称相近但不是同一工作；若 GraphRAG-R1 无法接入私有图，可再核验 Graph-R1 的 inference 可行性。 |
| E2RAG | Related Work only | 叙事文本 temporal-causal KG-RAG，任务域与 Agriculture/Legal 私有文档 QA 不完全匹配。 |
| BRINK | Related Work / evaluation reference | benchmark/evaluation，不是完整方法 baseline。 |
| Wikontic | Related Work only | KG construction 工作，不是完整 RAG QA baseline。 |
| mKG-RAG | Related Work only | 多模态 KG-RAG，与当前纯文本语料不匹配。 |

## 5. 公平对比口径

所有方法必须满足同一实验口径：

1. 使用同一批数据集：`Agriculture-full` 和 `Legal-full` 为主；如果 Legal-full 外部 baseline 成本无法承受，至少要完成 Legal smoke 并在技术说明中说明限制。
2. 使用同一批问题：`datasets/processed/<dataset>/questions.jsonl`。
3. 外部 baseline 允许使用自己的 chunk/index pipeline，但其 preprocessing、index build、LLM extraction、embedding、storage 都计入该 baseline 的 offline cost。
4. 外部 baseline 不允许使用 Signpost 的 signpost annotations、online signpost recommendations 或 enriched observation。
5. 所有方法最终输出统一 prediction schema：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
outputs/<dataset>/metrics/<method>.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

6. 正式 H200 实验必须使用同一套本地服务：

```text
Chat:      http://localhost:8000/v1, model=/data/srl/Llama-3.3-70B-FP8
Embedding: http://localhost:8001/v1/embeddings, model=/data/srl/nemotron-8b
```

不得混用 ECNU 或外部 API，否则时间、失败率、token 和延迟不可比。

## 6. 对现有 in-house baseline 的影响

目前已经完成：

- baseline harness；
- `Vanilla LLM`；
- `Vanilla RAG`；
- `run_baseline_method.sh`；
- 统一输出 schema 和本地测试。

这些代码不需要推倒重写，但需要做两个口径调整：

1. **技术说明命名改为 Hybrid RAG。**
   现有 `vanilla_rag` 在 `USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu` 时，本质就是 BM25+dense hybrid retrieval + generator。正式实验和技术说明表格应命名为 `Hybrid RAG`。

2. **代码可以保留 `vanilla_rag` 入口，但建议增加 alias。**
   为避免历史脚本断裂，可以保留：

```text
scripts/baselines/run_baseline_method.sh vanilla_rag <dataset> <namespace>
```

   同时后续增加：

```text
scripts/baselines/run_baseline_method.sh hybrid_rag <dataset> <namespace>
```

   该 alias 内部复用 `vanilla_rag` 实现，但输出 method name 改为 `hybrid_rag`，文件名改为 `hybrid_rag.jsonl`。这样技术说明表格、metrics 文件和方法名一致。

本地 smoke 可以继续用 `USE_ES=0 MODE=bm25`，但正式结果必须用 `USE_ES=1 MODE=hybrid`。

## 7. 建议接入顺序

不要同时铺开所有外部 baseline。建议顺序：

1. 固化 in-house 控制组：`vanilla_llm` 和 `hybrid_rag` alias。
2. 接 `Clue-RAG`：它与 Signpost 最接近，也是 ICDE 2026 同届同会议，优先级最高。
3. 接 `AGRAG`：检验 PPR/MCMI 图检索是否能替代 Signpost。
4. 接 `LinearRAG`：覆盖 topology/hierarchy slot。
5. 只在前三个图 baseline 完成后，再接 `HiPRAG`。
6. 最后接 `GraphRAG-R1`；如果私有图适配成本过高，用 `Youtu-GraphRAG` 或 `Graph-R1` 作为 fallback。

每接一个 baseline，都必须先完成：

```text
legal_test --limit 3 smoke
Agriculture --limit 3 smoke
统一 prediction schema 转换
basic_eval/query_metrics/method_summary/cost_quality
```

smoke 成功后再上 Agriculture-full。不要在 H200 上临时开发复杂 wrapper；本地先完成 wrapper 和小样本格式测试，再迁移到 H200 跑真实模型和真实时间。
