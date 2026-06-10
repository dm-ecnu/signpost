# In-house 控制组 Baseline：Vanilla LLM 与 Hybrid RAG

本文档说明已经固化的两个 in-house 控制组 baseline。更细的本地/H200 操作命令分别见：

```text
docs/baselines/vanilla_llm/runbook.zh.md
docs/baselines/hybrid_rag/runbook.zh.md
```

两个控制组 baseline：

1. `vanilla_llm`：无检索下界。
2. `hybrid_rag`：BM25+dense flat retrieval 控制组。

旧入口 `vanilla_rag` 保留兼容，但正式论文和正式实验使用 `hybrid_rag`。

## 1. 文件目录结构

```text
signpost/baselines/
  common.py              # baseline 公共 schema、query log、成本字段、context 拼接
  vanilla_llm.py         # Vanilla LLM 控制组
  vanilla_rag.py         # flat chunk RAG 实现，保留旧入口
  hybrid_rag.py          # Hybrid RAG alias，复用 vanilla_rag 实现但输出 method=hybrid_rag

scripts/baselines/
  run_baseline_method.sh # 统一 shell 入口：运行、评估、汇总
  run_vanilla_llm.py     # python -m 入口
  run_vanilla_rag.py     # 旧 python -m 入口
  run_hybrid_rag.py      # 正式 Hybrid RAG python -m 入口

tests/
  test_baselines.py      # fake LLM schema 测试，不访问真实模型服务

docs/baselines/
  final_baseline_selection_zh.md # 最终 baseline 选择
  in_house_controls_zh.md        # 本文档
  vanilla_llm/runbook.zh.md      # Vanilla LLM 独立操作手册
  hybrid_rag/runbook.zh.md       # Hybrid RAG 独立操作手册
```

`baselines/` 根目录保留给外部 baseline 官方仓库或适配说明。in-house 控制组属于 Signpost 代码包本身，不需要复制到 `baselines/` 目录。

## 2. 统一输入输出

两个控制组都读取同一套 processed dataset：

```text
datasets/processed/<dataset>/questions.jsonl
datasets/processed/<dataset>/chunks.jsonl      # Hybrid RAG 需要
```

统一输出：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
outputs/<dataset>/metrics/<method>.basic_eval.json
outputs/<dataset>/metrics/<method>.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

其中 `<method>` 为：

```text
vanilla_llm
hybrid_rag
```

旧的 `vanilla_rag` 仍可运行，但不建议进入正式论文主表。

## 2.1 统一成本口径

两个 in-house 控制组遵循全局口径：

```text
F6 chunk-level entity/relation extraction 作为共享语义标注阶段，只记录不计入方法离线成本。
需要实体/关系的外部 baseline 后续统一复用 F6 产物。
vanilla_llm 不使用任何离线索引。
hybrid_rag 只使用 F5 chunk index，不使用 F6 实体/关系。
```

在线阶段必须记录：

```text
每次检索时间、LLM 调用次数、input/output/total tokens、tool calls、总 latency。
```

## 3. Baseline 1：Vanilla LLM

### 3.1 实验目的

`vanilla_llm` 不检索任何外部文档，只把问题直接发给本地 chat model。

它回答的问题是：

```text
模型参数知识和直接生成能力本身是否足以回答 Agriculture/Legal 的任务？
```

它是无检索下界，不产生 evidence recall，也不会有 `retrieved_chunks`。

### 3.2 覆盖文件

```text
signpost/baselines/vanilla_llm.py
scripts/baselines/run_vanilla_llm.py
scripts/baselines/run_baseline_method.sh
```

### 3.3 运行命令

本地或 H200 smoke：

```bash
LIMIT=3 scripts/baselines/run_baseline_method.sh vanilla_llm legal_test legal_test
```

正式 Agriculture：

```bash
scripts/baselines/run_baseline_method.sh vanilla_llm agriculture agriculture
```

正式 Legal：

```bash
scripts/baselines/run_baseline_method.sh vanilla_llm legal legal
```

## 4. Baseline 2：Hybrid RAG

### 4.1 实验目的

`hybrid_rag` 是 flat chunk retrieval + generator 控制组，正式设置为：

```text
BM25 + dense vector hybrid retrieval
top-k chunk context
single generator call
no graph objects
no signpost metadata
no agentic multi-step search
```

它回答的问题是：

```text
Signpost 的收益是否只是来自更强的底层 BM25+dense 检索？
```

### 4.2 覆盖文件

```text
signpost/baselines/vanilla_rag.py  # 实际 flat RAG 实现
signpost/baselines/hybrid_rag.py   # 正式 alias，输出 method=hybrid_rag
scripts/baselines/run_hybrid_rag.py
scripts/baselines/run_baseline_method.sh
```

### 4.3 正式实验口径

正式结果必须使用：

```bash
USE_ES=1
MODE=hybrid
EMBEDDING_PROVIDER=ecnu
```

这里的 `ecnu` 是历史 provider 名称。在 H200 上它通过环境变量指向本地 embedding 服务：

```text
http://localhost:8001/v1/embeddings
model=/data/srl/nemotron-8b
```

`hybrid_rag` 复用 F5 chunk index，因此对应的离线成本包含：

```text
F5_chunk_index
```

它不使用 F6-F12 的 semantic graph、unified graph 或 signpost artifacts。

### 4.4 运行命令

本地 smoke，不依赖 ES：

```bash
LIMIT=3 USE_ES=0 MODE=bm25 scripts/baselines/run_baseline_method.sh hybrid_rag legal_test legal_test
```

H200 smoke，使用 ES + local embedding：

```bash
LIMIT=3 USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu \
  scripts/baselines/run_baseline_method.sh hybrid_rag legal_test legal_test
```

正式 Agriculture：

```bash
USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu \
  scripts/baselines/run_baseline_method.sh hybrid_rag agriculture agriculture
```

正式 Legal：

```bash
USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu \
  scripts/baselines/run_baseline_method.sh hybrid_rag legal legal
```

可调参数：

```text
LIMIT=<n>                 # smoke 时限制问题数
TOP_K=5                   # 检索 chunk 数量
MAX_CONTEXT_TOKENS=3500   # 拼接给 generator 的最大上下文
USE_ES=0|1
MODE=bm25|dense|hybrid
EMBEDDING_PROVIDER=ecnu|hash
```

## 5. H200 环境配置

进入项目：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
```

加载 H200 本地模型服务环境：

```bash
set -a
source .env.h200
set +a
```

必要环境变量：

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

验证服务：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

正式 `hybrid_rag` 还要求该 dataset 已经有 chunk ES index。通常先跑：

```bash
scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>
```

如果 Signpost full pipeline 已经跑完，则 F5 chunk index 已存在，可以直接跑 `hybrid_rag`。

## 6. 本地测试

只跑 baseline 单测：

```bash
python -m pytest tests/test_baselines.py
```

只检查 shell 语法：

```bash
bash -n scripts/baselines/run_baseline_method.sh
```

CLI help：

```bash
python -m scripts.baselines.run_vanilla_llm --help
python -m scripts.baselines.run_hybrid_rag --help
```

## 7. 与旧 `vanilla_rag` 的关系

保留旧命令：

```bash
scripts/baselines/run_baseline_method.sh vanilla_rag <dataset> <namespace>
```

但正式论文和正式结果使用：

```bash
scripts/baselines/run_baseline_method.sh hybrid_rag <dataset> <namespace>
```

区别：

```text
vanilla_rag -> outputs/<dataset>/predictions/vanilla_rag.jsonl, metadata.method=vanilla_rag
hybrid_rag  -> outputs/<dataset>/predictions/hybrid_rag.jsonl,  metadata.method=hybrid_rag
```

这样既不破坏已有测试和历史实验，又保证论文表格和输出文件名一致。

## 8. 下一步

完成本控制组后，外部 baseline 的推荐接入顺序是：

```text
1. Clue-RAG
2. AGRAG
3. LinearRAG
4. HiPRAG
5. GraphRAG-R1
```

每个外部 baseline 必须先完成：

```text
legal_test LIMIT=3 smoke
Agriculture LIMIT=3 smoke
统一 prediction schema 转换
basic_eval/query_metrics/method_summary/cost_quality
```

smoke 成功后再迁移到 H200 full run。
