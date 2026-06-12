# Hybrid RAG Baseline Runbook

`hybrid_rag` 是正式技术说明中的 flat retrieval 控制组。它使用 BM25+dense hybrid chunk retrieval，再把 top-k chunks 拼接给 chat model 生成答案。

旧入口 `vanilla_rag` 保留兼容，但正式实验、技术说明表格和结果文件统一使用 `hybrid_rag`。

本文档只写 `hybrid_rag` 自己额外需要什么环境、怎么在本地用 ECNU 真实模型和本地 ES 验收、以及搬到 H200 后如何切换模型 endpoint。Signpost/H200 已经配置好的总环境不在本文重复。

## 1. 实验目的

它回答的问题是：

```text
Signpost 的收益是否只是来自强 BM25+dense flat retriever？
```

`hybrid_rag` 不使用 Signpost 图、不使用 signpost sketch、不使用 scene-aware recommendation，也不进行 agentic 多轮检索。

## 2. 涉及代码

```text
signpost/baselines/vanilla_rag.py
signpost/baselines/hybrid_rag.py
signpost/baselines/common.py
signpost/retrieval/chunk_search.py
scripts/baselines/run_hybrid_rag.py
scripts/baselines/run_baseline_method.sh
tests/test_baselines.py
```

统一方法名：

```text
hybrid_rag
```

## 3. Baseline 专属环境需求

结论：

```text
不需要新建 conda 环境。
不需要新数据库。
不需要 MinIO。
不需要 Redis/Valkey。
不需要额外官方仓库。
需要已有 Elasticsearch chunk index。
需要 chat model endpoint。
需要 embedding endpoint。
```

使用已有环境：

```text
本地：/home/ruolinsu/signpost/signpost_re + conda env signpost-re
H200：/data/srl/signpost_re + conda env signpost-re
```

使用已有 ES：

```text
本地：Signpost docker compose 中的 Elasticsearch，默认 http://127.0.0.1:9200
H200：服务器上已经配置好的 Elasticsearch
```

`hybrid_rag` 自己不新建数据库表。它只读取：

```text
datasets/processed/<dataset>/questions.jsonl
datasets/processed/<dataset>/chunks.jsonl
F5_chunk_index 写入的 Elasticsearch chunk index
```

## 4. 输入与输出

输入：

```text
datasets/processed/<dataset>/questions.jsonl
datasets/processed/<dataset>/chunks.jsonl
```

正式 ES 模式还要求 F5 chunk index 已经存在。通常由以下脚本生成：

```bash
scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>
```

输出：

```text
outputs/<dataset>/predictions/hybrid_rag.jsonl
outputs/<dataset>/logs/hybrid_rag.query.jsonl
outputs/<dataset>/metrics/hybrid_rag.basic_eval.json
outputs/<dataset>/metrics/hybrid_rag.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

## 5. 成本记录口径

正式离线成本：

```text
F5_chunk_index
```

不计入：

```text
F3/F3.5/F4 shared preprocessing
F6 entity/relation extraction
F7/F8/F9/F10 Signpost graph organization and graph index
```

在线成本记录：

```text
latency_seconds
retrieval_latency_seconds
agent_reasoning_latency_seconds
llm_calls
online_llm_calls
tool_calls
knowledge_search_calls
input_tokens
output_tokens
total_tokens
retrieved_chunks
```

比较口径：

```text
1. 与 agentic RAG 方法比较时，重点看 LLM calls、tokens、tool calls 和总 latency。
2. 与非 agentic 单次检索方法比较时，重点看 retrieval_latency_seconds 和 top-k 检索结果质量。
3. hybrid_rag 每个问题应只有一次检索和一次生成调用。
```

## 6. 本地 ECNU + ES 调试流程

本地最终验收必须使用真实 ECNU chat、真实 ECNU embedding 和本地 Elasticsearch，不使用 fake/mock 或 `USE_ES=0` 作为最终通过标准。

ECNU 配置来自：

```text
/home/ruolinsu/signpost/ecnu.txt
```

本地 `.env.local.ecnu` 至少需要：

```bash
ECNU_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
ECNU_API_KEY=<your-ecnu-api-key>
ECNU_CHAT_MODEL=ecnu-plus
ECNU_REASONING_MODEL=ecnu-max

ECNU_EMBEDDING_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
ECNU_EMBEDDING_API_KEY=<your-ecnu-api-key>
ECNU_EMBEDDING_MODEL=ecnu-embedding-small
EMBEDDING_PROVIDER=ecnu
```

进入项目：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda activate signpost-re
set -a
source .env.local.ecnu
set +a
```

先跑单测，不访问真实模型服务，用于检查 schema 和指标口径：

```bash
python -m pytest tests/test_baselines.py tests/test_llm_client.py tests/test_benchmark_metrics.py
```

确认本地 ES 可用：

```bash
curl http://127.0.0.1:9200
```

确认真实 ECNU chat 和 embedding 可用：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

预期 embedding 维度：

```text
embedding_dimensions = 1024
```

如果本地 `legal_test` 尚未建过 F5 chunk index，先运行：

```bash
SEMANTIC_EXTRACTOR=llm \
EMBEDDING_PROVIDER=ecnu \
scripts/run_signpost_dataset_pipeline.sh legal_test legal_test
```

说明：

```text
这个 pipeline 可能执行 F6 语义抽取。F6 的时间、tokens、LLM calls 要记录，但按 v10 口径作为共享语义标注，不计入 hybrid_rag 方法离线成本。
hybrid_rag 方法离线成本只计 F5_chunk_index。
```

最终验收必须跑正式模式：

```bash
LIMIT=3 USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu \
  scripts/baselines/run_baseline_method.sh hybrid_rag legal_test legal_test
```

检查产物：

```bash
test -s outputs/legal_test/predictions/hybrid_rag.jsonl
test -s outputs/legal_test/logs/hybrid_rag.query.jsonl
test -s outputs/legal_test/metrics/hybrid_rag.basic_eval.json
test -s outputs/legal_test/metrics/hybrid_rag.query_metrics.json
test -s outputs/legal_test/metrics/method_summaries.json
test -s outputs/legal_test/metrics/cost_quality.json
```

## 7. H200 切换方式

H200 不需要为 `hybrid_rag` 新增 conda 环境、新数据库或额外组件。只使用已经部署好的 Signpost 环境、ES、H200 本地 chat 和 embedding 服务。

进入服务器项目目录：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
```

加载本地模型服务配置：

```bash
set -a
source .env.h200
set +a
```

确认 chat 和 embedding 服务：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

确认 F5 chunk index 已存在。如果该 dataset 的 Signpost pipeline 已经跑过，可以直接进入 smoke；否则先构建：

```bash
scripts/run_signpost_dataset_pipeline.sh legal_test legal_test
```

H200 smoke：

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

如果服务器正式数据集名称是 `Agriculture-full` 或 `Legal-full`，命令中的 dataset 和 namespace 必须与 `datasets/processed/<dataset>/` 目录一致。

## 8. 参数固定建议

正式主表建议固定：

```bash
USE_ES=1
MODE=hybrid
EMBEDDING_PROVIDER=ecnu
TOP_K=5
MAX_CONTEXT_TOKENS=3500
```

说明：

```text
EMBEDDING_PROVIDER=ecnu 是历史名称。H200 上实际通过 ECNU_EMBEDDING_API_BASE 指向 localhost embedding 服务。
不要为了改名字大范围修改代码，否则会增加迁移风险。
```

## 9. 结果检查

快速检查行数：

```bash
wc -l outputs/agriculture/predictions/hybrid_rag.jsonl
wc -l outputs/agriculture/logs/hybrid_rag.query.jsonl
```

检查 query metrics：

```bash
python -m json.tool outputs/agriculture/metrics/hybrid_rag.query_metrics.json | sed -n '1,160p'
```

预期现象：

```text
metadata.method = hybrid_rag
metadata.retrieval = flat_chunk_rag
metadata.use_es = true
metadata.mode = hybrid
retrieved_chunks 非空
tool_calls ≈ 1 per query
knowledge_search_calls ≈ 1 per query
read_file_calls = 0
llm_calls ≈ 1 per query
```

## 10. 常见问题

如果 ES 检索失败：

```text
1. 确认 dataset pipeline 已经跑过 F5_chunk_index。
2. 确认 namespace 和构建索引时一致。
3. 用 USE_ES=0 MODE=bm25 先确认非 ES 版本的 schema 和评估链路没问题。
```

如果 dense 或 hybrid 模式失败：

```text
检查 ECNU_EMBEDDING_API_BASE、ECNU_EMBEDDING_MODEL，以及 python -m signpost.llm.smoke --embedding。
```

如果输出里 `retrieved_chunks` 为空：

```text
1. 检查 chunk index 是否为空。
2. 检查 namespace 是否写错。
3. 检查 questions.jsonl 的问题文本是否为空。
```

如果 `read_file_calls` 不是 0：

```text
hybrid_rag 不调用 read_file。先检查 prediction JSONL 是否由 hybrid_rag 新入口生成，再重新运行 query_metrics。
不要把 citations 数量当成 read_file_calls。
```

如果正式结果误写成 `vanilla_rag`：

```text
不要直接把文件改名。重新用 hybrid_rag 入口运行，确保 metadata.method、prediction 文件名、query log 文件名一致。
```
