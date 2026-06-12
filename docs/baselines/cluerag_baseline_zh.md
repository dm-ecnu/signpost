# Clue-RAG Baseline 接入说明

> 当前正式实验默认使用 `baselines/ClueRAG/signpost_adapter/H200_MIGRATION_RUNBOOK.zh.md`
> 中的 `shared_es` 路径：复用 Signpost 已有 `chunks.jsonl` 和
> `semantic_llm.extractions.jsonl`，不使用 OceanBase，不运行 ClueRAG 官方重抽取流程；
> 但会按 ClueRAG 自己的 chunk / knowledge unit / entity 多层图组织重新建图，
> 并写入独立 ClueRAG ES index，不读取 Signpost unified/navigation graph。
> 本文后面保留的 `RUN_OFFICIAL=1` / `official_outputs` 内容仅用于
> `official_oceanbase` 可选诊断路径，不作为技术说明主表默认流程。

本文档说明 Clue-RAG 作为 Signpost 外部 baseline 的目录结构、输入输出、环境配置和运行命令。

Clue-RAG 是 ICDE 2026 录用的 graph-based RAG 方法，核心机制是：

```text
Clue-Index: chunk / knowledge unit / entity 多分区图索引
Q-Iter: query-driven iterative retrieval
```

在本文实验中，它对应的反事实问题是：

```text
如果已经有多粒度图索引和查询驱动迭代检索，是否仍然需要 Signpost 的 navigation-cue index？
```

## 1. 文件目录结构

官方仓库：

```text
baselines/ClueRAG/
  main.py
  data/
  dataset/
  index/
  retrieval/
  generation/
  llm/
  utils/
  requirements.txt
```

Signpost 侧适配器：

```text
signpost/baselines/cluerag.py
scripts/baselines/run_cluerag.py
scripts/baselines/run_cluerag_method.sh
docs/baselines/cluerag_baseline_zh.md
docs/baselines/cluerag_environment_h200_zh.md
tests/test_baselines.py
```

职责划分：

```text
baselines/ClueRAG/                  # 官方代码，尽量不改
signpost/baselines/cluerag.py       # 数据转换、可选调用官方 pipeline、输出 schema 转换
scripts/baselines/run_cluerag.py    # python -m 入口
scripts/baselines/run_cluerag_method.sh # 端到端实验入口：prepare/run/convert/eval/summary
```

## 2. 输入与转换

正式 `shared_es` 路径只使用 Signpost 共享阶段产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

它不会重新切 chunk，也不会重新抽 entity/relation。`semantic_llm.extractions.jsonl`
会在 adapter 内转换成 ClueRAG 自己的三层组织：

```text
chunk nodes
knowledge unit nodes
entity nodes
chunk <-> knowledge unit <-> entity edges
```

同时仍保留 prepare step，把文档/问题转换成 ClueRAG 官方格式，供
`official_oceanbase` 诊断路径使用：

```text
datasets/processed/<dataset>/documents.jsonl
datasets/processed/<dataset>/questions.jsonl
```

转换为 Clue-RAG 官方格式：

```text
baselines/ClueRAG/data/signpost_<dataset>.json
baselines/ClueRAG/data/signpost_<dataset>_corpus.json
```

`*_corpus.json` 每条记录：

```json
{
  "idx": 0,
  "title": "doc.txt",
  "text": "document text"
}
```

`*.json` 每条记录：

```json
{
  "_id": "q1",
  "id": "q1",
  "question": "...",
  "answer": "..."
}
```

转换 manifest：

```text
outputs/<dataset>/baselines/cluerag/manifest.json
```

## 3. 输出格式

`shared_es` 默认输出：

```text
outputs/<dataset>/baselines/cluerag/shared_graph/
  manifest.json
  graph_cache.json
  chunks.jsonl
  knowledge_units.jsonl
  entities.jsonl

outputs/<dataset>/baselines/cluerag/shared_outputs/COSINE_1.00/
  retrieval_results.json
  generation_results.json
```

`official_oceanbase` 可选诊断路径输出：

```text
outputs/<dataset>/baselines/cluerag/official_outputs/COSINE_1.00/
  retrieval_results.json
  generation_results.json
```

ClueRAG 中间输出：

```text
outputs/<dataset>/predictions/cluerag.jsonl
outputs/<dataset>/logs/cluerag.query.jsonl
outputs/<dataset>/metrics/cluerag.basic_eval.json
outputs/<dataset>/metrics/cluerag.query_metrics.json
```

注意：上述 `cluerag.jsonl` 使用默认 ClueRAG/adapter final prompt，只作为 graph/retrieval 中间流程的历史产物保留，不进入技术说明表格。技术说明中的 ClueRAG baseline 使用统一生成约束后的输出：

```text
outputs/<dataset>/predictions/cluerag_prompt_normalized.jsonl
outputs/<dataset>/logs/cluerag_prompt_normalized.query.jsonl
outputs/<dataset>/metrics/cluerag_prompt_normalized.basic_eval.json
outputs/<dataset>/metrics/cluerag_prompt_normalized.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

`cluerag_prompt_normalized.jsonl` 会写入：

```text
metadata.method=cluerag_prompt_normalized
retrieved_chunks=[{"chunk_id": "...", "score_source": "cluerag_q_iter"}]
trace=[{"tool": "cluerag", ...}]
```

`shared_es` 路径保留 Signpost 共享 `chunk_id`，因此 evidence recall 可以直接用统一评测。只有 `official_oceanbase` 路径可能出现官方内部 chunk id 与 Signpost `chunk_id` 不一致的问题。

## 4. 环境配置

Clue-RAG 的 H200 依赖配置、OceanBase/rerank 要求、时间与 token 统计方式见：

```text
docs/baselines/cluerag_environment_h200_zh.md
```

### 4.1 Signpost 环境

本地开发：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda activate signpost-re
```

H200：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a
```

验证本地模型服务：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
```

### 4.2 Clue-RAG 依赖

默认 `shared_es` 不需要安装 Clue-RAG 官方依赖，也不需要 OceanBase；直接使用
`signpost-re` 环境即可。官方依赖只用于 `official_oceanbase` 诊断路径，依赖文件在：

```text
baselines/ClueRAG/requirements.txt
```

它包含较重依赖，例如：

```text
pymilvus
milvus-lite
spacy
en_core_web_trf
FlagEmbedding
torch
openai
tiktoken
```

如果确实要跑官方路径，建议单独建环境，而不是污染 `signpost-re`：

```bash
conda create -n cluerag python=3.11 -y
conda activate cluerag
pip install -r baselines/ClueRAG/requirements.txt
```

如果 `en_core_web_trf` 下载失败，可先只做数据转换和 schema 测试；正式运行前必须解决 spaCy 模型依赖。

### 4.3 H200 本地模型映射

适配器会把 Signpost 的 `.env.h200` 映射到 Clue-RAG 的 config：

```text
llm_base_url          <- ECNU_API_BASE
llm_name              <- ECNU_CHAT_MODEL
api_key               <- ECNU_API_KEY
embedding_model_url   <- ECNU_EMBEDDING_API_BASE
embedding_model_name  <- ECNU_EMBEDDING_MODEL
```

正式实验目标：

```text
Chat:      http://localhost:8000/v1, model=/data/srl/Llama-3.3-70B-FP8
Embedding: http://localhost:8001/v1/embeddings, model=/data/srl/nemotron-8b
```

### 4.4 Rerank 服务

正式 `shared_es` 路径默认使用 H200 本地 NVIDIA rerank API：

```text
rerank_url=http://127.0.0.1:8033/v1/rerank
model=/data/srl/llama-nemotron-rerank-1b-v2
```

正式结果不要设置 rerank fallback。只有临时排查时才允许：

```bash
export CLUERAG_ALLOW_RERANK_FALLBACK=1
```

## 5. 运行命令

### 5.1 正式 shared_es 路径

正式 Agriculture：

```bash
export CLUERAG_BACKEND=shared_es
export CLUERAG_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export CLUERAG_RERANK_URL=http://127.0.0.1:8033/v1/rerank
scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

正式 Legal：

```bash
export CLUERAG_BACKEND=shared_es
export CLUERAG_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export CLUERAG_RERANK_URL=http://127.0.0.1:8033/v1/rerank
scripts/baselines/run_cluerag_method.sh legal legal
```

默认每次会重建 ClueRAG 自己的 ES 图索引并记录离线成本：

```bash
REUSE_GRAPH=0 scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

重复调参时可以复用已有 ClueRAG ES index：

```bash
REUSE_GRAPH=1 scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

### 5.2 shared_local 函数级排查

`shared_local` 只用于本地或 H200 临时排查，不作为技术说明主结果：

```bash
LIMIT=3 CLUERAG_BACKEND=shared_local RERANK_URL= \
  scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

### 5.3 官方 OceanBase 诊断路径

只有明确要跑官方 full pipeline 时才使用：

```bash
export CLUERAG_BACKEND=official_oceanbase
export CLUERAG_DB_HOST=127.0.0.1
export CLUERAG_DB_PORT=<oceanbase_port>
export CLUERAG_DB_USER=<oceanbase_user>
export CLUERAG_DB_PASSWORD=<oceanbase_password>
export CLUERAG_DB_NAME=clueragdb
scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

如果已经手动跑完官方代码，只做输出转换：

```bash
CONVERT_ONLY=1 OFFICIAL_OUTPUT_DIR=outputs/legal_test/baselines/cluerag/official_outputs/COSINE_1.00 \
  scripts/baselines/run_cluerag_method.sh legal_test legal_test
```

## 6. 功能点覆盖

| 功能点 | 文件 | 输入 | 输出 |
|---|---|---|---|
| F-BL-CR-1 数据转换 | `signpost/baselines/cluerag.py::prepare_cluerag_inputs` | `documents.jsonl`, `questions.jsonl` | `baselines/ClueRAG/data/signpost_<dataset>*.json`, `manifest.json` |
| F-BL-CR-2 shared graph 构建 | `signpost/baselines/cluerag.py::run_cluerag_shared` | `chunks.jsonl`, `semantic_llm.extractions.jsonl` | `shared_graph/*`, 独立 ClueRAG ES index |
| F-BL-CR-3 Q-Iter/rerank/generation | `signpost/baselines/cluerag.py::run_cluerag_shared` | shared graph, local chat/embedding/rerank | `shared_outputs/*` |
| F-BL-CR-4 输出转换 | `signpost/baselines/cluerag.py::convert_cluerag_outputs` | `generation_results.json` | unified `predictions/cluerag.jsonl`, `logs/cluerag.query.jsonl` |
| F-BL-CR-5 端到端脚本 | `scripts/baselines/run_cluerag_method.sh` | dataset name | prediction, eval, query metrics, method summary, cost quality |
| F-BL-CR-6 官方 pipeline 调用 | `signpost/baselines/cluerag.py::run_cluerag_official` | Clue-RAG data files, H200 env | official `retrieval_results.json`, `generation_results.json` |
| F-BL-CR-7 单测 | `tests/test_baselines.py` | fake tmp data | conversion/shared graph/schema tests |

## 7. 当前状态与下一步

已完成：

```text
1. 官方仓库已克隆到 baselines/ClueRAG。
2. Signpost -> Clue-RAG 数据转换。
3. Clue-RAG generation_results -> Signpost prediction schema 转换。
4. shared_es adapter：从共享 chunks/F6 semantics 构建 ClueRAG 自己的 chunk/KU/entity 多层图。
5. 独立 ClueRAG ES index 与 REUSE_GRAPH 复用逻辑。
6. per-query LLM/rerank/embedding/tool/retrieved chunk 记录。
7. baseline-level offline/online/total metrics。
8. 可选官方 pipeline 调用入口。
9. 端到端 shell runner、文档和单测。
```

尚未完成：

```text
1. 将本地 patch 上传 H200。
2. 用 shared_es 重新跑 agriculture/legal 的 ClueRAG 正式结果。
3. 下载每次 run 的 outputs/<dataset>/baselines/cluerag 和统一 metrics。
4. 如需官方诊断路径，再单独安装 Clue-RAG/OceanBase 依赖。
```

本地已覆盖的轻量检查：

```bash
conda run -n signpost-re python -m pytest tests/test_baselines.py
```
