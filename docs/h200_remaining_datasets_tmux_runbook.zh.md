# H200 全数据集正式实验 tmux 执行手册

本文档用于在 H200 上跑论文数据集：

```text
agriculture
legal
mix
graphrag-bench-medical
graphrag-bench-novel
```

执行约束：

- 每个数据集必须先完成 Signpost 主实验离线构建、Signpost full 和 Signpost 消融实验；
- 该数据集的 Signpost 主实验和消融实验完成后，才能开始该数据集的 baseline；
- baseline 复用 `datasets/processed/<dataset>/chunks.jsonl`、`semantic_llm.extractions.jsonl`、`questions.jsonl`；
- 不重新切 chunk，不重新抽实体，不改 Signpost 主流程；
- ClueRAG 使用自己的 baseline 适配层构建 ClueRAG 多层图和 ES index；
- 每个长任务都在 tmux 中执行。

维护约束：

```text
之后每处理好一个 baseline，必须同步更新本文档：
1. 在每个相关数据集的操作流程中加入该 baseline 的正式运行命令。
2. 写清该 baseline 哪些步骤只是中间产物，哪些 prediction 才进入论文。
3. 在本文档的 prompt registry 中写出该 baseline final generation 阶段使用的完整 prompt。
4. 如果该 baseline 保留自己的输出格式，必须写清“统一了哪些 Signpost 回答约束，保留了哪些输出格式约束”。
5. 更新完整性检查清单，加入该 baseline 应产生的 predictions/logs/metrics/run_metrics 文件。
```

## 1. 固定服务器环境

H200 当前固定目录和服务：

```bash
PROJECT_DIR=/home/srl/signpost_re
CHAT_BASE=http://localhost:8000/v1
CHAT_MODEL=/data/srl/Llama-3.3-70B-FP8
EMBED_BASE=http://localhost:8001/v1/embeddings
EMBED_MODEL=/data/srl/nemotron-8b
RERANK_URL=http://localhost:8033/v1/rerank
RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
ES_URL=http://127.0.0.1:9200
```

每个 tmux 窗口开始后先执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
```

服务检查：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

## 2. 总体顺序

对每个数据集执行同一个顺序：

```text
1. scripts/run_signpost_dataset_pipeline.sh <dataset> <namespace>
2. scripts/run_signpost_ablation_suite.sh <dataset> <namespace>
3. scripts/baselines/run_baseline_method.sh vanilla_llm <dataset> <namespace>
4. scripts/baselines/run_baseline_method.sh hybrid_rag <dataset> <namespace>
5. scripts/baselines/run_cluerag_method.sh <dataset> <namespace>  # 仅用于构建 ClueRAG 图、retrieval 和中间产物；默认 prompt 生成结果不进入论文
6. 复用 ClueRAG retrieval，正式生成论文 baseline：`cluerag_prompt_normalized`
7. AGRAG：使用各数据集小节中的直接命令，例如 `scripts/baselines/run_baseline_method.sh agrag agriculture agriculture`
8. LinearRAG：使用各数据集小节中的直接命令，例如 `scripts/baselines/run_baseline_method.sh linearrag agriculture agriculture`
9. HiPRAG：使用各数据集小节中的直接命令，例如 `scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture`
10. GraphRAG-R1：使用各数据集小节中的直接命令，例如 `scripts/baselines/run_baseline_method.sh graphrag_r1 agriculture agriculture`
11. MemGraphRAG：使用各数据集小节或 MemGraphRAG 小节中的直接命令，例如 `scripts/baselines/run_baseline_method.sh memgraphrag agriculture agriculture`
12. 重算 basic/query/method/cost 指标
13. 检查产物完整性
```

建议 namespace 与 dataset 同名。GraphRAG-Bench 数据集名里有连字符，ES index 可以正常工作。

注意：

```text
默认 scripts/baselines/run_cluerag_method.sh 跑出的 `cluerag` 只作为 ClueRAG 图构建、retrieval 和 raw 中间产物，不进入论文主表。
论文中的 ClueRAG baseline 使用 `cluerag_prompt_normalized`：设置 CLUERAG_GENERATION_ONLY=1 和 CLUERAG_PROMPT_STYLE=signpost_fewshot 后，复用 ClueRAG retrieval，只重跑 final generation。
cluerag_prompt_normalized 必须在中间 `cluerag` 跑完之后执行，因为它复用 `outputs/<dataset>/baselines/cluerag/shared_outputs/COSINE_1.00/retrieval_results.json`。
AGRAG 必须在该数据集 Signpost 共享阶段完成后执行；它复用 `chunks.jsonl`、`semantic_llm.extractions.jsonl`、`questions.jsonl`，只在 `outputs/<dataset>/baselines/agrag/` 下构建自己的 graph/triple artifacts。
LinearRAG 必须在该数据集 Signpost 共享阶段完成后执行；它复用 `chunks.jsonl`、`semantic_llm.extractions.jsonl`、`questions.jsonl`，只在 `outputs/<dataset>/baselines/linearrag/` 下构建自己的 relation-free entity/sentence/passage graph artifacts。
HiPRAG 必须在该数据集 Signpost 共享阶段完成后执行；它复用 `chunks.jsonl` 和 `questions.jsonl`，不读 Signpost graph/index/navigation-cue index，只在 `outputs/<dataset>/baselines/hiprag/` 下记录自己的本地 agentic chunk retrieval index 指标。
GraphRAG-R1 必须在该数据集 Signpost 共享阶段完成后执行；它复用 `chunks.jsonl`、`semantic_llm.extractions.jsonl`、`questions.jsonl`，不读 Signpost graph/index/navigation-cue index，只在 `outputs/<dataset>/baselines/graphrag_r1/` 下构建自己的 agentic graph retrieval artifacts。正式输出保留 `<think>/<answer>` 与 `<|begin_of_query|>...<|end_of_query|>` 检索标签契约，只迁移 Signpost evidence-grounded 回答约束。
MemGraphRAG 必须在该数据集 Signpost 共享阶段完成后执行；它只复用公共 `chunk/entity/type/relation`，不复用 Signpost fact/provenance、银证据或 target units。它在 `outputs/<dataset>/baselines/memgraphrag/` 下构建自己的 OpenIE-like observations、schema/fact/passage memory、fact-to-passage links、entity-passage PPR graph 和 embedding cache。正式输出保留 `Thought:` / `Answer:` 契约，只迁移 Signpost evidence-grounded 回答约束。
```

## 2.0 无人值守顺序执行所有实验

本节用于“每个数据集一个 tmux，启动一次后自动按顺序跑完”。每个数据集内部严格顺序执行：

```text
Signpost dataset pipeline -> Signpost full/ablations -> vanilla_llm -> hybrid_rag -> cluerag 中间步骤 -> cluerag_prompt_normalized -> agrag -> linearrag -> hiprag -> graphrag_r1
```

脚本使用 `set -euo pipefail`；任意一步失败后该数据集 tmux 会停止在失败位置，不会继续跑后续 baseline。这样可以避免前置产物不完整时产生无效 prediction。

注意：下面命令会让多个数据集在不同 tmux 中并发执行。若需要严格比较 wall-clock，或 8001 embedding 服务不稳定，不要同时启动所有数据集；可以只启动 1 个数据集，等它完成后再启动下一个。结构化指标仍以 `outputs/<dataset>/logs/stage_timing.jsonl`、`predictions/*.jsonl`、`metrics/*.json` 为准；`tee` 日志只用于排查。

默认 `REUSE_BASELINE_INDEX=0`，用于首次正式运行：AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 / MemGraphRAG 会构建自己的 baseline-owned graph/index，并写入 `outputs/<dataset>/baselines/<method>/index.pkl` cache。后续若只想让 baseline 部分重跑 online query 和 final generation，启动 tmux 时设置 `REUSE_BASELINE_INDEX=1`；ClueRAG 会同步设置 `REUSE_GRAPH=1`，复用 `shared_graph` 和 baseline-owned ES index。

### 2.0.1 在 H200 创建通用无人值守脚本

在 H200 任意 shell 中执行一次：

```bash
cat > /home/srl/run_signpost_dataset_all_unattended.sh <<'RUN_DATASET_ALL'
#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_signpost_dataset_all_unattended.sh <dataset> [namespace]}"
NAMESPACE="${2:-$DATASET}"
PROJECT_DIR=/home/srl/signpost_re
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="/home/srl/${DATASET}_all_experiments_${STAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[all] start dataset=${DATASET} namespace=${NAMESPACE} log=${LOG_FILE}"
date

cd "${PROJECT_DIR}"
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export REUSE_BASELINE_INDEX="${REUSE_BASELINE_INDEX:-0}"
export REUSE_GRAPH="${REUSE_GRAPH:-${REUSE_BASELINE_INDEX}}"

curl -fsS http://127.0.0.1:9200 >/tmp/es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/chat_models.ok
curl -fsS http://localhost:8001/v1/models >/tmp/embed_models.ok
curl -fsS http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["embedding health check"]}' >/tmp/embed_smoke.ok
curl -fsS http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' >/tmp/rerank_smoke.ok

echo "[all] services ok"

if [[ "${REUSE_BASELINE_INDEX}" == "1" ]]; then
  echo "[all] reuse mode: skip Signpost offline dataset pipeline; rerun Signpost full/ablations and baseline online query/final generation"
else
  scripts/run_signpost_dataset_pipeline.sh "${DATASET}" "${NAMESPACE}"
fi

scripts/run_signpost_ablation_suite.sh "${DATASET}" "${NAMESPACE}"

scripts/baselines/run_baseline_method.sh vanilla_llm "${DATASET}" "${NAMESPACE}"
scripts/baselines/run_baseline_method.sh hybrid_rag "${DATASET}" "${NAMESPACE}"

export CLUERAG_BACKEND=shared_es
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"

export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"
unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR

export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag "${DATASET}" "${NAMESPACE}"

export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag "${DATASET}" "${NAMESPACE}"

export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag "${DATASET}" "${NAMESPACE}"

export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 "${DATASET}" "${NAMESPACE}"

export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export MEMGRAPHRAG_EMBED_BATCH_SIZE=32
export MEMGRAPHRAG_SCHEMA_MIN_COUNT=2
export MEMGRAPHRAG_RETRIEVAL_TOP_K=200
export MEMGRAPHRAG_QA_TOP_K=5
export MEMGRAPHRAG_LINKING_TOP_K=5
export MEMGRAPHRAG_PPR_DAMPING=0.5
export MEMGRAPHRAG_PPR_ITERATIONS=20
export MEMGRAPHRAG_PASSAGE_NODE_WEIGHT=0.05
export MEMGRAPHRAG_SYNONYMY_EDGES=1
scripts/baselines/run_baseline_method.sh memgraphrag "${DATASET}" "${NAMESPACE}"

echo "[all] completed dataset=${DATASET} namespace=${NAMESPACE}"
date
RUN_DATASET_ALL

chmod +x /home/srl/run_signpost_dataset_all_unattended.sh
```

### 2.0.2 每个数据集一个 tmux 后台启动

首次执行会完整跑 Signpost pipeline、Signpost full/ablations、vanilla_llm、hybrid_rag、ClueRAG graph/retrieval、ClueRAG prompt-normalized generation、AGRAG、LinearRAG、HiPRAG、GraphRAG-R1、MemGraphRAG，并为 baseline-owned graph/index 写 cache。后续复用执行会显式传入 `REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1`；baseline 部分中，ClueRAG 复用 `shared_graph` 和自有 ES index，AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 / MemGraphRAG 复用 `index.pkl`，只重跑 online query 和 final generation。

首次执行：如果确认 8000/8001/8033/ES 服务稳定，可以一次性启动全部数据集：

```bash
tmux new-session -d -s sp-auto-agriculture "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh agriculture agriculture'"
tmux new-session -d -s sp-auto-legal "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh legal legal'"
tmux new-session -d -s sp-auto-mix "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh mix mix'"
tmux new-session -d -s sp-auto-medical "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-medical graphrag-bench-medical'"
tmux new-session -d -s sp-auto-novel "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-novel graphrag-bench-novel'"
```

首次执行：如果只想按数据集逐个启动，复制对应一行即可：

```bash
tmux new-session -d -s sp-auto-agriculture "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh agriculture agriculture'"
tmux new-session -d -s sp-auto-legal "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh legal legal'"
tmux new-session -d -s sp-auto-mix "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh mix mix'"
tmux new-session -d -s sp-auto-medical "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-medical graphrag-bench-medical'"
tmux new-session -d -s sp-auto-novel "bash -lc '/home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-novel graphrag-bench-novel'"
```

后续复用执行：若该数据集已经完成过首次 baseline-owned graph/index 构建，baseline 部分只重跑 online query 和 final generation，用下面的复用版启动命令。复用版要求对应目录下已存在：

```text
outputs/<dataset>/baselines/cluerag/shared_graph/manifest.json
outputs/<dataset>/baselines/agrag/index.pkl
outputs/<dataset>/baselines/linearrag/index.pkl
outputs/<dataset>/baselines/hiprag/index.pkl
outputs/<dataset>/baselines/graphrag_r1/index.pkl
```

后续复用执行：一次性启动全部数据集：

```bash
tmux new-session -d -s sp-reuse-agriculture "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh agriculture agriculture'"
tmux new-session -d -s sp-reuse-legal "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh legal legal'"
tmux new-session -d -s sp-reuse-mix "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh mix mix'"
tmux new-session -d -s sp-reuse-medical "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-medical graphrag-bench-medical'"
tmux new-session -d -s sp-reuse-novel "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-novel graphrag-bench-novel'"
```

后续复用执行：如果只想按数据集逐个启动，复制对应一行即可：

```bash
tmux new-session -d -s sp-reuse-agriculture "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh agriculture agriculture'"
tmux new-session -d -s sp-reuse-legal "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh legal legal'"
tmux new-session -d -s sp-reuse-mix "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh mix mix'"
tmux new-session -d -s sp-reuse-medical "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-medical graphrag-bench-medical'"
tmux new-session -d -s sp-reuse-novel "bash -lc 'REUSE_BASELINE_INDEX=1 REUSE_GRAPH=1 /home/srl/run_signpost_dataset_all_unattended.sh graphrag-bench-novel graphrag-bench-novel'"
```

### 2.0.3 监控和失败定位

查看 tmux 和日志：

```bash
tmux ls
tmux attach -t sp-auto-agriculture
tmux attach -t sp-auto-legal
tmux attach -t sp-auto-mix
tmux attach -t sp-auto-medical
tmux attach -t sp-auto-novel

ls -lh /home/srl/*_all_experiments_*.log
tail -n 80 /home/srl/agriculture_all_experiments_*.log
tail -n 80 /home/srl/legal_all_experiments_*.log
tail -n 80 /home/srl/mix_all_experiments_*.log
tail -n 80 /home/srl/graphrag-bench-medical_all_experiments_*.log
tail -n 80 /home/srl/graphrag-bench-novel_all_experiments_*.log
```

检查每个数据集是否完整：

```bash
for d in agriculture legal mix graphrag-bench-medical graphrag-bench-novel; do
  echo "===== $d ====="
  wc -l outputs/$d/predictions/signpost.full.jsonl \
        outputs/$d/predictions/signpost.no_offline.jsonl \
        outputs/$d/predictions/signpost.no_online.jsonl \
        outputs/$d/predictions/signpost.no_semantic_cues.jsonl \
        outputs/$d/predictions/signpost.no_provenance_cues.jsonl \
        outputs/$d/predictions/signpost.no_vertical_cues.jsonl \
        outputs/$d/predictions/signpost.no_horizontal_cues.jsonl \
        outputs/$d/predictions/vanilla_llm.jsonl \
        outputs/$d/predictions/hybrid_rag.jsonl \
        outputs/$d/predictions/cluerag_prompt_normalized.jsonl \
        outputs/$d/predictions/agrag.jsonl \
        outputs/$d/predictions/linearrag.jsonl \
        outputs/$d/predictions/hiprag.jsonl \
        outputs/$d/predictions/graphrag_r1.jsonl
  ls -lh outputs/$d/baselines/cluerag/run_metrics.json \
         outputs/$d/baselines/cluerag_prompt_normalized/run_metrics.json \
         outputs/$d/baselines/agrag/run_metrics.json \
         outputs/$d/baselines/linearrag/run_metrics.json \
         outputs/$d/baselines/hiprag/run_metrics.json \
         outputs/$d/baselines/graphrag_r1/run_metrics.json
  tail -n 12 outputs/$d/logs/stage_timing.jsonl
done
```

失败时先看该数据集日志和 structured stage log：

```bash
DATASET=mix
tail -n 120 /home/srl/${DATASET}_all_experiments_*.log
grep '"status":"failed"' outputs/${DATASET}/logs/stage_timing.jsonl || true
tail -n 30 outputs/${DATASET}/logs/stage_timing.jsonl
```

不要直接重启整套无人值守脚本覆盖已有结果；先按失败 stage 或失败 baseline 单独补跑。

### 2.0.4 Baseline index 复用与 ClueRAG conversion 热修

如果看到 ClueRAG 在 `baseline_cluerag_full` 的 conversion 阶段报错：

```text
AttributeError: 'Result' object has no attribute 'embedding_calls'
```

说明 H200 还没有应用 `baseline_index_reuse_and_cluerag_cost_hotfix`。该错误发生在 ClueRAG 已经完成 retrieval/generation 后，把 `generation_results.json` 转成统一 prediction 时；不是 graph 构建本身失败。

应用热修后：

```text
1. ClueRAG conversion 使用完整 BaselineResult cost schema，不再因为 embedding_calls/rerank_calls 缺字段失败。
2. REUSE_GRAPH=1 时，ClueRAG 跳过 prepare 和 shared_graph 重建，从 outputs/<dataset>/baselines/cluerag/shared_graph/ 加载 graph artifacts；USE_ES=1 时要求 ClueRAG 自己的 ES index 已存在。
3. REUSE_BASELINE_INDEX=1 时，AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 / MemGraphRAG 从 index.pkl 加载自己的 graph/index，只重跑 online query 和 final generation。
```

H200 应用热修包：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/baseline_index_reuse_and_cluerag_cost_hotfix_<STAMP>.tar.gz

conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re

python -m py_compile \
  signpost/baselines/cluerag.py \
  signpost/baselines/agrag.py \
  signpost/baselines/linearrag.py \
  signpost/baselines/hiprag.py \
  signpost/baselines/graphrag_r1.py \
  scripts/baselines/run_cluerag.py \
  scripts/baselines/run_agrag.py \
  scripts/baselines/run_linearrag.py \
  scripts/baselines/run_hiprag.py \
  scripts/baselines/run_graphrag_r1.py
bash -n scripts/baselines/run_cluerag_method.sh scripts/baselines/run_baseline_method.sh
```

ClueRAG 首次正式运行显式设置 `REUSE_GRAPH=0`：

```bash
export CLUERAG_BACKEND=shared_es
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export REUSE_GRAPH=0
scripts/baselines/run_cluerag_method.sh mix mix
```

ClueRAG 后续只复用 graph/index 重跑 retrieval + generation：

```bash
export CLUERAG_BACKEND=shared_es
export USE_ES=1
export REUSE_GRAPH=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh mix mix
unset REUSE_GRAPH
```

ClueRAG prompt-normalized 论文正式结果仍然只重跑 final generation：

```bash
export REUSE_GRAPH=1
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/mix/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh mix mix
unset REUSE_GRAPH CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR
```

## 2.1 2026-05-24 8001 embedding 服务中断恢复操作

恢复时间点：2026-05-24。故障表现为 H200 本地 embedding 服务 `http://localhost:8001/v1/embeddings` 对客户端返回 `Connection refused`、`TimeoutError` 或 HTTP 500；`embed` tmux 中 vLLM 可能报 `CUDA error: an illegal memory access was encountered`，随后 8001 API server 退出。

2026-05-24 HiPRAG agriculture 再次触发同类故障：第一次运行在 offline embedding 阶段返回 HTTP 500，第二次热修后明确显示 `Connection refused`。服务检查中 8000 chat、8033 rerank、ES 均正常，只有 8001 无响应。这说明不是单个 baseline 的 final generation 问题，而是 embedding 服务可用性问题；所有正在跑且需要 embedding 的 tmux 窗口都可能被迫中断，必须先盘点再恢复。

### 2.1.1 Baseline embedding 稳定性统一约束

之后所有需要离线或在线 embedding 的 baseline 正式运行，必须统一设置：

```bash
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
```

各 baseline 的专属变量只能作为覆盖项，不应比统一值更激进：

```text
AGRAG_EMBED_BATCH_SIZE      默认继承 BASELINE_EMBED_BATCH_SIZE
LINEARRAG_EMBED_BATCH_SIZE  默认继承 BASELINE_EMBED_BATCH_SIZE
HIPRAG_EMBED_BATCH_SIZE     默认继承 BASELINE_EMBED_BATCH_SIZE
CLUERAG_EMBED_BATCH_SIZE    正式命令建议显式设为 32
```

如果 8001 再次 HTTP 500 或进程退出，先不要继续提交新的 embedding-heavy 任务；恢复 8001 后，再从失败 stage 或失败 baseline 单独补跑。`stage_timing.jsonl` 中的 failed 记录必须保留作审计。

本次恢复原则：

```text
1. 不删除 outputs/*/logs/stage_timing.jsonl 中的 failed 记录；failed 记录保留作审计。
2. 已应用统计热修：method_summary 和 baseline artifact_summary 只统计 status=ok 的 stage。
3. 不整套重跑已完成的 suite；用 scripts/run_signpost_method.sh 单独补失败 variant。
4. 每次最多运行 1 个正式实验任务，尤其是需要 embedding 的 F15/F5/F10/AGRAG/LinearRAG/HiPRAG。否则 wall time 和服务稳定性都会被并发污染。
5. 如果 legal 仍在 F6_semantic_graph_llm，它主要使用 chat 服务；可以等它自然结束。为了论文时间口径稳定，不建议同时再启动多个 F15/baseline 正式任务。
6. 发生 8001 中断后，必须先盘点所有 tmux 和 stage log，再决定哪些继续观察、哪些单独补跑；不要凭单个窗口的报错推断全局状态。
```

### 2.1.2 恢复前检查服务

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

ss -ltnp | grep 8001
curl -sS http://localhost:8001/v1/models | head
curl -sS http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["test"]}' | head
python -m signpost.llm.smoke --embedding
```

如果 `curl -sS http://localhost:8001/v1/models` 没有返回、`ss -ltnp | grep 8001` 没有监听，先不要重跑任何需要 embedding 的实验。

### 2.1.3 重启 8001 embedding 服务

H200 的 embedding 服务运行在 `embed` tmux session，启动命令来自 `docs/h200_local_signpost_migration.zh.md`。如果 session 还在：

```bash
tmux attach -t embed
```

如果已经回到 shell prompt，直接重新启动；如果仍有旧进程占用端口，先确认监听进程：

```bash
ss -ltnp | grep 8001 || echo "8001 not listening"
ps -ef | grep -E '8001|nemotron|vllm' | grep -v grep
```

确认旧服务已退出或是崩溃残留后，在 `embed` session 中执行：

```bash
cd /data/srl
conda activate /data/srl/.conda_envs/vllm

CUDA_VISIBLE_DEVICES=2 \
VLLM_USE_DEEP_GEMM=0 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/nemotron-8b \
  --runner pooling \
  --port 8001 \
  --trust-remote-code
```

重启后在另一个 tmux 或 shell 中检查：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

ss -ltnp | grep 8001
curl -sS http://localhost:8001/v1/models | head
curl -sS http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["embedding health check"]}' | head
python -m signpost.llm.smoke --embedding
```

只有上述检查通过后，才允许恢复中断任务。

### 2.1.4 盘点哪些 tmux 受影响

先保存所有 tmux 窗口最近输出：

```bash
tmux ls
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_current_path} active=#{pane_active}'

for p in $(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}'); do
  echo "===== $p ====="
  tmux capture-pane -pt "$p" -S -80
done > /home/srl/tmux_recovery_status_$(date +%Y%m%d_%H%M).txt
```

检查所有 failed stage：

```bash
grep -H '"status":"failed"' outputs/*/logs/stage_timing.jsonl > /home/srl/failed_stages_$(date +%Y%m%d_%H%M).txt || true
```

检查哪些任务还在跑，哪些 Python 已退出：

```bash
ps -ef | grep -E 'run_signpost|run_cluerag|run_baseline|signpost.agent|semantic_graph|run_agrag|run_linearrag|run_hiprag' | grep -v grep
```

检查 prediction 和最近 stage：

```bash
for d in agriculture legal mix graphrag-bench-medical graphrag-bench-novel; do
  echo "===== $d ====="
  wc -l outputs/$d/predictions/*.jsonl 2>/dev/null
  tail -n 12 outputs/$d/logs/stage_timing.jsonl 2>/dev/null
done > /home/srl/prediction_counts_$(date +%Y%m%d_%H%M).txt
```

判定规则：

```text
tmux 仍有 Python 进程且 pane 还在持续输出：继续观察，不要重复启动同一任务。
tmux 已回到 shell prompt 且 stage_timing 最近是 status=failed：按失败 method/stage 单独补跑。
tmux 已回到 shell prompt 但 prediction/log/metrics 完整，stage_timing 最近 status=ok：视为完成，不重跑。
F6_semantic_graph_llm 主要依赖 chat，不依赖 8001；如果它还在跑且没有 failed，可继续等待。
F5/F10/F15、Hybrid RAG、ClueRAG graph/retrieval、AGRAG、LinearRAG、HiPRAG 都可能依赖 embedding；8001 中断后重点检查这些任务。
```

### 2.1.5 中断任务的恢复方式

恢复原则是“补失败项，不从头重跑数据集”。先看 `/home/srl/failed_stages_*.txt` 和对应 tmux pane 的最后输出。

Signpost ablation 单个 method 失败时，用 method 级脚本补跑：

```bash
scripts/run_signpost_method.sh <dataset> <variant> <namespace>
```

示例：

```bash
scripts/run_signpost_method.sh mix no_online mix
scripts/run_signpost_method.sh graphrag-bench-medical no_offline graphrag-bench-medical
```

Baseline 失败时，只重跑该 baseline。若该 method 还没有成功写出自己的 index cache，使用首次构建模式：

```bash
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5

scripts/baselines/run_baseline_method.sh agrag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh linearrag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh hiprag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh graphrag_r1 <dataset> <namespace>
```

若 `outputs/<dataset>/baselines/<method>/index.pkl` 已存在，且 ClueRAG 已有 `outputs/<dataset>/baselines/cluerag/shared_graph/manifest.json` 与自有 ES index，baseline 部分只重跑 online query 和 final generation：

```bash
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5

scripts/baselines/run_cluerag_method.sh <dataset> <namespace>
scripts/baselines/run_baseline_method.sh agrag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh linearrag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh hiprag <dataset> <namespace>
scripts/baselines/run_baseline_method.sh graphrag_r1 <dataset> <namespace>
```

HiPRAG agriculture 本次 8001 中断后的恢复命令：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4

{
  date
  scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture
  date
} 2>&1 | tee /home/srl/agriculture_hiprag_recovery_$(date +%Y%m%d_%H%M).log
```

如果某个 tmux 窗口还在运行，不要在另一个窗口重复启动同一 dataset/method；先等它完成或确认失败。

### 2.1.6 2026-05-24 20:36 CST 最新 tmux 判定

本节基于 2026-05-24 20:36 CST 左右的 tmux capture。中断发生在 8001 embedding 服务先 HTTP 500 后退出，多个窗口随后出现 `Connection refused` 或 `TimeoutError`。20:35 CST 已在 `embed` session 重启 8001，日志显示：

```text
GET /v1/models HTTP/1.1 200 OK
POST /v1/embeddings HTTP/1.1 200 OK
POST /v1/embeddings HTTP/1.1 200 OK
```

未受影响或已完成，不需要重跑：

```text
cluerag-agriculture:
  cluerag 默认 prompt 已完成两次，最后一次 count=100，stage=baseline_cluerag_full status=ok。
  注意：outputs/agriculture/predictions/cluerag.jsonl 仍只作为 ClueRAG graph/retrieval 中间/历史产物，不进论文。

cluerag-prompt-agri:
  cluerag_prompt_normalized 已完成，count=100，stage=baseline_cluerag_prompt_normalized_generation status=ok。
  论文正式 ClueRAG 使用 outputs/agriculture/predictions/cluerag_prompt_normalized.jsonl。

embed:
  旧 8001 服务因 HTTP 500 退出；20:35 CST 已重启，/v1/models 和 /v1/embeddings smoke 均 200 OK。

llama:
  8000 chat 服务持续 200 OK，仍有请求处理记录。

rerank:
  8033 rerank 服务持续 200 OK。

vanilla-agri:
  vanilla_llm agriculture 已完成，count=100，stage=baseline_vanilla_llm status=ok。

signpost-agriculture:
  agriculture pipeline 已完成到 F10_graph_es_sync。
  Signpost full/no_offline/no_online/no_semantic_cues/no_provenance_cues/no_vertical_cues/no_horizontal_cues 均已完成，count=100，stage status=ok。

signpost-legal:
  仍在 F6 semantic extraction，窗口持续输出 semantic_extract_cache extracting/processed。
  F6 主要依赖 chat，不依赖 8001；未见 failed stage，继续观察，不从头重跑。
```

已被 8001 中断，需要单独恢复：

```text
signpost-agri-en1:
  HiPRAG agriculture 失败，stage=baseline_hiprag status=failed。
  失败原因：8001 Connection refused。20:35 CST 重启 8001 后可单独重跑 hiprag agriculture。

sp-medical:
  graphrag-bench-medical no_offline 失败两次。
  第一次：stage=F15_agent_batch_signpost.no_offline status=failed wall_time_seconds=1663.251。
  第二次恢复命令在 19:25 CST 开始，但 8001 仍未恢复，20:05 CST 再次失败，stage=F15_agent_batch_signpost.no_offline status=failed wall_time_seconds=2407.221。
  恢复起点仍为 no_offline。

sp-mix:
  mix no_online 恢复成功，count=130，stage=F15_agent_batch_signpost.no_online status=ok。
  后续 no_semantic_cues 在 20:05 CST 因 8001 Connection refused 失败，stage=F15_agent_batch_signpost.no_semantic_cues status=failed wall_time_seconds=3926.054。
  恢复起点改为 no_semantic_cues，不再重跑 no_online。

sp-novel:
  graphrag-bench-novel pipeline 已完成到 F10_graph_es_sync。
  Signpost full 因 8001 TimeoutError 失败，stage=F15_agent_batch_signpost.full status=failed wall_time_seconds=11662.112。
  恢复起点为 full。
```

### 2.1.7 先应用 baseline embedding 热修包

本地已生成的热修包：

```text
/home/ruolinsu/signpost/h200/baseline_embedding_recovery_hotfix_20260524_1229.tar.gz
```

手动上传到 H200：

```text
H200 目标：/home/srl/baseline_embedding_recovery_hotfix_20260524_1229.tar.gz
```

H200 解压和静态检查：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/baseline_embedding_recovery_hotfix_20260524_1229.tar.gz

conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re

python -m py_compile \
  signpost/baselines/agrag.py \
  signpost/baselines/linearrag.py \
  signpost/baselines/hiprag.py \
  scripts/baselines/run_agrag.py \
  scripts/baselines/run_linearrag.py \
  scripts/baselines/run_hiprag.py
bash -n scripts/baselines/run_baseline_method.sh scripts/baselines/run_cluerag_method.sh
```

热修包内容：

```text
signpost/baselines/agrag.py
signpost/baselines/linearrag.py
signpost/baselines/hiprag.py
scripts/baselines/run_agrag.py
scripts/baselines/run_linearrag.py
scripts/baselines/run_hiprag.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

应用热修后，再确认 8001 健康：

```bash
ss -ltnp | grep 8001
curl -sS http://localhost:8001/v1/models | head
curl -sS http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["embedding health check"]}' | head
python -m signpost.llm.smoke --embedding
```

恢复命令按数据集合并为一整条顺序执行命令。命令使用 `&&` 串联：前一个 variant 成功才会继续下一个；任意一步失败会停止，避免在服务异常时继续产生连锁失败。

说明：`tee` 保存的是额外终端日志，便于排查；正式指标不依赖终端复制。论文统计需要的 stage timing、prediction、query metrics、method summary、cost quality 会由脚本写入：

```text
outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/logs/*.query.jsonl
outputs/<dataset>/predictions/*.jsonl
outputs/<dataset>/metrics/*.json
```

因此恢复命令应保留 `tee`，但不能用 `tee` 替代上述结构化指标文件。

通用环境初始化：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
```

Agriculture 恢复 HiPRAG，一整条执行：

```bash
{
  date
  USE_ES=0 \
  MODE=hybrid \
  MAX_CONTEXT_TOKENS=2500 \
  HIPRAG_EMBED_BATCH_SIZE=32 \
  HIPRAG_EMBED_RETRIES=3 \
  HIPRAG_EMBED_RETRY_SLEEP=5 \
  HIPRAG_SEARCH_TOP_K=3 \
  HIPRAG_MAX_STEPS=4 \
  scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture
  date
} 2>&1 | tee /home/srl/agriculture_hiprag_recovery_$(date +%Y%m%d_%H%M).log
```

Mix 从 `no_semantic_cues` 继续，一整条顺序执行：

```bash
{
  date
  scripts/run_signpost_method.sh mix no_semantic_cues mix && \
  scripts/run_signpost_method.sh mix no_provenance_cues mix && \
  scripts/run_signpost_method.sh mix no_vertical_cues mix && \
  scripts/run_signpost_method.sh mix no_horizontal_cues mix
  date
} 2>&1 | tee /home/srl/mix_recovery_$(date +%Y%m%d_%H%M).log
```

GraphRAG-Bench medical 从 `no_offline` 继续，一整条顺序执行：

```bash
{
  date
  scripts/run_signpost_method.sh graphrag-bench-medical no_offline graphrag-bench-medical && \
  scripts/run_signpost_method.sh graphrag-bench-medical no_online graphrag-bench-medical && \
  scripts/run_signpost_method.sh graphrag-bench-medical no_semantic_cues graphrag-bench-medical && \
  scripts/run_signpost_method.sh graphrag-bench-medical no_provenance_cues graphrag-bench-medical && \
  scripts/run_signpost_method.sh graphrag-bench-medical no_vertical_cues graphrag-bench-medical && \
  scripts/run_signpost_method.sh graphrag-bench-medical no_horizontal_cues graphrag-bench-medical
  date
} 2>&1 | tee /home/srl/medical_recovery_$(date +%Y%m%d_%H%M).log
```

GraphRAG-Bench novel 从 `full` 继续，一整条顺序执行：

```bash
{
  date
  scripts/run_signpost_method.sh graphrag-bench-novel full graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_offline graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_online graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_semantic_cues graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_provenance_cues graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_vertical_cues graphrag-bench-novel && \
  scripts/run_signpost_method.sh graphrag-bench-novel no_horizontal_cues graphrag-bench-novel
  date
} 2>&1 | tee /home/srl/novel_recovery_$(date +%Y%m%d_%H%M).log
```

Legal 继续监控：

```bash
tmux capture-pane -pt signpost-legal:0.0 -S -80
tail -n 20 outputs/legal/logs/stage_timing.jsonl
```

如果 legal pipeline 完成并输出 `[signpost-pipeline] done dataset=legal`，再按 legal 小节继续执行 Signpost ablation suite 和 baselines。若 legal 出现 `status=failed`，先记录失败 stage，不要直接从 F3 重跑。

查看当前 tmux 状态：

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_current_path} active=#{pane_active}'

for p in $(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}'); do
  echo "===== $p ====="
  tmux capture-pane -pt "$p" -S -40
done > /home/srl/tmux_recovery_status_20260524.txt
```

检查失败 stage：

```bash
grep '"status":"failed"' outputs/*/logs/stage_timing.jsonl > /home/srl/failed_stages_20260524.txt || true
```

检查 prediction 行数：

```bash
for d in agriculture legal mix graphrag-bench-medical graphrag-bench-novel; do
  echo "===== $d ====="
  wc -l outputs/$d/predictions/*.jsonl 2>/dev/null
  tail -n 8 outputs/$d/logs/stage_timing.jsonl 2>/dev/null
done > /home/srl/prediction_counts_20260524.txt
```

## 3. Agriculture 数据集

Agriculture 已经跑过多轮。若需要重新完整跑，使用本节命令；若只补某个 baseline，可以从对应 baseline 命令开始，但必须确认前置文件存在。

创建 tmux：

```bash
tmux new -s sp-agriculture
```

首次执行：窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0

DATASET=agriculture
NAMESPACE=agriculture

scripts/run_signpost_dataset_pipeline.sh "$DATASET" "$NAMESPACE"
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 AGRAG-owned entity/relation/passage graph，使用 PPR + MCMI 子图和 hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag agriculture agriculture

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 LinearRAG-owned relation-free entity/sentence/passage graph，使用 sentence bridging + PPR + hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag agriculture agriculture

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions，
# 构建 HiPRAG-owned 本地 agentic chunk retrieval index，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 GraphRAG-R1-owned agentic graph retrieval artifacts，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 agriculture agriculture
```

后续执行：窗口内执行（非首次，复用 baseline-owned graph/index；直接复制本代码框）：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
# 非首次执行：复用 baseline-owned graph/index。
# ClueRAG 复用 shared_graph 和自有 ES index；AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 复用 index.pkl。
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

DATASET=agriculture
NAMESPACE=agriculture

# 非首次执行不重跑 Signpost dataset pipeline / offline index 构建；
# 仍然重跑 Signpost full/ablations 的 online query/eval。
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/agrag/index.pkl，只重跑 PPR + MCMI 子图检索、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag agriculture agriculture

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/linearrag/index.pkl，只重跑 sentence bridging + PPR、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag agriculture agriculture

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/hiprag/index.pkl，只重跑 agentic chunk retrieval 和 final generation，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/graphrag_r1/index.pkl，只重跑 agentic graph retrieval 和 final generation，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 agriculture agriculture
```

Agriculture 当前论文口径中，`outputs/agriculture/predictions/cluerag.jsonl` 不进入论文；使用：

```text
outputs/agriculture/predictions/cluerag_prompt_normalized.jsonl
```

## 4. Legal 数据集

Legal 已经跑过主流程和部分 baseline。若重新完整跑，命令如下。Legal 的 F6 semantic extraction 可能耗时较长，必须保留 progress/cache，不要删除：

```text
datasets/processed/legal/semantic_llm.progress.jsonl
datasets/processed/legal/semantic_llm.extractions.jsonl
```

### 4.0 Legal F9 后 F10 embedding 中断自动恢复

2026-05-26 Legal 已完成 F6/F7/F8/F9，F10 `graph_es_sync` 期间 8001 embedding 服务多次 `EngineDeadError` 退出。此时不要从 F3 或 F6 重跑，也不要直接反复执行带 `--recreate` 的旧 F10 命令。

当前正式恢复口径：

```text
1. 默认 Signpost F10 不变：正常 graph object 仍是一 object、一 ES document、一 content_vector。
2. F10 auto-recovery 只在 Legal 恢复命令中显式启用。
3. 每个成功写入的 vector document 会写入 progress log；中断后从 checkpoint 继续，不从头重跑。
4. 如果同一个原始 graph object 连续失败，才只对这个 object 启用 multi-vector 子文档 fallback。
5. multi-vector fallback 不截断 content；失败 object 的完整 content 被拆成多个 ES 子文档，检索时按 graph_parent_id 归并。
6. 该恢复机制不调用 LLM，不新增 chat token 成本；只影响 F10 offline embedding/index wall time 和 embedding 调用次数。
```

对应审计文件：

```text
outputs/legal/logs/F10_graph_es_sync.progress.jsonl
outputs/legal/logs/F10_graph_es_sync.state.json
outputs/legal/logs/F10_graph_es_sync.multivector_parts.json
outputs/legal/logs/F10_graph_es_sync.recovery_decisions.jsonl
outputs/legal/logs/F10_graph_es_sync.multivector_objects.jsonl
```

论文说明记录在：

```text
signpost/paper_drafts/v10_metric_revision/f10_legal_embedding_recovery.md
```

H200 上应用包含该恢复机制的 patch 后，在原来的 `signpost-re` tmux 窗口中执行。该命令会自动在 `embed` tmux session 中用固定命令重启 8001：

```bash
cd /data/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/data/srl/signpost_re
export RAG_PROJECT_BASE=/data/srl/signpost_re

bash scripts/h200/run_legal_after_f9_auto_recovery.sh legal legal
```

`embed` tmux session 中自动使用的启动命令固定为：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_USE_DEEP_GEMM=0 python -m vllm.entrypoints.openai.api_server --model /data/srl/nemotron-8b --runner pooling --port 8001 --trust-remote-code
```

如果该命令最终成功，会继续顺序执行：

```text
F10_graph_es_sync auto-recovery
index_metrics
Signpost full + all ablations
vanilla_llm
vanilla_rag
hybrid_rag
cluerag
cluerag_prompt_normalized
agrag
linearrag
hiprag
graphrag_r1
basic/query/method/cost metrics recomputation
final_metrics if analysis_legal/targets exists
```

创建 tmux：

```bash
tmux new -s sp-legal
```

首次执行：窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0

DATASET=legal
NAMESPACE=legal

scripts/run_signpost_dataset_pipeline.sh "$DATASET" "$NAMESPACE"
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 AGRAG-owned entity/relation/passage graph，使用 PPR + MCMI 子图和 hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag legal legal

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 LinearRAG-owned relation-free entity/sentence/passage graph，使用 sentence bridging + PPR + hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag legal legal

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions，
# 构建 HiPRAG-owned 本地 agentic chunk retrieval index，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag legal legal

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 GraphRAG-R1-owned agentic graph retrieval artifacts，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 legal legal
```

后续执行：窗口内执行（非首次，复用 baseline-owned graph/index；直接复制本代码框）：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
# 非首次执行：复用 baseline-owned graph/index。
# ClueRAG 复用 shared_graph 和自有 ES index；AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 复用 index.pkl。
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

DATASET=legal
NAMESPACE=legal

# 非首次执行不重跑 Signpost dataset pipeline / offline index 构建；
# 仍然重跑 Signpost full/ablations 的 online query/eval。
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/agrag/index.pkl，只重跑 PPR + MCMI 子图检索、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag legal legal

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/linearrag/index.pkl，只重跑 sentence bridging + PPR、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag legal legal

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/hiprag/index.pkl，只重跑 agentic chunk retrieval 和 final generation，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag legal legal

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/graphrag_r1/index.pkl，只重跑 agentic graph retrieval 和 final generation，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 legal legal
```

Legal 当前论文口径中，`outputs/legal/predictions/cluerag.jsonl` 不进入论文；使用：

```text
outputs/legal/predictions/cluerag_prompt_normalized.jsonl
```

## 5. Mix 数据集

创建 tmux：

```bash
tmux new -s sp-mix
```

首次执行：窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0

DATASET=mix
NAMESPACE=mix

scripts/run_signpost_dataset_pipeline.sh "$DATASET" "$NAMESPACE"
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 AGRAG-owned entity/relation/passage graph，使用 PPR + MCMI 子图和 hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag mix mix

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 LinearRAG-owned relation-free entity/sentence/passage graph，使用 sentence bridging + PPR + hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag mix mix

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions，
# 构建 HiPRAG-owned 本地 agentic chunk retrieval index，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag mix mix

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 GraphRAG-R1-owned agentic graph retrieval artifacts，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 mix mix
```

后续执行：窗口内执行（非首次，复用 baseline-owned graph/index；直接复制本代码框）：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
# 非首次执行：复用 baseline-owned graph/index。
# ClueRAG 复用 shared_graph 和自有 ES index；AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 复用 index.pkl。
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

DATASET=mix
NAMESPACE=mix

# 非首次执行不重跑 Signpost dataset pipeline / offline index 构建；
# 仍然重跑 Signpost full/ablations 的 online query/eval。
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/agrag/index.pkl，只重跑 PPR + MCMI 子图检索、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag mix mix

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/linearrag/index.pkl，只重跑 sentence bridging + PPR、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag mix mix

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/hiprag/index.pkl，只重跑 agentic chunk retrieval 和 final generation，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag mix mix

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/graphrag_r1/index.pkl，只重跑 agentic graph retrieval 和 final generation，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 mix mix
```

脱离 tmux：

```text
Ctrl-b d
```

恢复窗口：

```bash
tmux attach -t sp-mix
```

## 6. GraphRAG-Bench Medical 数据集

创建 tmux：

```bash
tmux new -s sp-medical
```

首次执行：窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0

DATASET=graphrag-bench-medical
NAMESPACE=graphrag-bench-medical

scripts/run_signpost_dataset_pipeline.sh "$DATASET" "$NAMESPACE"
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 AGRAG-owned entity/relation/passage graph，使用 PPR + MCMI 子图和 hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-medical graphrag-bench-medical

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 LinearRAG-owned relation-free entity/sentence/passage graph，使用 sentence bridging + PPR + hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-medical graphrag-bench-medical

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions，
# 构建 HiPRAG-owned 本地 agentic chunk retrieval index，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-medical graphrag-bench-medical

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 GraphRAG-R1-owned agentic graph retrieval artifacts，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-medical graphrag-bench-medical
```

后续执行：窗口内执行（非首次，复用 baseline-owned graph/index；直接复制本代码框）：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
# 非首次执行：复用 baseline-owned graph/index。
# ClueRAG 复用 shared_graph 和自有 ES index；AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 复用 index.pkl。
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

DATASET=graphrag-bench-medical
NAMESPACE=graphrag-bench-medical

# 非首次执行不重跑 Signpost dataset pipeline / offline index 构建；
# 仍然重跑 Signpost full/ablations 的 online query/eval。
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/agrag/index.pkl，只重跑 PPR + MCMI 子图检索、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-medical graphrag-bench-medical

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/linearrag/index.pkl，只重跑 sentence bridging + PPR、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-medical graphrag-bench-medical

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/hiprag/index.pkl，只重跑 agentic chunk retrieval 和 final generation，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-medical graphrag-bench-medical

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/graphrag_r1/index.pkl，只重跑 agentic graph retrieval 和 final generation，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-medical graphrag-bench-medical
```


## 7. GraphRAG-Bench Novel 数据集

创建 tmux：

```bash
tmux new -s sp-novel
```

首次执行：窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export REUSE_BASELINE_INDEX=0
export REUSE_GRAPH=0

DATASET=graphrag-bench-novel
NAMESPACE=graphrag-bench-novel

scripts/run_signpost_dataset_pipeline.sh "$DATASET" "$NAMESPACE"
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 AGRAG-owned entity/relation/passage graph，使用 PPR + MCMI 子图和 hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-novel graphrag-bench-novel

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 LinearRAG-owned relation-free entity/sentence/passage graph，使用 sentence bridging + PPR + hybrid chunk retrieval。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-novel graphrag-bench-novel

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions，
# 构建 HiPRAG-owned 本地 agentic chunk retrieval index，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-novel graphrag-bench-novel

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions，
# 构建 GraphRAG-R1-owned agentic graph retrieval artifacts，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-novel graphrag-bench-novel
```

后续执行：窗口内执行（非首次，复用 baseline-owned graph/index；直接复制本代码框）：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
# 非首次执行：复用 baseline-owned graph/index。
# ClueRAG 复用 shared_graph 和自有 ES index；AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 复用 index.pkl。
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

DATASET=graphrag-bench-novel
NAMESPACE=graphrag-bench-novel

# 非首次执行不重跑 Signpost dataset pipeline / offline index 构建；
# 仍然重跑 Signpost full/ablations 的 online query/eval。
scripts/run_signpost_ablation_suite.sh "$DATASET" "$NAMESPACE"

scripts/baselines/run_baseline_method.sh vanilla_llm "$DATASET" "$NAMESPACE"
scripts/baselines/run_baseline_method.sh hybrid_rag "$DATASET" "$NAMESPACE"

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"

# 论文正式 ClueRAG baseline：统一使用 Signpost evidence-grounded 生成约束。
# 这一步复用上一条 cluerag 的 retrieval_results，只重跑 final generation；上一条 cluerag 默认 prompt 结果不进论文。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR

# 论文正式 AGRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/agrag/index.pkl，只重跑 PPR + MCMI 子图检索、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export TOP_K=5
export GRAPH_TOP_K=5
export LINK_TOP_K=8
export PPR_ALPHA=0.85
export MCMI_STEPS=20
export MAX_CONTEXT_TOKENS=3500
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export AGRAG_EMBED_BATCH_SIZE=32
export EMBEDDING_PROVIDER=ecnu
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-novel graphrag-bench-novel

# 论文正式 LinearRAG baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/linearrag/index.pkl，只重跑 sentence bridging + PPR、hybrid chunk retrieval 和 final generation。
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export LINEARRAG_EMBED_BATCH_SIZE=32
export LINEARRAG_EMBED_RETRIES=3
export LINEARRAG_EMBED_RETRY_SLEEP=5
export LINEARRAG_RETRIEVAL_TOP_K=5
export LINEARRAG_HYBRID_TOP_K=5
export LINEARRAG_SEED_TOP_K=8
export LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3
export LINEARRAG_ITERATION_THRESHOLD=0.5
export LINEARRAG_PASSAGE_RATIO=1.5
export LINEARRAG_PASSAGE_NODE_WEIGHT=0.05
export LINEARRAG_DAMPING=0.5
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-novel graphrag-bench-novel

# 论文正式 HiPRAG baseline：复用 Signpost chunks / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/hiprag/index.pkl，只重跑 agentic chunk retrieval 和 final generation，保留 <think>/<search>/<context>/<answer> XML 输出契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export HIPRAG_EMBED_BATCH_SIZE=32
export HIPRAG_EMBED_RETRIES=3
export HIPRAG_EMBED_RETRY_SLEEP=5
export HIPRAG_SEARCH_TOP_K=3
export HIPRAG_MAX_STEPS=4
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-novel graphrag-bench-novel

# 论文正式 GraphRAG-R1 baseline：复用 Signpost chunks / semantic_llm extractions / questions。
# 非首次执行复用 outputs/${DATASET}/baselines/graphrag_r1/index.pkl，只重跑 agentic graph retrieval 和 final generation，保留 <think>/<answer> 和 <|begin_of_query|> 检索标签契约。
export USE_ES=0
export MODE=hybrid
export MAX_CONTEXT_TOKENS=2500
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_EMBED_RETRIES=3
export GRAPHRAG_R1_EMBED_RETRY_SLEEP=5
export GRAPHRAG_R1_GRAPH_TOP_K=5
export GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8
export GRAPHRAG_R1_MAX_STEPS=4
export GRAPHRAG_R1_PPR_ALPHA=0.85
export GRAPHRAG_R1_PPR_ITERATIONS=20
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-novel graphrag-bench-novel
```


## 8. 进度检查

检查当前正在跑的任务：

```bash
tmux ls
ps -ef | grep -E 'run_signpost|run_cluerag|run_baseline|signpost.agent|semantic_graph' | grep -v grep
```

检查某个数据集是否完成 Signpost 共享阶段：

```bash
DATASET=mix

test -s datasets/processed/${DATASET}/questions.jsonl
test -s datasets/processed/${DATASET}/chunks.jsonl
test -s datasets/processed/${DATASET}/semantic_llm.extractions.jsonl
test -s datasets/processed/${DATASET}/graph.unified.json
test -s outputs/${DATASET}/metrics/index_metrics.json
```

检查某个数据集的 Signpost full 和消融结果：

```bash
DATASET=mix

wc -l outputs/${DATASET}/predictions/signpost.*.jsonl
ls -lh outputs/${DATASET}/metrics/signpost.*.query_metrics.json
```

检查 baseline 结果：

```bash
DATASET=mix

wc -l outputs/${DATASET}/predictions/vanilla_llm.jsonl
wc -l outputs/${DATASET}/predictions/hybrid_rag.jsonl
wc -l outputs/${DATASET}/predictions/cluerag.jsonl
wc -l outputs/${DATASET}/predictions/agrag.jsonl
wc -l outputs/${DATASET}/predictions/linearrag.jsonl
wc -l outputs/${DATASET}/predictions/hiprag.jsonl
wc -l outputs/${DATASET}/predictions/graphrag_r1.jsonl
wc -l outputs/${DATASET}/logs/vanilla_llm.query.jsonl
wc -l outputs/${DATASET}/logs/hybrid_rag.query.jsonl
wc -l outputs/${DATASET}/logs/cluerag.query.jsonl
wc -l outputs/${DATASET}/logs/agrag.query.jsonl
wc -l outputs/${DATASET}/logs/linearrag.query.jsonl
wc -l outputs/${DATASET}/logs/hiprag.query.jsonl
wc -l outputs/${DATASET}/logs/graphrag_r1.query.jsonl
ls -lh outputs/${DATASET}/baselines/cluerag/run_metrics.json
ls -lh outputs/${DATASET}/baselines/agrag/run_metrics.json
ls -lh outputs/${DATASET}/baselines/linearrag/run_metrics.json
ls -lh outputs/${DATASET}/baselines/hiprag/run_metrics.json
ls -lh outputs/${DATASET}/baselines/graphrag_r1/run_metrics.json
```

检查 ClueRAG 自有 ES 图索引：

```bash
DATASET=mix
curl -s "http://127.0.0.1:9200/_cat/indices/cluerag-${DATASET}-multilayer?v"
curl -s "http://127.0.0.1:9200/cluerag-${DATASET}-multilayer/_count?pretty"
```

## 9. 断点与重跑

如果 ClueRAG 是首次正式运行，显式使用 `REUSE_GRAPH=0`，这会构建 ClueRAG-owned shared_graph 和 ES index：

```bash
DATASET=mix
NAMESPACE=mix

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export REUSE_GRAPH=0
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
```

如果 ClueRAG 已经完成图构建和 ES index，但在线查询或生成中断，可以复用图，只重跑 online query 和 final generation：

```bash
DATASET=mix
NAMESPACE=mix

export CLUERAG_BACKEND=shared_es
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export REUSE_GRAPH=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
```

如果只需要把已有 ClueRAG raw output 转回统一 prediction，不重新检索或生成：

```bash
DATASET=mix
NAMESPACE=mix

export CONVERT_ONLY=1
export OFFICIAL_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CONVERT_ONLY
unset OFFICIAL_OUTPUT_DIR
```

注意：`CONVERT_ONLY=1` 只适合 raw output 已经完整存在的情况。

## 10. 跑完后重算指标

单个数据集重算 basic/query/method/cost：

```bash
DATASET=mix

for pred in outputs/${DATASET}/predictions/*.jsonl; do
  method=$(basename "$pred" .jsonl)
  python -m signpost.evaluation.evaluate_basic \
    --input "$pred" \
    --output "outputs/${DATASET}/metrics/${method}.basic_eval.json" \
    --normalize
  python -m signpost.benchmark.query_metrics \
    --input "$pred" \
    --output "outputs/${DATASET}/metrics/${method}.query_metrics.json" \
    --normalize --top-k 5 10
done

printf '[]\n' > outputs/${DATASET}/metrics/method_summaries.json

for qm in outputs/${DATASET}/metrics/*.query_metrics.json; do
  method=$(basename "$qm" .query_metrics.json)
  if [ "$method" = "cluerag" ]; then
    python -m signpost.benchmark.method_summary \
      --method "$method" \
      --dataset "$DATASET" \
      --query-metrics "$qm" \
      --stage-log "outputs/${DATASET}/logs/stage_timing.jsonl" \
      --offline-stage baseline_prepare_cluerag \
      --offline-stage baseline_cluerag_full \
      --output "outputs/${DATASET}/metrics/method_summaries.json"
  elif [ "$method" = "hybrid_rag" ] || [ "$method" = "vanilla_rag" ]; then
    python -m signpost.benchmark.method_summary \
      --method "$method" \
      --dataset "$DATASET" \
      --query-metrics "$qm" \
      --stage-log "outputs/${DATASET}/logs/stage_timing.jsonl" \
      --offline-stage F5_chunk_index \
      --output "outputs/${DATASET}/metrics/method_summaries.json"
  elif [[ "$method" == signpost.* ]]; then
    python -m signpost.benchmark.method_summary \
      --method "$method" \
      --dataset "$DATASET" \
      --query-metrics "$qm" \
      --stage-log "outputs/${DATASET}/logs/stage_timing.jsonl" \
      --offline-stage F7_structure_graph \
      --offline-stage F8_sequence_graph \
      --offline-stage F9_unified_graph \
      --offline-stage F10_graph_es_sync \
      --output "outputs/${DATASET}/metrics/method_summaries.json"
  else
    python -m signpost.benchmark.method_summary \
      --method "$method" \
      --dataset "$DATASET" \
      --query-metrics "$qm" \
      --output "outputs/${DATASET}/metrics/method_summaries.json"
  fi
done

python -m signpost.benchmark.cost_quality \
  --methods "outputs/${DATASET}/metrics/method_summaries.json" \
  --output "outputs/${DATASET}/metrics/cost_quality.json"
```

`final_metrics.py` 需要 silver evidence 和 target files。若该数据集已经有：

```text
analysis_<dataset>/targets/silver_evidence_chunks.jsonl
analysis_<dataset>/targets/target_entities.jsonl
analysis_<dataset>/targets/target_units.jsonl
analysis_<dataset>/targets/claim_units.jsonl
```

则执行：

```bash
DATASET=mix
ANALYSIS_DIR=analysis_${DATASET}

python -m signpost.benchmark.final_metrics \
  --predictions-dir outputs/${DATASET}/predictions \
  --targets-dir ${ANALYSIS_DIR}/targets \
  --output-dir ${ANALYSIS_DIR} \
  --chunks-file datasets/processed/${DATASET}/chunks.jsonl \
  --offline-stage-timing outputs/${DATASET}/logs/stage_timing.jsonl \
  --online-stage-timing outputs/${DATASET}/logs/stage_timing.jsonl
```

如果 targets 尚不存在，不要在正式 baseline 中途临时构造；等该数据集所有 prediction 固定后，再按评测协议统一生成 targets，并记录生成脚本和时间。

## 11. Baseline final generation prompt registry

维护要求：

```text
之后每新增或跑通一个 baseline，必须在本节登记该 baseline 的 final generation 完整 prompt。
登记内容必须包括：
1. method name
2. 是否保留 baseline 自己的输出格式
3. 进入论文的 prediction 文件名
4. 完整 system/user prompt 模板
5. 哪些默认/官方 prompt 结果被弃用
```

### 11.1 Signpost / JSON 输出格式通用 prompt

适用方法：

```text
signpost.full
signpost.* ablations
vanilla_llm
hybrid_rag
后续任何可以稳定输出 JSON {"rationale", "answer"} 的 baseline
```

完整 system prompt：

```text
Answer the question in English strictly based on the provided evidence.
You must format your output as a valid JSON object containing exactly two keys: "rationale" and "answer".

Follow these rules:
1. "rationale": Briefly analyze the core intent of the question and identify the relevant facts from the evidence. Keep your step-by-step thinking and analysis in this field.
2. "answer": Provide the final response text here.
   - Write complete, well-formed sentences that fully answer the question.
   - Include all necessary context and details supported by the evidence so that the answer is comprehensive and stands alone clearly.
   - DO NOT include citations (e.g., [file.txt:L1-L3]), file names, or line numbers. Source tracking is handled externally.
   - DO NOT include conversational filler (e.g., "Based on the provided text...", "According to the evidence...") or your reasoning process here.
   - If the evidence is insufficient to answer the question, output exactly: "Insufficient evidence."

Example Output:
```json
{
  "rationale": "The question asks about the specific innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence lists hydroponic growing, aquaponics, composting, and biodiesel production, alongside community engagement efforts.",
  "answer": "Greensgrow Farm employs innovative practices such as hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. They also focus on community engagement and education to promote sustainable food practices."
}
```
```

Signpost synthesis user prompt:

```text
Question:
{question}

Evidence:
{evidence}
```

Hybrid RAG / flat retrieval user prompt should use the same evidence-grounded constraints. If the implementation keeps the constraints in a single user prompt instead of a system prompt, the full text must still be equivalent to the system prompt above, followed by:

```text
Question:
{question}

Evidence:
{retrieved_context}
```

Vanilla LLM has no retrieved evidence. For paper comparability, it should still use the same JSON output schema and answer constraint, with evidence explicitly set to `None` or `No retrieved evidence provided`; it may answer `Insufficient evidence.` when the prompt requires strict evidence grounding.

### 11.2 ClueRAG / Thought-Answer 输出格式 prompt

论文不使用 ClueRAG 默认 final generation prompt 的结果。`outputs/<dataset>/predictions/cluerag.jsonl` 只作为历史/中间产物保留，不进入任何论文表格。论文中的 ClueRAG baseline 使用统一生成约束后的结果：

```text
cluerag_prompt_normalized
```

这个正式论文 baseline 只允许替换 final generation prompt 的任务说明与 few-shot 示例部分：

- 不重建 ClueRAG 图；
- 不改 ClueRAG 检索；
- 不改 rerank；
- 不改 retrieved chunks；
- 不改 `Thought:` / `Answer:` 输出格式；
- 不把 `outputs/<dataset>/predictions/cluerag.jsonl` 写入论文；论文使用 `outputs/<dataset>/predictions/cluerag_prompt_normalized.jsonl`。
- 保留 few-shot，但使用 evidence-grounded 的 Greensgrow 示例，而不是 ClueRAG 原始 Wikipedia 示例。

ClueRAG 原始 prompt 的关键输出格式约束是：

```text
Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
```

注意：不要直接套用 Signpost 的 JSON few-shot。Signpost 的示例输出是 JSON `{"rationale": ..., "answer": ...}`，如果直接套到 ClueRAG，会同时改变 prompt 内容和输出格式，导致结果不可解释。这里采用更干净的做法：保留 ClueRAG 的 `Thought:` / `Answer:` 输出契约，只把任务规则和 few-shot 内容改成 evidence-grounded 版本。

替换后的 prompt 是：

```text
As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence. Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence. Conclude with "Answer: " to present a complete, well-formed final response.

Follow these rules:
- Include all necessary context and details supported by the evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".

Example Input:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.

Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?
Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence lists hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.

Real Input:
{context}

Question: {question}
```

与原始 ClueRAG prompt 相比，这个版本把 Wikipedia few-shot 替换成 evidence-grounded few-shot，并把任务说明改成与 Signpost synthesis 规则一致的证据约束。输出格式仍然是：

```text
Thought: ...
Answer: ...
```

论文正式 ClueRAG baseline 输出：

```text
outputs/<dataset>/predictions/cluerag_prompt_normalized.jsonl
outputs/<dataset>/logs/cluerag_prompt_normalized.query.jsonl
outputs/<dataset>/metrics/cluerag_prompt_normalized.basic_eval.json
outputs/<dataset>/metrics/cluerag_prompt_normalized.query_metrics.json
```

论文写法建议：

```text
For ClueRAG, we keep its graph construction, retrieval, reranking, and Thought/Answer output contract unchanged, but replace its final generation instruction with the same evidence-grounded answer constraints used by Signpost. The default ClueRAG generation prompt is not reported in the main results.
```

### 11.3 AGRAG / Thought-Answer 输出格式 prompt

论文中的 AGRAG baseline 使用：

```text
agrag
```

AGRAG adapter 复用 Signpost 共享产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

它只在 baseline-owned 目录构建 AGRAG 图和 triple artifacts：

```text
outputs/<dataset>/baselines/agrag/graph.json
outputs/<dataset>/baselines/agrag/triples.jsonl
```

AGRAG 检索口径：

- 不重新切 chunk；
- 不重新抽实体或关系；
- 用共享 semantic extraction 构建 entity / relation / passage graph；
- 用 query-to-triple embedding 相似度选 anchor triples；
- 用 PPR 估计 graph influence；
- 用 MCMI-style greedy expansion 生成 reasoning subgraph；
- 与 hybrid chunk retrieval 融合后进入 final generation。

AGRAG 原始 HippoRAG-style QA prompt 使用 `Thought:` / `Answer:` 输出契约。正式 adapter 保留这个输出契约，只迁移 Signpost evidence-grounded 回答约束。论文正式输出：

```text
outputs/<dataset>/predictions/agrag.jsonl
outputs/<dataset>/logs/agrag.query.jsonl
outputs/<dataset>/metrics/agrag.basic_eval.json
outputs/<dataset>/metrics/agrag.query_metrics.json
outputs/<dataset>/baselines/agrag/run_metrics.json
outputs/<dataset>/baselines/agrag/run_status.json
```

完整 system prompt：

```text
As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence and AGRAG reasoning subgraph. Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence and reasoning subgraph. Conclude with "Answer: " to present a complete, well-formed final response.

Follow these rules:
- Include all necessary context and details supported by the evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".

Example Input:
AGRAG MCMI reasoning subgraph:
(Greensgrow Farm [SEP] uses [SEP] hydroponic growing)
(Greensgrow Farm [SEP] uses [SEP] aquaponics)
(Greensgrow Farm [SEP] promotes [SEP] community engagement and education)

Evidence:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.

Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?
Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence and AGRAG reasoning subgraph identify hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.
```

完整 user prompt 模板：

```text
AGRAG MCMI reasoning subgraph:
{reasoning_paths}

Evidence:
{retrieved_context}

Question: {question}
Thought:
```

### 11.4 LinearRAG / Thought-Answer 输出格式 prompt

论文中的 LinearRAG baseline 使用：

```text
linearrag
```

LinearRAG adapter 复用 Signpost 共享产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

它只在 baseline-owned 目录构建 LinearRAG 图和 index artifacts：

```text
outputs/<dataset>/baselines/linearrag/graph.json
outputs/<dataset>/baselines/linearrag/entities.jsonl
outputs/<dataset>/baselines/linearrag/sentences.jsonl
outputs/<dataset>/baselines/linearrag/passage_links.jsonl
```

LinearRAG 检索口径：

- 不重新切 chunk；
- 不重新跑 spaCy/NER 或 LLM entity extraction；
- 用共享 semantic extraction 中的 entities 作为 entity nodes；
- 用 chunk 内容切分 sentence bridge nodes，但不引入新实体；
- 构建 relation-free entity / sentence / passage graph；
- 用 question-to-entity embedding 和 lexical overlap 选 seed entities；
- 用 sentence bridging 扩展 activated entities；
- 用 passage dense score + activated entity score 形成 PPR personalization；
- 用 personalized PageRank 排序 passage nodes，并与 hybrid chunk retrieval 融合后进入 final generation。

LinearRAG 官方 QA prompt 使用 `Thought:` / `Answer:` 输出契约。正式 adapter 保留这个输出契约，只迁移 Signpost evidence-grounded 回答约束。论文正式输出：

```text
outputs/<dataset>/predictions/linearrag.jsonl
outputs/<dataset>/logs/linearrag.query.jsonl
outputs/<dataset>/metrics/linearrag.basic_eval.json
outputs/<dataset>/metrics/linearrag.query_metrics.json
outputs/<dataset>/baselines/linearrag/run_metrics.json
outputs/<dataset>/baselines/linearrag/run_status.json
```

完整 system prompt：

```text
As an advanced reading comprehension assistant, answer the question in English strictly based on the provided retrieved evidence and LinearRAG relation-free reasoning graph. Your response start after "Thought: ", where you briefly analyze the core intent of the question and identify the relevant facts from the evidence and reasoning graph. Conclude with "Answer: " to present a complete, well-formed final response.

Follow these rules:
- Include all necessary context and details supported by the evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the evidence is insufficient, write exactly: "Insufficient evidence." after "Answer: ".

Example Input:
LinearRAG relation-free reasoning graph:
Seed entities: Greensgrow Farm
Activated bridge entities: hydroponic growing; aquaponics; composting; biodiesel production; community engagement and education
Top passages are selected by entity/sentence bridging and personalized PageRank.

Evidence:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.

Question: What innovative practices does Greensgrow Farm use for sustainable urban farming?
Thought: The question asks about the innovative practices Greensgrow Farm uses for sustainable urban farming. The evidence and LinearRAG reasoning graph identify hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
Answer: Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.
```

完整 user prompt 模板：

```text
LinearRAG relation-free reasoning graph:
{graph_summary}

Evidence:
{retrieved_context}

Question: {question}
Thought:
```

### 11.5 HiPRAG / XML Agentic Search 输出格式 prompt

论文中的 HiPRAG baseline 使用：

```text
hiprag
```

HiPRAG adapter 复用 Signpost 共享产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl
```

它不读取 Signpost graph/index/navigation-cue index。正式 H200 口径中，HiPRAG 在 baseline-owned 目录构建本地 chunk retrieval index，并记录 index 与 agent search 指标：

```text
outputs/<dataset>/baselines/hiprag/retrieval_index.json
outputs/<dataset>/baselines/hiprag/run_metrics.json
outputs/<dataset>/baselines/hiprag/run_status.json
```

HiPRAG 检索口径：

- 不重新切 chunk；
- 不重新抽实体或关系；
- 不读 Signpost graph/index/navigation-cue index；
- 用共享 `chunks.jsonl` 构建 HiPRAG-owned 本地 chunk retrieval index；
- 在线阶段保留 HiPRAG agentic search 形态：`<think>` / `<step>` / `<reasoning>` / `<search>` / `<context>` / `<conclusion>` / `<answer>`；
- 搜索工具只返回私有语料 chunks，final answer 必须受 Signpost evidence-grounded 约束。

HiPRAG 官方 prompt 使用 XML agent 输出契约。正式 adapter 保留这个输出契约，只迁移 Signpost evidence-grounded 回答约束。论文正式输出：

```text
outputs/<dataset>/predictions/hiprag.jsonl
outputs/<dataset>/logs/hiprag.query.jsonl
outputs/<dataset>/metrics/hiprag.basic_eval.json
outputs/<dataset>/metrics/hiprag.query_metrics.json
outputs/<dataset>/baselines/hiprag/run_metrics.json
outputs/<dataset>/baselines/hiprag/run_status.json
```

完整 system prompt：

```text
You are a HiPRAG-style agentic retrieval assistant. Answer the question in English strictly based on retrieved evidence from the private corpus. You must preserve HiPRAG's XML output contract: reasoning and tool-use steps go inside a single <think> block, each step uses <step>, <reasoning>, optional <search>, optional <context>, and <conclusion>, and the final response is placed in <answer> after </think>.

Follow these evidence-grounded answer rules:
- Use the search tool to inspect the private corpus when evidence is needed.
- Include all necessary context and details supported by retrieved evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers in <answer>.
- Do not include conversational filler.
- If the retrieved evidence is insufficient, write exactly: "Insufficient evidence." inside <answer>.

Search tool contract:
- To search the corpus, output a single <search>query</search> tag inside the current <step>.
- The system will append retrieved evidence inside <context>...</context>.
- After enough evidence is available, close </think> and write <answer>...</answer>.

Example:
<think>
<step>
    <reasoning>The question asks what practices Greensgrow Farm uses for sustainable urban farming. I need corpus evidence listing those practices.</reasoning>
    <search>Greensgrow Farm sustainable urban farming practices</search>
    <context>Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.</context>
    <conclusion>The evidence identifies hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.</conclusion>
</step>
</think>
<answer>Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.</answer>
```

完整 user prompt 模板：

```text
User Question: {question}
<think>

Continue with the next HiPRAG XML step. If more corpus evidence is needed, include exactly one <search>query</search>. If the accumulated evidence is sufficient, close </think> and provide the final <answer>.
```

每个 search tool 调用后，系统追加：

```text
<context>{retrieved_context}</context>
<conclusion>The search returned {num_chunks} evidence chunks.</conclusion>
```

如果达到 `HIPRAG_MAX_STEPS` 后仍没有 `<answer>`，强制 final generation prompt 是：

```text
{transcript}
Now close the HiPRAG reasoning trace and write the final <answer>. Use only the accumulated retrieved evidence. If it is insufficient, write exactly <answer>Insufficient evidence.</answer>.
```

### 11.6 GraphRAG-R1 / XML Agentic Graph Retrieval 输出格式 prompt

论文中的 GraphRAG-R1 baseline 使用：

```text
graphrag_r1
```

GraphRAG-R1 adapter 复用 Signpost 共享产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

它不读取 Signpost graph/index/navigation-cue index。正式 H200 口径中，GraphRAG-R1 在 baseline-owned 目录构建自己的 agentic graph retrieval artifacts：

```text
outputs/<dataset>/baselines/graphrag_r1/graph.json
outputs/<dataset>/baselines/graphrag_r1/triples.jsonl
outputs/<dataset>/baselines/graphrag_r1/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1/run_status.json
```

GraphRAG-R1 检索口径：

- 不重新切 chunk；
- 不重新抽实体或关系；
- 不读 Signpost graph/index/navigation-cue index；
- 用共享 `semantic_llm.extractions.jsonl` 构建 baseline-owned entity / relation / passage graph；
- 在线阶段保留 GraphRAG-R1 agentic retrieval 形态：`<think>` / `<|begin_of_query|>` / `<|begin_of_documents|>` / `<answer>`；
- graph retrieval 返回 private corpus graph facts 和 chunks，final answer 必须受 Signpost evidence-grounded 约束。

GraphRAG-R1 官方 prompt 使用 XML agent 输出契约和 query/document 检索标签。正式 adapter 保留这个输出契约，只迁移 Signpost evidence-grounded 回答约束。论文正式输出：

```text
outputs/<dataset>/predictions/graphrag_r1.jsonl
outputs/<dataset>/logs/graphrag_r1.query.jsonl
outputs/<dataset>/metrics/graphrag_r1.basic_eval.json
outputs/<dataset>/metrics/graphrag_r1.query_metrics.json
outputs/<dataset>/baselines/graphrag_r1/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1/run_status.json
```

完整 system prompt：

```text
You are a GraphRAG-R1-style agentic graph retrieval assistant. Answer the question in English strictly based on retrieved evidence from the private corpus. Preserve the GraphRAG-R1 output contract: use a single <think> block for reasoning and retrieval, request graph retrieval by writing <|begin_of_query|>query<|end_of_query|>, read returned evidence inside <|begin_of_documents|>...<|end_of_documents|>, and place the final answer in <answer> after </think>.

Follow these evidence-grounded answer rules:
- Use graph retrieval when evidence is needed.
- Include all necessary context and details supported by retrieved evidence.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers in <answer>.
- Do not include conversational filler.
- If the retrieved evidence is insufficient, write exactly: "Insufficient evidence." inside <answer>.

Graph retrieval contract:
- To retrieve evidence, output exactly one query span using <|begin_of_query|> and <|end_of_query|>.
- The system will append graph facts and documents inside <|begin_of_documents|>...<|end_of_documents|>.
- After enough evidence is available, close </think> and write <answer>...</answer>.

Example:
<think>
The question asks what practices Greensgrow Farm uses for sustainable urban farming. I need graph-linked evidence about Greensgrow Farm practices.
<|begin_of_query|>Greensgrow Farm sustainable urban farming practices<|end_of_query|>
<|begin_of_documents|>
Graph facts:
(Greensgrow Farm [SEP] uses [SEP] hydroponic growing)
(Greensgrow Farm [SEP] uses [SEP] aquaponics)

Documents:
Greensgrow Farm uses hydroponic growing, aquaponics, composting, and biodiesel production as part of its sustainable urban farming practices. It also emphasizes community engagement and education to promote sustainable food practices.
<|end_of_documents|>
The evidence identifies hydroponic growing, aquaponics, composting, biodiesel production, and community engagement and education.
</think>
<answer>Greensgrow Farm employs hydroponic growing, aquaponics, composting, and biodiesel production to make urban farming sustainable. It also promotes sustainable food practices through community engagement and education.</answer>
```

完整 user prompt 模板：

```text
Question: {question}
<think>

Continue the GraphRAG-R1 reasoning trace. If more private-corpus evidence is needed, issue exactly one <|begin_of_query|>query<|end_of_query|> span. If the accumulated evidence is sufficient, close </think> and provide the final <answer>.
```

每个 graph retrieval 调用后，系统追加：

```text
<|begin_of_documents|>
Graph facts:
{graph_facts}

Documents:
{retrieved_context}
<|end_of_documents|>
```

如果达到 `GRAPHRAG_R1_MAX_STEPS` 后仍没有 `<answer>`，强制 final generation prompt 是：

```text
{transcript}
Now close the GraphRAG-R1 reasoning trace and write the final <answer>. Use only accumulated retrieved graph facts and documents. If they are insufficient, write exactly <answer>Insufficient evidence.</answer>.
```

### 11.6.1 MemGraphRAG / Thought-Answer 输出格式 prompt

论文中的 MemGraphRAG baseline 使用：

```text
memgraphrag
```

MemGraphRAG adapter 复用 Signpost 共享公共语料：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

边界必须写清：

```text
公共输入：chunk、entity、type、relation。
MemGraphRAG 自有派生产物：schema memory、fact memory、passage memory、fact-to-passage links、entity-passage PPR retrieval graph。
不使用：Signpost fact/provenance、银证据、target units、Signpost unified graph、Signpost navigation-cue index、Signpost online recommendations。
```

它只在 baseline-owned 目录构建 MemGraphRAG artifacts：

```text
outputs/<dataset>/baselines/memgraphrag/openie_observations.json
outputs/<dataset>/baselines/memgraphrag/filtered_openie.json
outputs/<dataset>/baselines/memgraphrag/memory.json
outputs/<dataset>/baselines/memgraphrag/facts.jsonl
outputs/<dataset>/baselines/memgraphrag/schemas.jsonl
outputs/<dataset>/baselines/memgraphrag/passages.jsonl
outputs/<dataset>/baselines/memgraphrag/index.pkl
outputs/<dataset>/baselines/memgraphrag/graph.json
outputs/<dataset>/baselines/memgraphrag/run_metrics.json
outputs/<dataset>/baselines/memgraphrag/run_status.json
```

MemGraphRAG 检索口径：

- 不重新切 chunk；
- 不重新抽实体、类型或关系；
- 用共享 relation observations 转成 MemGraphRAG OpenIE-like docs；
- 按 schema frequency 过滤 ontology，默认 `MEMGRAPHRAG_SCHEMA_MIN_COUNT=2`；
- 构建 schema / fact / passage 三层 memory；
- 编码 entity、fact、passage stores；
- 在线阶段用 query-to-fact 相似度选 top facts；
- 将 top facts 的 head/tail entity 作为 phrase seed，同时将 dense passage score 作为 passage seed；
- 用 Personalized PageRank 对 entity-passage graph 排序 passages；
- final generation 只读取 PPR 选出的 retrieved passages。

正式 H200 命令：

```bash
export USE_ES=0
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
export MEMGRAPHRAG_EMBED_BATCH_SIZE=32
export MEMGRAPHRAG_SCHEMA_MIN_COUNT=2
export MEMGRAPHRAG_RETRIEVAL_TOP_K=200
export MEMGRAPHRAG_QA_TOP_K=5
export MEMGRAPHRAG_LINKING_TOP_K=5
export MEMGRAPHRAG_PPR_DAMPING=0.5
export MEMGRAPHRAG_PPR_ITERATIONS=20
export MEMGRAPHRAG_PASSAGE_NODE_WEIGHT=0.05
export MEMGRAPHRAG_SYNONYMY_EDGES=1
export MAX_CONTEXT_TOKENS=3500

scripts/baselines/run_baseline_method.sh memgraphrag mix mix
scripts/baselines/run_baseline_method.sh memgraphrag legal legal
scripts/baselines/run_baseline_method.sh memgraphrag agriculture agriculture
scripts/baselines/run_baseline_method.sh memgraphrag graphrag-bench-medical graphrag-bench-medical
scripts/baselines/run_baseline_method.sh memgraphrag graphrag-bench-novel graphrag-bench-novel

# MuSiQue 仅在 datasets/processed/musique 完整汇入后执行。
scripts/baselines/run_baseline_method.sh memgraphrag musique musique
```

非首次执行如果复用 `outputs/<dataset>/baselines/memgraphrag/index.pkl`：

```bash
export REUSE_BASELINE_INDEX=1
scripts/baselines/run_baseline_method.sh memgraphrag agriculture agriculture
```

MemGraphRAG 官方 QA 代码读取 retrieved passages，输出 `Thought:` / `Answer:`。正式 adapter 保留这个输出契约，只迁移 Signpost evidence-grounded 回答约束。论文正式输出：

```text
outputs/<dataset>/predictions/memgraphrag.jsonl
outputs/<dataset>/logs/memgraphrag.query.jsonl
outputs/<dataset>/metrics/memgraphrag.basic_eval.json
outputs/<dataset>/metrics/memgraphrag.query_metrics.json
outputs/<dataset>/baselines/memgraphrag/run_metrics.json
outputs/<dataset>/baselines/memgraphrag/run_status.json
```

完整 system prompt：

```text
As an advanced reading comprehension assistant, answer the question in English strictly based on the retrieved passages selected by MemGraphRAG. Your response starts after "Thought: ", where you briefly analyze the question and identify the relevant evidence. Conclude with "Answer: " to present the final response.

Follow these rules:
- Include all necessary details supported by the retrieved passages.
- Do not use outside knowledge.
- Do not include citations, file names, chunk IDs, or line numbers.
- Do not include conversational filler.
- If the retrieved passages are insufficient, write exactly: "Insufficient evidence." after "Answer: ".
```

完整 user prompt 模板：

```text
Retrieved passages:
{retrieved_context}

Question: {question}
Thought:
```

### 11.7 H200 只重跑 ClueRAG final generation

先确认正式 ClueRAG 已经跑完：

```bash
DATASET=agriculture
test -s outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00/retrieval_results.json
test -s outputs/${DATASET}/predictions/cluerag.jsonl
```

创建 tmux：

```bash
tmux new -s cluerag-prompt-agri
```

tmux 窗口内执行：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re

export REUSE_GRAPH=1
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/agriculture/baselines/cluerag/shared_outputs/COSINE_1.00

scripts/baselines/run_cluerag_method.sh agriculture agriculture
```

输出文件：

```text
outputs/agriculture/predictions/cluerag_prompt_normalized.jsonl
outputs/agriculture/logs/cluerag_prompt_normalized.query.jsonl
outputs/agriculture/metrics/cluerag_prompt_normalized.basic_eval.json
outputs/agriculture/metrics/cluerag_prompt_normalized.query_metrics.json
outputs/agriculture/baselines/cluerag_prompt_normalized/run_metrics.json
outputs/agriculture/baselines/cluerag_prompt_normalized/shared_outputs/COSINE_1.00/generation_results.json
```

其他数据集只替换 `DATASET`：

```bash
DATASET=mix
NAMESPACE=mix
export REUSE_GRAPH=1
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
```

## 12. 最小完整性清单

每个数据集完成后至少应存在：

```text
datasets/processed/<dataset>/questions.jsonl
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/graph.unified.json

outputs/<dataset>/predictions/signpost.full.jsonl
outputs/<dataset>/predictions/signpost.no_offline.jsonl
outputs/<dataset>/predictions/signpost.no_online.jsonl
outputs/<dataset>/predictions/signpost.no_semantic_cues.jsonl
outputs/<dataset>/predictions/signpost.no_provenance_cues.jsonl
outputs/<dataset>/predictions/signpost.no_vertical_cues.jsonl
outputs/<dataset>/predictions/signpost.no_horizontal_cues.jsonl
outputs/<dataset>/predictions/vanilla_llm.jsonl
outputs/<dataset>/predictions/hybrid_rag.jsonl
outputs/<dataset>/predictions/cluerag.jsonl                     # 中间/历史产物，不进论文
outputs/<dataset>/predictions/cluerag_prompt_normalized.jsonl   # 论文正式 ClueRAG baseline
outputs/<dataset>/predictions/agrag.jsonl                       # 论文正式 AGRAG baseline
outputs/<dataset>/predictions/linearrag.jsonl                   # 论文正式 LinearRAG baseline
outputs/<dataset>/predictions/hiprag.jsonl                      # 论文正式 HiPRAG baseline
outputs/<dataset>/predictions/graphrag_r1.jsonl                 # 论文正式 GraphRAG-R1 baseline
outputs/<dataset>/predictions/graphrag_r1_hipporag2.jsonl       # GraphRAG-R1 + official HippoRAG2 v4 baseline，不覆盖 graphrag_r1
outputs/<dataset>/predictions/memgraphrag.jsonl                 # 论文正式 MemGraphRAG baseline

outputs/<dataset>/logs/stage_timing.jsonl
outputs/<dataset>/logs/*.query.jsonl
outputs/<dataset>/metrics/*.basic_eval.json
outputs/<dataset>/metrics/*.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
outputs/<dataset>/baselines/cluerag/run_metrics.json
outputs/<dataset>/baselines/cluerag/shared_graph/manifest.json
outputs/<dataset>/baselines/cluerag/shared_outputs/COSINE_1.00/retrieval_results.json
outputs/<dataset>/baselines/cluerag_prompt_normalized/run_metrics.json
outputs/<dataset>/baselines/cluerag_prompt_normalized/shared_outputs/COSINE_1.00/generation_results.json
outputs/<dataset>/baselines/agrag/graph.json
outputs/<dataset>/baselines/agrag/triples.jsonl
outputs/<dataset>/baselines/agrag/run_metrics.json
outputs/<dataset>/baselines/agrag/run_status.json
outputs/<dataset>/baselines/linearrag/graph.json
outputs/<dataset>/baselines/linearrag/entities.jsonl
outputs/<dataset>/baselines/linearrag/sentences.jsonl
outputs/<dataset>/baselines/linearrag/passage_links.jsonl
outputs/<dataset>/baselines/linearrag/run_metrics.json
outputs/<dataset>/baselines/linearrag/run_status.json
outputs/<dataset>/baselines/hiprag/retrieval_index.json
outputs/<dataset>/baselines/hiprag/run_metrics.json
outputs/<dataset>/baselines/hiprag/run_status.json
outputs/<dataset>/baselines/graphrag_r1/graph.json
outputs/<dataset>/baselines/graphrag_r1/triples.jsonl
outputs/<dataset>/baselines/graphrag_r1/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1/run_status.json
outputs/<dataset>/baselines/memgraphrag/graph.json
outputs/<dataset>/baselines/memgraphrag/openie_observations.json
outputs/<dataset>/baselines/memgraphrag/filtered_openie.json
outputs/<dataset>/baselines/memgraphrag/memory.json
outputs/<dataset>/baselines/memgraphrag/facts.jsonl
outputs/<dataset>/baselines/memgraphrag/schemas.jsonl
outputs/<dataset>/baselines/memgraphrag/passages.jsonl
outputs/<dataset>/baselines/memgraphrag/index.pkl
outputs/<dataset>/baselines/memgraphrag/run_metrics.json
outputs/<dataset>/baselines/memgraphrag/run_status.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/server/openie_results_ner_signpost_f6.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/server/
outputs/<dataset>/baselines/graphrag_r1_hipporag2/graph.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/run_status.json
```

GraphRAG-R1 + official HippoRAG2 v4 的完整 H200 隔离运行流程见：

```text
docs/baselines/graphrag_r1_hipporag2_h200_v4_runbook.zh.md
```
