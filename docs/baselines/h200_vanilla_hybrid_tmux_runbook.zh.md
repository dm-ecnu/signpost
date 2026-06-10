# H200 Vanilla LLM / Hybrid RAG Baseline 执行手册

本文档只覆盖两个 in-house baseline：

```text
vanilla_llm
hybrid_rag
```

服务器既有 Signpost conda 环境、Elasticsearch、本地 chat 服务、本地 embedding 服务不在本文重新部署。

## 1. 服务器前提

H200 固定信息：

```text
项目目录：/home/srl/signpost_re
Chat API：http://localhost:8000/v1
Chat model：/data/srl/Llama-3.3-70B-FP8
Embedding API：http://localhost:8001/v1/embeddings
Embedding model：/data/srl/nemotron-8b
Conda env：signpost-re
Elasticsearch：http://127.0.0.1:9200
```

说明：

```text
代码里的 provider 名称仍然叫 ecnu。
H200 上实际通过 .env.h200 指向 localhost 模型服务，不跨公网调用 ECNU。
```

## 2. 本地打包，手动上传到 H200

当前不要使用 `rsync`。采用“本地打 tar 包 -> 手动上传到临港服务器 -> H200 解压覆盖”的方式。

服务器已有项目目录：

```bash
PROJECT_DIR=/home/srl/signpost_re
```

如果你之后确认服务器项目被移动到了别的位置，只改这一行即可。当前不要再使用 `/data/srl/signpost_re` 作为项目目录。

如果 legal 的 Signpost 离线任务正在跑，不要上传或解压整项目代码包；只使用 2.1 的最小 baseline 代码包。最小包不会覆盖 Signpost pipeline 正在使用的主流程代码。

### 2.1 最小 baseline 代码包

在本地福建电脑执行：

```bash
cd /home/ruolinsu/signpost/signpost_re

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/vanilla_hybrid_baseline_patch.tar.gz \
  signpost/baselines \
  scripts/baselines \
  signpost/benchmark/final_metrics.py \
  docs/baselines
```

然后用你当前可用的方式把这个文件上传到 H200：

```text
本地文件：
/home/ruolinsu/signpost/h200/vanilla_hybrid_baseline_patch.tar.gz

上传到 H200：
/home/srl/vanilla_hybrid_baseline_patch.tar.gz
```

如果 `scp` 可用，可以本地执行：

```bash
scp /home/ruolinsu/signpost/h200/vanilla_hybrid_baseline_patch.tar.gz \
  srl@lingang-h200:/home/srl/
```

如果 `scp` 不可用，就用当前能用的 SFTP、VS Code Remote 上传、服务器网页文件管理或其他手动上传方式。只要最终文件在 `/home/srl/vanilla_hybrid_baseline_patch.tar.gz` 即可。

在 H200 上解压覆盖：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "${PROJECT_DIR}"
tar -xzf /home/srl/vanilla_hybrid_baseline_patch.tar.gz
```

这个最小包只覆盖：

```text
signpost/baselines/
scripts/baselines/
signpost/benchmark/final_metrics.py
docs/baselines/
```

不会覆盖：

```text
outputs/
datasets/processed/
```

### 2.2 可选整项目代码包

如果你希望 H200 上代码整体保持一致，可以打整项目包，但仍排除实验产物：

注意：如果 legal 离线正在跑，不要执行本小节。等 legal pipeline 完整结束后再考虑整项目包。

```bash
cd /home/ruolinsu/signpost

tar \
  --exclude='signpost_re/outputs' \
  --exclude='signpost_re/datasets/processed' \
  --exclude='signpost_re/.pytest_cache' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/signpost_re_code_patch.tar.gz \
  signpost_re
```

上传到 H200：

```text
/home/srl/signpost_re_code_patch.tar.gz
```

H200 解压：

```bash
cd /home/srl
tar -xzf /home/srl/signpost_re_code_patch.tar.gz
```

整项目包同样不包含 `outputs/` 和 `datasets/processed/`，不会覆盖已经跑出的正式结果和索引。

## 3. H200 通用环境检查

每个 tmux 窗口开始前都执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a
source .env.h200
set +a
```

检查服务：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
curl http://127.0.0.1:9200
```

检查数据集目录：

```bash
test -s datasets/processed/agriculture/questions.jsonl
test -s datasets/processed/agriculture/chunks.jsonl
test -s datasets/processed/legal/questions.jsonl
test -s datasets/processed/legal/chunks.jsonl
```

如果 `legal` 尚未构建，按第 10 节在 `tmux new -s signpost-legal` 里先跑 legal 的 Signpost pipeline 和 full/ablation。

如果只为了先跑 `vanilla_llm legal`，只需要 `questions.jsonl`。但正式 `hybrid_rag legal` 必须有 `chunks.jsonl` 和 F5 chunk ES index。

## 4. 输出组织

baseline 原始产物仍写入统一实验目录：

```text
outputs/<dataset>/predictions/<baseline>.jsonl
outputs/<dataset>/logs/<baseline>.query.jsonl
outputs/<dataset>/metrics/<baseline>.*.json
```

每个 baseline 额外归档到自己的清晰目录：

```text
outputs/baselines/vanilla_llm/<dataset>/
outputs/baselines/hybrid_rag/<dataset>/
```

归档命令见各 tmux 任务。

## 5. 正式计时原则

tmux 用于防止 SSH 断开，不表示四个实验应该并发运行。

正式计时建议一次只运行一个在线实验：

```text
1. vanilla_llm agriculture
2. vanilla_llm legal
3. hybrid_rag agriculture
4. hybrid_rag legal
```

如果多个 tmux 同时跑，会共享 H200 本地 Llama 服务、embedding 服务和 ES，latency/throughput 计时会互相污染。

## 6. Vanilla LLM Agriculture

创建 tmux：

```bash
tmux new -s vanilla-agri
```

窗口内执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a
source .env.h200
set +a

scripts/baselines/run_baseline_method.sh vanilla_llm agriculture agriculture

mkdir -p outputs/baselines/vanilla_llm/agriculture
cp outputs/agriculture/predictions/vanilla_llm.jsonl outputs/baselines/vanilla_llm/agriculture/
cp outputs/agriculture/logs/vanilla_llm.query.jsonl outputs/baselines/vanilla_llm/agriculture/
cp outputs/agriculture/metrics/vanilla_llm.basic_eval.json outputs/baselines/vanilla_llm/agriculture/
cp outputs/agriculture/metrics/vanilla_llm.query_metrics.json outputs/baselines/vanilla_llm/agriculture/
cp outputs/agriculture/metrics/method_summaries.json outputs/baselines/vanilla_llm/agriculture/method_summaries.after_vanilla_llm.json
cp outputs/agriculture/metrics/cost_quality.json outputs/baselines/vanilla_llm/agriculture/cost_quality.after_vanilla_llm.json
```

检查：

```bash
wc -l outputs/agriculture/predictions/vanilla_llm.jsonl
python -m json.tool outputs/agriculture/metrics/vanilla_llm.query_metrics.json | sed -n '1,80p'
```

## 7. Vanilla LLM Legal

创建 tmux：

```bash
tmux new -s vanilla-legal
```

窗口内执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a
source .env.h200
set +a

scripts/baselines/run_baseline_method.sh vanilla_llm legal legal

mkdir -p outputs/baselines/vanilla_llm/legal
cp outputs/legal/predictions/vanilla_llm.jsonl outputs/baselines/vanilla_llm/legal/
cp outputs/legal/logs/vanilla_llm.query.jsonl outputs/baselines/vanilla_llm/legal/
cp outputs/legal/metrics/vanilla_llm.basic_eval.json outputs/baselines/vanilla_llm/legal/
cp outputs/legal/metrics/vanilla_llm.query_metrics.json outputs/baselines/vanilla_llm/legal/
cp outputs/legal/metrics/method_summaries.json outputs/baselines/vanilla_llm/legal/method_summaries.after_vanilla_llm.json
cp outputs/legal/metrics/cost_quality.json outputs/baselines/vanilla_llm/legal/cost_quality.after_vanilla_llm.json
```

检查：

```bash
wc -l outputs/legal/predictions/vanilla_llm.jsonl
python -m json.tool outputs/legal/metrics/vanilla_llm.query_metrics.json | sed -n '1,80p'
```

## 8. Hybrid RAG Agriculture

前提：

```bash
test -s datasets/processed/agriculture/chunks.jsonl
curl http://127.0.0.1:9200
```

如果 agriculture Signpost pipeline 已经跑过，F5 chunk index 已存在，可以直接跑。

创建 tmux：

```bash
tmux new -s hybrid-agri
```

窗口内执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a
source .env.h200
set +a

USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu TOP_K=5 MAX_CONTEXT_TOKENS=3500 \
  scripts/baselines/run_baseline_method.sh hybrid_rag agriculture agriculture

mkdir -p outputs/baselines/hybrid_rag/agriculture
cp outputs/agriculture/predictions/hybrid_rag.jsonl outputs/baselines/hybrid_rag/agriculture/
cp outputs/agriculture/logs/hybrid_rag.query.jsonl outputs/baselines/hybrid_rag/agriculture/
cp outputs/agriculture/metrics/hybrid_rag.basic_eval.json outputs/baselines/hybrid_rag/agriculture/
cp outputs/agriculture/metrics/hybrid_rag.query_metrics.json outputs/baselines/hybrid_rag/agriculture/
cp outputs/agriculture/metrics/method_summaries.json outputs/baselines/hybrid_rag/agriculture/method_summaries.after_hybrid_rag.json
cp outputs/agriculture/metrics/cost_quality.json outputs/baselines/hybrid_rag/agriculture/cost_quality.after_hybrid_rag.json
```

检查：

```bash
wc -l outputs/agriculture/predictions/hybrid_rag.jsonl
python -m json.tool outputs/agriculture/metrics/hybrid_rag.query_metrics.json | sed -n '1,120p'
```

## 9. Hybrid RAG Legal

前提：

```bash
test -s datasets/processed/legal/chunks.jsonl
curl http://127.0.0.1:9200
```

如果 legal 还没构建，先按第 10 节在 `tmux new -s signpost-legal` 里完整跑：

```text
scripts/run_signpost_dataset_pipeline.sh legal legal
scripts/run_signpost_ablation_suite.sh legal legal
```

这样既能得到 legal 的 Signpost 主实验/消融结果，也能为 `hybrid_rag legal` 准备好 `chunks.jsonl` 和 F5 chunk ES index。

创建 tmux：

```bash
tmux new -s hybrid-legal
```

窗口内执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a
source .env.h200
set +a

USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu TOP_K=5 MAX_CONTEXT_TOKENS=3500 \
  scripts/baselines/run_baseline_method.sh hybrid_rag legal legal

mkdir -p outputs/baselines/hybrid_rag/legal
cp outputs/legal/predictions/hybrid_rag.jsonl outputs/baselines/hybrid_rag/legal/
cp outputs/legal/logs/hybrid_rag.query.jsonl outputs/baselines/hybrid_rag/legal/
cp outputs/legal/metrics/hybrid_rag.basic_eval.json outputs/baselines/hybrid_rag/legal/
cp outputs/legal/metrics/hybrid_rag.query_metrics.json outputs/baselines/hybrid_rag/legal/
cp outputs/legal/metrics/method_summaries.json outputs/baselines/hybrid_rag/legal/method_summaries.after_hybrid_rag.json
cp outputs/legal/metrics/cost_quality.json outputs/baselines/hybrid_rag/legal/cost_quality.after_hybrid_rag.json
```

检查：

```bash
wc -l outputs/legal/predictions/hybrid_rag.jsonl
python -m json.tool outputs/legal/metrics/hybrid_rag.query_metrics.json | sed -n '1,120p'
```

## 10. Legal Signpost 正式实验

如果 legal 的 Signpost 主实验和消融还没跑，先单独开一个 tmux 窗口完成索引与 Signpost full/ablation。这个步骤会构建 `datasets/processed/legal/chunks.jsonl` 和 ES chunk index，后续 `hybrid_rag legal` 依赖它。

```bash
tmux new -s signpost-legal
```

进入 tmux 窗口后执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
conda activate signpost-re
set -a; source .env.h200; set +a

SEMANTIC_EXTRACTOR=llm EMBEDDING_PROVIDER=ecnu \
  scripts/run_signpost_dataset_pipeline.sh legal legal

USE_ES=1 USE_LLM=1 EMBEDDING_PROVIDER=ecnu \
  scripts/run_signpost_ablation_suite.sh legal legal
```

检查 legal 索引和结果：

```bash
test -s datasets/processed/legal/chunks.jsonl
test -s datasets/processed/legal/graph.unified.json
test -s outputs/legal/predictions/signpost.full.jsonl
test -s outputs/legal/metrics/signpost.full.query_metrics.json
wc -l outputs/legal/predictions/signpost.full.jsonl
```

如果 pipeline 已经完成，只重跑在线 full/ablation：

```bash
USE_ES=1 USE_LLM=1 EMBEDDING_PROVIDER=ecnu \
  scripts/run_signpost_ablation_suite.sh legal legal
```

## 11. tmux 操作

detach：

```text
Ctrl-b 然后按 d
```

查看会话：

```bash
tmux ls
```

重新进入：

```bash
tmux attach -t vanilla-agri
tmux attach -t vanilla-legal
tmux attach -t hybrid-agri
tmux attach -t hybrid-legal
tmux attach -t signpost-legal
```

## 12. 跑完后打包下载

在 H200 上执行：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
tar -czf /home/srl/baseline_vanilla_hybrid_outputs.tar.gz \
  outputs/agriculture/predictions/vanilla_llm.jsonl \
  outputs/agriculture/logs/vanilla_llm.query.jsonl \
  outputs/agriculture/metrics/vanilla_llm.basic_eval.json \
  outputs/agriculture/metrics/vanilla_llm.query_metrics.json \
  outputs/agriculture/predictions/hybrid_rag.jsonl \
  outputs/agriculture/logs/hybrid_rag.query.jsonl \
  outputs/agriculture/metrics/hybrid_rag.basic_eval.json \
  outputs/agriculture/metrics/hybrid_rag.query_metrics.json \
  outputs/legal/predictions/vanilla_llm.jsonl \
  outputs/legal/logs/vanilla_llm.query.jsonl \
  outputs/legal/metrics/vanilla_llm.basic_eval.json \
  outputs/legal/metrics/vanilla_llm.query_metrics.json \
  outputs/legal/predictions/hybrid_rag.jsonl \
  outputs/legal/logs/hybrid_rag.query.jsonl \
  outputs/legal/metrics/hybrid_rag.basic_eval.json \
  outputs/legal/metrics/hybrid_rag.query_metrics.json \
  outputs/baselines
```

本地下载：

```bash
scp srl@lingang-h200:/home/srl/baseline_vanilla_hybrid_outputs.tar.gz \
  /home/ruolinsu/signpost/h200/
```

## 13. 常见错误

如果 `source .env.h200` 不存在：

```bash
PROJECT_DIR=/home/srl/signpost_re
cd "$PROJECT_DIR"
cat > .env.h200 <<'EOF'
PYTHONPATH=/home/srl/signpost_re
ECNU_API_BASE=http://localhost:8000/v1
ECNU_API_KEY=EMPTY
ECNU_CHAT_MODEL=/data/srl/Llama-3.3-70B-FP8
ECNU_REASONING_MODEL=/data/srl/Llama-3.3-70B-FP8
ECNU_EMBEDDING_API_BASE=http://localhost:8001/v1/embeddings
ECNU_EMBEDDING_API_KEY=EMPTY
ECNU_EMBEDDING_MODEL=/data/srl/nemotron-8b
ECNU_RERANK_MODEL=unused-local-rerank
LLM_TIMEOUT=600
LLM_RETRIES=6
RETRY_SLEEP=20
GLEANING_ROUNDS=0
SEMANTIC_EXTRACTOR=llm
EMBEDDING_PROVIDER=ecnu
EOF
```

如果 `hybrid_rag` 报 ES index 不存在：

```bash
EMBEDDING_PROVIDER=ecnu scripts/run_signpost_dataset_pipeline.sh <dataset> <dataset>
```

如果只想先 smoke：

```bash
LIMIT=3 scripts/baselines/run_baseline_method.sh vanilla_llm agriculture agriculture
LIMIT=3 USE_ES=1 MODE=hybrid EMBEDDING_PROVIDER=ecnu scripts/baselines/run_baseline_method.sh hybrid_rag agriculture agriculture
```

smoke 会覆盖同名 prediction 文件。正式实验前重新不带 `LIMIT` 跑一遍即可。
