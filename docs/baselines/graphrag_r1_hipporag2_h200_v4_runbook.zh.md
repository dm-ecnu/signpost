# GraphRAG-R1 + Official HippoRAG2 H200 v4 Runbook

更新时间：2026-06-03

本文档只处理新 baseline：

```text
graphrag_r1_hipporag2
```

旧 `graphrag_r1` 不覆盖、不重命名、不删除。新版本所有产物写入独立 method 名：

```text
outputs/<dataset>/predictions/graphrag_r1_hipporag2.jsonl
outputs/<dataset>/logs/graphrag_r1_hipporag2.query.jsonl
outputs/<dataset>/metrics/graphrag_r1_hipporag2.basic_eval.json
outputs/<dataset>/metrics/graphrag_r1_hipporag2.query_metrics.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/
```

论文口径：

```text
GraphRAG-R1 uses the released GraphRAG-R1 policy and the official HippoRAG2 retrieval server.
The server is initialized from the same fixed Signpost F6 OpenIE annotations
(`semantic_llm.extractions.jsonl`) converted to HippoRAG2 OpenIE format. We do not rechunk,
re-extract entities/relations, or read the Signpost graph/navigation index.
```

## 1. 新增/修改文件

本次本地 patch 文件：

```text
scripts/baselines/convert_signpost_f6_to_hipporag_openie.py
scripts/baselines/run_graphrag_r1_hipporag2.py
signpost/baselines/graphrag_r1_hipporag2.py
scripts/baselines/run_baseline_method.sh
baselines/GraphRAG-R1/server/config.py
baselines/GraphRAG-R1/server/server.py
baselines/GraphRAG-R1/server/src/hipporag/embedding_model/__init__.py
baselines/GraphRAG-R1/server/src/hipporag/embedding_model/OpenAI.py
docs/baselines/graphrag_r1_hipporag2_h200_v4_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

## 2. H200 上复制 v4 代码树

不要在正在运行的 v2 代码树上改。先复制 v4：

```bash
cd /home/srl
rsync -a \
  --exclude='outputs' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  /home/srl/signpost_re_v2/ /home/srl/signpost_re_v4/
```

## 3. 本地打包并上传 v4 patch

本地：

```bash
cd /home/ruolinsu/signpost/signpost_re_v2
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/graphrag_r1_hipporag2_v4_patch_${STAMP}.tar.gz \
  scripts/baselines/convert_signpost_f6_to_hipporag_openie.py \
  scripts/baselines/run_graphrag_r1_hipporag2.py \
  signpost/baselines/graphrag_r1_hipporag2.py \
  scripts/baselines/run_baseline_method.sh \
  baselines/GraphRAG-R1/server/config.py \
  baselines/GraphRAG-R1/server/server.py \
  baselines/GraphRAG-R1/server/src/hipporag/embedding_model/__init__.py \
  baselines/GraphRAG-R1/server/src/hipporag/embedding_model/OpenAI.py \
  docs/baselines/graphrag_r1_hipporag2_h200_v4_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md \
  docs/h200_remaining_datasets_tmux_runbook.zh.md
```

上传到 H200：

```bash
scp /home/ruolinsu/signpost/h200/graphrag_r1_hipporag2_v4_patch_${STAMP}.tar.gz \
  srl@lingang-h200:/home/srl/
```

H200 解压到 v4：

```bash
cd /home/srl/signpost_re_v4
tar -xzf /home/srl/graphrag_r1_hipporag2_v4_patch_<STAMP>.tar.gz
```

## 4. H200 静态检查

```bash
cd /home/srl/signpost_re_v4
conda activate signpost-re

export PYTHONPATH=/home/srl/signpost_re_v4
export RAG_PROJECT_BASE=/home/srl/signpost_re_v4

python -m py_compile \
  scripts/baselines/convert_signpost_f6_to_hipporag_openie.py \
  scripts/baselines/run_graphrag_r1_hipporag2.py \
  signpost/baselines/graphrag_r1_hipporag2.py \
  baselines/GraphRAG-R1/server/config.py \
  baselines/GraphRAG-R1/server/server.py \
  baselines/GraphRAG-R1/server/src/hipporag/embedding_model/__init__.py \
  baselines/GraphRAG-R1/server/src/hipporag/embedding_model/OpenAI.py

bash -n scripts/baselines/run_baseline_method.sh
```

## 5. qsample 解压和安装

`prebuilt_qsamples_question_only_20260602_1025.tar.gz` 是 question-only qsample，只含 `questions.jsonl` 和
`question_length_subset_manifest.json`。它可用于在线检索/生成；target/silver 评测需等对应 target/silver 文件补齐后再做。

H200：

```bash
cd /home/srl
tar -xzf /home/srl/prebuilt_qsamples_question_only_20260602_1025.tar.gz

for D in \
  legal_q3 legal_q100 \
  graphrag-bench-medical_q3 graphrag-bench-medical_q100 \
  graphrag-bench-novel_q3 graphrag-bench-novel_q100; do
  wc -l "/home/srl/prebuilt_qsamples/$D/questions.jsonl"
  test -s "/home/srl/prebuilt_qsamples/$D/question_length_subset_manifest.json" && echo "ok $D manifest"
done
```

在 v4 安装 qsample：

```bash
cd /home/srl/signpost_re_v4
conda activate signpost-re

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset legal \
  --prebuilt-dir /home/srl/prebuilt_qsamples/legal_q3 \
  --output-dataset legal_q3 \
  --expected-questions 3 \
  --overwrite

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset legal \
  --prebuilt-dir /home/srl/prebuilt_qsamples/legal_q100 \
  --output-dataset legal_q100 \
  --expected-questions 100 \
  --overwrite

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset graphrag-bench-medical \
  --prebuilt-dir /home/srl/prebuilt_qsamples/graphrag-bench-medical_q3 \
  --output-dataset graphrag-bench-medical_q3 \
  --expected-questions 3 \
  --overwrite

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset graphrag-bench-medical \
  --prebuilt-dir /home/srl/prebuilt_qsamples/graphrag-bench-medical_q100 \
  --output-dataset graphrag-bench-medical_q100 \
  --expected-questions 100 \
  --overwrite

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset graphrag-bench-novel \
  --prebuilt-dir /home/srl/prebuilt_qsamples/graphrag-bench-novel_q3 \
  --output-dataset graphrag-bench-novel_q3 \
  --expected-questions 3 \
  --overwrite

python scripts/h200/install_prebuilt_qsample.py \
  --root /home/srl/signpost_re_v4 \
  --source-dataset graphrag-bench-novel \
  --prebuilt-dir /home/srl/prebuilt_qsamples/graphrag-bench-novel_q100 \
  --output-dataset graphrag-bench-novel_q100 \
  --expected-questions 100 \
  --overwrite
```

MuSiQue：暂时不写正式命令。等 processed dataset、qsample、target/silver 处理完后，按同一流程补 `musique_q3` / `musique_q100`。

## 6. 单数据集运行流程

每个数据集单独跑，避免 8001 embedding 服务被多个 HippoRAG2 server 同时压垮。

### 6.1 转换 F6 OpenIE

设置三个变量，并在 HippoRAG2 server tmux 和 baseline tmux 中都粘贴同一组 `export`：

```bash
export OUT=<output_dataset>
export PROC=<processed_dataset>
export NS=<namespace>
```

例子，选择其中一组后粘贴成 `export` 命令：

```text
agriculture: OUT=agriculture, PROC=agriculture, NS=agriculture
mix:         OUT=mixv0, PROC=mix, NS=mixv0_offline_eff_20260528_2000
legal q3:    OUT=legal_q3, PROC=legal_q3, NS=legal
legal q100:  OUT=legal_q100, PROC=legal_q100, NS=legal
medical q3:  OUT=graphrag-bench-medical_q3, PROC=graphrag-bench-medical_q3, NS=graphrag-bench-medical-fixed
medical q100: OUT=graphrag-bench-medical_q100, PROC=graphrag-bench-medical_q100, NS=graphrag-bench-medical-fixed
novel q3:    OUT=graphrag-bench-novel_q3, PROC=graphrag-bench-novel_q3, NS=graphrag-bench-novel-fixed
novel q100:  OUT=graphrag-bench-novel_q100, PROC=graphrag-bench-novel_q100, NS=graphrag-bench-novel-fixed
```

转换：

```bash
cd /home/srl/signpost_re_v4
conda activate signpost-re
export PYTHONPATH=/home/srl/signpost_re_v4
export RAG_PROJECT_BASE=/home/srl/signpost_re_v4

mkdir -p "outputs/${OUT}/baselines/graphrag_r1_hipporag2/server"

python scripts/baselines/convert_signpost_f6_to_hipporag_openie.py \
  --dataset "${OUT}" \
  --chunks "datasets/processed/${PROC}/chunks.jsonl" \
  --extractions "datasets/processed/${PROC}/semantic_llm.extractions.jsonl" \
  --output "outputs/${OUT}/baselines/graphrag_r1_hipporag2/server/openie_results_ner_signpost_f6.json" \
  --manifest-output "outputs/${OUT}/baselines/graphrag_r1_hipporag2/server/openie_conversion_manifest.json"
```

### 6.2 启动 official HippoRAG2 server

在一个 tmux 窗口启动 server。`SERVER_PORT` 每次只跑一个数据集可固定 8090；若并发，必须换端口并避免 embedding 服务过载。

```bash
tmux new -s gr1-hippo-${OUT}
```

tmux 内：

```bash
cd /home/srl/signpost_re_v4/baselines/GraphRAG-R1/server
conda activate signpost-re

export PYTHONPATH=/home/srl/signpost_re_v4/baselines/GraphRAG-R1/server
export OPENAI_API_KEY=sk-
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""

export DATA_PATH="/home/srl/signpost_re_v4/outputs/${OUT}/baselines/graphrag_r1_hipporag2/server/openie_results_ner_signpost_f6.json"
export SAVE_DIR="/home/srl/signpost_re_v4/outputs/${OUT}/baselines/graphrag_r1_hipporag2/server"

export LLM_MODEL_NAME=/data/srl/Llama-3.3-70B-FP8
export LLM_BASE_URL=http://localhost:8000/v1
export EMBEDDING_MODEL_NAME=/data/srl/nemotron-8b
export EMBEDDING_BASE_URL=http://localhost:8001/v1

export RERANK_BASE_URL=http://localhost:8000/v1
export RERANK_MODEL=/data/srl/Llama-3.3-70B-FP8
export RERANK_API_KEY=sk-

export HIPPORAG_EMBEDDING_BATCH_SIZE=32
export HIPPORAG_RETRIEVAL_TOP_K=5
export HIPPORAG_LINKING_TOP_K=20
export HIPPORAG_MAX_QA_STEPS=3
export HIPPORAG_QA_TOP_K=5
export HIPPORAG_FORCE_INDEX_FROM_SCRATCH=0
export HIPPORAG_FORCE_OPENIE_FROM_SCRATCH=0

export SERVER_HOST=127.0.0.1
export SERVER_PORT=8090
export LOG_LEVEL=info

python server.py
```

server 启动时会读取 `DATA_PATH`，复用转换好的 OpenIE，并在 `SAVE_DIR` 下做 HippoRAG2 passage/entity/fact embedding 和图索引。看到 `/health` 可访问后再跑 baseline：

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1:8090/stats
```

### 6.3 跑新 baseline

另开 tmux 或 shell：

```bash
cd /home/srl/signpost_re_v4
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re_v4
export RAG_PROJECT_BASE=/home/srl/signpost_re_v4
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export BASELINE_QUERY_WORKERS=1

export GRAPHRAG_R1_API_BASE=http://127.0.0.1:8002/v1
export GRAPHRAG_R1_CHAT_MODEL=/data/srl/GraphRAG-R1
export GRAPHRAG_R1_HIPPORAG2_URL=http://127.0.0.1:8090
export GRAPHRAG_R1_HIPPORAG2_OPENIE_PATH="/home/srl/signpost_re_v4/outputs/${OUT}/baselines/graphrag_r1_hipporag2/server/openie_results_ner_signpost_f6.json"
export GRAPHRAG_R1_HIPPORAG2_SAVE_DIR="/home/srl/signpost_re_v4/outputs/${OUT}/baselines/graphrag_r1_hipporag2/server"
export GRAPHRAG_R1_HIPPORAG2_EMBED_BATCH_SIZE=32
export GRAPHRAG_R1_HIPPORAG2_RETRIEVAL_NUM=5
export GRAPHRAG_R1_HIPPORAG2_MAX_STEPS=4
export MAX_CONTEXT_TOKENS=2500

PROCESSED_DATASET="${PROC}" \
scripts/baselines/run_baseline_method.sh graphrag_r1_hipporag2 "${OUT}" "${NS}"
```

检查：

```bash
wc -l "outputs/${OUT}/predictions/graphrag_r1_hipporag2.jsonl"
wc -l "outputs/${OUT}/logs/graphrag_r1_hipporag2.query.jsonl"
ls -lh "outputs/${OUT}/metrics/graphrag_r1_hipporag2.basic_eval.json"
ls -lh "outputs/${OUT}/metrics/graphrag_r1_hipporag2.query_metrics.json"
ls -lh "outputs/${OUT}/baselines/graphrag_r1_hipporag2/graph.json"
ls -lh "outputs/${OUT}/baselines/graphrag_r1_hipporag2/run_metrics.json"
```

确认模型元数据：

```bash
python -m json.tool "outputs/${OUT}/baselines/graphrag_r1_hipporag2/graph.json" | sed -n '1,120p'
```

必须看到：

```text
chat_model_used: /data/srl/GraphRAG-R1
index_type: official_hipporag2_retrieval_server_over_signpost_f6_openie
openie_source: converted_signpost_f6_semantic_llm_extractions
uses_signpost_graph_or_navigation_index: false
```

## 7. q3 smoke

先跑 q3 smoke：

```bash
# legal
OUT=legal_q3
PROC=legal_q3
NS=legal

# medical
OUT=graphrag-bench-medical_q3
PROC=graphrag-bench-medical_q3
NS=graphrag-bench-medical-fixed

# novel
OUT=graphrag-bench-novel_q3
PROC=graphrag-bench-novel_q3
NS=graphrag-bench-novel-fixed
```

每个数据集按第 6 节：转换 OpenIE -> 启动 server -> 跑 baseline。q3 每个 prediction 应为 3 行。

## 8. 正式数据集

当前正式需要重跑 `graphrag_r1_hipporag2` 的数据集：

```text
agriculture
mixv0          # processed source is mix
legal_q100
graphrag-bench-medical_q100
graphrag-bench-novel_q100
```

MuSiQue 暂空，等 processed/qsample/target-silver 准备好后补。

## 9. 从 v4 回拷到 v2

不要拷 `method_summaries.json` 和 `cost_quality.json`，避免覆盖 v2 现有汇总。只拷新 method 文件：

```bash
copy_one_method() {
  local d="$1"
  mkdir -p \
    "/home/srl/signpost_re_v2/outputs/${d}/predictions" \
    "/home/srl/signpost_re_v2/outputs/${d}/logs" \
    "/home/srl/signpost_re_v2/outputs/${d}/metrics" \
    "/home/srl/signpost_re_v2/outputs/${d}/baselines"

  rsync -a "/home/srl/signpost_re_v4/outputs/${d}/predictions/graphrag_r1_hipporag2.jsonl" \
    "/home/srl/signpost_re_v2/outputs/${d}/predictions/"
  rsync -a "/home/srl/signpost_re_v4/outputs/${d}/logs/graphrag_r1_hipporag2.query.jsonl" \
    "/home/srl/signpost_re_v2/outputs/${d}/logs/"
  rsync -a "/home/srl/signpost_re_v4/outputs/${d}/metrics/graphrag_r1_hipporag2.basic_eval.json" \
    "/home/srl/signpost_re_v2/outputs/${d}/metrics/"
  rsync -a "/home/srl/signpost_re_v4/outputs/${d}/metrics/graphrag_r1_hipporag2.query_metrics.json" \
    "/home/srl/signpost_re_v2/outputs/${d}/metrics/"
  rsync -a "/home/srl/signpost_re_v4/outputs/${d}/baselines/graphrag_r1_hipporag2" \
    "/home/srl/signpost_re_v2/outputs/${d}/baselines/"
}

copy_one_method agriculture
copy_one_method mixv0
copy_one_method legal_q100
copy_one_method graphrag-bench-medical_q100
copy_one_method graphrag-bench-novel_q100
```

回拷后在 v2 统一重算总表/最终分析，不用覆盖旧 `graphrag_r1`。

## 10. 完整性清单

每个正式 dataset 至少有：

```text
outputs/<dataset>/predictions/graphrag_r1_hipporag2.jsonl
outputs/<dataset>/logs/graphrag_r1_hipporag2.query.jsonl
outputs/<dataset>/metrics/graphrag_r1_hipporag2.basic_eval.json
outputs/<dataset>/metrics/graphrag_r1_hipporag2.query_metrics.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/server/openie_conversion_manifest.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/server/openie_results_ner_signpost_f6.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/server/
outputs/<dataset>/baselines/graphrag_r1_hipporag2/graph.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1_hipporag2/run_status.json
```

旧结果应仍存在：

```text
outputs/<dataset>/predictions/graphrag_r1.jsonl
outputs/<dataset>/baselines/graphrag_r1/
```
