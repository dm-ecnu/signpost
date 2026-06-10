# Signpost full v1 独立工作区完整实验 Runbook

当前方案固定为：**v0 和 v1 是两个独立项目目录**。

- v0 项目：`/home/srl/signpost_re`
- v1 项目：`/home/srl/signpost_re_v1`
- v0 继续跑已有 tmux，不覆盖代码，不覆盖 outputs。
- v1 在独立副本里解压热修包并运行。
- v1 只共享 H200 服务：LLM `8000`、embedding `8001`、rerank `8033`、ES `9200`。

因此 H200 上没有“复原 v0 代码”的问题。v1 效果不好时，不采用 `signpost.full_rerank_v1` 结果即可；需要清理时只处理 `/home/srl/signpost_re_v1`。

## 当前 v1 改动

v1 只改在线回答质量相关逻辑，不重切 chunk、不重抽实体、不重建 Signpost 离线图/index。

热修包文件：

```text
scripts/run_signpost_method_variant.sh
scripts/run_signpost_ablation_suite_variant.sh
scripts/build_all_and_score.py
scripts/run_v1_full_dataset_suite.sh
signpost/agent/supervisor.py
signpost/agent/batch.py
docs/signpost_full_v1_safe_experiment_runbook.zh.md
```

新增/修改功能：

- `scripts/run_signpost_method_variant.sh` 支持独立 method id。
- `scripts/run_signpost_ablation_suite_variant.sh` 跑 v1 full 和 v1 ablations，并写独立 method id，避免覆盖复制过来的 v0 输出。
- v1 full method id：`signpost.full_rerank_v1`。
- v1 ablation method ids：`signpost.full_rerank_v1.no_offline`、`signpost.full_rerank_v1.no_online` 等。
- `SIGNPOST_EVIDENCE_RERANK=1` 时启用 evidence rerank、证据去重、证据 token 上限和 answer-slot 检查。
- `scripts/build_all_and_score.py` 是通用 LLM 对比评价脚本，支持 `--dataset agriculture|mix|legal`。

## 本地打包与上传

本地打包：

```bash
cd /home/ruolinsu/signpost/signpost_re

tar -czf /home/ruolinsu/signpost/signpost_full_rerank_v1_patch.tar.gz \
  scripts/run_signpost_method_variant.sh \
  scripts/run_signpost_ablation_suite_variant.sh \
  scripts/build_all_and_score.py \
  scripts/build_suffer_samples.py \
  scripts/run_v1_full_dataset_suite.sh \
  signpost/agent/supervisor.py \
  signpost/agent/batch.py \
  docs/signpost_full_v1_safe_experiment_runbook.zh.md

tar -tzf /home/ruolinsu/signpost/signpost_full_rerank_v1_patch.tar.gz
```

上传到 H200：

```bash
scp /home/ruolinsu/signpost/signpost_full_rerank_v1_patch.tar.gz \
  srl@lingang-h200:/home/srl/signpost_full_rerank_v1_patch.tar.gz
```

## H200 创建/更新 v1 工作区

不要在 `/home/srl/signpost_re` 解压 v1 热修包。

首次创建 v1 工作区：

```bash
test -s /home/srl/signpost_full_rerank_v1_patch.tar.gz || {
  echo "missing patch: upload /home/srl/signpost_full_rerank_v1_patch.tar.gz first"
  exit 1
}

V0_DIR=/home/srl/signpost_re
V1_DIR=/home/srl/signpost_re_v1

test ! -e "$V1_DIR" || {
  echo "v1 workspace already exists: $V1_DIR; use update command instead"
  exit 1
}

mkdir -p "$V1_DIR"
rsync -a --exclude datasets --exclude outputs "$V0_DIR"/ "$V1_DIR"/
ln -s "$V0_DIR/datasets" "$V1_DIR/datasets"
mkdir -p "$V1_DIR/outputs"
cp -a "$V0_DIR/outputs/agriculture" "$V1_DIR/outputs/"
cp -a "$V0_DIR/outputs/mix" "$V1_DIR/outputs/"

cd "$V1_DIR"
tar -xzf /home/srl/signpost_full_rerank_v1_patch.tar.gz
chmod +x scripts/run_signpost_method_variant.sh scripts/run_signpost_ablation_suite_variant.sh scripts/build_all_and_score.py scripts/run_v1_full_dataset_suite.sh
python -m py_compile scripts/build_all_and_score.py scripts/build_suffer_samples.py signpost/agent/supervisor.py signpost/agent/batch.py
```

如果 v1 工作区已存在，只更新 v1 热修包：

```bash
cd /home/srl/signpost_re_v1
tar -xzf /home/srl/signpost_full_rerank_v1_patch.tar.gz
chmod +x scripts/run_signpost_method_variant.sh scripts/run_signpost_ablation_suite_variant.sh scripts/build_all_and_score.py scripts/run_v1_full_dataset_suite.sh
python -m py_compile scripts/build_all_and_score.py scripts/build_suffer_samples.py signpost/agent/supervisor.py signpost/agent/batch.py
```

如果 H200 没有 `rsync`，用完整复制兜底：

```bash
cp -a /home/srl/signpost_re /home/srl/signpost_re_v1
cd /home/srl/signpost_re_v1
tar -xzf /home/srl/signpost_full_rerank_v1_patch.tar.gz
chmod +x scripts/run_signpost_method_variant.sh scripts/run_signpost_ablation_suite_variant.sh scripts/build_all_and_score.py scripts/run_v1_full_dataset_suite.sh
python -m py_compile scripts/build_all_and_score.py scripts/build_suffer_samples.py signpost/agent/supervisor.py signpost/agent/batch.py
```

## 服务检查

在开 v1 tmux 前检查：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

## 完整任务顺序

每个数据集的完整 v1 流程是：

1. 跑 v1 Signpost full 和 v1 ablations：`scripts/run_signpost_ablation_suite_variant.sh <dataset> <namespace> signpost.full_rerank_v1`。
2. 跑 baseline：`vanilla_llm`、`vanilla_rag`、`hybrid_rag`、`agrag`、`linearrag`、`hiprag`、`graphrag_r1`。
3. 跑 ClueRAG shared graph/retrieval 中间步骤：`cluerag`。
4. 跑论文正式 ClueRAG prompt：`cluerag_prompt_normalized`。
5. 跑 LLM 对比评价：`scripts/build_all_and_score.py --dataset <dataset>`。

命令中 baseline 默认如果 prediction 已存在就跳过；如果缺失则运行。这样 v1 工作区可以复用从 v0 复制来的已完成 baseline，也能补齐缺失 baseline。

## agriculture 完整并行命令

这条命令会按顺序完成 v1 full、v1 ablations、baselines、ClueRAG normalized 和 LLM 评价。

```bash
tmux new -d -s v1-agriculture "bash -lc '
cd /home/srl/signpost_re_v1
conda activate signpost-re
set -a
source .env.h200
set +a
export PYTHONPATH=/home/srl/signpost_re_v1
export RAG_PROJECT_BASE=/home/srl/signpost_re_v1
export SIGNPOST_EVIDENCE_RERANK=1
export SIGNPOST_RERANK_URL=http://127.0.0.1:8033/v1/rerank
export SIGNPOST_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export SIGNPOST_CANDIDATE_LOCATE_TOP_K=30
export SIGNPOST_RERANK_TOP_K=8
export SIGNPOST_READ_TOP_K=5
export SIGNPOST_EVIDENCE_MAX_TOKENS=5000
unset LIMIT
{
  scripts/run_v1_full_dataset_suite.sh agriculture agriculture
} 2>&1 | tee /home/srl/signpost_full_rerank_v1_agriculture_\$(date +%Y%m%d_%H%M%S).log
'"
```

## mix 完整并行命令

这条命令会按顺序完成 v1 full、v1 ablations、baselines、ClueRAG normalized 和 LLM 评价。

```bash
tmux new -d -s v1-mix "bash -lc '
cd /home/srl/signpost_re_v1
conda activate signpost-re
set -a
source .env.h200
set +a
export PYTHONPATH=/home/srl/signpost_re_v1
export RAG_PROJECT_BASE=/home/srl/signpost_re_v1
export SIGNPOST_EVIDENCE_RERANK=1
export SIGNPOST_RERANK_URL=http://127.0.0.1:8033/v1/rerank
export SIGNPOST_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export SIGNPOST_CANDIDATE_LOCATE_TOP_K=30
export SIGNPOST_RERANK_TOP_K=8
export SIGNPOST_READ_TOP_K=5
export SIGNPOST_EVIDENCE_MAX_TOKENS=5000
unset LIMIT
{
  scripts/run_v1_full_dataset_suite.sh mix mix
} 2>&1 | tee /home/srl/signpost_full_rerank_v1_mix_\$(date +%Y%m%d_%H%M%S).log
'"
```

## 查看与排错

查看 tmux：

```bash
tmux ls
tmux attach -t v1-agriculture
tmux attach -t v1-mix
```

如果 session 秒退，看最新日志：

```bash
tail -100 $(ls -t /home/srl/signpost_full_rerank_v1_agriculture_*.log | head -1)
tail -100 $(ls -t /home/srl/signpost_full_rerank_v1_mix_*.log | head -1)
```

检查输出：

```bash
cd /home/srl/signpost_re_v1
wc -l outputs/agriculture/predictions/signpost.full_rerank_v1.jsonl
wc -l outputs/mix/predictions/signpost.full_rerank_v1.jsonl
find ans/agriculture -maxdepth 1 -name "*.txt" | wc -l
find ans/mix -maxdepth 1 -name "*.txt" | wc -l
```

## LLM 评价中断恢复

如果日志最后出现：

```text
ModuleNotFoundError: No module named 'build_suffer_samples'
```

说明旧热修包漏带了 LLM 评价依赖 `scripts/build_suffer_samples.py`。这只影响最后的 LLM 打分阶段；已经生成的 v1 full、v1 ablations、baseline、ClueRAG prediction 不需要重跑。

先重新上传新版热修包，然后在 H200 的 v1 工作区更新：

```bash
cd /home/srl/signpost_re_v1
tar -xzf /home/srl/signpost_full_rerank_v1_patch.tar.gz
chmod +x scripts/run_signpost_method_variant.sh scripts/run_signpost_ablation_suite_variant.sh scripts/build_all_and_score.py scripts/run_v1_full_dataset_suite.sh
python -m py_compile scripts/build_all_and_score.py scripts/build_suffer_samples.py signpost/agent/supervisor.py signpost/agent/batch.py
```

只恢复 mix 的 LLM 评价：

```bash
tmux new -d -s v1-mix-score "bash -lc '
cd /home/srl/signpost_re_v1
conda activate signpost-re
set -a
source .env.h200
set +a
export PYTHONPATH=/home/srl/signpost_re_v1
export RAG_PROJECT_BASE=/home/srl/signpost_re_v1
{
  python scripts/build_all_and_score.py \
    --root /home/srl/signpost_re_v1 \
    --project-dir /home/srl/signpost_re_v1 \
    --dataset mix \
    --clean-all \
    --clean-ans
  find ans/mix -maxdepth 1 -name \"*.txt\" | wc -l
} 2>&1 | tee /home/srl/signpost_full_rerank_v1_mix_score_\$(date +%Y%m%d_%H%M%S).log
'"
```

只恢复 agriculture 的 LLM 评价：

```bash
tmux new -d -s v1-agriculture-score "bash -lc '
cd /home/srl/signpost_re_v1
conda activate signpost-re
set -a
source .env.h200
set +a
export PYTHONPATH=/home/srl/signpost_re_v1
export RAG_PROJECT_BASE=/home/srl/signpost_re_v1
{
  python scripts/build_all_and_score.py \
    --root /home/srl/signpost_re_v1 \
    --project-dir /home/srl/signpost_re_v1 \
    --dataset agriculture \
    --clean-all \
    --clean-ans
  find ans/agriculture -maxdepth 1 -name \"*.txt\" | wc -l
} 2>&1 | tee /home/srl/signpost_full_rerank_v1_agriculture_score_\$(date +%Y%m%d_%H%M%S).log
'"
```

## 清理 v1

v1 不覆盖 v0，所以没有 v0 代码复原步骤。

如果 v1 不采用：

```bash
tmux kill-session -t v1-agriculture
tmux kill-session -t v1-mix
```

确认不需要 v1 结果后，再手动处理 `/home/srl/signpost_re_v1`。不要在未确认前删除该目录。

## 结果判定

判定顺序：

1. 先看 `ans/agriculture`、`ans/mix` 的 LLM 对比评价均分，比较 `signpost.full_rerank_v1` 与 `signpost.full`。
2. 再看 `metrics/signpost.full_rerank_v1.query_metrics.json` 和 `metrics/signpost.full.query_metrics.json`。
3. 如果 v1 整体优于 v0，论文只写 v1。
4. 如果 v1 不稳定或低于 v0，论文继续写 v0，v1 结果保留为失败实验记录，不进入论文主表。
