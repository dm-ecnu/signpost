# Prompt 控制变量修改方案与 H200 重跑交接

本文档只整理方案，不修改运行代码。待确认后再按本文档修改 adapter / runner。

## 1. 目标

这次要把进入论文比较的最终回答生成约束统一到 Signpost 主实验的回答约束，减少 prompt 差异带来的控制变量问题。

统一原则：

```text
1. 所有方法最终回答都必须用英语回答。
2. 最终答案必须严格基于该方法可见的 evidence / retrieved context。
3. 答案要完整、独立成句，覆盖问题所需上下文和细节。
4. 不在最终答案里写 citation、文件名、行号。
5. 不写 conversational filler，例如 “Based on the provided text...”。
6. 如果可见 evidence 不足以回答，输出 exactly: Insufficient evidence.
7. 输出格式不强制统一；各方法可以保留自己的 JSON、plain text、Thought/Answer、XML/tag 格式。
```

注意：Vanilla LLM 没有 retrieval evidence。若严格套用 “strictly based on evidence”，它会系统性输出 `Insufficient evidence`。如果论文仍需要 Vanilla LLM 作为“无检索但允许参数知识”的 baseline，就不能说它与 RAG 方法有完全相同的 evidence-grounded 约束；只能统一语言、完整性、无 citation、无 filler 等 answer style 约束。下面给出两个可选口径，需人工确认。

## 2. 当前 prompt 与拟修改 prompt

### 2.1 Vanilla LLM

代码位置：

```text
signpost/baselines/vanilla_llm.py
```

当前 system prompt：

```text
Answer the question directly. Do not cite documents because no retrieval context is provided.
```

当前 user prompt：

```text
{question}
```

拟修改方案 A：严格 evidence-grounded，对 Vanilla LLM 最严格但会导致大量 Insufficient evidence。

System prompt：

```text
Answer the question in English strictly based on the evidence available to you.

Follow these rules:
1. Provide a complete, well-formed final response that directly answers the question.
2. Include all necessary context and details supported by the available evidence so that the answer is comprehensive and stands alone clearly.
3. DO NOT include citations, file names, or line numbers.
4. DO NOT include conversational filler such as "Based on the provided text..." or "According to the evidence...".
5. If the available evidence is insufficient to answer the question, output exactly: "Insufficient evidence."

No retrieval context is provided for this baseline.
```

User prompt：

```text
Question:
{question}
```

拟修改方案 B：保留 Vanilla LLM 的无检索参数知识 baseline 含义，只统一回答风格，不声称 evidence-grounded。

System prompt：

```text
Answer the question in English directly.

Follow these rules:
1. Provide a complete, well-formed final response that directly answers the question.
2. Include necessary context and details so that the answer is comprehensive and stands alone clearly.
3. DO NOT include citations, file names, or line numbers because no retrieval context is provided.
4. DO NOT include conversational filler such as "Based on the provided text..." or "According to the evidence...".
5. If you cannot answer the question, output exactly: "Insufficient evidence."
```

User prompt：

```text
Question:
{question}
```

建议：论文主表若强调 controlled evidence-grounded generation，应使用方案 A；若保留经典 Vanilla LLM 对照，应使用方案 B 并在论文中说明它不具备 retrieval evidence。

### 2.2 Hybrid RAG

代码位置：

```text
signpost/baselines/vanilla_rag.py
signpost/baselines/hybrid_rag.py
```

`hybrid_rag` 调用 `vanilla_rag` 的回答逻辑，只是检索模式使用 hybrid。

当前 system prompt：

```text
You are a retrieval-augmented QA baseline. Ground the answer in the provided chunks.
```

当前 user prompt：

```text
Question:
{question}

Retrieved context:
{context}

Answer using only the retrieved context. If the context is insufficient, say so briefly.
```

拟修改 system prompt：

```text
Answer the question in English strictly based on the provided evidence.

Follow these rules:
1. Provide a complete, well-formed final response that directly answers the question.
2. Include all necessary context and details supported by the evidence so that the answer is comprehensive and stands alone clearly.
3. DO NOT include citations, file names, or line numbers. Source tracking is handled externally.
4. DO NOT include conversational filler such as "Based on the provided text..." or "According to the evidence...".
5. If the evidence is insufficient to answer the question, output exactly: "Insufficient evidence."
```

拟修改 user prompt：

```text
Question:
{question}

Evidence:
{context}
```

输出格式：仍为 plain text，不强制 JSON。

### 2.3 Signpost full 与消融

代码位置：

```text
signpost/agent/supervisor.py
scripts/run_signpost_method.sh
scripts/run_signpost_ablation_suite.sh
signpost/retrieval/signpost_variants.py
```

Signpost full / no_offline / no_online / no_semantic_cues / no_provenance_cues / no_vertical_cues / no_horizontal_cues 目前共用同一个 synthesis prompt。消融只通过 `--signpost-variant` 屏蔽 retrieval result 中的 cue，不改最终回答 prompt。

当前 synthesis system prompt：

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

当前 synthesis user prompt：

```text
Question:
{question}

Evidence:
{evidence_text}
```

其中每个 evidence block：

```text
子问题：{subquestion}
{snippet.file_content_view}
```

拟修改：

```text
不改。
```

原因：Signpost 主实验和所有 Signpost 消融已经使用同一套最终回答约束。

### 2.4 需要一并检查但本轮暂不改的 baseline

如果后续要求所有论文 baseline 都统一 prompt，还需要逐个处理：

```text
cluerag_prompt_normalized
agrag
linearrag
hiprag
graphrag_r1
```

这些方法目前多数已经显式带有 evidence-grounded 约束，但保留各自输出格式。若确认“所有 baseline 都统一”，需要另开一节逐个列原 prompt 和拟改 prompt。

## 3. H200 两数据集一条命令重跑方案

目标：只重跑受 prompt 修改影响的 online final generation / prediction 阶段，不重新跑离线 shared pipeline 和 baseline graph/index。

数据集：

```text
agriculture
mixv0
```

注意：H200 processed dataset 名称可能是 `mix`，但当前本地 outputs 目录是 `mixv0`。重跑前需在 H200 上确认实际 dataset/output 命名。如果 H200 当前正式输出目录是 `outputs/mixv0`，命令就用 `mixv0`；如果是 `outputs/mix`，下载后本地再映射。

### 3.1 H200 一条命令自动跑完所有受影响实验

在 H200 执行：

```bash
cat > /home/srl/run_prompt_control_rerun_agri_mix.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/srl/signpost_re
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="/home/srl/prompt_control_rerun_agri_mix_${STAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

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
export REUSE_BASELINE_INDEX=1
export REUSE_GRAPH=1

curl -fsS http://127.0.0.1:9200 >/tmp/es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/chat_models.ok
curl -fsS http://localhost:8001/v1/models >/tmp/embed_models.ok
curl -fsS http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' >/tmp/rerank_smoke.ok

run_dataset() {
  local DATASET="$1"
  local NAMESPACE="$2"
  echo "[rerun] dataset=${DATASET} namespace=${NAMESPACE}"

  scripts/baselines/run_baseline_method.sh vanilla_llm "${DATASET}" "${NAMESPACE}"
  scripts/baselines/run_baseline_method.sh hybrid_rag "${DATASET}" "${NAMESPACE}"

  # Signpost full + all ablations use the same Signpost synthesis prompt.
  scripts/run_signpost_ablation_suite.sh "${DATASET}" "${NAMESPACE}"
}

run_dataset agriculture agriculture
run_dataset mixv0 mixv0

echo "[rerun] done log=${LOG_FILE}"
RUN

bash /home/srl/run_prompt_control_rerun_agri_mix.sh
```

如果 H200 上没有 `mixv0` processed dataset，而是 `mix`，把最后一行改成：

```bash
run_dataset mix mix
```

### 3.2 tmux 版本

```bash
tmux new -s prompt_control_agri_mix
bash /home/srl/run_prompt_control_rerun_agri_mix.sh
```

## 4. 下载到本地新目录

本地新目录命名建议：

```text
/home/ruolinsu/signpost/local_backup_prompt_control_YYYYMMDD_HHMM
```

下载内容应与 `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525` 类似，至少包含：

```text
datasets/processed/agriculture/
datasets/processed/mix/
outputs/agriculture/
outputs/mixv0/ 或 outputs/mix/
```

示例 rsync 命令（把 `H200_HOST` 替换成实际 ssh alias）：

```bash
STAMP="$(date +%Y%m%d_%H%M)"
DEST="/home/ruolinsu/signpost/local_backup_prompt_control_${STAMP}"
mkdir -p "${DEST}"

rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/datasets/processed/agriculture \
  "${DEST}/datasets/processed/"

rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/datasets/processed/mix \
  "${DEST}/datasets/processed/"

rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/outputs/agriculture \
  "${DEST}/outputs/"

rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/outputs/mixv0 \
  "${DEST}/outputs/" || \
rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/outputs/mix \
  "${DEST}/outputs/"
```

如果 LLM 评分文件也在 H200 上生成，需要同步：

```bash
rsync -av --info=progress2 \
  H200_HOST:/home/srl/signpost_re/ans \
  "${DEST}/"
```

## 5. 下载后的处理和评测步骤

### 5.1 完整性检查

```bash
ROOT="/home/ruolinsu/signpost/local_backup_prompt_control_YYYYMMDD_HHMM"

find "${ROOT}/outputs/agriculture/predictions" -maxdepth 1 -type f | sort
find "${ROOT}/outputs/mixv0/predictions" -maxdepth 1 -type f | sort

wc -l "${ROOT}/datasets/processed/agriculture/questions.jsonl"
wc -l "${ROOT}/datasets/processed/agriculture/llm_target_units.jsonl"
wc -l "${ROOT}/datasets/processed/agriculture/llm_silver_chunks.jsonl"

wc -l "${ROOT}/datasets/processed/mix/questions.jsonl"
wc -l "${ROOT}/datasets/processed/mix/llm_target_units.jsonl"
wc -l "${ROOT}/datasets/processed/mix/llm_silver_chunks.jsonl"
```

期望：

```text
agriculture: 100 questions / 100 target units rows / 100 silver rows
mix: 130 questions / 130 target units rows / 130 silver rows
```

### 5.2 不覆盖旧结果的评测输出目录

新评测结果不要写入旧目录：

```text
${ROOT}/final_eval_prompt_control/
```

### 5.3 指标口径待确认

当前已生成的 `final_eval_v2` 里：

```text
Answer Recall / legacy_recall = gold answer token overlap recall
legacy_precision = prediction token overlap precision
legacy_f1 = token overlap F1
```

这些是 lexical 指标，不使用 target units。

如果要按新版 target units 计算 answer recall，需要改成：

```text
TargetUnitRecall = 被预测答案覆盖的 required target_units / required target_units 总数
```

但 target-unit precision / F1 需要先定义 denominator：

```text
方案 1：只算 TargetUnitRecall，不算 precision/F1。
原因：当前只有 gold target_units，没有 predicted target_units；precision 没有自然分母。

方案 2：先对每个 prediction 抽取 predicted_units，再和 gold target_units 匹配。
TargetUnitPrecision = matched predicted_units / predicted_units
TargetUnitRecall = matched gold target_units / gold target_units
TargetUnitF1 = 2PR/(P+R)
这个方案最合理，但需要新增一次 LLM predicted-unit extraction 或规则抽取。

方案 3：用 lexical token precision/recall/F1 保留为 legacy 指标。
这个方案可复现，但不符合“recall 使用 targetunit”的要求。
```

建议：论文主指标改用方案 2；临时快速检查可以先输出方案 1 的 TargetUnitRecall 和 legacy precision/F1，明确标注 legacy。

