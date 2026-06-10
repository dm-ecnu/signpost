# Baseline 控制变量要求与新对话交接说明

本文档用于后续继续接入、迁移和评测外部 baseline。核心原则是：

```text
所有 baseline 的检索、图组织、索引构建保留其自身方法特征；
所有 baseline 的最终答案生成阶段统一使用 Signpost 的 evidence-grounded 约束；
如果 baseline 有自己的输出格式，则保留其输出格式，只迁移回答约束，不强制改成 Signpost JSON。
```

## 1. 当前服务器与项目环境

H200 固定信息：

```text
项目目录：/home/srl/signpost_re
工作目录：/home/srl
Conda 环境：signpost-re
Elasticsearch：http://127.0.0.1:9200
Chat API：http://localhost:8000/v1
Chat model：/data/srl/Llama-3.3-70B-FP8
Embedding API：http://localhost:8001/v1/embeddings
Embedding model：/data/srl/nemotron-8b
Rerank API：http://localhost:8033/v1/rerank
Rerank model：/data/srl/llama-nemotron-rerank-1b-v2
```

每个 tmux 窗口的基础环境：

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

## 2. Baseline 处理要求

1. 不改 Signpost 已调通的主流程代码，尤其不改现有 chunk、semantic extraction、graph organization、ES index sync、agent batch 的核心逻辑。

2. Baseline 必须复用 Signpost 共享阶段产物：

```text
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl
```

不能重新切 chunk，不能重新抽实体来混入变量。若 baseline 需要自己的 graph/index，只能基于这些共享产物构建 baseline-owned 图或索引，不能读取、复用或依赖 Signpost 的 graph ES index、Signpost unified graph、Signpost navigation-cue index 或在线 signpost recommendations。

Baseline 图/index 存储按原方法口径处理：

```text
如果原论文/官方实现使用数据库或外部图存储来保存图/index：
  H200 适配时用该 baseline 自己的独立 ES index 替代数据库，index 命名必须带 baseline method 和 dataset，例如 baseline-<method>-<dataset>-graph。
  该 ES index 只能写入该 baseline 自己从共享产物构建出的节点、边、triple、passage-link、retrieval metadata 等对象。

如果原论文/官方实现不使用数据库，而是本地内存、文件、pickle/json/jsonl、networkx/igraph 等方式建图：
  保留其自有建图和存储方式，不强制迁移到 ES。
  产物必须写在 outputs/<dataset>/baselines/<method>/ 或 baseline adapter 自己目录下，不能混入 Signpost 主流程产物。
```

无论使用 ES 还是本地文件/内存图，都必须把建图阶段作为 baseline offline cost 记录下来，包括 wall time、LLM calls/tokens、embedding calls、rerank calls、节点/边/triple/chunk-link/object 数量、baseline-owned disk bytes；若使用 ES，还必须记录 baseline-owned index 名称、写入文档数、bulk 写入耗时和 index bytes。

3. 每个 baseline 的适配代码放在自己的 baseline adapter 范围内，例如：

```text
signpost/baselines/<method>.py
scripts/baselines/run_<method>_method.sh
baselines/<ExternalRepo>/signpost_adapter/
docs/baselines/<method>*.md
```

4. 如果 baseline 原实现需要自己的输入输出格式，写 adapter：

```text
Signpost shared artifacts -> baseline internal format
baseline raw outputs -> unified predictions/*.jsonl
baseline raw query traces -> logs/*.query.jsonl
```

5. 统一输出必须完整：

```text
outputs/<dataset>/predictions/<method>.jsonl
outputs/<dataset>/logs/<method>.query.jsonl
outputs/<dataset>/metrics/<method>.basic_eval.json
outputs/<dataset>/metrics/<method>.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
outputs/<dataset>/baselines/<method>/run_metrics.json
outputs/<dataset>/baselines/<method>/run_status.json
```

6. 指标至少记录：

```text
per-query: llm_calls, input_tokens, output_tokens, total_tokens, tool_calls, retrieved_chunks
per-query: rerank_calls, embedding_calls, knowledge_search_calls if used
baseline-level: offline_wall_time_seconds, online_wall_time_seconds, total_wall_time_seconds
baseline-level: offline/online LLM calls and tokens
baseline-level: graph/index node/edge/object counts if applicable
baseline-level: disk bytes for baseline-owned artifacts
```

7. H200 不依赖公网 API。Chat、embedding、rerank 都走 localhost 服务。不能让 baseline 运行时在线下载复杂依赖；如必须新增依赖，需要写清楚离线 wheelhouse 方案。

8. H200 embedding 服务稳定性是 baseline 控制变量的一部分。所有需要离线或在线 embedding 的 baseline 正式运行必须统一设置：

```bash
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5
```

专属变量只能继承或覆盖为不更激进的值，例如：

```text
AGRAG_EMBED_BATCH_SIZE=32
LINEARRAG_EMBED_BATCH_SIZE=32
HIPRAG_EMBED_BATCH_SIZE=32
CLUERAG_EMBED_BATCH_SIZE=32
```

如果 8001 embedding 服务返回 HTTP 500、`Connection refused` 或退出，不能继续提交新的 embedding-heavy baseline。必须先按 `docs/h200_remaining_datasets_tmux_runbook.zh.md` 的 8001 中断恢复流程盘点 tmux、重启服务、确认 smoke test，再只补跑失败 stage/baseline。失败的 `stage_timing.jsonl` 记录保留作审计。

9. Baseline 可以并发跑，但论文主口径使用 LLM calls/tokens；wall-clock 仍保留。若要比较 wall-clock，避免多个实验共享同一模型服务并发污染。正式 H200 上需要 embedding 的任务不建议并发，避免 8001 稳定性和 wall time 被污染。

10. 不能覆盖其他 baseline 的结果。覆盖该 baseline 自己的结果可以接受，但每次下载归档时要保存 run 时间或压缩包名。

11. 新数据集顺序固定：

```text
Signpost dataset pipeline
Signpost full + all ablations
baseline runs
metrics recomputation
analysis/final metrics
```

该数据集的 Signpost 主实验和消融没跑完前，不开始该数据集的 baseline。

12. 每处理好一个 baseline，必须同步更新：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

`docs/h200_remaining_datasets_tmux_runbook.zh.md` 只记录 H200 tmux 正式运行流程、每个数据集的操作命令、prompt registry 和完整性检查清单；不要把 baseline 的最小迁移包、上传、解压、静态检查和回滚说明写在 runbook 里。

runbook 更新内容至少包括：

```text
每个相关数据集的正式运行命令
该 baseline 哪些产物只是中间/历史产物，哪些 prediction 进入论文
该 baseline final generation 阶段的完整 prompt
该 baseline 是否保留自己的输出格式，以及如何迁移 Signpost evidence-grounded 约束
完整性检查清单中的 predictions/logs/metrics/run_metrics 文件
```

`docs/baselines/baseline_control_requirements_and_handoff.zh.md` 记录 baseline 接入状态、当前本地/H200 状态、最小迁移包、上传、解压、环境配置、静态检查、正式运行命令和回滚说明。新增 baseline 的迁移说明格式参考本文档中“AGRAG 当前本地接入与 H200 最小迁移说明”。

13. 每接入一个新 baseline，必须同时在 `docs/baselines/baseline_control_requirements_and_handoff.zh.md` 给出 H200 最小迁移说明。本地开发完成不等于 H200 已接入；H200 有网络限制，只允许手动下载本地 patch 包、手动上传到服务器，再在服务器解压覆盖。

迁移说明必须遵守：

```text
只打包该 baseline 新增/修改的 adapter、脚本、必要 shared harness 小改和对应文档。
不要打包整个 signpost_re 项目。
不要打包 outputs、datasets、__pycache__、*.pyc、虚拟环境、外部 repo 大目录。
不要覆盖无关 Signpost 主流程文件，避免影响 H200 上正在跑的 tmux 进程。
如果必须修改 shared harness 或 metrics 字段，必须在迁移说明中逐个列出文件和原因。
```

每个 baseline 的 H200 迁移说明至少包括：

```text
1. 本地打包命令：tar 包路径、包含的精确文件列表、排除规则。
2. 手动上传目标：例如 /home/srl/<method>_baseline_patch.tar.gz。
3. H200 解压命令：在 /home/srl/signpost_re 下执行 tar -xzf。
4. H200 环境配置：conda activate signpost-re、source .env.h200、PYTHONPATH/RAG_PROJECT_BASE、本地 chat/embedding/rerank/ES 环境变量。
5. H200 静态检查：python -m py_compile 只检查该 baseline 相关文件；bash -n 只检查改过的 shell 脚本。
6. 正式运行命令：每个数据集的 tmux 命令和环境变量，必须是可直接复制执行的完整命令，显式写出 dataset 和 namespace，不要求运行者再补变量。
8. 回滚方式：列出本次覆盖的文件，必要时从本地重新打旧 patch，不能用 git reset --hard 干扰正在跑的进程。
```

迁移包命名建议：

```text
/home/ruolinsu/signpost/h200/<method>_baseline_patch_YYYYMMDD_HHMM.tar.gz
/home/srl/<method>_baseline_patch_YYYYMMDD_HHMM.tar.gz
```

后续对话继续工作时，必须先确认“本地已改”和“H200 已迁移/已验证”是否是两个不同状态，不要默认服务器已经有本地改动。

## 3. 统一 final generation prompt 口径

论文中的所有 baseline 最终答案生成阶段都要使用 Signpost 的 evidence-grounded 约束。允许保留 baseline 自己的输出格式。

### 3.1 Signpost JSON 输出格式方法

如果 baseline 可以输出 JSON，则使用 Signpost 原始 synthesis prompt：

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

### 3.2 ClueRAG Thought/Answer 输出格式方法

如果 baseline 原本依赖 `Thought:` / `Answer:`，不要强制改成 JSON。使用同等回答约束，但保留其输出契约：

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

原则：统一回答约束，不统一输出格式。

## 4. ClueRAG 当前正式口径

ClueRAG 的论文 baseline 是：

```text
cluerag_prompt_normalized
```

旧的：

```text
cluerag
```

只作为中间产物保留，用来构建 ClueRAG 自己的 graph/index/retrieval，并产出：

```text
outputs/<dataset>/baselines/cluerag/shared_outputs/COSINE_1.00/retrieval_results.json
```

论文中不使用：

```text
outputs/<dataset>/predictions/cluerag.jsonl
```

论文中使用：

```text
outputs/<dataset>/predictions/cluerag_prompt_normalized.jsonl
outputs/<dataset>/logs/cluerag_prompt_normalized.query.jsonl
outputs/<dataset>/metrics/cluerag_prompt_normalized.*.json
outputs/<dataset>/baselines/cluerag_prompt_normalized/run_metrics.json
```

H200 上跑法：

```bash
DATASET=agriculture
NAMESPACE=agriculture

# Step 1: ClueRAG 自身 graph/retrieval 中间步骤，默认 prompt 生成结果弃用。
export CLUERAG_BACKEND=shared_es
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

# Step 2: 论文正式 ClueRAG baseline，复用 retrieval，只重跑 final generation。
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh "$DATASET" "$NAMESPACE"
unset CLUERAG_GENERATION_ONLY
unset CLUERAG_PROMPT_STYLE
unset CLUERAG_METHOD_NAME
unset CLUERAG_SOURCE_OUTPUT_DIR
```

## 5. 已知当前状态

Agriculture：

```text
Signpost full + ablations：已跑
vanilla_llm / hybrid_rag：已跑
ClueRAG shared_es 自有图：已跑
ClueRAG prompt-normalized：已跑并评测
targets：已生成
```

Agriculture prompt-normalized 评测结果：

```text
cluerag_prompt_normalized:
AnsRec 0.5490
F1 0.3585
SilverHit@5 0.6600
SilverRecall@5 0.2392
TER@5 0.4179
EntityRecall@5 0.3231
ClaimCoverage@5 0.5463
Tokens/query 2926.61
```

旧 `cluerag` 默认 prompt 结果已弃用，不写入论文。

## 5.1 AGRAG 当前本地接入与 H200 最小迁移说明

### GraphRAG-R1 HippoRAG2 v4 补充口径

新增 `graphrag_r1_hipporag2`，用于 ICDE 论文中更严格的 GraphRAG-R1 baseline：

```text
released GraphRAG-R1 policy
+ official GraphRAG-R1 server/HippoRAG2 retrieval service
+ fixed Signpost F6 OpenIE annotations converted to HippoRAG2 JSON
```

它不覆盖旧 `graphrag_r1`。所有新产物写入 `outputs/<dataset>/.../graphrag_r1_hipporag2*` 和
`outputs/<dataset>/baselines/graphrag_r1_hipporag2/`。

H200 必须在复制出的 `/home/srl/signpost_re_v4` 上运行，避免影响正在执行的 v2 tmux。完整配置、qsample 安装、OpenIE 转换、HippoRAG2 server 启动、正式运行和回拷命令见：

```text
docs/baselines/graphrag_r1_hipporag2_h200_v4_runbook.zh.md
```

最小迁移文件清单：

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

正式需要重跑该 baseline 的当前数据集：

```text
agriculture
mixv0
legal_q100
graphrag-bench-medical_q100
graphrag-bench-novel_q100
```

q3 smoke 数据集：

```text
legal_q3
graphrag-bench-medical_q3
graphrag-bench-novel_q3
```

MuSiQue 暂时空置，等 processed/qsample/target-silver 完成后再补。

当前 AGRAG adapter 已在本地开发目录接入，H200 不会自动同步。迁移到 H200 时只能搬迁 AGRAG 新增部分和必要 harness 小改，不要打包整个项目。

本次 AGRAG 最小迁移文件清单：

```text
signpost/baselines/agrag.py                         # 新增 AGRAG adapter
signpost/baselines/artifact_summary.py              # 新增 baseline run_metrics/run_status 汇总
scripts/baselines/run_agrag.py                      # 新增 AGRAG module entry
scripts/baselines/run_baseline_method.sh            # 增加 agrag 方法分支
signpost/baselines/common.py                        # 增加 embedding/rerank/PPR cost 字段
signpost/benchmark/query_metrics.py                 # 增加 embedding/rerank cost 汇总字段
docs/h200_remaining_datasets_tmux_runbook.zh.md     # 增加 AGRAG 操作流程和 prompt registry
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

如果 H200 AGRAG 在离线建图阶段出现 embedding `TimeoutError`，原因通常是 triple 数量较多且单次 embedding 请求过大。需要使用分批 embedding 热修 patch，最小迁移文件只有：

```text
signpost/baselines/agrag.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

热修本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/agrag_embedding_batch_hotfix_${STAMP}.tar.gz \
  signpost/baselines/agrag.py \
  scripts/baselines/run_baseline_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/agrag_embedding_batch_hotfix_${STAMP}.tar.gz
```

手动上传到 H200 后解压：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/agrag_embedding_batch_hotfix_<STAMP>.tar.gz
python -m py_compile signpost/baselines/agrag.py scripts/baselines/run_agrag.py
bash -n scripts/baselines/run_baseline_method.sh
```

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/agrag_baseline_patch_${STAMP}.tar.gz \
  signpost/baselines/agrag.py \
  signpost/baselines/artifact_summary.py \
  scripts/baselines/run_agrag.py \
  scripts/baselines/run_baseline_method.sh \
  signpost/baselines/common.py \
  signpost/benchmark/query_metrics.py \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/agrag_baseline_patch_${STAMP}.tar.gz
```

手动下载该 tar 包到本机，再手动上传到 H200：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/agrag_baseline_patch_<STAMP>.tar.gz
H200 目标：/home/srl/agrag_baseline_patch_<STAMP>.tar.gz
```

H200 解压和静态检查：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/agrag_baseline_patch_<STAMP>.tar.gz

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

python -m py_compile \
  signpost/baselines/agrag.py \
  signpost/baselines/artifact_summary.py \
  scripts/baselines/run_agrag.py \
  signpost/baselines/common.py \
  signpost/benchmark/query_metrics.py
bash -n scripts/baselines/run_baseline_method.sh
```

H200 服务环境确认：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

H200 AGRAG 正式运行命令。必须在该数据集已完成 Signpost pipeline/full/ablations 后执行；下面命令可直接复制运行，不需要再手动补 `DATASET` 或 `NAMESPACE`：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
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

scripts/baselines/run_baseline_method.sh agrag agriculture agriculture
```

其他正式数据集命令：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
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

scripts/baselines/run_baseline_method.sh agrag legal legal
scripts/baselines/run_baseline_method.sh agrag mix mix
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-medical graphrag-bench-medical
scripts/baselines/run_baseline_method.sh agrag graphrag-bench-novel graphrag-bench-novel
```

AGRAG 输出检查，按数据集直接复制对应块：

```bash
wc -l outputs/agriculture/predictions/agrag.jsonl
wc -l outputs/agriculture/logs/agrag.query.jsonl
ls -lh outputs/agriculture/metrics/agrag.basic_eval.json
ls -lh outputs/agriculture/metrics/agrag.query_metrics.json
ls -lh outputs/agriculture/baselines/agrag/graph.json
ls -lh outputs/agriculture/baselines/agrag/triples.jsonl
ls -lh outputs/agriculture/baselines/agrag/run_metrics.json
ls -lh outputs/agriculture/baselines/agrag/run_status.json
```

AGRAG 正式运行命令写在：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

注意：AGRAG 当前按其 adapter 自有方式在 `outputs/<dataset>/baselines/agrag/` 构建本地图和 triples，不使用 Signpost graph ES index。若后续 baseline 原论文使用数据库/外部图存储，则在 H200 上用该 baseline 独立 ES index 替代；AGRAG 不属于必须强制迁移到 ES 的情况。

## 5.2 LinearRAG 当前本地接入与 H200 最小迁移说明

当前 LinearRAG adapter 已在本地开发目录接入，H200 不会自动同步。迁移到 H200 时只能搬迁 LinearRAG 新增部分、必要 baseline runner 小改和对应文档，不要打包整个项目。

本地验证状态：

```text
python -m py_compile signpost/baselines/linearrag.py scripts/baselines/run_linearrag.py：通过
bash -n scripts/baselines/run_baseline_method.sh：通过
agriculture LIMIT=1 USE_ES=0 EMBEDDING_PROVIDER=hash smoke：通过

注意：legal_test 缺少 datasets/processed/legal_test/semantic_llm.extractions.jsonl，按控制变量要求不能临时重抽实体，因此没有用 legal_test 做 LinearRAG smoke。
```

H200 已观察到一次 `LINEARRAG_EMBED_BATCH_SIZE=128` 时的 embedding 服务 `HTTP 500`，失败发生在离线 passage embedding 阶段。当前热修策略：

```text
scripts/baselines/run_baseline_method.sh 默认 LINEARRAG_EMBED_BATCH_SIZE=32。
signpost/baselines/linearrag.py 在单个 embedding batch 失败时先重试，再把失败 batch 二分拆小继续执行。
graph.json 记录 offline_embedding_calls、offline_embedding_retries、offline_embedding_failures。
H200 正式运行建议显式设置 LINEARRAG_EMBED_BATCH_SIZE=32、LINEARRAG_EMBED_RETRIES=3、LINEARRAG_EMBED_RETRY_SLEEP=5。
```

LinearRAG adapter 口径：

```text
复用：
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl

不做：
不重新切 chunk
不重新跑 spaCy/NER 或 LLM entity extraction
不读取 Signpost graph/index/navigation-cue index

自有产物：
outputs/<dataset>/baselines/linearrag/graph.json
outputs/<dataset>/baselines/linearrag/entities.jsonl
outputs/<dataset>/baselines/linearrag/sentences.jsonl
outputs/<dataset>/baselines/linearrag/passage_links.jsonl
outputs/<dataset>/baselines/linearrag/run_metrics.json
outputs/<dataset>/baselines/linearrag/run_status.json
```

本次 LinearRAG 最小迁移文件清单：

```text
signpost/baselines/linearrag.py                      # 新增 LinearRAG adapter
scripts/baselines/run_linearrag.py                   # 新增 LinearRAG module entry
scripts/baselines/run_baseline_method.sh             # 增加 linearrag 方法分支
docs/h200_remaining_datasets_tmux_runbook.zh.md      # 增加 LinearRAG 正式运行流程、prompt registry 和完整性清单
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/linearrag_baseline_patch_${STAMP}.tar.gz \
  signpost/baselines/linearrag.py \
  scripts/baselines/run_linearrag.py \
  scripts/baselines/run_baseline_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/linearrag_baseline_patch_${STAMP}.tar.gz
```

手动下载该 tar 包到本机，再手动上传到 H200：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/linearrag_baseline_patch_<STAMP>.tar.gz
H200 目标：/home/srl/linearrag_baseline_patch_<STAMP>.tar.gz
```

H200 解压和静态检查：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/linearrag_baseline_patch_<STAMP>.tar.gz

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

python -m py_compile \
  signpost/baselines/linearrag.py \
  scripts/baselines/run_linearrag.py
bash -n scripts/baselines/run_baseline_method.sh
```

H200 服务环境确认：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

H200 LinearRAG 正式运行命令。必须在该数据集已完成 Signpost pipeline/full/ablations 和前置 baseline 后执行；下面命令可直接复制运行，不需要再手动补 `DATASET` 或 `NAMESPACE`：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
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
```

其他正式数据集命令：

```bash
cd /home/srl/signpost_re
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re
export RAG_PROJECT_BASE=/home/srl/signpost_re
export EMBEDDING_PROVIDER=ecnu
export USE_ES=1
export MODE=hybrid
export MAX_CONTEXT_TOKENS=3500
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
scripts/baselines/run_baseline_method.sh linearrag mix mix
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-medical graphrag-bench-medical
scripts/baselines/run_baseline_method.sh linearrag graphrag-bench-novel graphrag-bench-novel
```

LinearRAG 输出检查，按数据集直接复制对应块：

```bash
wc -l outputs/agriculture/predictions/linearrag.jsonl
wc -l outputs/agriculture/logs/linearrag.query.jsonl
ls -lh outputs/agriculture/metrics/linearrag.basic_eval.json
ls -lh outputs/agriculture/metrics/linearrag.query_metrics.json
ls -lh outputs/agriculture/baselines/linearrag/graph.json
ls -lh outputs/agriculture/baselines/linearrag/entities.jsonl
ls -lh outputs/agriculture/baselines/linearrag/sentences.jsonl
ls -lh outputs/agriculture/baselines/linearrag/passage_links.jsonl
ls -lh outputs/agriculture/baselines/linearrag/run_metrics.json
ls -lh outputs/agriculture/baselines/linearrag/run_status.json
```

LinearRAG 正式运行流程和 final generation prompt registry 写在：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

回滚方式：

```text
本次覆盖文件只有：
signpost/baselines/linearrag.py
scripts/baselines/run_linearrag.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md

如需回滚，不要在 H200 使用 git reset --hard；从本地重新打旧版本最小 patch 并覆盖这些文件。
```

## 5.3 HiPRAG 当前本地接入与 H200 最小迁移说明

当前 HiPRAG adapter 已在本地开发目录接入，H200 不会自动同步。迁移到 H200 时只能搬迁 HiPRAG 新增部分、必要 baseline runner 小改和对应文档，不要打包整个项目。

本地验证状态：

```text
python -m py_compile signpost/baselines/hiprag.py scripts/baselines/run_hiprag.py：通过
bash -n scripts/baselines/run_baseline_method.sh：通过
agriculture LIMIT=0 USE_ES=0 MODE=bm25 EMBEDDING_PROVIDER=hash smoke：通过；该 smoke 只验证管道和 artifact 写入，不是正式 prediction。
```

H200 embedding 热修状态：

```text
2026-05-24 agriculture 正式运行在 HiPRAG offline embedding 阶段遇到 8001 embedding 服务 HTTP 500。
已为 signpost/baselines/hiprag.py 增加 embedding batch retry、失败 batch 二分拆小和 offline_embedding_retries/offline_embedding_failures 记录。
H200 正式运行建议显式设置 HIPRAG_EMBED_BATCH_SIZE=32、HIPRAG_EMBED_RETRIES=3、HIPRAG_EMBED_RETRY_SLEEP=5。
```

HiPRAG adapter 口径：

```text
复用：
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/questions.jsonl

不做：
不重新切 chunk
不重新抽实体或关系
不读取 Signpost graph/index/navigation-cue index

自有产物：
outputs/<dataset>/baselines/hiprag/retrieval_index.json
outputs/<dataset>/baselines/hiprag/run_metrics.json
outputs/<dataset>/baselines/hiprag/run_status.json
```

正式 H200 口径保持 HiPRAG 的 XML agentic search 输出契约：

```text
<think>
<step>
  <reasoning>...</reasoning>
  <search>...</search>
  <context>...</context>
  <conclusion>...</conclusion>
</step>
</think>
<answer>...</answer>
```

只迁移 Signpost evidence-grounded 回答约束，不改成 JSON。HiPRAG 搜索工具基于共享 `chunks.jsonl` 构建 baseline-owned 本地 chunk retrieval index；正式命令默认 `USE_ES=0`，避免读取 Signpost chunk ES index。若后续为了诊断设置 `USE_ES=1`，必须在论文口径外单独标记。

本次 HiPRAG 最小迁移文件清单：

```text
signpost/baselines/hiprag.py                         # 新增 HiPRAG adapter
scripts/baselines/run_hiprag.py                      # 新增 HiPRAG module entry
scripts/baselines/run_baseline_method.sh             # 增加 hiprag 方法分支和 run_metrics 汇总
docs/h200_remaining_datasets_tmux_runbook.zh.md      # 增加 HiPRAG 正式运行流程、prompt registry 和完整性清单
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/hiprag_baseline_patch_${STAMP}.tar.gz \
  signpost/baselines/hiprag.py \
  scripts/baselines/run_hiprag.py \
  scripts/baselines/run_baseline_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/hiprag_baseline_patch_${STAMP}.tar.gz
```

手动下载该 tar 包到本机，再手动上传到 H200：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/hiprag_baseline_patch_<STAMP>.tar.gz
H200 目标：/home/srl/hiprag_baseline_patch_<STAMP>.tar.gz
```

H200 解压和静态检查：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/hiprag_baseline_patch_<STAMP>.tar.gz

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

python -m py_compile \
  signpost/baselines/hiprag.py \
  scripts/baselines/run_hiprag.py
bash -n scripts/baselines/run_baseline_method.sh
```

H200 服务环境确认：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

H200 HiPRAG 正式运行命令。必须在该数据集已完成 Signpost pipeline/full/ablations 和前置 baseline 后执行；下面命令可直接复制运行，不需要再手动补 `DATASET` 或 `NAMESPACE`：

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

scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture
```

其他正式数据集命令：

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

scripts/baselines/run_baseline_method.sh hiprag legal legal
scripts/baselines/run_baseline_method.sh hiprag mix mix
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-medical graphrag-bench-medical
scripts/baselines/run_baseline_method.sh hiprag graphrag-bench-novel graphrag-bench-novel
```

HiPRAG 输出检查，按数据集直接复制对应块：

```bash
wc -l outputs/agriculture/predictions/hiprag.jsonl
wc -l outputs/agriculture/logs/hiprag.query.jsonl
ls -lh outputs/agriculture/metrics/hiprag.basic_eval.json
ls -lh outputs/agriculture/metrics/hiprag.query_metrics.json
ls -lh outputs/agriculture/baselines/hiprag/retrieval_index.json
ls -lh outputs/agriculture/baselines/hiprag/run_metrics.json
ls -lh outputs/agriculture/baselines/hiprag/run_status.json
```

HiPRAG 正式运行流程和 final generation prompt registry 写在：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

回滚方式：

```text
本次覆盖文件只有：
signpost/baselines/hiprag.py
scripts/baselines/run_hiprag.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md

如需回滚，不要在 H200 使用 git reset --hard；从本地重新打旧版本最小 patch 并覆盖这些文件。
```

## 5.4 GraphRAG-R1 当前本地接入与 H200 最小迁移说明

当前 GraphRAG-R1 adapter 已在本地开发目录接入，H200 不会自动同步。迁移到 H200 时只能搬迁 GraphRAG-R1 新增部分、必要 baseline runner 小改和对应文档，不要打包整个项目。

本地验证状态：

```text
python -m py_compile signpost/baselines/graphrag_r1.py scripts/baselines/run_graphrag_r1.py：通过
bash -n scripts/baselines/run_baseline_method.sh：通过
agriculture LIMIT=0 USE_ES=0 MODE=bm25 EMBEDDING_PROVIDER=hash smoke：通过；该 smoke 只验证 baseline-owned graph build、artifact 写入和 runner wiring，不是正式 prediction。本地 smoke 产生的空 graphrag_r1 输出已清理。
```

GraphRAG-R1 adapter 口径：

```text
复用：
datasets/processed/<dataset>/chunks.jsonl
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
datasets/processed/<dataset>/questions.jsonl

不做：
不重新切 chunk
不重新抽实体或关系
不读取 Signpost graph/index/navigation-cue index

自有产物：
outputs/<dataset>/baselines/graphrag_r1/graph.json
outputs/<dataset>/baselines/graphrag_r1/triples.jsonl
outputs/<dataset>/baselines/graphrag_r1/run_metrics.json
outputs/<dataset>/baselines/graphrag_r1/run_status.json
```

正式 H200 口径保持 GraphRAG-R1 的 agentic graph retrieval 输出契约：

```text
<think>...</think>
<answer>...</answer>

检索标签：
<|begin_of_query|>...<|end_of_query|>
<|begin_of_documents|>...<|end_of_documents|>
```

只迁移 Signpost evidence-grounded 回答约束，不改成 JSON。GraphRAG-R1 图检索基于共享 `semantic_llm.extractions.jsonl` 构建 baseline-owned entity/relation/passage graph；正式命令默认 `USE_ES=0`，避免读取 Signpost chunk ES index。若后续为了诊断设置 `USE_ES=1`，必须在论文口径外单独标记。

本次 GraphRAG-R1 最小迁移文件清单：

```text
signpost/baselines/graphrag_r1.py                   # 新增 GraphRAG-R1 adapter
scripts/baselines/run_graphrag_r1.py                # 新增 GraphRAG-R1 module entry
scripts/baselines/run_baseline_method.sh            # 增加 graphrag_r1 方法分支和 run_metrics 汇总
docs/h200_remaining_datasets_tmux_runbook.zh.md     # 增加 GraphRAG-R1 正式运行流程、prompt registry 和完整性清单
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/graphrag_r1_baseline_patch_${STAMP}.tar.gz \
  signpost/baselines/graphrag_r1.py \
  scripts/baselines/run_graphrag_r1.py \
  scripts/baselines/run_baseline_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/graphrag_r1_baseline_patch_${STAMP}.tar.gz
```

手动下载该 tar 包到本机，再手动上传到 H200：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/graphrag_r1_baseline_patch_<STAMP>.tar.gz
H200 目标：/home/srl/graphrag_r1_baseline_patch_<STAMP>.tar.gz
```

H200 解压和静态检查：

```bash
cd /home/srl/signpost_re
tar -xzf /home/srl/graphrag_r1_baseline_patch_<STAMP>.tar.gz

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

python -m py_compile \
  signpost/baselines/graphrag_r1.py \
  scripts/baselines/run_graphrag_r1.py
bash -n scripts/baselines/run_baseline_method.sh
```

H200 服务环境确认：

```bash
curl -s http://127.0.0.1:9200 >/tmp/es.ok && echo "es ok"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' | head
```

H200 GraphRAG-R1 正式运行命令。必须在该数据集已完成 Signpost pipeline/full/ablations 和前置 baseline 后执行；下面命令可直接复制运行，不需要再手动补 `DATASET` 或 `NAMESPACE`：

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

其他正式数据集命令：

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
scripts/baselines/run_baseline_method.sh graphrag_r1 mix mix
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-medical graphrag-bench-medical
scripts/baselines/run_baseline_method.sh graphrag_r1 graphrag-bench-novel graphrag-bench-novel
```

GraphRAG-R1 输出检查，按数据集直接复制对应块：

```bash
wc -l outputs/agriculture/predictions/graphrag_r1.jsonl
wc -l outputs/agriculture/logs/graphrag_r1.query.jsonl
ls -lh outputs/agriculture/metrics/graphrag_r1.basic_eval.json
ls -lh outputs/agriculture/metrics/graphrag_r1.query_metrics.json
ls -lh outputs/agriculture/baselines/graphrag_r1/graph.json
ls -lh outputs/agriculture/baselines/graphrag_r1/triples.jsonl
ls -lh outputs/agriculture/baselines/graphrag_r1/run_metrics.json
ls -lh outputs/agriculture/baselines/graphrag_r1/run_status.json
```

GraphRAG-R1 正式运行流程和 final generation prompt registry 写在：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

回滚方式：

```text
本次覆盖文件只有：
signpost/baselines/graphrag_r1.py
scripts/baselines/run_graphrag_r1.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md

如需回滚，不要在 H200 使用 git reset --hard；从本地重新打旧版本最小 patch 并覆盖这些文件。
```

## 5.5 Baseline index 复用 + ClueRAG conversion 热修与 H200 最小迁移说明

问题背景：

```text
旧版本中 AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 每次执行 scripts/baselines/run_baseline_method.sh 都会重新构建自己的 baseline-owned graph/index。
ClueRAG 的 REUSE_GRAPH 旧逻辑只跳过 ES 重建，仍会重新组织 shared_graph artifacts。
这会导致同一个 dataset/method 重跑时重复执行离线阶段；正确口径应是首次构建一次，后续只重跑 online query 和 final generation。
```

本次热修行为：

```text
首次运行 REUSE_BASELINE_INDEX=0：
  AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 正常构建 baseline-owned graph/index，并写入 outputs/<dataset>/baselines/<method>/index.pkl。

后续运行 REUSE_BASELINE_INDEX=1：
  AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 从 index.pkl 加载 index，只执行 online query 和 final generation。
  如果 index.pkl 不存在或 embedding_provider/mode/use_es 不匹配，直接报错，不静默重建。
  method_summary 不再把 baseline_<method> stage 计入 offline stage。
  graph.json / retrieval_index.json 标记 offline_reused=true，offline_wall_time_seconds=0。

ClueRAG：
  REUSE_GRAPH=1 时跳过 prepare 和 shared_graph 重建，直接从 outputs/<dataset>/baselines/cluerag/shared_graph/ 加载 graph artifacts；
  若 USE_ES=1，会要求 baseline-owned ClueRAG ES index 已存在。
  修复 convert_cluerag_outputs 中临时 Result 缺少 embedding_calls/rerank_calls 字段导致的 AttributeError。
```

本次最小迁移文件清单：

```text
signpost/baselines/agrag.py
signpost/baselines/linearrag.py
signpost/baselines/hiprag.py
signpost/baselines/graphrag_r1.py
signpost/baselines/cluerag.py
scripts/baselines/run_baseline_method.sh
scripts/baselines/run_cluerag_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md
```

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
mkdir -p /home/ruolinsu/signpost/h200
STAMP=$(date +%Y%m%d_%H%M)
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf /home/ruolinsu/signpost/h200/baseline_index_reuse_and_cluerag_cost_hotfix_${STAMP}.tar.gz \
  signpost/baselines/agrag.py \
  signpost/baselines/linearrag.py \
  signpost/baselines/hiprag.py \
  signpost/baselines/graphrag_r1.py \
  signpost/baselines/cluerag.py \
  scripts/baselines/run_baseline_method.sh \
  scripts/baselines/run_cluerag_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/baseline_index_reuse_and_cluerag_cost_hotfix_${STAMP}.tar.gz
```

手动上传：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/baseline_index_reuse_and_cluerag_cost_hotfix_<STAMP>.tar.gz
H200 目标：/home/srl/baseline_index_reuse_and_cluerag_cost_hotfix_<STAMP>.tar.gz
```

H200 解压和静态检查：

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
  signpost/baselines/agrag.py \
  signpost/baselines/linearrag.py \
  signpost/baselines/hiprag.py \
  signpost/baselines/graphrag_r1.py \
  signpost/baselines/cluerag.py \
  scripts/baselines/run_agrag.py \
  scripts/baselines/run_linearrag.py \
  scripts/baselines/run_hiprag.py \
  scripts/baselines/run_graphrag_r1.py \
  scripts/baselines/run_cluerag.py
bash -n scripts/baselines/run_baseline_method.sh scripts/baselines/run_cluerag_method.sh
```

H200 使用方式：

```bash
# 首次构建：不要设置 REUSE_BASELINE_INDEX，或显式设为 0。
scripts/baselines/run_baseline_method.sh agrag agriculture agriculture

# 后续只重跑 online query + final generation：
export REUSE_BASELINE_INDEX=1
scripts/baselines/run_baseline_method.sh agrag agriculture agriculture
scripts/baselines/run_baseline_method.sh linearrag agriculture agriculture
scripts/baselines/run_baseline_method.sh hiprag agriculture agriculture
scripts/baselines/run_baseline_method.sh graphrag_r1 agriculture agriculture

# ClueRAG 复用 graph/index，只重跑 retrieval + generation：
export REUSE_GRAPH=1
scripts/baselines/run_cluerag_method.sh agriculture agriculture

# ClueRAG prompt-normalized 仍然只重跑 final generation：
export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR=outputs/agriculture/baselines/cluerag/shared_outputs/COSINE_1.00
scripts/baselines/run_cluerag_method.sh agriculture agriculture
unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR
```

无人值守脚本使用方式写在：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md 的 “2.0 无人值守顺序执行所有实验”
```

注意：

```text
H200 上已经在旧代码下跑完的 AGRAG / LinearRAG / HiPRAG / GraphRAG-R1 artifact 目录通常没有 index.pkl。
这些旧结果无法无损恢复完整向量 index；应用热修后需要对每个 dataset/method 再跑一次 REUSE_BASELINE_INDEX=0 来生成 index.pkl。
从下一次开始才能设置 REUSE_BASELINE_INDEX=1 只重跑 online query 和 final generation。
```

剩余数据集：

```text
mix
graphrag-bench-medical
graphrag-bench-novel
```

运行手册：

```text
docs/h200_remaining_datasets_tmux_runbook.zh.md
```

## 5.7 MemGraphRAG 当前本地接入与 H200 最小迁移说明

当前本地开发目录已新增 `memgraphrag` baseline adapter。H200 不会自动同步；迁移时只搬本节列出的新增/修改文件，不要打包整个项目。

方法边界：

```text
公共输入：chunk、entity、type、relation。
baseline 自有派生产物：schema memory、fact memory、passage memory、fact-to-passage links、entity-passage PPR retrieval graph。
不使用：Signpost fact/provenance、银证据、target units、Signpost unified graph、Signpost navigation-cue index、Signpost online recommendations。
```

与官方 MemGraphRAG 对齐的保留部分：

```text
1. 将公共 relation observations 转成 MemGraphRAG OpenIE-like docs。
2. 按官方 ontology/schema frequency filtering 过滤 schema，默认 min_count=2。
3. 构建 schema / fact / passage 三层 memory，并维护 schema->fact、fact->passage、passage->fact 索引。
4. 编码 entity、fact、passage memory stores。
5. 在线阶段先做 query-to-fact 相似度，取 linking_top_k facts。
6. 将 selected fact 的 head/tail entity 作为 phrase seed，dense passage score 作为 passage seed。
7. 用 PPR 对 entity-passage graph 排序 passage，再把 top qa_top_k passages 送入 final generation。
8. final generation 保留 `Thought:` / `Answer:` 输出格式，只迁移 Signpost evidence-grounded 回答约束。
```

本地已验证：

```text
python -m py_compile signpost/baselines/memgraphrag.py scripts/baselines/run_memgraphrag.py：通过
bash -n scripts/baselines/run_baseline_method.sh：通过
LIMIT=0 EMBEDDING_PROVIDER=hash MEMGRAPHRAG_SCHEMA_MIN_COUNT=1 MEMGRAPHRAG_SYNONYMY_EDGES=0 BASELINE_QUERY_WORKERS=1 bash scripts/baselines/run_baseline_method.sh memgraphrag agriculture agriculture：通过
```

smoke 只验证管线，不作为论文结果；正式 H200 运行必须使用 `EMBEDDING_PROVIDER=ecnu` 和 H200 本地 embedding/chat 服务。

本地打包命令：

```bash
cd /home/ruolinsu/signpost/signpost_re_v2
mkdir -p /home/ruolinsu/signpost/h200
STAMP="$(date +%Y%m%d_%H%M)"
tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='outputs' \
  --exclude='datasets' \
  -czf /home/ruolinsu/signpost/h200/memgraphrag_baseline_patch_${STAMP}.tar.gz \
  signpost/baselines/memgraphrag.py \
  scripts/baselines/run_memgraphrag.py \
  scripts/baselines/run_baseline_method.sh \
  docs/h200_remaining_datasets_tmux_runbook.zh.md \
  docs/baselines/baseline_control_requirements_and_handoff.zh.md
ls -lh /home/ruolinsu/signpost/h200/memgraphrag_baseline_patch_${STAMP}.tar.gz
```

手动上传：

```text
本地 tar 包：/home/ruolinsu/signpost/h200/memgraphrag_baseline_patch_<STAMP>.tar.gz
H200 目标：/home/srl/memgraphrag_baseline_patch_<STAMP>.tar.gz
```

H200 解压、环境和静态检查：

```bash
cd /home/srl/signpost_re_v2
tar -xzf /home/srl/memgraphrag_baseline_patch_<STAMP>.tar.gz

conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re_v2
export RAG_PROJECT_BASE=/home/srl/signpost_re_v2
export EMBEDDING_PROVIDER=ecnu
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5

python -m py_compile signpost/baselines/memgraphrag.py scripts/baselines/run_memgraphrag.py
bash -n scripts/baselines/run_baseline_method.sh
```

正式运行命令。必须等对应 dataset 的 Signpost pipeline、full 和 ablations 完成后再执行：

```bash
cd /home/srl/signpost_re_v2
conda activate signpost-re
set -a
source .env.h200
set +a

export PYTHONPATH=/home/srl/signpost_re_v2
export RAG_PROJECT_BASE=/home/srl/signpost_re_v2
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

# MuSiQue 仅在 datasets/processed/musique/{chunks,semantic_llm.extractions,questions}.jsonl
# 已按 MuSiQue offline runbook 汇入 signpost_re_v2 后执行。
scripts/baselines/run_baseline_method.sh memgraphrag musique musique
```

非首次执行如果只重跑 online query + final generation：

```bash
export REUSE_BASELINE_INDEX=1
scripts/baselines/run_baseline_method.sh memgraphrag agriculture agriculture
```

完整性检查：

```bash
DATASET=agriculture
wc -l outputs/${DATASET}/predictions/memgraphrag.jsonl
wc -l outputs/${DATASET}/logs/memgraphrag.query.jsonl
ls -lh outputs/${DATASET}/metrics/memgraphrag.basic_eval.json
ls -lh outputs/${DATASET}/metrics/memgraphrag.query_metrics.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/graph.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/openie_observations.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/filtered_openie.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/memory.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/facts.jsonl
ls -lh outputs/${DATASET}/baselines/memgraphrag/schemas.jsonl
ls -lh outputs/${DATASET}/baselines/memgraphrag/passages.jsonl
ls -lh outputs/${DATASET}/baselines/memgraphrag/index.pkl
ls -lh outputs/${DATASET}/baselines/memgraphrag/run_metrics.json
ls -lh outputs/${DATASET}/baselines/memgraphrag/run_status.json
```

回滚方式：

```text
本次覆盖文件：
signpost/baselines/memgraphrag.py
scripts/baselines/run_memgraphrag.py
scripts/baselines/run_baseline_method.sh
docs/h200_remaining_datasets_tmux_runbook.zh.md
docs/baselines/baseline_control_requirements_and_handoff.zh.md

如需回滚，只重新打旧版本 patch 覆盖这些文件。不要在 H200 使用 git reset --hard，以免影响正在跑的 tmux 任务。
```

## 6. 新对话交接提示词

如果开启一个新对话继续处理 baseline，可以直接把下面内容发给新对话：

```text
我们在 /home/ruolinsu/signpost/signpost_re 开发，H200 服务器项目在 /home/srl/signpost_re。H200 本地服务是 chat http://localhost:8000/v1 model /data/srl/Llama-3.3-70B-FP8，embedding http://localhost:8001/v1/embeddings model /data/srl/nemotron-8b，rerank http://localhost:8033/v1/rerank model /data/srl/llama-nemotron-rerank-1b-v2，ES 在 http://127.0.0.1:9200。

Baseline 控制变量要求：
1. 不改 Signpost 主流程，只改 baseline adapter。
2. Baseline 复用 datasets/processed/<dataset>/chunks.jsonl、semantic_llm.extractions.jsonl、questions.jsonl。
3. 不重新切 chunk，不重新抽实体；baseline 自己的图/index 可以基于共享产物构建。
4. 所有 baseline final generation 必须使用 Signpost 的 evidence-grounded 回答约束；如果 baseline 有自己的输出格式，保留其输出格式，只迁移回答约束。
5. 输出统一写 predictions/*.jsonl、logs/*.query.jsonl、metrics/*.json，并记录 per-query LLM calls/tokens/tool/retrieved chunks/rerank/embedding，以及 baseline-level offline/online/total cost。
6. ClueRAG 的论文 baseline 是 cluerag_prompt_normalized，不使用旧 cluerag 默认 prompt 的结果。旧 cluerag 只作为 ClueRAG graph/retrieval 中间步骤。
7. 每个数据集必须先跑 Signpost pipeline、Signpost full 和 ablations，再跑 baseline。
8. 每处理好一个 baseline，必须更新 docs/h200_remaining_datasets_tmux_runbook.zh.md：只加入每个数据集的正式运行流程、该 baseline final generation 的完整 prompt、进入论文的 prediction 文件和完整性检查清单；不要把最小迁移包/上传/解压/回滚说明写进 runbook。
9. 本地接入不等于 H200 已接入。每个新 baseline 必须在 docs/baselines/baseline_control_requirements_and_handoff.zh.md 提供最小 H200 迁移包和手动上传/解压/环境配置/py_compile/正式运行命令；正式运行命令必须可直接复制执行，显式写出 dataset 和 namespace；只打包该 baseline 新增或必要修改文件，不能打包整个项目，避免影响 H200 正在跑的 tmux 进程。
10. baseline 自己的图/index 不能读 Signpost 图/index。若原方法使用数据库或外部图存储，H200 适配用该 baseline 独立 ES index；若原方法不用数据库，则保持其本地文件/内存图方式，但必须记录建图时间和图/index 指标。

请先阅读 docs/baselines/baseline_control_requirements_and_handoff.zh.md 和 docs/h200_remaining_datasets_tmux_runbook.zh.md，然后继续接入/运行下一个 baseline。
```
