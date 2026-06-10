# Vanilla LLM Baseline Runbook

`vanilla_llm` 是无检索控制组。它只把问题发给 chat model，不读取文档、不检索 chunk、不使用实体、关系或图。

本文档只写 `vanilla_llm` 自己额外需要什么环境、怎么在本地用 ECNU 真实模型验收、以及搬到 H200 后如何切换模型 endpoint。Signpost/H200 已经配置好的总环境不在本文重复。

## 1. 实验目的

它回答的问题是：

```text
模型参数知识和直接生成能力是否已经足够？
```

因此它是主表中的下界控制组，不应和 retrieval 方法比较 evidence recall。

## 2. 涉及代码

```text
signpost/baselines/vanilla_llm.py
signpost/baselines/common.py
scripts/baselines/run_vanilla_llm.py
scripts/baselines/run_baseline_method.sh
tests/test_baselines.py
```

统一方法名：

```text
vanilla_llm
```

## 3. Baseline 专属环境需求

结论：

```text
不需要新建 conda 环境。
不需要新数据库。
不需要 Elasticsearch。
不需要 MinIO。
不需要 Redis/Valkey。
不需要额外官方仓库。
```

使用已有环境：

```text
本地：/home/ruolinsu/signpost/signpost_re + conda env signpost-re
H200：/data/srl/signpost_re + conda env signpost-re
```

唯一要求是 chat model endpoint 可用：

```text
本地调试：ECNU OpenAI-compatible API
H200 正式：localhost Llama-3.3-70B-FP8 OpenAI-compatible API
```

## 4. 输入与输出

输入：

```text
datasets/processed/<dataset>/questions.jsonl
```

输出：

```text
outputs/<dataset>/predictions/vanilla_llm.jsonl
outputs/<dataset>/logs/vanilla_llm.query.jsonl
outputs/<dataset>/metrics/vanilla_llm.basic_eval.json
outputs/<dataset>/metrics/vanilla_llm.query_metrics.json
outputs/<dataset>/metrics/method_summaries.json
outputs/<dataset>/metrics/cost_quality.json
```

`retrieved_chunks` 应为空，`tool_calls` 应为 0，`llm_calls` 应为 1。

## 5. 成本记录口径

离线成本：

```text
无方法专属离线成本。
```

在线成本记录：

```text
latency_seconds
agent_reasoning_latency_seconds
llm_calls
online_llm_calls
input_tokens
output_tokens
total_tokens
```

说明：

```text
vanilla_llm 不使用 F5 chunk index，也不使用 F6 entity/relation extraction。
F6 即使存在，也与该 baseline 无关。
```

## 6. 本地 ECNU 调试流程

本地最终验收必须使用真实 ECNU 模型，不使用 fake/mock 作为最终通过标准。

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
```

`vanilla_llm` 实际只使用 chat 相关变量。embedding/rerank 变量可以和其他 baseline 共用同一个 `.env.local.ecnu`，但不是本 baseline 的必要条件。

进入项目：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda activate signpost-re
set -a
source .env.local.ecnu
set +a
```

先跑单测，不访问真实模型服务，用于检查 schema：

```bash
python -m pytest tests/test_baselines.py tests/test_llm_client.py
```

确认真实 ECNU chat 可用：

```bash
python -m signpost.llm.smoke --chat
```

用真实 ECNU 跑 `legal_test`：

```bash
LIMIT=3 scripts/baselines/run_baseline_method.sh vanilla_llm legal_test legal_test
```

检查产物：

```bash
test -s outputs/legal_test/predictions/vanilla_llm.jsonl
test -s outputs/legal_test/logs/vanilla_llm.query.jsonl
test -s outputs/legal_test/metrics/vanilla_llm.basic_eval.json
test -s outputs/legal_test/metrics/vanilla_llm.query_metrics.json
test -s outputs/legal_test/metrics/method_summaries.json
test -s outputs/legal_test/metrics/cost_quality.json
```

## 7. H200 切换方式

H200 不需要为 `vanilla_llm` 新增任何 conda 环境、数据库或组件。只使用已经部署好的 Signpost 环境和 H200 本地模型服务。

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

确认 chat 服务：

```bash
python -m signpost.llm.smoke --chat
```

H200 smoke：

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

如果服务器正式数据集名称是 `Agriculture-full` 或 `Legal-full`，命令中的 dataset 和 namespace 必须与 `datasets/processed/<dataset>/` 目录一致。

## 8. 结果检查

快速检查行数：

```bash
wc -l outputs/agriculture/predictions/vanilla_llm.jsonl
wc -l outputs/agriculture/logs/vanilla_llm.query.jsonl
```

检查 method summary 是否写入：

```bash
python -m json.tool outputs/agriculture/metrics/method_summaries.json | sed -n '1,120p'
```

预期现象：

```text
metadata.method = vanilla_llm
retrieved_chunks = []
tool_calls = 0
knowledge_search_calls = 0
read_file_calls = 0
llm_calls ≈ 1 per query
```

## 9. 常见问题

如果 `python -m signpost.llm.smoke --chat` 失败：

```text
优先检查 ECNU_API_BASE、ECNU_CHAT_MODEL、ECNU_API_KEY。
H200 上 ECNU_* 只是历史变量名，实际应指向 localhost 服务。
```

如果 `basic_eval` 失败：

```text
检查 prediction JSONL 是否每行都有 question_id、question、answer、prediction、metadata.method。
```

如果 `LIMIT=3` 输出不是 3 行：

```text
检查 questions.jsonl 是否存在且行数足够。
```
