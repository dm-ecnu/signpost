# Method / Ablation / Metric Audit, 2026-05-27

本文档基于当前代码和本地 H200 备份结果整理：

- 代码根目录：`/home/ruolinsu/signpost/signpost_re`
- 评测数据根目录：`/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525`
- 当前 TargetUnit/Silver 评测输出：`/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/target_unit_silver_eval_v1`

结论先行：

1. `hybrid_rag` 不是多轮检索。它是一次 top-k flat chunk retrieval，然后一次生成。
2. `signpost.full` 和全部 `signpost.no_*` 消融都仍然执行同一套 Supervisor-Researcher 多步在线检索流程。
3. `signpost.no_online` 并没有关闭在线检索/在线 agent。它只清空 `online_signpost` cues，因此命名容易误导。
4. 当前 `SilverHit@5 / SilverRecall@5 / MRR / ClaimCoverage@5` 更像“top-5 evidence access/retrieval reachability”指标，不等价于最终答案质量。它们可能出现 Hybrid RAG 或某些消融高于 `signpost.full`，这不一定说明 Signpost 答案更差。
5. 当前 TargetUnitRecall 是答案文本对 target units 的覆盖率；它和 LLM judge 关注点不同。若 LLM judge 显示 Signpost 最高，而 TargetUnitRecall 或 silver 指标不是最高，优先怀疑：指标定义、prompt 控制、证据序列口径、target/silver 抽取粒度，而不是直接否定 Signpost。

## 1. 各方法离线阶段和在线阶段

### Vanilla LLM

代码：`signpost/baselines/vanilla_llm.py`

离线阶段：

- 无检索索引构建。
- 不使用 chunks、graph、semantic extraction、silver evidence。

在线阶段：

- 只把问题发给 LLM。
- `tool_calls = 0`。
- 没有 `retrieved_chunks` / `citations`。

当前评测：

- 可以计算 TargetUnitRecall，因为有最终答案。
- 不能计算 silver evidence 指标，当前脚本填 NA。

### Hybrid RAG

代码：`signpost/baselines/vanilla_rag.py`

离线阶段：

- 使用共享 chunks：`datasets/processed/<dataset>/chunks.jsonl`。
- 如果 `use_es=True`，使用 ES 里的 chunk index。
- 如果本地检索，按 `mode` 做 bm25/dense/hybrid，hybrid 是 keyword + dense 的 RRF 融合。

在线阶段：

- 对原始 question 做一次 `retrieve(question)`。
- 默认 `top_k=5`。
- 把检索到的 chunks 拼成 context。
- 调一次 LLM 生成答案。

检索轮次：

- 一轮 flat chunk retrieval。
- 不是多轮 agent retrieval。

当前评测：

- TargetUnitRecall 来自最终答案。
- Silver 指标来自 `citations` 或 `retrieved_chunks` 的 top-5。
- 因为它直接返回 top-5 chunks，所以很适合算 `SilverHit@5 / SilverRecall@5 / MRR`。

### ClueRAG / cluerag_prompt_normalized

代码：`signpost/baselines/cluerag.py`

离线阶段：

- 使用共享 chunks 和共享 semantic extraction：
  - `datasets/processed/<dataset>/chunks.jsonl`
  - `datasets/processed/<dataset>/semantic_llm.extractions.jsonl`
- 构建 ClueRAG-style knowledge units、entity links、KU/entity graph。
- 当前 fair-comparison adapter 不重新 chunk，也不重新跑一套独立 NER；它复用 Signpost shared artifacts。

在线阶段：

- `cluerag` 中间过程执行一次 retrieval pipeline：
  - direct chunk retrieval
  - query NER / KU matching
  - graph expansion
  - rerank
  - final generation
- `cluerag_prompt_normalized` 是正式纳入论文比较的版本：
  - 复用 `cluerag` 的 retrieval rows。
  - 只重新跑最终生成 prompt。

检索轮次：

- 不是 LLM 多轮 tool-use agent。
- 是一次 retrieval pipeline，内部包含多组件检索和 graph expansion。

当前评测：

- TargetUnitRecall 来自 normalized 后的最终答案。
- Silver 指标来自复用的 ClueRAG retrieval chunks。

### AGRAG

代码：`signpost/baselines/agrag.py`

离线阶段：

- 使用共享 chunks。
- 使用共享 semantic extraction 构建 entity/relation/passage graph。
- 把 relations 变成 triples。
- 为 triples 建 embedding。
- 缓存到 `outputs/<dataset>/baselines/agrag/index.pkl` 等 artifact。

在线阶段：

- 对 question embedding。
- 选择 anchor triples。
- 在 entity/relation graph 上跑 PPR。
- 用 MCMI-style greedy expansion 选 reasoning subgraph。
- 从 subgraph 取 graph chunks。
- 再做一次 hybrid chunk retrieval。
- 合并 graph chunks + hybrid chunks。
- 调一次 LLM 生成答案。

检索轮次：

- 一轮 graph + hybrid retrieval pipeline。
- 内部有 PPR/MCMI 迭代，但不是 LLM 多轮检索。

当前评测：

- TargetUnitRecall 来自最终答案。
- Silver 指标来自 `citations` 或 `retrieved_chunks` 的 top-5。

### LinearRAG

代码：`signpost/baselines/linearrag.py`

离线阶段：

- 使用共享 chunks。
- 使用共享 semantic extraction 抽到的 entity mentions。
- 构建 relation-free graph：
  - entity nodes
  - passage/chunk nodes
  - sentence bridge nodes
  - adjacent passage links
- 为 passages/entities/sentences 建 embeddings。

在线阶段：

- 对 question embedding。
- 找 seed entities。
- 通过 sentence bridge 激活更多 entities。
- 计算 passage weights。
- 跑 PPR。
- 取 graph chunks。
- 再做一次 hybrid chunk retrieval。
- 合并后调一次 LLM 生成。

检索轮次：

- 一轮 relation-free graph + hybrid retrieval pipeline。
- 内部有 entity activation/PPR 迭代，但不是 LLM 多轮检索。

当前评测：

- TargetUnitRecall 来自最终答案。
- Silver 指标来自 `citations` 或 `retrieved_chunks` 的 top-5。

### HipRAG

代码：`signpost/baselines/hiprag.py`

离线阶段：

- 使用共享 artifacts 构建私有 baseline index。

在线阶段：

- LLM 按 step 工作。
- 如果 LLM 输出 `<search>...</search>`，代码会执行 private chunk search，把结果加入上下文。
- 最多 `max_steps` 轮。
- 如果 LLM 直接输出 answer，不触发 search，则实际没有检索证据序列。

检索轮次：

- 代码支持真正的多轮 LLM/tool retrieval。
- 但实际日志中可能出现没有 search 或没有稳定 `retrieved_chunks` 的情况。

当前评测：

- TargetUnitRecall 来自最终答案。
- Silver 指标当前填 NA，因为没有可靠、可比的 evidence sequence。

### GraphRAG-R1

代码：`signpost/baselines/graphrag_r1.py`

离线阶段：

- 使用共享 chunks 和 semantic extraction 构建 GraphRAG-R1-style graph/index。

在线阶段：

- LLM 可输出 graph query。
- 代码执行 graph search 并把 documents 放回上下文。
- 最多 `max_steps` 轮。
- 如果模型没有按格式发 query，则实际检索会弱化或为空。

检索轮次：

- 代码支持真正的多轮 graph retrieval。

当前评测：

- TargetUnitRecall 来自最终答案。
- Silver 指标当前填 NA，因为没有可靠、可比的 evidence sequence。

### Signpost Full

代码：

- `signpost/agent/supervisor.py`
- `signpost/agent/tools.py`
- `signpost/retrieval/run.py`
- `signpost/retrieval/offline_signpost.py`
- `signpost/retrieval/online_signpost.py`

离线阶段：

- 共享数据准备、parse、chunk。
- 构建 chunk index。
- LLM semantic extraction。
- 构建 semantic graph。
- 构建 structure graph / sequence graph。
- merge 成 unified graph：`graph.unified.json`。
- retrieval 时为结果 attach offline signposts：
  - vertical
  - horizontal
  - provenance
  - semantic

在线阶段：

- Supervisor 把 question 分解成最多 3 个 subquestions。
- Researcher 对每个 subquestion 调一次 `knowledge_search`。
- `knowledge_search` 返回 text_group 和 graph_group：
  - text group: chunk items + summary items
  - graph group: entity/relation items
  - 每个 item 带 offline_signpost
  - 每个 group 带 online_signpost
- Researcher 从 retrieval result 收集 locates。
- 对 locates 调 `read_file`，默认每个 subquestion 读 top-3。
- Supervisor 汇总 evidence，调 LLM synthesis。

检索轮次：

- 多步 / 多 subquestion retrieval。
- 典型上限：最多 3 次 `knowledge_search` + 最多 9 次 `read_file`。

当前评测：

- TargetUnitRecall 来自最终答案 JSON 的 `answer` 字段。
- Silver 指标优先来自 trace 里的 `read_file` sequence。
- 如果没有 read_file，再退到 citations/retrieved_chunks。

## 2. Signpost 每个消融具体消融了什么

消融入口代码：`signpost/retrieval/signpost_variants.py`

注意：所有 `signpost.no_*` 仍然执行同一套 Supervisor-Researcher 流程。消融只发生在 `knowledge_search` 返回的 retrieval result 上，主要是删除或清空某些 cue 字段。

### signpost.full

不做过滤。

保留：

- offline vertical cues
- offline horizontal cues
- offline provenance cues
- offline semantic cues
- group-level online signpost cues
- source locates / source chunk ids

### signpost.no_offline

代码行为：

```python
item["offline_signpost"] = {}
```

具体移除：

- `offline_signpost.vertical`
- `offline_signpost.horizontal`
- `offline_signpost.provenance`
- `offline_signpost.semantic`

实际影响：

- Researcher 仍然会 `knowledge_search`。
- 但是 `collect_locates()` 主要从 `offline_signpost.provenance` 读 locate/source_locates。
- 因此 `no_offline` 会显著削弱 read_file 能力。
- 这解释了为什么 `no_offline` 在表格里大幅下降。

### signpost.no_online

代码行为：

```python
group["online_signpost"] = {
  "scene": previous_scene,
  "seeds": previous_seeds,
  "subgraph": {"nodes": 0, "edges": 0},
  "recommended_entities": []
}
```

具体移除：

- group-level online PPR subgraph size
- group-level recommended entities

保留：

- 原始 text_group / graph_group items
- offline vertical/horizontal/provenance/semantic cues
- source locates
- read_file 流程
- Supervisor/Researcher 多步在线检索流程

实际影响：

- 它不是“no online retrieval”。
- 它只是“no online signpost recommendations”。
- 如果论文中把它描述为去掉在线多智能体检索，需要修改描述或改代码。

### signpost.no_semantic_cues

代码行为：

```python
group["online_signpost"] = empty
signpost.pop("semantic", None)
```

具体移除：

- group-level online signpost recommended entities
- entity/relation item 的 offline semantic cues：
  - neighboring_entities
  - source_entity / target_entity
  - semantic neighbor jump 信息

保留：

- vertical
- horizontal
- provenance
- source locates
- read_file

实际影响：

- 删除语义跳转线索。
- 但保留 provenance，所以仍然能读到证据。
- 因此它的 silver evidence 指标可能不降，甚至可能高于 full。

### signpost.no_provenance_cues

代码行为：

```python
signpost.pop("provenance", None)
item.pop("source_locates", None)
item.pop("source_chunk_ids", None)
```

具体移除：

- offline provenance：
  - file_name
  - start_line
  - end_line
  - locate
  - source_locates
  - source_chunk_ids
  - source_mapping
- item 顶层的 `source_locates`
- item 顶层的 `source_chunk_ids`

保留：

- vertical
- horizontal
- semantic
- online signpost
- 原始检索 item 本身

实际影响：

- 这是对 `read_file` 伤害最大的消融之一。
- 因为 `collect_locates()` 明确依赖 provenance/source_locates。
- 表格里 `TargetUnitRecall` 大幅下降符合预期。
- 但 silver 指标未必同步大幅下降，因为当前 silver metric 对 evidence sequence 的 fallback 和 span-overlap 可能仍能从 citations/其他字段找到部分证据；这需要逐条核对。

### signpost.no_vertical_cues

代码行为：

```python
signpost.pop("vertical", None)
```

具体移除：

- chunk 的 section_path
- chunk 的 parent_summaries / nearest_parent_summary
- summary 的 level / section_path / parent_summary / child_summaries / child_chunks

保留：

- horizontal
- provenance
- semantic
- online signpost
- read_file

实际影响：

- 删除层级/章节结构线索。
- 不直接破坏 evidence locate。
- 因此 evidence reachability 指标可能接近 full，甚至个别超过 full。

### signpost.no_horizontal_cues

代码行为：

```python
signpost.pop("horizontal", None)
```

具体移除：

- chunk 的 previous_chunk
- chunk 的 next_chunk
- 顺序邻接提示

保留：

- vertical
- provenance
- semantic
- online signpost
- read_file

实际影响：

- 删除相邻 chunk 顺序线索。
- 不直接破坏 evidence locate。
- 因此 TargetUnitRecall 和 silver 指标可能接近 full。

## 3. 当前使用什么数据算指标

### 数据集

当前脚本 `scripts/h200_target_unit_silver_eval.py` 使用：

| 评测名 | processed dir | outputs dir |
| --- | --- | --- |
| agriculture | `datasets/processed/agriculture` | `outputs/agriculture` |
| mixv0 | `datasets/processed/mix` | `outputs/mixv0` |

完整路径：

- `agriculture` target units:
  - `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/agriculture/llm_target_units.jsonl`
- `agriculture` silver chunks:
  - `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/agriculture/llm_silver_chunks.jsonl`
- `mixv0` target units:
  - `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/mix/llm_target_units.jsonl`
- `mixv0` silver chunks:
  - `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/mix/llm_silver_chunks.jsonl`

Prediction 输入：

- `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/outputs/<dataset>/predictions/<method>.jsonl`

其中 `mixv0` 的 outputs 目录是：

- `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/outputs/mixv0`

### 当前纳入的方法

- `vanilla_llm`
- `hybrid_rag`
- `cluerag_prompt_normalized`
- `agrag`
- `linearrag`
- `hiprag`
- `graphrag_r1`
- `signpost.full`
- `signpost.no_offline`
- `signpost.no_online`
- `signpost.no_semantic_cues`
- `signpost.no_provenance_cues`
- `signpost.no_vertical_cues`
- `signpost.no_horizontal_cues`

`cluerag` 中间产物不纳入正式表，只纳入 `cluerag_prompt_normalized`。

## 4. 当前指标怎么算

评测脚本：`scripts/h200_target_unit_silver_eval.py`

### TargetUnitRecall

输入：

- method prediction 的最终答案
- `llm_target_units.jsonl` 中对应 question 的 `target_units`

算法：

1. 只取 `required=True` 的 target units。
2. 从 prediction 中抽最终答案：
   - 优先 JSON 的 `answer` 字段。
   - 其次 `<answer>...</answer>`。
   - 否则使用 prediction 文本。
3. 对 target unit 的 `text` 和 `aliases` 做匹配：
   - normalize 后短语包含算命中。
   - 2 token 以内要求 token 全覆盖。
   - 更长 unit 允许 0.75 token overlap。
4. 每题：

```text
TargetUnitRecall = covered_required_units / all_required_units
```

含义：

- 衡量最终答案覆盖了多少 gold target units。
- 它不是 LLM judge。
- 它不衡量表达质量、逻辑完整性、是否有不必要内容。

### SilverHit@5

输入：

- method prediction 的 evidence sequence
- `llm_silver_chunks.jsonl` 中对应 question 的 silver chunks

Evidence sequence 选择顺序：

1. trace 中的 `read_file` tool calls。
2. 如果没有 read_file，用 `citations`。
3. 如果没有 citations，用 `retrieved_chunks`。

对每题取 top-5 evidence items。

命中规则：

- 如果 evidence item 有 `chunk_id`，和 silver chunk_id 相等则命中。
- 否则如果 file_name 相等且 line range overlap，则命中。

每题：

```text
SilverHit@5 = 1 if top-5 命中至少一个 silver chunk else 0
```

含义：

- top-5 evidence 中是否至少拿到一个 silver chunk。
- 不是 precision。
- 一轮 Hybrid RAG 也可以很高，因为 top-5 里只要有一个命中，该题就是 1。

### SilverRecall@5

每题：

```text
SilverRecall@5 = top-5 命中的 silver chunk 数 / 该问题全部 silver chunk 数
```

含义：

- 衡量 top-5 evidence 覆盖了多少 silver chunks。
- 不以 5 为分母。
- 如果某题只有 1 个 silver chunk，命中即为 1.0。

### MRR

当前脚本用完整 evidence sequence，不只 top-5：

```text
MRR = 1 / 第一个 silver chunk 出现的 rank
```

如果没有命中，则为 0。

注意：

- 对 Hybrid RAG，rank 是 retrieved/cited chunks 的排名。
- 对 Signpost，rank 是 read_file 的调用顺序。
- 两者语义并不完全相同。

### ClaimCoverage@5

当前脚本：

1. 找到 top-5 命中的 silver chunks。
2. 读取这些 silver chunks 的 `supports` 字段。
3. 读取 target unit row 的 `facts`。
4. 如果某个 fact 的 `required_units` 全部包含在命中 silver chunks 的 supports 里，则该 fact 被覆盖。

每题：

```text
ClaimCoverage@5 = covered_facts / all_facts
```

注意：

- 它不是直接看答案是否覆盖 facts。
- 它看的是 top-5 命中的 silver chunks 是否支持 facts。
- 如果 silver chunk 的 `supports` 标注不完整，会直接影响该指标。

## 5. 为什么有些方法会超过 signpost.full

当前表格里确实有反直觉现象：

### Agriculture

- `TargetUnitRecall`：
  - `signpost.no_horizontal_cues` = 0.4507，高于 `signpost.full` = 0.4470。
  - 差异很小，可能是模型生成随机性或被删除 cue 减少干扰。
- `SilverHit@5`：
  - `hybrid_rag` = 0.7128，高于 `signpost.full` = 0.6702。
  - `signpost.no_vertical_cues` = 0.7021，高于 full。
  - `signpost.no_semantic_cues` = 0.6915，高于 full。
- `MRR`：
  - `hybrid_rag` = 0.5725，高于 full = 0.5069。
  - 多个消融也高于 full。

### Mixv0

- `SilverHit@5`：
  - `hybrid_rag` = 0.9672，高于 `signpost.full` = 0.9590。
  - `signpost.no_vertical_cues` = 0.9754，高于 full。
- `SilverRecall@5`：
  - `hybrid_rag` = 0.9093，高于 full = 0.8781。
  - 多个消融高于 full。
- `MRR`：
  - `hybrid_rag` = 0.8572，略高于 full = 0.8550。
- `ClaimCoverage@5`：
  - `hybrid_rag` = 0.7612，高于 full = 0.7376。
  - `signpost.no_vertical_cues` = 0.7526，高于 full。

这些现象可能来自以下几类原因。

### 原因 A：Silver 指标不是最终答案质量指标

Hybrid RAG 的 top-5 chunk retrieval 很直接：

- query = 原始问题
- 返回 top-5 chunks
- silver chunks 本身也是 chunk-level 标注

因此它天然容易在 `SilverHit@5 / SilverRecall@5 / MRR` 这种 chunk-level reachability 指标上得高分。

Signpost 的 evidence sequence 是：

- 先分解 subquestions
- 每个 subquestion search
- 再 read_file
- 评测取 read_file 顺序的 top-5

这会导致两个口径差异：

1. Signpost 可能读到了有用证据，但不是 silver chunk 精确 span。
2. Signpost 多 subquestion 的 read_file 顺序不等同于“原问题 top-k ranking”。

因此 silver 指标高不一定等价于答案质量高。

### 原因 B：top-5 截断可能惩罚多步 agent

Signpost 可能最多 3 个 subquestions，每个读 3 条 evidence。

如果评测只看全局前 5 个 read_file：

- 第三个 subquestion 的证据可能被截掉。
- 对多事实问题，后面的 evidence 可能支持关键 fact。
- Hybrid RAG 的 top-5 则全部服务于原始问题，不存在 subquestion 顺序截断问题。

因此 `SilverRecall@5` 可能偏向 one-shot retriever。

### 原因 C：消融删除 cue 可能减少噪声

如果某些 cue 质量不稳定，删除它可能让 retrieval/read_file 顺序更靠近 silver chunks。

例如：

- `no_vertical_cues` 删除 section hierarchy，不影响 provenance。
- `no_horizontal_cues` 删除 previous/next chunk，不影响 provenance。
- `no_semantic_cues` 删除 semantic jump，同时也清空 online recommendations，但保留 provenance。

这些消融仍能 read_file，而且可能少了一些干扰性推荐，所以个别 silver 指标超过 full 并不奇怪。

### 原因 D：prompt 变量没有完全控制

当前代码里各方法 final answer prompt 并不完全一致：

- `vanilla_llm` 使用非常短的 direct-answer prompt。
- `hybrid_rag` 使用 “Answer using only the retrieved context...”。
- `agrag` / `linearrag` 使用 Thought/Answer 格式。
- `cluerag_prompt_normalized` 使用 normalized prompt。
- Signpost 使用 JSON `rationale` / `answer` prompt。

TargetUnitRecall 对最终答案文本很敏感。不同 prompt 可能导致：

- 答案更短或更长。
- 是否列全事实不同。
- 是否输出 “Insufficient evidence” 不同。
- 是否把事实放在 rationale 而不是 answer 不同。

所以如果用 TargetUnitRecall 比最终答案质量，prompt 未完全统一会带来变量不受控风险。

### 原因 E：TargetUnit 抽取/匹配可能过于表层

当前 TargetUnitRecall 匹配规则是 normalize + phrase/token overlap。

风险：

- 同义改写可能没命中。
- 过短 unit 容易被偶然命中。
- 复杂事实拆成多个 unit 时，答案说对了但没有覆盖原文 token，会被低估。
- 答案中提到关键词但事实关系不对，也可能被高估。

这不一定是 target units 抽取错了，也可能是自动匹配函数太粗。

### 原因 F：Silver chunks 可能更贴近 flat chunk retrieval

Silver evidence 是 chunk/span 层级。

如果 silver chunk 抽取时偏向“直接包含答案字符串的 chunk”，那么：

- Hybrid RAG 这种直接 chunk search 更占优。
- Signpost 可能通过 summary/entity/relation 找到相关信息，再 read 原文附近 span，但 span 不一定与 silver chunk 对齐。

因此需要抽样检查：

- Hybrid 命中的 silver chunk 是否真的足以回答问题。
- Signpost 未命中的 read_file 是否其实读到了等价证据。
- Silver chunk 的 `supports` 是否完整覆盖 facts。

## 6. 当前评测是否有问题

我的判断：

1. `TargetUnitRecall` 的方向是对的，但匹配函数还偏粗，不能单独作为最终答案质量结论。
2. `SilverHit@5 / SilverRecall@5 / MRR / ClaimCoverage@5` 的定义本身可以保留，但它们应被解释为 evidence access 指标，不应和 answer quality 混用。
3. 当前 silver 指标对 Hybrid RAG 和 Signpost 的 evidence sequence 口径不同，导致比较不完全公平。
4. `no_online` 的实验命名/论文描述有明显风险；当前代码不是 no online retrieval。
5. prompt 控制确实是一个高风险变量，尤其影响 TargetUnitRecall。

所以，不应直接得出“Hybrid RAG 比 Signpost 好”。更准确的解释是：

- Hybrid RAG 在 chunk-level silver retrieval reachability 上很强。
- Signpost 在 LLM judge 的最终答案质量上更强。
- 当前自动指标需要拆分：retrieval/evidence 指标和 final answer 指标分别报告。

## 7. 建议的下一步核查

### 7.1 对 silver 指标做 per-query 差异抽样

优先抽这些 case：

- Hybrid `SilverRecall@5 = 1`，Signpost full 低。
- Signpost full TargetUnitRecall 高，但 SilverRecall@5 低。
- `no_vertical/no_semantic/no_online` 高于 full 的 query。

人工检查：

- Signpost read_file 是否其实读到等价证据。
- Silver chunk 是否过窄。
- 支持 fact 的 `supports` 是否漏标。

### 7.2 把 Signpost evidence 指标拆成两个版本

建议保留当前版本，同时新增：

1. `ReadFileSilver@5`：只看 read_file。
2. `SearchCandidateSilver@5`：只看 knowledge_search candidate items。
3. `AnyEvidenceSilver@k`：read_file + citations + retrieval candidates 全部归一后再算。

这样可以区分：

- 检索候选是否命中。
- agent 最终读了什么。
- top-5 read order 是否影响分数。

### 7.3 统一 prompt 后重跑最终答案相关指标

尤其是这些方法：

- `vanilla_llm`
- `hybrid_rag`
- `agrag`
- `linearrag`
- `cluerag_prompt_normalized`

目标：

- 让所有方法共享 Signpost 主实验的回答约束。
- 输出格式可保留各自原格式，以免改解析逻辑。
- 再比较 TargetUnitRecall 和 LLM judge。

### 7.4 明确修改论文里的 ablation 命名

如果不改代码，论文中建议把：

- `no_online` 改成 `no_online_signpost_cues`

或在方法部分明确：

- 它不关闭在线检索。
- 它只删除 group-level online PPR recommendations。

如果论文想表达“无在线多智能体检索”，则代码需要新增一个真正关闭 Supervisor/Researcher online loop 的 baseline，而不是复用当前 `no_online`。

