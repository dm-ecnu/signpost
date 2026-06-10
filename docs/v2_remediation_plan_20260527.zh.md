# V2 整改方案草案：方法、检索口径、Prompt、指标与 H200 并行运行

本文档只整理整改方案，不修改代码。确认后再创建独立 v2 项目并改代码。

相关目录：

- 本地原项目：`/home/ruolinsu/signpost/signpost_re`
- H200 原项目：`/home/srl/signpost_re`
- 本地 H200 备份：`/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525`
- 当前评测脚本：`/home/ruolinsu/signpost/signpost_re/scripts/h200_target_unit_silver_eval.py`
- Prompt 控制方案参考：`/home/ruolinsu/signpost/signpost_re/docs/prompt_control_and_rerun_plan_20260526.zh.md`

## 1. 当前质疑与整改目标

当前主要问题：

1. Hybrid RAG 是否用了 Signpost 的 ES index，导致看到 Signpost 离线线索。
2. Silver 指标到底是不是用抽好的 silver evidence 计算，`citations` / `retrieved_chunks` / `read_file` 的角色混乱。
3. ClueRAG pipeline 到底是一次检索还是多次检索，需要讲清楚。
4. HipRAG 和 GraphRAG-R1 虽然代码支持多轮，但实际可能没有稳定触发检索，需要整改为真正可执行的多轮检索 baseline。
5. Signpost 的 silver 指标当前用 `read_file` sequence，和其他方法用 `retrieved_chunks` 不同，比较口径不公平。
6. Signpost 消融解释需要更精确，尤其是 `no_offline`、`no_online`、`no_semantic_cues`、`no_provenance_cues`。
7. Prompt 变量没有完全控制，需要在 v2 统一最终回答约束。
8. 自动 TargetUnit 匹配函数需要在文档中明确说明。

V2 目标：

- 不影响原始项目和 H200 正在运行/已完成的实验。
- 创建独立 v2 项目目录，线上可与原始版本并行运行。
- 能复用不需要整改的离线数据，但不在原离线目录上写入。
- 多轮/多步方法输出统一的 `evidence_chunks`；单轮方法保留 `retrieved_chunks`，不伪造多轮 evidence log。
- 多轮 baseline 要修复当前几乎不触发检索的问题，但不强制固定检索轮数。
- 最终回答 prompt 统一到 Signpost 主实验约束。

## 2. Hybrid RAG 是否使用了 Signpost ES 数据库

当前代码：

- `signpost/baselines/vanilla_rag.py`
- `signpost/retrieval/chunk_search.py`
- `signpost/indexing/chunk_schema.py`

当前 Hybrid RAG 如果 `use_es=True`，会调用：

```python
search_chunks(namespace=self.namespace, mode=self.mode, top_k=self.top_k)
```

默认 index name：

```python
signpost-<namespace>-chunks
```

查询过滤：

```python
namespace == <namespace>
type == "chunk"
```

返回字段：

- `chunk_id`
- `doc_id`
- `file_name`
- `content`
- `start_line`
- `end_line`
- `section_path`
- `prev_chunk_id`
- `next_chunk_id`
- `score`
- `score_source`

chunk index schema 主要来自 `chunks.jsonl`，不是 unified graph。但代码里 `graph_es_sync.py` 有一个函数 `update_chunk_parent_fields()`，可能给同一个 chunk index 追加：

- `parent_summary_ids`
- `parent_summary_id`

当前 `search_chunks()` 没有读取这两个字段，Hybrid RAG prompt 也只使用 chunk content。但从控制变量角度，Hybrid RAG 共用 `signpost-<namespace>-chunks` 仍然容易被质疑。

V2 整改：

1. Hybrid RAG 不再默认使用 Signpost 的 `signpost-<namespace>-chunks`。
2. 采用方案 B：给 baseline 建独立 ES index，例如 `baseline-v2-<dataset>-chunks`，只写入 chunk content、基础定位字段、embedding，不写任何 Signpost graph/signpost 字段。
3. 论文中说明：
   - Hybrid RAG 使用同一套 chunking 和 embedding model，这是公平共享的数据准备。
   - 它不读取 Signpost offline/online cues。
   - 如果用 ES，则使用独立 baseline chunk index。

## 3. Silver 指标到底用什么

Silver evidence 已经抽取好，路径是：

- `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/agriculture/llm_silver_chunks.jsonl`
- `/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/datasets/processed/mix/llm_silver_chunks.jsonl`

这些文件是 gold/reference silver evidence，不是方法输出。

计算 silver 指标必须有两边，但 V2 不再把所有方法都纳入 silver evidence 指标主表：

1. Gold side：`llm_silver_chunks.jsonl` 中该问题的 silver chunks。
2. Method side：多轮/多步检索方法最终实际提供给 LLM 的证据 chunk/span。

单轮检索方法当然也有 retrieved chunks，但它们不是“多轮证据轨迹”。为了避免把 one-shot top-k retrieval 和 multi-round evidence acquisition 混在一起，V2 主表中 silver 相关指标只用于多轮/多步检索方法。

单轮方法处理：

- `hybrid_rag`、`agrag`、`linearrag`、`cluerag_prompt_normalized` 可以保留 `retrieved_chunks`，用于人工检查。
- 正式 silver 指标主表不计算这些单轮方法，填 NA。
- 这些方法仍参与最终答案指标，例如 TargetUnitRecall 和 LLM judge。

当前混乱点是 method side 有多个字段：

- `retrieved_chunks`：方法检索候选 chunk 列表。
- `citations`：方法最终引用/读取过的来源定位，通常是 `file_name + start_line/end_line`。
- `read_file` trace：Signpost agent 真实读取原文片段的工具调用顺序。

当前脚本为了兼容旧输出，用了 fallback：

```text
read_file trace -> citations -> retrieved_chunks
```

这导致不同方法口径不一致。V2 必须整改。

## 4. V2 evidence_chunks 记录口径

V2 不要求所有方法都生成 evidence log。单轮检索方法保留普通 `retrieved_chunks` 即可；只有 Signpost full/ablations 和两个多轮 baseline 需要新增统一字段：

```json
{
  "evidence_chunks": [
    {
      "rank": 1,
      "round": 1,
      "source": "read_file|retrieval_context",
      "query": "...",
      "chunk_id": "...",
      "doc_id": "...",
      "file_name": "...",
      "start_line": 1,
      "end_line": 10,
      "score": 0.0,
      "score_source": "...",
      "content_preview": "optional"
    }
  ]
}
```

统一原则：

- `retrieved_chunks` 继续保留，兼容旧逻辑。
- `citations` 不参与指标计算。
- silver 指标统一使用 `evidence_chunks`。
- 单轮检索方法不伪造 `round=1` 的 multi-round evidence，不进入多轮 silver 主表。
- 对多轮方法，`evidence_chunks` 只记录最终实际提供给 LLM 的证据，不记录中间候选。
- 对 Signpost，`evidence_chunks` 来自 `read_file` 后进入 synthesis prompt 的 snippets。
- 对 HipRAG，`evidence_chunks` 来自每次 `<search>` 后实际拼入 `<context>` 的 chunks。
- 对 GraphRAG-R1，`evidence_chunks` 来自每次 graph query 后实际拼入 `<|begin_of_documents|>` 的 document chunks。

公平性约束：

- 不是 Signpost 只算一个 chunk。Signpost 每个 subquestion 最多读 `read_top_k` 个 snippets，默认最多 3 个 subquestions、每个 3 个 snippets，因此最多约 9 个 evidence units。
- HipRAG/GraphRAG-R1 每轮可能给 LLM 多个 chunks。
- 指标统一在同一个 evidence budget 下算，例如 `@5`：按 evidence_chunks 的实际提供顺序取前 5 个证据单元。
- 同时输出每题 `num_evidence_chunks`，检查某方法是否系统性拿到更多证据。

推荐 V2 表：

1. 主答案质量表：所有方法都有 `AnswerTargetUnitRecall` 和 LLM judge。
2. Silver evidence 表：只包含 `signpost.full`、HipRAG、GraphRAG-R1；Signpost 消融不计算 Silver 指标；指标只用 `evidence_chunks`。

这样避免“单轮方法哪来的多轮 evidence log”的概念混乱，也避免 Signpost 只用一个 `read_file` 与其他方法多个 chunks 对比。

## 5. citations 是什么，V2 如何处理

当前 `citations` 不是 gold silver，也不是统一检索结果。

它是方法输出中的来源定位字段：

- Hybrid/AGRAG/LinearRAG：由最终用于 context 的 chunks 转成 `file_name/start_line/end_line/locate`。
- Signpost：由 `read_file` 读到的 snippets 转成 citation。
- ClueRAG normalized：当前 conversion 里 citations 可能为空，retrieved_chunks 有值。

问题：

- citations 语义是“最终引用/读取位置”，不一定等于“检索候选 top-k”。
- 有的方法有 citations，有的方法没有。
- 用 citations fallback 会造成口径不一致。

V2 整改：

- citations 仅保留做人工检查和 source tracking。
- 指标不再默认读取 citations。
- silver 指标统一读取 `evidence_chunks`。
- 单轮方法的 `retrieved_chunks` 不进入 silver 主表。

## 6. ClueRAG pipeline 是一次检索还是多次检索

当前 ClueRAG adapter 不是 LLM 多轮 tool-use agent。它是一次 query-level retrieval pipeline，内部包含多个检索子步骤。

当前 `run_cluerag_shared()` 对每个问题调用一次：

```python
_retrieve_shared_cluerag(...)
```

这个函数内部会产生多个候选来源：

1. direct chunk retrieval：对原始 question 做 chunk 检索。
2. query NER / entity linking：从 query 抽或匹配实体。
3. KU matching：检索 knowledge units。
4. graph expansion：围绕 KU/entity 在 ClueRAG graph 上扩展。
5. rerank：对候选 chunks rerank。
6. top_n chunks 进入 generation。

所以它不是“只一次 ES search 返回多个 chunk”这么简单，也不是“LLM 多轮检索”。更准确说：

```text
一次问题输入 -> 一个多组件检索 pipeline -> 多批候选合并/扩展/重排 -> top_n chunks
```

V2 文档和论文中应表述为：

- one-query multi-stage retrieval pipeline
- not iterative LLM-agent retrieval

V2 指标：

- ClueRAG 是单轮 multi-stage pipeline，不进入多轮 silver 主表。
- 不做 one-shot retrieval diagnostic。
- 仍保留原始 `retrieved_chunks`，只用于人工排查，不进正式 silver 指标。

## 7. HipRAG 和 GraphRAG-R1 多轮整改

当前代码支持多轮，但有一个问题：

- 每一步是否检索依赖 LLM 是否输出 `<search>...</search>` 或 graph query tag。
- 如果模型没有按格式输出 query，就会出现没有 search、没有 reliable retrieved chunks，最后 silver 指标只能 NA 或很差。

这不适合作为正式 baseline。这里的问题不是“所有 query 都不需要证据”，而是当前 adapter/prompt/解析很可能没有让模型稳定按方法预期发起检索。

V2 不强制固定每题检索轮数，也不强制每轮都必须检索。整改目标是修复“本该能检索但实际几乎不检索”的实现问题，让方法按原本机制自然决定是否继续检索。

### HipRAG-v2

当前逻辑：

- step prompt 要求 LLM 如果需要证据就输出 `<search>query</search>`。
- 不输出 search 就跳过检索。
- 没有 `read_file` 概念；检索到的 chunks 会直接被 `join_context()` 拼入 `<context>` 后提供给 LLM。

整改：

1. 先检查现有输出为什么几乎没有 search：
   - LLM 是否直接在第一步输出 `<answer>`。
   - prompt 是否给了“可以直接回答”的倾向。
   - tag 解析函数 `_extract_last_search()` 是否能解析 H200 模型实际输出。
   - stop condition 是否过早触发。
2. 修改 prompt，让模型在没有充分证据时优先发起 `<search>`，但不规定每题必须固定检索几轮。
3. 增强解析鲁棒性：
   - 支持大小写/空白/换行变体。
   - 支持模型输出接近但不完全标准的 search tag。
   - 记录无法解析的 raw step，便于 audit。
4. 如果模型第一步直接回答，但没有任何已检索证据，应视为 baseline 执行失败或 retry，而不是当作正常无检索结果。
5. 每次实际 search 后，把真正拼入 `<context>` 并提供给 LLM 的 chunks 写入 `evidence_chunks`，保留 `round` 和 `query`。
6. 最终生成只能使用累计检索 evidence；如果没有任何检索证据且方法要求 private-corpus QA，应输出失败状态或 `Insufficient evidence.`，不能用无检索参数知识回答。

这样 HipRAG-v2 保持原本“按需要多轮检索”的机制，同时避免因为格式/实现问题退化成无检索 LLM。

### GraphRAG-R1-v2

当前逻辑：

- step prompt 要求 LLM 输出 graph query tag。
- 不输出 graph query 就不检索。
- 没有 `read_file` 概念；graph search 返回 graph facts 和 chunks，chunks 会被 `join_context()` 拼入 `<|begin_of_documents|>` 后提供给 LLM。

整改：

1. 先检查现有输出为什么几乎没有 graph query：
   - LLM 是否直接输出 `<answer>`。
   - graph query tag 解析是否失败。
   - prompt 是否没有足够强调 private corpus evidence。
   - graph search 是否异常但被吞掉。
2. 修改 prompt，让模型在缺少证据时发起 graph query，但不强制固定轮数。
3. 增强 graph query 解析鲁棒性。
4. 如果没有任何 graph retrieval 就直接回答，应标记为失败或 retry。
5. 每次实际 graph search 后，把真正拼入 documents 并提供给 LLM 的 chunks 写入 `evidence_chunks`，保留 `round` 和 `query`。
6. 最终生成使用累计 graph facts + documents。

注意：

- 这属于 baseline adapter 修复，不是修改方法核心思想。
- 文档中需要说明 v2 不是强制固定检索轮数，而是修复当前实现没有稳定触发检索的问题。

## 8. Signpost Full 的 evidence 口径整改

当前 Signpost prediction 里已经有 `retrieved_chunks`，来自 `agent/batch.py` 的 `_retrieved_chunks(result)`：

- 遍历每个 research result 的 retrieval。
- 收集 `text_group` / `graph_group` items 的 `chunk_id/id/node_id`。
- 也收集 item 顶层 `source_chunk_ids`。

问题：

- 对 graph/entity/relation item，`id/node_id` 可能不是 chunk id。
- 旧 `_retrieved_chunks` 没有完整 file_name/start_line/end_line。
- 当前 silver 指标优先使用 `read_file` trace，这和 baseline 的 `retrieved_chunks` 口径不同。

V2 整改：

1. 不用 `knowledge_search` 候选计算 silver 指标。
2. Signpost 的 `evidence_chunks` 只来自最终 synthesis prompt 中实际使用的 `read_file` snippets。
3. 每个 snippet 记录 file_name/start_line/end_line/locate。如果能映射到 chunk_id，也记录 chunk_id；不能映射时用 file/line 与 silver chunk 做 overlap。
4. 父子章节、semantic cues、online PPR recommendations 不作为 evidence chunk 直接评分；它们只影响选哪些 locates 去 read_file。
5. 多轮 silver 主表只和 HipRAG/GraphRAG-R1 这类多轮 baseline 比较，不与 Hybrid/AGRAG/LinearRAG/ClueRAG 的单轮 top-k 混表。

这样可以避免“Signpost 用一个 read_file，其他方法用多个 chunks”的不公平：Signpost 的所有实际进入 synthesis 的 read_file snippets 都会按顺序进入 `evidence_chunks`，再统一取 `@5`。

## 9. Signpost 消融重新解释与可能整改

### 9.1 no_offline 到底去掉了什么

当前代码：

```python
item["offline_signpost"] = {}
```

这会删除 item 上整个 offline signpost：

- `vertical`
- `horizontal`
- `provenance`
- `semantic`

不是“保留 offline_signpost.provenance”。相反，`provenance` 也被删除。

为什么会影响 read_file：

- `collect_locates()` 主要从 `offline_signpost.provenance.locate` 和 `offline_signpost.provenance.source_locates` 取原文位置。
- 删除 offline_signpost 后，这些位置消失。
- 代码还会尝试读 item 顶层的 `source_locates`，但 chunk item 往往依赖 provenance locate。

V2 文档中必须明确：

```text
no_offline = 删除所有离线 signpost cues，包括 provenance。因此它不仅去掉提示信息，也会削弱 read_file 定位能力。
```

V2 保持原本设计，不拆新消融。论文里需要承认：当前 `no_offline` 同时删除 provenance，所以它消融的是完整 offline signpost 信息，而不是只删除提示文字。

### 9.2 no_online 是不是只去掉 PPR 推荐

是。当前 `no_online` 只清空 group-level `online_signpost`：

- `subgraph.nodes = 0`
- `subgraph.edges = 0`
- `recommended_entities = []`

它保留：

- `knowledge_search`
- `read_file`
- offline signpost
- source locates
- Supervisor/Researcher 多步流程

因此当前 `no_online` 应解释为：

```text
no_online_signpost_recommendations
```

不是：

```text
no_online_retrieval
```

V2 保持原本消融设计，不新增消融，不拆分消融。需要做的是在文档和论文里把含义说准确：

```text
no_online = 删除 online PPR/signpost recommendations，不是关闭在线检索流程。
```

### 9.3 no_semantic_cues 为什么也删 online signpost

当前代码：

```python
if variant in {NO_ONLINE, NO_SEMANTIC_CUES}:
    group["online_signpost"] = empty
if variant == NO_SEMANTIC_CUES:
    signpost.pop("semantic", None)
```

因此 `no_semantic_cues` 实际删除两类东西：

1. offline semantic cues。
2. group-level online signpost recommended entities。

这不是单纯的 offline semantic cue ablation，而是 composite ablation。代码可能这样设计的原因是：online PPR recommendations 本质上也会推荐 entity jump，和 semantic navigation 相关。

V2 保持原本设计，不拆成更多消融。文档和论文中必须明确：

```text
no_semantic_cues = 删除 offline semantic cues，同时删除 online signpost recommendations。
```

### 9.4 no_provenance_cues 是不是删掉溯源，导致读不了原始 chunk

是。当前代码：

```python
signpost.pop("provenance", None)
item.pop("source_locates", None)
item.pop("source_chunk_ids", None)
```

它删除：

- locate
- file_name/start_line/end_line provenance
- source_locates
- source_chunk_ids
- source_mapping

实际影响：

- Researcher 很难通过 `read_file` 定位原文。
- 对 chunk item，如果顶层仍有 file_name/start_line/end_line，理论上可以直接构造 locate，但当前 `collect_locates()` 没有这么做。

V2 保持原本设计。也就是说：

```text
no_provenance_cues = 删除溯源信息，因此会削弱或阻断 read_file 精确读取原始 chunk 的能力。
```

这应被解释为 provenance 对“能否定位并读取原文证据”的消融，而不是只删除展示给模型看的 citation 文本。

## 10. Prompt 控制整改

参考已有文档：

`docs/prompt_control_and_rerun_plan_20260526.zh.md`

V2 统一原则：

1. 所有 RAG/graph/multi-round 方法最终回答必须严格基于该方法可见 evidence。
2. 答案必须英语。
3. 答案完整、独立成句。
4. 不在 final answer 中写 citation、文件名、行号。
5. 不写 conversational filler。
6. evidence 不足时输出 exactly `Insufficient evidence.`
7. 输出格式可保留原方法格式，避免破坏解析逻辑。

Vanilla LLM 特殊：

- Vanilla LLM 保持原本 prompt，不改。
- 它作为“无检索参数知识 baseline”，不参与 evidence-grounded prompt 统一。
- 论文中需要明确它不是 evidence-grounded RAG 方法。

## 11. TargetUnitRecall 匹配函数说明

当前脚本函数：

- `target_unit_recall()`
- `unit_phrase_matches()`

当前算法：

1. 从 `llm_target_units.jsonl` 读取每题 target units。
2. 只评估 `required=True` 的 units。
3. 从 prediction 中抽最终答案：
   - JSON 里有 `answer` 则用 `answer`。
   - 否则如果有 `<answer>...</answer>` 则用标签内文本。
   - 否则用整个 prediction。
4. 对答案和 target unit 做 `normalize_answer()`。
5. 对每个 required unit：
   - 候选文本包括 `unit["text"]` 和 `unit["aliases"]`。
   - 如果 normalized unit phrase 是 normalized answer 的子串，命中。
   - 如果 unit token 数 <= 2，则要求 unit tokens 是 answer tokens 的子集。
   - 如果 unit token 数 > 2，则要求 token overlap >= 0.75。
6. 每题：

```text
TargetUnitRecall = 命中的 required target units 数 / required target units 总数
```

更详细解释：

- `normalize_answer()` 会做标准化，通常包括小写化、去掉多余标点/空白、统一一些冠词或格式差异。目的是让 `Carbon sequestration` 和 `carbon sequestration` 这类大小写差异不影响匹配。
- `text` 是抽取脚本为某个 target unit 写出的标准表达，也就是这个应答要点的 canonical form。例如某个 unit 的 `text` 可能是 `biodiesel production`。
- `aliases` 是同一个 target unit 的可接受替代表达或同义表述列表。例如同一个 unit 可能有 aliases：`biofuel production`、`producing biodiesel`。如果答案没有逐字写出 `text`，但写中了任意一个 alias，也算覆盖这个 target unit。
- 如果某个 target unit 没有 aliases，就只用 `text` 匹配。
- “短 unit”指 normalize 后 token 数小于等于 2 的 unit。例如 `composting`、`crop rotation`。短 unit 不做 0.75 overlap，因为一个词的 0.75 overlap 太容易误判；它要求答案里包含这些 token。
- “长 unit”指 token 数大于 2 的 unit。例如 `hydroponic growing and aquaponics`。这类 unit 允许答案换词序或轻微改写，只要 unit 里的 token 有至少 75% 出现在答案中，就算命中。
- 例子 1：target unit 是 `hydroponic growing`，答案含有 `hydroponic growing`，命中。
- 例子 2：target unit 是 `community engagement and education`，答案含有 `community engagement and local education programs`，token overlap 足够高，可能命中。
- 例子 3：target unit 是 `biodiesel production`，答案只写了 `biofuel`，如果 alias 里没有 `biofuel`，当前规则可能不命中。
- 例子 4：答案只堆了关键词但事实关系错了，当前规则仍可能误命中，因为它不是语义判定，只是 deterministic text/token matching。

风险：

- 同义改写可能漏匹配。
- 短词可能偶然匹配。
- token overlap 不判断事实关系是否正确。
- 答案把信息写在 rationale 而非 answer 时，取 answer 会影响分数。

V2 整改建议：

- 保留当前 deterministic recall，作为可复现指标。
- 输出 per-query covered/missed units，便于人工检查。

不新增 LLM verified 指标。

## 12. Silver chunks 是否更贴近 flat retrieval

需要谨慎分析。

用户补充：silver evidence 抽取是跳过 hybrid 获取的，因此它不是直接由 Hybrid RAG 检索产生。

但仍可能出现 silver 指标更利于 flat chunk retrieval，原因不是“silver 来自 Hybrid”，而是：

- silver chunk 是 chunk/span 层级的 gold evidence。
- Hybrid RAG 正好也是直接 chunk top-k。
- Signpost 可能通过 entity/relation/summary 找到间接路径，再 read_file 到相邻或聚合证据，和 silver chunk span 不完全对齐。

V2 先暂时保留原本 silver 抽取设计，不重新抽 silver。需要做 per-query audit：

- Hybrid 命中 silver 但答案没覆盖 target unit。
- Signpost 没命中 silver 但答案覆盖 target unit。
- Signpost 读到的 span 与 silver span 是否等价但不重叠。

## 13. V2 项目隔离方案

H200 上新建独立项目目录：

```text
/home/srl/signpost_re_v2
```

本地对应：

```text
/home/ruolinsu/signpost/signpost_re_v2
```

创建方式：

```bash
rsync -a --exclude outputs --exclude .git /home/srl/signpost_re/ /home/srl/signpost_re_v2/
```

或从本地修改后同步到 H200 v2 目录。

环境：

- 复用同一个 conda env：`signpost-re`
- 复用同一套 H200 服务：
  - chat `http://localhost:8000/v1`
  - embedding `http://localhost:8001/v1/embeddings`
  - rerank `http://localhost:8033/v1/rerank`
  - ES `http://127.0.0.1:9200`

输出目录必须独立：

```text
/home/srl/signpost_re_v2/outputs_v2/<dataset>/...
```

或：

```text
/home/srl/signpost_re_v2/outputs/<dataset>/...
```

只要不写 `/home/srl/signpost_re/outputs` 即可。

## 14. 离线数据复用策略

可以复用的只读数据：

- `documents.jsonl`
- `chunks.jsonl`
- `questions.jsonl`
- `semantic_llm.extractions.jsonl`
- `graph.semantic.json`
- `graph.structure.json`
- `graph.sequence.json`
- `graph.unified.json`
- AGRAG/LinearRAG/ClueRAG 已构建的 baseline graph/index cache，如果代码没有改其离线结构。
- `llm_target_units.jsonl`
- `llm_silver_chunks.jsonl`

不能直接复用或建议重建的部分：

- Hybrid RAG 的 ES chunk index：建议建 v2 baseline 独立 index。
- Signpost 的 v2 `evidence_chunks` 输出：必须重跑在线。
- HipRAG/GraphRAG-R1：由于要修复当前几乎不触发检索的问题，必须重跑在线；若离线 index 结构不变，可复用离线 cache。
- Prompt 修改影响最终答案，所有纳入论文质量比较的方法都要重跑 online generation。

原则：

- V2 只读原始 processed artifacts。
- V2 不修改原始 outputs。
- 如果某个离线 cache 复用，复制到 v2 outputs/cache 目录后再读，避免写回原目录。

## 15. H200 并发运行设计

因为论文不汇报 online time，本轮 online 可并发。

V2 runner 需要支持：

- `--workers N`
- `--shard-index i --num-shards N`
- 或内部 ThreadPool/ProcessPool 并发。

推荐实现：

1. 每个 method/dataset 一条 job。
2. job 内按 question 并发。
3. 每个 worker 写独立 shard 文件：

```text
predictions/<method>.shard-000.jsonl
predictions/<method>.shard-001.jsonl
...
```

4. 全部完成后 merge，按 question_id 排序，生成：

```text
predictions/<method>.jsonl
```

避免多个进程同时 append 同一个 JSONL。

并发限制建议：

- chat LLM 并发：根据 H200 70B 服务吞吐先从 4 或 8 开始。
- embedding 并发：baseline retrieval 可适当高一些，但避免压垮 embedding 服务。
- rerank 并发：ClueRAG rerank 先限制 2-4。

## 16. V2 评测脚本整改

新增脚本建议：

```text
scripts/h200_target_unit_silver_eval_v2.py
```

输入：

- `llm_target_units.jsonl`
- `llm_silver_chunks.jsonl`
- `predictions/<method>.jsonl`

主指标：

```text
TargetUnitRecall
SilverHit@5
SilverRecall@5
MRR
ClaimCoverage@5
num_evidence_chunks
```

计算口径：

- `TargetUnitRecall`：所有方法都计算。
- `Silver* / MRR / ClaimCoverage@5`：只对 `signpost.full`、HipRAG、GraphRAG-R1 计算，读取 `evidence_chunks`。
- 单轮检索方法的 silver 指标主表填 NA。
- 不再 fallback 到 citations。
- 如果方法没有 `evidence_chunks`，silver 指标填 NA，不做隐式转换。
- `num_evidence_chunks` 记录每题实际进入 LLM 的证据单元数量，用于检查证据预算是否公平。

额外输出：

- 每题 missed target units。
- 每题 hit/missed silver chunks。
- evidence_chunks 明细。
- method-level summary。

## 17. V2 方法整改清单

### 必须改

- Hybrid RAG：
  - 使用独立 baseline ES index：`baseline-v2-<dataset>-chunks`。
  - 不输出 `evidence_chunks`；保留 `retrieved_chunks` 用于人工检查。
  - 统一 prompt。

- ClueRAG prompt normalized：
  - 明确 multi-stage one-query pipeline。
  - 不进入 silver 主表。
  - 统一 final prompt。

- AGRAG / LinearRAG：
  - 不输出 `evidence_chunks`；保留 `retrieved_chunks`。
  - 统一 final prompt。

- HipRAG / GraphRAG-R1：
  - 修复当前实现几乎不触发检索的问题。
  - 不强制固定轮数，不强制每题必须检索相同轮数。
  - 每次实际提供给 LLM 的检索 chunks 输出到 `evidence_chunks`。
  - 统一 final prompt。

- Signpost full / ablations：
  - 输出最终 synthesis prompt 实际使用的 read_file snippets 到 `evidence_chunks`。
  - Silver 指标只计算 `signpost.full`，不计算消融。
  - 保持原本消融设计，只修正文档/论文解释。

- Evaluation：
  - silver 指标只用 `evidence_chunks`。
  - 单轮检索方法 silver 主表填 NA。
  - 不使用 citations，不使用 fallback。

### 已确认决策

- `no_online` 保持原本实现，不新增真正 no-online-retrieval；论文中解释为删除 online signpost recommendations。
- `no_semantic_cues` 保持原本 composite ablation，不拆分。
- `no_provenance_cues` 保持原本设计，即删除溯源并削弱 read_file。
- Vanilla LLM 保持原本 prompt。

## 18. 推荐执行顺序

1. 确认本文档方案。
2. 本地创建 `signpost_re_v2` 或在当前 repo 开 v2 分支/目录。
3. 实现多轮/多步方法的 `evidence_chunks` schema。
4. 修 Hybrid RAG 独立 baseline index。
5. 修 HipRAG/GraphRAG-R1 不稳定触发检索的问题。
6. 修 Signpost retrieval/read evidence 输出。
7. 统一 prompt。
8. 写 `h200_target_unit_silver_eval_v2.py`。
9. 在本机 v2 目录用 ecnu 配置做 HipRAG/GraphRAG-R1 小样本 smoke test：
   - 不在原始 `signpost_re` 里改代码或写输出。
   - 先用 `--limit 2` 或更小样本确认能跑通。
   - 检查 HipRAG 至少能产生 `<search>` 并执行 `hiprag_private_chunk_search`。
   - 检查 GraphRAG-R1 至少能产生 graph query 并执行 `graphrag_r1_graph_search`。
   - 检查每条 prediction 有 `evidence_chunks`，且其中记录的是实际进入 LLM context/documents 的 chunks。
   - 检查没有检索却直接回答的 case 被标记为失败或 retry，而不是当作正常结果。
10. 根据本机 smoke test 的真实输出，再最终确认 HipRAG/GraphRAG-R1 的 evidence_chunks 字段。
11. 同步到 H200 `/home/srl/signpost_re_v2`。
12. 复用只读 processed/offline artifacts。
13. 并发跑 agriculture + mixv0。
14. 下载到新的本地目录，例如：

```text
/home/ruolinsu/signpost/h200_v2_backup_20260527
```

15. 跑 v2 评测。
16. 抽样人工核查反直觉 query。

## 19. 本机 v2 smoke test 结果

本机 v2 目录：

```text
/home/ruolinsu/signpost/signpost_re_v2
```

ECNU 配置来自：

```text
/home/ruolinsu/signpost/ecnu.txt
/home/ruolinsu/signpost/signpost_re_v2/.env.local.ecnu
```

本机 ECNU 模型：

```text
ECNU_API_BASE=https://chat.ecnu.edu.cn/open/api/v1
ECNU_CHAT_MODEL=ecnu-plus
ECNU_EMBEDDING_MODEL=ecnu-embedding-small
```

已完成的检查：

1. Python 静态检查通过：

```bash
python -m py_compile \
  signpost/baselines/common.py \
  signpost/baselines/hiprag.py \
  signpost/baselines/graphrag_r1.py \
  signpost/baselines/vanilla_rag.py \
  signpost/agent/batch.py \
  scripts/h200_target_unit_silver_eval_v2.py
```

2. HipRAG v2 agriculture `LIMIT=1` smoke：

```text
mode=hybrid
tool_calls=2
llm_calls=3
evidence_chunks=4
retrieved_chunks=5
实际触发 hiprag_private_chunk_search 两次
```

第一条样本可见：

- 第 1 轮 query：`steps extracting handling honey strain honey importance`
- 第 2 轮 query：`steps uncapping filtering straining debris wax after extraction honey`
- `evidence_chunks` 记录了实际拼入 `<context>` 的 chunks。

3. GraphRAG-R1 v2 agriculture `LIMIT=1` smoke：

```text
mode=bm25
tool_calls=1
llm_calls=2
evidence_chunks=1
retrieved_chunks=6
实际触发 graphrag_r1_graph_search 一次
```

第一条样本可见：

- graph query：`steps involved in extracting and handling honey and importance of straining honey after extraction`
- `evidence_chunks` 记录了实际拼入 `<|begin_of_documents|>` 的 document chunk。

注意：

- GraphRAG-R1 第一次本机 smoke 使用 ECNU embedding 构建 full agriculture graph index，23,699 triples，共 741 个 embedding batches，用时约 12 分钟。
- 构建完成后复用 `index.pkl`，online `LIMIT=1` 用时约 18 秒。
- 本机 ECNU chat 偶发 HTTP 500，已在 baseline shared `chat_once()` 中加入 `LLM_RETRIES/RETRY_SLEEP` retry。H200 也应设置 `LLM_RETRIES=5 RETRY_SLEEP=5`。

## 20. H200 v2 部署与运行

目标：不影响原始 H200 项目 `/home/srl/signpost_re`，在独立目录运行 v2：

```text
/home/srl/signpost_re_v2
```

### 20.1 同步 v2 项目

本地打包建议：

```bash
cd /home/ruolinsu/signpost
tar --exclude='signpost_re_v2/outputs' \
    --exclude='signpost_re_v2/.pytest_cache' \
    --exclude='signpost_re_v2/**/__pycache__' \
    --exclude='signpost_re_v2/**/*.pyc' \
    -czf /home/ruolinsu/signpost/h200/signpost_re_v2.tar.gz \
    signpost_re_v2
```

这是完整 v2 项目包，但不包含 `outputs`、pytest cache 和 Python bytecode。

上传到 H200 后：

```bash
cd /home/srl
tar -xzf /home/srl/signpost_re_v2.tar.gz
```

如果 H200 已有 `/home/srl/signpost_re_v2`，建议先备份该目录或只 rsync 覆盖代码文件，避免覆盖已跑出的 outputs。

### 20.2 H200 环境

v2 脚本会显式设置：

```text
PROJECT_DIR=/home/srl/signpost_re_v2
chat base: http://localhost:8000/v1
chat model: /data/srl/Llama-3.3-70B-FP8
embedding base: http://localhost:8001/v1/embeddings
embedding model: /data/srl/nemotron-8b
rerank: http://localhost:8033/v1/rerank
rerank model: /data/srl/llama-nemotron-rerank-1b-v2
ES: http://127.0.0.1:9200
```

### 20.3 每个数据集一条指令

脚本：

```text
/home/srl/signpost_re_v2/scripts/h200/run_v2_dataset_all.sh
```

每个数据集一条命令：

```bash
tmux new-session -d -s v2-agriculture "bash -lc '/home/srl/signpost_re_v2/scripts/h200/run_v2_dataset_all.sh agriculture agriculture agriculture'"
tmux new-session -d -s v2-mixv0 "bash -lc '/home/srl/signpost_re_v2/scripts/h200/run_v2_dataset_all.sh mixv0 mix mix'"
```

说明：

- 第一个参数是 output dataset。
- 第二个参数是 namespace。
- 第三个参数是 processed dataset。
- `mixv0` 的 outputs 写到 `outputs/mixv0`，processed 输入读 `datasets/processed/mix`。
- `mixv0` 的 namespace 使用 `mix`，复用 H200 旧 ES 中的 `signpost-mix-*` 和 `cluerag-mix-multilayer`。

这两条 tmux 可以同时启动，实现数据集级并发。单个数据集内部按依赖顺序执行，避免多个方法同时写同一个 metrics 文件造成竞态；方法内部按 question 并发，默认 `V2_QUERY_WORKERS=8`，H200 服务稳定后可以提高到 12 或 16。

### 20.4 单数据集内部执行顺序

脚本内部顺序：

```text
1. 服务 smoke check
2. 构建 baseline 独立 ES chunk index: baseline-v2-<dataset>-chunks
3. Signpost full + ablations
4. vanilla_llm
5. hybrid_rag，使用 baseline-v2-<dataset>-chunks，不用 Signpost chunk index
6. cluerag 中间检索/图构建
7. cluerag_prompt_normalized
8. agrag
9. linearrag
10. hiprag
11. graphrag_r1
12. h200_target_unit_silver_eval_v2.py
```

Silver 指标只对以下方法计算：

```text
signpost.full
hiprag
graphrag_r1
```

以下方法的 Silver 指标填 NA：

```text
vanilla_llm
hybrid_rag
cluerag_prompt_normalized
agrag
linearrag
signpost.no_offline
signpost.no_online
signpost.no_semantic_cues
signpost.no_provenance_cues
signpost.no_vertical_cues
signpost.no_horizontal_cues
```

所有方法仍计算 TargetUnitRecall。

### 20.5 并发与复用建议

并发：

- 建议 agriculture 和 mixv0 两个 tmux 同时跑。
- 单个数据集内部保持顺序，避免 metrics 文件写入竞态。
- H200 上 8000/8001/8033 若稳定，可以再开更高并发；但 embedding-heavy baseline 首次建 index 时会明显占用 8001。

首次运行：

- Hybrid RAG 会构建 v2 自己的 `baseline-v2-<dataset>-chunks`，不要复用 Signpost chunk index。
- agriculture 未看到 AGRAG/LinearRAG/HiPRAG/GraphRAG-R1 的旧 `index.pkl`，不要全局设置 `REUSE_BASELINE_INDEX=1`。
- mix 可以按方法设置 `REUSE_AGRAG_INDEX=1 REUSE_LINEARRAG_INDEX=1 REUSE_HIPRAG_INDEX=1 REUSE_GRAPHRAG_R1_INDEX=1` 读取旧 `index.pkl`。
- ClueRAG 只软链旧 `shared_graph`；`shared_outputs` 不软链到 v2 输出，避免写回旧项目。

后续只重跑 online/prompt：

```bash
tmux new-session -d -s v2-agriculture-rerun "bash -lc 'cd /home/srl/signpost_re_v2 && REUSE_BASELINE_CHUNK_INDEX=1 REUSE_GRAPH=1 /home/srl/signpost_re_v2/scripts/h200/run_v2_dataset_all.sh agriculture agriculture agriculture'"
tmux new-session -d -s v2-mixv0-rerun "bash -lc 'cd /home/srl/signpost_re_v2 && REUSE_BASELINE_CHUNK_INDEX=1 REUSE_GRAPH=1 REUSE_AGRAG_INDEX=1 REUSE_LINEARRAG_INDEX=1 REUSE_HIPRAG_INDEX=1 REUSE_GRAPHRAG_R1_INDEX=1 /home/srl/signpost_re_v2/scripts/h200/run_v2_dataset_all.sh mixv0 mix mix'"
```

### 20.6 完整性检查

每个数据集跑完后检查：

```bash
cd /home/srl/signpost_re_v2
for d in agriculture mixv0; do
  echo "== ${d} =="
  ls outputs/${d}/predictions/*.jsonl
  ls outputs/${d}/logs/*.query.jsonl
  ls outputs/${d}/metrics/*.json
  test -f outputs/${d}/metrics/target_unit_silver_eval_v2/method_target_unit_silver_metrics.tsv
done
```

检查 HipRAG/GraphRAG-R1 evidence：

```bash
cd /home/srl/signpost_re_v2
python - <<'PY'
import json
for dataset in ["agriculture", "mixv0"]:
    for method in ["hiprag", "graphrag_r1", "signpost.full"]:
        path = f"outputs/{dataset}/predictions/{method}.jsonl"
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        counts = [len(row.get("evidence_chunks") or []) for row in rows]
        print(dataset, method, "rows", len(rows), "zero_evidence", sum(1 for c in counts if c == 0), "avg_evidence", sum(counts)/max(1,len(counts)))
PY
```
