# 外部 Baseline 指标公式与 Prompt 清单

本文档补充 `external_baseline_evaluation_audit_zh.md`，专门回答两个问题：

1. 技术说明里出现的指标，能否从技术说明或代码确认计算方式。
2. 每个 baseline 使用了哪些 prompt，这些 prompt 分别用于什么阶段。

结论先行：有些指标能从代码精确复现，例如 LinearRAG 的 `Contain Accuracy`、GraphRAG-R1 的规则 `F1/EM/SBERT`、HiPRAG 的 `cover_exact_match`；有些指标只能从技术说明得到定义，当前下载的官方代码没有完整实现，例如 ClueRAG 技术说明里的 QA `F1/Acc.`、LinearRAG 的 `Context Relevance/Evidence Recall`、AGRAG 的 `coverage_score/faithfulness` wrapper。

正式 项目实验仍建议以 Signpost 统一输出和统一评测为主；官方指标只作为 baseline 技术说明口径说明和附录补充。

## 1. 指标可复现性总表

| 方法 | 指标 | 来源 | 计算方式是否明确 | 备注 |
| --- | --- | --- | --- | --- |
| ClueRAG | `Accuracy / Acc.` | 技术说明 | 明确但代码入口缺失 | 技术说明定义为 gold answer 是否包含在 generated answer 中；当前仓库 `main.py` 引用的 `calculate_metric_scores` 在本地代码中不存在 |
| ClueRAG | `F1` | 技术说明 | 明确但代码入口缺失 | 技术说明称为 generated answer 与 gold answer 的 token-level overlap F1 |
| ClueRAG | `Recall@2/5/10` | 代码 | 明确 | top-k retrieved chunk ids 命中 gold supporting chunks 的比例 |
| ClueRAG | offline/online token cost | 技术说明 + 代码 metadata | 部分明确 | 技术说明要求统计 prompt + completion tokens；当前代码能记录部分 LLM metadata |
| LinearRAG | `Contain Accuracy` | 代码 + 技术说明 | 明确 | normalize 后判断 gold 是否为 prediction 子串 |
| LinearRAG | `LLM/GPT Accuracy` | 代码 + 技术说明 | 明确 | LLM judge 返回 `correct/incorrect`，平均 0/1 |
| LinearRAG | `Context Relevance` | 技术说明 | 概念明确，代码缺完整实现 | 衡量 retrieved passages 与 question 的语义相关性 |
| LinearRAG | `Evidence Recall` | 技术说明 | 概念明确，代码缺完整实现 | 衡量 retrieved contents 是否包含回答所需全部证据 |
| AGRAG | `ACC.` | 技术说明 + 部分代码 | 部分明确 | 分类任务为 label exact match；其他任务由 LLM 打 0/0.5/1，当前 wrapper 依赖外部 `metrics` |
| AGRAG | `REC.` | 技术说明 + 分类代码 | 部分明确 | 技术说明定义为 TP/(TP+FN)，分类任务使用 macro recall；旧分类脚本使用 sklearn weighted recall |
| AGRAG | `ROUGE-L` | 技术说明 + wrapper | 部分明确 | 基于 longest common subsequence；当前代码调用外部 `rouge_score` |
| AGRAG | `Coverage` | 技术说明 + wrapper | 概念明确，代码缺实现 | 衡量 response 覆盖 reference evidences 的程度 |
| AGRAG | `Faithfulness` | 技术说明 + wrapper | 概念明确，代码缺实现 | answer claims 中被 context 支持的比例 |
| HiPRAG | `CEM` | 代码 + 技术说明 | 明确 | normalize 后任一 gold answer 是 prediction 子串即正确 |
| HiPRAG | format correctness | 代码 | 明确 | 正则检查 `<think><step>...</step></think><answer>...</answer>` |
| HiPRAG | `OSR/USR` | 技术说明 + analysis 代码 | 明确但依赖 LLM judge | over-search / all search steps；under-search / all non-search steps |
| HiPRAG | `Avg. #Searches` | 技术说明 + 输出结构 | 明确 | 每题 search step 数量平均 |
| GraphRAG-R1 | `EM` | 代码 | 明确 | normalize 后 exact match |
| GraphRAG-R1 | `cover_em_1/2` | 代码 | 明确 | gold tokens 是否被 pred 覆盖；`cover_em_2` 要求连续或子串 |
| GraphRAG-R1 | `F1 / Precision / Recall` | 代码 + 技术说明 | 明确 | normalize 后 token Counter overlap |
| GraphRAG-R1 | `SBERT` | 代码 + 技术说明 | 明确 | `shibing624/text2vec-base-chinese` embedding cosine |
| GraphRAG-R1 | `ACCL / LLM judge accuracy` | 代码 + 技术说明 | 明确 | Qwen judge 返回 True/False |
| GraphRAG-R1 | `#Calls / #Token / ACCF` | 技术说明 + 代码输出 | 部分明确 | calls/tokens 来自推理输出；ACCF 是格式完全正确比例 |

## 2. 指标计算细节

### 2.1 ClueRAG

代码证据：

```text
baselines/ClueRAG/retrieval/retrieval.py
baselines/ClueRAG/utils/prompt.py
```

检索 `Recall@k`：

```text
hits = number of retrieved chunk ids in top-k that appear in ground_truth_chunks
recall@k = hits / len(ground_truth_chunks)
k in {2, 5, 10}
final score = average over questions
```

技术说明 QA 指标：

```text
Accuracy:
  generated answer contains the gold answer => 1
  otherwise => 0

F1:
  token-level overlap F1 between generated answer and gold answer
```

当前限制：

```text
ClueRAG/main.py imports calculate_metric_scores,
but baselines/ClueRAG/utils/utils.py does not define it.
```

所以 ClueRAG 的技术说明 `F1/Acc.` 能从技术说明确认，但不能从当前官方代码直接调用。接入我们的实验时，应转成 Signpost prediction schema，再跑统一 `basic_eval/query_metrics`。

### 2.2 LinearRAG

代码证据：

```text
baselines/LinearRAG/src/evaluate.py
baselines/LinearRAG/src/utils.py
```

`normalize_answer`：

```text
lowercase
remove punctuation
remove English articles a/an/the
fix whitespace
```

`Contain Accuracy`：

```text
s1 = normalize(pred_answer)
s2 = normalize(gold_answer)
score = 1 if s2 in s1 else 0
final score = average over samples
```

`LLM Accuracy / GPT-Acc.`：

```text
Prompt asks an evaluator model to compare generated answer and gold answer.
If response.strip().lower() == "correct": score = 1
else: score = 0
final score = average over samples
```

技术说明里的 retrieval-quality 指标：

```text
Context Relevance:
  semantic relevance between question and retrieved passages.

Evidence Recall:
  whether retrieved contents include all evidence required to answer the question.
```

当前限制：这两个 retrieval 指标在技术说明中有定义，但当前 `src/evaluate.py` 没有完整实现；如果要复现，需要额外接入 GraphRAG-Bench/RAGAS 类 evaluator，或转成我们自己的统一检索质量指标。

### 2.3 AGRAG

代码证据：

```text
baselines/AGRAG/GraphRAG_bench/generation_eval_vllm.py
baselines/AGRAG/baselines/run.py
```

技术说明指标：

```text
ACC.:
  text classification: exact match between predicted label and gold label
  other generation tasks: LLM score in {0, 0.5, 1}

REC.:
  TP / (TP + FN)
  used as macro recall for fixed-label classification tasks in the paper

ROUGE-L:
  longest-common-subsequence based overlap between generated answer and gold answer

Coverage:
  proportion of necessary reference evidences covered by the response

Faithfulness:
  proportion of answer claims supported by retrieved context
```

GraphRAG-Bench wrapper 按题型选择：

```text
Fact Retrieval:
  rouge_score, answer_correctness

Complex Reasoning:
  rouge_score, answer_correctness

Contextual Summarize:
  answer_correctness, coverage_score

Creative Generation:
  answer_correctness, coverage_score, faithfulness
```

当前限制：`rouge_score/answer_correctness/coverage_score/faithfulness` 从外部 `metrics` 模块导入，当前下载仓库没有完整实现。因此技术说明指标的定义能确认，但 QA wrapper 的逐项打分细节不能仅靠当前代码完全复现。

旧分类脚本 `baselines/run.py` 使用 sklearn：

```text
accuracy_score
precision_score(..., average="weighted", zero_division=0)
recall_score(..., average="weighted", zero_division=0)
f1_score(..., average="weighted", zero_division=0)
```

该分类路径不适合作为我们 QA 主实验指标。

### 2.4 HiPRAG

代码证据：

```text
baselines/HiPRAG/reward.py
baselines/HiPRAG/analysis.py
```

`normalize_string`：

```text
lowercase
remove English articles a/an/the
replace punctuation with spaces
fix whitespace
```

`CEM / cover_exact_match`：

```text
norm_pred = normalize(pred)
score = True if any(normalize(gold) in norm_pred for gold in gold_answers)
else False
final score = average boolean score
```

格式正确性：

```text
The output must contain:
  <think>
    one or more <step>...</step>
  </think>
  <answer>...</answer>

Each step must be one of:
  <reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion>
  <reasoning>...</reasoning><conclusion>...</conclusion>
```

Search-R1 reward：

```text
answer_correct and format_correct: 1.0
answer_correct and not format_correct: 1.0 - lambda_f
not answer_correct and format_correct: lambda_f
not answer_correct and not format_correct: 0.0
```

`OSR / USR`：

```text
OSR = oversearch / (search + oversearch)
USR = undersearch / (non-search + undersearch)
```

其中 over-search / under-search 依赖 verifier prompt 和 LLM judge，不是纯规则指标。

### 2.5 GraphRAG-R1

代码证据：

```text
baselines/GraphRAG-R1/eval/calc_rule.py
baselines/GraphRAG-R1/eval/eval_online.py
```

`normalize_answer`：

```text
replace underscore with spaces
lowercase
remove punctuation plus curly quotes/backticks
remove English articles a/an/the
fix whitespace
map True -> yes, False -> no
```

规则指标：

```text
EM:
  normalize(pred) == normalize(gold)

cover_em_1:
  every normalized gold token appears somewhere in normalized pred tokens

cover_em_2:
  normalized gold token sequence appears contiguously in normalized pred tokens
  OR normalized gold string is substring of normalized pred string

Precision:
  token_overlap_count / len(pred_tokens)

Recall:
  token_overlap_count / len(gold_tokens)

F1:
  2 * Precision * Recall / (Precision + Recall)

yes/no/noanswer special case:
  if one side is yes/no/noanswer and normalized strings differ, P/R/F1 = 0
```

多个 gold answers：

```text
For each metric, compute against every gold answer and take the maximum score.
```

SBERT：

```text
model = shibing624/text2vec-base-chinese
score = cosine(embedding(normalize(pred)), embedding(normalize(gold)))
```

LLM judge accuracy / ACCL：

```text
Qwen judge prompt compares Question, Golden Answer, Predicted Answer.
Return True if prediction fully aligns with the meaning and key information of gold.
Accuracy = number of True / total.
```

效率指标：

```text
average retrieve_num
variance / median of retrieve_num
average response time per retrieval round and per sample
average generation time per round and per sample
average retrieve tokens per round and per sample
average reasoning tokens per round and per sample
```

## 3. Prompt 清单

### 3.1 ClueRAG

代码证据：

```text
baselines/ClueRAG/utils/prompt.py
baselines/ClueRAG/generation/generation.py
baselines/ClueRAG/retrieval/retrieval.py
```

| Prompt | 用途 | 输入 | 输出要求 |
| --- | --- | --- | --- |
| `EXTRACTION_PROMPT` | 离线把 passage 切成自包含 knowledge units | `{passage}` | JSON object: `{"knowledge_units": [...]}` |
| `NER_PROMPT` | 离线 passage NER；在线 question NER | `{passage}`，question NER 时 passage 实际是 question text | JSON object: `{"named_entities": [...]}` |
| `GENERATION_PROMPT` | 根据检索上下文回答问题 | `{context}`, `{question}` | 代码使用 `GENERATION_SCHEMA`，要求 JSON object: `{"thought": "...", "answer": "..."}` |
| `ACCURACY_PROMPT` | LLM judge 评测 generated answer | `{question}`, `{gold_answer}`, `{generated_answer}` | JSON object: `{"reasoning": "...", "label": "CORRECT"|"WRONG"}` |

说明：

```text
GENERATION_PROMPT 文本本身要求 Thought/Answer 风格，
但 OpenAI 调用同时传了 strict JSON schema，
所以最终可解析输出应以 schema 为准。
```

### 3.2 LinearRAG

代码证据：

```text
baselines/LinearRAG/src/LinearRAG.py
baselines/LinearRAG/src/evaluate.py
```

| Prompt | 用途 | 输入 | 输出要求 |
| --- | --- | --- | --- |
| QA system prompt | 根据检索 passages 做 reading comprehension | retrieved passages + question | 先输出 `Thought:` 推理，最后用 `Answer:` 给简短答案 |
| QA user prompt | 拼接检索上下文与问题 | `passages`, `question` | 末尾固定 `Question: ...\n Thought:`，模型续写 |
| LLM judge prompt | 计算 `LLM Accuracy` | `Generated answer`, `Gold answer` | 只能返回 `correct` 或 `incorrect` |

QA prompt 核心要求：

```text
As an advanced reading comprehension assistant...
Conclude with "Answer: " to present a concise, definitive response.
```

评测 prompt 核心标准：

```text
correct if generated answer:
1. contains key information from gold answer
2. is factually accurate and consistent
3. has no contradiction
```

### 3.3 AGRAG

代码证据：

```text
baselines/AGRAG/GraphRAG_bench/run_graphrag.py
baselines/AGRAG/GraphRAG_bench/HippoRAG2/information_extraction/openie_openai.py
baselines/AGRAG/prompt/*.txt
```

| Prompt | 用途 | 输入 | 输出要求 |
| --- | --- | --- | --- |
| GraphRAG QA `SYSTEM_PROMPT` | 根据 Knowledge Base 直接回答 query | `{history}`, `{context_data}`, user query | plain text，直接简洁回答；未知则 `I don't know` |
| HippoRAG QA prompt | reading comprehension + chain-of-thought QA | retrieved docs + question | `Thought:` 推理，最后 `Answer:` 简洁答案 |
| NER prompt | OpenIE NER | passage | JSON object: `{"named_entities": [...]}` |
| relation/triple extraction prompt | OpenIE 关系抽取 | passage + `{"named_entities": [...]}` | JSON object: `{"triples": [[s,p,o], ...]}` |
| rerank/triplet filter prompt | LLM-based fact filtering | question + candidate facts | JSON object: `{"fact": [[s,p,o], ...]}`，最多选 4 条相关 facts |

说明：

```text
AGRAG 仓库实际包含 GraphRAG-Bench/HippoRAG2 组件。
其 NER/triple/filter prompt 与 HippoRAG 风格一致，
属于离线图构建或检索增强过程的一部分。
```

### 3.4 HiPRAG

代码证据：

```text
baselines/HiPRAG/prompt.py
baselines/HiPRAG/inference.py
baselines/HiPRAG/analysis.py
baselines/HiPRAG/scripts/data_process/nq_search.py
```

| Prompt | 用途 | 输入 | 输出要求 |
| --- | --- | --- | --- |
| `AGENT_PROMPT_BASE/V1/V2` | agentic search reasoning 主 prompt | user question | XML-like: `<think><step>...</step></think><answer>...</answer>` |
| `AGENT_PROMPT_V2_SHORT` | 默认推理 prompt，简化版 agent 格式控制 | user question | 同上；允许多步，每步可 search 或不 search |
| Search-R1 data prefix | 构造训练/测试 parquet prompt | dataset question | `<think>`, `<search>`, `<information>`, `<answer>` 格式 |
| `SEARCH_STEP_VERIFY_PROMPT_V1` | 判断两个 statement 是否语义等价，用于 over-search/under-search 分析 | Statement 1 + Statement 2 | `<answer>True</answer>` 或 `<answer>False</answer>` |
| `NON_SEARCH_STEP_VERIFY_PROMPT_V1` | 检查未搜索 step 是否事实正确、逻辑成立 | one step content | `<answer>True</answer>` 或 `<answer>False</answer>` |

默认 agent 输出格式：

```text
<think>
<step>
  <reasoning>...</reasoning>
  <search>...</search>
  <context>...</context>
  <conclusion>...</conclusion>
</step>
...
</think>
<answer>final answer</answer>
```

无搜索 step 可省略 `<search>/<context>`：

```text
<step>
  <reasoning>...</reasoning>
  <conclusion>...</conclusion>
</step>
```

### 3.5 GraphRAG-R1

代码证据：

```text
baselines/GraphRAG-R1/eval/qwen_base.py
baselines/GraphRAG-R1/eval/qwen_instruct.py
baselines/GraphRAG-R1/eval/eval_online.py
baselines/GraphRAG-R1/server/src/hipporag/prompts/templates/*.py
baselines/GraphRAG-R1/server/src/hipporag/prompts/filter_default_prompt.py
baselines/GraphRAG-R1/server/src/hipporag/prompts/linking.py
```

| Prompt | 用途 | 输入 | 输出要求 |
| --- | --- | --- | --- |
| `qwen_base.py` prompt `v0b` | base model agentic retrieval inference | question | `<think>...</think><answer>...</answer>`；检索用 `<\|begin_of_query\|>...<\|end_of_query\|>` |
| `qwen_instruct.py` prompt `v0c` | instruct model agentic retrieval inference | question | 同上，但要求答案 concise |
| `eval_online.py` judge prompt | LLM judge accuracy / ACCL | question + golden answer + predicted answer | 返回 `True` 或 `False` |
| `ner.py` | passage NER | passage | JSON object: `{"named_entities": [...]}` |
| `ner_query.py` | question/query NER | query | JSON object: `{"named_entities": [...]}` |
| `triple_extraction.py` | RDF triple extraction | passage + named entities | JSON object: `{"triples": [[s,p,o], ...]}` |
| `rag_qa_musique.py` | HippoRAG QA over retrieved docs | prompt_user | `Thought:` then `Answer:` |
| `ircot_hotpotqa.py` / `ircot_musique.py` | iterative CoT thought generation | prompt_user | 只生成当前 step thought；最终 step 用 `So the answer is:` |
| `filter_default_prompt.py` | LLM fact filtering/reranking | question + candidate facts | JSON object: `{"fact": [[s,p,o], ...]}` |
| `linking.py` instructions | embedding retrieval instruction | linking method | 返回不同检索模式的 natural-language instruction |

GraphRAG-R1 推理 prompt 的关键检索标签：

```text
<|begin_of_query|> query <|end_of_query|>
<|begin_of_documents|> documents <|end_of_documents|>
<answer> final answer </answer>
```

说明：

```text
GraphRAG-R1 同时包含 agentic retrieval prompt 和 server-side HippoRAG/OpenIE prompt。
如果只复现技术说明推理结果，主要关注 qwen_base/qwen_instruct 与 eval_online。
如果复现其图检索服务，还必须处理 NER、triple extraction、fact filter、linking instructions。
```

## 4. 对我们实验的影响

1. 不建议把不同 baseline 的官方指标直接混到主表。ClueRAG `Acc.`、HiPRAG `CEM`、GraphRAG-R1 `cover_em` 都是“宽松包含式”指标，但细节不同。
2. 主表继续使用 Signpost 已有 `EM / Precision / Recall / F1` 和成本指标，保证所有方法同一套评测逻辑。
3. 外部 baseline 官方指标可以作为附录：说明“按官方技术说明口径，该方法还报告了哪些指标”，但需要标明是否可由当前代码复现。
4. 需要实体或关系输入的 baseline，包括 ClueRAG、LinearRAG、AGRAG、GraphRAG-R1，正式接入时应优先复用 Signpost F6 统一实体/关系抽取产物；否则必须在 method card 中声明偏离。
5. Prompt 适配时应让 baseline 输出进入我们的统一 prediction schema，而不是修改 Signpost 统一评测逻辑去适配某个 baseline。
