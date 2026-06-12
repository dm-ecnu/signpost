# 外部 Baseline 官方评测口径审计

本文档只记录外部 baseline 官方代码中的数据格式、评测指标和实体抽取依赖。正式实验仍以 Signpost 统一输出 schema 和统一评测脚本为准；官方指标只用于理解方法原始技术说明/仓库的实验口径，不直接替代我们的 `basic_eval` 和 `query_metrics`。

## 1. 已下载仓库快照

| 方法 | 本地目录 | Remote | Commit |
| --- | --- | --- | --- |
| ClueRAG | `baselines/ClueRAG` | `https://github.com/Feesuu/ClueRAG.git` | `cb44eaa` |
| LinearRAG | `baselines/LinearRAG` | `https://github.com/DEEP-PolyU/LinearRAG` | `5da5262` |
| AGRAG | `baselines/AGRAG` | `https://github.com/Wyb0627/AGRAG` | `26fb4a9` |
| HiPRAG | `baselines/HiPRAG` | `https://github.com/qualidea1217/HiPRAG` | `3e64036` |
| GraphRAG-R1 | `baselines/GraphRAG-R1` | `https://github.com/ycygit/GraphRAG-R1` | `0c80b80` |

## 2. 统一实验原则

我们的技术说明实验不应混用各官方仓库的指标作为主表指标。主表建议统一使用 Signpost 已有指标：

```text
EM, Precision, Recall, F1, latency, retrieval latency, LLM calls, input/output/total tokens, tool calls
```

需要实体或关系输入的方法，原则上必须复用 Signpost F6 的 LLM 语义抽取产物。若官方代码无法直接复用，需要在该 baseline 的 method card 中标注偏离点，并解释官方抽取器是否会带来额外离线成本或不公平优势。

## 3. 总览

| 方法 | 官方原始数据格式 | 官方指标 | 是否有类似 target entity | 对我们实验的处理建议 |
| --- | --- | --- | --- | --- |
| ClueRAG | `data/<dataset>.json` + `data/<dataset>_corpus.json`；多跳 QA 中可含 `paragraphs/is_supporting` 或 `supporting_facts/context` | 生成质量：LLM judge `CORRECT/WRONG`；检索：`recall@2/5/10` | 有离线 KU/entity 抽取和在线 question NER，但不是我们的 target entity 字段 | 必须适配为复用 F6 实体/关系，或记录官方抽取器偏离；最终输出转为统一 prediction JSONL |
| LinearRAG | `dataset/<dataset>/questions.json` + `chunks.json`；问题含 `question/answer`，chunk 是字符串列表 | `LLM Accuracy`、`Contain Accuracy` | 用 spaCy 对 passage/question 做 NER；没有数据集自带 target entity | 若执行严格公平口径，需要替换/旁路 spaCy NER 为 F6 实体；否则标注为方法内部 NER 偏离 |
| AGRAG | GraphRAG-Bench 风格 JSON；结果含 `question_type/question/gold_answer/generated_answer/context` | 按题型使用 `rouge_score`、`answer_correctness`、`coverage_score`、`faithfulness`；另有旧分类脚本的 weighted accuracy/precision/recall/F1 | HippoRAG2/OpenIE 路径会抽取 `extracted_entities` 和 `extracted_triples` | QA 路径需要优先对接 F6 实体/关系；GraphRAG-Bench 指标可做附录，不做主表 |
| HiPRAG | Search-R1/VERL parquet；`reward_model.ground_truth.target` 是 golden answers | `cover_exact_match`、格式正确性、Search-R1 reward、oversearch/undersearch rate | 没有 target entity；`target` 表示答案，不是实体 | 作为 agentic RAG 对比时重点记录 LLM calls/tokens/retrieval rounds；QA 结果转统一 schema |
| GraphRAG-R1 | `datasets/<dataset>/Question.json` 和 `Corpus.json`；输出含 `pred_ans/answer/retrieve_num/...` | EM、cover EM、F1/P/R、SBERT similarity、ROUGE、LLM judge accuracy、检索次数/时间/tokens | server 内部 HippoRAG/OpenIE 使用 `extracted_entities/extracted_triples`；问题文件没有 target entity | 若使用其图索引路径，应复用 F6 语义抽取；规则 F1 与我们接近但仍以 Signpost 评测为准 |

## 4. ClueRAG

### 4.1 官方数据格式

主流程读取：

```text
data/<dataset>.json
data/<dataset>_corpus.json
```

`dataset/dataclass.py` 中，问题文件被转成：

```json
{
  "question": "...",
  "answer": "...",
  "ground_truth_chunks": ["..."]
}
```

`ground_truth_chunks` 的来源：

- MuSiQue 风格：`paragraphs` 中 `is_supporting=true` 的段落。
- HotpotQA/2Wiki 风格：`supporting_facts` 指向的 title，再回到 `context` 拼接全文并 hash。

如果我们的数据没有 supporting paragraph / supporting facts，官方 `recall@k` 会缺少 gold chunk 依据，不能直接作为可靠检索指标。

### 4.2 官方评测指标

检索指标在 `retrieval/retrieval.py`：

```text
recall@k = top-k retrieved chunk ids 命中 ground_truth_chunks 的数量 / ground_truth_chunks 数量
k in {2, 5, 10}
```

所有 query 的 `recall@k` 再做平均。

生成质量方面，`utils/prompt.py` 中 `ACCURACY_PROMPT` 使用 LLM judge，把 generated answer 标成 `CORRECT` 或 `WRONG`。但是当前仓库 `main.py` 引用了 `utils.utils.calculate_metric_scores`，而本地 `utils/utils.py` 没有这个函数，说明官方仓库当前评测入口不完整或代码未同步。

### 4.3 实体抽取

ClueRAG 官方索引阶段会抽取：

- `knowledge_units`
- `named_entities`

在线检索阶段还会对 question 做 LLM NER，并把 query entities link 到图中的 entity table。它没有使用我们数据中的 target entity 字段。按当前实验公平口径，ClueRAG 需要适配为复用 F6 的 LLM 实体/关系结果，否则应明确记录为偏离。

## 5. LinearRAG

### 5.1 官方数据格式

`run.py` 读取：

```text
dataset/<dataset_name>/questions.json
dataset/<dataset_name>/chunks.json
```

`chunks.json` 是 chunk 字符串列表，代码会转成：

```text
<idx>:<chunk_text>
```

`questions.json` 中每条问题至少需要：

```json
{
  "question": "...",
  "answer": "..."
}
```

生成后会写入 `pred_answer` 和 `gold_answer`，再进入 `src/evaluate.py`。

### 5.2 官方评测指标

`src/evaluate.py` 中有两个指标：

```text
LLM Accuracy:
  LLM judge 比较 pred_answer 与 gold_answer。
  若模型只返回 "correct"，该样本得 1，否则得 0。
  总分为样本平均值。

Contain Accuracy:
  normalize(pred_answer) 和 normalize(gold_answer) 后，
  若 gold 是 pred 的子串，则得 1，否则得 0。
  总分为样本平均值。
```

`normalize_answer` 会 lowercase、去标点、去英文冠词 `a/an/the`、合并空白。

### 5.3 实体抽取

`src/ner.py` 使用 spaCy：

- passage/chunk 建图时抽取实体。
- question 检索时抽取 query entities。
- 排除 `ORDINAL` 和 `CARDINAL`。

这不是我们 F6 的 target/entity 产物。如果严格执行“需要实体抽取的方法统一使用我们 LLM 抽取实体”，LinearRAG 需要适配 NER 输入；如果短期不改，应在实验限制中标明它仍使用官方 spaCy NER。

## 6. AGRAG

### 6.1 官方数据格式

AGRAG 仓库里有两条不同实验线：

1. `baselines/run.py`：旧的文本分类/持续学习脚本，数据是 `dataset/*/split/n*/train.txt|test.txt`，标签在行首。
2. `GraphRAG_bench/*`：更接近我们 QA/RAG 实验的 GraphRAG-Bench 路径。

GraphRAG-Bench 评测脚本 `generation_eval_vllm.py` 读取 JSON 列表，每个 item 需要：

```json
{
  "question_type": "Fact Retrieval",
  "question": "...",
  "gold_answer": "...",
  "generated_answer": "...",
  "context": "..."
}
```

`context` 也可能是 dict，此时代码读取 `context["compressed_prompt"]`。

### 6.2 官方评测指标

`generation_eval_vllm.py` 按 `question_type` 选择指标：

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

这些函数从 `metrics` 模块导入，仓库当前没有随附 `metrics.py`，应来自 GraphRAG-Bench/RAGAS 环境依赖。脚本对每个 metric 取非 NaN 样本均值。

旧分类路径 `baselines/run.py` 使用 sklearn：

```text
accuracy_score
precision_score(..., average="weighted", zero_division=0)
recall_score(..., average="weighted", zero_division=0)
f1_score(..., average="weighted", zero_division=0)
```

该分类路径不适合作为我们 QA 主实验指标。

### 6.3 实体抽取

GraphRAG-Bench/HippoRAG2 路径使用 OpenIE：

- NER 输出 `extracted_entities`
- triple extraction 输出 `extracted_triples`
- graph 中使用 entity/fact/passage 相关 embedding store

`openie_openai.py` 还包含 TF-IDF NER 路径，参数包括 `ner_threshold` 和 `max_ngram_length`。因此 AGRAG 属于需要实体/关系抽取的方法，正式接入时应优先复用 F6 产物，避免每个方法各自抽取实体导致成本和质量不可比。

## 7. HiPRAG

### 7.1 官方数据格式

`scripts/data_process/nq_search.py` 从 FlashRAG NQ 构造 parquet。每条样本包含：

```json
{
  "data_source": "nq",
  "prompt": [{"role": "user", "content": "..."}],
  "ability": "fact-reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "target": ["gold answer 1", "gold answer 2"]
    }
  },
  "extra_info": {
    "split": "train/test",
    "index": 0
  }
}
```

这里的 `target` 是 golden answers，不是 target entities。

### 7.2 官方评测指标

`reward.py` 中核心指标：

```text
cover_exact_match(pred, gold):
  normalize(pred) 后，只要任一 normalize(gold answer) 是 pred 的子串，即为 True。

format_correct:
  检查输出是否满足 <think>...<step>...</step>...</think><answer>...</answer> 格式。

search_r1_format:
  answer_correct 且 format_correct: 1.0
  answer_correct 但 format 不正确: 1.0 - lambda_f
  answer 不正确但 format 正确: lambda_f
  answer 不正确且 format 不正确: 0.0
```

`analysis.py` 还统计：

```text
accuracy = cover_exact_match(<answer> 中最后一个答案, golden_answers) 的平均值
over_search_rate = oversearch / (search + oversearch)
under_search_rate = undersearch / (non-search + undersearch)
```

### 7.3 实体抽取

HiPRAG 是 reasoning-search interleaving / agentic RAG 方法，官方数据和 reward 没有类似 target entity 的字段。对我们来说，重点不是实体复用，而是在线阶段要准确记录：

```text
retrieval rounds, LLM calls, input/output/total tokens, retrieval latency
```

## 8. GraphRAG-R1

### 8.1 官方数据格式

官方数据目录：

```text
datasets/<dataset>/Question.json
datasets/<dataset>/Corpus.json
```

`Question.json` 是 JSONL，每条样例类似：

```json
{"question": "What is George Rankin's occupation?", "answer": "politician"}
```

`Corpus.json` 是包含 docs 的 JSON，其中每个 chunk 可能已经带有：

```json
{
  "idx": "chunk-...",
  "passage": "...",
  "extracted_entities": ["..."],
  "extracted_triples": [["s", "p", "o"]],
  "propositions": [{"text": "...", "entities": ["..."]}],
  "title": "..."
}
```

推理输出通常包含：

```text
generated_answer, pred_ans, answer, retrieve_num, response_times, generate_times, retrieve_tokens, reasoning_tokens
```

### 8.2 官方评测指标

`eval/calc_rule.py` 的规则指标：

```text
EM:
  normalize(pred_ans) == normalize(answer)

cover_em_1:
  gold 的每个 token 都出现在 pred token 列表中，不要求连续。

cover_em_2:
  gold token 序列在 pred token 中连续出现，或 gold string 是 pred string 子串。

F1 / Precision / Recall:
  normalize 后按 token Counter 求重叠。
  precision = overlap / len(pred_tokens)
  recall = overlap / len(gold_tokens)
  f1 = 2 * precision * recall / (precision + recall)
  对 yes/no/noanswer 有特殊处理：若一方是该类答案且不相等，直接 0。

SBERT similarity:
  使用 `shibing624/text2vec-base-chinese` 编码 pred/gold，计算 cosine similarity。

ROUGE:
  normalize 后计算 rouge-1/rouge-2/rouge-l 的 F 值。
```

若一个问题有多个 gold answers，官方代码对每个 metric 取最高分。默认 `eval/config.json` 开启 `use_em`、`use_f1`、`use_sbert_sim`，未开启 `cover_em_1/2` 和 ROUGE。

`eval/eval_online.py` 使用 Qwen judge，提示模型判断 predicted answer 是否与 golden answer 语义一致，返回 `True/False`，最后计算 accuracy。

此外，`calc_rule.py` 会统计在线行为：

```text
平均检索次数、检索次数方差/中位数
每轮/每条响应时间
每轮/每条生成时间
每轮/每条检索 tokens
每轮/每条推理 tokens
```

### 8.3 实体抽取

GraphRAG-R1 的 server 目录内包含 HippoRAG/OpenIE 路径，会使用 `extracted_entities` 和 `extracted_triples` 建图。官方 `Corpus.json` 样例中这些字段已经存在，因此它不是直接读取我们的问题 target entity，而是依赖 corpus 侧 OpenIE 语义标注。

若要纳入我们的正式公平实验，图索引输入应优先从 F6 产物转换，而不是重新跑其官方 OpenIE；否则需要明确记录它使用官方预抽取实体/三元组。

## 9. 对后续接入的结论

1. 主表指标统一用 Signpost 的 `basic_eval` / `query_metrics`，不要直接混用官方 LLM judge、RAGAS 或 cover EM。
2. ClueRAG、AGRAG、GraphRAG-R1 明确依赖实体/三元组；LinearRAG 依赖 spaCy NER；这些都需要检查是否能接入 F6 共享实体。
3. HiPRAG 没有 target entity 概念，更适合作为 agentic RAG 在线成本对照，重点比较 LLM calls、tokens、检索轮数和延迟。
4. ClueRAG 当前仓库评测入口不完整，正式接入前必须先做 adapter smoke，不能直接相信 `main.py` 可跑通。
5. AGRAG 的 GraphRAG-Bench 评测依赖外部 `metrics` 模块，官方 QA 指标复现实验需要额外环境；我们主实验可先只做统一输出和统一评测。

## 10. 技术说明补充审计

上一版主要依据官方代码。进一步下载并检索技术说明后，需要补充：部分技术说明报告的指标并没有在当前开源代码中完整暴露，尤其是 ClueRAG 的 `F1 / Acc.` 和 LinearRAG 的 retrieval-quality 指标。因此后续写技术说明或 method card 时应区分“技术说明报告口径”和“代码可调用口径”。

### 10.1 已下载技术说明

| 方法 | 技术说明页面 | 本地 PDF | 本地文本 |
| --- | --- | --- | --- |
| ClueRAG | `https://arxiv.org/abs/2507.08445` | `docs/baselines/papers/cluerag_2507.08445.pdf` | `docs/baselines/papers/cluerag_2507.08445.txt` |
| LinearRAG | `https://arxiv.org/abs/2510.10114` | `docs/baselines/papers/linearrag_2510.10114.pdf` | `docs/baselines/papers/linearrag_2510.10114.txt` |
| AGRAG | `https://arxiv.org/abs/2511.05549` | `docs/baselines/papers/agrag_2511.05549.pdf` | `docs/baselines/papers/agrag_2511.05549.txt` |
| HiPRAG | `https://arxiv.org/abs/2510.07794` | `docs/baselines/papers/hiprag_2510.07794.pdf` | `docs/baselines/papers/hiprag_2510.07794.txt` |
| GraphRAG-R1 | `https://arxiv.org/abs/2507.23581` | `docs/baselines/papers/graphrag_r1_2507.23581.pdf` | `docs/baselines/papers/graphrag_r1_2507.23581.txt` |

### 10.2 ClueRAG 技术说明指标

ClueRAG 技术说明 Section 4.1 明确报告两类指标：

```text
QA performance:
  Accuracy / Acc.
  F1 score

Cost efficiency:
  total token expenditure for offline indexing
  average token consumption per online query
```

`Accuracy` 不是 strict exact match，而是判断 golden answer 是否包含在 generated answer 中。`F1` 是 generated answer 与 golden answer 的 token-level overlap F1，用于平衡 answer completeness 和 correctness。

技术说明主实验 Table 2 报告 `F1 / Acc.`，Table 3 报告 chunk-selection 策略下的 `F1 / Acc.`，Table 4 报告消融实验 `F1 / Acc.`。Appendix A.1 说明 token cost 统计 prompt tokens + completion tokens；如果方法在线需要 LLM query preprocessing，例如 keyword/entity extraction，这部分 token cost 也计入 online retrieval。

这意味着：ClueRAG 技术说明确实有 F1，不应只按当前代码里能看到的 LLM judge 或 retrieval recall 理解。当前开源代码缺 `calculate_metric_scores`，所以需要我们自己把输出转成 Signpost 统一 schema，再用统一评测算 EM/P/R/F1。

### 10.3 LinearRAG 技术说明指标

LinearRAG 技术说明 Section 4.1 使用四个指标：

```text
End-to-end QA:
  Contain-Match Accuracy / Contain-Acc.
  GPT-Evaluation Accuracy / GPT-Acc.

Retrieval quality:
  Context Relevance
  Evidence Recall
```

定义：

```text
Contain-Acc.:
  检查正确答案是否出现在 generated response 中。

GPT-Acc.:
  用 LLM judge 判断 predicted answer 是否匹配 ground truth。

Context Relevance:
  衡量 question 与 retrieved passages 的语义相关性。

Evidence Recall:
  衡量 retrieved contents 是否包含回答问题所需的全部证据。
```

技术说明 Table 1 报告 `Contain-Acc. / GPT-Acc.`。Medical dataset 因 gold answer 是长描述，只报告 `GPT-Acc.`。Table 2 中的 `Accuracy` 是 `Contain-Acc.` 和 `GPT-Acc.` 的平均值，同时报告 indexing/retrieval time、prompt tokens、completion tokens。Appendix Table 4 报告 GraphRAG-Bench Medical 上不同问题类型的 `Recall / Relevance`。

当前代码只直接实现 `LLM Accuracy` 和 `Contain Accuracy`。如果要复现技术说明中的 retrieval-quality 表，需要额外接入 GraphRAG-Bench/RAGAS 风格评测或按统一口径实现。

### 10.4 AGRAG 技术说明指标

AGRAG 技术说明 Section IV 使用 5 个指标：

```text
ACC. / Accuracy:
  文本分类任务中为 predicted label 与 gold label 的 exact-match 比例；
  其他任务中由 LLM 打 0 / 0.5 / 1，分别表示 inaccurate / partially accurate / exactly accurate。

REC. / Recall:
  true positives / (true positives + false negatives)；
  用于固定 label set 的 text-classification task，使用 macro recall。

ROG. / ROUGE-L:
  基于 generated answer G 和 gold answer Y 的 longest common subsequence；
  用于 fact retrieval 和 complex reasoning。

COV. / Coverage:
  判断 response 是否覆盖 reference evidences；
  用于 contextual summarization 和 creative generation。

FS. / Faithfulness:
  answer claims 中被 retrieved context 支持的比例；
  用于 creative generation。
```

技术说明还报告效率：

```text
Time Cost:
  indexing minutes
  querying minutes

Token Cost:
  input tokens per query
  output tokens per query
```

代码里的 `generation_eval_vllm.py` 是 GraphRAG-Bench wrapper，按 question type 调 `rouge_score / answer_correctness / coverage_score / faithfulness`。这些函数来自外部 `metrics` 模块，当前仓库没有随附完整实现。

### 10.5 HiPRAG 技术说明指标

HiPRAG 技术说明 Section 4.1 使用：

```text
CEM / Cover Exact Match:
  检查 ground-truth answer string 是否出现在 generated answer 中。
  技术说明选择 CEM 而不是 strict EM，因为 LLM 往往输出较长解释。

OSR / Over-search Rate:
  over-search steps / all identifiable search steps。

USR / Under-search Rate:
  under-search steps / all identifiable non-search steps。
```

技术说明还报告：

```text
Avg. #Searches:
  每个问题的平均 search steps 数量。

Judge Accuracy:
  over-search / under-search LLM judge 与人工标注的一致率。

Runtime breakdown:
  每个 RL training step 中 rollout、re-generation、external verifier API calls 等部分的 wall-clock 占比。
```

Table 1 是七个 QA benchmark 上的 CEM；Table 2 汇总 `CEM / OSR / USR`；Table 12 报告 `Avg. #Searches`。

### 10.6 GraphRAG-R1 技术说明指标

GraphRAG-R1 技术说明 Section 4.1.2 使用三个主指标：

```text
F1 Score:
  衡量 generated answer 与 ground-truth reference 的 lexical overlap。

SBERT Similarity / SBERT:
  用 SBERT sentence embeddings 计算 generated answer 与 reference 的 cosine similarity。

LLM-as-Judge Accuracy / ACCL:
  用 LLM judge 判断回答是否正确，技术说明使用 Qwen3-Turbo 作为 evaluator。
```

技术说明还报告：

```text
#Calls:
  retrieval calls 数量，用于 PRA/CAF 消融。

ACCF:
  完全符合格式要求的输出比例，用于 phase-dependent training 消融。

#Token:
  retrieval/reasoning 过程中的 token consumption，用于 hybrid graph-text retrieval 分析。
```

CAF reward 直接使用 F1 并惩罚 retrieval count：

```text
R_CAF = F1 * a * exp(-b * N)
N = total number of retrieval operations
```

代码里的 `calc_rule.py` 还额外支持 EM、cover EM、ROUGE 等辅助指标，但技术说明主表不是这些指标。

### 10.7 更新后的接入判断

1. ClueRAG 技术说明确实有 `F1 / Acc.`；当前代码评测入口不完整，所以后续不能只看代码，需要用 Signpost 统一评测复现 F1。
2. LinearRAG 不只是 `LLM Accuracy / Contain Accuracy`，技术说明还有 `Context Relevance / Evidence Recall`。这两个可作为补充检索质量指标，但主表仍建议统一用 Signpost EM/P/R/F1。
3. AGRAG 的指标最接近 GraphRAG-Bench/RAGAS 体系，技术说明里的 `ACC / REC / ROUGE-L / Coverage / Faithfulness` 不能直接和我们的 token-level F1 混为一谈。
4. HiPRAG 和 GraphRAG-R1 都强调 retrieval behavior / tool-call efficiency，应重点记录 calls、tokens、检索轮数和延迟，用来和我们的 online cost 对齐。
5. 所有 baseline 最终仍必须输出 `outputs/<dataset>/predictions/<method>.jsonl`，进入 Signpost 的统一 `basic_eval` 和 `query_metrics`；官方技术说明指标只作为附录或 method card 参考。
