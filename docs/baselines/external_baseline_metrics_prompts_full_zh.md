# 外部 Baseline 指标中文解释与原始 Prompt 完整整理

本文档是 `external_baseline_metrics_prompts_zh.md` 的完整口径版。这里不再只写摘要，而是尽量逐项说明：

1. 每个 baseline 的每个指标到底怎么算。
2. 哪些指标能从代码确认，哪些只能从技术说明确认，哪些当前无法确认。
3. 能在当前本地仓库中定位到的原始 prompt，按源码原文整理。

注意：正式实验仍以 Signpost 统一 `basic_eval` / `query_metrics` 为主。这里整理的是外部 baseline 的官方技术说明/代码口径，主要用于写 method card、附录和公平性说明。

## 1. ClueRAG

### 1.1 指标中文解释

#### Accuracy / Acc.

来源：技术说明。

是否能从当前代码确认：不能完整确认。技术说明定义明确，但当前下载的 `baselines/ClueRAG/main.py` 引用了 `calculate_metric_scores`，而 `baselines/ClueRAG/utils/utils.py` 中没有该函数。

中文解释：

```text
对每个问题，比较模型生成答案 generated answer 和标准答案 gold answer。
如果标准答案的核心内容被包含在生成答案中，则该样本记为 1。
否则记为 0。
最后对所有样本取平均。
```

可以理解为宽松的“答案包含式准确率”，不是严格 EM。

无法确认的部分：

```text
当前代码缺少官方 calculate_metric_scores，因此无法确认官方实现中：
1. 是否做了大小写归一化。
2. 是否去掉标点和冠词。
3. 多个 gold answer 时是否取最大值。
4. 是否用 LLM judge 辅助判断。
```

#### F1

来源：技术说明。

是否能从当前代码确认：不能完整确认。技术说明说明是 generated answer 与 gold answer 的 token-level overlap F1，但当前代码缺少对应实现。

中文解释：

```text
先把生成答案和标准答案切成 token。
计算两者共同 token 的数量 overlap。

Precision = overlap / 生成答案 token 数
Recall = overlap / 标准答案 token 数
F1 = 2 * Precision * Recall / (Precision + Recall)
```

无法确认的部分：

```text
当前代码无法确认 tokenization 和 normalization 细节。
例如是否 lowercase、是否去标点、是否去 a/an/the、是否处理 yes/no。
```

#### Recall@2 / Recall@5 / Recall@10

来源：代码 `baselines/ClueRAG/retrieval/retrieval.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
每个问题有一组 gold supporting chunks，记为 ground_truth_chunks。
检索器返回重排序后的 chunk id 列表 final_chunks。

Recall@k = final_chunks 前 k 个 chunk 中命中 ground_truth_chunks 的数量 / ground_truth_chunks 的数量

k 取 2、5、10。
所有问题的 Recall@k 再取平均。
```

特殊情况：

```text
如果某个问题没有 ground_truth_chunks，则该问题 recall@k = 0。
```

#### Offline / Online Token Cost

来源：技术说明 + 部分代码 metadata。

是否能从当前代码确认：只能部分确认。

中文解释：

```text
技术说明要求统计 LLM prompt tokens + completion tokens。
offline token cost 指离线索引/知识单元抽取/实体抽取等阶段产生的 token。
online token cost 指每个 query 检索和生成阶段产生的 token。
```

无法确认的部分：

```text
当前代码能记录部分 LLM metadata，但由于官方评测入口不完整，
无法确认技术说明表格中的 token cost 是否完全由当前代码复现。
```

### 1.2 原始 Prompt

来源：`baselines/ClueRAG/utils/prompt.py`。

#### EXTRACTION_PROMPT

用途：离线把 passage 抽成自包含 knowledge units。

```text
Extract independent facts from the "Content".
Your goal is to make each sentence self-contained by resolving pronouns, WITHOUT adding new information or changing the original meaning.

Guidelines:
1. **Strict Faithfulness**: Do not hallucinate, embellish, or add details that are not explicitly present in the text. Use the original wording as much as possible.
2. **Coreference Resolution Only**: The ONLY major modification allowed is replacing pronouns (he, she, it, they, his, etc.) and relative references (the company, the team) with their specific, full entity names.
3. **Atomic Units**: If a sentence contains multiple distinct facts, you may split them, but do not merge separate thoughts into complex, flowery sentences.
4. **Contextual Independence**: Ensure each unit makes sense on its own (Who, When, Where, What) without needing the surrounding context.

Example:
Input: Jesús Aranguren. His 13-year professional career was solely associated with Athletic Bilbao, with which he played in nearly 400 official games.
Output: {{ "knowledge_units": [
    "Jesús Aranguren's 13-year professional career was solely associated with Athletic Bilbao.",
    "Jesús Aranguren played in nearly 400 official games for Athletic Bilbao."
]}}

Input: {passage}
Output:
```

#### NER_PROMPT

用途：离线 passage NER；在线 question NER。

```text
Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
Example:
Input: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
Output: {{"named_entities":["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}}

Input: {passage}
Output:
```

#### ACCURACY_PROMPT

用途：LLM judge，判断 generated answer 是否正确。

```text
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
```

#### GENERATION_PROMPT

用途：根据检索上下文回答问题。源码同时配置了 strict JSON schema，要求输出 `thought` 和 `answer` 两个字段。

```text
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.

Wikipedia Title: The Last Horse
The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.
Wikipedia Title: Southampton
The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.
Wikipedia Title: Stanton Township, Champaign County, Illinois
Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.
Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.
Wikipedia Title: Finding Nemo
Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million

Question: When was Neville A. Stanton's employer founded?
Thought: The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862.
Answer: 1862.

Real Input:
{context}

Question: {question}
```

## 2. LinearRAG

### 2.1 指标中文解释

#### Contain Accuracy / Contain-Acc.

来源：代码 `baselines/LinearRAG/src/evaluate.py`，技术说明也报告该指标。

是否能从当前代码确认：可以确认。

中文解释：

```text
先分别归一化 pred_answer 和 gold_answer：
1. 全部转小写。
2. 去掉标点。
3. 去掉英文冠词 a/an/the。
4. 合并多余空白。

如果归一化后的 gold_answer 是归一化后的 pred_answer 的子串，则该样本得 1。
否则得 0。
最后对所有样本取平均。
```

这是一种宽松包含式准确率。生成答案比标准答案长，只要包含标准答案，就可能得 1。

#### LLM Accuracy / GPT-Acc.

来源：代码 `baselines/LinearRAG/src/evaluate.py`，技术说明也报告该指标。

是否能从当前代码确认：可以确认。

中文解释：

```text
对每个样本，把 generated answer 和 gold answer 交给一个 evaluator LLM。
prompt 要求 evaluator 只返回 correct 或 incorrect。

如果返回文本去空白、转小写后严格等于 correct，则该样本得 1。
否则得 0。
最后对所有样本取平均。
```

#### Context Relevance

来源：技术说明。

是否能从当前代码确认：无法确认。

中文解释：

```text
技术说明说该指标衡量 question 与 retrieved passages 的语义相关性。
也就是检索出来的上下文是否和问题相关。
```

无法确认的部分：

```text
当前 LinearRAG 代码没有实现该指标。
无法确认是否使用 RAGAS、LLM judge、embedding similarity，或人工标注。
无法确认分数范围、平均方式、prompt 和阈值。
```

#### Evidence Recall

来源：技术说明。

是否能从当前代码确认：无法确认。

中文解释：

```text
技术说明说该指标衡量 retrieved contents 是否包含回答问题所需的全部证据。
直观上，如果答案需要 3 条证据，而检索结果覆盖了其中 2 条，则 recall 应反映这种覆盖程度。
```

无法确认的部分：

```text
当前 LinearRAG 代码没有实现该指标。
无法确认 gold evidence 如何定义、如何匹配 evidence、是否由 LLM judge 判定。
```

### 2.2 原始 Prompt

来源：`baselines/LinearRAG/src/LinearRAG.py` 与 `baselines/LinearRAG/src/evaluate.py`。

#### QA system prompt

用途：根据检索 passage 回答问题。

```text
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
```

#### QA user prompt 模板

用途：把检索 passage 和 question 拼起来。源码中 `sorted_passage` 会逐条追加到前面。

```text
{passage_1}
{passage_2}
...
{passage_n}
Question: {question}
 Thought: 
```

#### LLM judge prompt

用途：计算 LLM Accuracy。

System：

```text
You are an expert evaluator. 
```

User：

```text
Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
```

## 3. AGRAG

### 3.1 指标中文解释

#### ACC. / Accuracy

来源：技术说明；旧分类脚本有 sklearn 实现；GraphRAG-Bench QA wrapper 依赖外部 `metrics`。

是否能从当前代码确认：只能部分确认。

中文解释：

```text
在文本分类任务中：
  Accuracy = 预测标签等于标准标签的样本数 / 总样本数。

在生成式 QA / summarization / creative generation 等任务中：
  技术说明说由 LLM 打分，分数为 0、0.5 或 1。
  0 表示错误，0.5 表示部分正确，1 表示完全正确。
```

无法确认的部分：

```text
当前 GraphRAG-Bench wrapper 调用 answer_correctness，
但当前 AGRAG 仓库没有随附完整 metrics.py。
因此无法确认 LLM judge 的完整 prompt、模型、阈值和平均细节。
```

#### REC. / Recall

来源：技术说明；旧分类代码有 sklearn weighted recall。

是否能从当前代码确认：只能部分确认。

中文解释：

```text
技术说明定义：
Recall = TP / (TP + FN)

如果是多类别分类，技术说明说明使用 macro recall。
也就是先分别计算每个类别的 recall，再对类别取平均。
```

当前代码差异：

```text
旧分类脚本使用 sklearn recall_score(..., average="weighted")。
weighted recall 会按每个类别的样本数量加权平均，不是 macro recall。
所以技术说明口径和当前旧分类脚本不完全一致。
```

#### ROUGE-L / ROG.

来源：技术说明；GraphRAG-Bench wrapper 调用外部 `rouge_score`。

是否能从当前代码确认：只能确认概念，无法确认完整实现。

中文解释：

```text
ROUGE-L 基于 generated answer 和 gold answer 的最长公共子序列 LCS。
LCS 越长，说明生成答案和标准答案的顺序重合内容越多。
```

无法确认的部分：

```text
当前代码没有 rouge_score 实现。
无法确认使用 ROUGE-L precision、recall、F1 中哪一个作为最终值。
无法确认是否做 normalization。
```

#### Coverage / COV.

来源：技术说明；GraphRAG-Bench wrapper 调用外部 `coverage_score`。

是否能从当前代码确认：无法确认完整实现。

中文解释：

```text
Coverage 衡量生成回答是否覆盖参考证据中的必要信息。
如果参考证据里有多个关键点，回答覆盖得越完整，coverage 越高。
```

无法确认的部分：

```text
当前 AGRAG 仓库没有 coverage_score 的实现。
无法确认 evidence 如何切分、如何判定 covered、是否由 LLM judge 打分。
```

#### Faithfulness / FS.

来源：技术说明；GraphRAG-Bench wrapper 调用外部 `faithfulness`。

是否能从当前代码确认：无法确认完整实现。

中文解释：

```text
Faithfulness 衡量生成回答中的 claims 是否被检索上下文支持。
如果回答中的每个事实陈述都能在 context 中找到依据，faithfulness 高。
如果回答有幻觉或 context 不支持的陈述，faithfulness 低。
```

无法确认的部分：

```text
当前 AGRAG 仓库没有 faithfulness 的实现。
无法确认 claim 如何抽取、support 如何判断、是否使用 LLM judge。
```

#### 旧分类 Precision / Recall / F1

来源：代码 `baselines/AGRAG/baselines/run.py`。

是否能从当前代码确认：可以确认，但这是分类路径，不适合作为我们的 QA 主实验指标。

中文解释：

```text
accuracy_score:
  预测标签等于真实标签的样本比例。

precision_score(..., average="weighted"):
  先算每个类别 precision，再按各类别真实样本数加权平均。

recall_score(..., average="weighted"):
  先算每个类别 recall，再按各类别真实样本数加权平均。

f1_score(..., average="weighted"):
  先算每个类别 F1，再按各类别真实样本数加权平均。
```

### 3.2 原始 Prompt

当前下载的 AGRAG 仓库中，`GraphRAG_bench/HippoRAG2` 调用了 prompt manager，但没有随附完整 `prompts/templates` 源目录。本地能确认的原文来自：

```text
baselines/AGRAG/GraphRAG_bench/run_graphrag.py
baselines/AGRAG/prompt/qa_prompt.txt
baselines/AGRAG/prompt/relation_extraction_prompt.txt
baselines/AGRAG/prompt/rerank_triplet_filter_prompt.txt
```

#### GraphRAG QA SYSTEM_PROMPT

用途：根据 Knowledge Base 直接回答 query。

```text
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know". 
Respond with no more than 4096 tokens. 

---Conversation History---
{history}

---Knowledge Base---
{context_data}
```

#### HippoRAG Final QA Prompt

用途：retrieved passages 上的阅读理解 QA。

System：

```text
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
```

One-shot user：

```text
Wikipedia Title: The Last Horse
The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.

Wikipedia Title: Southampton
The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.

Wikipedia Title: Stanton Township, Champaign County, Illinois
Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.

Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.

Wikipedia Title: Finding Nemo
Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million


Question: When was Neville A. Stanton's employer founded?
Thought: 
```

One-shot assistant：

```text
The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. 
Answer: 1862.
```

New user 模板：

```text
${prompt_user}
```

`prompt_user` 构造方式：

```text
Wikipedia Title: {passage_1}

Wikipedia Title: {passage_2}

...
Question: {question}
Thought: 
```

#### Relation / Triple Extraction Prompt

用途：把 passage 和 named entities 转成 RDF triples。

System：

```text
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
```

User 模板：

~~~text
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
${passage}
```

${named_entity_json}
~~~

Assistant 输出格式：

```text
{"triples": [
    ["subject", "predicate", "object"]
]}
```

#### Rerank / Triplet Filter Prompt

用途：从候选 facts 中筛选最多 4 条与 query 最相关的 facts。

System：

```text
Your input fields are:
1. `question` (str): Query for retrieval
2. `fact_before_filter` (str): Candidate facts to be filtered

Your output fields are:
1. `fact_after_filter` (Fact): Filtered facts in JSON format

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

[[ ## fact_after_filter ## ]]
{fact_after_filter}        # note: the value you produce must be pareseable according to the following JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": "array", "items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.
```

User 模板：

```text
[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

Respond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`.
```

Assistant 输出模板：

```text
[[ ## fact_after_filter ## ]]
{fact_after_filter}

[[ ## completed ## ]]
```

NER prompt：

```text
无法从当前 AGRAG 官方代码目录确认完整原文。
原因：HippoRAG2 代码调用 prompt_template_manager.render(name='ner', ...)，
但当前下载的 AGRAG/HippoRAG2 目录没有随附 prompts/templates/ner.py。
```

## 4. HiPRAG

### 4.1 指标中文解释

#### CEM / Cover Exact Match

来源：代码 `baselines/HiPRAG/reward.py`，技术说明也报告。

是否能从当前代码确认：可以确认。

中文解释：

```text
先归一化 pred 和 gold：
1. 全部转小写。
2. 去掉英文冠词 a/an/the。
3. 把标点替换为空格。
4. 合并多余空白。

如果任意一个 gold answer 的归一化字符串，是 pred 的归一化字符串的子串，则该样本正确。
否则错误。
最后对所有样本取平均。
```

#### Format Correctness

来源：代码 `baselines/HiPRAG/reward.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
检查模型输出是否满足指定 XML-like 格式。
整体必须是：
<think>
  一个或多个 <step>...</step>
</think>
<answer>...</answer>

每个 step 必须是两种格式之一：
1. 带搜索：
   <reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion>
2. 不带搜索：
   <reasoning>...</reasoning><conclusion>...</conclusion>
```

#### Search-R1 Format Reward

来源：代码 `baselines/HiPRAG/reward.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
如果 answer 正确且格式正确：得 1.0。
如果 answer 正确但格式错误：得 1.0 - lambda_f。
如果 answer 错误但格式正确：得 lambda_f。
如果 answer 错误且格式错误：得 0.0。
```

#### OSR / Over-search Rate

来源：技术说明 + `baselines/HiPRAG/analysis.py`。

是否能从当前代码确认：可以确认公式，但具体判定依赖 LLM judge。

中文解释：

```text
OSR = oversearch / (search + oversearch)

search 表示合理的搜索 step 数。
oversearch 表示不需要搜索但模型进行了搜索的 step 数。
```

无法纯规则确认的部分：

```text
某个搜索 step 是否 over-search 由额外 verifier prompt + LLM judge 判断。
因此结果依赖 evaluator 模型。
```

#### USR / Under-search Rate

来源：技术说明 + `baselines/HiPRAG/analysis.py`。

是否能从当前代码确认：可以确认公式，但具体判定依赖 LLM judge。

中文解释：

```text
USR = undersearch / (non-search + undersearch)

non-search 表示合理的不搜索 step 数。
undersearch 表示本应搜索但模型没有搜索的 step 数。
```

#### Avg. #Searches

来源：技术说明。

是否能从当前代码确认：可以从输出格式统计。

中文解释：

```text
统计每个问题输出中 <search>...</search> 的次数。
对所有问题取平均。
```

### 4.2 原始 Prompt

来源：

```text
baselines/HiPRAG/prompt.py
baselines/HiPRAG/scripts/data_process/nq_search.py
```

#### AGENT_PROMPT_V2_SHORT

用途：HiPRAG 默认推理 prompt。

```text
Answer user questions by thinking step-by-step. Your entire reasoning process must be encapsulated within a single <think></think> block, which contains one or more <step></step> blocks. Each step must begin with your analysis in <reasoning>. If you identify a knowledge gap, you may use <search>query</search> to query a search engine; search results will then be provided in a <context> tag. Every step must end with a <conclusion> summarizing what you learned in that step. After your thinking process is complete, provide the final, conclusive answer inside an <answer> tag placed immediately after the closing </think> tag. You can use as many steps as you need. Ensure all XML tags are properly formed and nested.

**## Output Format Specification**

Your output must follow this overall structure. The `<think>` block contains all the steps, and the `<answer>` block follows it.

<think>
<step>
    ...
</step>
<step>
    ...
</step>
</think>
<answer>Your final, conclusive answer to the user's question.</answer>

**## Step Formats (to be used inside <think>)**

Format 1: Step with a Search

<step>
    <reasoning>Your detailed analysis of what you know and what you need to find out.</reasoning>
    <search>The precise search query you will use.</search>
    <context>[This will be provided by the system after your search]</context>
    <conclusion>Your conclusion or answer to the reasoning and search query at this step.</conclusion>
</step>

Format 2: Step without a Search (Internal Reasoning)

<step>
    <reasoning>Your detailed analysis of what you know and what you need to find out.</reasoning>
    <conclusion>Your conclusion or answer to the reasoning at this step.</conclusion>
</step>
```

#### SEARCH_STEP_VERIFY_PROMPT_V1

用途：判断两个 statement 是否语义等价，用于 over-search/under-search 分析。

```text
You are an expert in Natural Language Understanding and Semantic Analysis. Your goal is to determine if these two statements are semantically equivalent—that is, if they mean the same thing and convey the same core information. Provide your answers with a single boolean value "True" or "False" in the tag <answer></answer> (e.g. <answer>True</answer> or <answer>False</answer>).
```

#### NON_SEARCH_STEP_VERIFY_PROMPT_V1

用途：检查未搜索 step 的事实正确性和内部逻辑。

```text
You are an expert Fact-Checker and Logic Verifier. Your task is to evaluate a single, isolated reasoning step from an AI agent.

This step was generated without using a search tool. Your goal is to determine if the agent made a mistake by not searching, based only on the information within this single step and your own general knowledge.

Analyze the provided step by asking two questions:
1. Factual Accuracy: Is the statement in the <reasoning></reasoning> and <conclusion></conclusion> factually correct?
2. Internal Logic: Does the <conclusion></conclusion> logically follow from the <reasoning></reasoning> provided within this same step?

If both questions are answered correctly, provide your answers with a single boolean value "True" or "False" in the tag <answer></answer> (e.g. <answer>True</answer> or <answer>False</answer>).
```

#### Search-R1 data prefix

用途：构造 parquet 数据里的 user prompt。

```text
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

说明：

```text
AGENT_PROMPT_BASE / V1 / V2 也是源码中存在的完整长 prompt 变体。
但 inference.py 和 analysis.py 默认参数使用 AGENT_PROMPT_V2_SHORT。
因此本节优先记录实际默认使用的 V2_SHORT。
```

## 5. GraphRAG-R1

### 5.1 指标中文解释

#### EM / Exact Match

来源：代码 `baselines/GraphRAG-R1/eval/calc_rule.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
先对 pred_ans 和 answer 归一化：
1. True 映射成 yes，False 映射成 no。
2. 下划线替换为空格。
3. 全部转小写。
4. 去掉标点、英文弯引号和反引号。
5. 去掉英文冠词 a/an/the。
6. 合并空白。

如果归一化后的 pred_ans 与归一化后的 gold answer 完全相等，则 EM = 1。
否则 EM = 0。
```

#### cover_em_1

来源：代码。

是否能从当前代码确认：可以确认。

中文解释：

```text
把归一化后的 pred 和 gold 切成 token。
如果 gold 中每一个 token 都出现在 pred token 列表中，则得 1。
否则得 0。

不要求顺序，也不要求连续。
```

#### cover_em_2

来源：代码。

是否能从当前代码确认：可以确认。

中文解释：

```text
把归一化后的 pred 和 gold 切成 token。
如果 gold token 序列在 pred token 序列中连续出现，则得 1。
或者 gold 字符串是 pred 字符串的子串，也得 1。
否则得 0。

这个比 cover_em_1 更严格，因为要求顺序连续或整体子串匹配。
```

#### Precision / Recall / F1

来源：代码；技术说明主指标也包含 F1。

是否能从当前代码确认：可以确认。

中文解释：

```text
归一化 pred 和 gold 后切 token。
用 Counter 计算共同 token 数 overlap。

Precision = overlap / pred token 数
Recall = overlap / gold token 数
F1 = 2 * Precision * Recall / (Precision + Recall)
```

特殊处理：

```text
如果 pred 或 gold 是 yes/no/noanswer，并且两者不相等，
则 Precision、Recall、F1 全部直接记为 0。
```

多个 gold answer：

```text
如果一个问题有多个 gold answer，
代码会对每个 gold 分别计算指标，然后取最高分。
```

#### SBERT Similarity

来源：代码 + 技术说明。

是否能从当前代码确认：可以确认。

中文解释：

```text
先归一化 pred 和 gold。
用 sentence-transformers 模型 shibing624/text2vec-base-chinese 分别编码。
计算两个 embedding 的 cosine similarity。
```

多个 gold answer 时，同样取最高分。

#### ROUGE

来源：代码支持，但技术说明主表不一定使用。

是否能从当前代码确认：可以确认。

中文解释：

```text
先归一化 pred 和 gold。
调用 rouge 包计算 rouge-1、rouge-2、rouge-l 的 F 值。
如果 pred 或 gold 为空，则返回 0。
```

#### ACCL / LLM-as-Judge Accuracy

来源：技术说明 + 代码 `baselines/GraphRAG-R1/eval/eval_online.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
对每个样本，把 Question、Golden Answer、Predicted Answer 交给 Qwen evaluator。
如果 evaluator 输出中包含 true，则该样本正确。
否则错误。
Accuracy = 正确样本数 / 总样本数。
```

#### retrieve_num / response_times / generate_times / retrieve_tokens / reasoning_tokens

来源：代码输出和 `calc_rule.py`。

是否能从当前代码确认：可以确认。

中文解释：

```text
retrieve_num:
  每个问题调用检索的次数。

response_times:
  每轮检索响应时间列表。

generate_times:
  每轮生成时间列表。

retrieve_tokens:
  每轮检索返回文本 token 数列表。

reasoning_tokens:
  推理生成 token 数，代码取最后一个值计入总推理 token。
```

汇总方式：

```text
平均检索次数 = sum(retrieve_num) / valid_count
检索次数方差 = variance(retrieve_num_list)
检索次数中位数 = median(retrieve_num_list)
每轮平均响应时间 = 所有 response_times 求和 / response_times 总数量
每条平均响应时间 = 所有 response_times 求和 / 有 retrieve_num 的样本数
```

#### #Calls / #Token / ACCF

来源：技术说明。

是否能从当前代码确认：只能部分确认。

中文解释：

```text
#Calls:
  检索调用次数，可近似对应 retrieve_num。

#Token:
  检索和推理阶段 token consumption，可由 retrieve_tokens / reasoning_tokens 统计。

ACCF:
  完全符合格式要求的输出比例。
```

无法确认的部分：

```text
当前 eval/calc_rule.py 没有完整 ACCF 实现。
技术说明中 ACCF 的格式判定细节需要结合训练/evaluator 代码进一步确认。
```

### 5.2 原始 Prompt

#### qwen_base.py prompt v0b

用途：base model agentic retrieval inference。

```text
Answer the given question. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <|begin_of_query|> query <|end_of_query|> and it will return the top searched results between <|begin_of_documents|> and <|end_of_documents|>. During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags. For example, <answer> Beijing </answer>.
User:{question}Assistant: <think>
```

#### qwen_instruct.py prompt v0c

用途：instruct model agentic retrieval inference。

```text
Answer the given question. Reasoning step by step. After reasoning, if you find you lack some knowledge, you can call a search engine by <|begin_of_query|> query <|end_of_query|> and it will return the top searched results between <|begin_of_documents|> and <|end_of_documents|>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations, keep concise. For example, <answer> Yes </answer> or <answer> Miller Brewing </answer> .
User:{question}Assistant: <think>
```

#### eval_online.py judge prompt

用途：LLM-as-Judge Accuracy / ACCL。

```text
Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.

    Question: {question}
    Golden Answer: {reference}
    Predicted Answer: {prediction}
```

#### server-side NER prompt

来源：`server/src/hipporag/prompts/templates/ner.py`。

用途：passage NER。

```text
Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
```

One-shot user：

```text
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
```

One-shot assistant：

```text
{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
```

New user：

```text
${passage}
```

#### server-side query NER prompt

来源：`server/src/hipporag/prompts/templates/ner_query.py`。

System：

```text
You're a very effective entity extraction system.
```

One-shot user：

```text
Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?
```

One-shot assistant：

```text
{"named_entities": ["First for Women", "Arthur's Magazine"]}
```

New user：

```text
Question: ${query}
```

#### server-side triple extraction prompt

来源：`server/src/hipporag/prompts/templates/triple_extraction.py`。

System：

```text
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
```

User 模板：

~~~text
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
~~~

Assistant 输出格式：

```text
{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
```

#### server-side RAG QA prompt

来源：`server/src/hipporag/prompts/templates/rag_qa_musique.py`。

System：

```text
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
```

New user：

```text
${prompt_user}
```

`prompt_user` 格式：

```text
Wikipedia Title: {passage_1}
...
Question: {question}
Thought: 
```

#### IRCoT prompt

来源：`server/src/hipporag/prompts/templates/ircot_hotpotqa.py` / `ircot_musique.py`。

System 主体：

```text
You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".
```

New user：

```text
${prompt_user}
```

#### linking.py retrieval instructions

用途：不同 linking method 的 embedding retrieval instruction。

```text
ner_to_node: Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.
query_to_node: Given a question, retrieve relevant phrases that are mentioned in this question.
query_to_fact: Given a question, retrieve relevant triplet facts that matches this question.
query_to_sentence: Given a question, retrieve relevant sentences that best answer the question.
query_to_passage: Given a question, retrieve relevant documents that best answer the question.
default: Given a question, retrieve relevant documents that best answer the question.
```

#### filter_default_prompt.py

用途：fact filtering / reranking。该 prompt 与 AGRAG 中记录的 DSPy filter prompt 同源，要求输入 `question` 和 `fact_before_filter`，输出 `fact_after_filter` JSON。

完整 system instruction：

```text
You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.
```

## 6. 汇总判断

```text
ClueRAG:
  技术说明 QA Acc/F1 定义能确认，但当前代码实现缺失，不能直接复现官方 QA 指标。
  检索 Recall@k 可从代码确认。

LinearRAG:
  Contain Accuracy 和 LLM Accuracy 可从代码确认。
  Context Relevance 和 Evidence Recall 当前无法从代码确认。

AGRAG:
  技术说明指标定义能大致确认。
  GraphRAG-Bench QA wrapper 依赖外部 metrics，Coverage/Faithfulness/answer_correctness 的完整实现无法从当前仓库确认。

HiPRAG:
  CEM、格式检查、Search-R1 reward 可从代码确认。
  OSR/USR 公式可确认，但具体 step 判定依赖 LLM judge。

GraphRAG-R1:
  EM、cover EM、F1/P/R、SBERT、ROUGE、LLM judge prompt 和在线统计都可从代码确认。
  ACCF 的完整实现当前未在 eval/calc_rule.py 中确认。
```
