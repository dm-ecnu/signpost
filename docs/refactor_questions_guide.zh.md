# Signpost 重构问答式理解手册

本文档用于整理重构过程中已经讨论过的问题，重点解释“每个功能点为什么这样做、输入输出是什么、对后续阶段有什么影响”。它和 `refactor_implementation_notes.zh.md` 的区别是：

- `refactor_implementation_notes.zh.md`：按功能点记录代码结构、命令和实现边界。
- 本文档：按你实际问过的问题整理解释，方便回看和理解。

后续每次继续追问某个功能点或运行命令问题，都可以把对应解释追加到这里。

## Q1. 在子目录运行 `python -m signpost...` 为什么找不到 `signpost` 包？

### 问题现象

在 `signpost_re/docker` 目录运行：

```bash
conda run -n signpost-re python -m signpost.config.smoke --namespace legal
```

曾出现：

```text
ModuleNotFoundError: No module named 'signpost'
```

### 原因

Python 默认只会把当前目录和已安装包加入模块搜索路径。如果当前目录是：

```text
/home/ruolinsu/signpost/signpost_re/docker
```

那么它不会自动把上一级项目根目录：

```text
/home/ruolinsu/signpost/signpost_re
```

加入搜索路径。

### 解决方式

已把 `signpost_re` 改成可 editable install 的项目，并执行：

```bash
conda run -n signpost-re python -m pip install -e /home/ruolinsu/signpost/signpost_re
```

之后无论在项目根目录还是 `docker` 子目录，都可以运行：

```bash
python -m signpost.config.smoke --namespace legal
```

### 对应修改

- `pyproject.toml`
  - 增加 `build-system`。
  - 增加 `setuptools` 包发现配置。
- `docs/environment_setup.zh.md`
  - 将 `PYTHONPATH` 从旧项目 `signpost-main` 改为新项目 `signpost_re`。
  - 明确推荐安装新项目本体。

## Q2. 路径 `/datasets/...` 为什么报权限错误？

### 问题现象

运行：

```bash
python -m signpost.parsing.parse_documents \
  --input /datasets/processed/graphrag-bench-medical/raw_corpus.jsonl \
  --output /datasets/processed/graphrag-bench-medical/documents.jsonl
```

出现：

```text
PermissionError: [Errno 13] Permission denied: '/datasets'
```

### 原因

`/datasets/...` 是 Linux 系统根目录下的绝对路径，不是项目里的 `datasets` 目录。普通用户通常没有权限在系统根目录创建 `/datasets`。

### 正确写法

在 `signpost_re` 项目中，应使用项目相对路径：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/graphrag-bench-medical/raw_corpus.jsonl \
  --output datasets/processed/graphrag-bench-medical/documents.jsonl
```

或者完整绝对路径：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input /home/ruolinsu/signpost/signpost_re/datasets/processed/graphrag-bench-medical/raw_corpus.jsonl \
  --output /home/ruolinsu/signpost/signpost_re/datasets/processed/graphrag-bench-medical/documents.jsonl
```

### 额外注意

Shell 续行符 `\` 后面不能有空格。错误示例：

```bash
--input xxx \ 
```

正确示例：

```bash
--input xxx \
```

## Q3. F4 里 tree 和 chunk 是怎么出来的？

### 简短结论

F4 的顺序是：

```text
documents.jsonl
  -> 识别 headers
  -> 构建 document tree
  -> 根据 tree 做 tree-aware chunking
  -> 输出 document_trees.jsonl 和 chunks.jsonl
```

也就是：**先有 tree，再根据 tree 优先决定 chunk 边界。**

### 输入

F3.5 输出的 `documents.jsonl`，每行至少包含：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "text": "第一章 概述\n...",
  "lines": [
    {"line_no": 1, "text": "第一章 概述"},
    {"line_no": 2, "text": "Signpost 是一种路标机制。"}
  ],
  "metadata": {"dataset": "mini"}
}
```

### headers 怎么识别？

默认不用 LLM，走确定性章节识别。规则包括：

- Markdown 标题：`#`、`##`。
- 中文章节：`第一章`、`第二节`、`第三条`。
- 英文法律/章节格式：`ARTICLE I`、`CHAPTER 2`、`Section 1.2`。
- 数字编号标题：`1.1 Background`、`2.3.4 Method`。

如果加：

```bash
--use-llm
```

则走论文中的双路径章节识别：

- 短文档：LLM 将全文转换成 Markdown，再解析标题。
- 长文档：分窗口让 LLM 抽取标题 JSON，再合并。

### tree 怎么构建？

识别出 headers 后，根据标题级别和行号算出每个标题覆盖的内容范围，然后用栈式算法构建父子层级。

示例：

```text
第一章 总则
1.1 背景
1.2 方法
第二章 实验
```

会构建成：

```text
[ROOT]
  第一章 总则
    1.1 背景
    1.2 方法
  第二章 实验
```

输出到 `document_trees.jsonl`。

### chunk 怎么生成？

chunker 按 tree 做两阶段分块：

```text
遍历每个章节节点
  如果这个章节子树 <= max_tokens
    整个子树合成一个 chunk
  否则
    当前节点或子节点继续递归
    如果仍超预算，则按行切分，并保留 overlap
```

chunk 内容前会追加章节路径：

```text
第二章 方法

[CONTENT]

它使用图结构和 PPR 推荐。
```

这样后续 embedding、BM25、LLM 抽取时，即使只看到一个 chunk，也知道它属于哪个章节。

## Q4. F4 的 `split_long_line` 是什么？是否改变论文设计？

### 简短结论

没有改变文档树逻辑，也没有改变论文中的 tree-aware chunking 设计。

`split_long_line` 只是一个边界 fallback：当某个原始逻辑行本身超过 `max_tokens` 时，在这一行内部继续按词切成小 chunk。

### 为什么需要？

GraphRAG-Bench 的 medical/novel 数据中，有些样本会把一整篇文本压成一行。旧逻辑“按行切分”会失效，因为一行本身可能有几万 tokens。

旧结果示例：

```text
graphrag-bench-novel rows 20
最大 chunk 约 95k tokens
```

这会导致 ECNU embedding 直接失败。

### 修复后的逻辑

```text
如果某一行本身也超过 max_tokens
  -> 在这一行内部继续按词切成小 chunk
  -> metadata.merge = "split_long_line"
```

### 对后续有什么影响？

不会新增节点类型或边类型。

后续 F6/F7/F8/F9 主要读取：

```text
chunk_id
doc_id
file_name
content
start_line
end_line
section_path
prev_chunk_id
next_chunk_id
metadata.token_count
```

`metadata.merge = split_long_line` 只是说明该 chunk 的来源切分方式。

会影响的是 chunk 数量和 chunk_id。例如：

```text
旧：4_c00000 是整篇 95k token 文档
新：4_c00000, 4_c00001, 4_c00002, ...
```

因此，某个数据集如果重新跑了 F4，就需要从 F5 往后重跑。

### 重跑建议

如果 F4 产物来自旧版本，建议重跑：

```text
F4 -> F5 -> F6 -> F7 -> F8 -> F9 -> F10
```

F11-F13/F15 取决于是否已经基于旧图跑过检索或预测，通常也建议重新生成。

## Q5. F5 里的 namespace、chunk id、bulk 写入、mapping 是什么？

### namespace 是什么？

`namespace` 是实验或数据集隔离名，例如：

```text
mini
legal-ecnu
graphrag-bench-novel-ecnu
```

默认 ES chunk index 命名为：

```text
signpost-<namespace>-chunks
```

例如：

```text
signpost-legal-ecnu-chunks
```

同一个 ES 可以存多个实验索引，互不混淆。

### chunk id 是什么？

chunk id 来自 F4：

```text
<doc_id>_c00000
<doc_id>_c00001
```

例如：

```text
mini_doc_001_c00000
```

表示 `mini_doc_001` 的第 0 个 chunk。

### bulk 写入是什么？

ES bulk 是批量写入接口。相比一条一条写：

```text
insert chunk 1
insert chunk 2
...
```

bulk 会一次请求写入多条文档，适合几千到几十万 chunk 的索引构建。

### 为什么需要 mapping？

ES 需要知道每个字段如何索引：

- `content`: `text`，用于 BM25。
- `content_vector`: `dense_vector`，用于向量检索。
- `namespace`: `keyword`，用于过滤。
- `start_line/end_line`: `integer`，用于溯源。
- `section_path`: `keyword`，用于章节路径保存和过滤。

所以 F5 需要：

```text
index name
mapping
chunk -> ES document 转换
```

## Q6. ES 是向量数据库吗？BM25、dense、hybrid 为什么都需要？

### 简短结论

Elasticsearch 是搜索引擎，不只是向量数据库。它既支持 BM25 文本检索，也支持 dense vector 相似度检索。

ES 不会自动把文本变成 embedding。embedding 由 ECNU 或 hash provider 生成，ES 只负责保存向量和计算相似度。

### BM25

BM25 是关键词检索，不使用 embedding。

适合：

- 精确术语。
- 法律条款编号。
- 专有名词。
- 字面匹配强的问题。

### dense

dense 是向量检索。

流程：

```text
query -> ECNU embedding -> query_vector
chunk.content -> ECNU embedding -> content_vector
ES 计算 cosineSimilarity(query_vector, content_vector)
```

适合：

- 语义相近但字面不同的问题。
- 改写、同义表达。

### hybrid

hybrid 同时跑 BM25 和 dense，然后用 RRF 合并排名。

原因是 BM25 和 dense 各有强弱：

- BM25 擅长精确词。
- dense 擅长语义相似。

RRF 不直接比较原始分数，而是融合排名，稳定性更好。

## Q7. F5 调 ECNU embedding 为什么会 HTTP 500、SSL EOF 或看起来卡住？

### HTTP 500：常见原因之一是 chunk 太长

如果报错里显示：

```text
chars=168290 tokens=34385
```

说明 chunk 极端超长，embedding 服务处理不了。解决方式是先重新跑 F4，确保 chunk 大小正常。

### SSL EOF：远程连接中断

如果报错是：

```text
ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING
```

说明 HTTPS 连接被服务端或网络中途断开。这通常是远程服务稳定性或网络问题，不代表 chunk 一定太长。

当前 F5 已增强：

- batch 失败后默认重试。
- batch 多次失败会自动拆成更小 batch。
- 只有单条 chunk 多次失败才最终报错。

推荐真实数据集命令：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace <dataset>-ecnu \
  --chunks datasets/processed/<dataset>/chunks.jsonl \
  --embedding-provider ecnu \
  --batch-size 4 \
  --progress-every 10 \
  --embedding-retries 5 \
  --retry-sleep 3 \
  --recreate
```

### 看起来卡住

如果数据集有很多 chunk，例如 legal 有上万条，`batch-size=4` 会产生几千次远程 embedding 请求。之前没有进度输出时，看起来像卡住。

现在可以用：

```bash
--progress-every 10
```

看到类似：

```text
indexed=401/12692 batches=100
```

### 命令续行注意

反斜杠后面不能有空格。错误示例：

```bash
--namespace legal-ecnu \    
```

正确示例：

```bash
--namespace legal-ecnu \
```

## Q8. F6 是不是 LLM 抽取图元素？是否写数据库？

### 简短结论

F6 是从 `chunks.jsonl` 中抽取实体和关系，生成语义图文件：

```text
chunks.jsonl -> graph.semantic.json
```

它暂时不写数据库，也不写 ES。真正把图对象同步到 ES 是 F10。

### F6 是否使用 LLM？

有两种模式：

生产路径：

```bash
--extractor llm
```

会对每个 chunk 调用 ECNU/OpenAI-compatible chat，让模型输出实体和关系 JSON。

本地 smoke 路径：

```bash
--extractor deterministic
```

不用模型，靠规则抽词和共现关系，用于测试管线是否跑通。

### LLM 输出什么？

LLM 需要返回：

```json
{
  "entities": [
    {
      "name": "PPR",
      "type": "CONCEPT",
      "description": "..."
    }
  ],
  "relations": [
    {
      "source": "Signpost",
      "target": "PPR",
      "description": "...",
      "keywords": ["recommendation"],
      "weight": 1.0
    }
  ]
}
```

支持 gleaning，多轮补充抽取，默认最多 2 轮。

### F6 输出哪些节点？

一类是 chunk 节点：

```json
{
  "node_id": "chunk:mini_doc_001_c00001",
  "node_type": "chunk",
  "chunk_id": "mini_doc_001_c00001",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "start_line": 4,
  "end_line": 6,
  "section_path": ["第二章 方法"]
}
```

另一类是 entity 节点：

```json
{
  "node_id": "entity:b4a9ca4f6d48",
  "node_type": "entity",
  "name": "PPR",
  "entity_type": "CONCEPT",
  "description": "...",
  "source_chunk_ids": ["mini_doc_001_c00001"],
  "source_locates": ["mini.txt:L4-L6"],
  "source_mapping": {
    "mini_doc_001:mini_doc_001_c00001": {
      "description": "LLM 抽取出来的实体描述",
      "entity_type": "CONCEPT",
      "file_name": "mini.txt",
      "start_line": 4,
      "end_line": 6
    }
  },
  "type_counts": {"CONCEPT": 1},
  "auto_created": false
}
```

### F6 输出哪些边？

一类是实体关系边：

```json
{
  "source": "entity:...",
  "target": "entity:...",
  "edge_type": "semantic_relation",
  "description": "...",
  "relation_types": ["co_occurs"],
  "weight": 1.0,
  "source_chunk_ids": ["mini_doc_001_c00001"],
  "source_locates": ["mini.txt:L4-L6"],
  "source_mapping": {
    "mini_doc_001:mini_doc_001_c00001": {
      "description": "LLM 抽取出来的关系描述",
      "relation_types": ["co_occurs"],
      "weight": 1.0,
      "file_name": "mini.txt",
      "start_line": 4,
      "end_line": 6
    }
  }
}
```

另一类是实体到 chunk 的溯源边：

```json
{
  "source": "entity:...",
  "target": "chunk:mini_doc_001_c00001",
  "edge_type": "source",
  "source_chunk_ids": ["mini_doc_001_c00001"],
  "source_locates": ["mini.txt:L4-L6"]
}
```

### `split_long_line` 对 F6 有什么影响？

不改变 F6 的节点/边类型。

它只让输入 chunk 粒度更细：

```text
旧：一个巨大 chunk 抽很多实体关系，容易超过 LLM/embedding 上限
新：多个小 chunk 分别抽取，source_locates 更细，LLM 输入更稳定
```

实体合并逻辑仍然是：

```text
同名实体 -> 同一个 entity node
同一实体对关系 -> 合并为一条 semantic_relation
```

source 边仍然是：

```text
entity -> chunk
```

只是 chunk 变得更细。

## Q9. F6 命令输出 `chunks=1` 是不是只处理了 1 个 chunk？

### 问题现象

运行：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace agriculture-llm \
  --chunks datasets/processed/agriculture/chunks.jsonl \
  --output datasets/processed/agriculture/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --max-chunks 1
```

输出：

```text
chunks=1 entities=8 relations=9 source_edges=8
```

### 原因

这是 `--max-chunks 1` 的预期行为。

`--max-chunks` 是调试/烟测参数，用来限制只处理输入文件前 N 个 chunk：

```python
chunks = list(read_jsonl(chunks_path))
if max_chunks is not None:
    chunks = chunks[:max_chunks]
```

所以：

```text
--max-chunks 1  -> 只处理前 1 个 chunk
--max-chunks 10 -> 只处理前 10 个 chunk
不传该参数       -> 处理全部 chunk
```

### 为什么文档里 smoke 用 `--max-chunks 1`？

因为 `--extractor llm` 会对每个 chunk 调用一次或多次 LLM。对于 agriculture、legal 这种几千到上万个 chunk 的数据集，全量运行会非常慢，也会消耗大量模型额度。

所以推荐流程是：

```text
先 --max-chunks 1 验证 LLM 抽取链路
再 --max-chunks 10 看输出稳定性
最后移除 --max-chunks 做全量
```

### 全量运行怎么写？

移除 `--max-chunks`：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace agriculture-llm \
  --chunks datasets/processed/agriculture/chunks.jsonl \
  --output datasets/processed/agriculture/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1
```

### 输出里的 `chunks` 表示什么？

F6 输出里的：

```text
chunks=1
```

表示“本次实际参与语义抽取的 chunk 数”，不是原始数据集总 chunk 数。

如果想确认输入文件实际有多少 chunk，可以运行：

```bash
wc -l datasets/processed/agriculture/chunks.jsonl
```

### 后续文档维护规则

如果之后某个命令参数、实现细节或运行报错会影响理解，除了直接回答，也要同步更新：

- `docs/refactor_questions_guide.zh.md`
- 必要时更新 `docs/refactor_implementation_notes.zh.md`

## Q10. F6 全量 LLM 抽取为什么会 timeout？`gleaning` 是什么？

### 问题现象

全量运行：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace agriculture-llm \
  --chunks datasets/processed/agriculture/chunks.jsonl \
  --output datasets/processed/agriculture/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1
```

可能出现：

```text
TimeoutError: The read operation timed out
```

### 原因

这是 ECNU/OpenAI-compatible chat 远程请求超时。F6 的 LLM extractor 会对每个 chunk 调用模型抽取实体和关系。全量 agriculture 有几千个 chunk，所以会产生大量远程 chat 请求。

这类错误通常不是图构建逻辑错误，而是远程模型接口在某次请求中响应慢、网络波动或服务端临时不稳定。

### `gleaning` 是什么？

`gleaning` 可以理解成“补充抽取”。

F6 对一个 chunk 的 LLM 抽取流程是：

```text
第 1 次：抽取这个 chunk 里的实体和关系
第 2 次：把上一次结果给模型，让它只补充遗漏的实体和关系
第 3 次：继续补充
...
```

参数：

```bash
--gleaning-rounds 1
```

表示：

```text
首轮抽取 + 1 轮补充抽取
```

所以每个 chunk 最多 2 次 LLM 调用。

如果：

```bash
--gleaning-rounds 2
```

就是每个 chunk 最多 3 次 LLM 调用。

### 成本估算

假设数据集有 4000 个 chunk：

```text
gleaning-rounds=0 -> 最多 4000 次 LLM 调用
gleaning-rounds=1 -> 最多 8000 次 LLM 调用
gleaning-rounds=2 -> 最多 12000 次 LLM 调用
```

所以全量 LLM 抽取非常耗时，也会消耗较多模型额度。

### 当前代码怎么处理 timeout？

F6 已支持：

- `--progress-every`：输出处理进度。
- `--progress-file`：把进度事件写入 JSONL 文件，便于 `tail -f` 查看。
- `--extractions-cache`：逐 chunk 保存 LLM 抽取结果，支持中断后恢复。
- `--llm-retries`：单次 LLM 请求失败后的重试次数。
- `--retry-sleep`：重试间隔。
- `--llm-timeout`：单次请求超时时间。

推荐先用小规模 smoke：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace agriculture-llm-smoke \
  --chunks datasets/processed/agriculture/chunks.jsonl \
  --output datasets/processed/agriculture/graph.semantic.llm.smoke.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --max-chunks 10 \
  --progress-every 1 \
  --progress-file datasets/processed/agriculture/semantic_llm.smoke.progress.jsonl \
  --extractions-cache datasets/processed/agriculture/semantic_llm.smoke.extractions.jsonl \
  --llm-retries 5 \
  --retry-sleep 3 \
  --llm-timeout 180
```

全量运行：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace agriculture-llm \
  --chunks datasets/processed/agriculture/chunks.jsonl \
  --output datasets/processed/agriculture/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --progress-every 10 \
  --progress-file datasets/processed/agriculture/semantic_llm.progress.jsonl \
  --extractions-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --llm-retries 5 \
  --retry-sleep 3 \
  --llm-timeout 180
```

如果终端仍看不到实时输出，可以另开一个终端运行：

```bash
tail -f datasets/processed/agriculture/semantic_llm.progress.jsonl
```

进度文件每行是一个 JSON 事件，例如：

```json
{"event":"extracting","index":12,"total":4321,"chunk_id":"...","tokens":1200}
{"event":"processed","index":12,"total":4321,"entities":80,"relations":65}
```

如果只出现 `extracting` 很久没有对应 `processed`，说明当前卡在这个 chunk 的 LLM 请求上。

### 为什么需要 `--extractions-cache`？

F6 的最终输出 `graph.semantic.llm.json` 是合并后的全局语义图，只有全部 chunk 合并完才会完整写出。如果只做最终图文件，长任务中断后就会浪费已经完成的 LLM 抽取。

`--extractions-cache` 解决这个问题：每完成一个 chunk，就立即追加一行抽取结果：

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "file_name": "...",
  "start_line": 1,
  "end_line": 3,
  "extraction": {
    "entities": [],
    "relations": []
  }
}
```

重跑同一命令时，会先读取 cache：

```text
cache 里已有的 chunk -> 不再调用 LLM，直接复用
cache 里没有的 chunk -> 继续调用 LLM 抽取
最后统一合并成 graph.semantic.llm.json
```

如果使用 `--extractor llm` 且没有显式传 `--extractions-cache`，CLI 会默认在输出文件旁边创建：

```text
graph.semantic.llm.extractions.jsonl
```

但长任务仍建议显式传路径，方便自己查看和管理。

### 实验建议

如果当前目标只是验证后续 F7-F16 流程，先用：

```bash
--extractor deterministic
```

如果目标是论文最终质量实验，再用：

```bash
--extractor llm
```

并且建议先按数据集抽样检查 `graph.semantic.llm.json` 的实体和关系质量，再做全量。

## Q11：ICDE 实验需要补哪些指标？怎么记录日志？

实验设计文档里需要的指标不只包括答案 EM/F1，还包括离线索引成本、在线查询成本、图结构指标、弱证据命中、摊销成本和 break-even。现在新增的指标代码集中在：

```text
signpost/benchmark/
```

### 需要记录哪些日志？

阶段级日志：

```text
outputs/{dataset}/logs/stage_timing.jsonl
```

用于回答：

- F3/F3.5/F4 shared preprocessing 花了多久？
- F5-F10 每个离线索引阶段花了多久？
- 哪个阶段失败了，失败码是什么？

query 级日志或 prediction：

```text
outputs/{dataset}/predictions/{method}.jsonl
outputs/{dataset}/logs/{method}.query.jsonl
```

用于回答：

- 每条 query 的 LLM calls、tool calls、tokens、latency 是多少？
- p90/p95 尾延迟是多少？
- ReadFile 调了几次？
- graph PPR 调了几次？

F6 语义抽取 cache：

```text
datasets/processed/{dataset}/semantic_llm.extractions.jsonl
```

用于回答：

- 抽取了多少 chunks？
- 估算 LLM calls 是多少？
- 每个 chunk 抽出多少实体和关系？

### 新增脚本怎么用？

统计 query 指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

统计离线和图结构指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/agriculture/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/agriculture/graph.unified.json \
  --output outputs/agriculture/metrics/index_metrics.json
```

统计成本-质量派生指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.cost_quality \
  --methods outputs/agriculture/metrics/method_summaries.json \
  --output outputs/agriculture/metrics/cost_quality.json
```

完整字段说明在：

```text
docs/experiment_metrics_guide.zh.md
```

如果觉得字段太多，可以先读更口语化的入门版：

```text
docs/experiment_metrics_plain_guide.zh.md
```
