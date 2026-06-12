# Signpost 重构功能点计划

本文档说明 `signpost_re` 应该复现哪些技术说明功能点、每个功能点如何重新实现、依赖什么环境、输入输出格式是什么，以及数据应该如何准备。

核心原则：

- 不直接复制旧项目的大块代码。
- 旧项目 `signpost-main` 只作为参考实现和算法线索。
- 新项目按技术说明功能点重新实现，模块边界重新设计。
- 每个功能点都必须能独立运行、独立验证、独立测速。
- 在数据未整理好之前，先用 `samples/` 下的极小样例验证功能，不依赖完整实验集。

## 1. 总体目标

`signpost_re` 是一个科研实验系统，不是产品后端。

目标工作流：

```text
原始语料
-> 数据标准化
-> Index 阶段
   -> 文档解析与文本规范化
   -> 文档切块
   -> embedding
   -> ES chunk index
   -> 语义图构建
   -> 结构视图/RAPTOR 构建
   -> 多视图统一图持久化
   -> 图对象同步到 ES
-> Retrieval 阶段
   -> chunk/RAPTOR/entity/edge 检索
   -> 离线路标生成
   -> 在线 PPR 路标生成
   -> 可选 Agent 检索
-> Eval 阶段
   -> 生成预测 JSONL
   -> LLM-as-Judge 或其他指标评估
```

新项目不保留：

- 多用户系统
- 租户权限
- 登录鉴权
- 前端
- Canvas
- MCP
- 对话产品功能
- API token 管理
- 文件夹/缩略图等产品管理逻辑

旧项目中这些字段如果出现在算法代码里，应被替换成实验上下文：

```text
tenant_id -> namespace
user_id   -> namespace
kb_id     -> dataset_id 或 index_id
```

## 2. 推荐目录结构

```text
signpost_re/
  signpost/
    config/
    llm/
    storage/
    data/
    chunking/
    indexing/
    graph/
    retrieval/
    agent/
    evaluation/
    benchmark/
  scripts/
  samples/
  datasets/
  outputs/
  tests/
  docs/
```

说明：

- `config/`：配置、实验上下文、环境变量读取。
- `llm/`：ECNU/OpenAI-compatible 模型客户端。
- `storage/`：ES、MinIO、Redis、PostgreSQL 的最小封装。
- `data/`：数据集标准化、JSONL 读写、语料清洗。
- `parsing/`：文档解析、文本规范化、行号保留、占位符处理。
- `chunking/`：技术说明中的文档切块和章节路径处理。
- `indexing/`：index 阶段入口和流水线。
- `graph/`：多视图拓扑图的数据结构、保存、校验。
- `retrieval/`：普通检索、图检索、Signpost 路标。
- `agent/`：Supervisor-Researcher，多工具 ReAct。
- `evaluation/`：预测输出格式转换和评估适配。
- `benchmark/`：统一测速 wrapper，但具体指标之后再定。

## 3. 功能点清单

下面每个功能点都应该有：

```text
独立模块
独立 CLI
样例输入
样例输出
smoke test
可选测速入口
```

### F0. 配置与实验上下文

技术说明对应：

- 第五章系统实现中的基础设施层。

要实现什么：

- 读取 `.env` 和 YAML 配置。
- 定义统一实验上下文 `ExperimentContext`。
- 用 `namespace` 表示一次实验或数据集命名空间。
- 不再引入用户、租户、权限。

建议数据结构：

```python
ExperimentContext(
    namespace="legal",
    dataset_id="legal",
    run_id="2026xxxx",
    output_dir="outputs/..."
)
```

环境依赖：

- 无外部服务依赖。

输入：

- `.env`
- `conf/*.yaml`

输出：

- Python 配置对象。

独立验证：

```bash
python -m signpost.config.smoke --namespace legal
```

旧项目参考：

- `core/config.py`
- `api/settings.py`

注意：

- 只参考配置字段含义，不复制全局单例写法。

### F1. 模型客户端

技术说明对应：

- 实体关系抽取。
- RAPTOR 摘要生成。
- Agent 推理。
- LLM-as-Judge。
- embedding。
- rerank。

要实现什么：

- ECNU 作为默认 provider。
- 保留 OpenAI-compatible fallback。
- chat、stream chat、embedding、rerank 分开实现。
- 支持 ECNU `thinking: {"type": "enabled"}`，但作为可选参数。
- 所有 key 从环境变量读取，不写入文件。

默认模型：

```text
chat: ecnu-plus
reasoning/complex: ecnu-max
embedding: ecnu-embedding-small
rerank: ecnu-rerank
```

环境依赖：

- ECNU API 或 OpenAI-compatible API。

输入：

- messages
- texts
- query/doc pairs

输出：

- chat text
- embedding vectors
- rerank scores

独立验证：

```bash
python -m signpost.llm.smoke --chat
python -m signpost.llm.smoke --embedding
python -m signpost.llm.smoke --rerank
```

旧项目参考：

- `core/llm/core.py`
- `core/llm/chat_model.py`
- `core/llm/embedding_model.py`
- `core/llm/rerank_model.py`

注意：

- 不复制旧项目的多 provider 历史包袱。
- 先只实现 ECNU/OpenAI-compatible。

### F2. 存储连接

技术说明对应：

- 第五章基础设施层。
- ES 文档/向量检索。
- MinIO 图持久化。
- Redis 锁。
- PostgreSQL cache/metadata。

要实现什么：

- ES 最小封装：创建 index、写入、查询、删除、健康检查。
- MinIO 最小封装：put/get/list/delete。
- Redis 最小封装：get/set/lock。
- PostgreSQL 最小封装：后续给 LLM cache 或运行记录用。

环境依赖：

- Elasticsearch
- MinIO
- Valkey/Redis
- PostgreSQL

输入：

- 配置文件

输出：

- 可用 client。

独立验证：

```bash
python -m signpost.storage.smoke --es
python -m signpost.storage.smoke --minio
python -m signpost.storage.smoke --redis
python -m signpost.storage.smoke --db
```

旧项目参考：

- `core/storage/es_conn.py`
- `core/storage/minio_conn.py`
- `core/storage/redis_conn.py`
- `core/db/models.py`

注意：

- 不复制旧 ORM 的用户、会话、Canvas 等表。
- ES mapping 重新按科研对象设计。

### F3. 数据标准化

技术说明对应：

- 第二章 2.2 文档解析之前的数据组织。
- 第五章实验数据集与任务设置。
- UltraDomain: Agriculture、Legal、Mix。
- GraphRAG-Bench。

要实现什么：

- 把不同来源数据集统一成进入“文档解析”之前的项目内部格式。
- 区分 raw corpus manifest 和 questions。
- 不在这一阶段做复杂文档解析。
- 不要求所有原文都已经变成纯文本；允许 txt、md、json、jsonl 等来源先登记到 manifest。
- 保留标准答案、推理路径、原始 metadata。

推荐目录：

```text
datasets/
  raw/
    legal/
    agriculture/
    mix/
    graphrag-bench/
  processed/
    legal/
      raw_corpus.jsonl
      questions.jsonl
    agriculture/
      raw_corpus.jsonl
      questions.jsonl
```

`raw_corpus.jsonl` 格式：

```json
{
  "doc_id": "legal_doc_001",
  "file_name": "xxx.txt",
  "source_path": "datasets/raw/legal/xxx.txt",
  "source_format": "txt",
  "text": null,
  "metadata": {
    "domain": "legal",
    "source_path": "datasets/raw/legal/xxx.txt"
  }
}
```

说明：

- 如果原文已经是 `.txt` 或 `.md`，可以把 `text` 留空，只记录 `source_path`，由 F3.5/F4 文档解析阶段读取。
- 如果原始数据本身是 JSONL 且每行已经包含正文，可以在 `text` 字段直接放正文，同时保留原始字段到 `metadata.raw`。
- 这一阶段的目标是“知道有哪些文档、它们在哪里、属于哪个数据集”，不是生成最终可索引文本。

`questions.jsonl` 格式：

```json
{
  "question_id": "legal_q_0001",
  "question": "问题文本",
  "answer": "标准答案",
  "rationale": "专家推理路径或参考依据",
  "metadata": {
    "dataset": "legal"
  }
}
```

环境依赖：

- 无外部服务依赖。

输入：

- 原始 txt/md/json/jsonl 数据目录或文件。

输出：

- `processed/<dataset>/raw_corpus.jsonl`
- `processed/<dataset>/questions.jsonl`

独立验证：

```bash
python -m signpost.data.prepare --dataset legal
python -m signpost.data.validate --dataset legal
```

旧项目参考：

- `eval/`
- `deepresearch/batch_runner.py`

注意：

- 在真实数据未整理好前，创建 `samples/mini/raw_corpus.jsonl` 和 `samples/mini/questions.jsonl`。
- 后续所有功能先基于 mini 数据跑通。

### F3.5. 文档解析与文本规范化

技术说明对应：

- 第二章 2.2 文档解析。
- 第三章 3.2 章节层次识别的输入前提。
- 第三章 3.2.1 中的文本规范化、编号行、行号定位。
- 第三章 3.2.3 中的多模态占位符处理思想。

为什么必须单独列出：

- 技术说明中的结构视图依赖“可定位的行号”和“较干净的文本”。
- 原始数据可能来自 txt、Markdown、JSONL、教材语料或已经 OCR 后的文本。
- 如果不先把文档解析结果规范化，后面的章节识别、chunking、溯源读取都会不稳定。

要实现什么：

- 读取 `raw_corpus.jsonl` 中登记的原始文档。
- 将不同来源统一成“可解析文档”格式。
- 保留文件名、原始路径、原始行号。
- 做 Unicode NFKC 规范化。
- 做基础标点标准化。
- 过滤空行和冗余空白，但不能破坏行号定位。
- 为图片、表格、链接等非纯文本元素保留占位符。
- 初版只支持 txt/md/json/jsonl，PDF/DOCX 暂不作为第一阶段目标。

`documents.jsonl` 格式：

```json
{
  "doc_id": "legal_doc_001",
  "file_name": "xxx.txt",
  "source_path": "datasets/raw/legal/xxx.txt",
  "text": "规范化后的全文",
  "lines": [
    {"line_no": 1, "text": "第一章 总则"},
    {"line_no": 2, "text": "......"}
  ],
  "placeholders": [
    {
      "placeholder": "![table_1](1)",
      "type": "table",
      "line_no": 25,
      "raw": "<table>...</table>"
    }
  ],
  "metadata": {
    "dataset": "legal",
    "source_format": "txt"
  }
}
```

环境依赖：

- 初版无外部服务依赖。
- 如果后续支持 PDF/DOCX，可选依赖 MinerU、python-docx 或其他解析器。

输入：

- `processed/<dataset>/raw_corpus.jsonl`

输出：

- `processed/<dataset>/documents.jsonl`

独立验证：

```bash
python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal/raw_corpus.jsonl \
  --output datasets/processed/legal/documents.jsonl

python -m signpost.parsing.validate_documents \
  --input datasets/processed/legal/documents.jsonl
```

旧项目参考：

- `preprocessing/chapter_extractor/text_normalizer.py`
- `preprocessing/chapter_extractor/read_file_tool.py`
- `api/utils/file_utils.py`

注意：

- 不复制旧项目的文件服务和权限逻辑。
- PDF/DOCX 不在第一阶段强行支持，先让 txt/md/jsonl 跑通。
- 行号是后续 Signpost 溯源的核心，任何清洗都要保留 line mapping。

### F4. 文档切块与章节路径

技术说明对应：

- 第三章结构视图。
- 学术文档分块。
- 章节层次识别。
- 顺序视图基础。
- 第三章 3.2 的短文档 Markdown 转换和长文档迭代式章节抽取。
- 第三章 3.3 的文档树构建和层次化分块。

要实现什么：

- 输入解析后的 `documents.jsonl`。
- 识别或读取章节标题序列。
- 构建文档树。
- 按章节结构和 token 预算切块。
- 输出 chunk JSONL。
- 每个 chunk 必须带行号、文件名、章节路径、顺序位置。

`chunks.jsonl` 格式：

```json
{
  "chunk_id": "legal_doc_001_c0001",
  "doc_id": "legal_doc_001",
  "file_name": "xxx.txt",
  "content": "chunk 文本",
  "start_line": 10,
  "end_line": 35,
  "section_path": ["第一章", "第一节"],
  "prev_chunk_id": null,
  "next_chunk_id": "legal_doc_001_c0002",
  "metadata": {}
}
```

环境依赖：

- 可先无 LLM。
- 如果启用 LLM 章节识别，则依赖 ECNU chat。

输入：

- `documents.jsonl`

输出：

- `chunks.jsonl`

独立验证：

```bash
python -m signpost.chunking.run \
  --input samples/mini/documents.jsonl \
  --output outputs/mini/chunks.jsonl

python -m signpost.chunking.validate \
  --chunks outputs/mini/chunks.jsonl
```

旧项目参考：

- `chunking/`
- `preprocessing/chapter_extractor/`
- `preprocessing/doc_tree_builder/`

注意：

- 初版可以先实现规则/标题切块，不急着接 LLM 章节识别。
- LLM 章节识别作为增强功能单独接入。

### F5. Chunk Index

技术说明对应：

- 基础 RAG 检索。
- 后续 GraphRAG 从 ES 读取 chunk。

要实现什么：

- 对 chunks 生成 embedding。
- 写入 ES。
- 支持按 namespace/dataset 创建独立 index。
- 支持简单查询验证。

ES 文档核心字段：

```json
{
  "id": "chunk_id",
  "type": "chunk",
  "namespace": "legal",
  "doc_id": "legal_doc_001",
  "content": "...",
  "content_vector": [0.1, 0.2],
  "file_name": "xxx.txt",
  "start_line": 10,
  "end_line": 35,
  "section_path": ["第一章", "第一节"],
  "prev_chunk_id": null,
  "next_chunk_id": "..."
}
```

环境依赖：

- ES
- ECNU embedding

输入：

- `chunks.jsonl`

输出：

- ES index

独立验证：

```bash
python -m signpost.indexing.chunk_index \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl

python -m signpost.retrieval.chunk_search \
  --namespace mini \
  --query "测试问题"
```

旧项目参考：

- `worker/doc_pipeline.py`
- `core/nlp/search.py`
- `core/storage/es_conn.py`

注意：

- 初版只做 dense retrieval 也可以。
- BM25 + dense hybrid 可以作为后续增强。

### F6. 语义视图：实体关系图

技术说明对应：

- 第三章语义视图：知识图谱层构建。

要实现什么：

- 对 chunk 调用 LLM 抽取实体和关系。
- 保留来源映射。
- 构建 entity node、relation edge、entity-chunk source edge。
- 保存为 NetworkX 图或自定义图 JSON。

图节点示例：

```json
{
  "node_id": "entity:xxx",
  "node_type": "entity",
  "name": "实体名",
  "description": "局部或统一描述",
  "source_chunk_ids": ["chunk1", "chunk2"]
}
```

图边示例：

```json
{
  "source": "entity:a",
  "target": "entity:b",
  "edge_type": "semantic_relation",
  "description": "关系描述",
  "source_chunk_ids": ["chunk1"]
}
```

环境依赖：

- ECNU chat
- MinIO 或本地文件系统

输入：

- `chunks.jsonl`

输出：

- `graph.semantic.json`

独立验证：

```bash
python -m signpost.indexing.semantic_graph \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl

python -m signpost.graph.inspect \
  --graph outputs/mini/graph.semantic.json
```

旧项目参考：

- `graphrag/indexing/extraction/`
- `graphrag/indexing/builder.py`
- `graphrag/indexing/unified_graph.py`

注意：

- 初版先不做复杂实体消解。
- 先保证实体、关系、source mapping 正确。

### F7. 结构视图：文档树与 RAPTOR

技术说明对应：

- 第三章结构视图。
- 基于文档树的层次化摘要。
- RAPTOR 摘要节点。

要实现什么：

- 从 chunk 的 `section_path` 重建文档树。
- 自底向上生成摘要节点。
- 建立 parent/child 边。
- 保存 source chunk 定位。

RAPTOR 节点示例：

```json
{
  "node_id": "raptor:xxx",
  "node_type": "raptor",
  "title": "章节标题",
  "content": "摘要内容",
  "level": 1,
  "parent_node_id": null,
  "child_node_ids": ["..."],
  "source_chunk_ids": ["chunk1", "chunk2"],
  "source_locates": ["xxx.txt:L10-L35"]
}
```

环境依赖：

- ECNU chat
- ECNU embedding 可选

输入：

- `chunks.jsonl`

输出：

- `graph.structure.json`

独立验证：

```bash
python -m signpost.indexing.structure_graph \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl

python -m signpost.graph.inspect \
  --graph outputs/mini/graph.structure.json
```

旧项目参考：

- `graphrag/indexing/raptor/`
- `preprocessing/doc_tree_builder/`

注意：

- 初版可以先用 section_path 聚合生成摘要。
- 标准 RAPTOR 聚类可后续再加。

### F8. 顺序视图

技术说明对应：

- 第三章顺序视图。
- 保持原始叙事顺序，支持上下文补全。

要实现什么：

- 根据 chunk 顺序建立 prev/next 边。
- 保留文档内位置。
- 支持检索结果向前/向后扩展上下文。

环境依赖：

- 无外部服务依赖。

输入：

- `chunks.jsonl`

输出：

- `graph.sequence.json` 或直接合入 unified graph。

独立验证：

```bash
python -m signpost.indexing.sequence_graph \
  --chunks outputs/mini/chunks.jsonl
```

旧项目参考：

- 旧项目中顺序视图不够独立，应重新实现。

### F9. 多视图统一图

技术说明对应：

- 第三章多视图拓扑图。
- 结构视图、语义视图、顺序视图融合。

要实现什么：

- 合并 semantic graph、structure graph、sequence graph。
- 统一 node/edge 类型。
- 支持保存、加载、校验。
- 支持图对象同步到 ES。

输出：

```text
graph.unified.json
```

环境依赖：

- MinIO 或本地文件系统
- ES，如果同步索引

独立验证：

```bash
python -m signpost.graph.merge \
  --semantic outputs/mini/graph.semantic.json \
  --structure outputs/mini/graph.structure.json \
  --sequence outputs/mini/graph.sequence.json \
  --output outputs/mini/graph.unified.json

python -m signpost.graph.validate \
  --graph outputs/mini/graph.unified.json
```

旧项目参考：

- `graphrag/indexing/unified_graph.py`
- `core/storage/graph_repository.py`

注意：

- 旧项目的 Lazy Merge 可以后置。
- 初版先做确定性 merge。

### F10. 图对象同步到 ES

技术说明对应：

- 第五章 Elasticsearch 索引同步。
- entity、edge、RAPTOR 摘要节点可检索。

要实现什么：

- 将 unified graph 中的 entity、edge、raptor nodes 转成 ES 文档。
- embedding 后写入 ES。
- chunk 文档中可附加父 RAPTOR 节点或 section 信息。

环境依赖：

- ES
- ECNU embedding

输入：

- `graph.unified.json`

输出：

- ES index 中的 graph documents。

独立验证：

```bash
python -m signpost.indexing.graph_es_sync \
  --namespace mini \
  --graph outputs/mini/graph.unified.json

python -m signpost.retrieval.graph_search \
  --namespace mini \
  --query "测试实体"
```

旧项目参考：

- `graphrag/indexing/es_syncer/`

### F11. 离线路标

技术说明对应：

- 第四章路标机制。
- 离线路标：层次、相邻位置、溯源信息。

要实现什么：

对每个检索结果生成结构化 signpost：

chunk 结果：

```json
{
  "file_name": "xxx.txt",
  "start_line": 10,
  "end_line": 35,
  "section_path": ["第一章"],
  "prev_chunk_id": "...",
  "next_chunk_id": "..."
}
```

RAPTOR 结果：

```json
{
  "parent_node_id": "...",
  "child_node_ids": ["..."],
  "source_locates": ["xxx.txt:L10-L35"]
}
```

entity/edge 结果：

```json
{
  "source_chunk_ids": ["..."],
  "neighboring_entities": ["..."]
}
```

环境依赖：

- unified graph
- ES 检索结果

独立验证：

```bash
python -m signpost.retrieval.offline_signpost \
  --namespace mini \
  --query "测试问题"
```

旧项目参考：

- `graphrag/retrieval/signpost.py`
- `graphrag/retrieval/kg_retrieval.py`

### F12. 在线路标：PPR 推荐

技术说明对应：

- 第四章在线路标。
- 基于 Personalized PageRank 的关联推荐。

要实现什么：

- 根据当前检索结果选 seed nodes。
- 在裁剪后的子图上运行 PPR。
- 返回 related entities。
- 支持 text seeds 和 graph seeds 两种模式。

环境依赖：

- unified graph
- NetworkX

独立验证：

```bash
python -m signpost.retrieval.online_signpost \
  --graph outputs/mini/graph.unified.json \
  --seed chunk:xxx
```

旧项目参考：

- `graphrag/retrieval/subgraph.py`
- `graphrag/retrieval/signpost.py`
- `graphrag/retrieval/kg_retrieval.py`

注意：

- PPR 先实现可解释版本，不急着优化速度。

### F13. 图检索引擎

技术说明对应：

- 第四章路标驱动检索。
- 第五章图检索引擎。

要实现什么：

一次 query 返回：

```json
{
  "text_group": {
    "items": [],
    "online_signpost": {}
  },
  "graph_group": {
    "items": [],
    "online_signpost": {}
  }
}
```

检索来源：

- chunk
- RAPTOR node
- entity
- edge

环境依赖：

- ES
- unified graph
- ECNU embedding
- 可选 rerank

独立验证：

```bash
python -m signpost.retrieval.run \
  --namespace mini \
  --query "测试问题" \
  --output outputs/mini/retrieval_result.json
```

旧项目参考：

- `graphrag/retrieval/kg_retrieval.py`

### F14. ReadFile / 溯源读取

技术说明对应：

- Agent 根据路标读取原文片段。
- 结果保留文件名和行号引用。

要实现什么：

- 按 `file_name`、`start_line`、`end_line` 读取窗口。
- 支持向前/向后扩展若干行。
- 输出带行号文本。

环境依赖：

- 原始 corpus 或 MinIO。

独立验证：

```bash
python -m signpost.retrieval.read_file \
  --dataset mini \
  --file sample.txt \
  --start-line 10 \
  --end-line 20
```

旧项目参考：

- `api/utils/file_utils.py`
- `deepresearch/tools.py`

注意：

- 不需要旧项目的知识库权限和文件夹逻辑。

### F15. Supervisor-Researcher Agent

技术说明对应：

- 第四章多智能体检索增强框架。

要实现什么：

- Supervisor：拆解问题、分配子研究。
- Researcher：调用 `KnowledgeSearchTool` 和 `ReadFileTool`。
- 记录 trace。
- 输出最终 answer。

环境依赖：

- ECNU chat
- retrieval engine
- read file

独立验证：

```bash
python -m signpost.agent.run \
  --namespace mini \
  --question "测试问题"
```

批量运行：

```bash
python -m signpost.agent.batch \
  --namespace legal \
  --questions datasets/processed/legal/questions.jsonl \
  --output outputs/predictions/signpost/legal.jsonl
```

旧项目参考：

- `deepresearch/agent.py`
- `deepresearch/supervisor.py`
- `deepresearch/researcher.py`
- `deepresearch/tools.py`

注意：

- Agent 必须最后接入。
- 不用 Agent 调试 index 和 retrieval。

### F16. 预测输出与评估适配

技术说明对应：

- 第五章系统级评估。
- LLM-as-Judge。

要实现什么：

- 将 retrieval/agent 输出转成统一 JSONL。
- 兼容现有 `eval/` 所需字段。

预测文件格式：

```json
{
  "question_id": "legal_q_0001",
  "question": "...",
  "answer": "...",
  "rationale": "...",
  "prediction": "...",
  "metadata": {
    "method": "signpost",
    "dataset": "legal"
  }
}
```

环境依赖：

- 如果运行 LLM-as-Judge，则依赖 ECNU/OpenAI-compatible chat。

独立验证：

```bash
python -m signpost.evaluation.validate_predictions \
  --input outputs/predictions/signpost/legal.jsonl
```

旧项目参考：

- `eval/`

## 4. 数据准备流程

### 4.1 当前没有真实数据时

先创建最小样例：

```text
samples/
  mini/
    raw/
      mini.txt
    raw_corpus.jsonl
    documents.jsonl
    questions.jsonl
```

`samples/mini/raw/mini.txt` 示例：

```text
第一章 概述
Signpost 是一种路标机制。
第二章 方法
它使用图结构和 PPR 推荐。
```

`samples/mini/raw_corpus.jsonl` 示例：

```json
{"doc_id":"mini_doc_001","file_name":"mini.txt","source_path":"samples/mini/raw/mini.txt","source_format":"txt","text":null,"metadata":{"dataset":"mini"}}
```

`samples/mini/documents.jsonl` 是文档解析阶段的输出，可以由 F3.5 生成；如果暂时还没实现解析器，也可以手工创建用于后续功能 smoke test：

```json
{"doc_id":"mini_doc_001","file_name":"mini.txt","source_path":"samples/mini/raw/mini.txt","text":"第一章 概述\nSignpost 是一种路标机制。\n第二章 方法\n它使用图结构和 PPR 推荐。","lines":[{"line_no":1,"text":"第一章 概述"},{"line_no":2,"text":"Signpost 是一种路标机制。"},{"line_no":3,"text":"第二章 方法"},{"line_no":4,"text":"它使用图结构和 PPR 推荐。"}],"placeholders":[],"metadata":{"dataset":"mini","source_format":"txt"}}
```

`samples/mini/questions.jsonl` 示例：

```json
{"question_id":"mini_q_001","question":"Signpost 使用什么机制进行推荐？","answer":"它使用图结构和个性化 PageRank 推荐。","rationale":"文档第二章提到它使用图结构和 PPR 推荐。","metadata":{"dataset":"mini"}}
```

所有功能先用 mini 数据跑通。

### 4.2 真实数据整理

每个数据集最终整理成：

```text
datasets/processed/<dataset>/
  raw_corpus.jsonl      # 进入文档解析前
  documents.jsonl       # 文档解析后
  questions.jsonl
```

如果原始数据是文件夹形式：

```text
datasets/raw/legal/
  doc1.txt
  doc2.txt
  Question.jsonl
```

则转换规则：

- 每个 `.txt` 先变成 `raw_corpus.jsonl` 的一行，只登记路径、格式和 metadata。
- F3.5 文档解析阶段再读取 `raw_corpus.jsonl`，生成 `documents.jsonl`。
- `Question.jsonl` 转成统一 `questions.jsonl`。
- 原始字段不确定时，放进 `metadata.raw`。

进入文档解析之前，数据只需要满足：

```text
1. 每个文档有稳定 doc_id。
2. 每个文档有 file_name。
3. 每个文档有 source_path 或 text。
4. 每个文档有 source_format。
5. 每个问题有 question_id、question。
6. 如果有标准答案和推理路径，保留 answer、rationale。
```

也就是说，文档解析之前不需要完成章节识别、切块、embedding 或图构建；只需要把原始语料和问题整理成稳定、可追踪、可重复读取的格式。

### 4.3 需要你后续确认的数据问题

你需要后续帮忙确认：

- 每个数据集原文在哪里。
- 原文是 txt、json、jsonl 还是其他格式。
- 问题文件字段名是什么。
- 标准答案字段名是什么。
- 是否有 `Rationale` 或专家推理路径。
- GraphRAG-Bench 和 UltraDomain 的字段是否一致。

在这些确认前，重构工作可以先基于 `samples/mini` 进行。

## 5. 推荐实现顺序

不要从 API 开始，也不要从 Agent 开始。

推荐顺序：

```text
F0 配置与实验上下文
F1 模型客户端
F2 存储连接
F3 数据标准化
F3.5 文档解析与文本规范化
F4 文档切块与章节路径
F5 Chunk Index
F6 语义视图
F7 结构视图/RAPTOR
F8 顺序视图
F9 多视图统一图
F10 图对象同步 ES
F11 离线路标
F12 在线路标/PPR
F13 图检索引擎
F14 ReadFile/溯源读取
F15 Agent
F16 预测输出与评估适配
```

每一步完成标准：

```text
代码完成
CLI 可运行
mini 样例可跑通
输出格式可检查
失败时错误位置明确
```

## 6. 合作方式

你需要辅助的内容：

1. 确认模型默认选择。

   当前建议：

   ```text
   chat: ecnu-plus
   complex/reasoning: ecnu-max
   embedding: ecnu-embedding-small
   rerank: ecnu-rerank
   ```

2. 后续提供或说明真实数据目录。

3. 告诉我哪些技术说明功能点优先级更高。

我会负责：

- 逐个功能点实现。
- 每个功能点写独立 CLI。
- 每个功能点准备 mini smoke test。
- 避免复制旧项目冗余 API。
- 只从旧项目提炼算法逻辑和必要数据结构。
