# Signpost 重构实现说明

本文档用于记录 `signpost_re` 每个功能点落地后的代码结构、输入输出、验证命令和实现边界。后续每完成一个功能点，都在这里追加对应说明，避免代码结构只存在于实现者脑中。

## 当前状态总览

| 功能点                             | 状态     | 说明                                                                                          |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------- |
| F0 配置与实验上下文                     | 已完成最小版 | 支持 `.env`、YAML 配置读取和 `ExperimentContext`。                                                   |
| F1 模型客户端                        | 已完成最小版 | 支持 ECNU/OpenAI-compatible chat、embedding、rerank 的统一接口；默认 smoke 只检查配置，不发起网络请求。               |
| F2 存储连接                         | 已完成最小版 | 支持 Elasticsearch、MinIO、Redis/Valkey、PostgreSQL 的健康检查入口；不引入业务表和 ORM。                         |
| F3 数据标准化                        | 已完成最小版 | 保留原 `scripts/prepare_datasets.py`，新增包内 CLI 入口和 processed 数据校验。                              |
| F3.5 文档解析与文本规范化                 | 已完成初版  | 支持 txt/md/json/jsonl/jsonl\_context，输出 `documents.jsonl`。                                   |
| F4 文档切块与章节路径                    | 已完成初版  | 支持双路径章节识别接口、确定性章节识别、栈式文档树、树感知两阶段分块、chunk 校验。                                                |
| F5 Chunk Index                  | 已完成初版  | 支持 ECNU/hash embedding、ES chunk index、BM25/dense/hybrid 检索和 smoke 验证。                       |
| F6 语义视图：实体关系图                   | 已完成初版  | 支持 LLM/deterministic 抽取、多轮 gleaning、source mapping、语义边、溯源边、可选统一描述综合。                        |
| F7 结构视图：文档树与 RAPTOR             | 已完成初版  | 支持文档树自底向上摘要、无层级文档的标准 RAPTOR fallback、结构边、摘要节点溯源和 LLM/deterministic 摘要器。                     |
| F8 顺序视图：叙事连续性                   | 已完成初版  | 支持同一文档内相邻 chunk 双向顺序边、文档内位置保留、检索命中前后文扩展。                                                    |
| F9 多视图统一图                       | 已完成初版  | 支持 semantic/structure/sequence 三视图确定性融合、技术说明三类节点和四类边归一化、本地原子持久化、统一图校验。                        |
| F10 图对象同步到 ES                   | 已完成初版  | 支持 entity/relation/RAPTOR summary 图对象适配、embedding、ES bulk 写入、graph search、可选 chunk 父摘要字段更新。 |
| F11 离线路标                        | 已完成初版  | 支持 chunk/summary/entity/relation 结果的纵向、横向、溯源和语义邻居线索附着，支持 query 检索结果增强。                      |
| F12 在线路标：PPR 推荐                 | 已完成初版  | 支持文本场景和图谱场景子图裁剪、Personalized PageRank、组级关联实体推荐和可解释子图统计。                                     |
| F13 图检索引擎                       | 已完成初版  | 支持一次 query 返回 text\_group/graph\_group，融合 chunk、RAPTOR、entity、relation 检索结果，并附着离线/在线路标。     |
| F14 ReadFile / 溯源读取             | 已完成初版  | 支持按 file+line 或 locate 坐标读取源文档片段、前后窗口扩展、带行号文本输出和 JSON 输出。                                   |
| F15 Supervisor-Researcher Agent | 已完成初版  | 支持问题拆解、Researcher 调用 KnowledgeSearch/ReadFile、trace 记录、证据引用答案合成和 batch 预测 JSONL。            |
| F16 预测输出与评估适配                   | 已完成初版  | 支持 agent/retrieval 输出标准化为评估 JSONL、schema 校验、基础 EM/PRF/F1 指标和可选 LLM-as-Judge 适配。             |

## F0 配置与实验上下文

### 目标

F0 的目标是替代旧项目中的用户、租户、知识库上下文，提供科研实验所需的最小上下文：

- `namespace`：一次实验或数据集命名空间。
- `dataset_id`：数据集标识。
- `run_id`：一次运行标识。
- `output_dir`：实验输出目录。

### 对应文件

- `signpost/config/context.py`
  - 定义项目根目录 `PROJECT_ROOT`。
  - 定义 `ExperimentContext`。
  - 提供 `resolve_project_path()`，统一把相对路径解释为 `signpost_re` 下的路径。
- `signpost/config/settings.py`
  - 读取 `.env`。
  - 读取 `conf/service_conf.yaml`。
  - 合并环境变量和配置文件，生成 `Settings`。
- `signpost/config/smoke.py`
  - F0 独立验证入口。

### 输入

- `.env`
- `conf/service_conf.yaml`
- CLI 参数：`--namespace`、`--dataset-id`、`--run-id`

### 输出

`ExperimentContext` 和 `Settings` 对象。smoke CLI 输出 JSON 摘要。

### 验证命令

```bash
conda run -n signpost-re python -m signpost.config.smoke --namespace legal
```

## F1 模型客户端

### 目标

F1 提供科研版统一模型调用接口，先支持 ECNU/OpenAI-compatible API，不迁移旧项目多 provider 历史包袱。

### 对应文件

- `signpost/llm/client.py`
  - 定义 `LLMConfig`。
  - 定义 `OpenAICompatibleClient`。
  - 实现 `chat()`、`embedding()`、`rerank()` 三类接口。
  - 通过 `urllib` 调用 HTTP，避免第一阶段引入额外 SDK 复杂度。
- `signpost/llm/smoke.py`
  - 默认只检查配置，不调用外部 API。
  - 显式传入 `--chat`、`--embedding` 或 `--rerank` 时才发起真实请求。

### 输入

- `.env` 中的 `ECNU_API_BASE`、`ECNU_API_KEY`、模型名。
- messages、texts、query/doc pairs。

### 输出

- chat 文本。
- embedding 向量列表。
- rerank 分数列表。

### 验证命令

```bash
conda run -n signpost-re python -m signpost.llm.smoke
```

真实调用需要有效 API key：

```bash
conda run -n signpost-re python -m signpost.llm.smoke --chat
```

## F2 存储连接

### 目标

F2 提供外部服务健康检查和最小连接封装，不迁移旧 ORM、用户表、Canvas、API token 等产品逻辑。

### 对应文件

- `signpost/storage/health.py`
  - Elasticsearch：HTTP GET `/`。
  - MinIO：HTTP GET `/minio/health/live`。
  - Redis/Valkey：TCP socket 发送 RESP `PING`。
  - PostgreSQL：第一阶段做 TCP 连通性检查。
- `signpost/storage/smoke.py`
  - F2 独立验证入口。

### 输入

- `.env`
- `conf/service_conf.yaml`

### 输出

服务健康检查 JSON 摘要。

### 验证命令

```bash
conda run -n signpost-re python -m signpost.storage.smoke --all
```

## F3 数据标准化

### 目标

F3 的目标是把原始数据整理成文档解析前的统一格式，不做章节识别、切块、embedding 或图构建。

### 对应文件

- `scripts/prepare_datasets.py`
  - 已有数据下载和标准化脚本。
  - 继续保留，方便独立执行。
- `signpost/data/prepare.py`
  - 包内入口，复用 `scripts/prepare_datasets.py`。
  - 使计划文档中的 `python -m signpost.data.prepare` 可运行。
- `signpost/data/validate.py`
  - 校验 `datasets/processed/<dataset>/raw_corpus.jsonl` 和 `questions.jsonl`。

### 输入格式

`raw_corpus.jsonl` 每行示例：

```json
{"doc_id":"legal_doc_001","file_name":"xxx.txt","source_path":"datasets/raw/legal/xxx.txt","source_format":"txt","text":null,"metadata":{"dataset":"legal"}}
```

`questions.jsonl` 每行示例：

```json
{"question_id":"legal_q_0001","question":"问题文本","answer":"标准答案","rationale":"推理路径","metadata":{"dataset":"legal"}}
```

### 输出

- `datasets/processed/<dataset>/raw_corpus.jsonl`
- `datasets/processed/<dataset>/questions.jsonl`
- `datasets/manifest.json`

### 验证命令

```bash
conda run -n signpost-re python -m signpost.data.validate --dataset legal
```

## F3.5 文档解析与文本规范化

### 目标

F3.5 把 `raw_corpus.jsonl` 变成后续章节识别、chunking、溯源读取可用的 `documents.jsonl`。

本阶段解决的问题：

- 从 `text` 或 `source_path` 读取正文。
- 支持 `txt`、`md`、`json`、`jsonl`、`jsonl_context`。
- 做 Unicode NFKC 规范化。
- 标准化常见空白、引号、破折号、省略号。
- 过滤空行，但保留原始 `line_no`。
- 扫描 Markdown 图片、Markdown 链接、HTML table，占位符进入 `placeholders`。

### 对应文件

- `signpost/parsing/io.py`
  - JSONL 读写工具。
- `signpost/parsing/normalizer.py`
  - 文本和单行规范化。
- `signpost/parsing/parser.py`
  - F3.5 核心解析逻辑。
- `signpost/parsing/parse_documents.py`
  - CLI：`raw_corpus.jsonl -> documents.jsonl`。
- `signpost/parsing/validate_documents.py`
  - CLI：校验 `documents.jsonl`。
- `samples/mini/`
  - mini 样例输入。
- `tests/test_parsing.py`
  - mini smoke test。

### 输入格式

输入文件为 `raw_corpus.jsonl`，每行必须至少包含：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "source_path": "samples/mini/raw/mini.txt",
  "source_format": "txt",
  "text": null,
  "metadata": {"dataset": "mini"}
}
```

如果 `text` 非空，解析器优先使用 `text`；否则读取 `source_path`。

### 输出格式

输出文件为 `documents.jsonl`，每行格式：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "source_path": "samples/mini/raw/mini.txt",
  "text": "规范化后的全文",
  "lines": [
    {"line_no": 1, "text": "第一章 概述"}
  ],
  "placeholders": [
    {"placeholder": "[image_5_1]", "type": "image", "line_no": 5, "raw": "![示意图](figure.png)"}
  ],
  "metadata": {"dataset": "mini", "source_format": "txt"}
}
```

### 已完成功能

- mini 样例解析通过。
- `agriculture` 真实数据解析通过。
- 输出中保留 `line_no`，为空行过滤后的行号仍对应原始文件行号。
- 输出中保留 `metadata`，后续 chunking 和评估可以继续追踪数据来源。

### 验证命令

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input samples/mini/raw_corpus.jsonl \
  --output outputs/mini/documents.jsonl

conda run -n signpost-re python -m signpost.parsing.validate_documents \
  --input outputs/mini/documents.jsonl

conda run -n signpost-re python -m pytest tests/test_parsing.py
```

## F4 文档切块与章节路径

### 目标

F4 将 F3.5 的 `documents.jsonl` 转换成后续 embedding、结构视图、顺序视图和溯源读取可使用的 `chunks.jsonl`，同时输出 `document_trees.jsonl` 保存章节树。

技术说明中涉及的能力在本阶段对应为：

- 双路径章节层次识别：
  - 短文档：端到端 Markdown 转换接口。
  - 长文档：迭代式窗口抽取接口，支持周期性历史压缩。
  - 默认运行使用确定性章节识别，避免 smoke test 消耗模型额度；传 `--use-llm` 时走 LLM 路径。
- 栈式文档树构建：
  - 根据标题 level 构建父子层级。
  - 每个节点保留 `start_line`、`end_line` 和 `section_path`。
- 基于文档树的两阶段分块：
  - 阶段一：如果子树 token 数低于预算，则合并为一个 chunk。
  - 阶段二：超预算节点按行边界二次切分，保留 overlap。
  - 边界 fallback：如果数据集把很长的原文压成单个逻辑行，且这一行本身超过 token 预算，则在该行内部按词继续切分，`merge` 标记为 `split_long_line`。该 fallback 不改变章节识别和文档树，只保证技术说明中的 token 预算约束在异常长行数据上仍然生效。
- 章节路径编码：
  - 每个 chunk 的 `section_path` 写入字段。
  - chunk 内容前追加章节路径，方便脱离树结构后仍保留层次位置。
- 顺序关系：
  - 每个 chunk 写入 `prev_chunk_id` 和 `next_chunk_id`，为 F8 顺序视图做准备。

### 对应文件

- `signpost/chunking/models.py`
  - 定义 `Header`、`TreeNode`、`Chunk`。
- `signpost/chunking/tokenizer.py`
  - 轻量 token 计数器，用于 chunk 预算和 overlap。
- `signpost/chunking/headers.py`
  - 章节识别。
  - 包含确定性识别、短文档 Markdown LLM 路径、长文档迭代 LLM 路径。
- `signpost/chunking/tree.py`
  - 栈式文档树构建。
- `signpost/chunking/chunker.py`
  - 文档树两阶段分块。
  - 处理超长单行 fallback，避免单个 chunk 超过 embedding 模型输入上限。
  - 生成 chunk id、prev/next 链接和 tree payload。
- `signpost/chunking/run.py`
  - CLI：`documents.jsonl -> chunks.jsonl + document_trees.jsonl`。
- `signpost/chunking/validate.py`
  - CLI：校验 `chunks.jsonl`。
- `tests/test_chunking.py`
  - mini smoke test。

### 输入格式

输入为 F3.5 的 `documents.jsonl`，每行至少包含：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "text": "第一章 概述\n...",
  "lines": [
    {"line_no": 1, "text": "第一章 概述"}
  ],
  "metadata": {"dataset": "mini"}
}
```

### 输出格式

`chunks.jsonl` 每行格式：

```json
{
  "chunk_id": "mini_doc_001_c00000",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "第一章 概述\n\n[CONTENT]\n\n第一章 概述\nSignpost 是一种路标机制。",
  "start_line": 1,
  "end_line": 3,
  "section_path": ["第一章 概述"],
  "prev_chunk_id": null,
  "next_chunk_id": "mini_doc_001_c00001",
  "metadata": {
    "merge": "subtree",
    "chunk_index": 0,
    "token_count": 22
  }
}
```

`document_trees.jsonl` 每行格式：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "headers": [
    {"title": "第一章 概述", "level": 1, "content_start": 1, "content_end": 3}
  ],
  "tree": {
    "title": "[ROOT]",
    "level": 0,
    "children": []
  }
}
```

说明：

- `document_trees.jsonl` 由 headers 和栈式树构建算法生成，不受 `split_long_line` fallback 影响。
- `chunks.jsonl` 的 chunk 边界优先由文档树决定；只有章节子树或节点超出 `--max-tokens` 时才继续切分。
- 如果原始数据中整篇文章被写成一行，例如 GraphRAG-Bench novel/medical 中的部分样本，旧版本可能产生几万 token 的超长 chunk。当前版本会把这种单行继续拆成多个小 chunk，但这些 chunk 仍保留原始 `file_name`、`start_line`、`end_line` 和 `section_path`。

### 验证命令

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input outputs/mini/documents.jsonl \
  --output outputs/mini/chunks.jsonl \
  --tree-output outputs/mini/document_trees.jsonl \
  --max-tokens 1200 \
  --overlap-tokens 100

conda run -n signpost-re python -m signpost.chunking.validate \
  --chunks outputs/mini/chunks.jsonl

conda run -n signpost-re python -m pytest tests/test_chunking.py
```

## F5 Chunk Index

### 目标

F5 将 F4 输出的 `chunks.jsonl` 写入 Elasticsearch，形成基础文本检索和向量检索能力。该阶段是后续 GraphRAG 语义图构建、结构图同步、图检索和 Agent 检索的底座。

技术说明中涉及的能力在本阶段对应为：

- 对 chunk 生成 embedding。
- 在 ES 中建立 chunk 级索引。
- 保留 namespace/dataset/doc/chunk/line/section\_path 等定位字段。
- 支持 BM25 文本检索。
- 支持 dense vector 检索。
- 支持 hybrid 检索，当前使用 Reciprocal Rank Fusion 合并 BM25 和 dense 结果。
- 结果返回 prev/next chunk 和章节路径，为离线路标与顺序视图提供输入。

### 对应文件

- `signpost/storage/elasticsearch.py`
  - 最小 ES HTTP client。
  - 支持 index 创建、bulk 写入、refresh 和 JSON 请求。
- `signpost/indexing/embedding.py`
  - `ECNUEmbeddingProvider`：生产路径，调用 ECNU/OpenAI-compatible embedding。
  - `HashEmbeddingProvider`：本地确定性 embedding，用于 smoke test 和不消耗模型额度的管线测试。
- `signpost/indexing/chunk_schema.py`
  - ES index name、mapping 和 chunk 文档转换。
- `signpost/indexing/chunk_index.py`
  - CLI：读取 `chunks.jsonl`，生成 embedding，写入 ES。
- `signpost/retrieval/chunk_search.py`
  - CLI：BM25、dense、hybrid chunk 检索。
- `tests/test_chunk_index.py`
  - mapping、索引命名和 hash embedding 的单元测试。

### 输入格式

输入为 F4 的 `chunks.jsonl`：

```json
{
  "chunk_id": "mini_doc_001_c00000",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "章节路径\n\n[CONTENT]\n\n正文",
  "start_line": 1,
  "end_line": 3,
  "section_path": ["第一章 概述"],
  "prev_chunk_id": null,
  "next_chunk_id": "mini_doc_001_c00001",
  "metadata": {"chunk_index": 0, "token_count": 22}
}
```

### ES 文档格式

写入 ES 的文档核心字段：

```json
{
  "id": "mini_doc_001_c00000",
  "type": "chunk",
  "namespace": "mini",
  "dataset_id": "mini",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "...",
  "content_vector": [0.1, 0.2],
  "start_line": 1,
  "end_line": 3,
  "section_path": ["第一章 概述"],
  "prev_chunk_id": null,
  "next_chunk_id": "mini_doc_001_c00001",
  "chunk_index": 0,
  "token_count": 22,
  "metadata": {}
}
```

### 输出

- Elasticsearch index：默认命名为 `signpost-<namespace>-chunks`。
- 搜索结果 JSON，包含 chunk id、doc id、内容、行号、章节路径、prev/next 指针和 score。

### 验证命令

本地不消耗模型额度的 smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl \
  --embedding-provider hash \
  --recreate

conda run -n signpost-re python -m signpost.retrieval.chunk_search \
  --namespace mini \
  --query "PPR 推荐" \
  --embedding-provider hash \
  --mode hybrid
```

ECNU 真实 embedding smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace mini-ecnu \
  --chunks outputs/mini/chunks.jsonl \
  --embedding-provider ecnu \
  --batch-size 8 \
  --progress-every 50 \
  --embedding-retries 3 \
  --recreate

conda run -n signpost-re python -m signpost.retrieval.chunk_search \
  --namespace mini-ecnu \
  --query "PPR 推荐" \
  --embedding-provider ecnu \
  --mode hybrid
```

真实数据集使用 ECNU embedding 前，必须确认对应 `chunks.jsonl` 是当前 F4 版本生成的，尤其是 GraphRAG-Bench 这类可能把整篇文本压成单行的数据。推荐先重新跑 F4：

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/<dataset>/documents.jsonl \
  --output datasets/processed/<dataset>/chunks.jsonl \
  --tree-output datasets/processed/<dataset>/document_trees.jsonl \
  --max-tokens 1200 \
  --overlap-tokens 100
```

再跑 F5：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace <dataset>-ecnu \
  --chunks datasets/processed/<dataset>/chunks.jsonl \
  --embedding-provider ecnu \
  --batch-size 4 \
  --progress-every 50 \
  --embedding-retries 3 \
  --recreate
```

说明：ECNU embedding 是远程批量调用，真实数据集可能需要数百次请求。`chunk_index` 默认每 50 个 batch 输出一次进度；如果希望更频繁，可以设为 `--progress-every 10`。如果设置为 `0`，则只在结束时输出。远程接口偶发 HTTP 500 或 SSL EOF 时会按 `--embedding-retries` 重试；整批持续失败时会自动拆成更小 batch，再继续索引。GraphRAG-Bench 这类长文本数据建议从 `--batch-size 4` 开始，稳定后再调大。

如果 embedding 阶段报错包含 `chars` 和 `tokens` 很大的 chunk，例如几万 tokens，说明使用的是旧切块产物或当前 `--max-tokens` 不适合该 embedding 模型，需要先重新生成 F4 chunks。

真实数据结构 smoke，不消耗模型额度：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace graphrag-bench-medical-smoke \
  --chunks datasets/processed/graphrag-bench-medical/chunks.jsonl \
  --embedding-provider hash \
  --recreate
```

## F6 语义视图：实体关系图

### 目标

F6 从 F4 的 chunks 中抽取实体与实体关系，构建语义视图。该语义图用于建立跨文档概念关联，后续 F9 会与结构视图和顺序视图合并成统一图，F11/F12 会基于其中的实体、关系和溯源边生成路标。

技术说明中涉及的能力在本阶段对应为：

- 每个 chunk 独立调用 LLM 抽取实体和关系。
- 实体记录名称、类型、描述。
- 关系记录源实体、目标实体、描述、关系关键词和权重。
- 支持多轮补充抽取（gleaning），默认最多 2 轮。
- 保留逐来源证据字典 `source_mapping`，key 为 `doc_id:chunk_id`。
- 同名实体聚合为一个实体节点。
- 同一实体对的关系边按无向规范键合并。
- 关系端点缺失时自动创建占位实体节点。
- 创建实体到 chunk 的溯源边 `edge_type=source`。
- 支持可选 LLM 统一描述综合 `--synthesize-descriptions`。

### 对应文件

- `signpost/indexing/semantic_extractor.py`
  - `LLMSemanticExtractor`：生产路径，调用 ECNU/OpenAI-compatible chat。
  - `DeterministicSemanticExtractor`：本地 smoke 路径，不消耗模型额度。
  - `parse_extraction_response()`：解析 LLM JSON 输出。
- `signpost/graph/semantic.py`
  - 将逐 chunk 抽取结果合并成语义图。
  - 生成 entity 节点、chunk 节点、semantic\_relation 边和 source 边。
  - 保留 `source_mapping`、`source_chunk_ids`、`source_locates`。
- `signpost/indexing/semantic_graph.py`
  - CLI：`chunks.jsonl -> graph.semantic.json`。
- `signpost/graph/validate.py`
  - 通用 graph JSON 校验。
- `signpost/graph/inspect.py`
  - 图摘要查看。
- `tests/test_semantic_graph.py`
  - mini semantic graph smoke test。

### 输入格式

输入为 F4 的 `chunks.jsonl`：

```json
{
  "chunk_id": "mini_doc_001_c00001",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "第二章 方法\n\n[CONTENT]\n\n它使用图结构和 PPR 推荐。",
  "start_line": 4,
  "end_line": 6,
  "section_path": ["第二章 方法"]
}
```

### 输出格式

输出为 `graph.semantic.json`：

```json
{
  "metadata": {
    "namespace": "mini",
    "graph_type": "semantic",
    "chunks": 2,
    "entities": 10,
    "relations": 8,
    "source_edges": 10
  },
  "nodes": [
    {
      "node_id": "entity:be6d241ec3a7",
      "node_type": "entity",
      "name": "第一章",
      "entity_type": "CONCEPT",
      "description": "...",
      "source_chunk_ids": ["mini_doc_001_c00000"],
      "source_locates": ["mini.txt:L1-L3"],
      "source_mapping": {
        "mini_doc_001:mini_doc_001_c00000": {
          "description": "...",
          "entity_type": "CONCEPT",
          "file_name": "mini.txt",
          "start_line": 1,
          "end_line": 3
        }
      }
    }
  ],
  "edges": [
    {
      "source": "entity:...",
      "target": "entity:...",
      "edge_type": "semantic_relation",
      "relation_types": ["co_occurs"],
      "weight": 1.0,
      "source_chunk_ids": ["mini_doc_001_c00000"]
    },
    {
      "source": "entity:...",
      "target": "chunk:mini_doc_001_c00000",
      "edge_type": "source",
      "source_locates": ["mini.txt:L1-L3"]
    }
  ]
}
```

### 验证命令

本地 deterministic smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl \
  --output outputs/mini/graph.semantic.json \
  --extractor deterministic

conda run -n signpost-re python -m signpost.graph.validate \
  --graph outputs/mini/graph.semantic.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph outputs/mini/graph.semantic.json
```

LLM extraction smoke：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace mini-llm \
  --chunks outputs/mini/chunks.jsonl \
  --output outputs/mini/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --progress-every 1 \
  --progress-file outputs/mini/semantic_llm.progress.jsonl \
  --extractions-cache outputs/mini/semantic_llm.extractions.jsonl \
  --llm-retries 3 \
  --llm-timeout 120 \
  --max-chunks 1
```

说明：`--max-chunks 1` 是 smoke test 限制，只处理输入 `chunks.jsonl` 的前 1 个 chunk，用于确认 LLM 抽取链路、JSON 解析和图文件写出正常。输出中 `chunks=1` 正是这个参数的预期结果，不代表数据集只有 1 个 chunk。

全量 LLM 抽取时移除 `--max-chunks`：

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

注意：全量 `--extractor llm` 会对每个 chunk 调用 LLM，成本和耗时会随 chunk 数线性增长。建议先用 `--max-chunks 1`、`--max-chunks 10` 逐步 smoke，确认输出质量后再全量运行。`--gleaning-rounds 1` 表示每个 chunk 最多执行“首轮抽取 + 1 轮补充抽取”，因此 LLM 调用次数最多约为 `chunks * 2`。远程 chat 超时或断连时会按 `--llm-retries` 重试。使用 `conda run` 跑长任务时建议加 `--no-capture-output`，否则 conda 可能缓存输出，终端看不到实时进度；同时可以用 `tail -f <progress-file>` 查看 JSONL 进度文件。`--extractions-cache` 会在每个 chunk 抽取完成后立即追加一行结果，任务中断后重跑同一命令会跳过已缓存 chunk，再合并生成最终 `graph.semantic.json`。

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace graphrag-bench-medical-smoke \
  --chunks datasets/processed/graphrag-bench-medical/chunks.jsonl \
  --output datasets/processed/graphrag-bench-medical/graph.semantic.json \
  --extractor deterministic \
  --max-chunks 10
```

## F7 结构视图：文档树与 RAPTOR

### 目标

F7 从 F4 的 `chunks.jsonl` 和 `document_trees.jsonl` 构建结构视图。该视图对应技术说明中的文档层级结构与层级摘要，用于表达章节、子章节、chunk 之间的父子关系，并为后续统一图融合、结构检索和路标生成提供可追溯的摘要节点。

技术说明中涉及的能力在本阶段对应为：

- 优先使用 F4 生成的文档树，自底向上生成章节摘要。
- 如果文档没有可识别章节树，回退到标准 RAPTOR 风格的递归聚合。
- RAPTOR fallback 按 chunk 顺序和 token budget 分组；当超长 chunk 导致分组退化时，强制相邻节点配对继续形成多层结构。
- 生成 `node_type=raptor` 的摘要节点，记录 `level`、`parent_node_id`、`child_node_ids`。
- 生成 `edge_type=structure` 的结构边，边方向为父摘要节点指向子摘要节点或 chunk 节点。
- 每个摘要节点保留 `source_chunk_ids` 和 `source_locates`，支持回溯到原文 chunk 与行号。
- 支持 `deterministic` 摘要器做本地 smoke，也支持 `llm` 摘要器调用 ECNU/OpenAI-compatible chat。

### 对应文件

- `signpost/indexing/summarizer.py`
  - `Summarizer` 协议定义。
  - `DeterministicSummarizer`：本地摘要器，截断并拼接输入文本，不消耗模型额度。
  - `LLMSummarizer`：生产摘要器，调用 F1 模型客户端，要求模型返回 JSON 标题和摘要正文。
  - `create_summarizer()`：CLI 使用的摘要器工厂。
- `signpost/graph/structure.py`
  - F7 核心构图逻辑。
  - `build_structure_graph()`：入口函数，合并 chunk 节点、RAPTOR 节点和结构边。
  - 文档树路径：对章节树递归生成摘要，并在多顶层章节时补充文档根摘要。
  - RAPTOR fallback 路径：对无章节结构文档递归分组、摘要、连边。
- `signpost/indexing/structure_graph.py`
  - CLI：`chunks.jsonl + document_trees.jsonl -> graph.structure.json`。
  - 支持 `--summarizer deterministic|llm`、`--max-chunks`、`--cluster-token-budget` 等参数。
- `signpost/graph/validate.py`
  - 通用 graph JSON 校验，可验证 structure graph 的节点和边引用。
- `signpost/graph/inspect.py`
  - 查看 structure graph 的 metadata、节点类型计数和边类型计数。
- `tests/test_structure_graph.py`
  - mini structure graph smoke test，覆盖文档树摘要路径。

### 输入格式

输入一：F4 输出的 `chunks.jsonl`，每行一个 chunk：

```json
{
  "chunk_id": "mini_doc_001_c00001",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "第二章 方法\n\n[CONTENT]\n\n它使用图结构和 PPR 推荐。",
  "start_line": 4,
  "end_line": 6,
  "section_path": ["第二章 方法"]
}
```

输入二：F4 输出的 `document_trees.jsonl`，每行一棵文档树：

```json
{
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "tree": {
    "title": "mini.txt",
    "children": [
      {
        "title": "第二章 方法",
        "level": 1,
        "start_line": 4,
        "end_line": 6,
        "section_path": ["第二章 方法"],
        "children": []
      }
    ]
  }
}
```

### 输出格式

输出为 `graph.structure.json`：

```json
{
  "metadata": {
    "namespace": "mini",
    "graph_type": "structure",
    "chunks": 2,
    "raptor_nodes": 3,
    "structure_edges": 4
  },
  "nodes": [
    {
      "node_id": "chunk:mini_doc_001_c00001",
      "node_type": "chunk",
      "chunk_id": "mini_doc_001_c00001",
      "doc_id": "mini_doc_001",
      "file_name": "mini.txt",
      "start_line": 4,
      "end_line": 6,
      "section_path": ["第二章 方法"]
    },
    {
      "node_id": "raptor:...",
      "node_type": "raptor",
      "title": "第二章 方法",
      "content": "章节摘要正文",
      "level": 1,
      "parent_node_id": "raptor:...",
      "child_node_ids": ["chunk:mini_doc_001_c00001"],
      "source_chunk_ids": ["mini_doc_001_c00001"],
      "source_locates": ["mini.txt:L4-L6"],
      "section_path": ["第二章 方法"],
      "metadata": {"mode": "document_tree"}
    }
  ],
  "edges": [
    {
      "source": "raptor:...",
      "target": "chunk:mini_doc_001_c00001",
      "edge_type": "structure"
    }
  ]
}
```

### 验证命令

本地 deterministic smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl \
  --document-trees outputs/mini/document_trees.jsonl \
  --output outputs/mini/graph.structure.json \
  --summarizer deterministic

conda run -n signpost-re python -m signpost.graph.validate \
  --graph outputs/mini/graph.structure.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph outputs/mini/graph.structure.json
```

LLM summary smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace mini-llm \
  --chunks outputs/mini/chunks.jsonl \
  --document-trees outputs/mini/document_trees.jsonl \
  --output outputs/mini/graph.structure.llm.json \
  --summarizer llm \
  --max-summary-tokens 256
```

无章节结构数据集的 RAPTOR fallback smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace graphrag-bench-medical-smoke \
  --chunks datasets/processed/graphrag-bench-medical/chunks.jsonl \
  --document-trees datasets/processed/graphrag-bench-medical/document_trees.jsonl \
  --output datasets/processed/graphrag-bench-medical/graph.structure.json \
  --summarizer deterministic \
  --max-chunks 10 \
  --cluster-token-budget 2400
```

## F8 顺序视图：叙事连续性

### 目标

F8 从 F4 的 `chunks.jsonl` 构建顺序视图。该视图对应技术说明 3.5 节的 `Eseq`：在同一文档内，按照 chunk 的行号位置建立相邻文本块之间的双向边，保持原始叙事顺序，并支持检索命中后的上下文补全。

技术说明中涉及的能力在本阶段对应为：

- 仅在同一文档内部建立顺序边，不跨文档连接。
- 同一文档内按 `start_line`、`end_line` 和 `chunk_id` 稳定排序。
- 相邻 chunk 之间建立双向 `edge_type=sequence` 边。
- 边上记录 `direction=next|prev`，用于区分向后阅读和向前回溯。
- chunk 节点保留 `doc_position`、`doc_chunk_count`、`prev_chunk_id`、`next_chunk_id`。
- 提供 `expand_sequence_context()` 和 CLI，可围绕检索命中 chunk 向前/向后扩展上下文窗口。

### 对应文件

- `signpost/graph/sequence.py`
  - F8 核心构图逻辑。
  - `build_sequence_graph()`：读取 chunk 列表，生成 chunk 节点和双向顺序边。
  - `expand_sequence_context()`：基于 sequence graph 对命中 chunk 做前后文扩展。
- `signpost/indexing/sequence_graph.py`
  - CLI：`chunks.jsonl -> graph.sequence.json`。
  - 支持 `--namespace`、`--chunks`、`--output`、`--max-chunks`。
- `signpost/retrieval/sequence_context.py`
  - CLI：围绕一个或多个 `--chunk-id` 展开上下文。
  - 输出按文档位置排序的 chunk 节点列表，并标记 `hop_from_seed`。
- `tests/test_sequence_graph.py`
  - mini sequence graph smoke test。
  - 覆盖双向边数量、图校验和上下文扩展。

### 输入格式

输入为 F4 输出的 `chunks.jsonl`：

```json
{
  "chunk_id": "mini_doc_001_c00001",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "content": "第二章 方法\n\n[CONTENT]\n\n它使用图结构和 PPR 推荐。",
  "start_line": 4,
  "end_line": 6,
  "section_path": ["第二章 方法"],
  "prev_chunk_id": "mini_doc_001_c00000",
  "next_chunk_id": null,
  "metadata": {"chunk_index": 1, "token_count": 35}
}
```

### 输出格式

输出为 `graph.sequence.json`：

```json
{
  "metadata": {
    "namespace": "mini",
    "graph_type": "sequence",
    "chunks": 2,
    "documents": 1,
    "sequence_edges": 2
  },
  "nodes": [
    {
      "node_id": "chunk:mini_doc_001_c00001",
      "node_type": "chunk",
      "chunk_id": "mini_doc_001_c00001",
      "doc_id": "mini_doc_001",
      "file_name": "mini.txt",
      "content": "第二章 方法...",
      "start_line": 4,
      "end_line": 6,
      "section_path": ["第二章 方法"],
      "doc_position": 1,
      "doc_chunk_count": 2,
      "prev_chunk_id": "mini_doc_001_c00000",
      "next_chunk_id": null
    }
  ],
  "edges": [
    {
      "source": "chunk:mini_doc_001_c00000",
      "target": "chunk:mini_doc_001_c00001",
      "edge_type": "sequence",
      "direction": "next",
      "doc_id": "mini_doc_001",
      "source_chunk_id": "mini_doc_001_c00000",
      "target_chunk_id": "mini_doc_001_c00001",
      "source_locate": "mini.txt:L1-L3",
      "target_locate": "mini.txt:L4-L6",
      "distance": 1
    }
  ]
}
```

上下文扩展输出为 chunk 节点列表，并额外带有 `hop_from_seed`：

```json
[
  {
    "chunk_id": "mini_doc_001_c00000",
    "doc_position": 0,
    "hop_from_seed": -1
  },
  {
    "chunk_id": "mini_doc_001_c00001",
    "doc_position": 1,
    "hop_from_seed": 0
  }
]
```

### 验证命令

本地 mini sequence graph smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.sequence_graph \
  --namespace mini \
  --chunks outputs/mini/chunks.jsonl \
  --output outputs/mini/graph.sequence.json

conda run -n signpost-re python -m signpost.graph.validate \
  --graph outputs/mini/graph.sequence.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph outputs/mini/graph.sequence.json
```

上下文扩展 smoke：

```bash
conda run -n signpost-re python -m signpost.retrieval.sequence_context \
  --graph outputs/mini/graph.sequence.json \
  --chunk-id mini_doc_001_c00001 \
  --before 1 \
  --after 1
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.sequence_graph \
  --namespace graphrag-bench-medical-smoke \
  --chunks datasets/processed/graphrag-bench-medical/chunks.jsonl \
  --output datasets/processed/graphrag-bench-medical/graph.sequence.json \
  --max-chunks 10
```

## F9 多视图统一图

### 目标

F9 将 F6 语义视图、F7 结构视图和 F8 顺序视图融合为统一的多视图拓扑图，对应技术说明 3.6 节的形式化定义：

```text
V = Vchunk ∪ Vsummary ∪ Ventity
E = Estruct ∪ Esem ∪ Eseq ∪ Esource
```

本阶段解决的问题：

- 合并 `graph.semantic.json`、`graph.structure.json`、`graph.sequence.json`。
- 将节点类型归一化为 `chunk`、`summary`、`entity`。
- 将边类型归一化为 `structure`、`semantic`、`sequence`、`source`。
- 对来自多个视图的同一 chunk 节点做属性合并，保留 `views` 字段。
- 对重复边做确定性去重。
- 保留原始类型字段，例如 RAPTOR 节点会保留 `original_node_type=raptor` 和 `summary_type=raptor`。
- 支持本地 JSON 原子写入、加载和统一图校验。

### 对应文件

- `signpost/graph/unified.py`
  - F9 核心 merge 逻辑。
  - `merge_graphs()`：合并三视图图对象。
  - `validate_unified_graph()`：校验统一图的三类节点、四类边和边引用。
  - `save_graph_atomic()` / `load_graph()`：统一图本地 JSON 持久化。
- `signpost/graph/merge.py`
  - CLI：`semantic + structure + sequence -> graph.unified.json`。
- `signpost/graph/repository.py`
  - `LocalGraphRepository`：科研版本地图存储封装。
  - 保留旧项目“统一图持久化”的有用语义，但不迁移用户、租户、知识库等业务逻辑。
- `tests/test_unified_graph.py`
  - mini 三视图 merge smoke test。
  - 覆盖 chunk/summary/entity 节点、四类边和通用图校验。

### 输入格式

输入一：F6 输出的 `graph.semantic.json`：

```json
{
  "metadata": {"graph_type": "semantic"},
  "nodes": [
    {"node_id": "chunk:mini_doc_001_c00000", "node_type": "chunk"},
    {"node_id": "entity:...", "node_type": "entity", "source_chunk_ids": ["mini_doc_001_c00000"]}
  ],
  "edges": [
    {"source": "entity:...", "target": "entity:...", "edge_type": "semantic_relation"},
    {"source": "entity:...", "target": "chunk:mini_doc_001_c00000", "edge_type": "source"}
  ]
}
```

输入二：F7 输出的 `graph.structure.json`：

```json
{
  "metadata": {"graph_type": "structure"},
  "nodes": [
    {"node_id": "raptor:...", "node_type": "raptor", "source_chunk_ids": ["mini_doc_001_c00000"]}
  ],
  "edges": [
    {"source": "raptor:...", "target": "chunk:mini_doc_001_c00000", "edge_type": "structure"}
  ]
}
```

输入三：F8 输出的 `graph.sequence.json`：

```json
{
  "metadata": {"graph_type": "sequence"},
  "nodes": [
    {"node_id": "chunk:mini_doc_001_c00000", "node_type": "chunk", "doc_position": 0}
  ],
  "edges": [
    {"source": "chunk:mini_doc_001_c00000", "target": "chunk:mini_doc_001_c00001", "edge_type": "sequence", "direction": "next"}
  ]
}
```

### 输出格式

输出为 `graph.unified.json`：

```json
{
  "metadata": {
    "namespace": "mini",
    "graph_type": "unified",
    "views": ["semantic", "structure", "sequence"],
    "nodes": 15,
    "edges": 24,
    "chunk_nodes": 2,
    "summary_nodes": 3,
    "entity_nodes": 10,
    "structure_edges": 4,
    "semantic_edges": 8,
    "sequence_edges": 2,
    "source_edges": 10
  },
  "nodes": [
    {
      "node_id": "chunk:mini_doc_001_c00000",
      "node_type": "chunk",
      "views": ["semantic", "structure", "sequence"]
    },
    {
      "node_id": "raptor:...",
      "node_type": "summary",
      "original_node_type": "raptor",
      "summary_type": "raptor",
      "views": ["structure"]
    }
  ],
  "edges": [
    {
      "source": "raptor:...",
      "target": "chunk:mini_doc_001_c00000",
      "edge_type": "structure",
      "views": ["structure"]
    },
    {
      "source": "entity:...",
      "target": "entity:...",
      "edge_type": "semantic",
      "original_edge_type": "semantic_relation",
      "views": ["semantic"]
    }
  ]
}
```

### Merge 规则

- 节点按 `node_id` 合并。
- 列表字段去重合并，例如 `views`、`source_chunk_ids`、`source_locates`。
- 字典字段递归合并，例如 `source_mapping`、`metadata`。
- chunk 节点优先保留 F8 的 `content`、`doc_position`、`prev_chunk_id`、`next_chunk_id`，同时合并 F6/F7 的行号、章节路径和来源字段。
- `semantic_relation` 边归一化为 `semantic`，并保留 `original_edge_type`。
- `raptor` 节点归一化为 `summary`，并保留 `summary_type=raptor`。
- `semantic` 边按无向实体对去重；`sequence` 边保留方向；`structure` 和 `source` 边按有向边去重。

### 验证命令

mini 三视图 merge：

```bash
conda run -n signpost-re python -m signpost.graph.merge \
  --namespace mini \
  --semantic outputs/mini/graph.semantic.json \
  --structure outputs/mini/graph.structure.json \
  --sequence outputs/mini/graph.sequence.json \
  --output outputs/mini/graph.unified.json

conda run -n signpost-re python -m signpost.graph.validate \
  --graph outputs/mini/graph.unified.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph outputs/mini/graph.unified.json
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.graph.merge \
  --namespace graphrag-bench-medical-smoke \
  --semantic datasets/processed/graphrag-bench-medical/graph.semantic.json \
  --structure datasets/processed/graphrag-bench-medical/graph.structure.json \
  --sequence datasets/processed/graphrag-bench-medical/graph.sequence.json \
  --output datasets/processed/graphrag-bench-medical/graph.unified.json
```

## F10 图对象同步到 ES

### 目标

F10 将 F9 的 `graph.unified.json` 中可检索的图对象同步到 Elasticsearch。该阶段对应技术说明第五章的“Elasticsearch 索引同步”：实体、关系和 RAPTOR 摘要节点通过适配器转换为 ES 文档，embedding 后批量写入，使检索阶段可以同时召回原始文本块和图对象。

本阶段解决的问题：

- 将 `entity` 节点转换为实体 ES 文档。
- 将 `semantic` 边转换为关系 ES 文档。
- 将 `summary` 节点转换为 RAPTOR 摘要 ES 文档。
- 为每个图对象生成 `content` 和 `content_vector`，支持 BM25、dense、hybrid 检索。
- 保留 `source_chunk_ids`、`source_locates`、`level`、`parent_node_id`、`child_node_ids` 等后续离线路标需要的字段。
- 支持可选地把 chunk 的直接父摘要节点写回 chunk index。

### 对应文件

- `signpost/indexing/graph_schema.py`
  - `graph_index_name()`：生成 namespace 级图对象索引名。
  - `graph_index_mapping()`：定义 ES mapping，包含 `content_vector`、对象类型、来源、层级等字段。
  - `graph_to_index_documents()`：从 unified graph 生成 entity/relation/summary ES 文档。
  - `chunk_parent_updates()`：从 structure 边计算 chunk 的直接父 summary 节点。
- `signpost/indexing/graph_es_sync.py`
  - CLI：`graph.unified.json -> Elasticsearch graph index`。
  - 支持 ECNU/hash embedding、批量写入、重建索引、可选 chunk parent 更新。
- `signpost/retrieval/graph_search.py`
  - 图对象检索 CLI。
  - 支持 `bm25`、`dense`、`hybrid`，支持按 `entity`、`relation`、`summary` 过滤。
- `signpost/storage/elasticsearch.py`
  - 新增 `update_doc()`，用于可选更新 chunk index 的父摘要字段。
- `tests/test_graph_es_sync.py`
  - 覆盖 ES 文档适配、mapping、hash embedding 同步、chunk parent 更新。

### 输入格式

输入为 F9 输出的 `graph.unified.json`：

```json
{
  "metadata": {"namespace": "mini", "graph_type": "unified"},
  "nodes": [
    {
      "node_id": "entity:...",
      "node_type": "entity",
      "name": "推荐",
      "entity_type": "CONCEPT",
      "description": "统一描述",
      "source_chunk_ids": ["mini_doc_001_c00001"],
      "source_locates": ["mini.txt:L4-L6"]
    },
    {
      "node_id": "raptor:...",
      "node_type": "summary",
      "title": "第二章 方法",
      "content": "摘要正文",
      "level": 1,
      "parent_node_id": "raptor:root",
      "child_node_ids": ["chunk:mini_doc_001_c00001"]
    }
  ],
  "edges": [
    {
      "source": "entity:...",
      "target": "entity:...",
      "edge_type": "semantic",
      "description": "关系描述",
      "relation_types": ["co_occurs"]
    }
  ]
}
```

### 输出格式

输出写入 Elasticsearch graph index，默认索引名：

```text
signpost-<namespace>-graph
```

实体文档示例：

```json
{
  "id": "entity:62b46f24ae40",
  "type": "graph",
  "namespace": "mini",
  "object_type": "entity",
  "node_id": "entity:62b46f24ae40",
  "name": "推荐",
  "entity_type": "CONCEPT",
  "title": "推荐",
  "content": "推荐\nCONCEPT\n统一描述\nmini.txt:L4-L6",
  "content_vector": [0.1, 0.2],
  "source_chunk_ids": ["mini_doc_001_c00001"],
  "source_locates": ["mini.txt:L4-L6"]
}
```

关系文档示例：

```json
{
  "id": "edge:...",
  "type": "graph",
  "namespace": "mini",
  "object_type": "relation",
  "source": "entity:...",
  "target": "entity:...",
  "title": "推荐 -> PPR",
  "content": "推荐 -> PPR\nco_occurs\n关系描述",
  "relation_types": ["co_occurs"],
  "source_chunk_ids": ["mini_doc_001_c00001"]
}
```

摘要文档示例：

```json
{
  "id": "raptor:...",
  "type": "graph",
  "namespace": "mini",
  "object_type": "summary",
  "node_id": "raptor:...",
  "title": "第二章 方法",
  "content": "第二章 方法\n摘要正文",
  "level": 1,
  "parent_node_id": "raptor:root",
  "child_node_ids": ["chunk:mini_doc_001_c00001"],
  "source_chunk_ids": ["mini_doc_001_c00001"],
  "source_locates": ["mini.txt:L4-L6"]
}
```

### 验证命令

mini graph index 写入：

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace mini \
  --graph outputs/mini/graph.unified.json \
  --embedding-provider hash \
  --hash-dimensions 128 \
  --recreate
```

图对象检索：

```bash
conda run -n signpost-re python -m signpost.retrieval.graph_search \
  --namespace mini \
  --query "PPR 推荐" \
  --embedding-provider hash \
  --mode hybrid \
  --top-k 3
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace graphrag-bench-medical-smoke \
  --graph datasets/processed/graphrag-bench-medical/graph.unified.json \
  --embedding-provider hash \
  --hash-dimensions 128 \
  --recreate
```

可选更新 chunk index 的父摘要字段：

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace mini \
  --graph outputs/mini/graph.unified.json \
  --embedding-provider hash \
  --update-chunk-parents \
  --chunk-index signpost-mini-chunks
```

## F11 离线路标

### 目标

F11 为检索结果附着离线路标，对应技术说明 4.2.1 节的预计算结构化元数据。离线路标不做运行时 PageRank，只从 `graph.unified.json` 中读取已构建好的结构、顺序、语义和溯源关系，为每个检索结果补充导航线索。

技术说明中涉及的能力在本阶段对应为：

- 纵向线索：文本块所属父摘要节点；摘要节点的父摘要和子节点列表。
- 横向线索：文本块的前后相邻 chunk 文件定位。
- 溯源线索：文件名、行号范围、source chunk、source locate；摘要 source locate 会合并连续或重叠行号区间。
- 语义线索：实体和关系结果携带相邻实体，便于后续在线路标或 agent 决策使用。
- 实例级设计：chunk、summary、entity、relation 四类结果携带不同字段子集。

### 对应文件

- `signpost/retrieval/offline_signpost.py`
  - `build_offline_signpost()`：为单个节点、边或检索结果构造离线路标。
  - `attach_offline_signposts()`：批量增强检索结果。
  - `GraphIndex`：从 unified graph 构建轻量索引，包括结构父子关系、顺序前后关系、语义邻居和溯源边。
  - CLI：支持直接指定 `--node-id`、`--chunk-id`，也支持 `--query` 调用已有 chunk/graph search 后统一附着离线路标。
- `tests/test_offline_signpost.py`
  - 覆盖 chunk 的纵向/横向/溯源线索。
  - 覆盖 summary 的父子节点和 source locate 合并。
  - 覆盖 entity/relation 的语义邻居线索。

### 输入格式

输入一：F9 输出的 `graph.unified.json`。

输入二：检索结果，可以是 chunk 结果：

```json
{
  "chunk_id": "mini_doc_001_c00001",
  "file_name": "mini.txt",
  "start_line": 4,
  "end_line": 6,
  "section_path": ["第二章 方法"]
}
```

也可以是 graph result：

```json
{
  "id": "edge:...",
  "object_type": "relation",
  "source": "entity:62b46f24ae40",
  "target": "entity:b4a9ca4f6d48"
}
```

### 输出格式

chunk 离线路标：

```json
{
  "result_type": "chunk",
  "vertical": {
    "section_path": ["第二章 方法"],
    "parent_summaries": [
      {"node_id": "raptor:...", "title": "第二章 方法", "level": 1}
    ],
    "nearest_parent_summary": {"node_id": "raptor:...", "title": "第二章 方法"}
  },
  "horizontal": {
    "previous_chunk": {"chunk_id": "mini_doc_001_c00000", "locate": "mini.txt:L1-L3"},
    "next_chunk": null
  },
  "provenance": {
    "file_name": "mini.txt",
    "start_line": 4,
    "end_line": 6,
    "locate": "mini.txt:L4-L6"
  }
}
```

summary 离线路标：

```json
{
  "result_type": "summary",
  "vertical": {
    "level": 1,
    "parent_summary": {"node_id": "raptor:root", "title": "mini.txt"},
    "child_summaries": [],
    "child_chunks": [{"chunk_id": "mini_doc_001_c00001", "locate": "mini.txt:L4-L6"}]
  },
  "provenance": {
    "source_chunk_ids": ["mini_doc_001_c00001"],
    "source_locates": ["mini.txt:L4-L6"]
  }
}
```

entity/relation 离线路标：

```json
{
  "result_type": "entity",
  "provenance": {
    "source_chunk_ids": ["mini_doc_001_c00001"],
    "source_locates": ["mini.txt:L4-L6"]
  },
  "semantic": {
    "neighboring_entities": [
      {"node_id": "entity:...", "name": "PPR", "entity_type": "CONCEPT"}
    ]
  }
}
```

### 验证命令

直接增强 chunk：

```bash
conda run -n signpost-re python -m signpost.retrieval.offline_signpost \
  --graph outputs/mini/graph.unified.json \
  --chunk-id mini_doc_001_c00001
```

直接增强 RAPTOR/summary：

```bash
conda run -n signpost-re python -m signpost.retrieval.offline_signpost \
  --graph outputs/mini/graph.unified.json \
  --node-id raptor:03e3fb2c8643cdc5
```

直接增强 entity：

```bash
conda run -n signpost-re python -m signpost.retrieval.offline_signpost \
  --graph outputs/mini/graph.unified.json \
  --node-id entity:62b46f24ae40
```

query 模式，先检索再附着离线路标：

```bash
conda run -n signpost-re python -m signpost.retrieval.offline_signpost \
  --namespace mini \
  --query "PPR 推荐" \
  --embedding-provider hash \
  --mode hybrid \
  --top-k 2
```

## F12 在线路标：PPR 推荐

### 目标

F12 实现技术说明 4.2.2 节的在线路标：根据当前检索结果组选择 seed nodes，在面向场景裁剪后的子图上运行 Personalized PageRank，返回 top-k 关联实体作为组级探索建议。

本阶段解决的问题：

- 文本关联场景：seed 为 chunk 或 summary，目标是从文本内容发现相关实体。
- 图结构场景：seed 为 entity 或 relation 端点，目标是在实体关系网络中发现相关实体。
- 子图裁剪：文本场景保留 chunk/entity 和相关 summary 路径；图谱场景保留 chunk/entity 并剪除 summary。
- PPR：使用 unified graph 的边权重运行可解释的 PageRank，不引入额外图数据库依赖。
- 输出组级在线路标，不附着在单个检索结果上。

### 对应文件

- `signpost/retrieval/online_signpost.py`
  - `compute_online_signpost()`：F12 主入口。
  - `personalized_pagerank()`：纯 Python PPR 实现。
  - `OnlineGraphIndex`：从 unified graph 构建邻接表、结构父子索引和 seed 解析器。
  - CLI：支持 `--seed`、`--result-json`、`--scene text|graph|auto`。
- `tests/test_online_signpost.py`
  - 覆盖文本场景 chunk seed。
  - 覆盖图谱场景 entity seed。
  - 覆盖 auto 场景判断。

### 输入格式

输入一：F9 输出的 `graph.unified.json`。

输入二：seed 节点，可直接指定节点 ID：

```text
chunk:mini_doc_001_c00001
entity:62b46f24ae40
raptor:03e3fb2c8643cdc5
```

也可以传入检索结果 JSON 列表，模块会从 `node_id`、`id`、`chunk_id`、`source`、`target` 中解析 seed。

### 输出格式

输出为组级在线路标：

```json
{
  "scene": "graph",
  "seeds": ["entity:62b46f24ae40"],
  "subgraph": {
    "nodes": 12,
    "edges": 19
  },
  "recommended_entities": [
    {
      "node_id": "entity:b4a9ca4f6d48",
      "name": "PPR",
      "entity_type": "CONCEPT",
      "score": 0.12035729495835824,
      "source_chunk_ids": ["mini_doc_001_c00001"],
      "source_locates": ["mini.txt:L4-L6"]
    }
  ]
}
```

### 场景规则

- `scene=text`：用于文本组 seed，包括 chunk 和 summary。
- `scene=graph`：用于图谱组 seed，包括 entity 和 relation 端点。
- `scene=auto`：当所有 seed 都是 entity 时走 graph，否则走 text。
- PPR 推荐只返回 entity 节点，并排除 seed entity 本身。

### 验证命令

mini 文本场景：

```bash
conda run -n signpost-re python -m signpost.retrieval.online_signpost \
  --graph outputs/mini/graph.unified.json \
  --seed chunk:mini_doc_001_c00001 \
  --scene text \
  --top-k 3
```

mini 图谱场景：

```bash
conda run -n signpost-re python -m signpost.retrieval.online_signpost \
  --graph outputs/mini/graph.unified.json \
  --seed entity:62b46f24ae40 \
  --scene graph \
  --top-k 3
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.retrieval.online_signpost \
  --graph datasets/processed/graphrag-bench-medical/graph.unified.json \
  --seed chunk:0_c00000 \
  --scene text \
  --top-k 5
```

## F13 图检索引擎

### 目标

F13 将 F5/F10 的 ES 检索、F11 离线路标和 F12 在线路标组合为技术说明第五章描述的图检索引擎。一次 query 返回文本组和图谱组两个结果组：

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

本阶段解决的问题：

- 同时检索原始 chunk、RAPTOR summary、entity、relation。
- 将 chunk 和 summary 归入 `text_group`。
- 将 entity 和 relation 归入 `graph_group`。
- 为每条结果附着 F11 离线路标。
- 为 text group 运行 F12 文本场景 PPR。
- 为 graph group 运行 F12 图谱场景 PPR。
- 输出可直接供后续 Researcher/Agent 消费的结构化检索结果。

### 对应文件

- `signpost/retrieval/run.py`
  - `run_retrieval()`：F13 主入口，调用 ES 检索并组装双组结果。
  - `build_grouped_retrieval_result()`：纯函数，便于测试和后续 Agent 复用。
  - CLI：`python -m signpost.retrieval.run`。
- `tests/test_retrieval_run.py`
  - 使用内存 unified graph 和假检索结果测试 text/graph 双组结构。
  - 覆盖离线路标和在线路标的组合。

### 输入格式

必需输入：

- `namespace`
- `query`
- `graph.unified.json`

依赖索引：

- F5 chunk index：`signpost-<namespace>-chunks`
- F10 graph index：`signpost-<namespace>-graph`

### 输出格式

输出为 `retrieval_result.json`：

```json
{
  "query": "PPR 推荐",
  "text_group": {
    "items": [
      {
        "retrieval_type": "chunk",
        "chunk_id": "mini_doc_001_c00001",
        "content": "...",
        "offline_signpost": {
          "result_type": "chunk",
          "vertical": {},
          "horizontal": {},
          "provenance": {}
        }
      },
      {
        "retrieval_type": "summary",
        "node_id": "raptor:...",
        "title": "第二章 方法",
        "offline_signpost": {
          "result_type": "summary"
        }
      }
    ],
    "online_signpost": {
      "scene": "text",
      "recommended_entities": []
    }
  },
  "graph_group": {
    "items": [
      {
        "retrieval_type": "entity",
        "node_id": "entity:...",
        "offline_signpost": {
          "result_type": "entity"
        }
      },
      {
        "retrieval_type": "relation",
        "source": "entity:...",
        "target": "entity:...",
        "offline_signpost": {
          "result_type": "relation"
        }
      }
    ],
    "online_signpost": {
      "scene": "graph",
      "recommended_entities": []
    }
  },
  "metadata": {
    "text_items": 4,
    "graph_items": 2,
    "ppr_top_k": 3
  }
}
```

### 验证命令

mini 检索：

```bash
conda run -n signpost-re python -m signpost.retrieval.run \
  --namespace mini \
  --query "PPR 推荐" \
  --graph outputs/mini/graph.unified.json \
  --embedding-provider hash \
  --mode hybrid \
  --chunk-top-k 2 \
  --summary-top-k 2 \
  --graph-top-k 2 \
  --ppr-top-k 3 \
  --output outputs/mini/retrieval_result.json
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.retrieval.run \
  --namespace graphrag-bench-medical-smoke \
  --query "basal cell carcinoma treatment" \
  --graph datasets/processed/graphrag-bench-medical/graph.unified.json \
  --embedding-provider hash \
  --mode hybrid \
  --chunk-top-k 2 \
  --summary-top-k 2 \
  --graph-top-k 2 \
  --ppr-top-k 3 \
  --output datasets/processed/graphrag-bench-medical/retrieval_result.json
```

## F14 ReadFile / 溯源读取

### 目标

F14 对应技术说明中的 ReadFile 工具：当检索结果或离线路标给出文件名和行号范围后，Researcher 可以按坐标回读原始文档片段，获得更精确的证据文本，并可向前或向后扩展局部上下文。

本阶段只保留科研实验需要的溯源读取能力，不迁移旧项目中的用户、知识库权限、目录授权和前端文件预览逻辑。

### 对应文件

- `signpost/retrieval/read_file.py`
  - 提供 `read_file_window()`，按 `file_name`、`start_line`、`end_line` 读取文档片段。
  - 提供 `read_locate()`，读取 `file.txt:L10-L20` 形式的技术说明溯源坐标。
  - 提供 `parse_locate()`，解析 locate 字符串。
  - 提供 `format_file_view()`，输出带行号的可读文本。
  - 提供 CLI：`python -m signpost.retrieval.read_file`。
- `tests/test_read_file.py`
  - 覆盖 locate 解析、行号窗口扩展、read\_locate 入口和空结果格式化。

### 输入

F14 默认读取 F3.5 文档解析产物 `documents.jsonl`。如果只传 `--dataset`，会按顺序查找：

1. `datasets/processed/<dataset>/documents.jsonl`
2. `outputs/<dataset>/documents.jsonl`
3. `samples/<dataset>/documents.jsonl`

也可以显式传入 `--documents` 指向任意 `documents.jsonl`。

支持两种定位方式：

- 显式坐标：

```bash
--file mini.txt --start-line 4 --end-line 6
```

- locate 坐标：

```bash
--locate mini.txt:L4-L6
```

可选上下文扩展：

- `--before N`：向前扩展 N 行。
- `--after N`：向后扩展 N 行。

注意：当前读取的是 F3.5 规范化后的 `documents.jsonl`。空行会在解析阶段被过滤，但 `line_no` 保留原文行号，因此输出行号仍可稳定回指原文位置。

### 输出

默认输出带行号的文本视图：

```text
=== mini.txt:L4-L6 ===
     4 | 第二章 方法
     5 | 它使用图结构和 PPR 推荐。
     6 | ![示意图](figure.png)
```

传入 `--json` 时输出结构化 JSON：

```json
{
  "tool": "read_file",
  "dataset": "mini",
  "documents_path": ".../outputs/mini/documents.jsonl",
  "doc_id": "mini_doc_001",
  "file_name": "mini.txt",
  "requested": {
    "start_line": 4,
    "end_line": 6,
    "before": 0,
    "after": 0
  },
  "resolved": {
    "start_line": 4,
    "end_line": 6
  },
  "lines": [
    {
      "line_no": 4,
      "text": "第二章 方法"
    }
  ],
  "file_content_view": "..."
}
```

### 验证命令

单元测试：

```bash
conda run -n signpost-re python -m pytest tests/test_read_file.py
```

mini 文本视图：

```bash
conda run -n signpost-re python -m signpost.retrieval.read_file \
  --dataset mini \
  --file mini.txt \
  --start-line 4 \
  --end-line 6 \
  --before 1
```

mini JSON 输出：

```bash
conda run -n signpost-re python -m signpost.retrieval.read_file \
  --dataset mini \
  --locate mini.txt:L4-L6 \
  --json
```

真实数据结构 smoke：

```bash
conda run -n signpost-re python -m signpost.retrieval.read_file \
  --dataset graphrag-bench-medical \
  --locate 0.txt:L1-L2 \
  --after 1
```

## F15 Supervisor-Researcher Agent

### 目标

F15 对应技术说明第四章的 Supervisor-Researcher 多智能体检索增强框架，形成“查询拆解 -> 路标引导检索 -> 源文档回读 -> 答案综合”的闭环。

本阶段只保留技术说明实验需要的 Agent 算法链路，不迁移旧项目中的 Web SSE、Redis 任务恢复、租户隔离、前端事件流和用户权限逻辑。

### 对应文件

- `signpost/agent/tools.py`
  - `KnowledgeSearchTool`：封装 F13 检索，返回 `text_group`、`graph_group`、离线路标和在线 PPR 路标。
  - `ReadFileTool`：封装 F14，根据 `file.txt:Lx-Ly` 坐标回读证据片段。
  - `default_search_config()`：默认定位 `outputs/<namespace>/graph.unified.json` 和 `outputs/<namespace>/chunks.jsonl`。
  - 本地 artifact fallback：用于 mini smoke 和无 ES 环境测试；正式实验可通过 `--use-es` 调用 ES 检索入口。
- `signpost/agent/supervisor.py`
  - `Supervisor`：拆解问题、调度 Researcher、综合最终答案。
  - `Researcher`：调用 `KnowledgeSearchTool`，从路标中抽取 locate 坐标，再调用 `ReadFileTool` 回读源文档。
  - `TraceRecorder`：记录 agent\_start、plan、tool\_call、tool\_error、final\_answer 等结构化轨迹。
  - `deterministic_decompose()` 和 `deterministic_synthesize()`：无 LLM 或 smoke 场景下的确定性 fallback。
- `signpost/agent/run.py`
  - 单问题 CLI：`python -m signpost.agent.run`。
- `signpost/agent/batch.py`
  - 批量预测 CLI：读取 questions JSONL，输出 F16 可继续适配的 predictions JSONL。
- `tests/test_agent.py`
  - 覆盖问题拆解、locate 提取、完整 Supervisor-Researcher 流程和 batch 输出。

### 输入

单问题输入：

- `--namespace`：实验或数据集命名空间，例如 `mini`。
- `--question`：用户问题。
- `--use-llm`：可选，启用 ECNU/OpenAI-compatible chat 进行问题拆解和答案综合。
- `--use-es`：可选，启用 F13 的 Elasticsearch 检索路径；默认使用本地 artifacts，便于离线 smoke。

默认读取：

- `outputs/<namespace>/graph.unified.json`
- `outputs/<namespace>/chunks.jsonl`
- `outputs/<namespace>/documents.jsonl`

批量输入：

```json
{
  "question_id": "mini_q_001",
  "question": "Signpost 使用什么机制进行推荐？",
  "answer": "标准答案",
  "rationale": "可选推理路径",
  "metadata": {
    "dataset": "mini"
  }
}
```

问题字段兼容 `question`、`query`、`input`；ID 字段兼容 `id`、`qid`、`question_id`。

### 输出

单问题输出 JSON：

```json
{
  "trace_id": "...",
  "namespace": "mini",
  "question": "PPR 推荐是什么",
  "subquestions": [
    "PPR 推荐是什么"
  ],
  "answer": "带引用的最终答案",
  "citations": [
    {
      "file_name": "mini.txt",
      "start_line": 4,
      "end_line": 6,
      "locate": "mini.txt:L4-L6"
    }
  ],
  "research": [
    {
      "subquestion": "...",
      "retrieval": {
        "text_group": {},
        "graph_group": {}
      },
      "evidence": [],
      "locates": []
    }
  ],
  "trace": []
}
```

批量输出 JSONL：

```json
{
  "id": "mini_q_001",
  "dataset": "mini",
  "method": "signpost",
  "question": "Signpost 使用什么机制进行推荐？",
  "answer": "带引用的最终答案",
  "citations": [],
  "trace_id": "...",
  "trace": [],
  "gold_answer": "标准答案",
  "metadata": {}
}
```

### 验证命令

单元测试：

```bash
conda run -n signpost-re python -m pytest tests/test_agent.py
```

mini 单问题 smoke：

```bash
conda run -n signpost-re python -m signpost.agent.run \
  --namespace mini \
  --question "PPR 推荐是什么"
```

mini 批量 smoke：

```bash
conda run -n signpost-re python -m signpost.agent.batch \
  --namespace mini \
  --questions samples/mini/questions.jsonl \
  --output outputs/mini/agent_predictions.jsonl \
  --limit 1
```

走 ES 检索路径时追加：

```bash
--use-es
```

## F16 预测输出与评估适配

### 目标

F16 对应技术说明第五章系统级评估：把不同检索或 Agent 方法的输出统一为稳定 JSONL，供基础指标、GraphRAG-Bench 开放式评分、LLM-as-Judge 生成质量评分和后续测速统计使用。

本阶段不迁移旧项目的批处理队列、Web UI、评测看板和外部任务管理，只保留技术说明实验需要的预测文件格式、校验、转换和评估入口。

### 对应文件

- `signpost/evaluation/schema.py`
  - 定义 F16 预测 JSONL 必需字段。
  - `normalize_prediction_record()`：把 F15 agent 输出或其他 baseline 输出转成统一 schema。
  - `build_prediction_text()`：生成旧评估脚本兼容的 `<think>...</think><answer>...</answer>` 格式。
  - `validate_prediction_record()`：检查字段完整性和 `metadata.method/dataset`。
- `signpost/evaluation/validate_predictions.py`
  - CLI 校验入口。
  - 支持 `--normalize` 和 `--output`，可边转换边校验。
- `signpost/evaluation/convert_predictions.py`
  - 只做转换的 CLI，内部复用校验逻辑。
- `signpost/evaluation/metrics.py`
  - 基础非 LLM 指标：答案抽取、EM、Precision、Recall、F1。
- `signpost/evaluation/evaluate_basic.py`
  - 基础指标 CLI。
- `signpost/evaluation/llm_judge.py`
  - 可选 LLM-as-Judge 适配，复用 F1 的 ECNU/OpenAI-compatible chat client。
  - 当前支持 `answer_correctness`、`factuality`、`completeness` 三个维度，输出归一化 0-1 分数和原始 judge response。
- `signpost/agent/batch.py`
  - 已调整为直接输出 F16 schema，不再把生成答案写入 gold `answer` 字段。
- `tests/test_evaluation.py`
  - 覆盖 schema 标准化、校验、答案抽取、基础指标和转换写出。

### 输入

标准预测 JSONL：

```json
{
  "question_id": "legal_q_0001",
  "question": "...",
  "answer": "标准答案",
  "rationale": "标准推理路径，可为空",
  "prediction": "<think>模型推理</think><answer>模型答案</answer>",
  "metadata": {
    "method": "signpost",
    "dataset": "legal"
  }
}
```

兼容输入字段：

- ID：`question_id`、`id`、`qid`
- 问题：`question`、`query`、`input`
- 标准答案：`answer` 或 F15 旧输出中的 `gold_answer`
- 生成答案：`prediction`；如果缺失，则从 `generated_answer`、`model_answer` 或 `answer` 包装成 `<answer>`。

### 输出

校验输出：

```json
{
  "input": ".../predictions.jsonl",
  "output": null,
  "num_rows": 1,
  "valid": true,
  "issues": []
}
```

基础评估输出：

```json
{
  "num_samples": 1,
  "num_scored": 1,
  "num_skipped": 0,
  "metrics": {
    "exact_match": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "per_example": []
}
```

LLM-as-Judge 输出会在每条预测上追加：

```json
{
  "llm_judge": {
    "answer_correctness": {
      "score": 0.8,
      "raw_score": 8.0,
      "raw_response": "..."
    }
  }
}
```

### 验证命令

F16 单元测试：

```bash
conda run -n signpost-re python -m pytest tests/test_evaluation.py
```

生成 mini 标准预测文件：

```bash
conda run -n signpost-re python -m signpost.agent.batch \
  --namespace mini \
  --questions samples/mini/questions.jsonl \
  --output outputs/mini/agent_predictions.jsonl \
  --limit 1
```

校验预测文件：

```bash
conda run -n signpost-re python -m signpost.evaluation.validate_predictions \
  --input outputs/mini/agent_predictions.jsonl
```

基础指标：

```bash
conda run -n signpost-re python -m signpost.evaluation.evaluate_basic \
  --input outputs/mini/agent_predictions.jsonl \
  --output outputs/mini/basic_eval.json
```

转换旧格式输出：

```bash
conda run -n signpost-re python -m signpost.evaluation.convert_predictions \
  --input outputs/mini/agent_predictions.jsonl \
  --output outputs/mini/predictions.f16.jsonl \
  --dataset mini
```

可选 LLM-as-Judge：

```bash
conda run -n signpost-re python -m signpost.evaluation.llm_judge \
  --input outputs/mini/agent_predictions.jsonl \
  --output outputs/mini/llm_judge.jsonl \
  --dimension answer_correctness
```

## 项目实验指标补充

为了支撑 `project_experiment_design*.md` 中的成本-质量实验，新增了 `signpost.benchmark` 指标层。它不实现 baseline，也不实现剪枝，只读取现有工件并生成技术说明表格/曲线需要的统计量。

### 新增文件

- `signpost/benchmark/stats.py`：通用 sum/mean/median/p90/p95 统计。
- `signpost/benchmark/query_metrics.py`：从 prediction/query log 计算 EM/F1、在线 token/call/latency、弱证据 Recall@k/MRR。
- `signpost/benchmark/index_metrics.py`：从 stage log、semantic extraction cache、graph JSON 计算离线索引和图结构指标。
- `signpost/benchmark/cost_quality.py`：计算摊销成本、break-even、Pareto frontier、每多答对一个问题的额外成本。
- `signpost/benchmark/time_stage.py`：包装任意 F3-F16 阶段命令并追加 `stage_timing.jsonl`。
- `docs/experiment_metrics_guide.zh.md`：完整指标说明和使用方式。
- `tests/test_benchmark_metrics.py`：新增指标单元测试。

### 典型用法

记录阶段耗时：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset agriculture \
  --stage F6_semantic_graph \
  --method-scope signpost_offline_index \
  --input-path datasets/processed/agriculture/chunks.jsonl \
  --output-path datasets/processed/agriculture/graph.semantic.llm.json \
  --log outputs/agriculture/logs/stage_timing.jsonl \
  -- \
  conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
    --namespace agriculture-llm \
    --chunks datasets/processed/agriculture/chunks.jsonl \
    --output datasets/processed/agriculture/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --extractions-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl
```

生成 query 指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/agriculture/predictions/signpost.jsonl \
  --output outputs/agriculture/metrics/signpost.query_metrics.json
```

生成离线索引和图结构指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/agriculture/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/agriculture/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/agriculture/graph.unified.json \
  --output outputs/agriculture/metrics/index_metrics.json
```

生成成本-质量派生指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.cost_quality \
  --methods outputs/agriculture/metrics/method_summaries.json \
  --output outputs/agriculture/metrics/cost_quality.json
```

如果先理解实验该怎么看，读 `docs/experiment_metrics_plain_guide.zh.md`；如果要查字段和命令细节，再读 `docs/experiment_metrics_guide.zh.md`。
