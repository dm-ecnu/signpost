# Signpost 当前版功能点手册（复制更新版）

本文档是 `docs/refactor_function_plan.zh.md` 的当前实现版副本。原文件保留不动；本文件按现在 `signpost_re` 的代码、数据目录、指标模块和运行方式重新整理。

本文档默认工作目录：

```text
/home/ruolinsu/signpost/signpost_re
```

本文档默认数据根目录：

```text
/home/ruolinsu/signpost/signpost_re/datasets
```

当前不再使用 `samples/mini` 作为主说明对象。流程测试使用 `legal_test`，论文轻量 Legal 实验使用 `legal_lite`。

## 1. 当前工程目标

`signpost_re` 是论文实验系统，不是产品后端。它保留论文算法链路：数据标准化、文档解析、章节树与 chunk、ES 文本/向量索引、语义图、结构图、顺序图、统一图、图对象索引、离线/在线路标、检索引擎、Agent、评估和实验指标。

不保留旧项目里的无关产品逻辑：用户系统、租户权限、登录鉴权、前端、Canvas、API token 管理、文件夹管理、缩略图等。

旧项目字段在新项目中的替换关系：

| 旧字段         | 新含义                       |
| ----------- | ------------------------- |
| `tenant_id` | `namespace`               |
| `user_id`   | `namespace`               |
| `kb_id`     | `dataset_id` 或 `index_id` |

## 2. 当前目录结构

| 路径                     | 作用                                                      |
| ---------------------- | ------------------------------------------------------- |
| `signpost/config/`     | F0 配置和实验上下文。                                            |
| `signpost/llm/`        | F1 ECNU/OpenAI-compatible chat、embedding、rerank client。 |
| `signpost/storage/`    | ES 最小 HTTP client。                                      |
| `signpost/data/`       | F3 数据集准备、校验、UltraDomain raw 子集抽取。                       |
| `signpost/parsing/`    | F3.5 文档解析、文本规范化、行号保留。                                   |
| `signpost/chunking/`   | F4 标题识别、文档树、chunk。                                      |
| `signpost/indexing/`   | F5/F6/F7/F8/F10 索引与图构建入口。                               |
| `signpost/graph/`      | 语义图、结构图、顺序图、统一图、校验、查看。                                  |
| `signpost/retrieval/`  | F5/F10/F11/F12/F13/F14 检索与路标。                           |
| `signpost/agent/`      | F15 Supervisor-Researcher Agent。                        |
| `signpost/evaluation/` | F16 prediction schema、基础评估、LLM judge。                   |
| `signpost/benchmark/`  | 实验测速、查询成本、索引成本、成本质量分析。                                  |
| `datasets/`            | 原始数据和处理后数据。                                             |
| `outputs/`             | 预测、日志、指标输出。                                             |
| `docs/`                | 说明文档。                                                   |
| `tests/`               | 单元测试和 smoke test。                                       |

## 3. 数据集约定

### 3.1 原始数据

UltraDomain 原始数据放在：

```text
datasets/raw/ultradomain/<dataset>.jsonl
```

当前重点数据集：

| dataset                  | 用途                                                         |
| ------------------------ | ---------------------------------------------------------- |
| `legal_test`             | 小型流程测试集，1 个完整 legal document，9 个问题，当前 F4 生成 88 个 chunks。   |
| `legal_lite`             | 论文 Plan B 的 Legal-Lite 子集，12 个完整 legal documents，约 94 个问题。 |
| `agriculture`            | 农业数据集。                                                     |
| `legal`                  | 全量 Legal 数据集。                                              |
| `graphrag-bench-medical` | GraphRAG Bench medical。                                    |
| `graphrag-bench-novel`   | GraphRAG Bench novel。                                      |

### 3.2 处理后数据

F3 之后统一放在：

```text
datasets/processed/<dataset>/
```

典型文件：

| 文件                               | 生成阶段 | 含义                    |
| -------------------------------- | ---- | --------------------- |
| `raw_corpus.jsonl`               | F3   | 标准化后的文档语料。            |
| `questions.jsonl`                | F3   | 标准化后的问题和答案。           |
| `documents.jsonl`                | F3.5 | 解析后的文档，含行号。           |
| `chunks.jsonl`                   | F4   | 文档 chunk。             |
| `document_trees.jsonl`           | F4   | 章节树。                  |
| `graph.semantic.json`            | F6   | deterministic 语义图。    |
| `graph.semantic.llm.json`        | F6   | LLM 语义图。              |
| `semantic_llm.extractions.jsonl` | F6   | 每个 chunk 的抽取缓存，可断点续跑。 |
| `semantic_llm.progress.jsonl`    | F6   | 进度日志。                 |
| `graph.structure.json`           | F7   | 结构/RAPTOR 视图。         |
| `graph.sequence.json`            | F8   | 顺序视图。                 |
| `graph.unified.json`             | F9   | 多视图统一图。               |

### 3.3 指标到底是怎么产生的

先明确一个原则：

> 时间、调用次数、token、磁盘大小这些原始指标，必须在运行功能点时产生。后面的 `signpost.benchmark.*` 不是重新“测”这些东西，而是读取运行时留下的日志和工件，把它们汇总成论文表格需要的指标。

所以实验链路分两步：

```text
第一步：运行功能点，同时采集原始测量数据

  F4 chunking 命令
    -> 功能输出：chunks.jsonl、document_trees.jsonl
    -> 测量输出：stage_timing.jsonl 里追加一行 F4_chunk_tree

  F6 semantic graph 命令
    -> 功能输出：graph.semantic.llm.json、semantic_llm.extractions.jsonl
    -> 测量输出：stage_timing.jsonl 里追加一行 F6_semantic_graph
    -> 运行中原始数据：semantic_llm.progress.jsonl、semantic_llm.extractions.jsonl

  F15 agent batch 命令
    -> 功能输出：predictions/signpost.ecnu.jsonl
    -> 测量输出：prediction/trace/query log 中包含 latency、tool calls、tokens 等字段

第二步：汇总原始测量数据，不重新运行功能

  index_metrics.py
    读取 stage_timing.jsonl、semantic cache、graph JSON
    -> 输出 index_metrics.json

  query_metrics.py
    读取 predictions/query log
    -> 输出 query_metrics.json

  method_summary.py
    读取 query_metrics.json 和 stage_timing.jsonl
    -> 输出 method_summaries.json

  cost_quality.py
    读取 method_summaries.json
    -> 输出 amortized cost、break-even、Pareto
```

换句话说：

- `time_stage.py` 是“带计时的运行器”，不是事后测试器。它包住真正的功能命令，功能命令跑完的同时写一行 `stage_timing.jsonl`。
- `index_metrics.py`、`query_metrics.py`、`cost_quality.py` 是“汇总器”，只能汇总已有日志，不能凭空知道某个功能点花了多久。
- 如果某个功能点没有通过 `time_stage.py` 包起来跑，也没有自己写日志，那么这次运行的 wall-clock time 就丢了，后面不能精确补回来。
- 如果 LLM API 没有记录 token usage，那么 token 也不会凭空出现，只能用 tokenizer 或 prompt/response 长度后处理估算。

`time_stage.py` 能记录什么、不能记录什么：

| 内容                            | `time_stage.py` 是否负责 | 说明                                                                                      |
| ----------------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| 阶段开始/结束时间                     | 是                    | 记录 `started_at`、`finished_at`、`wall_time_seconds`。                                      |
| 命令、输入、输出路径                    | 是                    | 记录 `command`、`input_path`、`output_path`。                                                |
| 返回码和失败状态                      | 是                    | 记录 `status`、`return_code`。                                                              |
| 输出文件/目录大小                     | 是                    | 通过 `--disk-path` 或 `--output-path` 记录 `disk_bytes`。                                     |
| stdout/stderr 文本              | 可选                   | 传 `--stdout-log`、`--stderr-log` 时边运行边写文件；长任务仍必须用功能点自己的 progress/cache 文件记录结构化进度。        |
| 阶段自定义数值                       | 可选                   | 如果功能点额外写出一个 JSON object，可用 `--metrics-json` 合并到 `stage_timing.jsonl` 的 `extra_metrics`。 |
| 每个 chunk 的 LLM 抽取结果           | 否                    | 由 F6 的 `semantic_llm.extractions.jsonl` 和 `semantic_llm.progress.jsonl` 负责。             |
| 每条 query 的 tool calls / trace | 否                    | 由 F15 prediction/trace 或 `<method>.query.jsonl` 负责。                                     |
| 精确 token usage                | 不自动保证                | 如果 LLM API 返回 usage 或功能点记录 token，才能写入；否则只能后处理估算。                                        |

因此，`time_stage.py` 不是“只记录时间”，但它也不是万能日志系统。它负责每个阶段的一行总账；每个 chunk、每条 query、每次 LLM 调用的明细账，必须由对应功能点自己写。完整原始日志来自：

```text
stage_timing.jsonl                 # 阶段级时间/命令/工件大小
semantic_llm.extractions.jsonl     # F6 per-chunk 抽取结果
semantic_llm.progress.jsonl        # F6 per-chunk 进度
predictions/<method>.jsonl         # F15/F16 每题预测、trace、citations
logs/<method>.query.jsonl          # 可选，每题更细的在线成本日志
graph.*.json                       # 图结构指标的原始工件
stage_metrics/*.json               # 可选，某阶段自己统计出的数值，交给 time_stage --metrics-json 合并
```

因此，后面每个功能点的“输出”要分成两类：

| 输出类型   | 什么时候产生                           | 例子                                                                 | 用途                 |
| ------ | -------------------------------- | ------------------------------------------------------------------ | ------------------ |
| 功能输出   | 功能点运行时                           | `chunks.jsonl`、`graph.unified.json`、`predictions.jsonl`            | 下一阶段输入，或论文方法结果     |
| 原始测量输出 | 功能点运行时同时产生                       | `stage_timing.jsonl`、`semantic_llm.progress.jsonl`、trace/query log | 后续计算时间、调用、token、延迟 |
| 汇总指标输出 | 功能点跑完后，由 benchmark 汇总器读取原始测量数据产生 | `index_metrics.json`、`query_metrics.json`、`cost_quality.json`      | 论文表格和图             |

### 3.4 ICDE 实验测量输出约定

当前 ICDE 实验不能只保留功能输出，还必须保留原始测量数据。所有阶段统一写到：

```text
outputs/<dataset>/
  logs/
    stage_timing.jsonl
    <method>.query.jsonl
  predictions/
    <method>.jsonl
  metrics/
    index_metrics.json
    <method>.query_metrics.json
    method_summaries.json
    cost_quality.json
```

其中：

- `stage_timing.jsonl` 是 F3-F10/F13/F15/F16 运行时产生的阶段级原始测量数据。
- `<method>.query.jsonl` 是每条 query 运行时产生的在线原始测量数据。
- `index_metrics.json` 是运行结束后，从 stage log、semantic cache、graph JSON 汇总出的离线指标。
- `<method>.query_metrics.json` 是运行结束后，从 prediction/query log 汇总出的质量、token、latency、tool calls 等指标。
- `cost_quality.json` 是运行结束后，从 method summaries 汇总出的摊销成本、break-even 和 Pareto frontier。

### 3.5 每个功能点必须产出的功能输出和测量输出

| 功能点            | 功能输出                                                                                  | 原始测量输出                                                                                                         | 汇总指标输出                                                           | 主要代码                                                                        |
| -------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| F0 配置          | smoke JSON/stdout                                                                     | 可选 `stage_timing.jsonl`                                                                                        | 无                                                                | `signpost/config/smoke.py`                                                  |
| F1 模型客户端       | chat/embedding/rerank smoke 输出                                                        | 可选 `stage_timing.jsonl`；真实调用时记录调用耗时                                                                            | 无                                                                | `signpost/llm/smoke.py`、`signpost/llm/client.py`                            |
| F2 存储服务        | ES health/search 输出                                                                   | 可选 `stage_timing.jsonl`                                                                                        | 无                                                                | `signpost/storage/elasticsearch.py`                                         |
| F3 数据准备        | `raw_corpus.jsonl`、`questions.jsonl`、`manifest.json`                                  | `stage_timing.jsonl` 中 `F3_data_prepare`                                                                       | shared preprocessing time                                        | `signpost/data/prepare.py`                                                  |
| F3.5 文档解析      | `documents.jsonl`                                                                     | `stage_timing.jsonl` 中 `F3_5_parse_normalize`，记录 `disk_bytes`                                                  | shared preprocessing time、docs/lines/placeholders 可由文件统计         | `signpost/parsing/parse_documents.py`                                       |
| F4 文档树与 chunk  | `chunks.jsonl`、`document_trees.jsonl`                                                 | `stage_timing.jsonl` 中 `F4_chunk_tree`，记录 `disk_bytes`                                                         | shared preprocessing time、chunks/token\_count 分布                 | `signpost/chunking/run.py`                                                  |
| F5 Chunk Index | ES chunk index                                                                        | `stage_timing.jsonl` 中 `F5_chunk_index`，记录 embedding/token/耗时；ES 索引大小可用 `disk_bytes` 或 ES stats                | BM25/Dense/Hybrid offline index time                             | `signpost/indexing/chunk_index.py`                                          |
| F6 语义视图        | `graph.semantic*.json`、`semantic_llm.extractions.jsonl`、`semantic_llm.progress.jsonl` | `stage_timing.jsonl` 中 `F6_semantic_graph`；cache/progress 是 per-chunk 原始抽取数据                                   | entities/relations/source\_edges、estimated LLM calls、graph stats | `signpost/indexing/semantic_graph.py`、`signpost/benchmark/index_metrics.py` |
| F7 结构视图        | `graph.structure.json`                                                                | `stage_timing.jsonl` 中 `F7_structure_graph`                                                                    | summary nodes、structure edges、graph stats                        | `signpost/indexing/structure_graph.py`                                      |
| F8 顺序视图        | `graph.sequence.json`                                                                 | `stage_timing.jsonl` 中 `F8_sequence_graph`                                                                     | sequence edges、doc chunk counts、graph stats                      | `signpost/indexing/sequence_graph.py`                                       |
| F9 统一图         | `graph.unified.json`                                                                  | `stage_timing.jsonl` 中 `F9_unified_graph`                                                                      | node/edge counts、edge ratio、degree、components                    | `signpost/graph/merge.py`、`signpost/benchmark/index_metrics.py`             |
| F10 图对象同步      | ES graph index                                                                        | `stage_timing.jsonl` 中 `F10_graph_es_sync`，记录 graph object embedding/write time                                | graph index offline time、ES object count                         | `signpost/indexing/graph_es_sync.py`                                        |
| F11 离线路标       | signpost-enhanced result JSON/stdout                                                  | 单独检查时写 `stage_timing.jsonl`；正式方法中由 F13/F15 trace 记录是否调用、返回多少路标                                                 | offline signpost attach count/time                               | `signpost/retrieval/offline_signpost.py`                                    |
| F12 在线 PPR     | online signpost JSON/stdout                                                           | 单独检查时写 `stage_timing.jsonl`；正式方法中由 F13/F15 trace/query log 记录 `ppr_latency_seconds`、PPR calls、返回节点数            | PPR calls、subgraph nodes/edges、PPR latency                       | `signpost/retrieval/online_signpost.py`                                     |
| F13 检索引擎       | `retrieval_result.json`                                                               | 单问题检查写 `stage_timing.jsonl`；批量评测应写 per-query retrieval log 或保存在 prediction trace 中                             | retrieval latency、retrieved chunks、PPR calls                     | `signpost/retrieval/run.py`、`signpost/benchmark/query_metrics.py`           |
| F14 ReadFile   | read file JSON/stdout                                                                 | 单独检查写 `stage_timing.jsonl`；正式 Agent 中由 trace 记录 read\_file 调用次数、输入 locate、返回行数                                 | read\_file\_calls、read\_file latency                             | `signpost/retrieval/read_file.py`                                           |
| F15 Agent      | `predictions/<method>.jsonl`、trace                                                    | `stage_timing.jsonl` 中 `F15_agent_batch`；每题必须在 prediction trace 或 `<method>.query.jsonl` 记录 latency/tool calls | LLM calls、tool calls、tokens、latency、p95                          | `signpost/agent/batch.py`、`signpost/benchmark/query_metrics.py`             |
| F16 评估         | `basic_eval.json`、`llm_judge.jsonl`                                                   | `stage_timing.jsonl` 中 `F16_evaluation`；judge 另记调用耗时/成本                                                        | EM、Precision、Recall、F1、LLM judge score                           | `signpost/evaluation/*`、`signpost/benchmark/query_metrics.py`               |
| F17 指标汇总       | `index_metrics.json`、`query_metrics.json`、`cost_quality.json`                         | 不产生新的原始测量，只读取前面所有 logs/artifacts                                                                               | paper tables/figures 的汇总指标                                       | `signpost/benchmark/*`                                                      |

### 3.6 统一计时命令模板

所有阶段都应该用 `signpost.benchmark.time_stage` 包一层运行。这样功能输出和原始时间日志会在同一次运行中产生：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset <dataset> \
  --stage <stage_name> \
  --method-scope <shared_preprocess|method_offline_index|online_query|evaluation> \
  --method <method_name_if_any> \
  --input-path <input_artifact> \
  --output-path <output_artifact> \
  --disk-path <artifact_or_directory_to_measure> \
  --metrics-json outputs/<dataset>/logs/stage_metrics/<stage>.json \
  --stdout-log outputs/<dataset>/logs/<stage>.stdout.log \
  --stderr-log outputs/<dataset>/logs/<stage>.stderr.log \
  --log outputs/<dataset>/logs/stage_timing.jsonl \
  -- \
  <原本要运行的命令>
```

这个命令实际做的是：

```text
记录 started_at
运行 <原本要运行的命令>
记录 finished_at
计算 wall_time_seconds
统计 output_path/disk_path 的 disk_bytes
如果 --metrics-json 存在，把其中的 JSON object 复制到 extra_metrics
把这些字段追加到 stage_timing.jsonl
返回原命令的退出码
```

所以，功能点不是先跑完、再单独“测时间”；而是要从一开始就用这个 wrapper 跑。

文档后面每个功能点如果出现裸命令，它只表示“内部功能命令”。正式跑实验时必须套上本节这个 wrapper。推荐把每个阶段实际执行的命令都保存进 `stage_timing.jsonl`，后续论文中的时间数字才能追溯。

`stage_timing.jsonl` 每行字段：

| 字段                               | 含义                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `dataset`                        | 数据集。                                                                         |
| `method`                         | 方法名；shared 阶段可为空。                                                            |
| `stage`                          | 阶段名，例如 `F4_chunk_tree`。                                                      |
| `method_scope`                   | 成本归属：`shared_preprocess`、`method_offline_index`、`online_query`、`evaluation`。 |
| `input_path` / `output_path`     | 输入输出工件路径。                                                                    |
| `command`                        | 实际运行命令。                                                                      |
| `started_at` / `finished_at`     | Unix 时间戳。                                                                    |
| `wall_time_seconds`              | 墙钟耗时。                                                                        |
| `llm_calls`                      | 可选，命令行传入或后处理补充。                                                              |
| `input_tokens` / `output_tokens` | 可选，命令行传入或后处理补充。                                                              |
| `disk_bytes`                     | `--disk-path` 或 `--output-path` 对应工件大小。                                      |
| `stdout_log` / `stderr_log`      | 可选 stdout/stderr 保存路径。                                                       |
| `metrics_path`                   | 可选阶段自定义指标 JSON 路径。                                                           |
| `extra_metrics`                  | 可选阶段自定义指标，只汇总 JSON object 里的数值字段。                                            |
| `status`                         | `ok` / `failed`。                                                             |
| `return_code`                    | 进程退出码。                                                                       |

注意：如果某阶段的 token/calls 只能在运行后由 cache 或 query log 计算，可以先在 `stage_timing.jsonl` 里保留 0，再由 `index_metrics.py` / `query_metrics.py` 汇总。

正式实验的运行规则：

- F3-F10、F13、F15、F16 不能直接跑裸命令，必须用 `time_stage.py` 包起来跑。
- F6 必须同时写 `semantic_llm.progress.jsonl` 和 `semantic_llm.extractions.jsonl`；否则只能知道总耗时，不知道每个 chunk 抽取了什么、是否断点续跑、完成了多少。
- F15 必须在 `predictions/<method>.jsonl` 中保留 trace/citations；如果后续补了更细的 query log，则写到 `outputs/<dataset>/logs/<method>.query.jsonl`。
- `index_metrics.py`、`query_metrics.py`、`method_summary.py`、`cost_quality.py` 只能在所有原始日志已经存在后运行。它们生成论文表格，不负责重新跑实验。
- 如果某功能点现在还不能输出某个明细日志，就必须在本文件里标为“当前缺口”，不能假装已经有这个指标。

### 3.7 每个功能点的正式运行口径

下表是正式实验时的执行口径。后面 F0-F16 各节里的命令用于说明“内部功能命令怎么调用”；真正投稿实验跑数时，以本表的 stage 名、输入、输出和日志为准，并用 3.6 的 `time_stage.py` 模板包装。

| 功能点            | 正式 stage 名                              | 输入                                       | 功能输出                                                                                  | 必须同步产出的原始测量数据                                                                                                                                        | 后续汇总器                                            |
| -------------- | --------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| F0 配置          | `F0_config_smoke`                       | `.env`、`conf/service_conf.yaml`          | stdout/smoke JSON                                                                     | 可选 `stage_timing.jsonl`，主要用于环境确认                                                                                                                     | 无                                                |
| F1 模型客户端       | `F1_llm_smoke`                          | `.env`、模型配置                              | chat/embedding/rerank smoke 输出                                                        | 可选 `stage_timing.jsonl`；真实实验中的 token/calls 由调用阶段记录                                                                                                   | 无                                                |
| F2 存储服务        | `F2_storage_smoke`                      | ES URL、index mapping                     | ES health/search stdout                                                               | 可选 `stage_timing.jsonl`                                                                                                                              | 无                                                |
| F3 数据准备        | `F3_data_prepare`                       | `datasets/raw/...`                       | `raw_corpus.jsonl`、`questions.jsonl`、`manifest.json`                                  | `stage_timing.jsonl`，`method_scope=shared_preprocess`，`disk_bytes` 指向 `datasets/processed/<dataset>`                                                 | `index_metrics.py` 汇总 shared 时间                  |
| F3.5 文档解析      | `F3_5_parse_normalize`                  | `raw_corpus.jsonl`                       | `documents.jsonl`                                                                     | `stage_timing.jsonl`，`method_scope=shared_preprocess`，`disk_bytes=documents.jsonl`                                                                   | `index_metrics.py`                               |
| F4 文档树/chunk   | `F4_chunk_tree`                         | `documents.jsonl`                        | `chunks.jsonl`、`document_trees.jsonl`                                                 | `stage_timing.jsonl`，`method_scope=shared_preprocess`，`disk_bytes` 指向两个输出所在目录                                                                        | `index_metrics.py`，另由 chunks 统计 chunk 数/token 分布 |
| F5 Chunk Index | `F5_chunk_index`                        | `chunks.jsonl`                           | ES chunk index                                                                        | `stage_timing.jsonl`，`method_scope=method_offline_index`；embedding calls/token 若能记录则写 `extra_metrics` 或 query/cache 日志                               | `method_summary.py` 汇总 offline cost              |
| F6 语义视图        | `F6_semantic_graph`                     | `chunks.jsonl`                           | `graph.semantic.llm.json`、`semantic_llm.extractions.jsonl`                            | `stage_timing.jsonl` + `semantic_llm.progress.jsonl` + `semantic_llm.extractions.jsonl`；可选 `stage_metrics/F6_semantic_graph.json` 记录完成 chunks、实体/关系数 | `index_metrics.py`                               |
| F7 结构视图        | `F7_structure_graph`                    | `chunks.jsonl`、`document_trees.jsonl`    | `graph.structure.json`                                                                | `stage_timing.jsonl`；如果用 LLM summary，必须额外记录 summary calls/token                                                                                      | `index_metrics.py`                               |
| F8 顺序视图        | `F8_sequence_graph`                     | `chunks.jsonl`                           | `graph.sequence.json`                                                                 | `stage_timing.jsonl`，`disk_bytes=graph.sequence.json`                                                                                                | `index_metrics.py`                               |
| F9 统一图         | `F9_unified_graph`                      | semantic/structure/sequence graphs       | `graph.unified.json`                                                                  | `stage_timing.jsonl`，`disk_bytes=graph.unified.json`                                                                                                 | `index_metrics.py`                               |
| F10 图对象同步      | `F10_graph_es_sync`                     | `graph.unified.json`                     | ES graph index                                                                        | `stage_timing.jsonl`；embedding/write 数量可写 `extra_metrics`                                                                                            | `method_summary.py`                              |
| F11 离线路标       | `F11_offline_signpost`                  | `graph.unified.json`、seed/result         | signpost JSON/stdout                                                                  | 单独检查时写 `stage_timing.jsonl`；正式 query 中写入 F13/F15 trace                                                                                               | `query_metrics.py` 从 trace 汇总                    |
| F12 在线 PPR     | `F12_online_ppr`                        | `graph.unified.json`、seed/result         | PPR signpost JSON/stdout                                                              | 单独检查时写 `stage_timing.jsonl`；正式 query 中写 PPR calls、latency、returned nodes                                                                             | `query_metrics.py`                               |
| F13 检索引擎       | `F13_retrieval` 或 `F13_retrieval_batch` | query、ES indexes、`graph.unified.json`    | `retrieval_result.json` 或 per-query retrieval results                                 | `stage_timing.jsonl`；批量实验必须保留 per-query retrieval latency/retrieved chunks/PPR info 到 trace/query log                                                | `query_metrics.py`                               |
| F14 ReadFile   | `F14_read_file`                         | documents、locate/file+line range         | read file JSON/stdout                                                                 | 单独检查时写 `stage_timing.jsonl`；正式 Agent 中写 read\_file tool calls、locate、返回行数到 trace                                                                     | `query_metrics.py`                               |
| F15 Agent      | `F15_agent_batch`                       | `questions.jsonl`、检索工具、documents/indexes | `predictions/<method>.jsonl`                                                          | `stage_timing.jsonl` + prediction trace；可选 `logs/<method>.query.jsonl` 记录每题 latency/tokens/tool calls                                                | `query_metrics.py`                               |
| F16 评估         | `F16_evaluation`                        | `predictions/<method>.jsonl`             | `basic_eval.json`、`llm_judge.jsonl`                                                   | `stage_timing.jsonl`；LLM judge 若使用模型，另记 judge calls/token                                                                                            | `query_metrics.py`/评估脚本                          |
| F17 指标汇总       | `F17_metrics_summary`                   | 前面所有 logs/artifacts                      | `index_metrics.json`、`query_metrics.json`、`method_summaries.json`、`cost_quality.json` | 不产生新的实验原始测量；只记录汇总脚本自身运行状态即可                                                                                                                          | 论文表格/图                                           |

最重要的判断标准：

- 一个阶段如果没有 `stage_timing.jsonl`，这个阶段的运行时间就没有正式实验依据。
- 一个 LLM 抽取/Agent 查询如果没有 cache/trace/query log，只能说明功能跑过，不能支撑 token/call/latency 细分分析。
- 汇总脚本输出的是“论文指标”，不是“原始实验记录”。原始记录必须在功能点运行时产生。

## 4. F0 配置与实验上下文

### 目标

读取 `.env` 和 `conf/service_conf.yaml`，定义研究系统需要的最小配置，不引入用户、租户、权限。

### 对应文件

| 文件                            | 作用                                       |
| ----------------------------- | ---------------------------------------- |
| `signpost/config/context.py`  | `PROJECT_ROOT`、路径解析、`ExperimentContext`。 |
| `signpost/config/settings.py` | `.env` 和 YAML 配置读取。                      |
| `signpost/config/smoke.py`    | 配置 smoke。                                |

### 输入

| 输入                       | 含义                      |
| ------------------------ | ----------------------- |
| `.env`                   | ECNU/OpenAI、ES 等环境变量。   |
| `conf/service_conf.yaml` | 服务配置。                   |
| `--namespace`            | 实验命名空间，例如 `legal_test`。 |

### 输出

Python 配置对象和 smoke 输出。

`ExperimentContext` 字段：

| 字段           | 类型        | 含义                                        |
| ------------ | --------- | ----------------------------------------- |
| `namespace`  | string    | 一次实验或数据集索引命名空间。                           |
| `dataset_id` | string    | 数据集 ID。                                   |
| `run_id`     | string    | 运行 ID，默认 `default`。                       |
| `output_dir` | Path/null | 输出目录；为空时为 `outputs/<namespace>/<run_id>`。 |

### 命令

正式实验/流程检查命令。会写 `outputs/legal_test/logs/stage_timing.jsonl`，并保存 stdout/stderr：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F0_config_smoke \
  --method-scope environment_check \
  --output-path outputs/legal_test/logs/F0_config_smoke.stdout.log \
  --stdout-log outputs/legal_test/logs/F0_config_smoke.stdout.log \
  --stderr-log outputs/legal_test/logs/F0_config_smoke.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.config.smoke --namespace legal_test
```

功能输出和测量输出：

| 输出              | 路径                                                   | 用途                |
| --------------- | ---------------------------------------------------- | ----------------- |
| 配置 smoke stdout | `outputs/legal_test/logs/F0_config_smoke.stdout.log` | 确认配置读取成功。         |
| 阶段总账            | `outputs/legal_test/logs/stage_timing.jsonl`         | 记录 F0 检查耗时、命令、状态。 |

内部裸命令，只用于理解模块调用方式，正式实验不要只跑这一条：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run -n signpost-re python -m signpost.config.smoke --namespace legal_test
```

## 5. F1 模型客户端

### 目标

统一 ECNU/OpenAI-compatible 模型调用，供 embedding、实体关系抽取、summary、Agent 和 LLM-as-Judge 使用。

### 对应文件

| 文件                       | 作用                                 |
| ------------------------ | ---------------------------------- |
| `signpost/llm/client.py` | chat、embedding、rerank HTTP client。 |
| `signpost/llm/smoke.py`  | 模型 smoke。                          |

### 输入

| 输入                    | 含义                                  |
| --------------------- | ----------------------------------- |
| `messages`            | chat messages。                      |
| `texts`               | embedding 文本数组。                     |
| `query` + `documents` | rerank 输入。                          |
| `.env`                | `ECNU_API_BASE`、`ECNU_API_KEY`、模型名。 |

### 输出

| 输出                | 含义            |
| ----------------- | ------------- |
| chat text         | LLM 回复文本。     |
| embedding vectors | 每段文本一个向量。     |
| rerank scores     | 每个候选文档的相关性分数。 |

默认模型环境变量：

| 变量                     | 默认值                    |
| ---------------------- | ---------------------- |
| `ECNU_CHAT_MODEL`      | `ecnu-plus`            |
| `ECNU_REASONING_MODEL` | `ecnu-max`             |
| `ECNU_EMBEDDING_MODEL` | `ecnu-embedding-small` |
| `ECNU_RERANK_MODEL`    | `ecnu-rerank`          |

配置优先级：

```text
shell 环境变量 > .env 文件
```

因此如果某个 key 达到每日次数限制，可以临时在命令前覆盖 `ECNU_API_KEY`，不需要把新 key 写进 `.env`，也避免把 key 记录进 `stage_timing.jsonl` 的内部命令字段。

### 命令

正式 smoke 命令。每个子功能单独记一行 stage log：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F1_llm_chat_smoke \
  --method-scope environment_check \
  --output-path outputs/legal_test/logs/F1_llm_chat_smoke.stdout.log \
  --stdout-log outputs/legal_test/logs/F1_llm_chat_smoke.stdout.log \
  --stderr-log outputs/legal_test/logs/F1_llm_chat_smoke.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.llm.smoke --chat

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F1_llm_embedding_smoke \
  --method-scope environment_check \
  --output-path outputs/legal_test/logs/F1_llm_embedding_smoke.stdout.log \
  --stdout-log outputs/legal_test/logs/F1_llm_embedding_smoke.stdout.log \
  --stderr-log outputs/legal_test/logs/F1_llm_embedding_smoke.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.llm.smoke --embedding

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F1_llm_rerank_smoke \
  --method-scope environment_check \
  --output-path outputs/legal_test/logs/F1_llm_rerank_smoke.stdout.log \
  --stdout-log outputs/legal_test/logs/F1_llm_rerank_smoke.stdout.log \
  --stderr-log outputs/legal_test/logs/F1_llm_rerank_smoke.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.llm.smoke --rerank
```

功能输出和测量输出：

| 输出                  | 路径                                           | 用途                             |
| ------------------- | -------------------------------------------- | ------------------------------ |
| smoke stdout/stderr | `outputs/legal_test/logs/F1_*.log`           | 检查 chat/embedding/rerank 是否可用。 |
| 阶段总账                | `outputs/legal_test/logs/stage_timing.jsonl` | 记录每个 smoke 调用耗时和状态。            |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.llm.smoke --chat
conda run -n signpost-re python -m signpost.llm.smoke --embedding
conda run -n signpost-re python -m signpost.llm.smoke --rerank
```

## 6. F2 存储服务

### 目标

提供实验所需的最小 ES 访问能力。当前主流程只强依赖 Elasticsearch；MinIO、Redis、PostgreSQL 不作为论文核心链路展开。

### 对应文件

| 文件                                  | 作用                                                      |
| ----------------------------------- | ------------------------------------------------------- |
| `signpost/storage/elasticsearch.py` | ES HTTP client、index create/delete、bulk、search、refresh。 |

### 输入

| 输入            | 含义                             |
| ------------- | ------------------------------ |
| `.env`        | `ELASTICSEARCH_URL` 等连接信息。     |
| index mapping | F5/F10 提供。                     |
| documents     | 待写入 ES 的 chunk 或 graph object。 |

### 输出

| 输出          | 含义                                                            |
| ----------- | ------------------------------------------------------------- |
| ES index    | `signpost-<namespace>-chunks` 或 `signpost-<namespace>-graph`。 |
| search JSON | ES 查询结果。                                                      |

### 命令

正式 storage smoke 命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F2_storage_es_smoke \
  --method-scope environment_check \
  --output-path outputs/legal_test/logs/F2_storage_es_smoke.stdout.log \
  --stdout-log outputs/legal_test/logs/F2_storage_es_smoke.stdout.log \
  --stderr-log outputs/legal_test/logs/F2_storage_es_smoke.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.storage.smoke --es
```

功能输出和测量输出：

| 输出                     | 路径                                                  | 用途                |
| ---------------------- | --------------------------------------------------- | ----------------- |
| ES smoke stdout/stderr | `outputs/legal_test/logs/F2_storage_es_smoke.*.log` | 检查 ES 是否可用。       |
| 阶段总账                   | `outputs/legal_test/logs/stage_timing.jsonl`        | 记录 F2 检查耗时、命令、状态。 |

## 7. F3 数据集准备

### 目标

把原始数据统一转换成文档语料 `raw_corpus.jsonl` 和问题集 `questions.jsonl`。这是所有后续阶段的入口。

### 对应文件

| 文件                                           | 作用                          |
| -------------------------------------------- | --------------------------- |
| `scripts/prepare_datasets.py`                | F3 主准备脚本。                   |
| `signpost/data/prepare.py`                   | package CLI 包装。             |
| `signpost/data/validate.py`                  | F3 输出校验。                    |
| `signpost/data/create_ultradomain_subset.py` | raw-level UltraDomain 子集抽取。 |

### F3 实际代码流

F3 不是直接读 `processed`，而是从 `datasets/raw` 生成 `datasets/processed`。入口命令虽然是：

```bash
python -m signpost.data.prepare
```

但实际执行逻辑在：

```text
scripts/prepare_datasets.py
```

`signpost/data/prepare.py` 的作用只是把脚本包装成 package module，方便统一用 `python -m signpost.data.prepare` 调用。

当前代码流：

```text
datasets/raw/ultradomain/<dataset>.jsonl
  -> prepare_ultradomain()
  -> datasets/processed/<dataset>/raw_corpus.jsonl
  -> datasets/processed/<dataset>/questions.jsonl

datasets/raw/graphrag-bench/<config>_corpus.json
datasets/raw/graphrag-bench/<config>_questions.json
  -> prepare_graphrag_bench()
  -> datasets/processed/graphrag-bench-<config>/raw_corpus.jsonl
  -> datasets/processed/graphrag-bench-<config>/questions.jsonl
```

也就是说，F3 的职责边界是：

```text
raw 原始数据
-> 去重、字段清洗、标准化 ID、拆分文档和问题
-> processed/raw_corpus.jsonl + processed/questions.jsonl
```

F3 不做：

- 文档行号解析。
- chunking。
- embedding。
- ES 写入。
- 图构建。

### UltraDomain raw 输入

路径：

```text
datasets/raw/ultradomain/<dataset>.jsonl
```

常见字段：

| 字段           | 类型            | 含义       |
| ------------ | ------------- | -------- |
| `_id`        | string        | 原始问题 ID。 |
| `input`      | string        | 问题文本。    |
| `context`    | string        | 文档全文。    |
| `context_id` | string        | 文档 ID。   |
| `answers`    | list/string   | 标准答案。    |
| `label`      | string        | 数据集标签。   |
| `length`     | number/string | 原始长度信息。  |
| `meta`       | object        | 原始元数据。   |

UltraDomain raw 到 processed 的关键转换规则：

| raw 字段       | processed 位置                                        | 说明                        |
| ------------ | --------------------------------------------------- | ------------------------- |
| `context_id` | `raw_corpus.doc_id`、`questions.doc_ids[]`           | 同一个 `context_id` 只生成一个文档。 |
| `context`    | `raw_corpus.text`                                   | 文档全文。                     |
| `input`      | `questions.question`                                | 问题文本。                     |
| `_id`        | `questions.question_id`、`questions.metadata.raw_id` | 原始问题 ID。                  |
| `answers`    | `questions.answer`、`questions.answers`              | 标准答案。                     |
| `label`      | `metadata.label`                                    | 原始标签。                     |
| `meta.title` | `raw_corpus.metadata.title`、文件名候选                   | 文档标题。                     |

如果 raw 数据中多个问题对应同一个 `context_id`，F3 会只保留一份文档到 `raw_corpus.jsonl`，同时在 `questions.jsonl` 中保留多条问题，每条问题用 `doc_ids` 指向该文档。

### GraphRAG-Bench raw 输入

路径：

```text
datasets/raw/graphrag-bench/medical_corpus.json
datasets/raw/graphrag-bench/medical_questions.json
datasets/raw/graphrag-bench/novel_corpus.json
datasets/raw/graphrag-bench/novel_questions.json
```

GraphRAG-Bench raw 到 processed 的关键转换规则：

| raw 来源                                                            | processed 位置                              | 说明       |
| ----------------------------------------------------------------- | ----------------------------------------- | -------- |
| corpus 中的 `id` / `doc_id` / `source`                              | `raw_corpus.doc_id`                       | 文档 ID。   |
| corpus 中的 `text` / `content` / `context` / `document` / `passage` | `raw_corpus.text`                         | 文档全文。    |
| questions 中的 `question` / `query` / `input`                       | `questions.question`                      | 问题文本。    |
| questions 中的 `answer`                                             | `questions.answer`、`questions.answers`    | 标准答案。    |
| questions 中的 `evidence` / `contexts`                              | `questions.rationale`、`metadata.evidence` | 证据或解释信息。 |
| questions 中的 `source`                                             | `questions.doc_ids[]`                     | 关联文档。    |

### 输出一：`raw_corpus.jsonl`

路径：

```text
datasets/processed/<dataset>/raw_corpus.jsonl
```

字段：

| 字段              | 类型     | 含义                        |
| --------------- | ------ | ------------------------- |
| `doc_id`        | string | 标准文档 ID。                  |
| `file_name`     | string | 稳定文件名，通常为 `<doc_id>.txt`。 |
| `source_path`   | string | 原始来源路径或来源描述。              |
| `source_format` | string | `text`、`jsonl` 等。         |
| `text`          | string | 文档全文。                     |
| `metadata`      | object | 数据集、来源、原始 doc id 等信息。     |

### 输出二：`questions.jsonl`

路径：

```text
datasets/processed/<dataset>/questions.jsonl
```

字段：

| 字段            | 类型            | 含义                |
| ------------- | ------------- | ----------------- |
| `question_id` | string        | 标准问题 ID。          |
| `question`    | string        | 问题文本。             |
| `answer`      | string/list   | 标准答案。             |
| `answers`     | list          | 多答案形式。            |
| `rationale`   | string        | 标准解释或证据说明；没有时可为空。 |
| `doc_ids`     | list\[string] | 该问题关联的文档 ID。      |
| `metadata`    | object        | 原始 ID、数据集、来源等信息。  |

### 命令：raw 到 processed

正式带日志命令。会同时得到功能输出、阶段总账和自动工件统计：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_data_prepare \
  --method-scope shared_preprocess \
  --output-path datasets/processed/legal_test/raw_corpus.jsonl \
  --disk-path datasets/processed/legal_test \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F3_data_prepare.json \
  --stdout-log outputs/legal_test/logs/F3_data_prepare.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_data_prepare.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.data.prepare --datasets legal_test

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_data_validate \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/raw_corpus.jsonl \
  --output-path outputs/legal_test/logs/F3_data_validate.stdout.log \
  --stdout-log outputs/legal_test/logs/F3_data_validate.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_data_validate.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.data.validate --dataset legal_test
```

功能输出和原始测量输出：

| 输出            | 路径                                                           | 用途                                       |
| ------------- | ------------------------------------------------------------ | ---------------------------------------- |
| 标准文档语料        | `datasets/processed/legal_test/raw_corpus.jsonl`             | F3.5 输入。                                 |
| 标准问题集         | `datasets/processed/legal_test/questions.jsonl`              | F15/F16 输入。                              |
| manifest      | `datasets/manifest.json`                                     | 数据准备记录。                                  |
| 阶段自定义指标       | `outputs/legal_test/logs/stage_metrics/F3_data_prepare.json` | `docs`、`questions`、`raw_text_chars`。     |
| 阶段总账          | `outputs/legal_test/logs/stage_timing.jsonl`                 | F3 wall time、命令、状态、产物大小、`extra_metrics`。 |
| stdout/stderr | `outputs/legal_test/logs/F3_data_prepare.*.log`              | 失败排查。                                    |

下面是内部裸命令，只用于理解模块调用方式；正式实验不要只跑裸命令，否则没有阶段总账：

准备 `legal_test`。输入是：

```text
datasets/raw/ultradomain/legal_test.jsonl
```

输出是：

```text
datasets/processed/legal_test/raw_corpus.jsonl
datasets/processed/legal_test/questions.jsonl
datasets/manifest.json
```

命令：

```bash
conda run -n signpost-re python -m signpost.data.prepare --datasets legal_test
conda run -n signpost-re python -m signpost.data.validate --dataset legal_test
```

准备 `legal_lite`。输入是：

```text
datasets/raw/ultradomain/legal_lite.jsonl
```

输出是：

```text
datasets/processed/legal_lite/raw_corpus.jsonl
datasets/processed/legal_lite/questions.jsonl
datasets/manifest.json
```

命令：

```bash
conda run -n signpost-re python -m signpost.data.prepare --datasets legal_lite
conda run -n signpost-re python -m signpost.data.validate --dataset legal_lite
```

一次准备多个数据集：

```bash
conda run -n signpost-re python -m signpost.data.prepare \
  --datasets legal_test legal_lite agriculture graphrag-bench-medical graphrag-bench-novel
```

只校验已经生成的 processed 文件，不重新生成：

```bash
conda run -n signpost-re python -m signpost.data.prepare \
  --datasets legal_test legal_lite \
  --validate-only
```

对内置下载数据集强制重新下载 raw，再生成 processed：

```bash
conda run -n signpost-re python -m signpost.data.prepare \
  --datasets agriculture legal graphrag-bench-medical graphrag-bench-novel \
  --force-download
```

注意：`legal_test` 和 `legal_lite` 是本地 raw-level 子集。它们的 raw 文件已经放在 `datasets/raw/ultradomain/` 下时，F3 会直接读取本地文件，不会重新下载。

重新生成 raw-level `legal_test`：

```bash
conda run -n signpost-re python -m signpost.data.create_ultradomain_subset \
  --source-dataset legal \
  --target-dataset legal_test \
  --doc-id legal_doc_b9eb62b7885ca06a
```

重新生成 raw-level `legal_lite` 的完整 doc list 见 `docs/legal_lite_subset_design.zh.md`。

## 8. F3.5 文档解析

### 目标

将 F3 的 `raw_corpus.jsonl` 转成带行号的 `documents.jsonl`，为 F4 chunk、F14 ReadFile 和证据定位提供基础。

### 对应文件

| 文件                                       | 作用                     |
| ---------------------------------------- | ---------------------- |
| `signpost/parsing/parser.py`             | raw row 到 document。    |
| `signpost/parsing/normalizer.py`         | Unicode NFKC、空白、标点规范化。 |
| `signpost/parsing/parse_documents.py`    | CLI。                   |
| `signpost/parsing/validate_documents.py` | 校验。                    |
| `signpost/parsing/io.py`                 | JSONL IO。              |

### 输入

```text
datasets/processed/<dataset>/raw_corpus.jsonl
```

### 输出：`documents.jsonl`

路径：

```text
datasets/processed/<dataset>/documents.jsonl
```

字段：

| 字段             | 类型            | 含义                   |
| -------------- | ------------- | -------------------- |
| `doc_id`       | string        | 文档 ID。               |
| `file_name`    | string        | 文件名。                 |
| `source_path`  | string        | 原始来源路径。              |
| `text`         | string        | 规范化后的全文。             |
| `lines`        | list\[object] | 带行号文本行。              |
| `placeholders` | list\[object] | 表格、图片等占位符；纯文本数据通常为空。 |
| `metadata`     | object        | 数据集和来源信息。            |

`lines[]` 字段：

| 字段        | 类型     | 含义          |
| --------- | ------ | ----------- |
| `line_no` | int    | 1-based 行号。 |
| `text`    | string | 该行文本。       |

`placeholders[]` 字段：

| 字段            | 含义                 |
| ------------- | ------------------ |
| `placeholder` | 正文中的占位符文本。         |
| `type`        | `table`、`image` 等。 |
| `line_no`     | 所在行。               |
| `raw`         | 原始内容。              |

### 命令

正式带日志命令：LLM summary 版本，用于论文正式结构视图：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_5_parse_normalize \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/raw_corpus.jsonl \
  --output-path datasets/processed/legal_test/documents.jsonl \
  --disk-path datasets/processed/legal_test/documents.jsonl \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F3_5_parse_normalize.json \
  --stdout-log outputs/legal_test/logs/F3_5_parse_normalize.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_5_parse_normalize.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.parsing.parse_documents \
    --input datasets/processed/legal_test/raw_corpus.jsonl \
    --output datasets/processed/legal_test/documents.jsonl

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_5_validate_documents \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path outputs/legal_test/logs/F3_5_validate_documents.stdout.log \
  --stdout-log outputs/legal_test/logs/F3_5_validate_documents.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_5_validate_documents.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.parsing.validate_documents \
    --input datasets/processed/legal_test/documents.jsonl
```

功能输出和原始测量输出：

| 输出            | 路径                                                                | 用途                                                            |
| ------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| 解析后文档         | `datasets/processed/legal_test/documents.jsonl`                   | F4/F14 输入，含行号。                                                |
| 阶段自定义指标       | `outputs/legal_test/logs/stage_metrics/F3_5_parse_normalize.json` | `documents`、`document_lines`、`document_chars`、`placeholders`。 |
| 阶段总账          | `outputs/legal_test/logs/stage_timing.jsonl`                      | F3.5 wall time、命令、状态、文件大小、`extra_metrics`。                    |
| stdout/stderr | `outputs/legal_test/logs/F3_5_*.log`                              | 失败排查。                                                         |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal_test/raw_corpus.jsonl \
  --output datasets/processed/legal_test/documents.jsonl

conda run -n signpost-re python -m signpost.parsing.validate_documents \
  --input datasets/processed/legal_test/documents.jsonl
```

## 9. F4 文档树与 Chunk

### 目标

识别章节标题，构建 document tree，并按 token 预算切出 chunk。F4 的输出是 F5/F6/F7/F8 的共同输入。

### 逻辑说明

F4 先做标题识别，再构建树，最后切 chunk：

```text
documents.jsonl
-> 确定性标题识别或可选 LLM 标题识别
-> document_trees.jsonl
-> 按章节子树和 token 预算生成 chunks.jsonl
```

如果某个章节子树超过 `--max-tokens`，会继续按行范围拆分；如果单行本身超过 token 预算，会在行内按词拆分，`metadata.merge` 记为 `split_long_line`。这只是 chunk 来源类型，不影响后续图和边的 schema。

### 对应文件

| 文件                               | 作用                      |
| -------------------------------- | ----------------------- |
| `signpost/chunking/headers.py`   | 确定性标题识别。                |
| `signpost/chunking/tree.py`      | 文档树。                    |
| `signpost/chunking/chunker.py`   | chunk 切分、overlap、超长行拆分。 |
| `signpost/chunking/tokenizer.py` | token 估计。               |
| `signpost/chunking/run.py`       | CLI。                    |
| `signpost/chunking/validate.py`  | 校验。                     |

### 输入

```text
datasets/processed/<dataset>/documents.jsonl
```

### 输出一：`chunks.jsonl`

字段：

| 字段              | 类型            | 含义                              |
| --------------- | ------------- | ------------------------------- |
| `chunk_id`      | string        | chunk ID，通常为 `<doc_id>_c00000`。 |
| `doc_id`        | string        | 所属文档。                           |
| `file_name`     | string        | 文件名。                            |
| `content`       | string        | chunk 文本，含章节路径和 `[CONTENT]` 分隔。 |
| `start_line`    | int           | 起始行。                            |
| `end_line`      | int           | 结束行。                            |
| `section_path`  | list\[string] | 章节路径。                           |
| `prev_chunk_id` | string/null   | 同一文档前一个 chunk。                  |
| `next_chunk_id` | string/null   | 同一文档后一个 chunk。                  |
| `metadata`      | object        | chunk 参数和统计。                    |

`metadata` 常见字段：

| 字段            | 含义                                              |
| ------------- | ----------------------------------------------- |
| `merge`       | 生成方式：`subtree`、`split_range`、`split_long_line`。 |
| `chunk_index` | 文档内序号。                                          |
| `token_count` | token 估计数。                                      |

### 输出二：`document_trees.jsonl`

字段：

| 字段          | 类型            | 含义      |
| ----------- | ------------- | ------- |
| `doc_id`    | string        | 文档 ID。  |
| `file_name` | string        | 文件名。    |
| `headers`   | list\[object] | 识别出的标题。 |
| `tree`      | object        | 树形章节结构。 |

`headers[]` 字段：

| 字段              | 含义     |
| --------------- | ------ |
| `title`         | 标题文本。  |
| `level`         | 标题层级。  |
| `content_start` | 内容起始行。 |
| `content_end`   | 内容结束行。 |

`tree` 节点字段：

| 字段                        | 含义                |
| ------------------------- | ----------------- |
| `title`                   | 标题，根节点为 `[ROOT]`。 |
| `level`                   | 层级。               |
| `start_line` / `end_line` | 覆盖行范围。            |
| `children`                | 子节点。              |

### 命令

正式带日志命令。这是你跑 F4 时应该使用的命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F4_chunk_tree \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path datasets/processed/legal_test/chunks.jsonl \
  --disk-path datasets/processed/legal_test \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F4_chunk_tree.json \
  --stdout-log outputs/legal_test/logs/F4_chunk_tree.stdout.log \
  --stderr-log outputs/legal_test/logs/F4_chunk_tree.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.chunking.run \
    --input datasets/processed/legal_test/documents.jsonl \
    --output datasets/processed/legal_test/chunks.jsonl \
    --tree-output datasets/processed/legal_test/document_trees.jsonl \
    --max-tokens 1200 \
    --overlap-tokens 100

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F4_validate_chunks \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path outputs/legal_test/logs/F4_validate_chunks.stdout.log \
  --stdout-log outputs/legal_test/logs/F4_validate_chunks.stdout.log \
  --stderr-log outputs/legal_test/logs/F4_validate_chunks.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.chunking.validate \
    --chunks datasets/processed/legal_test/chunks.jsonl
```

功能输出和原始测量输出：

| 输出             | 路径                                                         | 用途                                                                                    |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| chunks         | `datasets/processed/legal_test/chunks.jsonl`               | F5/F6/F7/F8 输入。                                                                       |
| document trees | `datasets/processed/legal_test/document_trees.jsonl`       | F7 结构视图输入，也用于论文报告章节树统计。                                                               |
| 阶段自定义指标        | `outputs/legal_test/logs/stage_metrics/F4_chunk_tree.json` | `chunks`、`chunk_docs`、`chunk_tokens_sum/mean/max`、`chunks_merge_*`、`trees`、`headers`。 |
| 阶段总账           | `outputs/legal_test/logs/stage_timing.jsonl`               | F4 wall time、命令、状态、产物大小、`extra_metrics`。                                              |
| stdout/stderr  | `outputs/legal_test/logs/F4_*.log`                         | 失败排查。                                                                                 |

内部裸命令，只用于理解模块调用方式；正式实验不要只跑裸命令：

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_test/documents.jsonl \
  --output datasets/processed/legal_test/chunks.jsonl \
  --tree-output datasets/processed/legal_test/document_trees.jsonl \
  --max-tokens 1200 \
  --overlap-tokens 100

conda run -n signpost-re python -m signpost.chunking.validate \
  --chunks datasets/processed/legal_test/chunks.jsonl
```

可选 LLM 双路径章节识别说明：

`--use-llm` 只影响 F4 的“标题识别/章节树构建”这一步。它不会生成 embedding，不会写 ES，也不会抽取 F6 的实体关系。

当前 F4 有两种章节识别路径：

| 路径          | 做什么                                                           | 是否建议主实验使用               |
| ----------- | ------------------------------------------------------------- | ----------------------- |
| 默认确定性识别     | 用 Markdown 标题、中文“第 x 章/条”、英文 `ARTICLE`/`Section`、多级编号等规则识别标题。 | 建议默认使用。稳定、便宜、可复现。       |
| `--use-llm` | 短文档让 LLM 转 Markdown 再解析标题；长文档分窗口让 LLM 抽取标题 JSON。              | 可选，不是必须。适合标题格式混乱时做质量对照。 |

你不使用 `--use-llm` 完全可以。对论文实验而言，默认确定性 F4 更容易保证可复现，也不会消耗 LLM 额度。只有当你发现 `document_trees.jsonl` 明显识别不到章节，或者想报告“LLM 章节识别 vs 确定性章节识别”的消融实验时，才需要跑这个可选命令。

可选 LLM 双路径章节识别的正式带日志命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F4_chunk_tree_llm_headers \
  --method-scope shared_preprocess \
  --method llm_headers \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path datasets/processed/legal_test/chunks.jsonl \
  --disk-path datasets/processed/legal_test \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F4_chunk_tree_llm_headers.json \
  --stdout-log outputs/legal_test/logs/F4_chunk_tree_llm_headers.stdout.log \
  --stderr-log outputs/legal_test/logs/F4_chunk_tree_llm_headers.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.chunking.run \
    --input datasets/processed/legal_test/documents.jsonl \
    --output datasets/processed/legal_test/chunks.jsonl \
    --tree-output datasets/processed/legal_test/document_trees.jsonl \
    --max-tokens 1200 \
    --overlap-tokens 100 \
    --use-llm
```

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_test/documents.jsonl \
  --output datasets/processed/legal_test/chunks.jsonl \
  --tree-output datasets/processed/legal_test/document_trees.jsonl \
  --use-llm
```

## 10. F5 Chunk Index

### 目标

对 chunk 生成 embedding，写入 Elasticsearch，支持 BM25、dense vector、hybrid 检索。hybrid 当前使用 BM25 和 dense 的 Reciprocal Rank Fusion 合并。

### 对应文件

| 文件                                   | 作用                                |
| ------------------------------------ | --------------------------------- |
| `signpost/indexing/embedding.py`     | ECNU embedding 和 hash embedding。  |
| `signpost/indexing/chunk_schema.py`  | ES index name、mapping、chunk 文档转换。 |
| `signpost/indexing/chunk_index.py`   | chunks 写入 ES。                     |
| `signpost/retrieval/chunk_search.py` | chunk 检索。                         |

### 输入

```text
datasets/processed/<dataset>/chunks.jsonl
```

### 输出

ES index：

```text
signpost-<namespace>-chunks
```

ES 文档字段：

| 字段                                | 类型            | 含义                |
| --------------------------------- | ------------- | ----------------- |
| `id`                              | string        | chunk ID。         |
| `type`                            | string        | 固定为 `chunk`。      |
| `namespace`                       | string        | 实验命名空间。           |
| `dataset_id`                      | string        | 数据集 ID。           |
| `doc_id`                          | string        | 文档 ID。            |
| `file_name`                       | string        | 文件名。              |
| `content`                         | text          | chunk 文本，用于 BM25。 |
| `content_vector`                  | dense\_vector | embedding，用于向量检索。 |
| `start_line` / `end_line`         | int           | 行范围。              |
| `section_path`                    | keyword\[]    | 章节路径。             |
| `prev_chunk_id` / `next_chunk_id` | string/null   | 顺序邻接。             |
| `chunk_index`                     | int           | 文档内序号。            |
| `token_count`                     | int           | token 估计。         |
| `metadata`                        | object        | 其他字段。             |

### 命令

### hash 与 ECNU 怎么选

| provider | 做什么                                            | 什么时候用                                  |
| -------- | ---------------------------------------------- | -------------------------------------- |
| `hash`   | 本地确定性伪 embedding，不调用模型，不代表真实语义向量质量。            | smoke test、调通 ES、调通 F5-F16 流程、调试日志和指标。 |
| `ecnu`   | 调 ECNU embedding 模型，把 chunk 转成真实 dense vector。 | 正式实验、论文结果、检索质量测试、与 dense/hybrid 相关的测评。 |

BM25 只使用 `content` 文本字段，不依赖 embedding；dense 和 hybrid 依赖 `content_vector`。如果用 `hash`，dense/hybrid 只是流程可运行，不应用来报告真实检索效果。如果要报告论文实验结果，F5 应使用 `ecnu`。

调试带日志命令：hash smoke。用于快速验证写 ES、mapping、bulk、日志和后续流程，不用于论文正式结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F5_chunk_index \
  --method-scope method_offline_index \
  --method hybrid_rag \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path signpost-legal_test-hash-chunks \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F5_chunk_index.json \
  --stdout-log outputs/legal_test/logs/F5_chunk_index.stdout.log \
  --stderr-log outputs/legal_test/logs/F5_chunk_index.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace legal_test-hash \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --embedding-provider hash \
    --hash-dimensions 128 \
    --recreate
```

正式带日志命令：ECNU embedding。用于正式实验和论文结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F5_chunk_index_ecnu \
  --method-scope method_offline_index \
  --method hybrid_rag \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path signpost-legal_test-ecnu-chunks \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F5_chunk_index_ecnu.json \
  --stdout-log outputs/legal_test/logs/F5_chunk_index_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F5_chunk_index_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace legal_test-ecnu \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --embedding-provider ecnu \
    --batch-size 4 \
    --progress-every 10 \
    --embedding-retries 5 \
    --retry-sleep 3 \
    --recreate
```

如果当前 `.env` 里的 ECNU key 达到每日限制，可以临时换 key 重跑。不要把真实 key 写进文档或提交到仓库：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F5_chunk_index_ecnu \
  --method-scope method_offline_index \
  --method hybrid_rag \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path signpost-legal_test-ecnu-chunks \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F5_chunk_index_ecnu.json \
  --stdout-log outputs/legal_test/logs/F5_chunk_index_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F5_chunk_index_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace legal_test-ecnu \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --embedding-provider ecnu \
    --batch-size 4 \
    --progress-every 10 \
    --embedding-retries 5 \
    --retry-sleep 3 \
    --recreate
```

功能输出和原始测量输出：

| 输出             | 路径                                                                    | 用途                                                                              |
| -------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| ES chunk index | `signpost-legal_test-hash-chunks` 或 `signpost-legal_test-ecnu-chunks` | BM25/dense/hybrid 检索输入。                                                         |
| hash 阶段自定义指标   | `outputs/legal_test/logs/stage_metrics/F5_chunk_index.json`           | 从输入 chunks 统计 `input_chunks`、`input_chunk_tokens_*`，作为 indexed chunks/token 基线。 |
| ECNU 阶段自定义指标   | `outputs/legal_test/logs/stage_metrics/F5_chunk_index_ecnu.json`      | 同上，用于正式 embedding 索引阶段。                                                         |
| 阶段总账           | `outputs/legal_test/logs/stage_timing.jsonl`                          | F5 wall time、命令、状态、`extra_metrics`。                                             |
| stdout/stderr  | `outputs/legal_test/logs/F5_chunk_index*.log`                         | embedding/index 写入失败排查。                                                         |

内部裸命令：

ECNU embedding：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace legal_test-ecnu \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --embedding-provider ecnu \
  --batch-size 4 \
  --progress-every 10 \
  --embedding-retries 5 \
  --retry-sleep 3 \
  --recreate
```

本地 hash smoke，不消耗模型额度：

```bash
conda run -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace legal_test-hash \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --embedding-provider hash \
  --hash-dimensions 128 \
  --recreate
```

检索：

```bash
conda run -n signpost-re python -m signpost.retrieval.chunk_search \
  --namespace legal_test-ecnu \
  --query "What is the purpose of the agreement?" \
  --mode hybrid \
  --top-k 5
```

## 11. F6 语义视图

### 目标

对每个 chunk 抽取实体和实体关系，构建语义图。真实实验使用 LLM；流程测试可以使用 deterministic extractor。

### 逻辑说明

F6 是逐 chunk 调 LLM 抽图元素，不是一次性把全文给 LLM。每个 chunk 产生：

```text
entities: name/type/description
relations: source_entity/target_entity/description/keywords/weight
```

然后代码合并同名实体、合并同一实体对关系，并生成 entity 到 chunk 的 source 边。最终先保存为 JSON 图文件，F10 才同步到 ES。

`gleaning-rounds` 是补充抽取轮数。第一轮抽完后，后续轮次要求模型检查是否遗漏实体关系。轮数越大，召回可能更高，但 LLM 调用更多、时间更长。

### 对应文件

| 文件                                        | 作用                                             |
| ----------------------------------------- | ---------------------------------------------- |
| `signpost/indexing/semantic_extractor.py` | LLM/deterministic extractor、JSON 解析、cache 序列化。 |
| `signpost/graph/semantic.py`              | 合并 entity、relation、source edge。                |
| `signpost/indexing/semantic_graph.py`     | CLI，支持 progress、cache、retry、timeout。           |

### 输入

```text
datasets/processed/<dataset>/chunks.jsonl
```

### 输出一：`graph.semantic.json` 或 `graph.semantic.llm.json`

顶层字段：

| 字段         | 类型            | 含义                                                             |
| ---------- | ------------- | -------------------------------------------------------------- |
| `metadata` | object        | namespace、graph\_type、chunks、entities、relations、source\_edges。 |
| `nodes`    | list\[object] | entity 节点和 chunk 节点。                                           |
| `edges`    | list\[object] | semantic/source 边。                                             |

Entity 节点字段：

| 字段                 | 含义                                     |
| ------------------ | -------------------------------------- |
| `node_id`          | `entity:<hash>`。                       |
| `node_type`        | `entity`。                              |
| `name`             | 实体名。                                   |
| `entity_type`      | 实体类型。                                  |
| `description`      | 实体描述，来自 LLM 或 deterministic extractor。 |
| `source_chunk_ids` | 来源 chunks。                             |
| `source_locates`   | 来源行号，例如 `file.txt:L10-L20`。            |
| `source_mapping`   | 逐来源证据字典，key 为 `doc_id:chunk_id`。       |

Semantic relation 边字段：

| 字段                  | 含义                                            |
| ------------------- | --------------------------------------------- |
| `source` / `target` | 两端 entity node\_id。                           |
| `edge_type`         | 语义图内为 `semantic_relation`，统一图中规范为 `semantic`。 |
| `relation_types`    | 关系关键词。                                        |
| `description`       | 关系描述。                                         |
| `weight`            | 权重。                                           |
| `source_chunk_ids`  | 来源 chunks。                                    |
| `source_locates`    | 来源行号。                                         |
| `source_mapping`    | 逐来源证据。                                        |

Source 边字段：

| 字段               | 含义                  |
| ---------------- | ------------------- |
| `source`         | entity node\_id。    |
| `target`         | `chunk:<chunk_id>`。 |
| `edge_type`      | `source`。           |
| `source_locates` | 来源行号。               |

### 输出二：extraction cache

路径：

```text
datasets/processed/<dataset>/semantic_llm.extractions.jsonl
```

字段：

| 字段                        | 含义            |
| ------------------------- | ------------- |
| `chunk_id`                | 已完成抽取的 chunk。 |
| `doc_id`                  | 文档 ID。        |
| `file_name`               | 文件名。          |
| `start_line` / `end_line` | chunk 行范围。    |
| `extraction.entities`     | 本 chunk 抽取实体。 |
| `extraction.relations`    | 本 chunk 抽取关系。 |

### 命令

正式带日志命令。真实论文语义视图必须用这个形式跑；它会同时写阶段总账、阶段指标、per-chunk progress 和 extraction cache：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.llm.json \
  --disk-path datasets/processed/legal_test/graph.semantic.llm.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F6_semantic_graph.json \
  --stdout-log outputs/legal_test/logs/F6_semantic_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F6_semantic_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.semantic_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --progress-every 10 \
    --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
    --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
    --llm-retries 5 \
    --retry-sleep 3 \
    --llm-timeout 180
```

如果 stderr 中出现：

```text
HTTP 429: 请求过于频繁，超过应用的每天次数限制
```

说明当前 ECNU key 的每日额度已经用完。可以临时换 key 重跑。不要把真实 key 写进文档或提交到仓库：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.llm.json \
  --disk-path datasets/processed/legal_test/graph.semantic.llm.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F6_semantic_graph.json \
  --stdout-log outputs/legal_test/logs/F6_semantic_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F6_semantic_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.semantic_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --progress-every 10 \
    --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
    --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
    --llm-retries 5 \
    --retry-sleep 3 \
    --llm-timeout 180
```

功能输出和原始测量输出：

| 输出             | 路径                                                             | 用途                                                                                                |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 语义图            | `datasets/processed/legal_test/graph.semantic.llm.json`        | F9 合并输入。                                                                                          |
| per-chunk 抽取缓存 | `datasets/processed/legal_test/semantic_llm.extractions.jsonl` | 断点续跑；统计 completed chunks、实体/关系、估算 LLM calls。                                                      |
| per-chunk 进度   | `datasets/processed/legal_test/semantic_llm.progress.jsonl`    | 查看跑到哪个 chunk、是否失败、是否重试。                                                                           |
| 阶段自定义指标        | `outputs/legal_test/logs/stage_metrics/F6_semantic_graph.json` | `graph_nodes/edges`、`semantic_completed_chunks`、`entities_before_merge`、`relations_before_merge`。 |
| 阶段总账           | `outputs/legal_test/logs/stage_timing.jsonl`                   | F6 wall time、命令、状态、图文件大小、`extra_metrics`。                                                         |
| stdout/stderr  | `outputs/legal_test/logs/F6_semantic_graph.*.log`              | 长任务排查；不要替代 progress/cache。                                                                        |

运行中查看进度：

```bash
tail -f datasets/processed/legal_test/semantic_llm.progress.jsonl
```

查看 stdout 中的人类可读进度：

```bash
tail -f outputs/legal_test/logs/F6_semantic_graph.stdout.log
```

查看是否有报错：

```bash
tail -f outputs/legal_test/logs/F6_semantic_graph.stderr.log
```

统计当前已经缓存了多少个 chunk：

```bash
wc -l datasets/processed/legal_test/semantic_llm.extractions.jsonl
```

内部裸命令：

Deterministic 调试：

```bash
conda run -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace legal_test \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --output datasets/processed/legal_test/graph.semantic.json \
  --extractor deterministic
```

LLM，可断点续跑：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace legal_test-llm \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --output datasets/processed/legal_test/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --progress-every 10 \
  --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
  --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --llm-retries 5 \
  --retry-sleep 3 \
  --llm-timeout 180
```

校验和查看：

```bash
conda run -n signpost-re python -m signpost.graph.validate \
  --graph datasets/processed/legal_test/graph.semantic.llm.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph datasets/processed/legal_test/graph.semantic.llm.json
```

## 12. F7 结构视图 / RAPTOR

### 目标

利用 F4 的 `document_trees.jsonl` 和 `chunks.jsonl` 生成结构图。结构图包含 summary 节点和父子结构边，用于表示章节层级和聚合摘要。

### 对应文件

| 文件                                     | 作用                                |
| -------------------------------------- | --------------------------------- |
| `signpost/graph/structure.py`          | 构建 summary/chunk 节点和 structure 边。 |
| `signpost/indexing/summarizer.py`      | deterministic/LLM summarizer。     |
| `signpost/indexing/structure_graph.py` | CLI。                              |

### 输入

| 文件                                                  | 含义        |
| --------------------------------------------------- | --------- |
| `datasets/processed/<dataset>/chunks.jsonl`         | F4 chunk。 |
| `datasets/processed/<dataset>/document_trees.jsonl` | F4 文档树。   |

### 输出：`graph.structure.json`

顶层字段：

| 字段         | 含义                                               |
| ---------- | ------------------------------------------------ |
| `metadata` | namespace、chunks、raptor\_nodes、structure\_edges。 |
| `nodes`    | summary 节点和 chunk 节点。                            |
| `edges`    | structure 边。                                     |

Summary 节点字段：

| 字段                 | 含义                |
| ------------------ | ----------------- |
| `node_id`          | `summary:<hash>`。 |
| `node_type`        | `summary`。        |
| `title`            | 章节标题。             |
| `content`          | 摘要内容。             |
| `level`            | 层级。               |
| `section_path`     | 章节路径。             |
| `source_chunk_ids` | 覆盖 chunks。        |
| `source_locates`   | 覆盖行号。             |

Structure 边字段：

| 字段          | 含义                 |
| ----------- | ------------------ |
| `source`    | 父 summary。         |
| `target`    | 子 summary 或 chunk。 |
| `edge_type` | `structure`。       |
| `weight`    | 权重。                |

### 命令

正式带日志命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F7_structure_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.structure.json \
  --disk-path datasets/processed/legal_test/graph.structure.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F7_structure_graph.json \
  --stdout-log outputs/legal_test/logs/F7_structure_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F7_structure_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --document-trees datasets/processed/legal_test/document_trees.jsonl \
    --output datasets/processed/legal_test/graph.structure.json \
    --summarizer llm \
    --max-summary-tokens 512 \
    --cluster-token-budget 4096
```

可选对照命令：如果需要保留 deterministic summary 对照，可输出到 `graph.structure.det.json`，不要用于论文正式结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F7_structure_graph_deterministic \
  --method-scope method_offline_index \
  --method signpost_debug \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.structure.det.json \
  --disk-path datasets/processed/legal_test/graph.structure.det.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F7_structure_graph_deterministic.json \
  --stdout-log outputs/legal_test/logs/F7_structure_graph_deterministic.stdout.log \
  --stderr-log outputs/legal_test/logs/F7_structure_graph_deterministic.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --document-trees datasets/processed/legal_test/document_trees.jsonl \
    --output datasets/processed/legal_test/graph.structure.det.json \
    --summarizer deterministic
```

如果 ECNU key 达到每日限制，可以临时覆盖 key 重跑。不要把真实 key 写进文档或提交到仓库：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F7_structure_graph \
  --method-scope method_offline_index \
  --method signpost_llm_summary \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.structure.json \
  --disk-path datasets/processed/legal_test/graph.structure.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F7_structure_graph.json \
  --stdout-log outputs/legal_test/logs/F7_structure_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F7_structure_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --document-trees datasets/processed/legal_test/document_trees.jsonl \
    --output datasets/processed/legal_test/graph.structure.json \
    --summarizer llm \
    --max-summary-tokens 512 \
    --cluster-token-budget 4096
```

功能输出和原始测量输出：

| 输出                  | 路径                                                                          | 用途                                        |
| ------------------- | --------------------------------------------------------------------------- | ----------------------------------------- |
| 结构图                 | `datasets/processed/legal_test/graph.structure.json`                        | F9 合并输入。                                  |
| deterministic 对照结构图 | `datasets/processed/legal_test/graph.structure.det.json`                    | 只用于调试/对照，不作为论文正式结构视图。                     |
| 阶段自定义指标             | `outputs/legal_test/logs/stage_metrics/F7_structure_graph.json`             | summary/chunk 节点数、structure 边数等图统计。       |
| LLM summary 阶段指标    | `outputs/legal_test/logs/stage_metrics/F7_structure_graph_llm_summary.json` | LLM summary 版本的图统计。                       |
| 阶段总账                | `outputs/legal_test/logs/stage_timing.jsonl`                                | F7 wall time、命令、状态、图文件大小、`extra_metrics`。 |
| stdout/stderr       | `outputs/legal_test/logs/F7_structure_graph*.log`                           | 失败排查。                                     |

内部裸命令：

LLM summary：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace legal_test-llm \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --document-trees datasets/processed/legal_test/document_trees.jsonl \
  --output datasets/processed/legal_test/graph.structure.json \
  --summarizer llm \
  --max-summary-tokens 512 \
  --cluster-token-budget 4096
```

Deterministic 调试：

```bash
conda run -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace legal_test \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --document-trees datasets/processed/legal_test/document_trees.jsonl \
  --output datasets/processed/legal_test/graph.structure.det.json \
  --summarizer deterministic
```

## 13. F8 顺序视图

### 目标

根据 chunk 的 `prev_chunk_id` / `next_chunk_id` 建立顺序边，支持检索后扩展前后文。

### 对应文件

| 文件                                       | 作用                 |
| ---------------------------------------- | ------------------ |
| `signpost/graph/sequence.py`             | 构建 sequence graph。 |
| `signpost/indexing/sequence_graph.py`    | CLI。               |
| `signpost/retrieval/sequence_context.py` | 基于顺序图扩展上下文。        |

### 输入

```text
datasets/processed/<dataset>/chunks.jsonl
```

### 输出：`graph.sequence.json`

顶层字段：

| 字段         | 含义                                          |
| ---------- | ------------------------------------------- |
| `metadata` | namespace、chunks、documents、sequence\_edges。 |
| `nodes`    | chunk 节点。                                   |
| `edges`    | sequence 边。                                 |

Chunk 节点字段：

| 字段                        | 含义                  |
| ------------------------- | ------------------- |
| `node_id`                 | `chunk:<chunk_id>`。 |
| `node_type`               | `chunk`。            |
| `chunk_id`                | F4 chunk ID。        |
| `doc_id`                  | 文档 ID。              |
| `file_name`               | 文件名。                |
| `start_line` / `end_line` | 行范围。                |
| `section_path`            | 章节路径。               |

Sequence 边字段：

| 字段                  | 含义                 |
| ------------------- | ------------------ |
| `source` / `target` | 相邻 chunk node\_id。 |
| `edge_type`         | `sequence`。        |
| `direction`         | `next` 或 `prev`。   |
| `weight`            | 权重。                |

### 命令

正式带日志命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F8_sequence_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.sequence.json \
  --disk-path datasets/processed/legal_test/graph.sequence.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F8_sequence_graph.json \
  --stdout-log outputs/legal_test/logs/F8_sequence_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F8_sequence_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.sequence_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.sequence.json
```

功能输出和原始测量输出：

| 输出            | 路径                                                             | 用途                                        |
| ------------- | -------------------------------------------------------------- | ----------------------------------------- |
| 顺序图           | `datasets/processed/legal_test/graph.sequence.json`            | F9 合并输入、F13/F15 顺序上下文。                    |
| 阶段自定义指标       | `outputs/legal_test/logs/stage_metrics/F8_sequence_graph.json` | chunk 节点数、sequence 边数。                    |
| 阶段总账          | `outputs/legal_test/logs/stage_timing.jsonl`                   | F8 wall time、命令、状态、图文件大小、`extra_metrics`。 |
| stdout/stderr | `outputs/legal_test/logs/F8_sequence_graph.*.log`              | 失败排查。                                     |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.indexing.sequence_graph \
  --namespace legal_test \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --output datasets/processed/legal_test/graph.sequence.json

conda run -n signpost-re python -m signpost.retrieval.sequence_context \
  --graph datasets/processed/legal_test/graph.sequence.json \
  --chunk-id legal_doc_b9eb62b7885ca06a_c00000 \
  --before 1 \
  --after 1
```

## 14. F9 多视图统一图

### 目标

合并 F6 语义图、F7 结构图、F8 顺序图，形成统一图 `graph.unified.json`。F11/F12/F13/F15 都基于该图做导航或检索增强。

### 对应文件

| 文件                           | 作用        |
| ---------------------------- | --------- |
| `signpost/graph/unified.py`  | 合并、保存、校验。 |
| `signpost/graph/merge.py`    | CLI。      |
| `signpost/graph/validate.py` | 图校验。      |
| `signpost/graph/inspect.py`  | 图摘要。      |

### 输入

| 文件                                                | 含义   |
| ------------------------------------------------- | ---- |
| `graph.semantic.json` 或 `graph.semantic.llm.json` | 语义图。 |
| `graph.structure.json`                            | 结构图。 |
| `graph.sequence.json`                             | 顺序图。 |

### 输出：`graph.unified.json`

顶层字段：

| 字段         | 含义                                  |
| ---------- | ----------------------------------- |
| `metadata` | namespace、graph\_type、节点数、边数、来源图统计。 |
| `nodes`    | 合并节点。                               |
| `edges`    | 合并边。                                |

节点类型：

| `node_type` | 含义            |
| ----------- | ------------- |
| `chunk`     | 文本块。          |
| `entity`    | 语义实体。         |
| `summary`   | 结构/RAPTOR 摘要。 |

边类型：

| `edge_type` | 含义                   |
| ----------- | -------------------- |
| `semantic`  | entity-entity 语义关系。  |
| `source`    | entity 到 chunk 来源边。  |
| `structure` | summary/chunk 父子结构边。 |
| `sequence`  | chunk 顺序边。           |

### 命令

正式带日志命令：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F9_unified_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.semantic.llm.json \
  --output-path datasets/processed/legal_test/graph.unified.json \
  --disk-path datasets/processed/legal_test/graph.unified.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F9_unified_graph.json \
  --stdout-log outputs/legal_test/logs/F9_unified_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F9_unified_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.graph.merge \
    --namespace legal_test \
    --semantic datasets/processed/legal_test/graph.semantic.llm.json \
    --structure datasets/processed/legal_test/graph.structure.json \
    --sequence datasets/processed/legal_test/graph.sequence.json \
    --output datasets/processed/legal_test/graph.unified.json
```

功能输出和原始测量输出：

| 输出            | 路径                                                            | 用途                                        |
| ------------- | ------------------------------------------------------------- | ----------------------------------------- |
| 统一图           | `datasets/processed/legal_test/graph.unified.json`            | F10/F11/F12/F13/F15 输入。                   |
| 阶段自定义指标       | `outputs/legal_test/logs/stage_metrics/F9_unified_graph.json` | 统一图节点/边数、各类型节点/边数量。                       |
| 阶段总账          | `outputs/legal_test/logs/stage_timing.jsonl`                  | F9 wall time、命令、状态、图文件大小、`extra_metrics`。 |
| stdout/stderr | `outputs/legal_test/logs/F9_unified_graph.*.log`              | 失败排查。                                     |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.graph.merge \
  --namespace legal_test \
  --semantic datasets/processed/legal_test/graph.semantic.json \
  --structure datasets/processed/legal_test/graph.structure.json \
  --sequence datasets/processed/legal_test/graph.sequence.json \
  --output datasets/processed/legal_test/graph.unified.json

conda run -n signpost-re python -m signpost.graph.validate \
  --graph datasets/processed/legal_test/graph.unified.json

conda run -n signpost-re python -m signpost.graph.inspect \
  --graph datasets/processed/legal_test/graph.unified.json
```

## 15. F10 图对象同步到 ES

### 目标

把 unified graph 中的 entity、relation、summary 写入 Elasticsearch，使图对象可以 BM25、dense、hybrid 检索。

### 对应文件

| 文件                                   | 作用               |
| ------------------------------------ | ---------------- |
| `signpost/indexing/graph_schema.py`  | 图对象 mapping 和转换。 |
| `signpost/indexing/graph_es_sync.py` | 图对象写入 ES。        |
| `signpost/retrieval/graph_search.py` | 图对象检索。           |

### 输入

```text
datasets/processed/<dataset>/graph.unified.json
```

### 输出

ES index：

```text
signpost-<namespace>-graph
```

ES 文档字段：

| 字段                                  | 类型            | 含义                                   |
| ----------------------------------- | ------------- | ------------------------------------ |
| `id`                                | string        | 图对象 ID。                              |
| `type`                              | string        | `entity`、`relation`、`summary`。       |
| `namespace`                         | string        | 命名空间。                                |
| `title`                             | text          | 名称或标题。                               |
| `content`                           | text          | 描述、摘要或关系文本。                          |
| `content_vector`                    | dense\_vector | embedding。                           |
| `source_node_id` / `target_node_id` | string        | relation 端点。                         |
| `source_chunk_ids`                  | keyword\[]    | 来源 chunks。                           |
| `source_locates`                    | keyword\[]    | 来源行号。                                |
| `metadata`                          | object        | 原始节点/边属性，作为 `_source` 保存但不展开建立子字段索引。 |

注意：`metadata.source_mapping` 这类字段可能包含大量动态 key。F10 的 ES mapping 会把 `metadata` 设置为 `enabled: false`，避免 ES 把每个证据 key 都展开成 mapping 字段并触发默认 `index.mapping.total_fields.limit=1000`。这不影响图对象检索，因为检索使用的是 `title`、`name`、`content`、`object_type`、`source_chunk_ids`、`source_locates` 和 `content_vector` 等显式字段。

### 命令

F10 的 `ecnu` 路径调用的是 ECNU embedding 模型，不是 chat LLM。它会把统一图里的 entity、relation、summary 文本转成向量，写入 ES graph index。`hash` 只用于 smoke test，不用于论文真实图检索效果。

调试带日志命令：hash smoke。用于快速验证 graph index mapping、bulk 写入和日志链路，不用于论文正式结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_es_sync \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path signpost-legal_test-hash-graph \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_graph_es_sync.json \
  --stdout-log outputs/legal_test/logs/F10_graph_es_sync.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_graph_es_sync.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test-hash \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider hash \
    --hash-dimensions 128 \
    --recreate
```

正式带日志命令：ECNU embedding。用于正式实验和论文图对象检索结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_es_sync_ecnu \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path signpost-legal_test-ecnu-graph \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_graph_es_sync_ecnu.json \
  --stdout-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --batch-size 4 \
    --recreate
```

如果 ECNU key 达到每日限制，可以临时覆盖 key 重跑。不要把真实 key 写进文档或提交到仓库：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_es_sync_ecnu \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path signpost-legal_test-ecnu-graph \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_graph_es_sync_ecnu.json \
  --stdout-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --batch-size 4 \
    --recreate
```

结构父节点写回说明：

`--update-chunk-parents` 会根据统一图里的 `structure` 边，给已经存在的 chunk ES 文档写入：

```text
parent_summary_id
parent_summary_ids
```

这些字段表示“这个 chunk 属于哪些 summary/章节节点”。它不会改 `chunks.jsonl`，只更新 ES 中的 chunk index。作用是让后续 chunk 检索结果可以直接带出父章节 summary，方便 F13/F15 做结构上下文、章节定位和路标展示。它不是必须步骤；如果只想先跑通图对象检索，可以先不写回。正式 Signpost 实验建议在 F5 chunk index 和 F10 graph index 都完成后写回。

正式带日志命令：把结构父节点写回 ECNU chunk index：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_update_chunk_parents_ecnu \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path signpost-legal_test-ecnu-chunks \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_update_chunk_parents_ecnu.json \
  --stdout-log outputs/legal_test/logs/F10_update_chunk_parents_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_update_chunk_parents_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --batch-size 4 \
    --update-chunk-parents \
    --chunk-index signpost-legal_test-ecnu-chunks \
    --recreate
```

注意：当前 `graph_es_sync` 的写回命令会同时重建 graph index，并在结束时更新 chunk parent 字段。因此它依然需要 embedding provider。若只想本地验证写回流程，可把 namespace/chunk index 换成 hash 版本。

正式带日志命令：图对象检索。`graph_search` 没有单独的 `--output` 参数，JSON 结果会写入 stdout log：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_search_ecnu \
  --method-scope online_query_component \
  --method signpost \
  --input-path signpost-legal_test-ecnu-graph \
  --output-path outputs/legal_test/logs/F10_graph_search_ecnu.stdout.log \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_graph_search_ecnu.json \
  --stdout-log outputs/legal_test/logs/F10_graph_search_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_graph_search_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.graph_search \
    --namespace legal_test-ecnu \
    --query "termination clause" \
    --mode hybrid \
    --object-type entity \
    --object-type summary \
    --top-k 5
```

功能输出和原始测量输出：

| 输出              | 路径                                                                         | 用途                                   |
| --------------- | -------------------------------------------------------------------------- | ------------------------------------ |
| ES graph index  | `signpost-legal_test-hash-graph` 或 `signpost-legal_test-ecnu-graph`        | F13/F15 图对象检索。                       |
| chunk parent 字段 | `signpost-legal_test-ecnu-chunks` 中的 `parent_summary_id(s)`                | 让 chunk 检索结果携带父章节 summary。           |
| hash 阶段自定义指标    | `outputs/legal_test/logs/stage_metrics/F10_graph_es_sync.json`             | 从输入统一图统计待同步节点/边数量。                   |
| ECNU 阶段自定义指标    | `outputs/legal_test/logs/stage_metrics/F10_graph_es_sync_ecnu.json`        | ECNU graph index 同步统计。               |
| 父节点写回指标         | `outputs/legal_test/logs/stage_metrics/F10_update_chunk_parents_ecnu.json` | 写回流程统计。                              |
| 图检索 stdout      | `outputs/legal_test/logs/F10_graph_search_ecnu.stdout.log`                 | graph\_search 返回 JSON。               |
| 阶段总账            | `outputs/legal_test/logs/stage_timing.jsonl`                               | F10 wall time、命令、状态、`extra_metrics`。 |
| stdout/stderr   | `outputs/legal_test/logs/F10_*.log`                                        | ES 写入/embedding/检索失败排查。              |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace legal_test-ecnu \
  --graph datasets/processed/legal_test/graph.unified.json \
  --embedding-provider ecnu \
  --batch-size 4 \
  --recreate
```

如果希望把结构父节点写回 chunk index：

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace legal_test-ecnu \
  --graph datasets/processed/legal_test/graph.unified.json \
  --embedding-provider ecnu \
  --batch-size 4 \
  --update-chunk-parents \
  --chunk-index signpost-legal_test-ecnu-chunks \
  --recreate
```

图对象检索：

```bash
conda run -n signpost-re python -m signpost.retrieval.graph_search \
  --namespace legal_test-ecnu \
  --query "termination clause" \
  --mode hybrid \
  --object-type entity \
  --object-type summary \
  --top-k 5
```

## 16. F11 离线路标

### 目标

对检索结果或指定节点补充预先存在的图邻居路标。离线路标不运行 PPR，只根据统一图中的邻接关系返回可导航节点。

### 对应文件

| 文件                                       | 作用          |
| ---------------------------------------- | ----------- |
| `signpost/retrieval/offline_signpost.py` | 离线路标生成 CLI。 |

### 输入

| 输入                        | 含义                    |
| ------------------------- | --------------------- |
| `--graph`                 | `graph.unified.json`。 |
| `--node-id`               | 直接指定图节点，可重复。          |
| `--chunk-id`              | 指定 chunk id，可重复。      |
| `--result-json`           | 检索结果 JSON，给结果补路标。     |
| `--query` + `--namespace` | 可先查 ES，再补路标。          |

### 输出字段

| 字段             | 含义                                            |
| -------------- | --------------------------------------------- |
| `results`      | 原始或查询得到的结果。                                   |
| `signposts`    | 每个 seed 对应的离线路标。                              |
| `seed_node_id` | 起点节点。                                         |
| `neighbors`    | 邻居节点和边。                                       |
| `edge_type`    | `semantic`、`source`、`structure`、`sequence` 等。 |
| `score`        | 路标分数。                                         |

### 命令

正式带日志命令。F11 通常是 F13/F15 内部组件；单独运行用于检查组件输出和组件耗时：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F11_offline_signpost \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/offline_signpost.json \
  --disk-path outputs/legal_test/retrieval/offline_signpost.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F11_offline_signpost.json \
  --stdout-log outputs/legal_test/logs/F11_offline_signpost.stdout.log \
  --stderr-log outputs/legal_test/logs/F11_offline_signpost.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.offline_signpost \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --chunk-id legal_doc_b9eb62b7885ca06a_c00000 \
    --top-k 5
```

功能输出和原始测量输出：

| 输出      | 路径                                                                | 用途                                                  |
| ------- | ----------------------------------------------------------------- | --------------------------------------------------- |
| 离线路标结果  | `outputs/legal_test/retrieval/offline_signpost.json`              | 组件检查。                                               |
| 阶段自定义指标 | `outputs/legal_test/logs/stage_metrics/F11_offline_signpost.json` | results/signposts 数量等。                              |
| 阶段总账    | `outputs/legal_test/logs/stage_timing.jsonl`                      | 单独组件 wall time。正式 online 指标主要看 F13/F15 query trace。 |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.retrieval.offline_signpost \
  --namespace legal_test-ecnu \
  --graph datasets/processed/legal_test/graph.unified.json \
  --chunk-id legal_doc_b9eb62b7885ca06a_c00000 \
  --top-k 5
```

## 17. F12 在线 PPR 路标

### 目标

以检索结果或指定节点为种子，在统一图上运行 Personalized PageRank，生成在线路标。它体现论文中“根据当前查询动态推荐路标”的部分。

### 对应文件

| 文件                                      | 作用          |
| --------------------------------------- | ----------- |
| `signpost/retrieval/online_signpost.py` | PPR 路标 CLI。 |

### 输入

| 输入              | 含义                    |
| --------------- | --------------------- |
| `--graph`       | `graph.unified.json`。 |
| `--seed`        | 种子 node id，可重复。       |
| `--result-json` | 从检索结果中取种子。            |
| `--scene`       | `auto`、文本场景或图场景。      |
| `--top-k`       | 返回路标数。                |
| `--damping`     | PPR damping。          |
| `--max-iter`    | PPR 最大迭代次数。           |

### 输出字段

| 字段          | 含义            |
| ----------- | ------------- |
| `scene`     | 使用的场景。        |
| `seeds`     | PPR 种子节点。     |
| `signposts` | PPR 排名后的路标节点。 |
| `score`     | PPR 分数。       |
| `node`      | 路标节点属性。       |

### 命令

正式带日志命令。F12 通常在 F13/F15 内部调用；单独运行用于检查 PPR 输出和组件耗时：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F12_online_ppr \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/online_ppr.json \
  --disk-path outputs/legal_test/retrieval/online_ppr.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F12_online_ppr.json \
  --stdout-log outputs/legal_test/logs/F12_online_ppr.stdout.log \
  --stderr-log outputs/legal_test/logs/F12_online_ppr.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.online_signpost \
    --graph datasets/processed/legal_test/graph.unified.json \
    --seed chunk:legal_doc_b9eb62b7885ca06a_c00000 \
    --scene auto \
    --top-k 5 \
    --damping 0.85 \
    --max-iter 100
```

功能输出和原始测量输出：

| 输出       | 路径                                                          | 用途                                                  |
| -------- | ----------------------------------------------------------- | --------------------------------------------------- |
| PPR 路标结果 | `outputs/legal_test/retrieval/online_ppr.json`              | 组件检查。                                               |
| 阶段自定义指标  | `outputs/legal_test/logs/stage_metrics/F12_online_ppr.json` | signposts/seeds 数量等。                                |
| 阶段总账     | `outputs/legal_test/logs/stage_timing.jsonl`                | 单独组件 wall time。正式 online 指标主要看 F13/F15 query trace。 |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.retrieval.online_signpost \
  --graph datasets/processed/legal_test/graph.unified.json \
  --seed chunk:legal_doc_b9eb62b7885ca06a_c00000 \
  --scene auto \
  --top-k 5 \
  --damping 0.85 \
  --max-iter 100
```

## 18. F13 图检索引擎

### 目标

整合 chunk 检索、图对象检索、在线 PPR、顺序上下文，输出可给 Agent 或评估使用的检索结果。

### 对应文件

| 文件                                       | 作用            |
| ---------------------------------------- | ------------- |
| `signpost/retrieval/run.py`              | F13 检索引擎 CLI。 |
| `signpost/retrieval/chunk_search.py`     | chunk 检索。     |
| `signpost/retrieval/graph_search.py`     | 图对象检索。        |
| `signpost/retrieval/online_signpost.py`  | PPR 路标。       |
| `signpost/retrieval/sequence_context.py` | 顺序上下文。        |

### 输入

| 输入                | 含义                       |
| ----------------- | ------------------------ |
| `--namespace`     | ES namespace。            |
| `--query`         | 查询文本。                    |
| `--graph`         | 可选 unified graph。        |
| `--mode`          | `bm25`、`dense`、`hybrid`。 |
| `--chunk-top-k`   | chunk 返回数。               |
| `--summary-top-k` | summary 返回数。             |
| `--graph-top-k`   | graph object 返回数。        |
| `--ppr-top-k`     | 在线路标返回数。                 |

### 输出字段

| 字段                 | 含义                            |
| ------------------ | ----------------------------- |
| `query`            | 原查询。                          |
| `namespace`        | 命名空间。                         |
| `mode`             | 检索模式。                         |
| `text_group`       | chunk 检索结果。                   |
| `graph_group`      | entity/relation/summary 检索结果。 |
| `online_signposts` | PPR 路标。                       |
| `sequence_context` | 相邻 chunk 上下文。                 |
| `metadata`         | top-k、索引名、耗时等信息。              |

### 命令

F13 的 `--namespace`、`--embedding-provider` 必须和 F5/F10 建索引时一致：

| 检索版本       | 依赖的 chunk index                   | 依赖的 graph index                  | 命令参数                                                    |
| ---------- | --------------------------------- | -------------------------------- | ------------------------------------------------------- |
| hash smoke | `signpost-legal_test-hash-chunks` | `signpost-legal_test-hash-graph` | `--namespace legal_test-hash --embedding-provider hash` |
| ECNU 正式实验  | `signpost-legal_test-ecnu-chunks` | `signpost-legal_test-ecnu-graph` | `--namespace legal_test-ecnu --embedding-provider ecnu` |

如果 namespace 和已建立的 ES index 不一致，会出现 `index_not_found_exception`。

调试带日志命令：hash smoke。单问题检索检查用这条；前提是已经跑过 F5 hash 和 F10 hash，不用于论文正式结果：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F13_retrieval_hash \
  --method-scope online_query_component \
  --method static_unified \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/retrieval_result.hash.json \
  --disk-path outputs/legal_test/retrieval/retrieval_result.hash.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F13_retrieval_hash.json \
  --stdout-log outputs/legal_test/logs/F13_retrieval_hash.stdout.log \
  --stderr-log outputs/legal_test/logs/F13_retrieval_hash.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.run \
    --namespace legal_test-hash \
    --query "What is the purpose of the agreement?" \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider hash \
    --mode hybrid \
    --chunk-top-k 5 \
    --summary-top-k 5 \
    --graph-top-k 5 \
    --ppr-top-k 5 \
    --output outputs/legal_test/retrieval/retrieval_result.hash.json
```

正式带日志命令：ECNU 正式实验。前提是已经跑过 F5 ECNU 和 F10 ECNU：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F13_retrieval_ecnu \
  --method-scope online_query_component \
  --method static_unified \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/retrieval_result.ecnu.json \
  --disk-path outputs/legal_test/retrieval/retrieval_result.ecnu.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F13_retrieval_ecnu.json \
  --stdout-log outputs/legal_test/logs/F13_retrieval_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F13_retrieval_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.run \
    --namespace legal_test-ecnu \
    --query "What is the purpose of the agreement?" \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --mode hybrid \
    --chunk-top-k 5 \
    --summary-top-k 5 \
    --graph-top-k 5 \
    --ppr-top-k 5 \
    --output outputs/legal_test/retrieval/retrieval_result.ecnu.json
```

如果 ECNU key 达到每日限制，可以临时覆盖 key 重跑：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F13_retrieval_ecnu \
  --method-scope online_query_component \
  --method static_unified \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/retrieval_result.ecnu.json \
  --disk-path outputs/legal_test/retrieval/retrieval_result.ecnu.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F13_retrieval_ecnu.json \
  --stdout-log outputs/legal_test/logs/F13_retrieval_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F13_retrieval_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.run \
    --namespace legal_test-ecnu \
    --query "What is the purpose of the agreement?" \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --mode hybrid \
    --chunk-top-k 5 \
    --summary-top-k 5 \
    --graph-top-k 5 \
    --ppr-top-k 5 \
    --output outputs/legal_test/retrieval/retrieval_result.ecnu.json
```

功能输出和原始测量输出：

| 输出           | 路径                                                              | 用途                                                |
| ------------ | --------------------------------------------------------------- | ------------------------------------------------- |
| hash 检索结果    | `outputs/legal_test/retrieval/retrieval_result.hash.json`       | hash smoke 单问题组件检查。                               |
| ECNU 检索结果    | `outputs/legal_test/retrieval/retrieval_result.ecnu.json`       | 正式 embedding 单问题组件检查。                             |
| hash 阶段自定义指标 | `outputs/legal_test/logs/stage_metrics/F13_retrieval_hash.json` | `text_items`、`graph_items`、`online_signposts` 等。  |
| ECNU 阶段自定义指标 | `outputs/legal_test/logs/stage_metrics/F13_retrieval_ecnu.json` | 正式 embedding 检索统计。                                |
| 阶段总账         | `outputs/legal_test/logs/stage_timing.jsonl`                    | 单次 retrieval wall time。批量论文指标看 F15 per-query log。 |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.retrieval.run \
  --namespace legal_test-ecnu \
  --query "What is the purpose of the agreement?" \
  --graph datasets/processed/legal_test/graph.unified.json \
  --embedding-provider ecnu \
  --mode hybrid \
  --chunk-top-k 5 \
  --summary-top-k 5 \
  --graph-top-k 5 \
  --ppr-top-k 5 \
  --output outputs/legal_test/retrieval/retrieval_result.ecnu.json
```

## 19. F14 ReadFile 工具

### 目标

根据来源定位读取原文行号，为 Agent 和人工检查提供证据回读能力。

### 对应文件

| 文件                                | 作用                  |
| --------------------------------- | ------------------- |
| `signpost/retrieval/read_file.py` | ReadFile CLI 和工具逻辑。 |

### 输入

| 输入                            | 含义                                                   |
| ----------------------------- | ---------------------------------------------------- |
| `--dataset`                   | 自动使用 `datasets/processed/<dataset>/documents.jsonl`。 |
| `--documents`                 | 显式 documents 路径。                                     |
| `--locate`                    | `file.txt:L10-L35`。                                  |
| `--file`                      | 文件名或 doc\_id。                                        |
| `--start-line` / `--end-line` | 行范围。                                                 |
| `--before` / `--after`        | 前后扩展行数。                                              |
| `--json`                      | 输出 JSON。                                             |

### 输出字段

| 字段                  | 含义              |
| ------------------- | --------------- |
| `tool`              | `read_file`。    |
| `dataset`           | 数据集。            |
| `documents_path`    | 实际读取路径。         |
| `doc_id`            | 文档 ID。          |
| `file_name`         | 文件名。            |
| `requested`         | 请求范围。           |
| `resolved`          | 实际读取范围。         |
| `lines`             | 行号和文本。          |
| `file_content_view` | 带行号文本，可直接给 LLM。 |

### 命令

正式带日志命令。F14 通常由 Agent 工具调用；单独运行用于检查行号回读能力：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F14_read_file \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path outputs/legal_test/retrieval/read_file.json \
  --disk-path outputs/legal_test/retrieval/read_file.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F14_read_file.json \
  --stdout-log outputs/legal_test/logs/F14_read_file.stdout.log \
  --stderr-log outputs/legal_test/logs/F14_read_file.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.read_file \
    --dataset legal_test \
    --file legal_doc_b9eb62b7885ca06a.txt \
    --start-line 1 \
    --end-line 20 \
    --before 2 \
    --after 2 \
    --json \
    --output outputs/legal_test/retrieval/read_file.json
```

功能输出和原始测量输出：

| 输出      | 路径                                                         | 用途                                                           |
| ------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| 原文回读结果  | `outputs/legal_test/retrieval/read_file.json`              | 检查 locate/行号能否回读证据。                                          |
| 阶段自定义指标 | `outputs/legal_test/logs/stage_metrics/F14_read_file.json` | 返回行数等。                                                       |
| 阶段总账    | `outputs/legal_test/logs/stage_timing.jsonl`               | 单次 read\_file wall time。正式 online 指标主要看 F15 trace/query log。 |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.retrieval.read_file \
  --dataset legal_test \
  --file legal_doc_b9eb62b7885ca06a.txt \
  --start-line 1 \
  --end-line 20 \
  --before 2 \
  --after 2

conda run -n signpost-re python -m signpost.retrieval.read_file \
  --dataset legal_test \
  --file legal_doc_b9eb62b7885ca06a.txt \
  --start-line 1 \
  --end-line 20 \
  --json
```

## 20. F15 Supervisor-Researcher Agent

### 目标

实现论文中的 Agent 检索流程：Supervisor 拆解问题，Researcher 调用检索和 ReadFile，最后合成答案，并保留 trace 用于指标统计。

### 对应文件

| 文件                             | 作用                                |
| ------------------------------ | --------------------------------- |
| `signpost/agent/supervisor.py` | Supervisor、Researcher、trace。      |
| `signpost/agent/tools.py`      | KnowledgeSearchTool、ReadFileTool。 |
| `signpost/agent/run.py`        | 单问题 CLI。                          |
| `signpost/agent/batch.py`      | 批量预测 CLI。                         |

### 单问题输出字段

| 字段             | 含义            |
| -------------- | ------------- |
| `trace_id`     | 一次运行 ID。      |
| `namespace`    | 检索 namespace。 |
| `question`     | 原问题。          |
| `subquestions` | 拆解问题。         |
| `answer`       | 最终答案。         |
| `citations`    | 证据引用。         |
| `research`     | 每个子问题的检索和证据。  |
| `trace`        | 事件轨迹。         |

`trace[]` 字段：

| 字段               | 含义                                                                   |
| ---------------- | -------------------------------------------------------------------- |
| `trace_id`       | trace ID。                                                            |
| `event_type`     | `supervisor_start`、`plan`、`tool_call`、`tool_error`、`final_answer` 等。 |
| `timestamp`      | Unix 时间戳。                                                            |
| `tool`           | 工具名。                                                                 |
| `input`          | 工具输入。                                                                |
| `output_summary` | 工具输出摘要。                                                              |

### 批量 prediction 输出字段

| 字段                 | 含义                              |
| ------------------ | ------------------------------- |
| `question_id`      | 问题 ID。                          |
| `question`         | 问题。                             |
| `answer`           | 标准答案。                           |
| `rationale`        | 标准 rationale。                   |
| `prediction`       | F16 兼容预测文本。                     |
| `citations`        | 引用。                             |
| `trace_id`         | trace ID。                       |
| `trace`            | trace 事件。                       |
| `metadata.method`  | 方法名，默认 `signpost`。              |
| `metadata.dataset` | processed 数据集名，例如 `legal_test`。 |

### 命令

F15 有两组身份需要区分：

| 参数                           | 含义                                                                                      | legal\_test 正式实验示例 |
| ---------------------------- | --------------------------------------------------------------------------------------- | ------------------ |
| `--namespace`                | ES 检索命名空间，必须和 F5/F10 建索引一致。                                                             | `legal_test-ecnu`  |
| `--dataset`                  | 本地 processed 数据集，用于读取 `graph.unified.json`、`chunks.jsonl`、`documents.jsonl` 和 ReadFile。 | `legal_test`       |
| `--embedding-provider`       | 查询时生成 query embedding 的模型，必须和 F5/F10 建索引时一致。当前默认就是 `ecnu`；hash 只用于 smoke/debug。         | `ecnu`             |
| `--use-llm` / `--no-use-llm` | F15 默认使用 LLM 做问题拆解和答案合成；`--no-use-llm` 只用于调试，走规则拆问和模板答案。                                | 默认 LLM             |

正式实验应使用 ES：

```text
--namespace legal_test-ecnu --dataset legal_test --embedding-provider ecnu --use-es
```

这会用 ES 检索：

```text
signpost-legal_test-ecnu-chunks
signpost-legal_test-ecnu-graph
```

同时仍然从本地读取图和原文：

```text
datasets/processed/legal_test/graph.unified.json
datasets/processed/legal_test/chunks.jsonl
datasets/processed/legal_test/documents.jsonl
```

不带 `--use-es` 的 local 模式只用于调试 Agent/ReadFile/trace，不用于论文检索结果。F15 默认已经使用 ECNU LLM 做问题拆解和答案合成；如果为了快速排查检索链路临时不想调用 LLM，可以显式加 `--no-use-llm`。

正式带日志命令：ECNU ES 检索版本。这个命令会同时写 prediction JSONL、per-query log、阶段总账和阶段自定义指标：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F15_agent_batch_ecnu \
  --method-scope online_query \
  --method signpost \
  --input-path datasets/processed/legal_test/questions.jsonl \
  --output-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --disk-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F15_agent_batch_ecnu.json \
  --stdout-log outputs/legal_test/logs/F15_agent_batch_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F15_agent_batch_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.agent.batch \
    --namespace legal_test-ecnu \
    --dataset legal_test \
    --questions datasets/processed/legal_test/questions.jsonl \
    --output outputs/legal_test/predictions/signpost.ecnu.jsonl \
    --query-log outputs/legal_test/logs/signpost.ecnu.query.jsonl \
    --embedding-provider ecnu \
    --use-es 
```

如果 ECNU key 达到每日限制，可以临时覆盖 key 重跑：

```bash
cd /home/ruolinsu/signpost/signpost_re
ECNU_API_KEY='替换成你的新 key' \
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F15_agent_batch_ecnu \
  --method-scope online_query \
  --method signpost \
  --input-path datasets/processed/legal_test/questions.jsonl \
  --output-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --disk-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F15_agent_batch_ecnu.json \
  --stdout-log outputs/legal_test/logs/F15_agent_batch_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F15_agent_batch_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.agent.batch \
    --namespace legal_test-ecnu \
    --dataset legal_test \
    --questions datasets/processed/legal_test/questions.jsonl \
    --output outputs/legal_test/predictions/signpost.ecnu.jsonl \
    --query-log outputs/legal_test/logs/signpost.ecnu.query.jsonl \
    --embedding-provider ecnu \
    --use-es \
    --limit 5
```

常见错误：如果 stderr 中出现 `The query vector has a different number of dimensions [128] than the document vectors [1024]`，说明查询端用了 hash 128 维 embedding 去查 ECNU 1024 维索引。解决方式是使用当前文档里的 ECNU 命令，确保命令中有 `--embedding-provider ecnu`；hash 只用于独立 smoke/debug，不用于正式论文结果。

功能输出和原始测量输出：

| 输出               | 路径                                                                | 用途                                                                                                                               |
| ---------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| prediction JSONL | `outputs/legal_test/predictions/signpost.ecnu.jsonl`              | F16 评估输入；每题包含 prediction、citations、trace、latency/tokens/tool calls。                                                              |
| per-query 原始日志   | `outputs/legal_test/logs/signpost.ecnu.query.jsonl`               | 每题 `latency_seconds`、`retrieval_latency_seconds`、`read_file_latency_seconds`、`tool_calls`、`llm_calls`、token 估计、retrieved chunks。 |
| 阶段自定义指标          | `outputs/legal_test/logs/stage_metrics/F15_agent_batch_ecnu.json` | queries、tool calls、tokens、latency 总量。                                                                                            |
| 阶段总账             | `outputs/legal_test/logs/stage_timing.jsonl`                      | F15 batch 总 wall time、命令、状态、prediction 文件大小、`extra_metrics`。                                                                     |
| stdout/stderr    | `outputs/legal_test/logs/F15_agent_batch_ecnu.*.log`              | 长任务失败排查。                                                                                                                         |

内部裸命令：

单问题：

```bash
conda run -n signpost-re python -m signpost.agent.run \
  --namespace legal_test-ecnu \
  --dataset legal_test \
  --question "What is the purpose of the agreement?" \
  --output outputs/legal_test/agent_single.ecnu.json \
  --embedding-provider ecnu \
  --use-es
```

批量 ES 检索：

```bash
conda run -n signpost-re python -m signpost.agent.batch \
  --namespace legal_test-ecnu \
  --dataset legal_test \
  --questions datasets/processed/legal_test/questions.jsonl \
  --output outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --query-log outputs/legal_test/logs/signpost.ecnu.query.jsonl \
  --embedding-provider ecnu \
  --use-es \
  --limit 5
```

## 21. F16 预测输出与评估

### 目标

统一 prediction JSONL schema，计算 EM、Precision、Recall、F1，并支持 LLM-as-Judge。

### 对应文件

| 文件                                            | 作用                      |
| --------------------------------------------- | ----------------------- |
| `signpost/evaluation/schema.py`               | prediction 规范化和校验。      |
| `signpost/evaluation/metrics.py`              | EM、Precision、Recall、F1。 |
| `signpost/evaluation/validate_predictions.py` | 校验 prediction。          |
| `signpost/evaluation/evaluate_basic.py`       | 基础指标。                   |
| `signpost/evaluation/convert_predictions.py`  | 旧格式转换。                  |
| `signpost/evaluation/llm_judge.py`            | LLM-as-Judge。           |

### 输入：prediction JSONL

| 字段            | 类型            | 含义                               |
| ------------- | ------------- | -------------------------------- |
| `question_id` | string        | 问题 ID。                           |
| `question`    | string        | 问题。                              |
| `answer`      | string/list   | 标准答案。                            |
| `rationale`   | string        | 标准推理或证据说明。                       |
| `prediction`  | string        | 预测文本，推荐含 `<answer>...</answer>`。 |
| `metadata`    | object        | 至少包含 `method` 和 `dataset`。       |
| `citations`   | list\[object] | 可选证据引用。                          |
| `trace`       | list\[object] | 可选 agent trace。                  |

### 输出：basic eval JSON

| 字段                    | 含义               |
| --------------------- | ---------------- |
| `num_samples`         | 输入样本数。           |
| `num_scored`          | 评分样本数。           |
| `num_skipped`         | 跳过样本数。           |
| `metrics.exact_match` | 完全匹配。            |
| `metrics.precision`   | token precision。 |
| `metrics.recall`      | token recall。    |
| `metrics.f1`          | token F1。        |
| `per_example`         | 每题分数。            |

### 命令

正式带日志命令。F16 是评估阶段，不算入方法 online latency，但需要记录评估耗时和评估输出：

```bash
cd /home/ruolinsu/signpost/signpost_re
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F16_evaluation \
  --method-scope evaluation \
  --method signpost \
  --input-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output-path outputs/legal_test/metrics/basic_eval.json \
  --disk-path outputs/legal_test/metrics/basic_eval.json \
  --auto-metrics \
  --metrics-json outputs/legal_test/logs/stage_metrics/F16_evaluation.json \
  --stdout-log outputs/legal_test/logs/F16_evaluation.stdout.log \
  --stderr-log outputs/legal_test/logs/F16_evaluation.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
    --output outputs/legal_test/metrics/basic_eval.json
```

功能输出和原始测量输出：

| 输出         | 路径                                                          | 用途                                      |
| ---------- | ----------------------------------------------------------- | --------------------------------------- |
| basic eval | `outputs/legal_test/metrics/basic_eval.json`                | EM、Precision、Recall、F1。                 |
| 阶段自定义指标    | `outputs/legal_test/logs/stage_metrics/F16_evaluation.json` | `num_samples`、`num_scored`、`eval_f1` 等。 |
| 阶段总账       | `outputs/legal_test/logs/stage_timing.jsonl`                | F16 wall time、命令、状态、评估文件大小。             |

内部裸命令：

```bash
conda run -n signpost-re python -m signpost.evaluation.validate_predictions \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl

conda run -n signpost-re python -m signpost.evaluation.evaluate_basic \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/basic_eval.json

conda run --no-capture-output -n signpost-re python -m signpost.evaluation.llm_judge \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/llm_judge.jsonl \
  --dimension answer_correctness
```

## 22. Benchmark 指标与实验日志

### 目标

补充 ICDE 实验需要的指标代码：离线成本、在线成本、图结构、证据命中、摊销成本、break-even、Pareto frontier。当前只统计已有工件，不实现检索剪枝，不实现 baseline。

### 对应文件

| 文件                                     | 作用                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------- |
| `signpost/benchmark/time_stage.py`     | 包装阶段命令，写 `stage_timing.jsonl`。                                          |
| `signpost/benchmark/query_metrics.py`  | 统计 prediction/query log。                                                |
| `signpost/benchmark/index_metrics.py`  | 统计 stage log、F6 cache、graph JSON。                                       |
| `signpost/benchmark/method_summary.py` | 把单个方法的 query metrics 和 offline stage log 汇总成 `method_summaries.json` 行。 |
| `signpost/benchmark/cost_quality.py`   | 摊销、break-even、Pareto。                                                   |
| `signpost/benchmark/stats.py`          | 通用统计。                                                                   |

### 推荐输出目录

```text
outputs/<dataset>/
  logs/
    stage_timing.jsonl
    <method>.query.jsonl
  predictions/
    <method>.jsonl
  metrics/
    <method>.query_metrics.json
    index_metrics.json
    method_summaries.json
    cost_quality.json
```

### Stage timing 输出字段

| 字段                               | 含义                                                    |
| -------------------------------- | ----------------------------------------------------- |
| `dataset`                        | 数据集。                                                  |
| `method`                         | 方法名，可为空。                                              |
| `stage`                          | 阶段名。                                                  |
| `method_scope`                   | 成本归属，例如 `shared_preprocess`、`signpost_offline_index`。 |
| `input_path` / `output_path`     | 输入输出路径。                                               |
| `command`                        | 实际命令。                                                 |
| `started_at` / `finished_at`     | 时间戳。                                                  |
| `wall_time_seconds`              | 墙钟耗时。                                                 |
| `llm_calls`                      | 该阶段 LLM 调用次数；不能直接拿到时先为 0，后续由 cache/query log 计算。      |
| `input_tokens` / `output_tokens` | 该阶段 token；不能直接拿到时先为 0。                                |
| `disk_bytes`                     | `--disk-path` 或 `--output-path` 对应文件/目录大小。            |
| `status`                         | `ok` 或 `failed`。                                      |
| `return_code`                    | 退出码。                                                  |

命令示例：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph \
  --method-scope signpost_offline_index \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.llm.json \
  --disk-path datasets/processed/legal_test/graph.semantic.llm.json \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.llm.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl
```

### Query metrics 输出字段

| 字段                      | 含义                            |
| ----------------------- | ----------------------------- |
| `quality`               | EM、precision、recall、F1。       |
| `quality_counts`        | 样本数、评分数、跳过数。                  |
| `cost.totals`           | 总 token、总 calls、总耗时等。         |
| `cost.means`            | 单 query 平均成本。                 |
| `cost.p95`              | p95 尾部成本。                     |
| `retrieval.recall_at_k` | 有 gold evidence 时的 Recall\@k。 |
| `retrieval.mrr`         | 有 gold evidence 时的 MRR。       |
| `per_query`             | 每题质量和成本。                      |

核心 cost 字段：

| 字段                                                | 含义             |
| ------------------------------------------------- | -------------- |
| `online_llm_calls` / `llm_calls`                  | LLM 调用次数。      |
| `tool_calls`                                      | Agent 工具调用次数。  |
| `input_tokens` / `output_tokens` / `total_tokens` | token 成本。      |
| `latency_seconds`                                 | 单题总耗时。         |
| `retrieval_latency_seconds`                       | 检索耗时。          |
| `ppr_latency_seconds`                             | PPR 耗时。        |
| `read_file_latency_seconds`                       | ReadFile 耗时。   |
| `retrieved_chunks`                                | 返回 chunk 数。    |
| `read_file_calls`                                 | ReadFile 调用次数。 |
| `graph_ppr_calls`                                 | PPR 调用次数。      |
| `max_context_tokens`                              | 最大上下文 token。   |

命令：

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/signpost.query_metrics.json
```

### Index metrics 输出字段

| 字段                     | 含义                                |
| ---------------------- | --------------------------------- |
| `stage_logs`           | 每阶段耗时、状态、LLM calls、tokens。        |
| `semantic_extractions` | F6 cache 的 chunks、实体、关系、估算 calls。 |
| `graphs`               | 图的节点、边、边类型比例、度、连通分量。              |

图结构字段：

| 字段                     | 含义       |
| ---------------------- | -------- |
| `nodes` / `edges`      | 图规模。     |
| `node_counts`          | 各节点类型数量。 |
| `edge_counts`          | 各边类型数量。  |
| `edge_type_ratio`      | 各边类型占比。  |
| `degree`               | 节点度统计。   |
| `connected_components` | 连通分量统计。  |

命令：

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/legal_test/graph.unified.json \
  --output outputs/legal_test/metrics/index_metrics.json
```

### Cost-quality 输出字段

输入文件：

```text
outputs/<dataset>/metrics/method_summaries.json
```

该文件由 `signpost.benchmark.method_summary` 生成或追加，每行对应一个方法：

| 字段            | 含义                                                                                  |
| ------------- | ----------------------------------------------------------------------------------- |
| `method`      | 方法名，例如 `hybrid_rag`、`graphsearch_unified`、`signpost_full`。                          |
| `dataset`     | 数据集。                                                                                |
| `num_queries` | query 数。                                                                            |
| `quality`     | 来自 query metrics 的 EM/F1 等。                                                         |
| `cost`        | 来自 query metrics 的在线 token/calls/latency。                                           |
| `retrieval`   | Recall\@k、MRR 等检索指标。                                                                |
| `offline`     | 从 `stage_timing.jsonl` 选定 offline stages 汇总的 wall time、tokens、LLM calls、disk bytes。 |

输出字段：

| 字段          | 含义                                          |
| ----------- | ------------------------------------------- |
| `methods`   | 归一化后的方法摘要。                                  |
| `amortized` | 不同 query 数下的摊销成本。                           |
| `pairwise`  | 方法两两比较，含 break-even 和每多答对一个问题成本。            |
| `pareto`    | online tokens vs quality 的 Pareto frontier。 |

命令：

先为每个方法生成或更新 method summary：

```bash
conda run -n signpost-re python -m signpost.benchmark.method_summary \
  --method signpost \
  --dataset legal_test \
  --query-metrics outputs/legal_test/metrics/signpost.query_metrics.json \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --offline-stage F5_chunk_index \
  --offline-stage F6_semantic_graph \
  --offline-stage F7_structure_graph \
  --offline-stage F8_sequence_graph \
  --offline-stage F9_unified_graph \
  --offline-stage F10_graph_es_sync \
  --output outputs/legal_test/metrics/method_summaries.json
```

再计算成本质量分析：

```bash
conda run -n signpost-re python -m signpost.benchmark.cost_quality \
  --methods outputs/legal_test/metrics/method_summaries.json \
  --workload-sizes 10 50 100 500 1000 5000 10000 \
  --output outputs/legal_test/metrics/cost_quality.json
```

更详细的人话解释见：

```text
docs/experiment_metrics_plain_guide.zh.md
docs/experiment_metrics_guide.zh.md
```

### legal\_test 完整计时命令示例

下面命令展示如何从功能命令同时得到原始测量数据。正式跑其他数据集时，把 `legal_test` 替换成目标数据集即可。

说明：下面示例都保留 `--stdout-log` 和 `--stderr-log`，方便失败后追溯。`--metrics-json` 是阶段自定义指标入口；如果该 JSON 文件还没有由功能点生成，`time_stage.py` 会把 `extra_metrics` 记为空对象，不会阻止功能命令运行。对于 F6/F15，真正关键的明细仍然是 `semantic_llm.*.jsonl` 和 prediction trace/query log。

F3 shared preprocessing：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_data_prepare \
  --method-scope shared_preprocess \
  --output-path datasets/processed/legal_test/raw_corpus.jsonl \
  --disk-path datasets/processed/legal_test \
  --stdout-log outputs/legal_test/logs/F3_data_prepare.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_data_prepare.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  conda run -n signpost-re python -m signpost.data.prepare --datasets legal_test
```

F3.5 shared preprocessing：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F3_5_parse_normalize \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/raw_corpus.jsonl \
  --output-path datasets/processed/legal_test/documents.jsonl \
  --disk-path datasets/processed/legal_test/documents.jsonl \
  --stdout-log outputs/legal_test/logs/F3_5_parse_normalize.stdout.log \
  --stderr-log outputs/legal_test/logs/F3_5_parse_normalize.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  conda run -n signpost-re python -m signpost.parsing.parse_documents \
    --input datasets/processed/legal_test/raw_corpus.jsonl \
    --output datasets/processed/legal_test/documents.jsonl
```

F4 shared preprocessing：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F4_chunk_tree \
  --method-scope shared_preprocess \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path datasets/processed/legal_test/chunks.jsonl \
  --disk-path datasets/processed/legal_test \
  --stdout-log outputs/legal_test/logs/F4_chunk_tree.stdout.log \
  --stderr-log outputs/legal_test/logs/F4_chunk_tree.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  conda run -n signpost-re python -m signpost.chunking.run \
    --input datasets/processed/legal_test/documents.jsonl \
    --output datasets/processed/legal_test/chunks.jsonl \
    --tree-output datasets/processed/legal_test/document_trees.jsonl \
    --max-tokens 1200 \
    --overlap-tokens 100
```

F5 method-specific offline index：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F5_chunk_index_ecnu \
  --method-scope method_offline_index \
  --method hybrid_rag \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path signpost-legal_test-ecnu-chunks \
  --metrics-json outputs/legal_test/logs/stage_metrics/F5_chunk_index_ecnu.json \
  --stdout-log outputs/legal_test/logs/F5_chunk_index_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F5_chunk_index_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace legal_test-ecnu \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --embedding-provider ecnu \
    --batch-size 4 \
    --progress-every 10 \
    --embedding-retries 5 \
    --retry-sleep 3 \
    --recreate
```

F6 method-specific offline index：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F6_semantic_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.semantic.json \
  --disk-path datasets/processed/legal_test/graph.semantic.json \
  --metrics-json outputs/legal_test/logs/stage_metrics/F6_semantic_graph.json \
  --stdout-log outputs/legal_test/logs/F6_semantic_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F6_semantic_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.semantic_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.semantic.json \
    --extractor llm \
    --gleaning-rounds 1 \
    --progress-every 10 \
    --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
    --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
    --llm-retries 5 \
    --retry-sleep 3 \
    --llm-timeout 180
```

F7/F8/F9 method-specific offline index：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F7_structure_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.structure.json \
  --disk-path datasets/processed/legal_test/graph.structure.json \
  --metrics-json outputs/legal_test/logs/stage_metrics/F7_structure_graph.json \
  --stdout-log outputs/legal_test/logs/F7_structure_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F7_structure_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace legal_test-llm \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --document-trees datasets/processed/legal_test/document_trees.jsonl \
    --output datasets/processed/legal_test/graph.structure.json \
    --summarizer llm \
    --max-summary-tokens 512 \
    --cluster-token-budget 4096

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F8_sequence_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/chunks.jsonl \
  --output-path datasets/processed/legal_test/graph.sequence.json \
  --disk-path datasets/processed/legal_test/graph.sequence.json \
  --metrics-json outputs/legal_test/logs/stage_metrics/F8_sequence_graph.json \
  --stdout-log outputs/legal_test/logs/F8_sequence_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F8_sequence_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.sequence_graph \
    --namespace legal_test \
    --chunks datasets/processed/legal_test/chunks.jsonl \
    --output datasets/processed/legal_test/graph.sequence.json

conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F9_unified_graph \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.semantic.json \
  --output-path datasets/processed/legal_test/graph.unified.json \
  --disk-path datasets/processed/legal_test/graph.unified.json \
  --metrics-json outputs/legal_test/logs/stage_metrics/F9_unified_graph.json \
  --stdout-log outputs/legal_test/logs/F9_unified_graph.stdout.log \
  --stderr-log outputs/legal_test/logs/F9_unified_graph.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.graph.merge \
    --namespace legal_test \
    --semantic datasets/processed/legal_test/graph.semantic.json \
    --structure datasets/processed/legal_test/graph.structure.json \
    --sequence datasets/processed/legal_test/graph.sequence.json \
    --output datasets/processed/legal_test/graph.unified.json
```

F10 method-specific offline index：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F10_graph_es_sync_ecnu \
  --method-scope method_offline_index \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path signpost-legal_test-ecnu-graph \
  --metrics-json outputs/legal_test/logs/stage_metrics/F10_graph_es_sync_ecnu.json \
  --stdout-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F10_graph_es_sync_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --batch-size 4 \
    --recreate
```

F11 offline signpost 单独检查：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F11_offline_signpost \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/offline_signpost.json \
  --disk-path outputs/legal_test/retrieval/offline_signpost.json \
  --stdout-log outputs/legal_test/logs/F11_offline_signpost.stdout.log \
  --stderr-log outputs/legal_test/logs/F11_offline_signpost.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.offline_signpost \
    --namespace legal_test-ecnu \
    --graph datasets/processed/legal_test/graph.unified.json \
    --chunk-id legal_doc_b9eb62b7885ca06a_c00000 \
    --top-k 5
```

说明：F11 在正式 F13/F15 中通常作为检索内部步骤，不一定单独运行；这个命令用于单独检查离线路标功能和记录组件耗时。

F12 online PPR 单独检查：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F12_online_ppr \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/online_ppr.json \
  --disk-path outputs/legal_test/retrieval/online_ppr.json \
  --stdout-log outputs/legal_test/logs/F12_online_ppr.stdout.log \
  --stderr-log outputs/legal_test/logs/F12_online_ppr.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.online_signpost \
    --graph datasets/processed/legal_test/graph.unified.json \
    --seed chunk:legal_doc_b9eb62b7885ca06a_c00000 \
    --scene auto \
    --top-k 5 \
    --damping 0.85 \
    --max-iter 100
```

说明：F12 在正式 F13/F15 中通常记录为每条 query 的 `ppr_latency_seconds` 和 `graph_ppr_calls`；单独运行只用于组件检查。

F13 retrieval engine：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F13_retrieval \
  --method-scope online_query_component \
  --method static_unified \
  --input-path datasets/processed/legal_test/graph.unified.json \
  --output-path outputs/legal_test/retrieval/retrieval_result.json \
  --disk-path outputs/legal_test/retrieval/retrieval_result.json \
  --stdout-log outputs/legal_test/logs/F13_retrieval.stdout.log \
  --stderr-log outputs/legal_test/logs/F13_retrieval.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.run \
    --namespace legal_test-ecnu \
    --query "What is the purpose of the agreement?" \
    --graph datasets/processed/legal_test/graph.unified.json \
    --embedding-provider ecnu \
    --mode hybrid \
    --chunk-top-k 5 \
    --summary-top-k 5 \
    --graph-top-k 5 \
    --ppr-top-k 5 \
    --output outputs/legal_test/retrieval/retrieval_result.json
```

F14 ReadFile：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F14_read_file \
  --method-scope online_query_component \
  --method signpost \
  --input-path datasets/processed/legal_test/documents.jsonl \
  --output-path outputs/legal_test/retrieval/read_file.json \
  --disk-path outputs/legal_test/retrieval/read_file.json \
  --stdout-log outputs/legal_test/logs/F14_read_file.stdout.log \
  --stderr-log outputs/legal_test/logs/F14_read_file.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.retrieval.read_file \
    --dataset legal_test \
    --file legal_doc_b9eb62b7885ca06a.txt \
    --start-line 1 \
    --end-line 20 \
    --before 2 \
    --after 2 \
    --json
```

F15 online agent batch（正式实验使用 ECNU ES 索引）：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F15_agent_batch_ecnu \
  --method-scope online_query \
  --method signpost \
  --input-path datasets/processed/legal_test/questions.jsonl \
  --output-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --disk-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --metrics-json outputs/legal_test/logs/stage_metrics/F15_agent_batch_ecnu.json \
  --stdout-log outputs/legal_test/logs/F15_agent_batch_ecnu.stdout.log \
  --stderr-log outputs/legal_test/logs/F15_agent_batch_ecnu.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.agent.batch \
    --namespace legal_test-ecnu \
    --dataset legal_test \
    --questions datasets/processed/legal_test/questions.jsonl \
    --output outputs/legal_test/predictions/signpost.ecnu.jsonl \
    --query-log outputs/legal_test/logs/signpost.ecnu.query.jsonl \
    --embedding-provider ecnu \
    --use-es \
    --limit 5
```

F16 evaluation：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.benchmark.time_stage \
  --dataset legal_test \
  --stage F16_evaluation \
  --method-scope evaluation \
  --method signpost \
  --input-path outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output-path outputs/legal_test/metrics/basic_eval.json \
  --disk-path outputs/legal_test/metrics/basic_eval.json \
  --metrics-json outputs/legal_test/logs/stage_metrics/F16_evaluation.json \
  --stdout-log outputs/legal_test/logs/F16_evaluation.stdout.log \
  --stderr-log outputs/legal_test/logs/F16_evaluation.stderr.log \
  --log outputs/legal_test/logs/stage_timing.jsonl \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
    --output outputs/legal_test/metrics/basic_eval.json
```

汇总离线指标和在线指标：

```bash
conda run -n signpost-re python -m signpost.benchmark.index_metrics \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --semantic-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --gleaning-rounds 1 \
  --graph datasets/processed/legal_test/graph.unified.json \
  --output outputs/legal_test/metrics/index_metrics.json

conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/signpost.query_metrics.json

conda run -n signpost-re python -m signpost.benchmark.method_summary \
  --method signpost \
  --dataset legal_test \
  --query-metrics outputs/legal_test/metrics/signpost.query_metrics.json \
  --stage-log outputs/legal_test/logs/stage_timing.jsonl \
  --offline-stage F5_chunk_index \
  --offline-stage F6_semantic_graph \
  --offline-stage F7_structure_graph \
  --offline-stage F8_sequence_graph \
  --offline-stage F9_unified_graph \
  --offline-stage F10_graph_es_sync \
  --output outputs/legal_test/metrics/method_summaries.json
```

## 23. legal\_test 内部功能命令参考

下面是小数据集内部功能命令参考。所有命令都从 `/home/ruolinsu/signpost/signpost_re` 运行。

注意：本节命令主要用于说明每个模块原本怎么调用。正式实验不要直接运行这些裸命令；正式实验应使用上一节 `legal_test 完整计时命令示例` 中的 `signpost.benchmark.time_stage ... -- <内部功能命令>` 形式，这样才能同时得到功能输出和原始测量日志。

### F3

```bash
conda run -n signpost-re python -m signpost.data.prepare --datasets legal_test
conda run -n signpost-re python -m signpost.data.validate --dataset legal_test
```

### F3.5

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal_test/raw_corpus.jsonl \
  --output datasets/processed/legal_test/documents.jsonl

conda run -n signpost-re python -m signpost.parsing.validate_documents \
  --input datasets/processed/legal_test/documents.jsonl
```

### F4

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_test/documents.jsonl \
  --output datasets/processed/legal_test/chunks.jsonl \
  --tree-output datasets/processed/legal_test/document_trees.jsonl \
  --max-tokens 1200 \
  --overlap-tokens 100

conda run -n signpost-re python -m signpost.chunking.validate \
  --chunks datasets/processed/legal_test/chunks.jsonl
```

### F5

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.chunk_index \
  --namespace legal_test-ecnu \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --embedding-provider ecnu \
  --batch-size 4 \
  --progress-every 10 \
  --embedding-retries 5 \
  --retry-sleep 3 \
  --recreate
```

### F6-F9 LLM 正式流程裸命令

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace legal_test-llm \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --output datasets/processed/legal_test/graph.semantic.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --progress-every 10 \
  --progress-file datasets/processed/legal_test/semantic_llm.progress.jsonl \
  --extractions-cache datasets/processed/legal_test/semantic_llm.extractions.jsonl \
  --llm-retries 5 \
  --retry-sleep 3 \
  --llm-timeout 180

conda run --no-capture-output -n signpost-re python -m signpost.indexing.structure_graph \
  --namespace legal_test-llm \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --document-trees datasets/processed/legal_test/document_trees.jsonl \
  --output datasets/processed/legal_test/graph.structure.json \
  --summarizer llm \
  --max-summary-tokens 512 \
  --cluster-token-budget 4096

conda run -n signpost-re python -m signpost.indexing.sequence_graph \
  --namespace legal_test \
  --chunks datasets/processed/legal_test/chunks.jsonl \
  --output datasets/processed/legal_test/graph.sequence.json

conda run -n signpost-re python -m signpost.graph.merge \
  --namespace legal_test \
  --semantic datasets/processed/legal_test/graph.semantic.json \
  --structure datasets/processed/legal_test/graph.structure.json \
  --sequence datasets/processed/legal_test/graph.sequence.json \
  --output datasets/processed/legal_test/graph.unified.json
```

### F10

```bash
conda run -n signpost-re python -m signpost.indexing.graph_es_sync \
  --namespace legal_test-ecnu \
  --graph datasets/processed/legal_test/graph.unified.json \
  --embedding-provider ecnu \
  --batch-size 4 \
  --recreate
```

### F13

```bash
conda run -n signpost-re python -m signpost.retrieval.run \
  --namespace legal_test-ecnu \
  --query "What is the purpose of the agreement?" \
  --graph datasets/processed/legal_test/graph.unified.json \
  --embedding-provider ecnu \
  --mode hybrid \
  --output outputs/legal_test/retrieval_result.json
```

### F15/F16

```bash
conda run -n signpost-re python -m signpost.agent.batch \
  --namespace legal_test-ecnu \
  --dataset legal_test \
  --questions datasets/processed/legal_test/questions.jsonl \
  --output outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --query-log outputs/legal_test/logs/signpost.ecnu.query.jsonl \
  --embedding-provider ecnu \
  --use-es \
  --limit 5

conda run -n signpost-re python -m signpost.evaluation.evaluate_basic \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/basic_eval.json
```

### Benchmark

```bash
conda run -n signpost-re python -m signpost.benchmark.query_metrics \
  --input outputs/legal_test/predictions/signpost.ecnu.jsonl \
  --output outputs/legal_test/metrics/signpost.query_metrics.json
```

## 24. legal\_lite 正式轻量实验顺序

`legal_lite` 和 `legal_test` 的命令结构一致，只需要把路径中的 `legal_test` 换成 `legal_lite`，namespace 建议使用 `legal_lite-ecnu` 或 `legal_lite-llm` 区分模型版本。

注意：这里列出的命令也是内部功能命令。正式实验同样要使用 `signpost.benchmark.time_stage` 包装运行，并写入 `outputs/legal_lite/logs/stage_timing.jsonl`。不要直接运行裸命令后再试图事后补时间。

准备数据：

```bash
conda run -n signpost-re python -m signpost.data.prepare --datasets legal_lite
conda run -n signpost-re python -m signpost.data.validate --dataset legal_lite
```

解析和 chunk：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal_lite/raw_corpus.jsonl \
  --output datasets/processed/legal_lite/documents.jsonl

conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_lite/documents.jsonl \
  --output datasets/processed/legal_lite/chunks.jsonl \
  --tree-output datasets/processed/legal_lite/document_trees.jsonl \
  --max-tokens 1200 \
  --overlap-tokens 100
```

LLM 语义图建议使用 cache 和 progress：

```bash
conda run --no-capture-output -n signpost-re python -m signpost.indexing.semantic_graph \
  --namespace legal_lite-llm \
  --chunks datasets/processed/legal_lite/chunks.jsonl \
  --output datasets/processed/legal_lite/graph.semantic.llm.json \
  --extractor llm \
  --gleaning-rounds 1 \
  --progress-every 10 \
  --progress-file datasets/processed/legal_lite/semantic_llm.progress.jsonl \
  --extractions-cache datasets/processed/legal_lite/semantic_llm.extractions.jsonl \
  --llm-retries 5 \
  --retry-sleep 3 \
  --llm-timeout 180
```

## 25. 当前实现边界

已经实现：

- F0-F16 主流程。
- raw-level `legal_test` 和 `legal_lite`。
- F5 BM25/dense/hybrid chunk 检索。
- F6 deterministic/LLM 抽取、progress、cache、retry、timeout。
- F7 deterministic/LLM summary。
- F8 顺序视图。
- F9 多视图统一图。
- F10 graph object ES 同步。
- F11/F12 路标。
- F13 grouped retrieval。
- F14 ReadFile。
- F15 deterministic/LLM agent。
- F16 EM/F1/LLM judge。
- Benchmark 指标与日志。

暂不包含：

- 前端。
- 用户/租户/权限。
- 产品 API。
- 检索剪枝算法实现。
- baseline 实现。
- PDF/DOCX 专门解析器。

