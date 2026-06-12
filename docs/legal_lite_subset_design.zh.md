# Legal 子集设计：legal_test 与 legal_lite

这次改成 **raw-level 子集**，不是从已经处理好的 `datasets/processed/legal` 里切数据。

原因是：后面会从头重新跑 F3/F3.5/F4/F5...，所以子集应该先放在原始数据层：

```text
datasets/raw/ultradomain/legal_test.jsonl
datasets/raw/ultradomain/legal_lite.jsonl
```

然后再由 F3 生成：

```text
datasets/processed/legal_test/
datasets/processed/legal_lite/
```

## 1. 两个子集的定位

### legal_test

`legal_test` 只用于开发和流程测试。它要足够小，方便快速跑通 F3-F16。

不建议把 `legal_test` 写进技术说明主实验表。

### legal_lite

`legal_lite` 是正式技术说明里的 Legal-Lite 数据集。它比 `legal_test` 大，但比全量 Legal 小，目标是在可控成本下保留法律文档的结构性。

技术说明里应该明确写：

```text
Legal-Lite is a document-complete subset of the UltraDomain Legal split.
```

不能写成全量 Legal。

## 2. 为什么必须按 document 抽样？

不能随机抽 chunk。法律文档强依赖章节、条款、定义、条件、终止、附表和前后顺序。

如果随机抽 chunk，会破坏：

- F4 的 document tree
- F7 的结构图
- F8 的顺序图
- F9 的统一图
- 问题到文档证据的对应关系

所以抽样规则是：

```text
选中完整 document
-> 保留这个 document 对应的所有 raw question rows
-> F3 重新生成 raw_corpus/questions
-> F3.5/F4 重新解析和切块
```

## 3. 当前已经生成的 raw 子集

### legal_test

文件：

```text
datasets/raw/ultradomain/legal_test.jsonl
datasets/raw/ultradomain/legal_test.selection.json
```

规模：

```text
documents: 1
questions/raw rows: 9
chunks after F4: 88
```

选择的文档：

| doc_id | 类型粗略说明 | chunks after F4 | questions |
| --- | --- | ---: | ---: |
| `legal_doc_b9eb62b7885ca06a` | Securities Purchase and Security Agreement | 88 | 9 |

这个文档也放进了 `legal_lite`，方便先在小集上调通流程，再扩大到正式子集。

### legal_lite

文件：

```text
datasets/raw/ultradomain/legal_lite.jsonl
datasets/raw/ultradomain/legal_lite.selection.json
```

规模：

```text
documents: 12
questions/raw rows: 94
```

根据当前已有 chunk 统计，这 12 个文档预计约：

```text
chunks: 1236
```

注意：这个 chunk 数只是基于当前 F4 结果的估计。后面重新跑 F4 后，如果 chunking 参数变化，实际 chunk 数可能略变。

选择的文档：

| doc_id | 类型粗略说明 | 预计 chunks | questions |
| --- | --- | ---: | ---: |
| `legal_doc_b9eb62b7885ca06a` | Securities Purchase and Security Agreement | 88 | 9 |
| `legal_doc_cdf6e75dfe800945` | Loan and Security Agreement | 42 | 7 |
| `legal_doc_83b8a2f37f53f8d3` | Lease Agreement | 29 | 6 |
| `legal_doc_fbf9955498d3f805` | In-Lease Agreement | 62 | 8 |
| `legal_doc_a4aa44f9c5a48d06` | Credit Agreement | 186 | 10 |
| `legal_doc_9fcfd6e2b2b56725` | Term Loan Credit Agreement | 120 | 9 |
| `legal_doc_b68214938b35f6b4` | Executive/Employment-style Agreement | 94 | 8 |
| `legal_doc_927526a46c4dbd78` | Credit Agreement Amendment | 120 | 8 |
| `legal_doc_edb94dd7d9c88cdb` | Sale and Servicing Agreement | 181 | 8 |
| `legal_doc_ddcc2faa8248996a` | LLC Agreement | 190 | 8 |
| `legal_doc_e33c72fc78e4ea3a` | Senior Secured Notes / Purchase Agreement | 65 | 7 |
| `legal_doc_336d9fd251c7651a` | Senior Notes Purchase Agreement | 59 | 6 |

设计意图：

- 保留完整法律文档结构。
- 覆盖 lease、loan、credit、security、purchase、servicing、LLC、notes 等不同合同形态。
- questions 接近 100，适合技术说明轻量主实验。
- 预计 chunks 约 1200，F6 LLM 抽取成本明显低于全量 Legal 的 12692 chunks。

## 4. 子集生成脚本

新增脚本：

```text
signpost/data/create_ultradomain_subset.py
```

它做的是 raw-level 抽样：

```text
输入：datasets/raw/ultradomain/legal.jsonl
输出：datasets/raw/ultradomain/<target_dataset>.jsonl
```

它不会生成：

```text
datasets/processed/<target_dataset>/
```

processed 数据需要之后从 F3 开始重新跑。

## 5. 重新生成 legal_test

```bash
conda run -n signpost-re python -m signpost.data.create_ultradomain_subset \
  --source-dataset legal \
  --target-dataset legal_test \
  --doc-id legal_doc_b9eb62b7885ca06a
```

## 6. 重新生成 legal_lite

```bash
conda run -n signpost-re python -m signpost.data.create_ultradomain_subset \
  --source-dataset legal \
  --target-dataset legal_lite \
  --doc-id legal_doc_b9eb62b7885ca06a \
  --doc-id legal_doc_cdf6e75dfe800945 \
  --doc-id legal_doc_83b8a2f37f53f8d3 \
  --doc-id legal_doc_fbf9955498d3f805 \
  --doc-id legal_doc_a4aa44f9c5a48d06 \
  --doc-id legal_doc_9fcfd6e2b2b56725 \
  --doc-id legal_doc_b68214938b35f6b4 \
  --doc-id legal_doc_927526a46c4dbd78 \
  --doc-id legal_doc_edb94dd7d9c88cdb \
  --doc-id legal_doc_ddcc2faa8248996a \
  --doc-id legal_doc_e33c72fc78e4ea3a \
  --doc-id legal_doc_336d9fd251c7651a
```

## 7. 从头跑 legal_test

F3：

```bash
conda run -n signpost-re python -m signpost.data.prepare \
  --datasets legal_test
```

F3 校验：

```bash
conda run -n signpost-re python -m signpost.data.validate \
  --dataset legal_test
```

F3.5：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal_test/raw_corpus.jsonl \
  --output datasets/processed/legal_test/documents.jsonl
```

F4：

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_test/documents.jsonl \
  --chunks-output datasets/processed/legal_test/chunks.jsonl \
  --trees-output datasets/processed/legal_test/document_trees.jsonl
```

F4 校验：

```bash
conda run -n signpost-re python -m signpost.chunking.validate \
  --chunks datasets/processed/legal_test/chunks.jsonl
```

后续 F5-F16 就按正常数据集跑，namespace 建议用：

```text
legal_test
legal_test-ecnu
legal_test-llm
```

## 8. 从头跑 legal_lite

和 `legal_test` 一样，只是 dataset 名换成：

```text
legal_lite
```

F3：

```bash
conda run -n signpost-re python -m signpost.data.prepare \
  --datasets legal_lite
```

F3.5：

```bash
conda run -n signpost-re python -m signpost.parsing.parse_documents \
  --input datasets/processed/legal_lite/raw_corpus.jsonl \
  --output datasets/processed/legal_lite/documents.jsonl
```

F4：

```bash
conda run -n signpost-re python -m signpost.chunking.run \
  --input datasets/processed/legal_lite/documents.jsonl \
  --chunks-output datasets/processed/legal_lite/chunks.jsonl \
  --trees-output datasets/processed/legal_lite/document_trees.jsonl
```

## 9. 当前文件状态

现在保留的是 raw 子集：

```text
datasets/raw/ultradomain/legal_test.jsonl
datasets/raw/ultradomain/legal_test.selection.json
datasets/raw/ultradomain/legal_lite.jsonl
datasets/raw/ultradomain/legal_lite.selection.json
```

刚才错误生成的：

```text
datasets/processed/legal_test/
```

已经删除。后续需要通过 F3 重新生成。
