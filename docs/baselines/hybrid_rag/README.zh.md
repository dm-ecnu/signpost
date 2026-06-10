# Hybrid RAG Baseline

完整操作手册见：

```text
docs/baselines/hybrid_rag/runbook.zh.md
```

H200 上同时安排 agriculture/legal 与 Vanilla LLM 的 tmux 执行手册见：

```text
docs/baselines/h200_vanilla_hybrid_tmux_runbook.zh.md
```

该 baseline 不需要新 conda 环境、数据库或额外官方仓库；它需要已有 Signpost 环境、Elasticsearch chunk index、chat endpoint 和 embedding endpoint。
