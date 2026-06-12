# ReAct 移植进 signpost_re — 进度与 H200 调试 Runbook（2026-06-09）

> 目标：把师兄完整 ReAct 多智能体（`signpost-main/deepresearch`，对应毕业技术说明算法4.1/表4.2：ReAct 多步、4 类 cue 序列化给 agent 自主导航、工作记忆压缩）移植进 项目实验用的 `signpost_re`，让技术说明 §5.1/§4 描述与实际跑的代码一致，然后 5 数据集重跑。
>
> 工作基底：`/home/ruolinsu/signpost/code/signpost_re_merged`（= 你给的 v2 权威 + v4 legal baseline 合并；原始两套在 `signpost_re_code_20260609/` 不动，可回退）。

## 背景结论（已查实）
- 项目技术说明数据是 `signpost_re`（简化版：Supervisor decompose + Researcher 单次检索只读 provenance locate → ReadFile + synthesize = **2 次 LLM 调用**）跑的。
- 完整 ReAct（多步、4 cue 喂 agent、压缩）在 `signpost-main/deepresearch`（2026-05-09，师兄毕业技术说明版），用 `KGSearchResult` 数据结构 + `>` 标记把路标序列化给 LLM。
- 移植 = 把 `deepresearch` 整栈搬进 `signpost_re`，写适配器把 signpost_re 的 ES 检索结果转成 `KGSearchResult`，LLMCore 接 H200。重跑后 **"2 LLM calls" 会变成 decompose(1)+N×ReAct+synthesize ≈ 表4.2 的 KnowledgeSearch 11-13 次/查询**，技术说明 2-calls 卖点要改。

## ✅ 已完成（本机静态）
1. **阶段1：搬入完整 ReAct 栈** → `signpost_re_merged/signpost/react/`：
   - `react/deepresearch/`（agent.py ReAct循环 / researcher.py 4工具+压缩 / tools.py / supervisor.py / entities/events/types/model_client）
   - `react/core/`（llm/ = LLMCore tool-calling+streaming；logging/ = TraceSession；utils/ = file_utils/knowledge_retrieval）
   - `react/graphrag/retrieval/`（KGSearchResult/InstanceSignpost/GroupSignpost 定义）
2. **核心适配器** `react/react_adapter.py`（语法✓，字段名✓对齐源定义）：
   - `SignpostReRetrievaler.process(query)` → 调 signpost_re 的 `search_chunks/search_graph` + `build_grouped_retrieval_result`（4 cue + PPR）→ `to_kg_search_result()` 转成 `KGSearchResult`。
   - 字段映射：semantic.neighboring_entities→neighboring_entities；vertical.parent_summary.title→parent_node_title；child_summaries→child_node_titles；provenance.source_locates/locate(+horizontal prev/next)→source_locates；online_signpost.recommended_entities→GroupSignpost.related_entities。
   - `RetrievalType` 是 `Literal["original_chunk","graphrag_entity","graphrag_edge","raptor_node"]`，_TYPE_MAP 已对齐。

## ⬜ 剩余步骤（要做）
1. **解 tools.py 依赖**：`from core import config`（kg_retrievaler 挂这里）、`from core.utils.knowledge_retrieval import get_kb_summary_from_es`（GetTOC/Overview 用）、`from core.utils.file_utils`（read_file）。
   - 方案：写 `react/config_shim.py` 提供 `config.kg_retrievaler`（=SignpostReRetrievaler 实例）；把 GetTOC/KnowledgeOverview 接 signpost_re 的文档/graph，或先桩成可选工具。
2. **ReadFileTool 接 signpost_re**：deepresearch 的 read_file 读它自己的存储；改成调 signpost_re 的 `retrieval/read_file.py:read_locate`（按 file:Lx-Ly 读 documents.jsonl）。
3. **LLMCore 配 H200**：`core/llm/core.py` 的 LLMCore 用 OpenAI SDK；配 `base_url=http://localhost:8000/v1`、`model=/data/srl/Llama-3.3-70B-FP8`、`api_key=EMPTY`（H200 .env.h200）。确认 H200 的 vLLM 支持 OpenAI function-calling（tool_calls）——**这是 ReAct 能否跑的关键前提，先验证**。
4. **TraceEmitter/事件**：deepresearch 用流式 TraceEmitter+事件；写最小 shim 把 LLM 调用计数/工具调用记进 signpost_re 的 trace（用于 llm_calls/tool_calls 指标）。
5. **Supervisor 入口**：写 `react/run_react.py`：decompose（可复用 signpost_re 的 supervisor.decompose 或 deepresearch supervisor）→ 每子问题跑 ReAct Researcher（带 SignpostReRetrievaler）→ synthesize → 输出 signpost_re 统一 prediction schema（answer/citations/llm_calls/tool_calls/tokens/trace）。
6. **接消融**：signpost_variant 透传给 SignpostReRetrievaler（full/no_vertical/.../no_offline），保证消融仍走 ReAct。
7. **运行脚本 + 重跑**：仿 `scripts/run_signpost_method.sh` 写 ReAct 版入口；5 数据集（agri/medical/novel/legal/mix）重跑 F15+eval+消融。

## H200 启动（smoke，单数据集）
```bash
cd /home/srl/signpost_re   # 或把 signpost_re_merged 同步上 H200
set -a; source .env.h200; set +a
# 先验证 H200 vLLM 支持 tool_calls（ReAct 前提）：
python3 -c "from openai import OpenAI; c=OpenAI(base_url='http://localhost:8000/v1',api_key='EMPTY'); print(c.chat.completions.create(model='/data/srl/Llama-3.3-70B-FP8', messages=[{'role':'user','content':'hi'}], tools=[{'type':'function','function':{'name':'t','parameters':{'type':'object','properties':{}}}}]).choices[0].message)"
# 若返回 tool_calls 字段 = 支持，ReAct 可跑;否则要改用 prompt-based ReAct(无 function-calling)
```
**把上面 tool_calls 验证的输出发我** —— 它决定移植走 function-calling 还是 prompt-based。

## ⚠️ 关键风险
- H200 的 Llama-3.3-70B-FP8 vLLM **是否支持 OpenAI function-calling（tool_calls）** 未知。不支持则 ReAct 要改 prompt-based（多写一层解析），工作量+1。
- 重跑会改 tab:online（2→多步多次）、tab:quality/silver/F7 全部数字，技术说明要重出表+改 §5 的 2-calls 卖点。
- 6.11 前完成有风险，取决于 H200 调试顺利度。

## 回退
原始代码 `signpost_re_code_20260609/{signpost_re_v2,signpost_re_v4}` 不动；`signpost_re_merged` 是工作副本，`react/` 是新增子包，删掉即恢复简化版。
