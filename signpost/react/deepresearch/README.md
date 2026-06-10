# DeepResearch V2

DeepResearch V2 是一个基于 ReAct 框架的深度研究 Agent 系统，支持多智能体协作、知识图谱检索和结构化日志追踪。

## 核心特性

### 1. ReAct Agent 架构
- **ReActAgent 基类**：实现 Reasoning-Acting-Observing 循环
- **Supervisor（监督者）**：负责任务分解和多研究协调
- **Researcher（研究者）**：负责单主题深度研究

### 2. 工具自包含格式化
- 每个工具通过 `format_for_llm()` 方法自行格式化输出
- 工具执行返回 JSON，格式化为 XML 给 LLM
- 单一职责原则，格式化逻辑随工具定义

### 3. 类型安全的枚举系统
- **ExitReason**：Agent 退出原因（MAX_ITERATIONS、NO_TOOL_CALLS）
- **ContextDecision**：上下文决策（CONTINUE、FORCE_FINISH）
- **TraceStatus**：追踪状态（SUCCESS、ERROR、COMPLETED）

### 4. 结构化日志追踪
- **TraceLogger**：JSONL 格式的结构化日志
- **TraceEmitter**：事件发射器，支持层级追踪
- **TraceSession**：会话管理，自动记录 Agent 生命周期

### 5. Pythonic API 设计
- Agent 实例可直接调用：`supervisor(task)`
- 流式输出：`for event in supervisor(task)`
- 符合 Python 习惯的接口设计

## 快速开始

### 基本使用

```python
from deepresearch_v2.configuration import Configuration, Language
from deepresearch_v2.supervisor import DeepResearchSupervisor
from deepresearch_v2.trace_logger import TraceLogger

# 创建配置
config = Configuration(
    kb_id="your_kb_id",
    tenant_id="your_tenant",
    model_id="qwen",
    language=Language.CHINESE,
)

# 初始化日志
trace_logger = TraceLogger(log_dir="./logs", kb_id=config.kb_id)

# 创建 Supervisor
supervisor = DeepResearchSupervisor(
    config=config,
    trace_logger=trace_logger,
)

# 执行研究（流式输出）
task = "请分析 GraphRAG 和 RAPTOR 的技术差异"
for event in supervisor(task):  # 使用 __call__ 方法
    if event.type == "stream_chunk":
        print(event.content, end="", flush=True)
    elif event.type == "final_report":
        print(f"\n\n最终报告：{event.content}")
```

### 环境变量配置

```bash
# OpenAI 兼容接口配置
export OPENAI_API_KEY=your_api_key
export OPENAI_API_BASE=https://api.openai.com/v1

# 知识库配置
export KB_ID=your_kb_id
export TENANT_ID=your_tenant_id

# 批处理脚本可选配置
export DEEPRESEARCH_DATASETS_ROOT=./datasets
export DEEPRESEARCH_KB_ID_LEGAL=your_legal_kb_id

# 可选配置
export LANGUAGE=chinese  # 或 english
export MAX_CONCURRENT_RESEARCHERS=3
```

### CLI 运行

```bash
# 运行研究任务
uv run python deepresearch_v2/cli_runner.py \
  --kb-id <KB_ID> \
  --task "请分析这些论文中的核心技术方案"

# 使用环境变量
export KB_ID=your_kb_id
uv run python deepresearch_v2/cli_runner.py \
  --task "研究任务描述"
```

## 架构设计

### 核心组件

```
┌──────────────────────────────────────────┐
│     DeepResearchSupervisor（监督者）      │
│  ┌────────────────────────────────────┐  │
│  │ 系统提示：包含知识库概览            │  │
│  │ 工具：                              │  │
│  │   - ResearchTool                   │  │
│  │   - ResearchCompleteTool           │  │
│  │ ReAct 循环：规划和协调              │  │
│  └────────────────────────────────────┘  │
└────────────────┬─────────────────────────┘
                 │ ResearchTool
                 ▼
┌──────────────────────────────────────────┐
│        Researcher（研究者）              │
│  ┌────────────────────────────────────┐  │
│  │ 工具：                              │  │
│  │   - KnowledgeSearchTool            │  │
│  │   - ReadFileTool                   │  │
│  │ ReAct 循环：检索和分析              │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### 执行流程

```
用户查询
  │
  ▼
Supervisor(task)
  │
  ├─> ReAct 循环
  │   ├─> LLM 生成研究计划
  │   ├─> 调用 ResearchTool("子主题1")
  │   │   └─> Researcher(topic)
  │   │       ├─> KnowledgeSearchTool.execute()
  │   │       │   └─> format_for_llm()  # 格式化为 XML
  │   │       ├─> ReadFileTool.execute()
  │   │       │   └─> format_for_llm()  # 格式化为 XML
  │   │       └─> 生成研究报告
  │   │
  │   ├─> 调用 ResearchTool("子主题2")
  │   │   └─> Researcher(topic)
  │   │       └─> ...
  │   │
  │   └─> 调用 ResearchCompleteTool
  │       └─> 检测到信号，调用 generate_final_answer()
  │           └─> 生成综合报告
  │
  └─> 返回 DeepResearchEvent 流
```

### 工具格式化架构

```python
# 工具执行返回 JSON
class KnowledgeSearchTool(Tool):
    def execute(self, query: str) -> str:
        # 调用检索系统
        results = kg_search.vector_retrieval(query)
        # 返回 JSON 格式
        return json.dumps({"chunks": results})

    def format_for_llm(self, result: str) -> str:
        # 格式化为 XML 给 LLM
        return format_tool_response_for_llm(result)
        # 输出: <knowledge_search><chunk>...</chunk></knowledge_search>

# Agent 自动调用格式化
class ReActAgent:
    def _collect_tool_results(self, tool_results):
        for tool_call_id, exec_result in tool_results.items():
            tool = self.tools.get(tool_name)
            # 自动调用工具的格式化方法
            formatted_result = tool.format_for_llm(exec_result.result)
            # 添加到 memory
            self.memory.add_message(Message(..., content=formatted_result))
```

## API 参考

### Configuration

配置类，管理所有 Agent 参数。

```python
config = Configuration(
    kb_id: str,                             # 知识库ID（必填）
    tenant_id: str = "default_tenant",      # 租户ID
    model_id: str = "qwen",                 # OpenAI 模型ID
    language: Language = Language.CHINESE,  # 提示词语言
    max_concurrent_researchers: int = 3,    # 最大并发研究者
    max_iterations: int = 15,               # 最大迭代次数
)
```

### DeepResearchSupervisor

研究监督者，负责任务分解和协调。

```python
from deepresearch_v2.supervisor import DeepResearchSupervisor

supervisor = DeepResearchSupervisor(
    config=config,
    trace_logger=trace_logger,
)

# 执行研究（流式）
for event in supervisor(task):
    # 处理事件
    pass
```

### Researcher

单主题研究者，负责检索和分析。

```python
from deepresearch_v2.researcher import Researcher

researcher = Researcher(
    config=config,
    trace_logger=trace_logger,
)

# 执行研究（流式）
for event in researcher(task):
    # 处理事件
    pass
```

### DeepResearchEvent

所有流式事件的统一格式。

```python
@dataclass
class DeepResearchEvent:
    type: str                    # 事件类型
    content: Optional[str]       # 事件内容
    metadata: Optional[Dict]     # 元数据
```

事件类型：
- `stream_chunk`: 流式内容块
- `final_report`: 最终研究报告
- `tool_call`: 工具调用
- `tool_result`: 工具执行结果
- `error`: 错误事件

### TraceLogger

结构化日志系统。

```python
from deepresearch_v2.trace_logger import TraceLogger, TraceStatus

trace_logger = TraceLogger(log_dir="./logs", kb_id="kb_123")

# 创建会话
with trace_logger.session(task="研究任务") as session:
    # 记录 Agent 启动
    session.emit_agent_start(
        system_prompt="...",
        tools=[...],
    )

    # 记录工具调用
    session.emit_tool_call(
        tool_call_id="call_123",
        tool_name="knowledge_search",
        arguments={"query": "..."},
    )

    # 记录工具结果
    session.emit_tool_result(
        tool_call_id="call_123",
        result="...",
    )

    # 记录最终结果
    session.emit_final(
        status=TraceStatus.SUCCESS,
        final_answer="...",
        token_usage={...},
    )

# 日志路径：./logs/kb_123/YYYY-MM-DD/HH-MM-SS_<trace_id>.jsonl
```

### 枚举类型

```python
from deepresearch_v2.types import ExitReason, ContextDecision, TraceStatus

# Agent 退出原因
exit_reason = ExitReason.MAX_ITERATIONS
print(exit_reason.description)  # "达到最大迭代次数"

# 上下文决策
decision = ContextDecision.FORCE_FINISH

# 追踪状态
status = TraceStatus.SUCCESS
```

## 工具系统

### 内置工具

#### KnowledgeSearchTool
调用 GraphRAG 检索系统，检索知识图谱中的相关内容。

```python
# 工具定义
{
    "name": "knowledge_search",
    "description": "在知识库中检索相关信息",
    "parameters": {
        "query": {
            "type": "string",
            "description": "检索查询"
        }
    }
}

# 使用示例（LLM 自动调用）
# 输入：{"query": "GraphRAG 实体抽取"}
# 输出（格式化后）：
# <knowledge_search>
#   <chunk>
#     <content>实体抽取使用 LLM...</content>
#     <source>doc_123.pdf</source>
#   </chunk>
# </knowledge_search>
```

#### ReadFileTool
读取知识库中的文件内容。

```python
# 工具定义
{
    "name": "read_file",
    "description": "读取知识库文件",
    "parameters": {
        "filename": {"type": "string"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"}
    }
}
```

#### ResearchTool（仅 Supervisor）
创建 Researcher 执行子研究。

```python
# 工具定义
{
    "name": "research",
    "description": "执行深度研究",
    "parameters": {
        "topic": {"type": "string"}
    }
}
```

#### ResearchCompleteTool（仅 Supervisor）
信号工具，触发最终报告生成。

```python
# 工具定义
{
    "name": "research_complete",
    "description": "完成研究并生成最终报告"
}

# 注意：execute() 方法会抛出 RuntimeError
# 这是信号工具，不会实际执行
```

### 自定义工具

```python
from deepresearch_v2.tools import Tool

class MyCustomTool(Tool):
    name = "my_tool"
    description = "自定义工具描述"
    inputs = {
        "param1": {
            "type": "string",
            "description": "参数1描述"
        }
    }
    output_type = "string"

    def execute(self, param1: str) -> str:
        # 工具执行逻辑
        result = {"data": f"处理 {param1}"}
        return json.dumps(result)

    def format_for_llm(self, result: str) -> str:
        # 格式化为 LLM 友好格式
        data = json.loads(result)
        return f"<my_tool>\n{data['data']}\n</my_tool>"

# 添加到 Researcher
researcher = Researcher(
    config=config,
    trace_logger=trace_logger,
    additional_tools=[MyCustomTool()],
)
```

## 测试

### 运行单元测试

```bash
# 运行所有 deepresearch_v2 测试
uv run pytest test/unit/test_deepresearch/ -v

# 运行特定测试文件
uv run pytest test/unit/test_deepresearch/test_agent.py -v
uv run pytest test/unit/test_deepresearch/test_tools.py -v

# 查看覆盖率
uv run pytest test/unit/test_deepresearch/ --cov=deepresearch_v2
```

### Mock 配置

单元测试使用 Mock 隔离外部依赖：

```python
from unittest.mock import Mock, patch

@patch('deepresearch_v2.agent.OpenAI')
def test_agent_execution(mock_openai):
    # Mock LLM 响应
    mock_openai.return_value.chat.completions.create.return_value = ...

    # 测试 Agent 逻辑
    agent = ReActAgent(...)
    for event in agent(task):
        assert event.type in ["stream_chunk", "final_report"]
```

## 设计原则

### 1. 工具自包含
- 每个工具拥有自己的格式化逻辑
- 工具执行返回 JSON，格式化为 XML
- 单一职责，易于测试和扩展

### 2. 类型安全
- 使用 Enum 代替魔法字符串
- 使用 dataclass 定义数据结构
- 类型提示覆盖所有公共 API

### 3. 流式优先
- 只提供流式接口（`__call__`）
- 所有输出通过事件流返回
- 适合实时交互场景

### 4. 结构化日志
- JSONL 格式，每行一个事件
- 支持层级追踪（trace_id、parent_id）
- 完整记录 Agent 生命周期

### 5. Pythonic API
- 使用 `__call__` 魔法方法
- 符合 Python 习惯的命名和接口
- 易于理解和使用

## 性能特性

### Token 管理
- 自动估算 token 消耗
- 达到阈值时压缩记忆
- 保留高分 chunk，丢弃低分内容

### 并发控制
- 可配置最大并发研究者数量
- 避免过多并发请求

### 错误恢复
- 完善的异常处理
- 工具执行失败自动记录错误
- Agent 可以根据错误信息调整策略

## 常见问题

### Q: 为什么使用 `__call__` 而不是 `run()`？
A: `__call__` 让 Agent 实例可以像函数一样调用，更符合 Python 习惯：
```python
supervisor = DeepResearchSupervisor(config, trace_logger)
for event in supervisor(task):  # 直接调用
    process(event)
```

### Q: 如何获取最终研究报告？
A: 监听 `final_report` 事件：
```python
for event in supervisor(task):
    if event.type == "final_report":
        final_report = event.content
        break
```

### Q: ResearchCompleteTool 为什么会抛出异常？
A: 这是一个信号工具，不会实际执行。当 Agent 检测到这个工具调用时，会直接调用 `generate_final_answer()` 生成最终报告，而不是执行 `execute()` 方法。

### Q: 如何调试 Agent 行为？
A: 通过监听所有事件或查看日志文件：
```python
# 方式1：实时监听
for event in supervisor(task):
    print(f"[{event.type}] {event}")

# 方式2：查看日志
# ./logs/kb_123/YYYY-MM-DD/HH-MM-SS_<trace_id>.jsonl
```

### Q: 工具格式化失败怎么办？
A: `format_for_llm()` 失败时，会返回原始 JSON 结果：
```python
def format_for_llm(self, result: str) -> str:
    formatted = format_tool_response_for_llm(result)
    return formatted if formatted else result  # 失败时返回原始结果
```

## 相关文档

- **项目总览**：[../README.md](../README.md)
- **项目指导**：[../CLAUDE.md](../CLAUDE.md)
- **GraphRAG 架构**：见 CLAUDE.md 的"GraphRAG 核心技术架构"章节
- **测试框架**：见 CLAUDE.md 的"测试框架"章节

## 许可证

Apache License 2.0
