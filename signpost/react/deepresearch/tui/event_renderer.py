"""事件渲染器"""

from datetime import datetime
from textual.widgets import RichLog
from ..events import (
    AgentStepStartedEvent,
    ErrorEvent,
    EventType,
    LLMContentDeltaEvent,
    LLMContentDoneEvent,
    ResearchCompletedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)


class EventRenderer:
    def __init__(self, llm_output: RichLog, status_log: RichLog):
        self.llm_output = llm_output
        self.status_log = status_log
        self.current_step = 0

    def render_event(self, event):
        event_type = event.event_type
        if event_type == EventType.LLM_CONTENT_DELTA:
            self._render_llm_delta(event)
        elif event_type == EventType.LLM_CONTENT_DONE:
            self._render_llm_done(event)
        elif event_type == EventType.TOOL_EXECUTION_STARTED:
            self._render_tool_started(event)
        elif event_type == EventType.TOOL_EXECUTION_COMPLETED:
            self._render_tool_completed(event)
        elif event_type == EventType.AGENT_STEP_STARTED:
            self._render_step_started(event)
        elif event_type == EventType.RESEARCH_COMPLETED:
            self._render_research_completed(event)
        elif event_type == EventType.ERROR_OCCURRED:
            self._render_error(event)

    def _render_llm_delta(self, event: LLMContentDeltaEvent):
        self.llm_output.write(event.content, end="")
        self.llm_output.scroll_end(animate=False)

    def _render_llm_done(self, event: LLMContentDoneEvent):
        self.llm_output.write("\n")

    def _render_tool_started(self, event: ToolExecutionStartedEvent):
        timestamp = self._format_timestamp()
        self.status_log.write(f"[dim]{timestamp}[/dim] [yellow]⟳[/yellow] [bold cyan]{event.tool_name}[/bold cyan] 执行中...\n")

    def _render_tool_completed(self, event: ToolExecutionCompletedEvent):
        timestamp = self._format_timestamp()
        duration_str = ""
        if hasattr(event, "duration") and event.duration:
            duration_str = f" ([green]{event.duration:.2f}s[/green])"
        self.status_log.write(f"[dim]{timestamp}[/dim] [green]✓[/green] [bold cyan]{event.tool_name}[/bold cyan] 完成{duration_str}\n")

    def _render_step_started(self, event: AgentStepStartedEvent):
        self.current_step = event.step_number
        separator = "─" * 60
        self.llm_output.write(f"\n[blue]{separator}[/blue]\n[bold blue]Step {self.current_step}[/bold blue]\n[blue]{separator}[/blue]\n\n")
        timestamp = self._format_timestamp()
        self.status_log.write(f"[dim]{timestamp}[/dim] [blue]📍[/blue] [bold]开始 Step {self.current_step}[/bold]\n")

    def _render_research_completed(self, event: ResearchCompletedEvent):
        timestamp = self._format_timestamp()
        separator = "═" * 60
        self.llm_output.write(f"\n[green]{separator}[/green]\n[bold green]✅ 研究完成[/bold green]\n[green]{separator}[/green]\n")
        self.status_log.write(f"[dim]{timestamp}[/dim] [green]✅[/green] [bold green]研究完成[/bold green]\n")

    def _render_error(self, event: ErrorEvent):
        timestamp = self._format_timestamp()
        self.status_log.write(f"[dim]{timestamp}[/dim] [red]❌[/red] [bold red]错误:[/bold red] {event.error_type}\n    {event.error_message}\n")

    @staticmethod
    def _format_timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")
