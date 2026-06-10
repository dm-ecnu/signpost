"""DeepResearch Textual TUI主应用"""

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, RichLog

from .event_renderer import EventRenderer
from .screens.main_screen import MainScreen
from .screens.save_dialog import SaveDialog
from .widgets.header_bar import HeaderBar
from ..configuration import Configuration
from ..events import EventType
from ..supervisor import Supervisor


class DeepResearchApp(App):
    """DeepResearch TUI应用"""

    CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        ("q", "quit", "退出"),
        ("space", "pause_resume", "暂停/继续"),
        ("s", "save_report", "保存报告"),
        ("ctrl+c", "quit", "强制退出"),
    ]

    def __init__(self, task: str, config: Configuration):
        super().__init__()
        self.research_task = task
        self.config = config
        self.paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.final_report = ""
        self.renderer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MainScreen(id="main")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "DeepResearch"
        self.sub_title = f"任务: {self.research_task[:50]}..."

        llm_output = self.query_one("#llm-output", RichLog)
        status_log = self.query_one("#status-log", RichLog)
        header_bar = self.query_one("#header-bar", HeaderBar)

        header_bar.update_info(
            kb_id=self.config.kb_id,
            model_id=self.config.model_id,
            total_steps=self.config.max_researcher_iterations,
        )

        self.renderer = EventRenderer(llm_output, status_log)
        self.run_worker(self._run_research(), exclusive=True)

    async def _run_research(self):
        try:
            supervisor = Supervisor(self.config)

            for event in supervisor(self.research_task):
                await self.pause_event.wait()
                self.call_from_thread(self.renderer.render_event, event)

                if event.event_type == EventType.RESEARCH_COMPLETED:
                    self.final_report = event.content

                await asyncio.sleep(0)

        except Exception as e:
            self.notify(f"错误: {str(e)}", severity="error")
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("研究执行失败")

    def action_quit(self) -> None:
        self.exit()

    def action_pause_resume(self) -> None:
        self.paused = not self.paused

        if self.paused:
            self.pause_event.clear()
            self.notify("研究已暂停", title="暂停")
            self.sub_title = f"[暂停] {self.research_task[:50]}..."
        else:
            self.pause_event.set()
            self.notify("研究继续", title="继续")
            self.sub_title = f"任务: {self.research_task[:50]}..."

    def action_save_report(self) -> None:
        if not self.final_report:
            self.notify("暂无报告内容", severity="warning")
            return

        self.push_screen(SaveDialog(self.final_report), self._handle_save_result)

    def _handle_save_result(self, result) -> None:
        if result is None:
            return

        if result.get("success"):
            filename = result.get("filename")
            self.notify(f"报告已保存到: {filename}", title="成功")
        else:
            error = result.get("error")
            self.notify(f"保存失败: {error}", severity="error", title="失败")
