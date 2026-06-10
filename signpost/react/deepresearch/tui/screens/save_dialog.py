"""Save dialog"""

from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class SaveDialog(ModalScreen):
    DEFAULT_FILENAME = "research_report.md"
    CSS = """
    SaveDialog { align: center middle; }
    #dialog-container { width: 60; height: 15; border: thick $accent; background: $surface; padding: 1; }
    #button-row { layout: horizontal; height: 3; align: center middle; }
    Button { margin: 0 1; }
    """

    def __init__(self, report_content: str):
        super().__init__()
        self.report_content = report_content

    def compose(self):
        yield Container(
            Vertical(
                Label("保存研究报告", id="title"),
                Label(""),
                Label("文件名:"),
                Input(value=self.DEFAULT_FILENAME, placeholder="report.md", id="filename-input"),
                Label(""),
                Container(
                    Button("保存", variant="primary", id="save-btn"),
                    Button("取消", variant="default", id="cancel-btn"),
                    id="button-row",
                ),
            ),
            id="dialog-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            filename_input = self.query_one("#filename-input", Input)
            filename = filename_input.value or self.DEFAULT_FILENAME
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.report_content)
                self.dismiss({"success": True, "filename": filename})
            except Exception as e:
                self.dismiss({"success": False, "error": str(e)})
        elif event.button.id == "cancel-btn":
            self.dismiss(None)
