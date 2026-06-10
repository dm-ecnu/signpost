"""Header bar"""

from textual.widgets import Static


class HeaderBar(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kb_id = ""
        self.model_id = ""
        self.current_step = 0
        self.total_steps = 0

    def on_mount(self) -> None:
        self.refresh_display()

    def update_info(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.refresh_display()

    def refresh_display(self):
        info_text = f"知识库: [cyan]{self.kb_id}[/cyan] | 模型: [yellow]{self.model_id}[/yellow] | 步骤: [green]{self.current_step}/{self.total_steps}[/green]"
        self.update(info_text)
