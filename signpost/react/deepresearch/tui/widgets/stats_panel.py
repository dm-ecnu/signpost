"""Stats panel"""

from textual.reactive import reactive
from textual.widgets import Static


class StatsPanel(Static):
    current_step = reactive(0)
    total_steps = reactive(0)
    token_count = reactive(0)
    elapsed_time = reactive(0.0)
    tool_calls = reactive(0)

    def render(self) -> str:
        return f"统计信息\n{'━' * 20}\n步骤: {self.current_step}/{self.total_steps}\nToken: {self.token_count:,}\n耗时: {self.elapsed_time:.1f}s\n工具调用: {self.tool_calls}次"

    def update_stats(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
