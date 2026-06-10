"""Main screen"""

from textual.containers import Container
from textual.screen import Screen
from textual.widgets import RichLog
from ..widgets.header_bar import HeaderBar
from ..widgets.stats_panel import StatsPanel


class MainScreen(Screen):
    def compose(self):
        yield HeaderBar(id="header-bar")
        yield RichLog(id="llm-output", highlight=True, markup=True, wrap=True)
        yield Container(
            StatsPanel(id="stats-panel"),
            RichLog(id="status-log", highlight=True, markup=True),
            id="bottom-container",
        )
