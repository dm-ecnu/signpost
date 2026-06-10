"""Click CLI 和 Textual TUI"""

import click

from .configuration import Configuration
from .tui.app import DeepResearchApp


@click.command()
@click.argument("task")
@click.option("--kb-id", required=True, help="知识库ID")
@click.option("--tenant-id", required=True, help="租户ID")
@click.option("--model", default="qwen", help="模型ID")
def run(task, kb_id, tenant_id, model):
    """运行DeepResearch深度研究TUI

    示例:
        deepresearch run "分析GraphRAG技术原理" --kb-id my_kb --tenant-id my_tenant
    """
    # 加载配置
    cfg = Configuration(kb_id=kb_id, tenant_id=tenant_id, model_id=model)

    # 启动TUI应用
    app = DeepResearchApp(task=task, config=cfg)
    app.run()


@click.group()
def cli():
    """DeepResearch CLI"""
    pass


cli.add_command(run)


if __name__ == "__main__":
    cli()
