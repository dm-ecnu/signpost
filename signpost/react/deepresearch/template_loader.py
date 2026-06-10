"""Unified prompt template management (load + render)

Features:
1. Load YAML templates from deepresearch.prompts package by profile and language
2. Use Jinja2 to render templates with knowledge base overview, tool info, and other variables
3. Provide template validation and tool formatting helper functions
"""

from datetime import datetime
from typing import Dict, Any, Literal
import importlib.resources as ir
import yaml
from jinja2 import Template, StrictUndefined

from .configuration import LanguageType

Profile = Literal["researcher", "supervisor"]

# 每个 profile 必需的模板键
REQUIRED_KEYS: Dict[str, set[str]] = {
    "researcher": {
        "system_prompt",
        "final_report_generation_prompt",
        "compact_system_prompt",
    },
    "supervisor": {
        "system_prompt",
        "final_report_generation_prompt",
    },
}


def _resolve_filename(profile: Profile, language: LanguageType) -> str:
    if profile == "researcher":
        return "researcher_en.yaml" if language == "en" else "researcher_zh.yaml"
    if profile == "supervisor":
        return "supervisor_en.yaml" if language == "en" else "supervisor_zh.yaml"
    raise ValueError(f"未知的模板profile: {profile}")


def _ensure_required_keys(profile: Profile, templates: Dict[str, Any]) -> None:
    """检查模板是否包含所有必需的键

    Args:
        profile: 角色类型
        templates: 已加载的模板字典

    Raises:
        ValueError: 缺少必需键时抛出
    """
    required = REQUIRED_KEYS[profile]
    missing = [k for k in required if k not in templates]
    if missing:
        raise ValueError(f"模板缺少必需键: {missing} (profile={profile})")


def load_templates(profile: Profile, language: LanguageType) -> Dict[str, Any]:
    """加载并校验提示词模板

    Args:
        profile: 角色类型 (researcher/supervisor)
        language: 语言 ('zh': 中文, 'en': 英文)

    Returns:
        模板字典

    Raises:
        ValueError: 文件为空、格式错误或缺少必需键
    """
    filename = _resolve_filename(profile, language)
    data = ir.files("deepresearch.prompts").joinpath(filename).read_text(encoding="utf-8")

    templates = yaml.safe_load(data)

    # 检查空文件
    if templates is None:
        raise ValueError(f"模板文件为空或内容为null: {filename}")

    # 检查类型
    if not isinstance(templates, dict):
        raise ValueError(f"模板文件格式错误: {filename}，期望字典但得到 {type(templates).__name__}")

    # 检查必需键
    _ensure_required_keys(profile, templates)

    return templates


# ==================== 模板渲染功能 ====================


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """使用 Jinja2 渲染模板

    Args:
        template: Jinja2 模板字符串
        variables: 变量字典

    Returns:
        渲染后的字符串

    Raises:
        Exception: 模板渲染失败时抛出
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"模板渲染错误: {type(e).__name__}: {e}") from e


def get_today_str() -> str:
    """Get today's date string (YYYY-MM-DD format)"""
    return datetime.now().strftime("%Y-%m-%d")


def get_default_variables() -> Dict[str, Any]:
    """Get default template variables

    Returns:
        dict: Default variables dict (empty for future extensibility)
    """
    return {}
