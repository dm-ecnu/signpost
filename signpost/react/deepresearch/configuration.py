"""配置管理模块

完全移除smolagents依赖，使用pydantic-settings自动处理环境变量。
"""

import logging
import os
from typing import List, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoTokenizer

# 支持的语言类型
LanguageType = Literal["zh", "en"]


class Configuration(BaseSettings):
    """深度研究代理配置

    使用pydantic-settings自动从环境变量加载配置。
    字段名自动映射为大写环境变量（如 model_id -> MODEL_ID）。
    特殊映射使用 validation_alias（如 OPENAI_API_KEY -> api_key）。
    """

    # 通用配置（注意：不能用 "language"，会与系统环境变量 LANGUAGE 冲突）
    prompt_language: LanguageType = Field(default="en", description="提示词和报告的语言设置（'zh': 中文, 'en': 英文）")

    # 研究配置
    max_researcher_iterations: int = Field(default=32, description="研究监督者的最大研究迭代次数")
    max_react_tool_calls: int = Field(default=16, description="单个研究者步骤中的最大工具调用迭代次数")
    max_parallel_tools: int = Field(default=5, ge=1, le=20, description="并行工具调用的最大线程数（修复 P2: 从 10 降低到 5，避免并发过载）")

    # 知识库配置
    kb_id: str = Field(default="", description="知识库ID，每个任务使用单一kb_id")
    tenant_id: str = Field(default="", description="租户ID（用于租户隔离）")

    # 模型配置 - 只支持OpenAI兼容接口
    model_id: str = Field(default="qwen-plus-thinking", description="OpenAI模型ID，用于所有研究任务")
    api_base: str = Field(
        default="",
        description="OpenAI API基础URL",
        validation_alias=AliasChoices("api_base", "OPENAI_API_BASE"),
    )
    api_key: str = Field(default="", description="OpenAI API密钥", validation_alias=AliasChoices("api_key", "OPENAI_API_KEY"))
    max_tokens: int = Field(default=32000, description="模型最大输出token数")

    # Compact功能配置（Researcher使用）
    keep_score_threshold: int = Field(default=7, description="Researcher压缩时保留的分数阈值，高于此分数的chunk不会被删除")
    min_keep_k: int = Field(default=8, description="Researcher压缩时每条消息至少保留的chunk数量")

    # 上下文管理配置（统一架构）
    max_context_length: int = Field(default=131000, description="模型最大上下文长度（token数）")
    enable_context_compress: bool = Field(default=False, description="是否启用上下文压缩（False则直接强制生成最终答案）")
    context_check_threshold: float = Field(default=0.83, description="强制生成最终答案的阈值（占max_context_length的比例，0.85 = 85%）")
    compress_min_reduction_ratio: float = Field(default=0.10, description="压缩至少需要释放的比例（否则视为失败），例如0.10表示至少释放10%")

    # 日志配置
    log_root_path: str = Field(default="logs", description="日志根目录路径")

    model_config = SettingsConfigDict(
        env_file=".env",  # 可选：支持从.env文件加载
        env_file_encoding="utf-8",
        case_sensitive=False,  # 环境变量不区分大小写
        extra="ignore",  # 忽略额外的环境变量
    )

    def get_kb_ids(self) -> List[str]:
        """获取知识库ID列表，为后续多kb扩展做准备

        Returns:
            List[str]: kb_id列表，当前只包含单一kb_id
        """
        return [self.kb_id]


# ========== 全局Tokenizer初始化（用于上下文长度管理） ==========
#
# Tokenizer 配置说明：
# - tokenizer 路径可通过环境变量 DEEPRESEARCH_TOKENIZER_PATH 覆盖
# - 默认值使用公开可访问的 Hugging Face 模型名
# - 使用单例模式，全局共享一个 tokenizer 实例
# - 必须与实际使用的 LLM 模型匹配，以确保 token 计算准确
#
# 如需更改路径，请设置环境变量 DEEPRESEARCH_TOKENIZER_PATH
#
TOKENIZER_PATH = os.getenv("DEEPRESEARCH_TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")
_GLOBAL_TOKENIZER = None


def get_global_tokenizer():
    """获取全局tokenizer实例（单例模式）

    tokenizer 路径通过常量 TOKENIZER_PATH 配置，可由环境变量
    DEEPRESEARCH_TOKENIZER_PATH 覆盖。

    Returns:
        AutoTokenizer: 全局tokenizer实例

    Raises:
        RuntimeError: tokenizer加载失败时抛出
    """
    global _GLOBAL_TOKENIZER

    if _GLOBAL_TOKENIZER is None:
        try:
            logging.info(f"正在加载tokenizer: {TOKENIZER_PATH}")
            _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            logging.info("Tokenizer加载成功")
        except Exception as e:
            error_msg = f"Tokenizer加载失败 [{TOKENIZER_PATH}]: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    return _GLOBAL_TOKENIZER
