"""
GraphRAG统一实体定义

重构后的实体定义，统一了Node、Edge、Chunk、Community等概念。
纯粹的领域实体，不包含存储层逻辑。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class Node:
    """GraphRAG节点实体，代表知识图谱中的一个实体"""

    name: str  # 全局唯一键，原 entity_name
    type: str  # 实体类别
    description: str  # Lazy Merge 后填充，之前为空
    doc_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    pagerank: float = 0.0
    degree: int = 0  # 节点的度数（连接数）
    communities: List[str] = field(default_factory=list)  # 所属社区
    # V2: 存储每个 chunk 的原始抽取内容，格式: {"doc_id:chunk_id": {"content": "...", "type": "PERSON"}}
    content_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def to_nx_node_attrs(self) -> Dict[str, Any]:
        """
        生成NetworkX节点属性字典

        Returns:
            适合NetworkX图节点的属性字典
        """
        return {
            "entity_name": self.name,
            "entity_type": self.type,
            "description": self.description,
            "doc_ids": self.doc_ids.copy(),
            "chunk_ids": self.chunk_ids.copy(),
            "pagerank": self.pagerank,
            # NOTE degree的含义是节点的度，在networkx中直接获取degree。只有es中需要存储degree字段用于检索后的操作。
            # "degree": self.degree,
            "communities": self.communities.copy(),
            "content_mapping": {k: v.copy() for k, v in self.content_mapping.items()},
        }


@dataclass
class Edge:
    """GraphRAG边实体，代表知识图谱中的关系"""

    source: str
    target: str
    relation_type: List[str] = field(default_factory=list)  # 统一的关系类型字段，支持多个类型
    description: str = ""  # Lazy Merge 后填充，之前为空
    weight: float = 1.0
    doc_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    # V2: 存储每个 chunk 的原始抽取内容，格式: {"doc_id:chunk_id": {"content": "...", "type": "RELATION_TYPE"}}
    content_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        # 确保relation_type不为空
        if not self.relation_type:
            self.relation_type = ["RELATED_TO"]

    def to_nx_edge_attrs(self) -> Dict[str, Any]:
        """
        生成NetworkX边属性字典

        Returns:
            适合NetworkX图边的属性字典
        """
        return {
            "src_id": self.source,
            "tgt_id": self.target,
            "description": self.description,
            "weight": self.weight,
            "relation_type": self.relation_type.copy(),
            "doc_ids": self.doc_ids.copy(),
            "chunk_ids": self.chunk_ids.copy(),
            "content_mapping": {k: v.copy() for k, v in self.content_mapping.items()},
        }


@dataclass
class Chunk:
    """GraphRAG文本块实体"""

    id: str
    doc_id: str
    kb_id: str
    content: str
    important_kwd: List[str] = field(default_factory=list)

    # 位置信息
    page_num: int = 0
    top: int = 0
    position: Optional[List[int]] = None

    # 时间信息
    create_time: Optional[datetime] = None
    create_timestamp: float = 0.0

    # 状态信息
    available: int = 1
    doc_type: str = ""
    img_id: str = ""
    docnm_kwd: str = ""

    def to_es(self, embedding_model=None) -> Dict[str, Any]:
        """Chunk不应该被GraphRAG修改"""
        raise NotImplementedError("Chunk entities should not be modified by GraphRAG. Use read-only operations with from_es() instead.")

    @classmethod
    def from_es(cls, doc: Dict[str, Any]) -> "Chunk":
        """从ES文档恢复Chunk实例"""
        # 解析位置信息
        position = doc.get("position_int")

        # 解析时间
        create_time = None
        create_time_str = doc.get("create_time")
        if create_time_str:
            try:
                create_time = datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        return cls(
            id=doc.get("id", ""),
            doc_id=doc.get("doc_id", ""),
            kb_id=doc.get("kb_id", ""),
            content=doc.get("content_with_weight", ""),
            important_kwd=doc.get("important_kwd", []),
            page_num=doc.get("page_num_int", 0),
            top=doc.get("top_int", 0),
            position=position,
            create_time=create_time,
            create_timestamp=doc.get("create_timestamp_flt", 0.0),
            available=doc.get("available_int", 1),
            doc_type=doc.get("doc_type_kwd", ""),
            img_id=doc.get("img_id", ""),
            docnm_kwd=doc.get("docnm_kwd", ""),
        )


@dataclass
class Community:
    """统一的GraphRAG社区实体 - 覆盖所有使用场景"""

    # === 基础标识字段 ===
    id: str  # 社区ID，如"Community_0_1"
    level: int  # 社区层级

    # === 结构字段 ===
    nodes: List[str] = field(default_factory=list)  # 社区包含的节点列表
    edges: List[Tuple[str, str]] = field(default_factory=list)  # 社区内的边列表 (统一使用Tuple格式)
    weight: float = 0.0  # 社区权重
    occurrence: float = 0.0  # 社区出现频率

    # === 报告字段 ===
    title: str = ""  # LLM生成的社区标题
    summary: str = ""  # LLM生成的社区摘要
    report: str = ""  # 格式化的完整报告文本
    findings: List[Dict[str, str]] = field(default_factory=list)  # LLM生成的发现列表
    rating: float = 0.0  # LLM生成的评分
    rating_explanation: str = ""  # LLM生成的评分说明

    # === 关联字段 ===
    doc_ids: List[str] = field(default_factory=list)  # 关联的文档ID列表
    chunk_ids: List[str] = field(default_factory=list)  # 关联的文档块ID列表

    # === 层次关系字段 ===
    sub_community_ids: List[str] = field(default_factory=list)  # 下级社区ID列表
    sub_community_names: List[str] = field(default_factory=list)  # 下级社区名称列表
    parent_community_ids: List[str] = field(default_factory=list)  # 上级社区ID列表

    def get_report_dict(self) -> Dict[str, Any]:
        """获取报告相关字段的字典格式"""
        return {"title": self.title, "summary": self.summary, "findings": self.findings, "rating": self.rating, "rating_explanation": self.rating_explanation}

    def set_report_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典设置报告数据"""
        self.title = data["title"]
        self.summary = data["summary"]
        self.findings = data["findings"]
        self.rating = data["rating"]
        self.rating_explanation = data["rating_explanation"]
        # 同时更新格式化文本
        self.report = self.to_formatted_string()

    def to_formatted_string(self) -> str:
        """将报告数据转换为格式化文本"""

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary", "")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation", "")

        report_sections = "\n\n".join(f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in self.findings)
        return f"# {self.title}\n\n{self.summary}\n\n{report_sections}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Community":
        """从字典创建Community实例"""
        # 处理edges格式转换
        edges = data.get("edges", [])
        if edges and isinstance(edges[0], list):
            # 转换 List[List[str]] 为 List[Tuple[str, str]]
            edges = [tuple(edge) for edge in edges]
        data["edges"] = edges

        return cls(**data)


@dataclass
class RaptorNode:
    """
    Raptor树节点数据结构

    每个节点包含title、content、层级信息、嵌入向量和溯源信息
    """

    title: str  # 节点标题
    content: str  # 节点内容
    level: int  # 层级，0为根节点（原始chunks），越高层级越抽象
    embedding: np.ndarray  # 嵌入向量
    chunks_locate: List[str] = field(default_factory=list)  # chunk位置信息
    node_id: str = ""  # 节点唯一ID
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)  # 原始 chunk ID 列表，用于直接关联
    is_split_merged: bool = False  # 是否由 split chunks 合并生成（用于溯源和检索优化）
    docnm_kwd: str = ""  # 完整的层级路径（Markdown格式），用于溯源和精确定位

    def to_dict(self) -> dict:
        """转换为字典格式，用于存储"""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "chunks_locate": self.chunks_locate,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "source_chunk_ids": self.source_chunk_ids,
            "is_split_merged": self.is_split_merged,
            "docnm_kwd": self.docnm_kwd,
            # embedding需要单独处理
        }

    @classmethod
    def from_chunk_text(cls, text: str, embedding: np.ndarray, locate: str, level: int = 0) -> "RaptorNode":
        """从原始chunk文本创建节点

        对于原始chunk（level=0），title设为空字符串，避免与content重复
        """
        from core.utils import get_uuid

        # 原始chunk的title为空，避免从content生成导致重复
        title = ""

        return cls(title=title, content=text, level=level, embedding=embedding, chunks_locate=[locate], node_id=get_uuid(), parent_id=None, child_ids=[])

    @staticmethod
    def _merge_chunks_locate(locates: List[str]) -> List[str]:
        """
        合并chunks_locate列表，处理同文件的重叠行范围

        输入格式: ["foo.py:L12-20", "foo.py:L18-25", "bar.py:L5-5", "bar.py:L9-10"]
        输出格式: ["foo.py:L12-25", "bar.py:L5, L9-10"]
        """
        if not locates:
            return []

        # 按文件分组
        file_ranges = {}
        for locate in locates:
            if ":L" not in locate:
                # 处理旧格式或异常格式
                continue

            filename, range_part = locate.split(":L", 1)
            if "-" in range_part:
                start_str, end_str = range_part.split("-", 1)
                try:
                    start, end = int(start_str), int(end_str)
                except ValueError:
                    continue
            else:
                # 单行格式，如 L5
                try:
                    start = end = int(range_part)
                except ValueError:
                    continue

            if filename not in file_ranges:
                file_ranges[filename] = []
            file_ranges[filename].append((start, end))

        # 对每个文件的行范围进行合并
        result = []
        for filename, ranges in file_ranges.items():
            # 排序并合并重叠范围
            ranges.sort()
            merged_ranges = []

            for start, end in ranges:
                if merged_ranges and start <= merged_ranges[-1][1] + 1:
                    # 重叠或相邻，合并
                    merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
                else:
                    # 不重叠，添加新范围
                    merged_ranges.append((start, end))

            # 格式化输出（单行也统一为 Lx-x 格式）
            range_strs = []
            for start, end in merged_ranges:
                range_strs.append(f"L{start}-{end}")

            result.append(f"{filename}:{', '.join(range_strs)}")

        return sorted(result)  # 按文件名排序


# ============================================================================
# GraphRAG V2: 统一图节点和边类型枚举
# ============================================================================


class NodeType:
    """
    统一图中的节点类型

    用于区分不同类型的节点：
    - ENTITY: GraphRAG 抽取的实体节点
    - CHUNK: 文档块节点（同时作为 RAPTOR level=0 叶子节点）
    - RAPTOR_SUMMARY: RAPTOR 摘要节点（仅 level > 0）
    """

    ENTITY = "entity"
    CHUNK = "chunk"
    RAPTOR_SUMMARY = "raptor"


class EdgeType:
    """
    统一图中的边类型

    用于区分不同类型的关系：
    - ENTITY_RELATION: Entity <-> Entity（实体关系）
    - ENTITY_SOURCE: Entity <-> Chunk（实体来源追溯）
    - RAPTOR_HIERARCHY: RAPTOR 层次关系（Summary<->Summary 或 Summary<->Chunk）
    """

    ENTITY_RELATION = "entity_relation"
    ENTITY_SOURCE = "entity_source"
    RAPTOR_HIERARCHY = "raptor_hierarchy"
