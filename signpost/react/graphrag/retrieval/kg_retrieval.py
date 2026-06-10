#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from typing import Any, Literal
from dataclasses import dataclass, field

from core.utils import get_float, num_tokens_from_string
from core.storage.doc_store_conn import OrderByExpr
from core.nlp.search import Dealer, index_name
from .signpost import SignpostBuilder, SignpostConfig

from core.db import LLMType
from core.db.services.knowledgebase_service import KnowledgebaseService
from core.db.services.llm_service import TenantLLMService
from graphrag.utils import get_uuid
from core.llm.embedding_model import Base as EmbeddingModelBase
import pandas as pd

logger = logging.getLogger(__name__)


# 检索类型定义（使用 Literal 替代 Enum，解决 JSON 序列化问题）
RetrievalType = Literal["original_chunk", "graphrag_entity", "graphrag_edge", "raptor_node"]


# ============================================================
# 实例级 Signpost - 属于单个检索结果
# ============================================================


@dataclass
class InstanceSignpost:
    """实例级 Signpost - 属于单个检索结果的导航信息

    设计原则：
    - 所有字段可选，不同检索类型使用不同子集
    - 结构化数据，避免字符串解析
    - 字段分为三类：图导航、层次导航、位置信息

    字段使用矩阵：
    ┌──────────────────────┬───────┬────────┬────────┬────────┐
    │ 字段                 │ Chunk │ Entity │ Edge   │ RAPTOR │
    ├──────────────────────┼───────┼────────┼────────┼────────┤
    │ neighboring_entities │   -   │   Y    │   Y    │   -    │
    │ source_chunk_ids     │   -   │   Y    │   Y    │   -    │
    │ parent_node_id       │   -   │   -    │   -    │   Y    │
    │ parent_node_title    │   -   │   -    │   -    │   Y    │
    │ child_node_ids       │   -   │   -    │   -    │   Y    │
    │ child_node_titles    │   -   │   -    │   -    │   Y    │
    │ source_locates       │   -   │   -    │   -    │   Y    │
    │ file_name            │   Y   │   -    │   -    │   -    │
    │ start_line           │   Y   │   -    │   -    │   -    │
    │ end_line             │   Y   │   -    │   -    │   -    │
    │ total_lines          │   Y   │   -    │   -    │   -    │
    └──────────────────────┴───────┴────────┴────────┴────────┘
    """

    # 图导航（Entity/Edge 使用）
    neighboring_entities: list[str] = field(default_factory=list)
    """PPR/图邻居算法得到的相邻实体"""

    source_chunk_ids: list[str] = field(default_factory=list)
    """源 chunk ID 列表（用于溯源）"""

    # 层次导航（RAPTOR 使用）
    parent_node_id: str | None = None
    """父 RAPTOR 节点 ID"""

    parent_node_title: str | None = None
    """父 RAPTOR 节点标题"""

    child_node_ids: list[str] = field(default_factory=list)
    """子 RAPTOR 节点 ID 列表"""

    child_node_titles: list[str] = field(default_factory=list)
    """子 RAPTOR 节点标题列表"""

    source_locates: list[str] = field(default_factory=list)
    """源 chunk 定位，格式: "<filename>:L<start>-<end>" """

    file_name: str | None = None
    """文件名"""

    start_line: int | None = None
    """起始行号"""

    end_line: int | None = None
    """结束行号"""

    total_lines: int | None = None
    """文件总行数"""


@dataclass
class GraphRetrievalItem:
    """单个检索结果

    Attributes:
        type: 检索类型
        title: 标题
        content: 内容
        similarity: 相似度分数
        signpost: 实例级导航信息（结构化）
        kb_id: 知识库ID
    """

    type: RetrievalType
    title: str
    content: str
    similarity: float
    signpost: InstanceSignpost = field(default_factory=InstanceSignpost)
    kb_id: str | None = None


# ============================================================
# 组级 Signpost - 属于一组检索结果
# ============================================================


@dataclass
class GroupSignpost:
    """组级 Signpost - 属于一组检索结果的探索建议

    PPR 结果不属于任何单个检索结果，而是基于整组结果计算的探索方向。
    """

    related_entities: list[str] = field(default_factory=list)
    """PPR 算法推荐的相关实体（探索建议）"""


@dataclass
class RetrievalGroup:
    """一组相关的检索结果

    Attributes:
        items: 检索结果列表
        group_signpost: 组级导航信息（PPR 探索建议）
    """

    items: list[GraphRetrievalItem] = field(default_factory=list)
    group_signpost: GroupSignpost = field(default_factory=GroupSignpost)


@dataclass
class KGSearchResult:
    """KGSearch 统一返回结果 - 按语义分组

    分组设计：
    - text_group: 文本语义空间（chunks + raptors + PPR scene_a）
    - graph_group: 图结构空间（entities + edges + PPR scene_b）

    PPR 场景语义：
    - scene_a 在 chunk/raptor 子图上运行，发现与文本内容相关的实体
    - scene_b 在 entity-only 子图上运行，发现与图结构相关的实体
    """

    text_group: RetrievalGroup = field(default_factory=RetrievalGroup)
    """文本组：chunks + raptors + PPR(scene_a)"""

    graph_group: RetrievalGroup = field(default_factory=RetrievalGroup)
    """图谱组：entities + edges + PPR(scene_b)"""

    @property
    def all_items(self) -> list[GraphRetrievalItem]:
        """获取所有检索结果（兼容旧接口）"""
        return self.text_group.items + self.graph_group.items


@dataclass
class RaptorTopNode:
    """Raptor最顶层节点数据结构"""

    title: str  # 节点标题
    content: str  # 节点内容
    level: int  # 层级
    node_id: str  # 节点唯一ID
    chunks_locate: list[str]  # 位置信息
    doc_id: str  # 所属文档ID
    docnm_kwd: str  # 文档名称

    @classmethod
    def from_es_doc(cls, doc: dict[str, Any]) -> "RaptorTopNode":
        """从ES文档创建RaptorTopNode实例"""
        # 格式: [TITLE]{title}\n[CONTENT]{content}
        content_full = doc.get("content_with_weight", "")
        if content_full.startswith("[TITLE]"):
            content_marker = "\n[CONTENT]"
            marker_pos = content_full.find(content_marker)
            if marker_pos != -1:
                title = content_full[7:marker_pos]  # len("[TITLE]") = 7
                content = content_full[marker_pos + len(content_marker) :]
            else:
                title = ""
                content = content_full
        else:
            title = doc.get("title_kwd", "")
            content = content_full

        return cls(
            title=title,
            content=content,
            level=doc.get("level_int", 0),
            node_id=doc.get("node_id_kwd", ""),
            chunks_locate=doc.get("source_chunks_kwd", []),
            doc_id=doc.get("doc_id", ""),
            docnm_kwd=doc.get("docnm_kwd", ""),
        )


@dataclass
class KnowledgeOverview:
    """知识库概览数据结构"""

    kb_id: str  # 知识库ID
    raptor_nodes: list[RaptorTopNode]  # Raptor最顶层节点

    def get_summary(self) -> dict[str, Any]:
        """获取概览摘要信息"""
        return {
            "kb_id": self.kb_id,
            "raptor_count": len(self.raptor_nodes),
            "raptor_titles": [node.title for node in self.raptor_nodes],
            "total_documents": len(set([node.doc_id for node in self.raptor_nodes])),
        }


class KGSearch(Dealer):
    def __init__(self, dataStore):
        super().__init__(dataStore)
        self._emb_cache: dict[tuple[str, str], EmbeddingModelBase] = {}
        self._cache_lock = threading.Lock()  # 修复 P2: 添加线程锁保护缓存

    def process(
        self,
        query: str,
        tenant_id: str,
        kb_ids: list[str],
        emb_mdl=None,
        similarity_threshold: float = 0.3,
        chunk_raptor_topn: int = 5,
        entity_topn: int = 5,
        edge_topn: int = 3,
        ppr_top_k: int = 10,
        **kwargs,
    ) -> KGSearchResult:
        """
        检索流程（支持子图 PPR）

        流程：
        1. 阶段1：纯 ES 查询（返回原始 doc）
        2. 阶段2：创建 SignpostBuilder
        3. 阶段3：构建 InstanceSignpost
        4. 阶段4：PPR 计算 -> GroupSignpost
        5. 阶段5：包装分组结果

        Args:
            tenant_id: 租户ID（单个字符串）
            emb_mdl: Embedding模型，如为None则根据kb_ids自动构造
            similarity_threshold: 相似度阈值
            chunk_raptor_topn: 原始 chunk + RAPTOR 节点的合并检索数量
            entity_topn: 实体检索数量
            edge_topn: 关系检索数量
            ppr_top_k: PPR 返回的实体数量

        Returns:
            KGSearchResult: 分组检索结果
                - text_group: chunks + raptors + PPR(scene_a)
                - graph_group: entities + edges + PPR(scene_b)
        """
        # 自动构造embedding模型
        if emb_mdl is None:
            emb_mdl = self._get_or_create_embedding(kb_ids)

        filters = self.get_filters({"kb_ids": kb_ids})
        idxnms = [index_name(kb_id) for kb_id in kb_ids]
        query_vector = self.get_vector(query, emb_mdl, 1024, similarity_threshold)

        # ===== 阶段1：纯 ES 查询 =====
        chunk_docs, raptor_docs = self._query_chunks_and_raptor(filters, idxnms, kb_ids, query_vector, chunk_raptor_topn, similarity_threshold)
        entity_docs = self._query_graphrag_entities(filters, idxnms, kb_ids, query_vector, entity_topn, similarity_threshold)
        edge_docs = self._query_graphrag_edges(filters, idxnms, kb_ids, query_vector, edge_topn, similarity_threshold)

        # ===== 阶段2：创建 SignpostBuilder =====
        signpost_builder = self._create_signpost_builder(kb_ids, tenant_id, filters, kwargs)

        # ===== 阶段3：构建 InstanceSignpost =====
        chunk_signposts = self._build_chunk_signposts(chunk_docs, signpost_builder)
        raptor_signposts = self._build_raptor_signposts(raptor_docs, signpost_builder)
        entity_signposts = self._build_entity_signposts(entity_docs, signpost_builder)
        edge_signposts = self._build_edge_signposts(edge_docs, signpost_builder)

        # ===== 阶段4：PPR 计算 -> GroupSignpost =====
        text_group_signpost, graph_group_signpost = self._calculate_ppr(chunk_docs, raptor_docs, entity_docs, edge_docs, signpost_builder, ppr_top_k)

        # ===== 阶段5：包装分组结果 =====
        text_group, graph_group = self._wrap_docs_to_groups(
            chunk_docs, raptor_docs, entity_docs, edge_docs, chunk_signposts, raptor_signposts, entity_signposts, edge_signposts, text_group_signpost, graph_group_signpost, kb_ids
        )

        return KGSearchResult(text_group=text_group, graph_group=graph_group)

    # ===== 阶段1：纯 ES 查询方法 =====

    def _query_chunks_and_raptor(
        self,
        filters: dict,
        idxnms: list,
        kb_ids: list,
        query_vector,
        max_results: int,
        similarity_threshold: float = 0.3,
    ) -> tuple[list[dict], list[dict]]:
        """ES 查询：返回原始 doc 列表（chunk + raptor）

        通过 chunk_source_kwd IN ["original", "raptor"] 实现单次 ES 查询，
        按相似度排序后根据 chunk_source_kwd 分离为两个列表。

        Returns:
            (chunk_docs, raptor_docs): 两个原始 doc 列表
        """
        combined_filters = deepcopy(filters)
        combined_filters["chunk_source_kwd"] = ["original", "raptor"]

        select_fields = [
            # 共用字段
            "content_with_weight",
            "chunk_source_kwd",
            # original chunk 字段
            "docnm_kwd",
            "doc_id",
            "chunk_id",
            "page_num_int",
            "start_line_int",
            "end_line_int",
            "line_count_int",
            "chunk_method_kwd",
            "total_lines_int",
            "file_name",
            # raptor 字段
            "title_kwd",
            "level_int",
            "node_id_kwd",
            "source_chunks_kwd",
            "parent_node_id_kwd",
            "child_node_ids_kwd",
        ]

        es_res = self.dataStore.search(
            select_fields,
            [],
            combined_filters,
            [query_vector],
            OrderByExpr(),
            0,
            max_results,
            idxnms,
            kb_ids,
        )

        chunk_docs: list[dict] = []
        raptor_docs: list[dict] = []

        for _, doc in self.dataStore.getFields(es_res, select_fields + ["_score"]).items():
            score = get_float(doc.get("_score", 0))
            if score < similarity_threshold:
                continue

            source_type = doc.get("chunk_source_kwd", "")
            if source_type == "original":
                chunk_docs.append(doc)
            elif source_type == "raptor":
                raptor_docs.append(doc)

        return chunk_docs, raptor_docs

    def _query_graphrag_entities(
        self,
        filters: dict,
        idxnms: list,
        kb_ids: list,
        query_vector,
        max_results: int,
        similarity_threshold: float = 0.3,
    ) -> list[dict]:
        """ES 查询：返回原始 entity doc 列表"""
        entity_filters = deepcopy(filters)
        entity_filters["knowledge_graph_kwd"] = "entity"
        entity_filters["chunk_source_kwd"] = "graphrag"

        select_fields = ["content_with_weight", "entity_kwd", "entity_type_kwd", "communities_kwd", "chunk_ids", "doc_ids"]

        es_res = self.dataStore.search(select_fields, [], entity_filters, [query_vector], OrderByExpr(), 0, max_results, idxnms, kb_ids)

        return [doc for _, doc in self.dataStore.getFields(es_res, select_fields + ["_score"]).items() if get_float(doc.get("_score", 0)) >= similarity_threshold]

    def _query_graphrag_edges(
        self,
        filters: dict,
        idxnms: list,
        kb_ids: list,
        query_vector,
        max_results: int,
        similarity_threshold: float = 0.3,
    ) -> list[dict]:
        """ES 查询：返回原始 edge doc 列表"""
        edge_filters = deepcopy(filters)
        edge_filters["knowledge_graph_kwd"] = "relation"
        edge_filters["chunk_source_kwd"] = "graphrag"

        select_fields = ["content_with_weight", "from_entity_kwd", "to_entity_kwd", "relation_type_kwd", "chunk_ids", "doc_ids"]

        es_res = self.dataStore.search(select_fields, [], edge_filters, [query_vector], OrderByExpr(), 0, max_results, idxnms, kb_ids)

        return [doc for _, doc in self.dataStore.getFields(es_res, select_fields + ["_score"]).items() if get_float(doc.get("_score", 0)) >= similarity_threshold]

    # ===== 阶段2：SignpostBuilder 创建 =====

    def _create_signpost_builder(
        self,
        kb_ids: list,
        tenant_id: str,
        filters: dict,
        kwargs: dict,
    ) -> SignpostBuilder:
        """创建 SignpostBuilder"""
        config = kwargs.get("signpost_config", SignpostConfig())
        idxnms = [index_name(kb_id) for kb_id in kb_ids]

        return SignpostBuilder(
            self.dataStore,
            idxnms,
            kb_ids,
            tenant_id,
            filters_hook=self.get_filters,
            config=config,
        )

    # ===== 阶段3：InstanceSignpost 构建方法 =====

    def _build_chunk_signposts(self, docs: list[dict], builder: SignpostBuilder) -> list[InstanceSignpost]:
        """为 chunk docs 构建 InstanceSignpost 列表

        填充字段：file_name, start_line, end_line, total_lines, parent_node_id, parent_node_title
        """
        # 第一遍：收集所有需要查询的 parent_node_ids
        all_parent_ids_to_query: set[str] = set()

        for doc in docs:
            parent_id = doc.get("parent_node_id_kwd")
            if parent_id:
                all_parent_ids_to_query.add(parent_id)

        # 批量查询所有父节点的 title
        node_id_to_title = builder.get_raptor_nodes_info(list(all_parent_ids_to_query))

        # 第二遍：构建 signposts
        signposts: list[InstanceSignpost] = []
        for doc in docs:
            # 使用 file_name 字段而不是 docnm_kwd（docnm_kwd 是层级路径，不是文件名）
            file_name = doc.get("file_name") or ""

            # 父节点（RAPTOR）
            parent_id = doc.get("parent_node_id_kwd")
            parent_title = node_id_to_title.get(parent_id, "") if parent_id else None

            signposts.append(
                InstanceSignpost(
                    file_name=file_name if file_name else None,
                    start_line=doc.get("start_line_int"),
                    end_line=doc.get("end_line_int"),
                    total_lines=doc.get("total_lines_int"),
                    parent_node_id=parent_id if parent_id else None,
                    parent_node_title=parent_title,
                )
            )
        return signposts

    def _build_raptor_signposts(self, docs: list[dict], builder: SignpostBuilder) -> list[InstanceSignpost]:
        """为 raptor docs 构建 InstanceSignpost 列表

        填充字段：source_locates, child_node_ids, child_node_titles, parent_node_id, parent_node_title

        注意：source_locates 使用 RaptorNode._merge_chunks_locate 处理重叠行范围
        """
        from core.entities import RaptorNode

        # 第一遍：收集所有需要查询的 node_ids
        all_node_ids_to_query: set[str] = set()

        for doc in docs:
            level = doc.get("level_int", 0)
            node_id = doc.get("node_id_kwd", "")
            child_ids = doc.get("child_node_ids_kwd", []) or []
            if not child_ids and level > 1:
                child_ids = builder.get_child_raptor_nodes(node_id, level - 1)

            parent_id = doc.get("parent_node_id_kwd")

            # 收集 child_ids
            all_node_ids_to_query.update(child_ids)
            # 收集 parent_id
            if parent_id:
                all_node_ids_to_query.add(parent_id)

        # 批量查询所有节点的 title
        node_id_to_title = builder.get_raptor_nodes_info(list(all_node_ids_to_query))

        # 第二遍：构建 signposts
        signposts: list[InstanceSignpost] = []
        for doc in docs:
            # 源定位（使用 _merge_chunks_locate 合并重叠行范围）
            source_locates_raw = doc.get("source_chunks_kwd", []) or []
            source_locates = RaptorNode._merge_chunks_locate(source_locates_raw)

            # 子节点
            level = doc.get("level_int", 0)
            node_id = doc.get("node_id_kwd", "")
            child_ids = doc.get("child_node_ids_kwd", []) or []
            if not child_ids and level > 1:
                child_ids = builder.get_child_raptor_nodes(node_id, level - 1)

            # 查询子节点的 titles
            child_titles = [node_id_to_title.get(cid, "") for cid in child_ids]

            # 父节点
            parent_id = doc.get("parent_node_id_kwd")
            parent_title = node_id_to_title.get(parent_id, "") if parent_id else None

            signposts.append(
                InstanceSignpost(
                    source_locates=source_locates,
                    child_node_ids=child_ids,
                    child_node_titles=child_titles,
                    parent_node_id=parent_id if parent_id else None,
                    parent_node_title=parent_title,
                )
            )
        return signposts

    def _build_entity_signposts(self, docs: list[dict], builder: SignpostBuilder) -> list[InstanceSignpost]:
        """为 entity docs 构建 InstanceSignpost 列表

        填充字段：neighboring_entities, source_chunk_ids

        注意：当前版本暂时返回空 signpost，待后续实现完整的 Entity 导航逻辑
        """
        # TODO: 实现 Entity signpost 构建逻辑
        # - neighboring_entities: builder.get_neighboring_entities()
        # - source_chunk_ids: doc.get("chunk_ids")
        return [InstanceSignpost() for _ in docs]

    def _build_edge_signposts(self, docs: list[dict], builder: SignpostBuilder) -> list[InstanceSignpost]:
        """为 edge docs 构建 InstanceSignpost 列表

        填充字段：neighboring_entities（两端实体的合并邻居）, source_chunk_ids

        注意：当前版本暂时返回空 signpost，待后续实现完整的 Edge 导航逻辑
        """
        # TODO: 实现 Edge signpost 构建逻辑
        # - neighboring_entities: 合并 from_entity 和 to_entity 的邻居
        # - source_chunk_ids: doc.get("chunk_ids")
        return [InstanceSignpost() for _ in docs]

    # ===== 阶段4：PPR 计算方法 =====

    def _calculate_ppr(
        self,
        chunk_docs: list[dict],
        raptor_docs: list[dict],
        entity_docs: list[dict],
        edge_docs: list[dict],
        signpost_builder: SignpostBuilder,
        ppr_top_k: int,
    ) -> tuple[GroupSignpost, GroupSignpost]:
        """计算 PPR 结果，返回两个 GroupSignpost

        Returns:
            tuple[GroupSignpost, GroupSignpost]:
                - text_group_signpost: 基于 chunk/raptor 种子的 PPR 结果
                - graph_group_signpost: 基于 entity/edge 种子的 PPR 结果
        """
        graph = signpost_builder._get_graph()

        # 修复 HIGH-3: 空图检查逻辑（检查 None 或节点数为 0）
        if graph is None or graph.number_of_nodes() == 0:
            logger.warning("[KGSearch] No unified graph available (graph is None or empty), skipping PPR")
            return GroupSignpost(), GroupSignpost()

        # 场景A：Chunk/RAPTOR 的 PPR -> text_group
        from .subgraph import build_chunk_raptor_subgraph

        chunk_seeds = [doc.get("chunk_id") for doc in chunk_docs if doc.get("chunk_id")]
        raptor_seeds = [doc.get("node_id_kwd") for doc in raptor_docs if doc.get("node_id_kwd")]
        scene_a_seeds = chunk_seeds + raptor_seeds

        ppr_entities_a: list[str] = []
        if scene_a_seeds:
            subgraph_a = build_chunk_raptor_subgraph(graph, scene_a_seeds)
            ppr_entities_a = signpost_builder.ppr_on_subgraph(scene_a_seeds, subgraph_a, top_k=ppr_top_k)

        # 场景B：Entity/Edge 的 PPR -> graph_group
        from .subgraph import build_entity_only_subgraph

        entity_seeds = [doc.get("entity_kwd") for doc in entity_docs if doc.get("entity_kwd")]
        edge_seeds = []
        for doc in edge_docs:
            from_entity = doc.get("from_entity_kwd")
            to_entity = doc.get("to_entity_kwd")
            if from_entity:
                edge_seeds.append(from_entity)
            if to_entity:
                edge_seeds.append(to_entity)
        scene_b_seeds = list(set(entity_seeds + edge_seeds))

        ppr_entities_b: list[str] = []
        if scene_b_seeds:
            subgraph_b = build_entity_only_subgraph(graph)
            ppr_entities_b = signpost_builder.ppr_on_subgraph(scene_b_seeds, subgraph_b, top_k=ppr_top_k)

        return (
            GroupSignpost(related_entities=ppr_entities_a),
            GroupSignpost(related_entities=ppr_entities_b),
        )

    # ===== 阶段5：包装 GraphRetrievalItem 方法 =====

    def _wrap_docs_to_groups(
        self,
        chunk_docs: list[dict],
        raptor_docs: list[dict],
        entity_docs: list[dict],
        edge_docs: list[dict],
        chunk_signposts: list[InstanceSignpost],
        raptor_signposts: list[InstanceSignpost],
        entity_signposts: list[InstanceSignpost],
        edge_signposts: list[InstanceSignpost],
        text_group_signpost: GroupSignpost,
        graph_group_signpost: GroupSignpost,
        kb_ids: list,
    ) -> tuple[RetrievalGroup, RetrievalGroup]:
        """将所有 docs 包装为分组的 RetrievalGroup

        Returns:
            tuple[RetrievalGroup, RetrievalGroup]:
                - text_group: chunks + raptors + PPR(scene_a)
                - graph_group: entities + edges + PPR(scene_b)
        """
        kb_id = kb_ids[0] if len(kb_ids) == 1 else ""

        # Text group: Chunks + Raptors
        text_items: list[GraphRetrievalItem] = []
        for doc, signpost in zip(chunk_docs, chunk_signposts):
            text_items.append(self._wrap_chunk_doc(doc, signpost, kb_id))
        for doc, signpost in zip(raptor_docs, raptor_signposts):
            text_items.append(self._wrap_raptor_doc(doc, signpost, kb_id))

        # Graph group: Entities + Edges
        graph_items: list[GraphRetrievalItem] = []
        for doc, signpost in zip(entity_docs, entity_signposts):
            graph_items.append(self._wrap_entity_doc(doc, signpost, kb_id))
        for doc, signpost in zip(edge_docs, edge_signposts):
            graph_items.append(self._wrap_edge_doc(doc, signpost, kb_id))

        return (
            RetrievalGroup(items=text_items, group_signpost=text_group_signpost),
            RetrievalGroup(items=graph_items, group_signpost=graph_group_signpost),
        )

    def _wrap_chunk_doc(self, doc: dict, signpost: InstanceSignpost, kb_id: str) -> GraphRetrievalItem:
        """将单个 chunk doc 包装为 GraphRetrievalItem"""
        return GraphRetrievalItem(
            type="original_chunk",
            title="",
            content=doc.get("content_with_weight", ""),
            signpost=signpost,
            similarity=get_float(doc.get("_score", 0)),
            kb_id=kb_id,
        )

    def _wrap_raptor_doc(self, doc: dict, signpost: InstanceSignpost, kb_id: str) -> GraphRetrievalItem:
        """将单个 raptor doc 包装为 GraphRetrievalItem"""
        level = doc.get("level_int", 0)
        title = doc.get("title_kwd", "Unknown")
        content = self._parse_raptor_content(doc.get("content_with_weight", ""))

        return GraphRetrievalItem(
            type="raptor_node",
            title=f"RAPTOR Level {level}: {title}",
            content=content,
            signpost=signpost,
            similarity=get_float(doc.get("_score", 0)),
            kb_id=kb_id,
        )

    def _wrap_entity_doc(self, doc: dict, signpost: InstanceSignpost, kb_id: str) -> GraphRetrievalItem:
        """将单个 entity doc 包装为 GraphRetrievalItem"""
        return GraphRetrievalItem(
            type="graphrag_entity",
            title=f"Entity: {doc.get('entity_kwd', 'Unknown')}",
            content=self._parse_entity_content(doc.get("content_with_weight", "")),
            signpost=signpost,
            similarity=get_float(doc.get("_score", 0)),
            kb_id=kb_id,
        )

    def _wrap_edge_doc(self, doc: dict, signpost: InstanceSignpost, kb_id: str) -> GraphRetrievalItem:
        """将单个 edge doc 包装为 GraphRetrievalItem"""
        from_entity = doc.get("from_entity_kwd", "Unknown")
        to_entity = doc.get("to_entity_kwd", "Unknown")
        relation_type = doc.get("relation_type_kwd", "related_to")

        return GraphRetrievalItem(
            type="graphrag_edge",
            title=f"Relation: {from_entity} -> {to_entity} ({relation_type})",
            content=self._parse_edge_content(doc.get("content_with_weight", "")),
            signpost=signpost,
            similarity=get_float(doc.get("_score", 0)),
            kb_id=kb_id,
        )

    # ===== 内容解析方法 =====

    def _parse_entity_content(self, content_str: str) -> str:
        """解析实体内容"""
        try:
            content_data = json.loads(content_str)
            return content_data.get("description", content_str)
        except (json.JSONDecodeError, TypeError, KeyError):
            return content_str

    def _parse_edge_content(self, content_str: str) -> str:
        """解析边内容"""
        try:
            content_data = json.loads(content_str)
            return content_data.get("description", content_str)
        except (json.JSONDecodeError, TypeError, KeyError):
            return content_str

    def _parse_raptor_content(self, content_str: str) -> str:
        """解析RAPTOR内容，返回纯内容部分（不含title标签）"""
        # 格式：[TITLE]{title}\n[CONTENT]{content}
        if content_str.startswith("[TITLE]"):
            content_marker = "\n[CONTENT]"
            marker_pos = content_str.find(content_marker)
            if marker_pos != -1:
                return content_str[marker_pos + len(content_marker) :]
        return content_str

    # ===== 知识库概览功能（供旧版 deepresearch/ 使用）=====
    # 注意：新版 deepresearch_v2/ 已改用 get_kb_summary_from_es() 直接查询预生成的 KB Summary
    # 这些方法仅供旧版 deepresearch/tools.py 中的 KnowledgeOverviewTool 使用

    def get_top_raptor_nodes(self, kb_id: str, tenant_id: str = "", limit: int = 10) -> list[RaptorTopNode]:
        """
        获取知识库的最顶层Raptor节点

        Args:
            kb_id: 知识库ID
            tenant_id: 租户ID
            limit: 返回节点数量限制

        Returns:
            list[RaptorTopNode]: 最顶层的Raptor节点列表
        """
        try:
            from core.nlp.search import index_name

            filters = self.get_filters({"kb_ids": [kb_id]})
            filters["chunk_source_kwd"] = "raptor"

            # Use kb_id for index name (unified graphrag_{kb_id} index)
            idxnms = [index_name(kb_id)]

            # 首先查询所有Raptor节点的最高层级
            agg_query = {"aggregations": {"max_level": {"max": {"field": "level_int"}}}, "size": 0}

            # 执行聚合查询获取最高层级
            es_res = self.dataStore.search([], [], filters, [], agg_query, 0, 0, idxnms, [kb_id])

            max_level = 0
            if "aggregations" in es_res and "max_level" in es_res["aggregations"]:
                max_level = int(es_res["aggregations"]["max_level"].get("value", 0))

            # 查询最高层级的所有Raptor节点
            filters["level_int"] = max_level

            es_res = self.dataStore.search(["content_with_weight", "title_kwd", "level_int", "node_id_kwd", "source_chunks_kwd", "doc_id", "docnm_kwd"], [], filters, [], {}, 0, limit, idxnms, [kb_id])

            # 转换为RaptorTopNode对象
            raptor_nodes = []
            for _, doc in self.dataStore.getFields(es_res, ["content_with_weight", "title_kwd", "level_int", "node_id_kwd", "source_chunks_kwd", "doc_id", "docnm_kwd"]).items():
                raptor_node = RaptorTopNode.from_es_doc(doc)
                raptor_nodes.append(raptor_node)

            return raptor_nodes

        except Exception as e:
            logging.error(f"Error retrieving top Raptor nodes: {e}")
            return []

    def get_knowledge_overview(self, kb_id: str, tenant_id: str = "", raptor_limit: int = 10) -> KnowledgeOverview:
        """
        获取知识库的完整概览

        Args:
            kb_id: 知识库ID
            tenant_id: 租户ID
            raptor_limit: Raptor节点数量限制

        Returns:
            KnowledgeOverview: 完整的知识库概览
        """
        try:
            # 获取Raptor节点
            raptor_nodes = self.get_top_raptor_nodes(kb_id, tenant_id, raptor_limit)

            return KnowledgeOverview(kb_id=kb_id, raptor_nodes=raptor_nodes)

        except Exception as e:
            logging.error(f"Error getting knowledge overview: {e}")
            return KnowledgeOverview(kb_id=kb_id, raptor_nodes=[])

    # ===== 从 hybrid_search.py 迁移的方法 =====

    def _get_or_create_embedding(self, kb_ids: list[str]) -> EmbeddingModelBase:
        """基于 kb 配置获取或创建 Embedding 模型，并在本实例内缓存。

        约束：强制所有 kb 的 tenant_id 和"底模名"（去掉供应商后缀）一致。
        """
        if not kb_ids:
            raise ValueError("kb_ids 不能为空")

        tenants: set[str] = set()
        base_models: set[str] = set()
        embd_ids: set[str] = set()

        # 收集配置
        for kb_id in kb_ids:
            ok, kb = KnowledgebaseService.get_by_id(kb_id)
            if not ok or not kb:
                raise LookupError(f"Knowledge base not found: {kb_id}")
            tenants.add(kb.tenant_id)
            base, _ = TenantLLMService.split_model_name_and_factory(kb.embd_id)
            base_models.add(base)
            embd_ids.add(kb.embd_id)

        if len(tenants) != 1:
            raise ValueError(f"选定的知识库属于不同租户，无法统一构造Embedding模型: {tenants}")
        if len(base_models) != 1:
            raise ValueError(f"选定的知识库使用不同底模，无法统一构造Embedding模型: {base_models}")

        tenant_id = next(iter(tenants))
        embd_id = next(iter(embd_ids))

        key = (tenant_id, embd_id)

        # 修复 P2: 双重检查锁定模式（避免重复构造）
        if key in self._emb_cache:
            return self._emb_cache[key]

        with self._cache_lock:
            # 再次检查（可能在等待锁期间已被其他线程创建）
            if key in self._emb_cache:
                return self._emb_cache[key]

            model_instance = TenantLLMService.model_instance(tenant_id, LLMType.EMBEDDING, embd_id)
            self._emb_cache[key] = model_instance
            return model_instance

    # ===== 兼容旧 API 的方法 =====

    def retrieval(
        self,
        question: str,
        tenant_id: str,
        kb_ids: list[str],
        emb_mdl=None,
        llm=None,  # 兼容旧接口，实际不使用
        max_token: int = 8196,
        ent_topn: int = 6,
        rel_topn: int = 6,
        comm_topn: int = 1,
        ent_sim_threshold: float = 0.3,
        rel_sim_threshold: float = 0.3,
        **kwargs,
    ) -> dict:
        """知识图谱检索方法（兼容旧接口）

        注意：
        - llm 参数已废弃，不再使用基于LLM的查询改写
        - 返回格式兼容旧的kg_retrieval.process() 返回的ES 文档格式
        - 内部调用新的 process() 方法，然后转换格式

        Returns:
            dict: ES 文档格式的检索结果
        """
        # 如果未提供 embedding，自动构造
        if emb_mdl is None:
            emb_mdl = self._get_or_create_embedding(kb_ids)

        # 调用新的统一检索
        kg_result: KGSearchResult = self.process(
            query=question,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            emb_mdl=emb_mdl,
            chunk_raptor_topn=max(ent_topn, rel_topn, 10),
            entity_topn=ent_topn,
            edge_topn=rel_topn,
            similarity_threshold=min(ent_sim_threshold, rel_sim_threshold),
            **kwargs,
        )

        # 格式转换：list[GraphRetrievalItem] -> 单个 ES 文档字典
        return self._convert_to_legacy_format(kg_result.all_items, max_token)

    def _convert_to_legacy_format(self, results: list[GraphRetrievalItem], max_token: int) -> dict:
        """将新格式检索结果转换为旧的 ES 文档格式

        旧格式示例（来自 kg_retrieval_backup.py:296-325）：
        {
            "id": "...",
            "doc_id": "",
            "kb_id": "...",
            "content_with_weight": "---- Entities ----\\n...",
            "docnm_kwd": "Related content in Knowledge Graph",
            "similarity": 1.0,
            ...
        }
        """
        if not results:
            # 返回空结果
            return {
                "id": get_uuid(),
                "doc_id": "",
                "kb_id": "",
                "content_with_weight": "",
                "docnm_kwd": "Related content in Knowledge Graph",
                "similarity": 0.0,
                "important_kwd": [],
                "page_num_int": [],
                "top_int": 0,
                "position_int": [],
                "create_timestamp_flt": 0.0,
                "available_int": 1,
                "doc_type_kwd": "",
                "img_id": "",
                "chunk_id": get_uuid(),
                "image_id": "",
                "vector": [],
                "positions": [],
                "vector_similarity": 0.0,
                "term_similarity": 0.0,
            }

        # 按类型分组
        entities = []
        relations = []
        raptors = []

        for item in results:
            if item.type == "graphrag_entity":
                entities.append(item)
            elif item.type == "graphrag_edge":
                relations.append(item)
            elif item.type == "raptor_node":
                raptors.append(item)

        # 构建内容（模拟旧格式的DataFrame输出）
        content_parts = []
        remaining_tokens = max_token

        # 1. 实体部分
        if entities:
            entity_rows = []
            for item in entities[:10]:  # 限制数量
                entity_rows.append({"Entity": item.title.replace("Entity: ", ""), "Score": f"{item.similarity:.2f}", "Description": item.content[:200] if item.content else ""})
                remaining_tokens -= num_tokens_from_string(str(entity_rows[-1]))
                if remaining_tokens <= 0:
                    entity_rows = entity_rows[:-1]
                    break

            if entity_rows:
                df = pd.DataFrame(entity_rows)
                content_parts.append(f"\n---- Entities ----\n{df.to_csv(index=False)}")

        # 2. 关系部分
        if relations and remaining_tokens > 0:
            relation_rows = []
            for item in relations[:10]:
                # 从 title 解析：格式 "Relation: A -> B (type)"
                title = item.title.replace("Relation: ", "")
                if " -> " in title:
                    parts = title.split(" -> ")
                    from_entity = parts[0]
                    rest = parts[1]
                    if " (" in rest:
                        to_entity = rest.split(" (")[0]
                    else:
                        to_entity = rest
                else:
                    from_entity = "Unknown"
                    to_entity = "Unknown"

                relation_rows.append(
                    {
                        "From Entity": from_entity,
                        "To Entity": to_entity,
                        "Score": f"{item.similarity:.2f}",
                        "Description": item.content[:200] if item.content else "",
                    }
                )
                remaining_tokens -= num_tokens_from_string(str(relation_rows[-1]))
                if remaining_tokens <= 0:
                    relation_rows = relation_rows[:-1]
                    break

            if relation_rows:
                df = pd.DataFrame(relation_rows)
                content_parts.append(f"\n---- Relations ----\n{df.to_csv(index=False)}")

        # 3. RAPTOR 摘要部分（如果有）
        if raptors and remaining_tokens > 0:
            raptor_parts = []
            for i, item in enumerate(raptors[:3]):
                text = f"# {i + 1}. {item.title}\n{item.content[:500] if item.content else ''}"
                raptor_parts.append(text)
                remaining_tokens -= num_tokens_from_string(text)
                if remaining_tokens <= 0:
                    raptor_parts = raptor_parts[:-1]
                    break

            if raptor_parts:
                content_parts.append("\n---- CHAPTER Summaries ----\n" + "\n\n".join(raptor_parts))

        # 合并所有内容
        final_content = "".join(content_parts)

        # 构建 ES 文档格式
        kb_id = results[0].kb_id if results else ""

        return {
            "id": get_uuid(),
            "doc_id": "",
            "kb_id": kb_id,
            "content_with_weight": final_content,
            "important_kwd": [],
            "page_num_int": [],
            "top_int": 0,
            "position_int": [],
            "create_timestamp_flt": 0.0,
            "available_int": 1,
            "doc_type_kwd": "",
            "img_id": "",
            "docnm_kwd": "Related content in Knowledge Graph",
            "chunk_id": get_uuid(),
            "image_id": "",
            "vector": [],
            "positions": [],
            "similarity": 1.0,
            "vector_similarity": 1.0,
            "term_similarity": 0.0,
        }
