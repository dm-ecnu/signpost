"""Signpost: 多层次知识导航机制

为检索结果附加结构化的关联信息，使 Agent 能够沿着知识图谱的拓扑结构
和文档的层次结构进行多跳探索，解决单轮检索的信息孤岛问题。

核心组件：
- SignpostConfig: 构建配置
- SignpostBuilder: 导航信息构建器

Signpost 字符串格式（list[str]）：
- 位置信息: "loc:<filename>:L<start>-<end>"
- 实体邻居: "entity:<entity_name>"
- 源chunk:  "chunk:<chunk_id>"
- RAPTOR子节点: "raptor:<node_id>"
- RAPTOR父节点: "raptor_parent:<node_id>"
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import networkx as nx

from core.storage.graph_repository import GraphRepository
from core.storage.doc_store_conn import OrderByExpr


@dataclass
class SignpostConfig:
    """Signpost 构建配置"""

    # PPR 配置
    use_ppr: bool = True
    ppr_damping: float = 0.5
    ppr_threshold: float = 0.001
    ppr_seed_top_k: int = 10
    ppr_filter_community_nodes: bool = False

    # 邻居限制
    entity_neighbor_limit: int = 10
    edge_neighbor_limit: int = 5

    # 其他配置
    max_nodes: int = 80
    path_pruning_enabled: bool = True
    ego_radius: int = 1
    source_node_bonus: float = 1_000_000


class SignpostBuilder:
    """Signpost 导航信息构建器

    职责：
    1. 提供邻居实体查询、子节点查询等底层能力
    2. 支持 PPR 计算
    3. kg_retrieval.py 调用这些方法构建 signpost 字符串列表
    """

    def __init__(
        self,
        data_store,
        idxnms: list[str],
        kb_ids: list[str],
        tenant_id: str,
        filters_hook: Callable[[dict], dict] | None = None,
        config: SignpostConfig | None = None,
    ) -> None:
        self.data_store = data_store
        self.idxnms = idxnms
        self.kb_ids = kb_ids
        self.tenant_id = tenant_id
        self.filters_hook = filters_hook
        self.config = config or SignpostConfig()
        self._graph: nx.Graph | None = None

    # ------------------------------------------------------------------
    # Public API: 获取导航信息
    # ------------------------------------------------------------------

    def get_neighboring_entities(self, entity_name: str, limit: int = 10) -> list[str]:
        """获取邻居实体，优先使用图方案，ES作为fallback"""
        try:
            return self._get_neighboring_entities_from_graph(entity_name, limit)
        except Exception as e:
            logging.warning(f"Graph neighbor query failed for {entity_name}, fallback to ES: {e}")
            return self._get_neighboring_entities_from_es(entity_name, limit)

    def get_child_raptor_nodes(self, parent_node_id: str, child_level: int) -> list[str]:
        """获取RAPTOR子节点（仅返回ID列表）"""
        if not parent_node_id:
            return []
        filters = self._base_filters()
        filters["chunk_source_kwd"] = "raptor"
        filters["level_int"] = child_level
        filters["parent_node_id_kwd"] = parent_node_id

        child_nodes: list[str] = []
        es_res = self.data_store.search(["node_id_kwd"], [], filters, [], OrderByExpr(), 0, 50, self.idxnms, self.kb_ids)
        for _, node in self.data_store.getFields(es_res, ["node_id_kwd"]).items():
            node_id = node.get("node_id_kwd", "")
            if node_id:
                child_nodes.append(node_id)
        return child_nodes

    def get_raptor_nodes_info(self, node_ids: list[str]) -> dict[str, str]:
        """批量查询 RAPTOR 节点的 title

        Args:
            node_ids: RAPTOR 节点 ID 列表

        Returns:
            node_id -> title 的映射字典
        """
        if not node_ids:
            return {}

        filters = self._base_filters()
        filters["chunk_source_kwd"] = "raptor"
        filters["node_id_kwd"] = node_ids

        node_info: dict[str, str] = {}
        es_res = self.data_store.search(["node_id_kwd", "title_kwd"], [], filters, [], OrderByExpr(), 0, len(node_ids), self.idxnms, self.kb_ids)
        for _, node in self.data_store.getFields(es_res, ["node_id_kwd", "title_kwd"]).items():
            node_id = node.get("node_id_kwd", "")
            title = node.get("title_kwd", "")
            if node_id:
                node_info[node_id] = title

        return node_info

    # ------------------------------------------------------------------
    # PPR 计算
    # ------------------------------------------------------------------

    def ppr_on_subgraph(
        self,
        seed_nodes: list[str],
        subgraph: nx.Graph,
        top_k: int = 10,
        damping: float = 0.85,
    ) -> list[str]:
        """在子图上运行 PPR，返回 top-k 实体"""
        if not subgraph or not seed_nodes:
            return []

        personalization = {node: 0.0 for node in subgraph.nodes()}
        valid_seeds = [s for s in seed_nodes if s in subgraph.nodes()]

        if not valid_seeds:
            logging.warning(f"No valid seed nodes found in subgraph (seeds: {seed_nodes[:5]}...)")
            return []

        seed_weight = 1.0 / len(valid_seeds)
        for seed in valid_seeds:
            personalization[seed] = seed_weight

        try:
            ppr_scores = nx.pagerank(subgraph, alpha=damping, personalization=personalization, weight="weight")
        except Exception as e:
            logging.warning(f"PPR on subgraph failed: {e}")
            return []

        from core.entities import NodeType

        entity_scores = []
        for node, score in ppr_scores.items():
            if node in valid_seeds:
                continue
            node_type = subgraph.nodes[node].get("node_type")
            if node_type == NodeType.ENTITY:
                entity_scores.append((node, score))

        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in entity_scores[:top_k]]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _base_filters(self) -> dict:
        if self.filters_hook:
            return self.filters_hook({"kb_ids": self.kb_ids})
        return {"kb_ids": self.kb_ids}

    def _get_neighboring_entities_from_graph(self, entity_name: str, limit: int = 10) -> list[str]:
        """从NetworkX图中获取邻居实体"""
        graph = self._get_graph()
        if graph is None or not graph.has_node(entity_name):
            return []

        neighbors = list(graph.neighbors(entity_name))
        neighbors_with_score = []
        for neighbor in neighbors:
            score = 0.0
            degree = graph.degree(neighbor)
            score += 0.4 * degree
            pagerank = graph.nodes[neighbor].get("pagerank", 0.0)
            score += 0.6 * pagerank
            neighbors_with_score.append((neighbor, score))

        neighbors_with_score.sort(key=lambda x: x[1], reverse=True)
        return [neighbor for neighbor, _ in neighbors_with_score[:limit]]

    def _get_neighboring_entities_from_es(self, entity_name: str, limit: int = 10) -> list[str]:
        """从ES查询邻居实体（fallback）"""
        if not entity_name:
            return []
        filters = self._base_filters()
        filters["knowledge_graph_kwd"] = "relation"
        filters["chunk_source_kwd"] = "graphrag"

        from_filters = deepcopy(filters)
        from_filters["from_entity_kwd"] = entity_name
        to_filters = deepcopy(filters)
        to_filters["to_entity_kwd"] = entity_name

        neighbors: list[str] = []

        es_res = self.data_store.search(["to_entity_kwd"], [], from_filters, [], OrderByExpr(), 0, limit, self.idxnms, self.kb_ids)
        for _, edge in self.data_store.getFields(es_res, ["to_entity_kwd"]).items():
            neighbor = edge.get("to_entity_kwd", "")
            if neighbor:
                neighbors.append(neighbor)

        es_res = self.data_store.search(["from_entity_kwd"], [], to_filters, [], OrderByExpr(), 0, limit, self.idxnms, self.kb_ids)
        for _, edge in self.data_store.getFields(es_res, ["from_entity_kwd"]).items():
            neighbor = edge.get("from_entity_kwd", "")
            if neighbor:
                neighbors.append(neighbor)

        return list(dict.fromkeys(neighbors))[:limit]

    def _get_graph(self) -> nx.Graph | None:
        """获取或加载NetworkX图"""
        if self._graph is not None:
            return self._graph
        if len(self.kb_ids) != 1:
            return None
        kb_id = self.kb_ids[0]
        graph = GraphRepository.instance().load_graph(self.tenant_id, kb_id)
        self._graph = graph
        return graph


__all__ = [
    "SignpostConfig",
    "SignpostBuilder",
]
