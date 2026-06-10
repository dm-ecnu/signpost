"""子图裁切模块

提供两种裁切策略：
1. chunk_raptor_subgraph: 用于 Chunk/RAPTOR 种子
2. entity_only_subgraph: 用于 Entity 种子
"""

import logging
import networkx as nx
from typing import List, Set
from core.entities import NodeType, EdgeType

logger = logging.getLogger(__name__)


def extract_raptor_path_nodes(
    graph: nx.Graph,
    seed_nodes: List[str],
) -> Set[str]:
    """
    提取需要保留的 RAPTOR 节点集合

    逻辑：
    - CHUNK 种子：不需要特殊处理（所有 CHUNK 都会被保留）
    - RAPTOR 种子：向下遍历，收集所有可达的 RAPTOR 节点

    Args:
        graph: 完整的统一图
        seed_nodes: 种子节点列表（CHUNK 或 RAPTOR_SUMMARY）

    Returns:
        需要保留的 RAPTOR 节点集合
    """
    path_nodes: Set[str] = set()

    for seed in seed_nodes:
        if seed not in graph.nodes():
            continue

        node_type = graph.nodes[seed].get("node_type")

        if node_type == NodeType.RAPTOR_SUMMARY:
            # RAPTOR 种子：加入自身，并向下遍历收集所有 RAPTOR 后代
            path_nodes.add(seed)
            _collect_raptor_descendants(graph, seed, path_nodes)

        # CHUNK 种子不需要处理，所有 CHUNK 都会在 build_chunk_raptor_subgraph 中保留

    return path_nodes


def _collect_raptor_descendants(graph: nx.Graph, node: str, result: Set[str]) -> None:
    """递归收集 RAPTOR 后代节点（仅 RAPTOR_SUMMARY，不包括 CHUNK）

    CHUNK 节点不需要加入 path_nodes，因为在 build_chunk_raptor_subgraph 中
    所有 CHUNK 都会被保留。这里只需要标记哪些 RAPTOR 节点在路径上。
    """
    for neighbor in graph.neighbors(node):
        edge_data = graph.edges[node, neighbor]
        if edge_data.get("edge_type") != EdgeType.RAPTOR_HIERARCHY:
            continue

        neighbor_type = graph.nodes[neighbor].get("node_type")
        if neighbor_type == NodeType.RAPTOR_SUMMARY and neighbor not in result:
            result.add(neighbor)
            _collect_raptor_descendants(graph, neighbor, result)


def build_chunk_raptor_subgraph(
    graph: nx.Graph,
    seed_nodes: List[str],
) -> nx.Graph:
    """
    场景A：为 Chunk/RAPTOR 种子构建子图

    裁切规则：
    1. 保留种子到 Chunk 的 RAPTOR 路径
    2. 裁切路径外的所有 RAPTOR_SUMMARY 节点
    3. 保留所有 CHUNK 节点
    4. 保留所有 ENTITY 节点及其边

    Args:
        graph: 完整的统一图
        seed_nodes: 种子节点列表（CHUNK 或 RAPTOR_SUMMARY）

    Returns:
        裁切后的子图（copy）
    """
    # 1. 获取需要保留的 RAPTOR 路径节点
    raptor_path_nodes = extract_raptor_path_nodes(graph, seed_nodes)

    # 2. 收集所有需要保留的节点
    nodes_to_keep = set()

    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type")

        if node_type == NodeType.ENTITY:
            # 保留所有 Entity
            nodes_to_keep.add(node)
        elif node_type == NodeType.CHUNK:
            # 保留所有 Chunk
            nodes_to_keep.add(node)
        elif node_type == NodeType.RAPTOR_SUMMARY:
            # 只保留路径上的 RAPTOR 节点
            if node in raptor_path_nodes:
                nodes_to_keep.add(node)

    # 3. 构建子图
    subgraph = graph.subgraph(nodes_to_keep).copy()

    logger.debug(f"[Subgraph A] Original: {graph.number_of_nodes()} nodes, Subgraph: {subgraph.number_of_nodes()} nodes (RAPTOR paths: {len(raptor_path_nodes)})")

    return subgraph


def build_entity_only_subgraph(graph: nx.Graph) -> nx.Graph:
    """
    场景B：为 Entity 种子构建子图

    裁切规则：
    1. 裁切所有 RAPTOR_SUMMARY 节点
    2. 保留所有 CHUNK 节点
    3. 保留所有 ENTITY 节点及其边

    Args:
        graph: 完整的统一图

    Returns:
        裁切后的子图（copy）
    """
    nodes_to_keep = set()

    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        if node_type in (NodeType.ENTITY, NodeType.CHUNK):
            nodes_to_keep.add(node)

    subgraph = graph.subgraph(nodes_to_keep).copy()

    logger.debug(f"[Subgraph B] Original: {graph.number_of_nodes()} nodes, Subgraph: {subgraph.number_of_nodes()} nodes (removed all RAPTOR nodes)")

    return subgraph


__all__ = [
    "extract_raptor_path_nodes",
    "build_chunk_raptor_subgraph",
    "build_entity_only_subgraph",
]
