"""
知识库检索接口

提供获取知识库最顶层Raptor节点和GraphRAG社区的统一接口
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.storage import es_conn
from core.storage.es_conn import OrderByExpr


@dataclass
class RaptorTopNode:
    """Raptor最顶层节点数据结构"""

    title: str  # 节点标题
    content: str  # 节点内容
    level: int  # 层级
    node_id: str  # 节点唯一ID
    chunks_locate: List[str]  # 位置信息
    doc_id: str  # 所属文档ID
    docnm_kwd: str  # 文档名称

    @classmethod
    def from_es_doc(cls, doc: Dict[str, Any]) -> "RaptorTopNode":
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
class GraphRAGTopCommunity:
    """GraphRAG最顶层社区数据结构"""

    id: str  # 社区ID
    title: str  # 社区标题
    summary: str  # 社区摘要
    level: int  # 层级（0为最高层）
    sub_community_names: List[str]  # 下级社区名称
    nodes: List[str]  # 包含的节点
    weight: float  # 社区权重
    rating: float  # 社区评分
    doc_ids: List[str]  # 关联文档ID

    @classmethod
    def from_es_doc(cls, doc: Dict[str, Any]) -> "GraphRAGTopCommunity":
        """从ES文档创建GraphRAGTopCommunity实例"""
        # 从content_with_weight解析社区信息
        import json

        content_data = {}
        try:
            content_str = doc.get("content_with_weight", "{}")
            content_data = json.loads(content_str)
        except:
            pass

        # 获取report_json中的摘要信息
        report_json = content_data.get("report_json", {})

        return cls(
            id=doc.get("id", ""),
            title=report_json.get("title", "未知社区"),
            summary=report_json.get("summary", ""),
            level=doc.get("level_int", 0),
            sub_community_names=doc.get("sub_community_names_kwd", []),
            nodes=content_data.get("nodes", []),
            weight=doc.get("weight_flt", 0.0),
            rating=doc.get("rating_flt", 0.0),
            doc_ids=doc.get("doc_ids", []),
        )


@dataclass
class KnowledgeOverview:
    """知识库概览数据结构"""

    kb_id: str  # 知识库ID
    raptor_nodes: List[RaptorTopNode]  # Raptor最顶层节点
    community_nodes: List[GraphRAGTopCommunity]  # GraphRAG最顶层社区

    def get_summary(self) -> Dict[str, Any]:
        """获取概览摘要信息"""
        return {
            "kb_id": self.kb_id,
            "raptor_count": len(self.raptor_nodes),
            "community_count": len(self.community_nodes),
            "raptor_titles": [node.title for node in self.raptor_nodes],
            "community_titles": [community.title for community in self.community_nodes],
            "total_documents": len(set([node.doc_id for node in self.raptor_nodes] + [doc_id for community in self.community_nodes for doc_id in community.doc_ids])),
        }


class KnowledgeRetrieval:
    """知识库检索服务"""

    def __init__(self, es_connection: Optional[es_conn.ESConnection] = None):
        """
        初始化知识检索服务

        Args:
            es_connection: ES连接实例，如果为None则使用默认连接
        """
        self.es_conn = es_connection
        if self.es_conn is None:
            # 使用默认的ES连接
            from core import config as settings

            self.es_conn = settings.retrievaler

    async def get_top_raptor_nodes(self, kb_id: str, tenant_id: str = "", limit: int = 10) -> List[RaptorTopNode]:
        """
        获取知识库的最顶层Raptor节点

        Args:
            kb_id: 知识库ID
            tenant_id: 租户ID
            limit: 返回节点数量限制

        Returns:
            List[RaptorTopNode]: 最顶层的Raptor节点列表
        """
        try:
            # 首先查询所有Raptor节点的最高层级
            max_level_query = {"agg": {"max_level": {"max": {"field": "level_int"}}}, "query": {"bool": {"must": [{"term": {"chunk_source_kwd": "raptor"}}, {"term": {"kb_id": kb_id}}]}}, "size": 0}

            if tenant_id:
                max_level_query["query"]["bool"]["must"].append({"term": {"tenant_id": tenant_id}})

            # 执行聚合查询获取最高层级
            agg_results = await self.es_conn.search(query=max_level_query.get("query", {}), agg=max_level_query.get("agg", {}), size=0)

            max_level = 0
            if agg_results.get("aggregations", {}).get("max_level"):
                max_level = int(agg_results["aggregations"]["max_level"]["value"] or 0)

            # 查询最高层级的所有Raptor节点
            top_nodes_query = {"bool": {"must": [{"term": {"chunk_source_kwd": "raptor"}}, {"term": {"kb_id": kb_id}}, {"term": {"level_int": max_level}}]}}

            if tenant_id:
                top_nodes_query["bool"]["must"].append({"term": {"tenant_id": tenant_id}})

            # 执行查询
            results = await self.es_conn.search(query=top_nodes_query, size=limit, fields=["content_with_weight", "title_kwd", "level_int", "node_id_kwd", "source_chunks_kwd", "doc_id", "docnm_kwd"])

            # 转换为RaptorTopNode对象
            raptor_nodes = []
            for doc in results.get("hits", {}).get("hits", []):
                source = doc["_source"]
                raptor_node = RaptorTopNode.from_es_doc(source)
                raptor_nodes.append(raptor_node)

            logging.info(f"Retrieved {len(raptor_nodes)} top-level Raptor nodes for kb_id={kb_id}")
            return raptor_nodes

        except Exception as e:
            logging.error(f"Error retrieving top Raptor nodes: {e}")
            return []

    async def get_top_communities(self, kb_id: str, tenant_id: str = "", limit: int = 10) -> List[GraphRAGTopCommunity]:
        """
        获取知识库的最顶层GraphRAG社区

        当前架构使用Lazy Merge策略，不生成社区，直接返回空列表。
        保留方法签名供 get_knowledge_overview 调用。
        """
        return []

    async def get_knowledge_overview(self, kb_id: str, tenant_id: str = "", raptor_limit: int = 10, community_limit: int = 10) -> KnowledgeOverview:
        """
        获取知识库的完整概览

        Args:
            kb_id: 知识库ID
            tenant_id: 租户ID
            raptor_limit: Raptor节点数量限制
            community_limit: 社区数量限制

        Returns:
            KnowledgeOverview: 完整的知识库概览
        """
        # 并行获取Raptor节点和GraphRAG社区
        import asyncio

        raptor_nodes, communities = await asyncio.gather(self.get_top_raptor_nodes(kb_id, tenant_id, raptor_limit), self.get_top_communities(kb_id, tenant_id, community_limit), return_exceptions=True)

        # 处理异常结果
        if isinstance(raptor_nodes, Exception):
            logging.error(f"Error getting Raptor nodes: {raptor_nodes}")
            raptor_nodes = []

        if isinstance(communities, Exception):
            logging.error(f"Error getting communities: {communities}")
            communities = []

        return KnowledgeOverview(kb_id=kb_id, raptor_nodes=raptor_nodes, community_nodes=communities)


# 便捷的全局实例
_global_retrieval: Optional[KnowledgeRetrieval] = None


def get_knowledge_retrieval() -> KnowledgeRetrieval:
    """获取知识检索服务的全局实例"""
    global _global_retrieval
    if _global_retrieval is None:
        _global_retrieval = KnowledgeRetrieval()
    return _global_retrieval


# 便捷函数
async def get_kb_overview(kb_id: str, tenant_id: str = "") -> KnowledgeOverview:
    """
    获取知识库概览的便捷函数

    Args:
        kb_id: 知识库ID
        tenant_id: 租户ID

    Returns:
        KnowledgeOverview: 知识库概览
    """
    retrieval = get_knowledge_retrieval()
    return await retrieval.get_knowledge_overview(kb_id, tenant_id)


async def get_kb_summary(kb_id: str, tenant_id: str = "") -> Dict[str, Any]:
    """
    获取知识库摘要信息的便捷函数

    Args:
        kb_id: 知识库ID
        tenant_id: 租户ID

    Returns:
        Dict[str, Any]: 摘要信息
    """
    overview = await get_kb_overview(kb_id, tenant_id)
    return overview.get_summary()


def get_kb_summary_from_es(kb_id: str, tenant_id: str) -> Optional[str]:
    """
    从 ES 查询 KB Summary（由 Lazy Merge 生成的知识库级别摘要）

    这是一个同步函数，用于快速获取预生成的 KB Summary。
    如果不存在，返回 None。

    Args:
        kb_id: 知识库 ID
        tenant_id: 租户 ID (未使用，保留参数兼容性)

    Returns:
        KB Summary 内容，如果不存在则返回 None
    """
    from core.nlp.search import index_name
    from core import config

    # 索引格式为 graphrag_{kb_id}，不是 graphrag_{tenant_id}
    idxnm = index_name(kb_id)

    try:
        # 构建精确匹配查询
        condition = {
            "chunk_source_kwd": "kb_summary",
        }

        es_res = config.docStoreConn.search(
            selectFields=["content_with_weight", "doc_count_int", "source_doc_names_kwd"],
            highlightFields=[],
            condition=condition,
            matchExprs=[],
            orderBy=OrderByExpr(),
            offset=0,
            limit=1,
            indexNames=[idxnm],
            knowledgebaseIds=[kb_id],
        )

        # 提取结果
        for _, doc in config.docStoreConn.getFields(es_res, ["content_with_weight", "doc_count_int", "source_doc_names_kwd"]).items():
            content = doc.get("content_with_weight", "")
            if content:
                doc_count = doc.get("doc_count_int", 0)
                logging.debug(f"[KB Summary] Found summary for kb {kb_id}: {doc_count} documents")
                return content

        logging.debug(f"[KB Summary] No summary found for kb {kb_id}")
        return None

    except Exception as e:
        logging.warning(f"[KB Summary] Failed to query ES: {e}")
        return None
