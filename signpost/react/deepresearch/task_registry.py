"""全局运行任务注册表

用于支持停止正在运行的研究任务。
提供线程安全的任务注册、查询和注销功能。
"""

import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .agent import ReActAgent


class TaskRegistry:
    """全局运行任务注册表（线程安全）

    职责：
    - 注册正在运行的 Agent 实例（含租户信息）
    - 提供通过 trace_id 查找 Agent 的接口（支持租户校验）
    - 支持取消操作

    使用示例：
        >>> from deepresearch.task_registry import task_registry
        >>> task_registry.register(trace_id, supervisor, tenant_id)
        >>> agent = task_registry.get(trace_id, tenant_id=tenant_id)
        >>> if agent:
        >>>     agent.cancel()
        >>> task_registry.unregister(trace_id)
    """

    def __init__(self):
        # 修改：存储 (agent, tenant_id) 元组
        self._tasks: Dict[str, Tuple["ReActAgent", str]] = {}
        self._lock = threading.Lock()

    def register(self, trace_id: str, agent: "ReActAgent", tenant_id: str) -> None:
        """注册运行中的任务（新增 tenant_id 参数）

        Args:
            trace_id: 任务的追踪ID
            agent: ReActAgent 实例（Supervisor 或 Researcher）
            tenant_id: 租户ID（用于权限校验）
        """
        with self._lock:
            self._tasks[trace_id] = (agent, tenant_id)

    def unregister(self, trace_id: str) -> None:
        """注销任务（任务完成或失败时调用）

        Args:
            trace_id: 任务的追踪ID
        """
        with self._lock:
            self._tasks.pop(trace_id, None)

    def get(self, trace_id: str, tenant_id: Optional[str] = None) -> Optional["ReActAgent"]:
        """获取正在运行的任务（支持可选租户校验）

        Args:
            trace_id: 任务的追踪ID
            tenant_id: 可选的租户ID，如果提供则校验租户权限

        Returns:
            ReActAgent 实例，如果任务不存在或租户不匹配则返回 None

        示例：
            # 不校验租户（向后兼容）
            agent = task_registry.get(trace_id)

            # 校验租户（推荐）
            agent = task_registry.get(trace_id, tenant_id=user_tenant_id)
            if not agent:
                # 任务不存在或无权限
                pass
        """
        with self._lock:
            task_info = self._tasks.get(trace_id)
            if not task_info:
                return None

            agent, task_tenant_id = task_info

            # 如果提供了 tenant_id，校验权限
            if tenant_id is not None and task_tenant_id != tenant_id:
                return None  # 租户不匹配，拒绝访问

            return agent

    def list_active_tasks(self) -> list[str]:
        """列出所有活跃任务的 trace_id

        Returns:
            活跃任务的 trace_id 列表
        """
        with self._lock:
            return list(self._tasks.keys())


# 全局单例
task_registry = TaskRegistry()
