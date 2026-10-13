"""异常层级。

设计原则：所有对外抛出的异常都继承 :class:`AgentMemoryError`，
调用方可以用一个 ``except`` 兜住整个记忆子系统，不让它把 Agent 主循环打挂。
"""

from __future__ import annotations

__all__ = [
    "AgentMemoryError",
    "MemoryNotFoundError",
    "MemoryValidationError",
    "EmbeddingDimensionError",
    "BudgetExceededError",
    "StoreError",
    "PolicyError",
]


class AgentMemoryError(Exception):
    """记忆子系统所有异常的基类。"""


class MemoryNotFoundError(AgentMemoryError):
    """按 id 找不到记忆条目。"""

    def __init__(self, memory_id: str) -> None:
        super().__init__(f"memory not found: {memory_id}")
        self.memory_id = memory_id


class MemoryValidationError(AgentMemoryError):
    """记忆条目的字段不合法（空文本、越界权重等）。"""


class EmbeddingDimensionError(AgentMemoryError):
    """向量维度与 embedder 声明的维度不一致。

    这是"换过一版 embedder 但没重建索引"这类事故的哨兵。
    """


class BudgetExceededError(AgentMemoryError):
    """单条内容就超过了 token 预算，无法裁剪。"""


class StoreError(AgentMemoryError):
    """持久层错误（schema 不兼容、写入冲突等）。"""


class PolicyError(AgentMemoryError):
    """写入策略配置非法。"""
