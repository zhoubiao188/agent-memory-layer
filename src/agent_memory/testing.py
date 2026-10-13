"""测试与示例用的小工具。生产代码不依赖本模块。"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from .budget import ContextAssembler
from .clock import DAY, HOUR, MINUTE, ManualClock, SystemClock  # noqa: F401  (re-export)
from .embedder import HashingEmbedder
from .extract import Turn
from .manager import MemoryManager
from .models import MemoryDraft, MemoryItem, MemoryKind, MemoryStatus, new_id
from .policy import WritePolicy
from .store import MemoryStore

__all__ = [
    "ManualClock",
    "SystemClock",
    "DAY",
    "HOUR",
    "MINUTE",
    "make_item",
    "make_manager",
    "seed_drafts",
    "sample_turns",
]


def make_item(
    text: str,
    *,
    kind: MemoryKind = MemoryKind.SEMANTIC,
    now: float | None = None,
    **overrides: Any,
) -> MemoryItem:
    """快速造一条记忆。默认时间是"现在"，方便直接和真实时钟对比。"""
    ts = time.time() if now is None else now
    base: dict[str, Any] = dict(
        kind=kind,
        text=text,
        created_at=ts,
        updated_at=ts,
    )
    base.update(overrides)
    if "id" not in base:
        base["id"] = new_id()
    return MemoryItem(**base)


def make_manager(
    *,
    path: str = ":memory:",
    clock: ManualClock | None = None,
    embedder: HashingEmbedder | None = None,
    policy: WritePolicy | None = None,
    assembler: ContextAssembler | None = None,
    **kwargs: Any,
) -> MemoryManager:
    """造一个全内存、时钟可控的 manager，供测试使用。"""
    return MemoryManager(
        store=MemoryStore(path),
        embedder=embedder or HashingEmbedder(dimension=128),
        clock=clock or ManualClock(),
        policy=policy,
        assembler=assembler,
        **kwargs,
    )


def seed_drafts(*texts: str, kind: MemoryKind = MemoryKind.SEMANTIC) -> list[MemoryDraft]:
    return [MemoryDraft(kind=kind, text=t) for t in texts]


def sample_turns() -> list[Turn]:
    """一段三段式的示例对话，覆盖偏好 / 事实 / 约束 / 情景四类抽取。"""
    return [
        Turn("user", "记住，回答尽量简短，不要长篇大论。", "t0"),
        Turn("assistant", "好的。", "t1"),
        Turn("user", "我们项目现在用 MySQL 8 做订单库。", "t2"),
        Turn("user", "以后都走灰度发布，上线前必须有回滚开关。", "t3"),
        Turn("user", "今天上线了新的对账服务，遇到了连接池打满的问题。", "t4"),
        Turn("assistant", "了解。", "t5"),
        Turn("user", "我们决定把订单库换成 PostgreSQL 16。", "t6"),
    ]


def collect_texts(items: Iterable[MemoryItem]) -> list[str]:
    return [i.text for i in items]


def active_texts(manager: MemoryManager) -> list[str]:
    return [i.text for i in manager.memories() if i.status is MemoryStatus.ACTIVE]
