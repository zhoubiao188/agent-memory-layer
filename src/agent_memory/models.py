"""记忆条目的数据契约。

一条记忆在系统里被三种消费者读写：

* **写入侧**（策略层）需要 ``subject/predicate/object`` 才能判断"这条是新事实还是旧事实的覆盖"；
* **检索侧**需要 ``embedding`` 与 ``importance``；
* **人类**需要 ``text`` 和 ``source_turns`` 来回答"它为什么会记住这个"。

所以 :class:`MemoryItem` 同时承载这三类字段，并用 ``status`` 表达生命周期，
而不是物理删除——**可回滚**是记忆系统和缓存最大的区别。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import MemoryValidationError

__all__ = [
    "MemoryKind",
    "MemoryStatus",
    "MemoryItem",
    "MemoryDraft",
    "MemoryQuery",
    "RecallHit",
    "WriteAction",
    "WriteDecision",
    "ConsolidationReport",
    "MemoryStats",
    "new_id",
]


def new_id() -> str:
    return uuid.uuid4().hex


class MemoryKind(str, Enum):
    """记忆的类型。分类不是为了好看，而是为了**不同的写入与淘汰策略**。"""

    EPISODIC = "episodic"  # 发生过什么：一次失败、一次发布
    SEMANTIC = "semantic"  # 稳定事实：用户的套餐、系统架构
    PROCEDURAL = "procedural"  # 怎么做事 / 约束：以后都要走灰度
    PREFERENCE = "preference"  # 用户偏好：喜欢简短回答


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # 被更新的同类事实覆盖
    EXPIRED = "expired"  # TTL 到期或被遗忘曲线淘汰


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _Validated(_Model):
    """校验规则放在共享基类里。

    :class:`MemoryDraft`（入库前）和 :class:`MemoryItem`（入库后）字段形状相同，
    如果只在校验草稿，就会出现"绕过抽取器直接构造 MemoryItem 写脏数据"的缺口。
    契约必须在**两个入口**都成立。
    """

    @field_validator("text", check_fields=False)
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise MemoryValidationError("memory text must not be blank")
        return v.strip()

    @field_validator("importance", "confidence", check_fields=False)
    @classmethod
    def _unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise MemoryValidationError(f"value out of [0,1]: {v}")
        return float(v)

    @field_validator("ttl_s", check_fields=False)
    @classmethod
    def _positive_ttl(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise MemoryValidationError(f"ttl_s must be positive: {v}")
        return v


class MemoryDraft(_Validated):
    """尚未落库的记忆候选。没有 id / 时间戳，由 store 补齐。"""

    kind: MemoryKind
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    importance: float = 0.5
    confidence: float = 0.7
    ttl_s: float | None = None
    tags: list[str] = Field(default_factory=list)
    source_turns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_item(self, *, now: float, item_id: str | None = None) -> "MemoryItem":
        return MemoryItem(
            **self.model_dump(),
            id=item_id or new_id(),
            created_at=now,
            updated_at=now,
        )


class MemoryItem(_Validated):
    """一条已落库的记忆。"""

    id: str = Field(default_factory=new_id)
    kind: MemoryKind
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None

    importance: float = 0.5
    confidence: float = 0.7
    status: MemoryStatus = MemoryStatus.ACTIVE

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    last_accessed_at: float | None = None
    access_count: int = 0

    ttl_s: float | None = None
    superseded_by: str | None = None

    tags: list[str] = Field(default_factory=list)
    source_turns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    #: 运行时字段，不参与序列化（落库走 metadata / 单独列）。
    embedding: list[float] | None = Field(default=None, exclude=True, repr=False)

    # ---------- 派生属性 ----------

    @property
    def key(self) -> tuple[str, str] | None:
        """冲突判定用的键：``(subject, predicate)`` 规范化后的二元组。

        缺少任一字段时返回 ``None``，表示"这条记忆不参与取代判定"，
        只能走相似度去重。**这个保守的设计很重要**：宁可不取代，
        也不要因为 key 提取错了把用户的正确记忆标成 superseded。
        """
        if not self.subject or not self.predicate:
            return None
        return (self.subject.strip().lower(), self.predicate.strip().lower())

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.created_at)

    def is_expired(self, now: float) -> bool:
        if self.status is MemoryStatus.EXPIRED:
            return True
        if self.ttl_s is None:
            return False
        return (now - self.created_at) > self.ttl_s

    def touch(self, now: float) -> None:
        """被召回时更新访问统计。写入访问频次是"重要性"最强的经验信号之一。"""
        self.access_count += 1
        self.last_accessed_at = now

    def recap(self) -> str:
        """给模型看的一行摘要，带时间与类型，便于它自己判断可信度。"""
        return f"[{self.kind.value}] {self.text}"


class MemoryQuery(_Model):
    text: str = ""
    kinds: list[MemoryKind] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    top_k: int = 8
    min_score: float = 0.0
    with_context: bool = True
    context_max_tokens: int = 800

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise MemoryValidationError(f"top_k must be > 0: {v}")
        return v


class RecallHit(_Model):
    item: MemoryItem
    score: float
    components: dict[str, float] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.item.id


class WriteAction(str, Enum):
    ADD = "add"  # 新记忆
    SUPERSEDE = "supersede"  # 覆盖旧的同类事实
    MERGE = "merge"  # 与高度相似的一条合并（取更长/更高分）
    NOOP = "noop"  # 完全重复，丢弃


class WriteDecision(_Model):
    action: WriteAction
    reason: str
    draft: MemoryDraft
    importance: float
    target_id: str | None = None
    similarity: float = 0.0

    @property
    def writes(self) -> bool:
        return self.action is not WriteAction.NOOP


class ConsolidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expired: int = 0
    merged: int = 0
    summarized: int = 0
    decayed: int = 0
    scanned: int = 0

    def as_dict(self) -> Mapping[str, int]:
        return {
            "scanned": self.scanned,
            "expired": self.expired,
            "merged": self.merged,
            "summarized": self.summarized,
            "decayed": self.decayed,
        }


class MemoryStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 0
    active: int = 0
    superseded: int = 0
    expired: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    tokens_stored: int = 0

    @property
    def active_ratio(self) -> float:
        return 0.0 if self.total == 0 else self.active / self.total
