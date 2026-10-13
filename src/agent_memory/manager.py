"""门面层：把 store / policy / extractor / retriever / assembler 串成一个 Agent 能用的对象。

对外只暴露 6 个动作，够 Agent 主循环用完：

===========  ================================================
动作          语义
===========  ================================================
``observe``   把一轮对话喂进来 → 抽取 → 策略决策 → 落库
``remember``  直接写一条记忆（Agent 主动调用"记住这个"时用）
``recall``    召回 → 组装上下文 → 刷新访问统计
``forget``    遗忘（软删除，可回滚）
``consolidate`` 手动触发记忆巩固
``stats``     观测指标（记忆条数、类型分布、token 占用）
===========  ================================================
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from .budget import AssembledContext, ContextAssembler
from .clock import Clock, SystemClock
from .consolidate import Consolidator
from .embedder import Embedder, HashingEmbedder
from .extract import ExtractionStrategy, RuleBasedExtractor, Turn
from .models import (
    ConsolidationReport,
    MemoryDraft,
    MemoryItem,
    MemoryKind,
    MemoryStats,
    MemoryStatus,
    RecallHit,
    WriteAction,
    WriteDecision,
)
from .policy import WritePolicy
from .retrieve import HybridRetriever, RetrievalWeights
from .store import MemoryStore

__all__ = ["MemoryManager", "RecallBundle"]


class RecallBundle:
    """一次召回的全部产物：命中的记忆 + 组装好的上下文 + 装箱账单。"""

    __slots__ = ("query", "hits", "context")

    def __init__(
        self, query: str, hits: list[RecallHit], context: AssembledContext | None
    ) -> None:
        self.query = query
        self.hits = hits
        self.context = context

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self.hits]

    @property
    def text(self) -> str:
        return self.context.text if self.context else ""

    def __len__(self) -> int:
        return len(self.hits)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"RecallBundle(query={self.query!r}, hits={len(self.hits)})"


class MemoryManager:
    """Agent 长期记忆门面。"""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        extractor: ExtractionStrategy | None = None,
        policy: WritePolicy | None = None,
        retriever: HybridRetriever | None = None,
        assembler: ContextAssembler | None = None,
        consolidator: Consolidator | None = None,
        clock: Clock | None = None,
        weights: RetrievalWeights | None = None,
        auto_consolidate_every: int | None = None,
        history_limit: int = 200,
    ) -> None:
        self.store = store or MemoryStore()
        self.embedder = embedder or HashingEmbedder()
        self.extractor = extractor or RuleBasedExtractor()
        self.policy = policy or WritePolicy()
        self.retriever = retriever or HybridRetriever(self.embedder, weights)
        self.assembler = assembler or ContextAssembler()
        self.consolidator = consolidator or Consolidator(
            self.store, self.embedder, self.policy
        )
        self.clock = clock or SystemClock()

        self.auto_consolidate_every = auto_consolidate_every
        self._observe_count = 0
        self._history_limit = history_limit
        self.decisions: list[WriteDecision] = []

    # ---------- 内部 ----------

    def _record(self, decision: WriteDecision) -> None:
        self.decisions.append(decision)
        if len(self.decisions) > self._history_limit:
            del self.decisions[: len(self.decisions) - self._history_limit]

    def _embed(self, text: str) -> list[float]:
        return self.embedder.embed_one(text)

    def _apply(self, decision: WriteDecision, now: float) -> MemoryItem | None:
        """把决策落到存储上。这是唯一会写库的地方。"""
        self._record(decision)
        if decision.action is WriteAction.NOOP:
            return None

        draft = decision.draft

        if decision.action is WriteAction.ADD:
            item = draft.to_item(now=now)
            item.importance = decision.importance
            item.embedding = self._embed(item.text)
            return self.store.add(item)

        if decision.action is WriteAction.SUPERSEDE:
            old = self.store.get(decision.target_id) if decision.target_id else None
            item = draft.to_item(now=now)
            item.importance = decision.importance
            item.embedding = self._embed(item.text)
            if old is not None:
                item.tags = sorted(set(item.tags) | set(old.tags))
            self.store.add(item)
            # 同 key 的历史记忆**全部**标为被取代：确认一下，别只处理最近一条，
            # 否则库里会留下"互相矛盾但都 active"的僵尸记忆。
            for stale in self.store.find_by_key(
                draft.subject or "", draft.predicate or ""
            ):
                if stale.id == item.id:
                    continue
                self.store.set_status(
                    stale.id,
                    MemoryStatus.SUPERSEDED,
                    superseded_by=item.id,
                    now=now,
                    reason=decision.reason,
                )
            return item

        # MERGE：更新既有条目，不新增
        target = self.store.get(decision.target_id) if decision.target_id else None
        if target is None:  # 目标被并发删掉了，降级成新增
            item = draft.to_item(now=now)
            item.importance = decision.importance
            item.embedding = self._embed(item.text)
            return self.store.add(item)

        if len(draft.text) > len(target.text):
            target.text = draft.text
        target.importance = max(target.importance, decision.importance)
        target.confidence = max(target.confidence, draft.confidence)
        target.tags = sorted(set(target.tags) | set(draft.tags))
        target.source_turns = sorted(set(target.source_turns) | set(draft.source_turns))
        target.embedding = self._embed(target.text)
        self.store.update(target, now=now)
        return target

    # ---------- 写入 ----------

    def remember(
        self,
        text: str,
        *,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        confidence: float = 0.8,
        tags: Sequence[str] = (),
        ttl_s: float | None = None,
        source_turns: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> WriteDecision:
        """直接写一条记忆，不做自然语言抽取。

        Agent 显式说"把这条记下来"时走这条路——此时意图是明确的，
        再抽取一次只会引入误差。
        """
        now = self.clock.now()
        draft = MemoryDraft(
            kind=kind,
            text=text,
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            ttl_s=ttl_s,
            tags=list(tags),
            source_turns=list(source_turns),
            metadata=dict(metadata or {}),
        )
        decision = self.policy.decide(
            draft, self.store.list(status=MemoryStatus.ACTIVE), embed=self._embed
        )
        self._apply(decision, now)
        return decision

    def observe(
        self,
        text: str,
        *,
        role: str = "user",
        turn_id: str | None = None,
    ) -> list[WriteDecision]:
        """把一轮发言喂进来，返回本轮产生的全部写入决策。"""
        if role not in getattr(self.extractor, "roles", ("user",)):
            return []
        now = self.clock.now()
        drafts = self.extractor.extract(text, turn_id=turn_id)
        decisions: list[WriteDecision] = []
        for draft in drafts:
            decision = self.policy.decide(
                draft, self.store.list(status=MemoryStatus.ACTIVE), embed=self._embed
            )
            self._apply(decision, now)
            decisions.append(decision)

        self._observe_count += 1
        if (
            self.auto_consolidate_every
            and self._observe_count % self.auto_consolidate_every == 0
        ):
            self.consolidate(now=now)
        return decisions

    def observe_turns(self, turns: Iterable[Turn]) -> list[WriteDecision]:
        out: list[WriteDecision] = []
        for turn in turns:
            out.extend(self.observe(turn.text, role=turn.role, turn_id=turn.turn_id))
        return out

    # ---------- 读取 ----------

    def recall(
        self,
        query: str,
        *,
        top_k: int = 8,
        kinds: Sequence[MemoryKind] | None = None,
        tags: Sequence[str] | None = None,
        min_score: float = 0.0,
        diversify: bool = False,
        mmr_lambda: float = 0.7,
        with_context: bool = True,
        max_tokens: int | None = None,
        touch: bool = True,
        use_rrf: bool = False,
    ) -> RecallBundle:
        now = self.clock.now()
        candidates = self.store.list(status=MemoryStatus.ACTIVE)
        if use_rrf:
            hits = self.retriever.retrieve_rrf(
                query, candidates, now=now, top_k=top_k, kinds=kinds, tags=tags
            )
        else:
            hits = self.retriever.retrieve(
                query,
                candidates,
                now=now,
                top_k=top_k,
                kinds=kinds,
                tags=tags,
                min_score=min_score,
                diversify=diversify,
                mmr_lambda=mmr_lambda,
            )
        if touch and hits:
            for hit in hits:
                self.store.record_access(hit.id, now)
                hit.item.touch(now)

        context = (
            self.assembler.assemble(
                hits, now=now, max_tokens=max_tokens or self.assembler.max_tokens
            )
            if with_context
            else None
        )
        return RecallBundle(query, hits, context)

    def context_for(self, query: str, *, max_tokens: int | None = None, **kwargs: Any) -> str:
        """最常用的一行接口：给我可直接拼进 prompt 的记忆块。"""
        return self.recall(query, with_context=True, max_tokens=max_tokens, **kwargs).text

    def get(self, memory_id: str) -> MemoryItem | None:
        return self.store.get(memory_id)

    def memories(self, *, include_inactive: bool = False) -> list[MemoryItem]:
        if include_inactive:
            return self.store.all()
        return self.store.list(status=MemoryStatus.ACTIVE)

    # ---------- 遗忘 ----------

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def forget_matching(self, predicate: Callable[[MemoryItem], bool]) -> int:
        """按条件批量遗忘。``predicate`` 是纯函数，便于测试。"""
        now = self.clock.now()
        n = 0
        for item in self.store.list(status=MemoryStatus.ACTIVE):
            if predicate(item):
                self.store.set_status(item.id, MemoryStatus.EXPIRED, now=now, reason="manual_bulk")
                n += 1
        return n

    # ---------- 维护 ----------

    def consolidate(self, *, now: float | None = None, dry_run: bool = False) -> ConsolidationReport:
        return self.consolidator.run(now if now is not None else self.clock.now(), dry_run=dry_run)

    def stats(self) -> MemoryStats:
        return self.store.stats()

    def events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.events(limit=limit)

    def export(self) -> list[dict[str, Any]]:
        return self.store.export()

    def import_(self, rows: Iterable[dict[str, Any]]) -> int:
        return self.store.import_items(rows)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
