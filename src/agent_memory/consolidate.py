"""记忆巩固：模拟"睡眠"时的整理与遗忘。

人类的记忆不是录音机。白天记下的碎片会在夜里被重放、裁剪、概括，
没用的淡出、重复的合并、零散的经历被抽象成规律。

一个只写不整理的记忆系统，三个月后一定退化成"什么都查不出来"——
因为库里 90% 是重复和噪音，检索预算全被它们吃掉。这一步是整个系统里
**唯一能让记忆质量随时间上升而不是下降的机制**。

四个动作，按代价从低到高：

1. ``expire``   —— TTL 到期 / 衰减低于阈值的，标记过期（不删）；
2. ``decay``    —— 剩余记忆按半衰期重新计算重要性（只改元数据，不删）；
3. ``merge``    —— 同类且高度相似的合并成一条；
4. ``summarize``——同一主题的多条情景记忆，抽象成一条语义记忆。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .embedder import Embedder, cosine
from .models import (
    ConsolidationReport,
    MemoryDraft,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
)
from .policy import WritePolicy
from .store import MemoryStore

__all__ = ["Consolidator"]

_MERGE_EPS = 1e-6


class _UnionFind:
    def __init__(self, ids: Iterable[str]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # 路径压缩
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in self._parent:
            out.setdefault(self.find(node), []).append(node)
        return out


class Consolidator:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        policy: WritePolicy | None = None,
        *,
        merge_threshold: float = 0.93,
        summarize_min_cluster: int = 3,
        summarize_max_chars: int = 160,
        decay_epsilon: float = 0.01,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.policy = policy or WritePolicy()
        self.merge_threshold = merge_threshold
        self.summarize_min_cluster = summarize_min_cluster
        self.summarize_max_chars = summarize_max_chars
        self.decay_epsilon = decay_epsilon

    # ---------- 工具 ----------

    def _vec(self, item: MemoryItem) -> list[float]:
        return item.embedding if item.embedding is not None else self.embedder.embed_one(item.text)

    # ---------- 四个动作 ----------

    def _expire(self, items: Sequence[MemoryItem], now: float, dry: bool) -> list[MemoryItem]:
        forgotten: list[MemoryItem] = []
        for item in items:
            if self.policy.should_forget(item, now):
                forgotten.append(item)
                if not dry:
                    self.store.set_status(
                        item.id,
                        MemoryStatus.EXPIRED,
                        now=now,
                        reason="decayed_below_threshold" if item.ttl_s is None else "ttl",
                    )
        return forgotten

    def _decay(self, items: Sequence[MemoryItem], now: float, dry: bool) -> list[MemoryItem]:
        changed: list[MemoryItem] = []
        for item in items:
            fresh = self.policy.decayed_importance(item, now)
            if abs(fresh - item.importance) > self.decay_epsilon:
                changed.append(item)
                if not dry:
                    item.importance = round(fresh, 6)
                    self.store.update(item, now=now)
        return changed

    def _merge(self, items: Sequence[MemoryItem], now: float, dry: bool) -> list[str]:
        """返回被"吸收"掉的记忆 id 列表。"""
        absorbed: list[str] = []
        by_kind: dict[MemoryKind, list[MemoryItem]] = {}
        for item in items:
            by_kind.setdefault(item.kind, []).append(item)

        for group in by_kind.values():
            if len(group) < 2:
                continue
            vectors = {i.id: self._vec(i) for i in group}
            uf = _UnionFind(i.id for i in group)
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    va, vb = vectors[group[a].id], vectors[group[b].id]
                    if len(va) != len(vb):
                        continue
                    if cosine(va, vb) >= self.merge_threshold - _MERGE_EPS:
                        uf.union(group[a].id, group[b].id)

            by_id = {i.id: i for i in group}
            for members in uf.groups().values():
                if len(members) < 2:
                    continue
                cluster = [by_id[m] for m in members]
                # 保留"分数最高、其次最长、再其次最早"的那条：稳定性优先
                keeper = sorted(
                    cluster,
                    key=lambda i: (-i.importance, -len(i.text), i.created_at),
                )[0]
                for loser in cluster:
                    if loser.id == keeper.id:
                        continue
                    absorbed.append(loser.id)
                    if dry:
                        continue
                    keeper.tags = sorted(set(keeper.tags) | set(loser.tags))
                    keeper.source_turns = sorted(
                        set(keeper.source_turns) | set(loser.source_turns)
                    )
                    keeper.importance = max(keeper.importance, loser.importance)
                    self.store.update(keeper, now=now)
                    self.store.set_status(
                        loser.id,
                        MemoryStatus.SUPERSEDED,
                        superseded_by=keeper.id,
                        now=now,
                        reason="merged_duplicate",
                    )
        return absorbed

    def _summarize(
        self, items: Sequence[MemoryItem], now: float, dry: bool
    ) -> list[str]:
        """把同主题的多条情景记忆抽象成一条语义记忆。返回被概括掉的 id 列表。"""
        buckets: dict[str, list[MemoryItem]] = {}
        for item in items:
            if item.kind is not MemoryKind.EPISODIC:
                continue
            topic = item.subject or (item.tags[0] if item.tags else None)
            if not topic:
                continue
            buckets.setdefault(topic, []).append(item)

        summarized: list[str] = []
        for topic, cluster in buckets.items():
            if len(cluster) < self.summarize_min_cluster:
                continue
            cluster = sorted(cluster, key=lambda i: i.importance, reverse=True)
            body = "；".join(i.text for i in cluster[:4])
            if len(body) > self.summarize_max_chars:
                body = body[: self.summarize_max_chars] + "…"
            text = f"{topic}方面近期反复出现：{body}"
            summary = MemoryDraft(
                kind=MemoryKind.SEMANTIC,
                text=text,
                subject=topic,
                predicate="归纳",
                object=body,
                importance=min(1.0, max(i.importance for i in cluster) + 0.1),
                confidence=0.65,
                tags=["consolidated"],
                source_turns=sorted({t for i in cluster for t in i.source_turns}),
                metadata={"consolidated_from": [i.id for i in cluster], "topic": topic},
            )
            if dry:
                summarized.extend(i.id for i in cluster)
                continue
            new_item = self.store.add(summary.to_item(now=now))
            new_item.embedding = self.embedder.embed_one(new_item.text)
            self.store.update(new_item, now=now)
            for member in cluster:
                self.store.set_status(
                    member.id,
                    MemoryStatus.SUPERSEDED,
                    superseded_by=new_item.id,
                    now=now,
                    reason="summarized",
                )
                summarized.append(member.id)
        return summarized

    # ---------- 编排 ----------

    def run(self, now: float, *, dry_run: bool = False, summarize: bool = True) -> ConsolidationReport:
        active = self.store.list(status=MemoryStatus.ACTIVE)
        report = ConsolidationReport(scanned=len(active))

        expired = self._expire(active, now, dry_run)
        report.expired = len(expired)
        survivors = [i for i in active if i not in expired]

        merged = self._merge(survivors, now, dry_run)
        report.merged = len(merged)
        survivors = [i for i in survivors if i.id not in set(merged)]

        if summarize:
            summarized = self._summarize(survivors, now, dry_run)
            report.summarized = len(summarized)
            survivors = [i for i in survivors if i.id not in set(summarized)]

        report.decayed = len(self._decay(survivors, now, dry_run))
        return report
