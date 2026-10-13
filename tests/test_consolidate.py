from __future__ import annotations

import pytest

from agent_memory.consolidate import Consolidator
from agent_memory.embedder import HashingEmbedder
from agent_memory.models import MemoryItem, MemoryKind, MemoryStatus
from agent_memory.policy import WritePolicy

from conftest import START

DAY = 86_400.0


@pytest.fixture()
def cons(store):
    return Consolidator(store, HashingEmbedder(dimension=32), WritePolicy())


def add(
    store,
    text: str,
    *,
    kind: MemoryKind = MemoryKind.SEMANTIC,
    embedding=None,
    now: float = START,
    **kw,
) -> MemoryItem:
    kw.setdefault("importance", 0.5)
    item = MemoryItem(
        kind=kind, text=text, created_at=now, updated_at=now, embedding=embedding, **kw
    )
    return store.add(item)


class TestExpire:
    def test_ttl_expiry(self, store, cons):
        item = add(store, "临时上下文", ttl_s=60.0, importance=0.9)
        report = cons.run(START + 120)
        assert report.expired == 1
        assert store.require(item.id).status is MemoryStatus.EXPIRED

    def test_decay_expiry_for_episodic(self, store, cons):
        item = add(store, "昨天的一次故障排查", kind=MemoryKind.EPISODIC, importance=0.4)
        report = cons.run(START + 30 * DAY)
        assert report.expired == 1
        assert store.require(item.id).status is MemoryStatus.EXPIRED

    def test_preference_never_expires(self, store, cons):
        item = add(store, "回答要简短", kind=MemoryKind.PREFERENCE, importance=0.1)
        report = cons.run(START + 1000 * DAY)
        assert report.expired == 0
        assert store.require(item.id).status is MemoryStatus.ACTIVE

    def test_expiry_is_soft(self, store, cons):
        item = add(store, "临时上下文", ttl_s=1.0, importance=0.9)
        cons.run(START + 10)
        # 记录还在，只是状态变了——记忆系统必须可复盘
        assert store.get(item.id) is not None


class TestMerge:
    def test_near_duplicates_merged(self, store, cons):
        keep = add(store, "订单库使用 PostgreSQL 16 做主库", embedding=[1.0, 0.0], importance=0.8)
        drop = add(store, "订单库使用 PostgreSQL 做主库", embedding=[1.0, 0.0], importance=0.5)
        report = cons.run(START)
        assert report.merged == 1
        assert store.require(drop.id).status is MemoryStatus.SUPERSEDED
        assert store.require(drop.id).superseded_by == keep.id
        assert store.require(keep.id).status is MemoryStatus.ACTIVE

    def test_merge_unions_tags_and_sources(self, store, cons):
        keep = add(
            store, "订单库用 PostgreSQL", embedding=[1.0, 0.0], importance=0.8,
            tags=["db"], source_turns=["t1"],
        )
        drop = add(
            store, "订单库用 PostgreSQL 16", embedding=[1.0, 0.0], importance=0.5,
            tags=["stack"], source_turns=["t2"],
        )
        cons.run(START)
        merged = store.require(keep.id)
        assert set(merged.tags) == {"db", "stack"}
        assert set(merged.source_turns) == {"t1", "t2"}

    def test_different_kinds_not_merged(self, store, cons):
        add(store, "文字一样的内容", kind=MemoryKind.SEMANTIC, embedding=[1.0, 0.0])
        add(store, "文字一样的内容", kind=MemoryKind.PROCEDURAL, embedding=[1.0, 0.0])
        report = cons.run(START)
        assert report.merged == 0

    def test_unrelated_not_merged(self, store, cons):
        add(store, "订单库用 PostgreSQL", embedding=[1.0, 0.0])
        add(store, "食堂今天排队很长", embedding=[0.0, 1.0])
        assert cons.run(START).merged == 0


class TestSummarize:
    def _three_episodic(self, store):
        return [
            add(
                store,
                f"第{i}次遇到连接池打满",
                kind=MemoryKind.EPISODIC,
                subject="事件",
                embedding=vec,
                importance=0.8,
                tags=["episodic"],
            )
            for i, vec in enumerate(([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]))
        ]

    def test_cluster_summarised_into_semantic(self, store, cons):
        members = self._three_episodic(store)
        report = cons.run(START)
        assert report.summarized == 3
        summaries = [i for i in store.list() if "consolidated" in i.tags]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.kind is MemoryKind.SEMANTIC
        assert summary.text.startswith("事件方面近期反复出现")
        for m in members:
            assert store.require(m.id).status is MemoryStatus.SUPERSEDED
            assert store.require(m.id).superseded_by == summary.id

    def test_below_threshold_not_summarised(self, store, cons):
        add(store, "第一次遇到连接池打满", kind=MemoryKind.EPISODIC, subject="事件",
            embedding=[1.0, 0.0], importance=0.8)
        add(store, "第二次遇到连接池打满", kind=MemoryKind.EPISODIC, subject="事件",
            embedding=[0.0, 1.0], importance=0.8)
        assert cons.run(START).summarized == 0

    def test_can_be_disabled(self, store, cons):
        self._three_episodic(store)
        assert cons.run(START, summarize=False).summarized == 0

    def test_summary_records_provenance(self, store, cons):
        members = self._three_episodic(store)
        cons.run(START)
        summary = [i for i in store.list() if "consolidated" in i.tags][0]
        assert set(summary.metadata["consolidated_from"]) == {m.id for m in members}


class TestDecay:
    def test_importance_decays(self, store, cons):
        item = add(store, "订单库使用 PostgreSQL 做主库", importance=0.8)
        report = cons.run(START + 90 * DAY)
        assert report.decayed >= 1
        assert store.require(item.id).importance < 0.8

    def test_no_decay_when_fresh(self, store, cons):
        add(store, "订单库使用 PostgreSQL 做主库", importance=0.8)
        assert cons.run(START).decayed == 0


class TestRun:
    def test_scan_count(self, store, cons):
        for i in range(4):
            add(store, f"记忆 {i}", embedding=[1.0, float(i)])
        assert cons.run(START).scanned == 4

    def test_empty_store(self, store, cons):
        report = cons.run(START)
        assert report.as_dict() == {
            "scanned": 0,
            "expired": 0,
            "merged": 0,
            "summarized": 0,
            "decayed": 0,
        }

    def test_dry_run_changes_nothing(self, store, cons):
        keep = add(store, "订单库使用 PostgreSQL", embedding=[1.0, 0.0], importance=0.8)
        drop = add(store, "订单库使用 PostgreSQL 16", embedding=[1.0, 0.0], importance=0.5)
        before = {i.id: (i.status, i.importance) for i in store.all()}
        report = cons.run(START + 90 * DAY, dry_run=True)
        after = {i.id: (i.status, i.importance) for i in store.all()}
        assert before == after
        # 但报告仍然如实反映"如果不 dry run 会做什么"
        assert report.merged == 1
        assert len(store.list()) == 2
        assert store.require(keep.id).importance == pytest.approx(0.8)
        assert store.require(drop.id).status is MemoryStatus.ACTIVE

    def test_idempotent_second_run(self, store, cons):
        add(store, "订单库使用 PostgreSQL", embedding=[1.0, 0.0], importance=0.8)
        add(store, "订单库使用 PostgreSQL 16", embedding=[1.0, 0.0], importance=0.5)
        first = cons.run(START)
        second = cons.run(START)
        assert first.merged == 1
        assert second.merged == 0
