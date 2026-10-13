from __future__ import annotations

import pytest

from agent_memory.manager import MemoryManager
from agent_memory.models import MemoryKind, MemoryStatus, WriteAction
from agent_memory.testing import make_manager

from conftest import START

DAY = 86_400.0


class TestObserve:
    def test_single_sentence_creates_memory(self, manager):
        decisions = manager.observe("我们项目现在用 MySQL 8 做订单库。")
        assert len(decisions) == 1
        assert decisions[0].action is WriteAction.ADD
        assert len(manager.memories()) == 1

    def test_assistant_turns_ignored(self, manager):
        assert manager.observe("我叫周彪。", role="assistant") == []
        assert manager.memories() == []

    def test_chitchat_produces_nothing(self, manager):
        assert manager.observe("嗯嗯，好的") == []
        assert manager.memories() == []

    def test_source_turn_recorded(self, manager):
        manager.observe("我叫周彪。", turn_id="t9")
        [item] = manager.memories()
        assert item.source_turns == ["t9"]

    def test_observe_turns_batch(self, manager):
        from agent_memory.extract import Turn

        turns = [
            Turn("user", "我叫周彪。", "t0"),
            Turn("assistant", "你好。", "t1"),
            Turn("user", "我的时区是 UTC+8。", "t2"),
        ]
        decisions = manager.observe_turns(turns)
        assert len(decisions) == 2
        assert len(manager.memories()) == 2

    def test_decisions_history_limited(self):
        mgr = make_manager(history_limit=2)
        vector = mgr.embedder
        assert vector is not None
        mgr.observe("我叫周彪。")
        mgr.observe("我的时区是 UTC+8。")
        mgr.observe("我是后端工程师。")
        assert len(mgr.decisions) == 2
        mgr.close()


class TestRemember:
    def test_direct_write(self, manager):
        decision = manager.remember(
            "订单库使用 PostgreSQL 16",
            kind=MemoryKind.SEMANTIC,
            subject="订单库",
            predicate="技术选型",
            object_="PostgreSQL 16",
            tags=["db"],
        )
        assert decision.action is WriteAction.ADD
        [item] = manager.memories()
        assert item.subject == "订单库"
        assert item.tags == ["db"]

    def test_duplicate_direct_write_is_noop(self, manager):
        for _ in range(2):
            manager.remember("订单库使用 PostgreSQL 16 做主库")
        assert len(manager.memories()) == 1

    def test_ttl_honoured(self):
        mgr = make_manager()
        mgr.remember("短期上下文", ttl_s=60.0)
        [item] = mgr.memories()
        assert item.ttl_s == 60.0
        mgr.clock.advance(120)
        assert mgr.recall("短期上下文").hits == []
        mgr.close()


class TestSupersedeAndMerge:
    def test_fact_change_supersedes_old(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        decisions = manager.observe("我们决定把订单库换成 PostgreSQL 16。")
        assert decisions[0].action is WriteAction.SUPERSEDE

        active = manager.memories()
        assert len(active) == 1
        assert "PostgreSQL" in active[0].text

        all_items = manager.memories(include_inactive=True)
        stale = [i for i in all_items if "MySQL" in i.text][0]
        assert stale.status is MemoryStatus.SUPERSEDED
        assert stale.superseded_by == active[0].id

    def test_supersede_inherits_tags(self, manager):
        first = manager.remember(
            "订单库使用 MySQL 8",
            subject="订单库",
            predicate="技术选型",
            object_="MySQL 8",
            tags=["db"],
        )
        second = manager.remember(
            "订单库使用 PostgreSQL 16",
            subject="订单库",
            predicate="技术选型",
            object_="PostgreSQL 16",
            tags=["stack"],
        )
        assert first.action is WriteAction.ADD
        assert second.action is WriteAction.SUPERSEDE
        [item] = manager.memories()
        assert set(item.tags) == {"db", "stack"}

    def test_constraints_coexist(self, manager):
        manager.observe("记住，回答尽量简短，不要长篇大论。")
        manager.observe("以后都走灰度发布，上线前必须有回滚开关。")
        kinds = {i.kind for i in manager.memories()}
        assert MemoryKind.PREFERENCE in kinds
        assert MemoryKind.PROCEDURAL in kinds
        assert len(manager.memories()) == 2


class TestRecall:
    def test_recall_returns_context(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        bundle = manager.recall("订单库用什么数据库？")
        assert len(bundle) == 1
        assert bundle.text.startswith("## 关于用户的已知记忆")

    def test_recall_touches_access_stats(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        manager.recall("订单库")
        [item] = manager.memories()
        assert item.access_count == 1
        assert item.last_accessed_at == START

    def test_recall_without_touch(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        manager.recall("订单库", touch=False)
        assert manager.memories()[0].access_count == 0

    def test_recall_without_context(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        bundle = manager.recall("订单库", with_context=False)
        assert bundle.context is None
        assert bundle.text == ""

    def test_ids_helper(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        bundle = manager.recall("订单库")
        assert bundle.ids == [h.id for h in bundle.hits]

    def test_context_for_returns_string(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        assert isinstance(manager.context_for("订单库"), str)
        assert "MySQL" in manager.context_for("订单库")

    def test_recall_empty_store(self, manager):
        bundle = manager.recall("随便问一句")
        assert len(bundle) == 0
        assert bundle.text == ""

    def test_recall_respects_token_budget(self, manager):
        for text in (
            "我们项目现在用 MySQL 8 做订单库。",
            "以后都走灰度发布，上线前必须有回滚开关。",
            "记住，回答尽量简短，不要长篇大论。",
        ):
            manager.observe(text)
        bundle = manager.recall("订单库 灰度 回答", max_tokens=40)
        assert bundle.context is not None
        assert bundle.context.tokens_used <= 40

    def test_recall_use_rrf(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        bundle = manager.recall("订单库", use_rrf=True)
        assert isinstance(bundle, type(bundle))
        assert len(bundle) >= 1

    def test_kind_filter(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        manager.observe("记住，回答尽量简短。")
        bundle = manager.recall("回答", kinds=[MemoryKind.PREFERENCE])
        assert all(h.item.kind is MemoryKind.PREFERENCE for h in bundle.hits)


class TestForgetAndMaintenance:
    def test_forget_soft_deletes(self, manager):
        manager.observe("我叫周彪。")
        [item] = manager.memories()
        assert manager.forget(item.id) is True
        assert manager.memories() == []
        assert manager.get(item.id).status is MemoryStatus.EXPIRED

    def test_forget_missing_returns_false(self, manager):
        assert manager.forget("ghost") is False

    def test_forget_matching(self, manager):
        manager.observe("记住，回答尽量简短。")
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        removed = manager.forget_matching(lambda i: i.kind is MemoryKind.PREFERENCE)
        assert removed == 1
        assert len(manager.memories()) == 1

    def test_consolidate(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        report = manager.consolidate()
        assert report.scanned == 1

    def test_stats(self, manager):
        manager.observe("记住，回答尽量简短。")
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        stats = manager.stats()
        assert stats.total == 2
        assert stats.active == 2
        assert set(stats.by_kind) == {"preference", "semantic"}
        assert stats.tokens_stored > 0

    def test_events(self, manager):
        manager.observe("我叫周彪。")
        assert any(e["action"] == "add" for e in manager.events())

    def test_auto_consolidate_trigger(self):
        mgr = make_manager(auto_consolidate_every=2)
        mgr.observe("我们项目现在用 MySQL 8 做订单库。")
        mgr.observe("我叫周彪。")
        # 第 2 次 observe 之后应已自动巩固过一轮
        assert mgr._observe_count == 2
        assert mgr.stats().active >= 2
        mgr.close()

    def test_export_import_between_managers(self, manager):
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        rows = manager.export()
        other = make_manager()
        assert other.import_(rows) == 1
        assert other.memories()[0].text == manager.memories()[0].text
        other.close()

    def test_context_manager_protocol(self):
        with make_manager() as mgr:
            mgr.observe("我叫周彪。")
            assert len(mgr.memories()) == 1

    def test_get_returns_none_for_unknown(self, manager):
        assert manager.get("ghost") is None

    def test_close_is_safe(self, manager):
        manager.close()
        manager.close()
