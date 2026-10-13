from __future__ import annotations

import pytest

from agent_memory.errors import MemoryNotFoundError, StoreError
from agent_memory.models import MemoryItem, MemoryKind, MemoryStatus
from agent_memory.store import MemoryStore


def item(text: str, **kw) -> MemoryItem:
    kw.setdefault("kind", MemoryKind.SEMANTIC)
    kw.setdefault("created_at", 1000.0)
    kw.setdefault("updated_at", 1000.0)
    return MemoryItem(text=text, **kw)


class TestCRUD:
    def test_add_and_get_roundtrip(self, store):
        original = item(
            "订单库用 PostgreSQL",
            subject="订单库",
            predicate="技术选型",
            object="PostgreSQL 16",
            tags=["db"],
            source_turns=["t1"],
            metadata={"pattern": "stack_choice"},
            embedding=[0.1, 0.2, 0.3],
        )
        store.add(original)
        loaded = store.get(original.id)
        assert loaded is not None
        assert loaded.text == original.text
        assert loaded.subject == "订单库"
        assert loaded.tags == ["db"]
        assert loaded.source_turns == ["t1"]
        assert loaded.metadata == {"pattern": "stack_choice"}
        assert loaded.embedding == pytest.approx([0.1, 0.2, 0.3])
        assert loaded.kind is MemoryKind.SEMANTIC
        assert loaded.status is MemoryStatus.ACTIVE

    def test_duplicate_id_raises_store_error(self, store):
        first = item("a")
        store.add(first)
        with pytest.raises(StoreError):
            store.add(item("b", id=first.id))

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_require_missing_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.require("nope")

    def test_update_missing_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.update(item("a", id="ghost"))

    def test_update_persists_changes(self, store):
        original = item("旧文本")
        store.add(original)
        original.text = "新文本"
        original.importance = 0.9
        store.update(original, now=2000.0)
        loaded = store.require(original.id)
        assert loaded.text == "新文本"
        assert loaded.importance == pytest.approx(0.9)
        assert loaded.updated_at == 2000.0

    def test_count(self, store):
        assert store.count() == 0
        store.add(item("a"))
        store.add(item("b"))
        assert store.count() == 2


class TestListing:
    def test_filter_by_status(self, store):
        a, b = item("a"), item("b")
        store.add(a)
        store.add(b)
        store.set_status(b.id, MemoryStatus.EXPIRED)
        assert [i.id for i in store.list(status=MemoryStatus.ACTIVE)] == [a.id]
        assert [i.id for i in store.list(status=MemoryStatus.EXPIRED)] == [b.id]
        assert len(store.list(status=None)) == 2

    def test_filter_by_kind(self, store):
        store.add(item("a", kind=MemoryKind.PREFERENCE))
        store.add(item("b", kind=MemoryKind.SEMANTIC))
        got = store.list(kinds=[MemoryKind.PREFERENCE])
        assert [i.kind for i in got] == [MemoryKind.PREFERENCE]

    def test_filter_by_ids(self, store):
        a, b = item("a"), item("b")
        store.add(a)
        store.add(b)
        assert {i.id for i in store.list(ids=[b.id, a.id])} == {a.id, b.id}

    def test_empty_ids_returns_empty(self, store):
        store.add(item("a"))
        assert store.list(ids=[]) == []

    def test_limit(self, store):
        for i in range(5):
            store.add(item(f"m{i}", created_at=1000.0 + i))
        assert len(store.list(limit=2)) == 2

    def test_ordered_by_created_at(self, store):
        store.add(item("late", created_at=2000.0))
        store.add(item("early", created_at=1000.0))
        assert [i.text for i in store.list()] == ["early", "late"]


class TestFindByKey:
    def test_case_insensitive_match(self, store):
        store.add(item("a", subject="OrderDB", predicate="Choice", object="MySQL"))
        assert len(store.find_by_key("orderdb", "choice")) == 1

    def test_excludes_inactive_by_default(self, store):
        a = item("a", subject="订单库", predicate="技术选型", object="MySQL")
        store.add(a)
        store.set_status(a.id, MemoryStatus.SUPERSEDED, superseded_by="x")
        assert store.find_by_key("订单库", "技术选型") == []
        assert len(store.find_by_key("订单库", "技术选型", include_inactive=True)) == 1


class TestLifecycle:
    def test_set_status_superseded(self, store):
        a = item("a")
        store.add(a)
        store.set_status(
            a.id, MemoryStatus.SUPERSEDED, superseded_by="new", now=1500.0, reason="fact_changed"
        )
        loaded = store.require(a.id)
        assert loaded.status is MemoryStatus.SUPERSEDED
        assert loaded.superseded_by == "new"
        assert loaded.updated_at == 1500.0

    def test_set_status_missing_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.set_status("ghost", MemoryStatus.EXPIRED)

    def test_soft_delete_marks_expired(self, store):
        a = item("a")
        store.add(a)
        assert store.delete(a.id) is True
        assert store.require(a.id).status is MemoryStatus.EXPIRED

    def test_soft_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False

    def test_hard_delete_removes_row(self, store):
        a = item("a")
        store.add(a)
        assert store.delete(a.id, hard=True) is True
        assert store.get(a.id) is None

    def test_record_access(self, store):
        a = item("a")
        store.add(a)
        store.record_access(a.id, 5000.0, increment=3)
        loaded = store.require(a.id)
        assert loaded.access_count == 3
        assert loaded.last_accessed_at == 5000.0

    def test_clear(self, store):
        store.add(item("a"))
        store.clear()
        assert store.count() == 0


class TestStatsAndEvents:
    def test_stats(self, store):
        store.add(item("a", kind=MemoryKind.PREFERENCE))
        b = item("b", kind=MemoryKind.SEMANTIC)
        store.add(b)
        store.set_status(b.id, MemoryStatus.SUPERSEDED, superseded_by="a")
        stats = store.stats()
        assert stats.total == 2
        assert stats.active == 1
        assert stats.superseded == 1
        assert stats.by_kind == {"preference": 1, "semantic": 1}
        assert stats.tokens_stored > 0

    def test_events_recorded(self, store):
        a = item("a")
        store.add(a)
        store.set_status(a.id, MemoryStatus.EXPIRED, reason="manual")
        actions = [e["action"] for e in store.events()]
        assert "add" in actions and "expire" in actions

    def test_events_limit(self, store):
        for i in range(5):
            store.add(item(f"m{i}", id=f"id{i}"))
        assert len(store.events(limit=2)) == 2


class TestPersistenceAndPortability:
    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "mem.db"
        s1 = MemoryStore(path)
        a = item("订单库用 PostgreSQL")
        s1.add(a)
        s1.close()

        s2 = MemoryStore(path)
        assert s2.require(a.id).text == "订单库用 PostgreSQL"
        s2.close()

    def test_export_import_roundtrip(self, tmp_path):
        s1 = MemoryStore(tmp_path / "a.db")
        s1.add(item("订单库用 PostgreSQL", tags=["db"], embedding=[1.0, 0.0]))
        rows = s1.export()
        assert len(rows) == 1

        s2 = MemoryStore(tmp_path / "b.db")
        assert s2.import_items(rows) == 1
        loaded = s2.list()[0]
        assert loaded.text == "订单库用 PostgreSQL"
        assert loaded.tags == ["db"]
        # embedding 属于"要重建的派生物"，不参与导出
        assert loaded.embedding is None

    def test_pragmatic_parent_dir_created(self, tmp_path):
        s = MemoryStore(tmp_path / "nested" / "deep" / "mem.db")
        s.add(item("a"))
        assert s.count() == 1
        s.close()

    def test_context_manager(self, tmp_path):
        with MemoryStore(tmp_path / "m.db") as s:
            s.add(item("a"))
            assert s.count() == 1
