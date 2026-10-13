from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_memory.errors import MemoryValidationError
from agent_memory.models import (
    ConsolidationReport,
    MemoryDraft,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    MemoryStats,
    MemoryStatus,
    RecallHit,
    WriteAction,
    WriteDecision,
    new_id,
)


class TestMemoryItem:
    def test_key_normalizes_case_and_space(self):
        item = MemoryItem(
            kind=MemoryKind.SEMANTIC,
            text="订单库用 PostgreSQL",
            subject=" 订单库 ",
            predicate="技术选型",
        )
        assert item.key == ("订单库", "技术选型")

    def test_key_is_none_without_subject_or_predicate(self):
        a = MemoryItem(kind=MemoryKind.EPISODIC, text="今天上线了")
        assert a.key is None
        b = MemoryItem(kind=MemoryKind.EPISODIC, text="今天上线了", subject="事件")
        assert b.key is None

    def test_blank_text_rejected(self):
        with pytest.raises(MemoryValidationError):
            MemoryItem(kind=MemoryKind.SEMANTIC, text="   ")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            MemoryItem(kind=MemoryKind.SEMANTIC, text="x", nope=1)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_importance_out_of_range(self, value):
        with pytest.raises(MemoryValidationError):
            MemoryItem(kind=MemoryKind.SEMANTIC, text="x", importance=value)

    def test_negative_ttl_rejected(self):
        with pytest.raises(MemoryValidationError):
            MemoryItem(kind=MemoryKind.SEMANTIC, text="x", ttl_s=-1)

    def test_age_is_never_negative(self):
        item = MemoryItem(kind=MemoryKind.SEMANTIC, text="x", created_at=100.0)
        assert item.age_s(99.0) == 0.0
        assert item.age_s(160.0) == 60.0

    def test_is_expired_by_ttl(self):
        item = MemoryItem(kind=MemoryKind.SEMANTIC, text="x", created_at=0.0, ttl_s=100.0)
        assert not item.is_expired(50.0)
        assert item.is_expired(101.0)

    def test_is_expired_when_status_expired(self):
        item = MemoryItem(
            kind=MemoryKind.SEMANTIC, text="x", status=MemoryStatus.EXPIRED
        )
        assert item.is_expired(0.0)

    def test_touch_updates_access_stats(self):
        item = MemoryItem(kind=MemoryKind.SEMANTIC, text="x")
        assert item.access_count == 0 and item.last_accessed_at is None
        item.touch(123.0)
        assert item.access_count == 1
        assert item.last_accessed_at == 123.0

    def test_recap_contains_kind_and_text(self):
        item = MemoryItem(kind=MemoryKind.PREFERENCE, text="回答要简短")
        assert "preference" in item.recap() and "回答要简短" in item.recap()

    def test_embedding_excluded_from_dump(self):
        item = MemoryItem(kind=MemoryKind.SEMANTIC, text="x", embedding=[1.0, 2.0])
        assert "embedding" not in item.model_dump()
        assert item.embedding == [1.0, 2.0]


class TestMemoryDraft:
    def test_to_item_fills_id_and_timestamps(self):
        draft = MemoryDraft(kind=MemoryKind.SEMANTIC, text="订单库用 PostgreSQL")
        item = draft.to_item(now=1234.0)
        assert item.created_at == 1234.0
        assert item.updated_at == 1234.0
        assert item.id
        assert item.status is MemoryStatus.ACTIVE

    def test_to_item_honours_explicit_id(self):
        draft = MemoryDraft(kind=MemoryKind.SEMANTIC, text="x")
        item = draft.to_item(now=1.0, item_id="fixed")
        assert item.id == "fixed"

    def test_text_is_stripped(self):
        draft = MemoryDraft(kind=MemoryKind.SEMANTIC, text="  x  ")
        assert draft.text == "x"


class TestQueryAndStats:
    def test_top_k_must_be_positive(self):
        with pytest.raises(MemoryValidationError):
            MemoryQuery(text="x", top_k=0)

    def test_query_defaults(self):
        q = MemoryQuery(text="订单库")
        assert q.top_k == 8
        assert q.with_context is True
        assert q.kinds == []

    def test_stats_active_ratio(self):
        stats = MemoryStats(total=4, active=3, superseded=1)
        assert stats.active_ratio == 0.75

    def test_stats_active_ratio_empty(self):
        assert MemoryStats().active_ratio == 0.0


class TestDecisions:
    def test_writes_flag(self):
        draft = MemoryDraft(kind=MemoryKind.SEMANTIC, text="x")
        keep = WriteDecision(
            action=WriteAction.ADD, reason="r", draft=draft, importance=0.5
        )
        skip = WriteDecision(
            action=WriteAction.NOOP, reason="r", draft=draft, importance=0.5
        )
        assert keep.writes and not skip.writes

    def test_hit_id_property(self):
        item = MemoryItem(kind=MemoryKind.SEMANTIC, text="x")
        hit = RecallHit(item=item, score=0.9)
        assert hit.id == item.id

    def test_report_as_dict(self):
        report = ConsolidationReport(scanned=10, expired=2, merged=3)
        d = report.as_dict()
        assert d["scanned"] == 10 and d["merged"] == 3
        assert set(d) == {"scanned", "expired", "merged", "summarized", "decayed"}

    def test_new_id_is_unique(self):
        assert len({new_id() for _ in range(1000)}) == 1000
