from __future__ import annotations

import pytest

from agent_memory.budget import ContextAssembler
from agent_memory.models import MemoryItem, MemoryKind, RecallHit

from conftest import START

DAY = 86_400.0


def hit(
    text: str,
    *,
    kind: MemoryKind = MemoryKind.SEMANTIC,
    score: float = 0.5,
    created_at: float = START,
    **kw,
) -> RecallHit:
    return RecallHit(
        item=MemoryItem(
            kind=kind, text=text, created_at=created_at, updated_at=created_at, **kw
        ),
        score=score,
    )


HITS = [
    hit("回答尽量简短，不要长篇大论", kind=MemoryKind.PREFERENCE, score=0.9),
    hit("以后都走灰度发布，上线前必须有回滚开关", kind=MemoryKind.PROCEDURAL, score=0.8),
    hit("订单库现在使用 PostgreSQL 16 做主库", score=0.7),
    hit("今天上线了对账服务，遇到连接池打满", kind=MemoryKind.EPISODIC, score=0.6),
]


class TestRenderLine:
    def test_age_humanised(self):
        asm = ContextAssembler()
        assert "刚刚" in asm.render_line(hit("x").item, now=START)
        assert "2 分钟前" in asm.render_line(hit("x").item, now=START + 120)
        assert "2 小时前" in asm.render_line(hit("x").item, now=START + 7200)
        assert "3 天前" in asm.render_line(hit("x").item, now=START + 3 * DAY)
        assert "2 个月前" in asm.render_line(hit("x").item, now=START + 70 * DAY)

    def test_citation_marker(self):
        asm = ContextAssembler()
        item = hit("x").item
        assert f"[m:{item.id[:6]}]" in asm.render_line(item, now=START)

    def test_citation_can_be_disabled(self):
        asm = ContextAssembler(cite=False)
        assert "[m:" not in asm.render_line(hit("x").item, now=START)

    def test_kind_can_be_shown(self):
        asm = ContextAssembler(show_kind=True)
        line = asm.render_line(hit("x", kind=MemoryKind.PREFERENCE).item, now=START)
        assert "preference" in line

    def test_age_can_be_hidden(self):
        asm = ContextAssembler(show_age=False)
        assert "刚刚" not in asm.render_line(hit("x").item, now=START)


class TestAssemble:
    def test_empty_hits(self):
        result = ContextAssembler().assemble([], now=START)
        assert result.text == ""
        assert result.is_empty
        assert result.tokens_used == 0

    def test_constraints_come_before_background(self):
        result = ContextAssembler(max_tokens=400).assemble(HITS, now=START)
        assert result.text.index("必须遵守的约定") < result.text.index("相关背景")
        assert result.text.startswith("## 关于用户的已知记忆")

    def test_all_items_included_with_room(self):
        result = ContextAssembler(max_tokens=400).assemble(HITS, now=START)
        assert len(result.included_ids) == 4
        assert result.dropped_ids == []
        assert result.truncated is False

    def test_overflow_is_dropped_not_cut(self):
        """放不下的整条丢弃并记账——半句话的记忆比没有记忆更危险。"""
        result = ContextAssembler(max_tokens=50).assemble(HITS, now=START)
        assert 0 < len(result.included_ids) < 4
        assert len(result.dropped_ids) == 4 - len(result.included_ids)
        assert result.truncated is True

    def test_never_exceeds_budget(self):
        for budget in (20, 40, 60, 90, 130, 200):
            result = ContextAssembler(max_tokens=budget).assemble(HITS, now=START)
            assert result.tokens_used <= budget
            assert result.budget == budget

    def test_budget_override_per_call(self):
        asm = ContextAssembler(max_tokens=400)
        result = asm.assemble(HITS, now=START, max_tokens=40)
        assert result.budget == 40
        assert result.truncated is True

    def test_header_too_large_returns_empty(self):
        result = ContextAssembler(max_tokens=1000).assemble(
            HITS, now=START, max_tokens=3
        )
        assert result.is_empty
        assert len(result.dropped_ids) == 4
        assert result.truncated is True

    def test_pin_constraints_off(self):
        result = ContextAssembler(max_tokens=400).assemble(
            HITS, now=START, pin_constraints=False
        )
        assert "必须遵守的约定" not in result.text
        assert "相关背景" in result.text
        assert len(result.included_ids) == 4

    def test_utilization(self):
        result = ContextAssembler(max_tokens=400).assemble(HITS, now=START)
        assert 0 < result.utilization < 1

    def test_repr_is_informative(self):
        result = ContextAssembler(max_tokens=400).assemble(HITS, now=START)
        assert "items=4" in repr(result)


class TestValidation:
    def test_constructor_rejects_zero(self):
        with pytest.raises(ValueError):
            ContextAssembler(max_tokens=0)

    def test_assemble_rejects_zero(self):
        with pytest.raises(ValueError):
            ContextAssembler().assemble(HITS, now=START, max_tokens=0)

    def test_included_and_dropped_are_disjoint(self):
        result = ContextAssembler(max_tokens=60).assemble(HITS, now=START)
        assert not (set(result.included_ids) & set(result.dropped_ids))
