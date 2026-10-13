from __future__ import annotations

import pytest

from agent_memory.embedder import HashingEmbedder
from agent_memory.errors import PolicyError
from agent_memory.models import MemoryDraft, MemoryKind, WriteAction
from agent_memory.policy import DEFAULT_HALF_LIFE_S, SINGLE_VALUED_KINDS, WritePolicy

from conftest import START, FakeEmbedder, make_vec_item

DAY = 86_400.0


def draft(text: str, kind: MemoryKind = MemoryKind.SEMANTIC, **kw) -> MemoryDraft:
    return MemoryDraft(kind=kind, text=text, **kw)


class TestImportanceScoring:
    def test_explicit_remember_marker_raises_score(self):
        policy = WritePolicy()
        plain = policy.score_importance(draft("回答要简短", MemoryKind.PREFERENCE))
        marked = policy.score_importance(draft("记住回答要简短", MemoryKind.PREFERENCE))
        assert marked > plain

    def test_preference_outranks_episodic_for_same_text(self):
        policy = WritePolicy()
        text = "请记住这个偏好设置"
        assert policy.score_importance(
            draft(text, MemoryKind.PREFERENCE)
        ) > policy.score_importance(draft(text, MemoryKind.EPISODIC))

    def test_constraint_marker_boosts(self):
        policy = WritePolicy()
        assert policy.score_importance(
            draft("不要使用表情符号", MemoryKind.PROCEDURAL)
        ) > policy.score_importance(draft("使用表情符号", MemoryKind.PROCEDURAL))

    def test_very_short_text_penalised(self):
        policy = WritePolicy()
        short = policy.score_importance(draft("重启", MemoryKind.SEMANTIC))
        longer = policy.score_importance(
            draft("重启对账服务需要先摘流量", MemoryKind.SEMANTIC)
        )
        assert longer > short

    def test_score_always_in_unit_range(self):
        policy = WritePolicy()
        texts = ["", "记住", "记住" * 50, "务必" * 30 + "不要" * 30, "a"]
        for t in texts:
            if not t.strip():
                continue
            s = policy.score_importance(draft(t))
            assert 0.0 <= s <= 1.0

    def test_low_confidence_reduces(self):
        policy = WritePolicy()
        high = policy.score_importance(draft("订单库切换到新集群", confidence=0.9))
        low = policy.score_importance(draft("订单库切换到新集群", confidence=0.3))
        assert low < high

    def test_slot_structure_bonus(self):
        policy = WritePolicy()
        with_slot = policy.score_importance(
            draft("订单库使用 PostgreSQL", subject="订单库", predicate="技术选型")
        )
        without = policy.score_importance(draft("订单库使用 PostgreSQL"))
        assert with_slot > without


class TestChitchat:
    @pytest.mark.parametrize("text", ["好的", "嗯嗯", "谢谢", "ok", "Got it", "哈哈"])
    def test_detected(self, text):
        assert WritePolicy().is_chitchat(text)

    @pytest.mark.parametrize(
        "text", ["好的，那我们把超时改成 3 秒", "谢谢，另外记住不要用表情", "OK 那订单库换成 PostgreSQL"]
    )
    def test_not_detected_when_informative(self, text):
        assert not WritePolicy().is_chitchat(text)

    def test_chitchat_becomes_noop(self):
        policy = WritePolicy()
        decision = policy.decide(
            draft("好的", MemoryKind.PREFERENCE),
            [],
            embed=HashingEmbedder(dimension=32).embed_one,
        )
        assert decision.action is WriteAction.NOOP
        assert decision.reason == "chitchat"


class TestWriteDecisions:
    def test_empty_store_is_add(self):
        policy = WritePolicy()
        d = policy.decide(
            draft("订单库使用 PostgreSQL 16", subject="订单库", predicate="技术选型", object="PostgreSQL 16"),
            [],
            embed=HashingEmbedder(dimension=64).embed_one,
        )
        assert d.action is WriteAction.ADD
        assert d.reason == "new_memory"

    def test_exact_duplicate_is_noop(self):
        policy = WritePolicy()
        emb = FakeEmbedder({"回答要简短": [1.0, 0.0]})
        existing = [make_vec_item("回答要简短", [1.0, 0.0], kind=MemoryKind.PREFERENCE)]
        d = policy.decide(
            draft("回答要简短", MemoryKind.PREFERENCE), existing, embed=emb.embed_one
        )
        assert d.action is WriteAction.NOOP
        assert d.reason.startswith("duplicate")
        assert d.similarity == pytest.approx(1.0)

    def test_richer_restatement_is_merged(self):
        policy = WritePolicy(duplicate_threshold=0.70, merge_threshold=0.60)
        old_text, new_text = "回答要简短", "回答要简短，并且不要用长篇大论解释背景"
        emb = FakeEmbedder({old_text: [1.0, 0.0], new_text: [1.0, 0.0]})
        existing = [make_vec_item(old_text, [1.0, 0.0], kind=MemoryKind.PREFERENCE)]
        d = policy.decide(
            draft(new_text, MemoryKind.PREFERENCE), existing, embed=emb.embed_one
        )
        assert d.action is WriteAction.MERGE
        assert d.reason.startswith("richer_restatement")
        assert d.target_id == existing[0].id

    def test_near_duplicate_is_merged(self):
        policy = WritePolicy(duplicate_threshold=0.99, merge_threshold=0.80)
        new_text = "回答尽量简短一些"
        emb = FakeEmbedder({"回答要简短": [1.0, 0.0], new_text: [0.9, 0.4358898943540674]})
        existing = [make_vec_item("回答要简短", [1.0, 0.0], kind=MemoryKind.PREFERENCE)]
        d = policy.decide(
            draft(new_text, MemoryKind.PREFERENCE), existing, embed=emb.embed_one
        )
        assert d.action is WriteAction.MERGE
        assert d.reason.startswith("near_duplicate")

    def test_fact_change_supersedes(self):
        policy = WritePolicy()
        emb = FakeEmbedder({"old": [1.0, 0.0], "new": [0.2, 0.9797958971132712]})
        existing = [
            make_vec_item(
                "old", [1.0, 0.0], subject="订单库", predicate="技术选型", object="MySQL 8"
            )
        ]
        d = policy.decide(
            draft(
                "new",
                subject="订单库",
                predicate="技术选型",
                object="PostgreSQL 16",
            ),
            existing,
            embed=emb.embed_one,
        )
        assert d.action is WriteAction.SUPERSEDE
        assert "fact_changed" in d.reason
        assert d.target_id == existing[0].id

    def test_same_fact_restated_is_not_supersede(self):
        policy = WritePolicy(duplicate_threshold=0.99, merge_threshold=0.5)
        emb = FakeEmbedder({"a": [1.0, 0.0], "b": [0.7, 0.714142842854285]})
        existing = [
            make_vec_item(
                "a", [1.0, 0.0], subject="订单库", predicate="技术选型", object="MySQL 8"
            )
        ]
        d = policy.decide(
            draft("b", subject="订单库", predicate="技术选型", object="MySQL 8"),
            existing,
            embed=emb.embed_one,
        )
        assert d.action is not WriteAction.SUPERSEDE

    def test_constraints_accumulate_not_supersede(self):
        """约束是可累积的：两条不同的规范必须共存，而不是后者取代前者。"""
        policy = WritePolicy()
        emb = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        existing = [
            make_vec_item(
                "a",
                [1.0, 0.0],
                kind=MemoryKind.PROCEDURAL,
                subject="团队",
                predicate="约束",
                object="长篇大论",
            )
        ]
        d = policy.decide(
            draft(
                "b",
                MemoryKind.PROCEDURAL,
                subject="团队",
                predicate="约束",
                object="有回滚开关",
            ),
            existing,
            embed=emb.embed_one,
        )
        assert d.action is WriteAction.ADD

    def test_preferences_accumulate_not_supersede(self):
        policy = WritePolicy()
        emb = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        existing = [
            make_vec_item(
                "a", [1.0, 0.0], kind=MemoryKind.PREFERENCE, subject="用户", predicate="偏好"
            )
        ]
        d = policy.decide(
            draft("b", MemoryKind.PREFERENCE, subject="用户", predicate="偏好"),
            existing,
            embed=emb.embed_one,
        )
        assert d.action is WriteAction.ADD

    def test_single_valued_kinds_default(self):
        assert SINGLE_VALUED_KINDS == frozenset({MemoryKind.SEMANTIC})

    def test_low_importance_is_dropped(self):
        policy = WritePolicy(min_importance=0.45)
        d = policy.decide(
            draft("好的吧", MemoryKind.EPISODIC),
            [],
            embed=HashingEmbedder(dimension=16).embed_one,
        )
        assert d.action is WriteAction.NOOP
        assert d.reason.startswith("low_importance")

    def test_uses_item_embedding_when_present(self):
        """已有条目的向量应当被复用，而不是每次重新算一遍。"""
        calls: list[str] = []

        def counting_embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0]

        policy = WritePolicy()
        existing = [make_vec_item("cached", [1.0, 0.0])]
        policy.decide(draft("query"), existing, embed=counting_embed)
        assert calls == ["query"]  # 只为新草稿算了一次

    def test_similarity_recorded_on_add(self):
        policy = WritePolicy()
        emb = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        existing = [make_vec_item("a", [1.0, 0.0])]
        d = policy.decide(draft("b"), existing, embed=emb.embed_one)
        assert d.action is WriteAction.ADD
        assert d.similarity == pytest.approx(0.0, abs=1e-9)


class TestDecay:
    def test_decays_with_age(self):
        policy = WritePolicy()
        item = make_vec_item("x", [1.0], kind=MemoryKind.EPISODIC, importance=0.4)
        assert policy.decayed_importance(item, START) == pytest.approx(0.4)
        assert policy.decayed_importance(item, START + 7 * DAY) == pytest.approx(0.2)

    def test_access_dampens_decay(self):
        policy = WritePolicy()
        cold = make_vec_item("cold", [1.0], kind=MemoryKind.EPISODIC, importance=0.4)
        hot = make_vec_item(
            "hot", [1.0], kind=MemoryKind.EPISODIC, importance=0.4, access_count=20
        )
        now = START + 14 * DAY
        assert policy.decayed_importance(hot, now) > policy.decayed_importance(cold, now)

    def test_half_life_per_kind_ordering(self):
        policy = WritePolicy()
        assert policy.half_life_for(MemoryKind.PREFERENCE) > policy.half_life_for(
            MemoryKind.EPISODIC
        )
        assert set(DEFAULT_HALF_LIFE_S) == set(MemoryKind)

    def test_unknown_kind_gets_default_half_life(self):
        policy = WritePolicy(half_life_s={})
        assert policy.half_life_for(MemoryKind.SEMANTIC) == 30 * DAY


class TestForgetting:
    def test_ttl_expiry(self):
        policy = WritePolicy()
        item = make_vec_item(
            "x", [1.0], kind=MemoryKind.SEMANTIC, ttl_s=60.0, importance=0.9
        )
        assert not policy.should_forget(item, START + 30)
        assert policy.should_forget(item, START + 61)

    def test_preference_never_forgotten_by_decay(self):
        policy = WritePolicy()
        item = make_vec_item("x", [1.0], kind=MemoryKind.PREFERENCE, importance=0.05)
        assert not policy.should_forget(item, START + 3650 * DAY)

    def test_old_episodic_forgotten(self):
        policy = WritePolicy()
        item = make_vec_item("x", [1.0], kind=MemoryKind.EPISODIC, importance=0.4)
        assert not policy.should_forget(item, START + 1 * DAY)
        assert policy.should_forget(item, START + 8 * DAY)

    def test_high_importance_semantic_survives_long(self):
        policy = WritePolicy()
        item = make_vec_item("x", [1.0], kind=MemoryKind.SEMANTIC, importance=1.0)
        assert not policy.should_forget(item, START + 30 * DAY)


class TestConfigValidation:
    def test_merge_must_be_below_duplicate(self):
        with pytest.raises(PolicyError):
            WritePolicy(duplicate_threshold=0.8, merge_threshold=0.9)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_min_importance_range(self, value):
        with pytest.raises(PolicyError):
            WritePolicy(min_importance=value)
