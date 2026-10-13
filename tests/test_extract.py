from __future__ import annotations

import pytest

from agent_memory.extract import (
    SLOT_NOUNS,
    RuleBasedExtractor,
    Turn,
    derive_slot,
    split_sentences,
)
from agent_memory.models import MemoryKind


@pytest.fixture()
def ex() -> RuleBasedExtractor:
    return RuleBasedExtractor()


class TestTurn:
    def test_valid_roles(self):
        for role in ("user", "assistant", "system", "tool"):
            assert Turn(role, "hi").role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError):
            Turn("robot", "hi")

    def test_turn_is_hashable_and_frozen(self):
        t = Turn("user", "hi", "t1")
        assert t.turn_id == "t1"
        with pytest.raises(Exception):
            t.role = "assistant"  # type: ignore[misc]


class TestSplitSentences:
    def test_splits_on_cjk_and_ascii_punctuation(self):
        got = split_sentences("第一句。第二句！第三句？第四句;第五句\n第六句")
        assert got == ["第一句", "第二句", "第三句", "第四句", "第五句", "第六句"]

    def test_drops_trailing_punctuation(self):
        assert split_sentences("订单库换掉了。") == ["订单库换掉了"]

    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_no_terminator(self):
        assert split_sentences("没有句号的一句话") == ["没有句号的一句话"]


class TestSlotDerivation:
    def test_known_nouns(self):
        assert derive_slot("订单库不用 MySQL 了") == "订单库"
        assert derive_slot("缓存现在是 Redis") == "缓存"

    def test_fallback(self):
        assert derive_slot("总共也没提具体部件") == "技术栈"

    def test_slot_nouns_are_nonempty(self):
        assert len(SLOT_NOUNS) >= 8


class TestPreferenceExtraction:
    def test_expression_preference(self, ex):
        [d] = ex.extract("记住，回答尽量简短，不要长篇大论。")
        assert d.kind is MemoryKind.PREFERENCE
        assert d.subject == "用户"
        assert d.object == "简短"
        # "记住" 是显式标记 → 置信度更高
        assert d.confidence == pytest.approx(0.9)

    def test_prefer_pattern(self, ex):
        [d] = ex.extract("我更喜欢用中文和我说话，不要太啰嗦。")
        assert d.kind is MemoryKind.PREFERENCE
        assert d.predicate == "偏好"
        assert d.object == "中文和我说话"

    def test_dislike_pattern(self, ex):
        [d] = ex.extract("我不喜欢被反复追问背景。")
        assert d.predicate == "反感"
        assert d.object == "被反复追问背景"

    def test_chitchat_yields_nothing(self, ex):
        assert ex.extract("嗯嗯") == []
        assert ex.extract("好的") == []


class TestFactExtraction:
    def test_name(self, ex):
        [d] = ex.extract("我叫周彪，是做后端的。")
        assert d.kind is MemoryKind.SEMANTIC
        assert d.predicate == "姓名"
        assert d.object == "周彪"

    def test_role(self, ex):
        [d] = ex.extract("我是后端工程师，主要写 Java。")
        assert d.predicate == "身份"
        assert d.object == "后端工程师"

    def test_attribute(self, ex):
        [d] = ex.extract("我的时区是 UTC+8。")
        assert d.predicate == "属性"
        assert d.object == "UTC+8"

    def test_plain_stack_choice(self, ex):
        [d] = ex.extract("我们项目现在用 MySQL 8 做订单库。")
        assert d.subject == "订单库"  # 槽位来自领域名词，而非句法主语
        assert d.predicate == "技术选型"
        assert d.object == "MySQL 8"

    def test_migration(self, ex):
        [d] = ex.extract("我们决定把订单库换成 PostgreSQL 16。")
        assert d.subject == "订单库"
        assert d.object == "PostgreSQL 16"

    def test_replacement_negative(self, ex):
        """'不用 MySQL，改用 PostgreSQL' 里的 object 必须是新值。"""
        [d] = ex.extract("订单库不用 MySQL 了，改用 PostgreSQL。")
        assert d.object == "PostgreSQL"

    def test_negative_lookbehind_blocks_false_positive(self, ex):
        # "不用 MySQL" 不该被当成一次 "用 MySQL" 的选型
        drafts = ex.extract("我们不用 Oracle 了。")
        assert drafts == []


class TestRuleExtraction:
    def test_constraint(self, ex):
        [d] = ex.extract("不要使用表情符号，正式场合要严肃。")
        assert d.kind is MemoryKind.PROCEDURAL
        assert d.predicate == "约束"
        assert d.object == "使用表情符号"  # 在子句边界断开

    def test_future_rule_wins_by_leftmost_match(self, ex):
        """'以后都走灰度发布，上线前必须有回滚开关' 同时命中 future_rule 与 constraint，
        整句主干是前者，取最靠左的匹配。"""
        [d] = ex.extract("以后都走灰度发布，上线前必须有回滚开关。")
        assert d.kind is MemoryKind.PROCEDURAL
        assert d.predicate == "流程"
        assert d.object == "走灰度发布"

    def test_episodic(self, ex):
        [d] = ex.extract("今天上线了新的对账服务，遇到了连接池打满的问题。")
        assert d.kind is MemoryKind.EPISODIC
        assert d.predicate == "经历"

    def test_episodic_hint_fallback(self, ex):
        # 没有任何句式命中，但出现了"故障"这类事件词 → 兜底归为情景记忆
        [d] = ex.extract("线上故障持续了十分钟。")
        assert d.kind is MemoryKind.EPISODIC
        assert d.tags == ["episodic_hint"]

    def test_text_is_full_sentence(self, ex):
        # 句末标点会被 split_sentences 剥掉，但整句内容保留
        [d] = ex.extract("我们项目现在用 MySQL 8 做订单库。")
        assert d.text == "我们项目现在用 MySQL 8 做订单库"

    def test_metadata_traces_pattern(self, ex):
        [d] = ex.extract("我们项目现在用 MySQL 8 做订单库。")
        assert d.metadata["pattern"] == "stack_choice"
        assert d.metadata["extractor"] == "rule"
        assert d.tags == ["stack_choice"]

    def test_min_chars_filters_fragments(self):
        ex = RuleBasedExtractor(min_chars=20)
        assert ex.extract("我叫周彪。") == []

    def test_max_per_turn_caps_output(self):
        ex = RuleBasedExtractor(max_per_turn=2)
        text = "我叫周彪。我的时区是 UTC+8。我是后端工程师。我喜欢简短回答。"
        assert len(ex.extract(text)) == 2

    def test_duplicate_sentences_deduped(self, ex):
        drafts = ex.extract("我叫周彪。我叫周彪。")
        assert len(drafts) == 1


class TestTurnBatch:
    def test_only_user_turns_extracted(self, ex):
        turns = [
            Turn("user", "我叫周彪。", "t0"),
            Turn("assistant", "我叫小助手，请多关照。", "t1"),
            Turn("user", "我喜欢简短回答。", "t2"),
        ]
        drafts = ex.extract_turns(turns)
        assert len(drafts) == 2
        assert all("小助手" not in d.text for d in drafts)

    def test_turn_id_propagated(self, ex):
        turns = [Turn("user", "我叫周彪。", "t7")]
        [d] = ex.extract_turns(turns)
        assert d.source_turns == ["t7"]

    def test_auto_turn_id_when_missing(self, ex):
        turns = [Turn("user", "我叫周彪。", None)]
        [d] = ex.extract_turns(turns)
        assert d.source_turns == ["turn-0"]

    def test_custom_roles(self):
        ex = RuleBasedExtractor(roles=("user", "assistant"))
        turns = [Turn("assistant", "系统约定：不要输出表情符号。", "t0")]
        assert len(ex.extract_turns(turns)) == 1

    def test_extract_sets_turn_id_directly(self, ex):
        [d] = ex.extract("我叫周彪。", turn_id="abc")
        assert d.source_turns == ["abc"]
