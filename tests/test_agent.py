from __future__ import annotations

import pytest

from agent_memory.agent import (
    AgentReply,
    MemoryAugmentedAgent,
    ScriptedResponder,
    build_prompt,
    kind_breakdown,
    summarize_writes,
)
from agent_memory.models import MemoryKind, WriteAction

from conftest import START


class TestBuildPrompt:
    def test_memory_block_placed_before_user_turn(self):
        prompt = build_prompt("灰度怎么落地？", "## 关于用户的已知记忆\n- 回答要简短")
        assert prompt.index("关于用户的已知记忆") < prompt.index("灰度怎么落地？")
        assert prompt.startswith("你是一个")

    def test_empty_memory_block_skipped(self):
        prompt = build_prompt("你好", "   ")
        assert prompt.count("\n\n") == 1
        assert "关于用户的已知记忆" not in prompt

    def test_custom_system_prompt(self):
        prompt = build_prompt("x", "", system_prompt="你是运维助手")
        assert prompt.startswith("你是运维助手")

    def test_custom_user_label(self):
        assert "提问：" in build_prompt("x", "", user_label="提问")


class TestScriptedResponder:
    def test_short_style_when_preference_present(self):
        out = ScriptedResponder()("### 必须遵守的约定\n- 回答尽量简短\n")
        assert "结论先行" in out

    def test_verbose_by_default(self):
        out = ScriptedResponder()("没有任何记忆")
        assert "展开讲一下" in out

    def test_notes_chinese_preference(self):
        out = ScriptedResponder()("### 必须遵守的约定\n- 我更喜欢用中文\n")
        assert "中文" in out

    def test_records_calls(self):
        responder = ScriptedResponder()
        responder("a")
        responder("b")
        assert responder.calls == ["a", "b"]

    def test_mentions_online_risk_when_constraint_present(self):
        out = ScriptedResponder()("### 必须遵守的约定\n- 上线前必须有回滚开关\n")
        assert "线上风险" in out


class TestSummarizeWrites:
    def test_counts(self, manager):
        manager.observe("记住，回答尽量简短。")
        manager.observe("我们项目现在用 MySQL 8 做订单库。")
        counts = summarize_writes(manager.decisions)
        assert counts["add"] == 2
        assert counts["supersede"] == 0

    def test_empty(self):
        assert summarize_writes([]) == {"add": 0, "supersede": 0, "merge": 0, "noop": 0}


class TestKindBreakdown:
    def test_all_kinds_present(self, manager):
        manager.observe("我叫周彪。")
        breakdown = kind_breakdown(manager)
        assert set(breakdown) == {k.value for k in MemoryKind}
        assert breakdown["semantic"] == 1
        assert breakdown["preference"] == 0


class TestMemoryAugmentedAgent:
    def test_first_turn_has_no_memory_context(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder(), context_max_tokens=300)
        reply = agent.chat("灰度发布怎么落地？", turn_id="t0")
        assert reply.context == ""
        assert reply.hit_ids == []

    def test_memory_written_from_user_turn(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder(), context_max_tokens=300)
        reply = agent.chat("记住，回答尽量简短，不要长篇大论。", turn_id="t0")
        assert [w.action for w in reply.wrote] == [WriteAction.ADD]
        assert len(manager.memories()) == 1

    def test_second_turn_uses_first_turn_memory(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder(), context_max_tokens=300)
        agent.chat("记住，回答尽量简短，不要长篇大论。", turn_id="t0")
        reply = agent.chat("灰度发布怎么落地？", turn_id="t1")
        assert "简短" in reply.context
        assert "结论先行" in reply.text

    def test_style_changes_when_preference_changes(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder(), context_max_tokens=300)
        a1 = agent.chat("出来聊聊发布流程。", turn_id="t0")
        a2 = agent.chat("记住，回答尽量简短。", turn_id="t1")
        a3 = agent.chat("再说一次发布流程。", turn_id="t2")
        assert "展开讲一下" in a1.text
        assert "结论先行" in a3.text and "展开讲一下" not in a3.text
        assert a2.wrote

    def test_run_helper(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder())
        replies = agent.run(["我叫周彪。", "我的时区是 UTC+8。"])
        assert len(replies) == 2
        assert all(isinstance(r, AgentReply) for r in replies)
        assert len(manager.memories()) == 2

    def test_reply_repr(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder())
        assert "AgentReply" in repr(agent.chat("我叫周彪。"))

    def test_budget_limits_context(self, manager):
        agent = MemoryAugmentedAgent(manager, ScriptedResponder(), context_max_tokens=30)
        for text in ("记住，回答尽量简短。", "以后都走灰度发布，必须有回滚开关。"):
            manager.observe(text)
        reply = agent.chat("发布流程？")
        assert len(reply.context) < 200
