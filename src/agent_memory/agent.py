"""门面：把记忆层接进一个"记忆增强"的 Agent 循环。

这一段是全文最想强调的地方——**记忆不是"往里灌数据"，而是两个时机**：

* **回合开始前**：根据用户这句话召回相关记忆，组装成带预算的上下文块；
* **回合结束后**：把这一轮发言交给抽取器，让策略决定记什么、覆盖什么。

中间那个"模型回复"的环节，在这个 Demo 里用一个确定性的 :class:`ScriptedResponder`
替代——它不需要 API Key，但**接口和真实 LLM 完全一致**（收 prompt、出文本）。
换成真的 OpenAI / 本地模型只需要把 responder 换掉。
"""

from __future__ import annotations

import re
from typing import Callable, Protocol, Sequence, runtime_checkable

from .manager import MemoryManager, RecallBundle
from .models import MemoryKind, WriteDecision

__all__ = [
    "Responder",
    "ScriptedResponder",
    "AgentReply",
    "MemoryAugmentedAgent",
    "build_prompt",
]


@runtime_checkable
class Responder(Protocol):
    """任何"收 prompt、出文本"的东西。真实 LLM 客户端天然满足这个协议。"""

    def __call__(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class ScriptedResponder:
    """离线演示用的确定性"模型"。

    它做两件事：从 prompt 里读出系统已知的偏好，然后按这些偏好组织回答。
    这样 Demo 的输出能直观证明"记忆真的影响了回答"，而不是打印一堆内部状态。
    """

    def __init__(self, *, salary_style: bool = True) -> None:
        self.salary_style = salary_style
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        facts = _extract_section(prompt, "必须遵守的约定")
        background = _extract_section(prompt, "相关背景")
        lines: list[str] = []

        joined = facts + background
        if re.search(r"简短|简洁|不要长|短一点", joined):
            lines.append("好的。结论先行：建议先做灰度发布。")
        else:
            lines.append(
                "好的，我把方案展开讲一下：建议先做灰度发布，"
                "观察 24 小时错误率，再逐步放量。"
            )
        if re.search(r"中文|不要英文|用中文", joined):
            lines.append("（已按你的偏好使用中文回答。）")
        if re.search(r"生产|线上|灰度|回滚|发布", joined):
            lines.append("已预先考虑线上风险：回滚开关与监控告警一并带上。")
        if not lines:
            lines.append("（这是我第一次和你聊这个话题，没有历史记忆可用。）")
        return "\n".join(lines)


def _extract_section(prompt: str, header: str) -> str:
    """从组装好的 prompt 里抠出某个小节，供 ScriptedResponder 判断偏好。

    真实系统里这一步由模型自己完成——这里只是让离线 Demo 有可验证的行为。
    """
    m = re.search(rf"{re.escape(header)}\n(.*?)(?=\n###|\Z)", prompt, re.S)
    return m.group(1) if m else ""


def build_prompt(
    user_text: str,
    memory_block: str,
    *,
    system_prompt: str = "你是一个资深技术顾问。回答时充分利用已知的用户偏好与历史。",
    user_label: str = "用户",
) -> str:
    """把记忆块拼进 system prompt。

    注意记忆块放在 **system** 而不是 user 消息里——它描述的是"这个人是谁"，
    属于长期设定，不该和这一轮的具体问题混在一起。
    """
    parts = [system_prompt]
    if memory_block.strip():
        parts.append(memory_block.strip())
    return "\n\n".join(parts) + f"\n\n{user_label}：{user_text}"


class AgentReply:
    __slots__ = ("user_text", "text", "context", "hit_ids", "writes")

    def __init__(
        self,
        user_text: str,
        text: str,
        context: str,
        hit_ids: list[str],
        writes: list[WriteDecision],
    ) -> None:
        self.user_text = user_text
        self.text = text
        self.context = context
        self.hit_ids = hit_ids
        self.writes = writes

    @property
    def wrote(self) -> list[WriteDecision]:
        return [w for w in self.writes if w.writes]

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"AgentReply(writes={len(self.wrote)}, hits={len(self.hit_ids)})"


class MemoryAugmentedAgent:
    """一个最小的"有记忆的 Agent"。"""

    def __init__(
        self,
        memory: MemoryManager,
        responder: Responder,
        *,
        system_prompt: str | None = None,
        top_k: int = 6,
        context_max_tokens: int = 260,
    ) -> None:
        self.memory = memory
        self.responder = responder
        self.system_prompt = system_prompt or (
            "你是一个资深技术顾问。回答时充分利用已知的用户偏好与历史。"
        )
        self.top_k = top_k
        self.context_max_tokens = context_max_tokens

    def chat(self, user_text: str, *, turn_id: str | None = None) -> AgentReply:
        # ---- 1. 回合前：召回 + 组装 ----
        bundle = self.memory.recall(
            user_text, top_k=self.top_k, max_tokens=self.context_max_tokens
        )
        prompt = build_prompt(
            user_text, bundle.text, system_prompt=self.system_prompt
        )

        # ---- 2. 生成 ----
        answer = self.responder(prompt) if callable(self.responder) else str(self.responder)

        # ---- 3. 回合后：抽取 + 写入 ----
        writes = self.memory.observe(user_text, role="user", turn_id=turn_id)
        return AgentReply(user_text, answer, bundle.text, bundle.ids, writes)

    def run(self, turns: Sequence[str]) -> list[AgentReply]:
        return [self.chat(t, turn_id=f"t{i}") for i, t in enumerate(turns)]


def summarize_writes(decisions: Sequence[WriteDecision]) -> dict[str, int]:
    """给 Demo 用的小统计。"""
    out: dict[str, int] = {"add": 0, "supersede": 0, "merge": 0, "noop": 0}
    for d in decisions:
        out[d.action.value] = out.get(d.action.value, 0) + 1
    return out


def kind_breakdown(manager: MemoryManager) -> dict[str, int]:
    stats = manager.stats()
    return {
        kind.value: stats.by_kind.get(kind.value, 0) for kind in MemoryKind
    }
