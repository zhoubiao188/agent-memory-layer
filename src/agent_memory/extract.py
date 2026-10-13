"""从对话里抽取"值得记住的东西"。

**为什么是规则而不是直接丢给 LLM**：

1. 抽取这一步跑在每一轮对话上，是**热路径**。加一次 LLM 调用 = 每轮多几百毫秒 + 每轮多一份钱；
2. 规则是**可解释**的。白天出了问题，你能一眼说出"这条为什么被记住"，而 LLM 抽取只能再问一次 LLM；
3. 规则是**可测试**的。它有确定的输入输出，所以能写单元测试；LLM 抽取只能做端到端评估。

但规则一定会漏。所以这里把抽取做成了 :class:`ExtractionStrategy` 协议——
生产里你可以用"规则打底 + 只在句子包含不确定语义时调用小模型"的混合策略，
两者的产物是同一种 :class:`~agent_memory.models.MemoryDraft`，下游一行都不用改。

### 一个必须讲清楚的取舍：`subject` 是"冲突槽位"，不是句法主语

中文句子大量省略主语（"订单库换成 PostgreSQL 了"），硬去做句法分析既不划算也不可靠。
所以这里反过来：**从句子里的领域名词派生一个稳定的槽位键**，
让"同一件事的不同说法"能落到同一个 ``(subject, predicate)`` 上，
从而触发"事实取代"而不是"并存两条互相矛盾的记忆"。

这也是规则抽取唯一需要人工维护的地方——但它是**可枚举、可评审**的，
比一个说不清为什么的相似度阈值可靠得多。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .models import MemoryDraft, MemoryKind

__all__ = [
    "Turn",
    "ExtractionStrategy",
    "RuleBasedExtractor",
    "split_sentences",
    "SLOT_NOUNS",
]


@dataclass(frozen=True)
class Turn:
    """一段对话轮次。``turn_id`` 用于回溯"这条记忆是哪轮说的"。"""

    role: str
    text: str
    turn_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unknown role: {self.role}")


@runtime_checkable
class ExtractionStrategy(Protocol):
    def extract(self, text: str, *, turn_id: str | None = None) -> list[MemoryDraft]:
        ...  # pragma: no cover - protocol


#: 领域名词表：用于把"同一件事"的不同说法归到同一个冲突槽位。
#: 这是本文件里唯一需要按业务维护的数据，也是最该做代码评审的地方。
SLOT_NOUNS: tuple[str, ...] = (
    "订单库",
    "用户库",
    "数据库",
    "缓存",
    "消息队列",
    "网关",
    "框架",
    "中间件",
    "运行时",
    "编程语言",
    "部署方式",
    "存储",
)

#: 技术名一般以拉丁字母开头，用这个约束把"用 MySQL"和"用中文回答"区分开。
_TECH_VALUE = r"([A-Za-z][A-Za-z0-9\.\-_]{1,20}(?:\s?[A-Za-z0-9\.\-_]+){0,3})"
#: 子句边界。抽取 object 时在这里断开，避免把下一个分句也吞进来。
_CLAUSE_END = r"[^，。；,;！!？?]"


def _value_group(max_len: int) -> str:
    """带捕获组的"到子句边界为止"的值。

    这里必须**显式带括号**：抽取器统一用 ``m.groups()[-1]`` 取值，
    如果值段不是捕获组，正则明明匹配了却拿不到 object，
    表现为"规则写了但一条都抽不出来"——一个非常容易看走眼的 bug。
    """
    return f"({_CLAUSE_END}{{1,{max_len}}})"


# (名称, 类型, 正则, subject, predicate)。subject 为 ``$slot`` 时按句子派生槽位。
_PATTERNS: tuple[tuple[str, MemoryKind, re.Pattern[str], str, str], ...] = (
    (
        "expression",
        MemoryKind.PREFERENCE,
        re.compile(
            r"(?:回答|回复|输出|讲解|解释|写代码|文档)(?:尽量|要|请|都)?"
            r"(简短|简洁|详细|通俗|直接|带例子|说人话|中文|英文|专业)"
        ),
        "用户",
        "表达偏好",
    ),
    (
        "prefer",
        MemoryKind.PREFERENCE,
        re.compile(
            r"我(?:更|最|比较)?(?:喜欢|偏好|倾向(?:于)?|习惯)(?:用|使用|吃|看|听|读)?"
            + _value_group(40)
        ),
        "用户",
        "偏好",
    ),
    (
        "dislike",
        MemoryKind.PREFERENCE,
        re.compile(
            r"我(?:很|比较|特别)?(?:不喜欢|讨厌|受不了|反感)" + _value_group(40)
        ),
        "用户",
        "反感",
    ),
    (
        "self_name",
        MemoryKind.SEMANTIC,
        re.compile(r"(?:我叫|我的名字叫|我的名字是)([\u4e00-\u9fffA-Za-z]{1,12})"),
        "用户",
        "姓名",
    ),
    (
        "self_role",
        MemoryKind.SEMANTIC,
        re.compile(
            r"我是(?:一[个名位])?([\u4e00-\u9fffA-Za-z]{1,12}?"
            r"(?:工程师|开发|架构师|经理|负责人|运维|产品经理|设计师|测试|老师|学生))"
        ),
        "用户",
        "身份",
    ),
    (
        "attribute",
        MemoryKind.SEMANTIC,
        re.compile(
            r"我的(?:[\u4e00-\u9fffA-Za-z]{1,12}?)(?:是|为)" + _value_group(40)
        ),
        "用户",
        "属性",
    ),
    (
        "migrate",
        MemoryKind.SEMANTIC,
        re.compile(
            r"(?:把|将)(?:[\u4e00-\u9fff]{1,10}?)(?:换成|改为|改成|迁移到|替换为|调整成)\s*"
            + _TECH_VALUE
        ),
        "$slot",
        "技术选型",
    ),
    (
        "replacement",
        MemoryKind.SEMANTIC,
        re.compile(
            r"(?:不是|不用|不要)\s*(?:[A-Za-z][\w\.\-]{1,20})[，,]?\s*"
            r"(?:而是|要换成|改用|换成|改成)\s*" + _TECH_VALUE
        ),
        "$slot",
        "技术选型",
    ),
    (
        "stack_choice",
        # 负向后顾：避免把"不用 MySQL"里的"用 MySQL"识别成一次技术选型。
        MemoryKind.SEMANTIC,
        re.compile(
            r"(?:决定|打算|准备|已经)?(?<![不没别未])"
            r"(?:用|使用|采用|选|选型为|换成|迁移到|改成)\s*" + _TECH_VALUE
        ),
        "$slot",
        "技术选型",
    ),
    (
        "constraint",
        MemoryKind.PROCEDURAL,
        re.compile(r"(?:不要|别|禁止|不准|不能再|务必|必须|一定要)" + _value_group(40)),
        "团队",
        "约束",
    ),
    (
        "future_rule",
        MemoryKind.PROCEDURAL,
        re.compile(r"(?:以后|今后|下次|后续)(?:都|一律|统一)?" + _value_group(40)),
        "团队",
        "流程",
    ),
    (
        "episodic",
        MemoryKind.EPISODIC,
        re.compile(
            r"(?:刚刚|今天|昨天|已经)?"
            r"(?:上线了|发布了|部署了|回滚了|修复了|完成了|遇到了|踩了)([^。！？!?]{2,50})"
        ),
        "事件",
        "经历",
    ),
)

#: 出现这些词说明这句话**在陈述一件已发生的事**，而不是在闲聊。
_EPISODIC_HINT = re.compile(r"上线|发布|部署|回滚|修复|完成|遇到|踩坑|故障|报错|挂了")
#: 明确的"请你记住"标记：命中则抬高置信度，并把整句作为记忆原文。
_EXPLICIT_REMEMBER = re.compile(r"记住|记一下|记下来|帮我记|请记得|别忘了")

_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_TRAILING = re.compile(r"[\s。！？!?；;，,、.]+$")


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留非空片段。"""
    out: list[str] = []
    for chunk in _SPLIT.split(text or ""):
        piece = _TRAILING.sub("", chunk.strip())
        if piece:
            out.append(piece)
    return out


def derive_slot(sentence: str) -> str:
    """从句子里的领域名词派生冲突槽位。找不到就落到通用槽"技术栈"。"""
    for noun in SLOT_NOUNS:
        if noun in sentence:
            return noun
    return "技术栈"


class RuleBasedExtractor:
    """确定性规则抽取器。

    ``roles`` 限制只从哪些角色的发言里抽取——默认只抽用户的，
    因为"助手说了一句话"不等于"用户认可了这句话"。
    """

    def __init__(
        self,
        *,
        min_chars: int = 4,
        roles: Sequence[str] = ("user",),
        max_per_turn: int = 6,
    ) -> None:
        self.min_chars = min_chars
        self.roles = tuple(roles)
        self.max_per_turn = max_per_turn

    @property
    def name(self) -> str:
        return "rule"

    # ---------- 单句 ----------

    def _match_sentence(self, sentence: str) -> list[MemoryDraft]:
        """一句最多产出一条候选。

        多个模式同时命中时取**最靠左**的那个（并列则按模式声明顺序）。
        理由是：中文里"以后都走灰度发布，上线前必须有回滚开关"同时命中
        ``future_rule`` 和 ``constraint``，但整句的主干是前者。
        真正的多意图切分交给 LLM 策略——这正是把抽取做成可替换接口的原因。
        """
        explicit = bool(_EXPLICIT_REMEMBER.search(sentence))
        best: tuple[int, int] | None = None
        chosen: tuple[str, MemoryKind, re.Match[str], str, str] | None = None

        for order, (name, kind, pattern, subject, predicate) in enumerate(_PATTERNS):
            m = pattern.search(sentence)
            if not m:
                continue
            key = (m.start(), order)
            if best is None or key < best:
                best = key
                chosen = (name, kind, m, subject, predicate)

        if chosen is None:
            if _EPISODIC_HINT.search(sentence) and len(sentence) >= self.min_chars:
                return [
                    MemoryDraft(
                        kind=MemoryKind.EPISODIC,
                        text=sentence,
                        subject="事件",
                        predicate="经历",
                        object=sentence,
                        confidence=0.6,
                        tags=["episodic_hint"],
                        metadata={"pattern": "episodic_hint", "extractor": self.name},
                    )
                ]
            return []

        name, kind, m, subject_spec, predicate = chosen
        groups = [g for g in m.groups() if g]
        if not groups:
            return []
        obj = groups[-1].strip("的　 ")
        if not obj:
            return []
        subject = derive_slot(sentence) if subject_spec == "$slot" else subject_spec
        # 记忆原文用**整句**，而不是正则匹配到的那一小段。
        # 结构化三元组已经落在 subject/predicate/object 上，原文保留整句能让
        # BM25 召回受益（"订单库"这个词在匹配片段里可能根本没出现），
        # 也让人工排查时看得懂上下文。
        text = sentence
        return [
            MemoryDraft(
                kind=kind,
                text=text,
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.9 if explicit else 0.72,
                tags=[name],
                metadata={"pattern": name, "extractor": self.name},
            )
        ]

    # ---------- 对外 ----------

    def extract(self, text: str, *, turn_id: str | None = None) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        seen: set[str] = set()
        for sentence in split_sentences(text):
            if len(sentence) < self.min_chars:
                continue
            for draft in self._match_sentence(sentence):
                key = draft.text.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                if turn_id:
                    draft.source_turns = [turn_id]
                drafts.append(draft)
                if len(drafts) >= self.max_per_turn:
                    return drafts
        return drafts

    def extract_turns(self, turns: Iterable[Turn]) -> list[MemoryDraft]:
        """从多轮对话里抽取，自动跳过非目标角色。"""
        out: list[MemoryDraft] = []
        for i, turn in enumerate(turns):
            if turn.role not in self.roles:
                continue
            tid = turn.turn_id or f"turn-{i}"
            out.extend(self.extract(turn.text, turn_id=tid))
        return out
