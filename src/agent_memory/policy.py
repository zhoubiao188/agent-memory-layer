"""写入策略：决定"这条要不要记、记成什么、要不要覆盖旧的"。

这是整个记忆系统里**最容易被跳过、也最容易出事故**的一层。
绝大多数 Demo 的做法是"把每轮对话都塞进向量库"，跑三天之后你会发现：

* 同一个偏好存了 12 遍（用户说了 12 次"我喜欢简短回答"）；
* 用户从"要用 MySQL"改成"要换 PostgreSQL"之后，两条事实同时存在于库里，
  检索时看运气返回哪一条；
* 库里 90% 是"好的""谢谢""嗯"这类零价值文本。

所以 :class:`WritePolicy` 做三件事：**打分（值不值得记）→ 去重（记过没有）→ 取代（是不是旧事实）**。
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from .embedder import cosine
from .errors import PolicyError
from .models import (
    MemoryDraft,
    MemoryItem,
    MemoryKind,
    WriteAction,
    WriteDecision,
)

__all__ = ["WritePolicy", "DEFAULT_HALF_LIFE_S", "SINGLE_VALUED_KINDS"]

#: **只有单值事实才谈得上"取代"。**
#:
#: 这一条是我踩过坑之后加的：早期的实现对所有类型都做 ``(subject, predicate)``
#: 冲突判定，结果"不要长篇大论"被后来的"必须有回滚开关"取代掉了——
#: 因为两条都被抽成 ``(团队, 约束)``，而 object 不同。
#: 但约束和偏好是**可累积**的：用户说了 5 条规范，就该记住 5 条。
#: 只有"某个槽位当前等于什么"这种单值事实，新值才有资格让旧值失效。
SINGLE_VALUED_KINDS: frozenset[MemoryKind] = frozenset({MemoryKind.SEMANTIC})

#: 各类型记忆的"半衰期"。情景记忆天然短命，偏好和规范应该活很久。
DEFAULT_HALF_LIFE_S: Mapping[MemoryKind, float] = {
    MemoryKind.EPISODIC: 7 * 86_400.0,
    MemoryKind.SEMANTIC: 90 * 86_400.0,
    MemoryKind.PROCEDURAL: 120 * 86_400.0,
    MemoryKind.PREFERENCE: 180 * 86_400.0,
}

#: 类型基线分：偏好 > 规范 > 事实 > 情景。
_BASE_IMPORTANCE: Mapping[MemoryKind, float] = {
    MemoryKind.PREFERENCE: 0.70,
    MemoryKind.PROCEDURAL: 0.66,
    MemoryKind.SEMANTIC: 0.56,
    MemoryKind.EPISODIC: 0.40,
}

#: (正则, 加权)。加权值经过调参，保证"用户明确要求记住"必然高过任何日常语句。
_SIGNALS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"记住|记一下|记下来|请记得|帮我记"), 0.26),
    (re.compile(r"务必|必须|一定要|千万|永远|以后都|今后都"), 0.17),
    (re.compile(r"不要|别|禁止|不能再|不准|拒绝"), 0.15),
    (re.compile(r"我(?:更|最)?(?:喜欢|偏好|习惯|倾向|讨厌|不喜欢|希望)"), 0.14),
    (re.compile(r"重要|关键|核心|红线|底线"), 0.10),
    (re.compile(r"生产|线上|正式环境|事故|故障|宕机|回滚"), 0.08),
    (re.compile(r"我的|我叫|我是"), 0.06),
    (re.compile(r"\d"), 0.04),
)

#: 纯寒暄/无信息量的整句，直接丢弃。**只匹配"整句就是它"的情况**，
#: 避免把"好的，那我们把超时改成 3 秒"这种有信息量的句子误杀。
_CHITCHAT = re.compile(
    r"^(?:好(?:的)?|嗯+|哦+|啊+|是的?|对|行|可以|收到|谢谢|感谢|辛苦|哈哈+|"
    r"ok|okay|thanks|thank you|got it|sure|cool|nice|hi|hello|hey)[\s。！!,~]*$",
    re.IGNORECASE,
)


def _norm_fact(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[\s,，。.、;；:：\"'“”‘’()（）]", "", value).lower()


class WritePolicy:
    """写入决策器。

    参数
    ----
    duplicate_threshold:
        余弦相似度高于此值视为"同一件事"。定得高（0.95）是因为
        宁可重复也不要错杀——漏记的代价远大于多存一条。
    merge_threshold:
        介于 merge 与 duplicate 之间时，把新表述**合并进旧的**（通常新表述更完整）。
    min_importance:
        低于此分直接丢弃，不入库。
    """

    def __init__(
        self,
        *,
        duplicate_threshold: float = 0.95,
        merge_threshold: float = 0.86,
        min_importance: float = 0.22,
        half_life_s: Mapping[MemoryKind, float] | None = None,
        single_valued_kinds: frozenset[MemoryKind] | None = None,
    ) -> None:
        if not 0.0 < min_importance < 1.0:
            raise PolicyError(f"min_importance out of range: {min_importance}")
        if merge_threshold >= duplicate_threshold:
            raise PolicyError("merge_threshold must be < duplicate_threshold")
        self.duplicate_threshold = duplicate_threshold
        self.merge_threshold = merge_threshold
        self.min_importance = min_importance
        self.single_valued_kinds = (
            SINGLE_VALUED_KINDS if single_valued_kinds is None else single_valued_kinds
        )
        # 注意用 `is None` 而不是 truthiness：显式传空 dict 表示"不要任何类型的
        # 专用半衰期，全部走默认值"，这与"没传"是两种不同意图。
        self.half_life_s = (
            dict(DEFAULT_HALF_LIFE_S) if half_life_s is None else dict(half_life_s)
        )

    # ---------- 打分 ----------

    def is_chitchat(self, text: str) -> bool:
        return bool(_CHITCHAT.match(text.strip()))

    def score_importance(self, draft: MemoryDraft) -> float:
        score = _BASE_IMPORTANCE[draft.kind]
        for pattern, weight in _SIGNALS:
            if pattern.search(draft.text):
                score += weight
        # 长度是弱信号：太短的信息量低（"重启"），太长往往是路径/报错粘贴
        length = len(draft.text)
        if 16 <= length <= 160:
            score += 0.05
        elif length < 6:
            score -= 0.10
        # 有明确主谓宾结构的记忆更可靠
        if draft.subject and draft.predicate:
            score += 0.04
        if draft.confidence < 0.4:
            score -= 0.08
        return max(0.02, min(1.0, score))

    # ---------- 衰减 ----------

    def half_life_for(self, kind: MemoryKind) -> float:
        return self.half_life_s.get(kind, 30 * 86_400.0)

    def decayed_importance(self, item: MemoryItem, now: float) -> float:
        """指数衰减。被访问过的记忆衰减更慢（用访问次数做阻尼）。"""
        half_life = self.half_life_for(item.kind)
        age = item.age_s(now)
        # 每被访问一次，"有效年龄"打个折；避免热门记忆永不衰减
        damping = 1.0 / (1.0 + 0.15 * item.access_count)
        effective_age = age * damping
        return max(0.0, item.importance * 0.5 ** (effective_age / half_life))

    def should_forget(self, item: MemoryItem, now: float) -> bool:
        """是否该被遗忘。

        **偏好类记忆永不因衰减被遗忘**——用户三个月前说"不要用表情符号"，
        今天依然成立。这是产品语义，不是数学。
        """
        if item.is_expired(now):
            return True
        if item.kind is MemoryKind.PREFERENCE:
            return False
        return self.decayed_importance(item, now) < self.min_importance

    # ---------- 决策 ----------

    def decide(
        self,
        draft: MemoryDraft,
        existing: Sequence[MemoryItem],
        *,
        embed: Callable[[str], list[float]],
    ) -> WriteDecision:
        importance = self.score_importance(draft)

        if self.is_chitchat(draft.text):
            return WriteDecision(
                action=WriteAction.NOOP,
                reason="chitchat",
                draft=draft,
                importance=importance,
            )

        candidates = [i for i in existing if i.kind is draft.kind] or list(existing)

        # ---- 1) 相似度去重 ----
        best_sim = 0.0
        best_item: MemoryItem | None = None
        new_vec = embed(draft.text)
        for item in candidates:
            vec = item.embedding if item.embedding is not None else embed(item.text)
            if len(vec) != len(new_vec):
                continue
            sim = cosine(new_vec, vec)
            if sim > best_sim:
                best_sim, best_item = sim, item

        if best_item is not None and best_sim >= self.duplicate_threshold:
            # 完全重复：只有当新表述明显更完整时才值得合并，否则直接丢
            if len(draft.text) >= len(best_item.text) * 1.3:
                return WriteDecision(
                    action=WriteAction.MERGE,
                    reason=f"richer_restatement(sim={best_sim:.3f})",
                    draft=draft,
                    importance=max(importance, best_item.importance),
                    target_id=best_item.id,
                    similarity=best_sim,
                )
            return WriteDecision(
                action=WriteAction.NOOP,
                reason=f"duplicate(sim={best_sim:.3f})",
                draft=draft,
                importance=importance,
                target_id=best_item.id,
                similarity=best_sim,
            )

        # ---- 2) 事实冲突（同 subject+predicate，object 不同）----
        # 只对"单值事实"生效：约束/偏好是可累积的，不参与取代。
        draft_key = (
            (draft.subject.strip().lower(), draft.predicate.strip().lower())
            if (draft.subject and draft.predicate)
            else None
        )
        if draft_key is not None and draft.kind in self.single_valued_kinds:
            for item in sorted(existing, key=lambda i: i.created_at, reverse=True):
                if item.key is None or item.key != draft_key:
                    continue
                if item.kind not in self.single_valued_kinds:
                    continue
                if _norm_fact(item.object) == _norm_fact(draft.object):
                    continue
                return WriteDecision(
                    action=WriteAction.SUPERSEDE,
                    reason=(f"fact_changed: {item.object!r} -> {draft.object!r}"),
                    draft=draft,
                    importance=max(importance, 0.55),
                    target_id=item.id,
                    similarity=best_sim,
                )

        # ---- 3) 高强度相似：合并而非新增 ----
        if best_item is not None and best_sim >= self.merge_threshold:
            return WriteDecision(
                action=WriteAction.MERGE,
                reason=f"near_duplicate(sim={best_sim:.3f})",
                draft=draft,
                importance=max(importance, best_item.importance),
                target_id=best_item.id,
                similarity=best_sim,
            )

        # ---- 4) 低于重要性阈值：不记 ----
        if importance < self.min_importance:
            return WriteDecision(
                action=WriteAction.NOOP,
                reason=f"low_importance({importance:.3f} < {self.min_importance})",
                draft=draft,
                importance=importance,
            )

        return WriteDecision(
            action=WriteAction.ADD,
            reason="new_memory",
            draft=draft,
            importance=importance,
            similarity=best_sim,
        )
