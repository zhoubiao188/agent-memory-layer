"""上下文组装：把召回结果塞进一个**有预算的**提示块。

这一步经常被忽略，但它是"记忆系统"和"往 prompt 里灌数据"的分水岭：

* 召回 20 条，全塞进去，prompt 直接涨 3000 token，还稀释了真正的关键信息；
* 只塞 top-3，可能漏掉那条"绝对不能做"的硬约束。

所以这里做两件事：

1. **硬约束优先**。偏好/规范类记忆单独成块，排在背景之前，享受"先占预算"的权利。
2. **按预算贪心装箱**。放不下的不截断、直接丢弃并记账（``dropped_ids``），
   因为半句话的记忆比没有记忆更危险——模型会把断句当完整事实。
"""

from __future__ import annotations

from typing import Sequence

from .models import MemoryKind, MemoryItem, RecallHit
from .text import estimate_tokens

__all__ = ["AssembledContext", "ContextAssembler"]

#: 这些类型的记忆不是"背景知识"，而是"必须遵守的约定"。
_CONSTRAINT_KINDS = {MemoryKind.PREFERENCE, MemoryKind.PROCEDURAL}


class AssembledContext:
    """组装结果。除了文本本身，还带完整的"装箱账单"。"""

    __slots__ = (
        "text",
        "included_ids",
        "dropped_ids",
        "tokens_used",
        "budget",
        "truncated",
    )

    def __init__(
        self,
        text: str,
        included_ids: list[str],
        dropped_ids: list[str],
        tokens_used: int,
        budget: int,
        truncated: bool,
    ) -> None:
        self.text = text
        self.included_ids = included_ids
        self.dropped_ids = dropped_ids
        self.tokens_used = tokens_used
        self.budget = budget
        self.truncated = truncated

    @property
    def is_empty(self) -> bool:
        return not self.included_ids

    @property
    def utilization(self) -> float:
        return 0.0 if self.budget <= 0 else self.tokens_used / self.budget

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"AssembledContext(items={len(self.included_ids)}, "
            f"tokens={self.tokens_used}/{self.budget}, truncated={self.truncated})"
        )


def _humanize_age(seconds: float) -> str:
    if seconds < 90:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    days = int(seconds // 86400)
    if days < 60:
        return f"{days} 天前"
    return f"{days // 30} 个月前"


class ContextAssembler:
    """带 token 预算的上下文组装器。"""

    def __init__(
        self,
        *,
        max_tokens: int = 800,
        header: str = "## 关于用户的已知记忆",
        constraint_header: str = "### 必须遵守的约定",
        background_header: str = "### 相关背景",
        show_age: bool = True,
        show_kind: bool = False,
        cite: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = int(max_tokens)
        self.header = header
        self.constraint_header = constraint_header
        self.background_header = background_header
        self.show_age = show_age
        self.show_kind = show_kind
        self.cite = cite

    # ---------- 渲染 ----------

    def render_line(self, item: MemoryItem, *, now: float) -> str:
        bits: list[str] = []
        if self.show_age:
            bits.append(_humanize_age(item.age_s(now)))
        if self.show_kind:
            bits.append(item.kind.value)
        prefix = f"（{', '.join(bits)}）" if bits else ""
        suffix = f" [m:{item.id[:6]}]" if self.cite else ""
        return f"- {prefix}{item.text}{suffix}"

    @staticmethod
    def _block_cost(header: str, lines: Sequence[str]) -> int:
        if not lines:
            return 0
        return estimate_tokens(header) + sum(estimate_tokens(line) for line in lines)

    # ---------- 组装 ----------

    def assemble(
        self,
        hits: Sequence[RecallHit],
        *,
        now: float,
        max_tokens: int | None = None,
        pin_constraints: bool = True,
    ) -> AssembledContext:
        budget = int(max_tokens if max_tokens is not None else self.max_tokens)
        if budget <= 0:
            raise ValueError("max_tokens must be > 0")

        if not hits:
            return AssembledContext("", [], [], 0, budget, False)

        constraints: list[tuple[MemoryItem, str]] = []
        background: list[tuple[MemoryItem, str]] = []
        for hit in hits:
            item = hit.item
            rendered = self.render_line(item, now=now)
            bucket = constraints if (pin_constraints and item.kind in _CONSTRAINT_KINDS) else background
            bucket.append((item, rendered))

        included: list[str] = []
        dropped: list[str] = []
        c_lines: list[str] = []
        b_lines: list[str] = []

        base = estimate_tokens(self.header)
        used = base
        if base > budget:
            # 连标题都放不下：直接返回空上下文，让调用方自己决定降级策略
            return AssembledContext(
                "", [], [h.item.id for h in hits], 0, budget, True
            )

        for item, line in constraints:
            cost = estimate_tokens(line) + (
                estimate_tokens(self.constraint_header) if not c_lines else 0
            )
            if used + cost > budget:
                dropped.append(item.id)
                continue
            c_lines.append(line)
            used += cost
            included.append(item.id)

        for item, line in background:
            cost = estimate_tokens(line) + (
                estimate_tokens(self.background_header) if not b_lines else 0
            )
            if used + cost > budget:
                dropped.append(item.id)
                continue
            b_lines.append(line)
            used += cost
            included.append(item.id)

        parts: list[str] = []
        if c_lines:
            parts.append(self.constraint_header)
            parts.extend(c_lines)
        if b_lines:
            parts.append(self.background_header)
            parts.extend(b_lines)
        if not parts:
            return AssembledContext("", [], [h.item.id for h in hits], 0, budget, True)

        body = "\n".join(parts)
        text = f"{self.header}\n{body}"
        return AssembledContext(
            text=text,
            included_ids=included,
            dropped_ids=dropped,
            tokens_used=estimate_tokens(text),
            budget=budget,
            truncated=bool(dropped),
        )
