"""可注入的时钟。

所有涉及时间的逻辑（TTL、时效衰减、归一化年龄）都不允许直接调
``time.time()``——否则测试要么睡真实时间，要么写不出确定性断言。
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "SystemClock", "ManualClock"]

DAY = 86_400.0
HOUR = 3_600.0
MINUTE = 60.0


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:  # pragma: no cover - protocol
        ...


class SystemClock:
    """生产用时钟。"""

    def now(self) -> float:
        return time.time()


class ManualClock:
    """测试用时钟：只有显式 ``advance`` 才会走。"""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("cannot go back in time")
        self._t += float(seconds)
        return self._t

    def set(self, value: float) -> None:
        self._t = float(value)

    # 语义糖，让测试读起来像自然语言
    def advance_days(self, days: float) -> float:
        return self.advance(days * DAY)

    def advance_hours(self, hours: float) -> float:
        return self.advance(hours * HOUR)
