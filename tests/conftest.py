"""让测试能直接 import src 下的包，无需先 pip install -e ."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_memory import (  # noqa: E402
    HashingEmbedder,
    ManualClock,
    MemoryItem,
    MemoryKind,
    MemoryStore,
    WritePolicy,
)
from agent_memory.testing import make_manager  # noqa: E402

START = 1_700_000_000.0


class FakeEmbedder:
    """精确可控的向量表：让"相似度刚好落在某个区间"这类测试变得可写。

    真实 embedder 的相似度是连续且难以预判的，用它去测"阈值分支"
    只能靠碰运气。把向量固定成 2 维，就能精确构造 0.98 / 0.90 / 0.40 这类边界。
    """

    def __init__(self, table: Mapping[str, Sequence[float]], dimension: int = 2) -> None:
        self.table = {k: list(v) for k, v in table.items()}
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        return list(self.table.get(text, [0.0] * self._dimension))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


@pytest.fixture()
def clock() -> ManualClock:
    return ManualClock(START)


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=128)


@pytest.fixture()
def policy() -> WritePolicy:
    return WritePolicy()


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def manager():
    m = make_manager()
    yield m
    m.close()


def make_vec_item(
    text: str,
    vec: Sequence[float],
    *,
    kind: MemoryKind = MemoryKind.SEMANTIC,
    now: float = START,
    **overrides,
) -> MemoryItem:
    """造一条"向量已知"的记忆，用于精确控制相似度。"""
    return MemoryItem(
        kind=kind,
        text=text,
        created_at=now,
        updated_at=now,
        embedding=list(vec),
        **overrides,
    )
