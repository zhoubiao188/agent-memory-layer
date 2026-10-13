"""向量化层。

这里的 :class:`HashingEmbedder` 是一个**可离线、可复现、零依赖**的 embedder：

* 用 ``hashlib.blake2b`` 而不是内置 ``hash()``。内置 hash 对 str 加了随机盐，
  换个进程就换一套向量——检索结果会在重启后漂移，这类 bug 极难排查。
* 特征 = 拉丁词 + 汉字 uni-gram + 汉字 bi-gram（与 :mod:`agent_memory.text` 共用同一套分词），
  再做 sublinear tf 加权与 L2 归一化。
* 效果上它接近"语义很弱的 TF-IDF"，但**接口与 OpenAI/本地模型完全一致**，
  所以生产里换成真模型只需要换一个实现类，其余代码一行不动。

这个"接口稳定、实现可替换"的取舍，是全文第一个工程决定。
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, Sequence, runtime_checkable

from .errors import EmbeddingDimensionError, MemoryValidationError
from .text import tokenize

__all__ = ["Embedder", "HashingEmbedder", "cosine", "normalize_vector"]


@runtime_checkable
class Embedder(Protocol):
    """向量化器协议。任何实现只要满足这三件事就能插进来。"""

    @property
    def dimension(self) -> int:  # pragma: no cover - protocol
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        ...


def normalize_vector(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。长度不一致直接抛错，而不是截断——静默截断会掩盖"换了 embedder"的事故。"""
    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"dimension mismatch: {len(a)} vs {len(b)}"
        )
    if not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _stable_index(feature: str, seed: str, dim: int) -> tuple[int, float]:
    """把一个特征稳定地映射到 ``(维度下标, 符号)``。

    用 blake2b 取 8 字节：前 4 字节做下标，后 4 字节的最低位做符号。
    符号位是必要的——只取正号会让哈希碰撞变成单调累加，
    相似度差异被抹平。
    """
    digest = hashlib.blake2b(
        feature.encode("utf-8"), digest_size=8, key=seed.encode("utf-8")[:64]
    ).digest()
    idx = struct.unpack("<I", digest[:4])[0] % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


class HashingEmbedder:
    """确定性哈希 embedder（hashing trick + sublinear TF + L2 归一化）。"""

    def __init__(self, dimension: int = 256, *, seed: str = "agent-memory") -> None:
        if dimension <= 0:
            raise MemoryValidationError(f"dimension must be > 0: {dimension}")
        self._dimension = int(dimension)
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self._dimension

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        vec = [0.0] * self._dimension
        for feature, tf in counts.items():
            weight = 1.0 + math.log(tf)  # sublinear tf：出现 10 次的词不该是 1 次的 10 倍权重
            idx, sign = _stable_index(feature, self._seed, self._dimension)
            vec[idx] += sign * weight
        return normalize_vector(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
