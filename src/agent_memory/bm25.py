"""纯内存 BM25 索引。

为什么向量检索之外还要一路 BM25：向量对**低频专有名词**很不友好。
用户说"把 idempotency-key 带上"，哈希向量里 ``idempotency`` 和 ``key``
会被打散成若干桶，跟"幂等键"三个字几乎不共享维度；
而 BM25 靠精确词匹配能一击命中。两路融合比任何单路都稳。

实现上刻意做成"每次检索现场构建"的轻量索引——记忆条数通常在 10^2~10^4，
构建成本远小于维护一份增量倒排的复杂度。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

from .text import tokenize

__all__ = ["BM25Index", "bm25_scores"]


class BM25Index:
    """标准 BM25（含 k1 / b 调参）的内存实现。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._doc_tokens: dict[str, list[str]] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0

    # ---------- 建索引 ----------

    def fit(self, docs: Mapping[str, str] | Iterable[tuple[str, str]]) -> "BM25Index":
        items = docs.items() if isinstance(docs, Mapping) else docs
        self._doc_tokens.clear()
        self._tf.clear()
        self._df.clear()
        total_len = 0
        for doc_id, text in items:
            tokens = tokenize(text)
            self._doc_tokens[doc_id] = tokens
            self._tf[doc_id] = Counter(tokens)
            total_len += len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._n = len(self._doc_tokens)
        self._avgdl = (total_len / self._n) if self._n else 0.0
        return self

    # ---------- 打分 ----------

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # 概率型 IDF 的平滑版本，保证非负：即使某词出现在所有文档里也有一点权重
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        tokens = self._tf.get(doc_id)
        if not tokens:
            return 0.0
        dl = len(self._doc_tokens[doc_id]) or 1
        total = 0.0
        for term in set(tokenize(query)):
            tf = tokens.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            total += self._idf(term) * (tf * (self.k1 + 1)) / denom
        return total

    def scores(self, query: str) -> dict[str, float]:
        return {doc_id: self.score(query, doc_id) for doc_id in self._doc_tokens}

    def __len__(self) -> int:
        return self._n


def bm25_scores(
    query: str, docs: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75
) -> dict[str, float]:
    """一次性便捷入口：适合测试与小数据集。"""
    return BM25Index(k1=k1, b=b).fit(docs).scores(query)


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """把任意量纲的分数压到 [0,1]。

    融合两路检索时**必须**先归一化：BM25 的值域是 ``[0, +inf)``，
    余弦是 ``[-1, 1]``，直接加权等于让 BM25 主导一切。
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # 全部同分：给一个恒定的中性值，而不是全 0，
        # 否则这一路在融合时就"消失"了，权重失去意义。
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[str]], *, k: int = 60
) -> dict[str, float]:
    """RRF 融合：只看排名不看分数，天然免去归一化。

    ``score(d) = Σ 1 / (k + rank_i(d))``。k 越大，头部排名差异被压得越平。
    当两路分数分布差异极大（例如一路是精确匹配），RRF 通常比加权求和更稳。
    """
    fused: dict[str, float] = {}
    for ranks in rank_lists:
        for position, doc_id in enumerate(ranks):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return fused
