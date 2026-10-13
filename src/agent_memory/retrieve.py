"""混合检索：向量 + BM25 + 时效 + 重要性 + 访问频次 → MMR 去冗余。

### 为什么单路向量不够

向量检索擅长"意思相近"，但在记忆场景下有三个硬伤：

1. **低频专有名词**。用户说"退款走 ``refund_v2`` 网关"，向量把 token 打散进桶，
   和"支付走 refund_v2"未必近；BM25 精确命中 ``refund_v2``。
2. **时间无关**。用户三个月前说"我用 Vim"，今天说"我改用 Neovim 了"。
   两条都在库里，纯向量可能把旧的排前面。
3. **重复霸屏**。用户说过 8 次"回答要简短"，8 条几乎一样的记忆会把 top-k 占满，
   真正的关键信息挤不进来。

对应三个补充信号：``lexical`` / ``recency`` / 以及去冗余的 ``MMR``。
**注意这些是"排序信号"而不是"过滤条件"**——过滤会引入硬边界，
排序可以做平滑加权的组合。
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .bm25 import BM25Index, min_max_normalize, reciprocal_rank_fusion
from .embedder import Embedder, cosine
from .errors import MemoryValidationError
from .models import MemoryItem, MemoryKind, MemoryStatus, RecallHit

__all__ = ["RetrievalWeights", "HybridRetriever"]

DAY = 86_400.0


class RetrievalWeights:
    """五路信号的权重。会自动归一化到和为 1。"""

    __slots__ = ("similarity", "lexical", "recency", "importance", "frequency")

    def __init__(
        self,
        *,
        similarity: float = 0.52,
        lexical: float = 0.24,
        recency: float = 0.12,
        importance: float = 0.08,
        frequency: float = 0.04,
    ) -> None:
        values = {
            "similarity": similarity,
            "lexical": lexical,
            "recency": recency,
            "importance": importance,
            "frequency": frequency,
        }
        for name, v in values.items():
            if v < 0:
                raise MemoryValidationError(f"weight {name} must be >= 0: {v}")
        total = sum(values.values())
        if total <= 0:
            raise MemoryValidationError("at least one weight must be > 0")
        for name, v in values.items():
            setattr(self, name, v / total)

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def lexical_heavy(cls) -> "RetrievalWeights":
        """词面级 embedder（哈希 / TF-IDF 这类）的推荐权重。

        理由：这类 embedder 的相似度**区分度很差**——实测两条人眼可辨的同义表述
        也只有 0.5~0.8 的余弦，而"完全不相关"的两条能到 0.5。
        如果继续让 ``similarity`` 占 0.52，排序会退化成"看起来都差不多"。
        此时应当把话语权交给 BM25 与重要性。

        换成真实语义向量（bge / voyage / text-embedding-3）后请用默认权重。
        **权重和阈值一样，是跟着 embedder 走的校准量，不是常数。**
        """
        return cls(
            similarity=0.30,
            lexical=0.40,
            recency=0.12,
            importance=0.12,
            frequency=0.06,
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.as_dict().items())
        return f"RetrievalWeights({parts})"


class HybridRetriever:
    """多信号混合检索器。"""

    def __init__(
        self,
        embedder: Embedder,
        weights: RetrievalWeights | None = None,
        *,
        recency_half_life_s: float = 30 * DAY,
        prefer_recent: bool = True,
    ) -> None:
        self.embedder = embedder
        self.weights = weights or RetrievalWeights()
        self.recency_half_life_s = float(recency_half_life_s)
        self.prefer_recent = prefer_recent

    # ---------- 过滤 ----------

    @staticmethod
    def filter_candidates(
        candidates: Iterable[MemoryItem],
        *,
        now: float,
        kinds: Sequence[MemoryKind] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        wanted_kinds = set(kinds or ())
        wanted_tags = {t.lower() for t in (tags or ())}
        out: list[MemoryItem] = []
        for item in candidates:
            if item.status is not MemoryStatus.ACTIVE:
                continue
            if item.is_expired(now):
                continue
            if wanted_kinds and item.kind not in wanted_kinds:
                continue
            if wanted_tags and not (wanted_tags & {t.lower() for t in item.tags}):
                continue
            out.append(item)
        return out

    # ---------- 单路信号 ----------

    def _lexical(self, query: str, items: Sequence[MemoryItem]) -> dict[str, float]:
        if not query.strip():
            return {i.id: 0.0 for i in items}
        index = BM25Index().fit({i.id: i.text for i in items})
        return min_max_normalize(index.scores(query))

    def _similarity(self, query_vec: Sequence[float], items: Sequence[MemoryItem]) -> dict[str, float]:
        out: dict[str, float] = {}
        for item in items:
            vec = item.embedding if item.embedding is not None else self.embedder.embed_one(item.text)
            if len(vec) != len(query_vec):
                out[item.id] = 0.0
                continue
            # 余弦在 [-1,1]，压到 [0,1] 才能和其他信号相加
            out[item.id] = (cosine(query_vec, vec) + 1.0) / 2.0
        return out

    def _recency(self, items: Sequence[MemoryItem], now: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for item in items:
            if self.prefer_recent:
                # 用"最近一次被用到"或"创建时间"里更近的那个：
                # 一条老记忆如果昨天刚被召回，它就是"当下相关"的。
                anchor = max(item.created_at, item.last_accessed_at or 0.0)
                age = max(0.0, now - anchor)
                out[item.id] = 0.5 ** (age / self.recency_half_life_s)
            else:
                out[item.id] = 1.0
        return out

    @staticmethod
    def _frequency(items: Sequence[MemoryItem]) -> dict[str, float]:
        counts = [i.access_count for i in items]
        peak = max(counts) if counts else 0
        if peak <= 0:
            return {i.id: 0.0 for i in items}
        denom = math.log1p(peak)
        return {i.id: math.log1p(i.access_count) / denom for i in items}

    # ---------- 融合 ----------

    def _fuse(
        self,
        items: Sequence[MemoryItem],
        *,
        similarities: Mapping[str, float],
        lexicals: Mapping[str, float],
        recencies: Mapping[str, float],
        frequencies: Mapping[str, float],
    ) -> dict[str, tuple[float, dict[str, float]]]:
        w = self.weights
        out: dict[str, tuple[float, dict[str, float]]] = {}
        for item in items:
            parts = {
                "similarity": similarities.get(item.id, 0.0),
                "lexical": lexicals.get(item.id, 0.0),
                "recency": recencies.get(item.id, 0.0),
                "importance": float(item.importance),
                "frequency": frequencies.get(item.id, 0.0),
            }
            score = (
                w.similarity * parts["similarity"]
                + w.lexical * parts["lexical"]
                + w.recency * parts["recency"]
                + w.importance * parts["importance"]
                + w.frequency * parts["frequency"]
            )
            parts = {k: round(v, 6) for k, v in parts.items()}
            out[item.id] = (round(score, 6), parts)
        return out

    # ---------- MMR ----------

    @staticmethod
    def mmr_select(
        ranked: Sequence[tuple[str, float]],
        vectors: Mapping[str, Sequence[float]],
        *,
        top_k: int,
        lam: float = 0.7,
    ) -> list[str]:
        """最大边际相关：``lam * 相关度 - (1-lam) * 已选集合的最大相似度``。

        ``lam=1`` 退化成纯按分数排序；``lam=0`` 退化成最不相似优先。
        0.7 是我在"事实类记忆"上调出来的值——再低会开始把相关但冗余的上下文挤掉，
        再高则防不住重复霸屏。
        """
        if top_k <= 0 or not ranked:
            return []
        scores = {doc_id: s for doc_id, s in ranked}
        # 分数归一到 [0,1]，否则"相关度"的量纲会压死相似度惩罚项
        lo = min(scores.values())
        hi = max(scores.values())
        span = hi - lo if hi - lo > 1e-12 else 1.0
        norm = {k: (v - lo) / span for k, v in scores.items()}

        remaining = [doc_id for doc_id, _ in ranked]
        selected: list[str] = []
        while remaining and len(selected) < top_k:
            best_id, best_value = None, -1e18
            for doc_id in remaining:
                redundancy = 0.0
                vec = vectors.get(doc_id)
                if vec is not None and selected:
                    redundancy = max(
                        cosine(vec, vectors[s]) if vectors.get(s) is not None else 0.0
                        for s in selected
                    )
                value = lam * norm[doc_id] - (1 - lam) * redundancy
                if value > best_value:
                    best_id, best_value = doc_id, value
            assert best_id is not None
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    # ---------- 对外 ----------

    def retrieve(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
        *,
        now: float,
        top_k: int = 8,
        kinds: Sequence[MemoryKind] | None = None,
        tags: Sequence[str] | None = None,
        min_score: float = 0.0,
        diversify: bool = False,
        mmr_lambda: float = 0.7,
    ) -> list[RecallHit]:
        pool = self.filter_candidates(candidates, now=now, kinds=kinds, tags=tags)
        if not pool or top_k <= 0:
            return []

        query_vec = self.embedder.embed_one(query) if query.strip() else [0.0] * self.embedder.dimension
        similarities = self._similarity(query_vec, pool) if query.strip() else {i.id: 0.0 for i in pool}
        lexicals = self._lexical(query, pool)
        recencies = self._recency(pool, now)
        frequencies = self._frequency(pool)

        fused = self._fuse(
            pool,
            similarities=similarities,
            lexicals=lexicals,
            recencies=recencies,
            frequencies=frequencies,
        )
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1][0], kv[0]))

        if diversify:
            vectors = {
                i.id: (i.embedding or self.embedder.embed_one(i.text)) for i in pool
            }
            # 注意：fused 的 value 是 (score, components)，MMR 只接受 (id, score)
            order = self.mmr_select(
                [(doc_id, score) for doc_id, (score, _) in ranked],
                vectors,
                top_k=top_k,
                lam=mmr_lambda,
            )
        else:
            order = [doc_id for doc_id, _ in ranked[:top_k]]

        by_id = {i.id: i for i in pool}
        hits: list[RecallHit] = []
        for doc_id in order:
            score, parts = fused[doc_id]
            if score < min_score:
                continue
            hits.append(RecallHit(item=by_id[doc_id], score=score, components=parts))
        return hits

    def retrieve_rrf(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
        *,
        now: float,
        top_k: int = 8,
        kinds: Sequence[MemoryKind] | None = None,
        tags: Sequence[str] | None = None,
        k: int = 60,
    ) -> list[RecallHit]:
        """RRF 变体：只融合"向量排名"和"BM25 排名"，分数由排名反推。

        当两路分数的分布形态差异极大时（例如 query 里有精确 ID），
        RRF 比加权求和更抗量纲问题。代价是丢掉了时效/重要性信号，
        所以生产里我通常用它做"召回"，再用加权融合做"重排"。
        """
        pool = self.filter_candidates(candidates, now=now, kinds=kinds, tags=tags)
        if not pool:
            return []
        vec_rank: list[str] = []
        if query.strip():
            qv = self.embedder.embed_one(query)
            sims = self._similarity(qv, pool)
            vec_rank = [d for d, _ in sorted(sims.items(), key=lambda kv: -kv[1])]
        bm_scores = self._lexical(query, pool)
        bm_rank = [d for d, v in sorted(bm_scores.items(), key=lambda kv: -kv[1]) if v > 0]
        fused = reciprocal_rank_fusion([vec_rank, bm_rank], k=k)
        by_id = {i.id: i for i in pool}
        hits = [
            RecallHit(item=by_id[d], score=round(s, 8), components={"rrf": round(s, 8)})
            for d, s in sorted(fused.items(), key=lambda kv: -kv[1])
            if d in by_id
        ]
        return hits[:top_k]
