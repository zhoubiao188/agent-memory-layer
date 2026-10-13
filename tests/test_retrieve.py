from __future__ import annotations

import pytest

from agent_memory.bm25 import reciprocal_rank_fusion
from agent_memory.embedder import HashingEmbedder
from agent_memory.errors import MemoryValidationError
from agent_memory.models import MemoryItem, MemoryKind, MemoryStatus
from agent_memory.retrieve import HybridRetriever, RetrievalWeights

from conftest import START, FakeEmbedder

DAY = 86_400.0


def make(text: str, kind: MemoryKind = MemoryKind.SEMANTIC, **kw) -> MemoryItem:
    kw.setdefault("created_at", START)
    kw.setdefault("updated_at", START)
    return MemoryItem(kind=kind, text=text, **kw)


@pytest.fixture()
def retriever() -> HybridRetriever:
    return HybridRetriever(HashingEmbedder(dimension=256))


CORPUS = [
    make("订单库现在使用 PostgreSQL 16 做主库", importance=0.5),
    make("回答尽量简短，不要长篇大论", MemoryKind.PREFERENCE, importance=0.9),
    make("今天上线了对账服务，遇到连接池打满", MemoryKind.EPISODIC, importance=0.5),
    make("以后都走灰度发布，上线前必须有回滚开关", MemoryKind.PROCEDURAL, importance=0.8),
]


class TestFilterCandidates:
    def test_excludes_non_active(self):
        items = [make("a"), make("b", status=MemoryStatus.SUPERSEDED)]
        kept = HybridRetriever.filter_candidates(items, now=START)
        assert [i.text for i in kept] == ["a"]

    def test_excludes_expired_ttl(self):
        items = [make("a", ttl_s=10.0), make("b")]
        kept = HybridRetriever.filter_candidates(items, now=START + 11)
        assert [i.text for i in kept] == ["b"]

    def test_filter_by_kind(self):
        kept = HybridRetriever.filter_candidates(
            CORPUS, now=START, kinds=[MemoryKind.PREFERENCE]
        )
        assert len(kept) == 1

    def test_filter_by_tags_is_or_semantics(self):
        items = [make("a", tags=["db"]), make("b", tags=["ops"]), make("c")]
        kept = HybridRetriever.filter_candidates(items, now=START, tags=["db", "ops"])
        assert {i.text for i in kept} == {"a", "b"}

    def test_no_filters_keeps_all(self):
        assert len(HybridRetriever.filter_candidates(CORPUS, now=START)) == 4


class TestWeights:
    def test_defaults_sum_to_one(self):
        w = RetrievalWeights()
        assert sum(w.as_dict().values()) == pytest.approx(1.0)
        assert set(w.as_dict()) == {
            "similarity",
            "lexical",
            "recency",
            "importance",
            "frequency",
        }

    def test_arbitrary_scale_normalized(self):
        w = RetrievalWeights(similarity=10, lexical=10, recency=0, importance=0, frequency=0)
        assert w.similarity == pytest.approx(0.5)
        assert w.lexical == pytest.approx(0.5)

    def test_negative_rejected(self):
        with pytest.raises(MemoryValidationError):
            RetrievalWeights(similarity=-1)

    def test_all_zero_rejected(self):
        with pytest.raises(MemoryValidationError):
            RetrievalWeights(similarity=0, lexical=0, recency=0, importance=0, frequency=0)

    def test_lexical_heavy_preset_prioritises_bm25(self):
        preset = RetrievalWeights.lexical_heavy()
        default = RetrievalWeights()
        assert preset.lexical > default.lexical
        assert preset.similarity < default.similarity
        assert sum(preset.as_dict().values()) == pytest.approx(1.0)

    def test_lexical_heavy_improves_low_discrimination_ranking(self):
        """词面级 embedder 下，lexical-heavy 预设能让精确匹配排得更前。"""
        items = [
            make("退款服务走 refund_v2 网关"),
            make("订单流程需要两级审批"),
        ]
        default = HybridRetriever(HashingEmbedder(dimension=512))
        heavy = HybridRetriever(
            HashingEmbedder(dimension=512), RetrievalWeights.lexical_heavy()
        )
        d_hits = default.retrieve("refund_v2", items, now=START, top_k=2)
        h_hits = heavy.retrieve("refund_v2", items, now=START, top_k=2)
        assert d_hits[0].item.text == h_hits[0].item.text
        # 命中词的领先幅度应当被放大
        assert (h_hits[0].score - h_hits[1].score) > (d_hits[0].score - d_hits[1].score)


class TestRetrieve:
    def test_relevant_document_ranks_first(self, retriever):
        hits = retriever.retrieve("订单库用什么数据库", CORPUS, now=START, top_k=4)
        assert hits[0].item.text.startswith("订单库现在使用 PostgreSQL")

    def test_components_reported(self, retriever):
        hits = retriever.retrieve("订单库", CORPUS, now=START, top_k=1)
        assert set(hits[0].components) == {
            "similarity",
            "lexical",
            "recency",
            "importance",
            "frequency",
        }

    def test_top_k_limits(self, retriever):
        assert len(retriever.retrieve("订单库", CORPUS, now=START, top_k=2)) == 2

    def test_top_k_zero(self, retriever):
        assert retriever.retrieve("订单库", CORPUS, now=START, top_k=0) == []

    def test_empty_candidates(self, retriever):
        assert retriever.retrieve("订单库", [], now=START) == []

    def test_empty_query_still_ranks_by_importance(self, retriever):
        hits = retriever.retrieve("", CORPUS, now=START, top_k=4)
        assert len(hits) == 4
        # 没有文本信号时，重要度最高的偏好记忆应该排前面
        assert hits[0].item.kind is MemoryKind.PREFERENCE

    def test_kinds_filter(self, retriever):
        hits = retriever.retrieve(
            "发布", CORPUS, now=START, kinds=[MemoryKind.PROCEDURAL]
        )
        assert all(h.item.kind is MemoryKind.PROCEDURAL for h in hits)

    def test_min_score_filters(self, retriever):
        all_hits = retriever.retrieve("订单库", CORPUS, now=START, top_k=4)
        cutoff = all_hits[0].score + 0.001
        filtered = retriever.retrieve(
            "订单库", CORPUS, now=START, top_k=4, min_score=cutoff
        )
        assert filtered == []

    def test_recency_prefers_newer(self):
        retriever = HybridRetriever(HashingEmbedder(dimension=64))
        old = make("订单库的说明文档", created_at=START - 365 * DAY)
        new = make("订单库的说明文档", created_at=START)
        hits = retriever.retrieve("订单库", [old, new], now=START, top_k=2)
        assert hits[0].item.id == new.id

    def test_recency_can_be_disabled(self):
        retriever = HybridRetriever(HashingEmbedder(dimension=64), prefer_recent=False)
        old = make("订单库的说明文档", created_at=START - 365 * DAY, importance=0.9)
        new = make("订单库的说明文档", created_at=START, importance=0.5)
        hits = retriever.retrieve("订单库", [old, new], now=START, top_k=2)
        # 关掉时效信号后，重要性重新成为决定性因素
        assert hits[0].item.id == old.id

    def test_frequency_signal(self):
        retriever = HybridRetriever(HashingEmbedder(dimension=64))
        cold = make("灰度发布的检查清单")
        hot = make("灰度发布的检查清单", access_count=30)
        hits = retriever.retrieve("灰度发布", [cold, hot], now=START, top_k=2)
        assert hits[0].components["frequency"] == pytest.approx(1.0)
        assert hits[0].item.id == hot.id

    def test_diversify_returns_unique_ids(self, retriever):
        hits = retriever.retrieve(
            "发布 数据库 回答", CORPUS, now=START, top_k=3, diversify=True
        )
        assert len({h.id for h in hits}) == len(hits)

    def test_lexical_boosts_exact_identifier(self):
        """低频专有名词靠 BM25 命中——这是向量路的盲区。"""
        retriever = HybridRetriever(HashingEmbedder(dimension=512))
        items = [
            make("退款服务走 refund_v2 网关"),
            make("退款流程需要两级审批"),
            make("订单库使用 PostgreSQL"),
        ]
        hits = retriever.retrieve("refund_v2 网关", items, now=START, top_k=3)
        assert hits[0].item.text.startswith("退款服务走 refund_v2")


class TestMMR:
    def test_pure_relevance_takes_top_scores(self):
        ranked = [("A", 0.9), ("B", 0.8), ("C", 0.75)]
        vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0]}
        assert HybridRetriever.mmr_select(ranked, vecs, top_k=2, lam=1.0) == ["A", "B"]

    def test_diversity_pushes_out_redundant_item(self):
        ranked = [("A", 0.9), ("B", 0.8), ("C", 0.75)]
        vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0]}
        assert HybridRetriever.mmr_select(ranked, vecs, top_k=2, lam=0.7) == ["A", "C"]

    def test_zero_lambda_picks_least_similar(self):
        ranked = [("A", 0.9), ("B", 0.8), ("C", 0.75)]
        vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0]}
        assert HybridRetriever.mmr_select(ranked, vecs, top_k=2, lam=0.0) == ["A", "C"]

    def test_empty_inputs(self):
        assert HybridRetriever.mmr_select([], {}, top_k=3) == []
        assert HybridRetriever.mmr_select([("A", 1.0)], {}, top_k=0) == []

    def test_missing_vector_treated_as_no_redundancy(self):
        ranked = [("A", 0.9), ("B", 0.8)]
        assert HybridRetriever.mmr_select(ranked, {}, top_k=2) == ["A", "B"]


class TestRRFRetrieval:
    def test_returns_rrf_component(self, retriever):
        hits = retriever.retrieve_rrf("订单库 PostgreSQL", CORPUS, now=START, top_k=2)
        assert hits
        assert set(hits[0].components) == {"rrf"}

    def test_top_hit_is_relevant(self, retriever):
        hits = retriever.retrieve_rrf("订单库 PostgreSQL", CORPUS, now=START, top_k=4)
        assert hits[0].item.text.startswith("订单库现在使用")

    def test_empty_candidates(self, retriever):
        assert retriever.retrieve_rrf("x", [], now=START) == []

    def test_rrf_rewards_agreement(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        assert fused["a"] > fused["b"]

    def test_dimension_mismatch_degrades_to_zero(self):
        """换过 embedder 却没重建索引时，不该抛异常拖垮召回。"""
        emb = FakeEmbedder({"query": [1.0, 0.0]}, dimension=2)
        retriever = HybridRetriever(emb)  # type: ignore[arg-type]
        stale = make("历史遗留记忆", embedding=[1.0, 0.0, 0.0, 0.0])
        hits = retriever.retrieve("query", [stale], now=START, top_k=1)
        assert hits[0].components["similarity"] == 0.0
