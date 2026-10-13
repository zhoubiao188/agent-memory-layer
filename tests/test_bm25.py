from __future__ import annotations

import pytest

from agent_memory.bm25 import BM25Index, bm25_scores, min_max_normalize, reciprocal_rank_fusion

DOCS = {
    "d1": "订单库现在使用 PostgreSQL 16 做主库",
    "d2": "今天上线了对账服务，遇到了连接池打满",
    "d3": "回答尽量简短，不要长篇大论",
}


class TestBM25Index:
    def test_exact_term_wins(self):
        scores = BM25Index().fit(DOCS).scores("订单库")
        assert scores["d1"] > 0
        assert scores["d2"] == 0.0
        assert scores["d3"] == 0.0

    def test_rare_term_has_higher_idf(self):
        index = BM25Index().fit(
            {
                "a": "灰度发布 灰度发布 灰度发布",
                "b": "灰度发布 PostgreSQL",
            }
        )
        # PostgreSQL 只出现在一篇里，IDF 更高
        assert index._idf("postgresql") > index._idf("灰度")

    def test_query_with_no_overlap(self):
        scores = BM25Index().fit(DOCS).scores("完全无关的词汇")
        assert all(v == 0.0 for v in scores.values())

    def test_empty_index(self):
        index = BM25Index().fit({})
        assert len(index) == 0
        assert index.scores("订单库") == {}

    def test_len(self):
        assert len(BM25Index().fit(DOCS)) == 3

    def test_refit_replaces(self):
        index = BM25Index().fit(DOCS)
        index.fit({"x": "只有这一条"})
        assert len(index) == 1
        assert index.scores("订单库") == {"x": 0.0}

    def test_longer_doc_penalised_by_length_norm(self):
        index = BM25Index().fit(
            {
                "short": "订单库",
                "long": "订单库 " + "填充词 " * 40,
            }
        )
        scores = index.scores("订单库")
        assert scores["short"] > scores["long"]

    def test_accepts_iterable_of_pairs(self):
        index = BM25Index().fit(list(DOCS.items()))
        assert len(index) == 3

    def test_convenience_function(self):
        scores = bm25_scores("连接池", DOCS)
        assert scores["d2"] > 0


class TestMinMaxNormalize:
    def test_bounds(self):
        out = min_max_normalize({"a": 1.0, "b": 5.0, "c": 3.0})
        assert out["a"] == 0.0 and out["b"] == 1.0
        assert out["c"] == pytest.approx(0.5)

    def test_all_equal_gets_neutral_value(self):
        out = min_max_normalize({"a": 2.0, "b": 2.0})
        assert out == {"a": 0.5, "b": 0.5}

    def test_empty(self):
        assert min_max_normalize({}) == {}

    def test_single_value(self):
        assert min_max_normalize({"a": 7.0}) == {"a": 0.5}


class TestRRF:
    def test_document_in_both_lists_ranks_first(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        assert fused["b"] > fused["a"]
        assert fused["b"] > fused["c"]

    def test_k_dampens_top_rank_advantage(self):
        small_k = reciprocal_rank_fusion([["a", "b"]], k=1)
        large_k = reciprocal_rank_fusion([["a", "b"]], k=1000)
        ratio_small = small_k["a"] / small_k["b"]
        ratio_large = large_k["a"] / large_k["b"]
        assert ratio_small > ratio_large

    def test_empty(self):
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}
