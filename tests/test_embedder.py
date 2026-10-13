from __future__ import annotations

import math

import pytest

from agent_memory.embedder import HashingEmbedder, cosine, normalize_vector
from agent_memory.errors import EmbeddingDimensionError, MemoryValidationError


class TestHashingEmbedder:
    def test_dimension_property(self):
        assert HashingEmbedder(dimension=64).dimension == 64

    def test_invalid_dimension(self):
        with pytest.raises(MemoryValidationError):
            HashingEmbedder(dimension=0)

    def test_deterministic_across_instances(self):
        a = HashingEmbedder(dimension=64).embed_one("订单库用 PostgreSQL")
        b = HashingEmbedder(dimension=64).embed_one("订单库用 PostgreSQL")
        assert a == b

    def test_seed_changes_vector(self):
        a = HashingEmbedder(dimension=64, seed="one").embed_one("订单库")
        b = HashingEmbedder(dimension=64, seed="two").embed_one("订单库")
        assert a != b

    def test_vector_is_l2_normalized(self):
        vec = HashingEmbedder(dimension=64).embed_one("把硬超时改成三秒")
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_blank_text_gives_zero_vector(self):
        vec = HashingEmbedder(dimension=32).embed_one("   ")
        assert vec == [0.0] * 32

    def test_batch_matches_single(self):
        emb = HashingEmbedder(dimension=32)
        texts = ["订单库", "灰度发布", ""]
        assert emb.embed(texts) == [emb.embed_one(t) for t in texts]

    def test_similar_text_scores_higher_than_unrelated(self):
        emb = HashingEmbedder(dimension=512)
        query = emb.embed_one("订单库现在用什么数据库？")
        near = emb.embed_one("订单库用 PostgreSQL 16")
        far = emb.embed_one("今天食堂的菜不错")
        assert cosine(query, near) > cosine(query, far)

    def test_repeated_term_does_not_dominate(self):
        """sublinear tf：重复 8 次不该得到 8 倍权重。"""
        emb = HashingEmbedder(dimension=256)
        once = emb.embed_one("灰度发布")
        many = emb.embed_one("灰度发布 灰度发布 灰度发布 灰度发布 灰度发布 灰度发布")
        assert cosine(once, many) > 0.95


class TestCosine:
    def test_identical_vectors(self):
        assert cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(EmbeddingDimensionError):
            cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors(self):
        assert cosine([], []) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestNormalizeVector:
    def test_unit_length(self):
        vec = normalize_vector([3.0, 4.0])
        assert vec == pytest.approx([0.6, 0.8])

    def test_zero_vector_unchanged(self):
        assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
