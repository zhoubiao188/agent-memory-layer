from __future__ import annotations

import pytest

from agent_memory.text import (
    STOPWORDS,
    estimate_tokens,
    normalize_text,
    tokenize,
    tokenize_all,
    tokenize_iter,
)


class TestNormalize:
    def test_strips_and_collapses_whitespace(self):
        assert normalize_text("  你好   世界 \n ") == "你好 世界"

    def test_fullwidth_to_halfwidth(self):
        assert normalize_text("ＡＢＣ１２３") == "ABC123"

    def test_fullwidth_space_becomes_ascii_space(self):
        assert normalize_text("a\u3000b") == "a b"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_keeps_cjk_ideographic_punctuation(self):
        # U+3002 属于 CJK 标点区，不参与全角→半角转换，保留原样
        assert normalize_text("世界。") == "世界。"

    def test_converts_fullwidth_ascii_punctuation(self):
        # U+FF0C 是"全角逗号"，属于 FF01-FF5E 区，会被转成 ASCII 逗号
        assert normalize_text("你好，世界") == "你好,世界"


class TestTokenize:
    def test_latin_word_lowercased_and_hyphen_kept(self):
        tokens = tokenize("Tool-Runtime 的硬超时")
        assert "tool-runtime" in tokens
        assert "超时" in tokens
        assert "硬" in tokens

    def test_numbers_stay_whole(self):
        tokens = tokenize("Python 3.11 上线")
        assert "3.11" in tokens
        assert "python" in tokens

    def test_cjk_unigrams_and_bigrams(self):
        tokens = tokenize("退款单")
        assert {"退", "款", "单"} <= set(tokens)
        assert {"退款", "款单"} <= set(tokens)

    def test_stopwords_removed_by_default(self):
        tokens = tokenize("the tool")
        assert "tool" in tokens
        assert "the" not in tokens

    def test_stopwords_kept_on_demand(self):
        tokens = tokenize("the tool", keep_stopwords=True)
        assert "the" in tokens

    def test_all_stopwords_keeps_bigram(self):
        # "的了" 的单字都是停用词，但 bigram "的了" 不在停用词表里，
        # 所以过滤后不会为空——避免"空 query 导致检索退化"。
        tokens = tokenize("的了")
        assert tokens == ["的了"]

    def test_blank_returns_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_cjk_only_sentence_yields_many_tokens(self):
        tokens = tokenize("订单库改用 PostgreSQL")
        assert len(tokens) >= 6

    def test_tokenize_all_and_iter_agree(self):
        texts = ["你好世界", "tool call"]
        assert tokenize_all(texts) == [tokenize(t) for t in texts]
        assert list(tokenize_iter(texts[0])) == tokenize(texts[0])

    def test_stopwords_table_nonempty(self):
        assert len(STOPWORDS) > 20
        assert "的" in STOPWORDS


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_single_chinese_char(self):
        assert estimate_tokens("中") == 1

    def test_grows_with_length(self):
        short = estimate_tokens("你好")
        long = estimate_tokens("你好，请把订单服务的超时配置改成三秒")
        assert long > short

    def test_latin_words_counted(self):
        assert estimate_tokens("hello world") >= 2

    def test_never_negative(self):
        for text in ["", " ", "a", "。", "中英混排 abc 123"]:
            assert estimate_tokens(text) >= 0

    @pytest.mark.parametrize("text", ["退款", "退款流程", "退款流程需要审批"])
    def test_monotonic_in_chinese_length(self, text):
        assert estimate_tokens(text) >= len(text) * 0.8
