"""中文友好的分词与 token 估算。

为什么不用 tiktoken / jieba：

* **可离线性**。整个仓库零 API Key、零模型权重也能跑通全部测试。
* **确定性**。分词结果必须可复现，否则检索分数会随机漂移，
  测试里就没法写断言。Python 内置 ``hash()`` 因为有 PYTHONHASHSEED 随机化，
  连"同一进程内的两次运行结果一致"都保证不了——所以后面的 embedder 用的是 blake2b。

token 估算是**启发式**，不是精确值。它的用途是"别把 prompt 撑爆"，不是计费。
"""

from __future__ import annotations

import math
import re
from typing import Iterator, Sequence

__all__ = [
    "tokenize",
    "tokenize_all",
    "estimate_tokens",
    "normalize_text",
    "STOPWORDS",
]

_LATIN_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

#: 极简停用词表。只处理"高频且几乎不携带检索价值"的词，
#: 保持短小是为了不误杀领域词（例如 "not"、"no" 在约束类记忆里是有意义的，
#: 所以我们**不**把英文否定词放进来）。
STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "也",
        "就",
        "都",
        "而",
        "及",
        "或",
        "一个",
        "这个",
        "那个",
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
        "are",
        "was",
        "were",
        "be",
        "and",
        "or",
        "for",
        "in",
        "on",
        "at",
        "it",
        "that",
        "this",
        "with",
    }
)

# 经验系数：一个汉字约等于 0.85 个 token（BPE 里常见汉字多为 1 token，
# 但中文标点和常用双字词会被合并，实测下来 0.8~0.9 更接近主流中文 tokenizer）。
_CJK_TOKEN_RATIO = 0.85
_LATIN_TOKEN_RATIO = 1.30
_PUNCT_TOKEN_RATIO = 0.50


def normalize_text(text: str) -> str:
    """折叠空白 + 全角标点转半角 + 小写化之前的中文保留。

    只做无损的规范化：不改字符种类，只折叠空白和统一常见标点，
    保证"同一句话的不同写法"能落到同一个 key 上。
    """
    if not text:
        return ""
    out = text.strip()
    # 全角字符 → 半角（仅处理 ASCII 可见范围内的全角形式）
    out = "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch
        for ch in out
    )
    out = out.replace("\u3000", " ")
    out = _WS.sub(" ", out)
    return out.strip()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """把文本切成检索用的 token 序列。

    产生的 token 种类：
    * 拉丁词：小写化，保留 ``-`` / ``_``（``tool-runtime`` 不该被切开）
    * 数字：整体保留（``3.11`` 是一个 token）
    * 汉字 uni-gram：单字，保证召回（"退款" 能命中 "退款单"）
    * 汉字 bi-gram：双字，保证精度（避免 "上海" 命中 "下海"）
    """
    text = normalize_text(text)
    if not text:
        return []

    tokens: list[str] = []
    tokens.extend(m.lower() for m in _LATIN_WORD.findall(text))
    tokens.extend(_NUMBER.findall(text))

    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        if len(run) >= 2:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    if not keep_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS]
        # 极端情况下（整句都是停用词）退回原始 token，避免空 query
        tokens = filtered or tokens
    return tokens


def tokenize_all(texts: Sequence[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


def tokenize_iter(text: str) -> Iterator[str]:
    yield from tokenize(text)


def estimate_tokens(text: str) -> int:
    """估算一段文本占用的 token 数（向上取整）。

    中文按字计、英文按词计、标点打折。误差通常 <20%，
    足够支撑"上下文预算"这类用途。
    """
    if not text:
        return 0
    cjk = len(_CJK_CHAR.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    number = len(_NUMBER.findall(text))
    punct = len(_PUNCT.findall(text))
    raw = (
        cjk * _CJK_TOKEN_RATIO
        + (latin + number) * _LATIN_TOKEN_RATIO
        + punct * _PUNCT_TOKEN_RATIO
    )
    return max(1, math.ceil(raw))
