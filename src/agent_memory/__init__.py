"""agent-memory-layer：给 AI Agent 加一层可运维的长期记忆。

```python
from agent_memory import MemoryManager

mem = MemoryManager()
mem.observe("记住，回答尽量简短。", role="user")
mem.observe("我们决定把订单库换成 PostgreSQL 16。", role="user")

bundle = mem.recall("订单库现在用什么？")
print(bundle.text)
```

分层：

* :mod:`agent_memory.store`        —— SQLite 持久层（软删除 + 事件审计）
* :mod:`agent_memory.policy`       —— 写入策略（打分 / 去重 / 事实取代）
* :mod:`agent_memory.extract`      —— 规则抽取（可替换为 LLM 策略）
* :mod:`agent_memory.retrieve`     —— 混合检索（向量 + BM25 + 时效 + MMR）
* :mod:`agent_memory.budget`       —— token 预算内组装上下文
* :mod:`agent_memory.consolidate`  —— 遗忘 / 合并 / 归纳
* :mod:`agent_memory.manager`      —— 门面
"""

from __future__ import annotations

from .agent import (
    AgentReply,
    MemoryAugmentedAgent,
    Responder,
    ScriptedResponder,
    build_prompt,
)
from .bm25 import BM25Index, min_max_normalize, reciprocal_rank_fusion
from .budget import AssembledContext, ContextAssembler
from .clock import MINUTE, HOUR, DAY, Clock, ManualClock, SystemClock
from .consolidate import Consolidator
from .embedder import Embedder, HashingEmbedder, cosine, normalize_vector
from .errors import (
    AgentMemoryError,
    BudgetExceededError,
    EmbeddingDimensionError,
    MemoryNotFoundError,
    MemoryValidationError,
    PolicyError,
    StoreError,
)
from .extract import (
    SLOT_NOUNS,
    ExtractionStrategy,
    RuleBasedExtractor,
    Turn,
    derive_slot,
    split_sentences,
)
from .manager import MemoryManager, RecallBundle
from .models import (
    ConsolidationReport,
    MemoryDraft,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    MemoryStats,
    MemoryStatus,
    RecallHit,
    WriteAction,
    WriteDecision,
)
from .policy import DEFAULT_HALF_LIFE_S, SINGLE_VALUED_KINDS, WritePolicy
from .retrieve import HybridRetriever, RetrievalWeights
from .store import MemoryStore
from .text import estimate_tokens, normalize_text, tokenize

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 核心
    "MemoryManager",
    "RecallBundle",
    "MemoryStore",
    "MemoryItem",
    "MemoryDraft",
    "MemoryKind",
    "MemoryStatus",
    "MemoryQuery",
    "MemoryStats",
    "RecallHit",
    "ConsolidationReport",
    "WriteAction",
    "WriteDecision",
    # 策略与抽取
    "WritePolicy",
    "DEFAULT_HALF_LIFE_S",
    "SINGLE_VALUED_KINDS",
    "RuleBasedExtractor",
    "ExtractionStrategy",
    "Turn",
    "split_sentences",
    "derive_slot",
    "SLOT_NOUNS",
    # 检索
    "HybridRetriever",
    "RetrievalWeights",
    "BM25Index",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    # 向量
    "Embedder",
    "HashingEmbedder",
    "cosine",
    "normalize_vector",
    # 上下文
    "ContextAssembler",
    "AssembledContext",
    # 巩固
    "Consolidator",
    # 时钟
    "Clock",
    "SystemClock",
    "ManualClock",
    "DAY",
    "HOUR",
    "MINUTE",
    # 文本
    "tokenize",
    "normalize_text",
    "estimate_tokens",
    # Agent 集成
    "MemoryAugmentedAgent",
    "AgentReply",
    "Responder",
    "ScriptedResponder",
    "build_prompt",
    # 异常
    "AgentMemoryError",
    "MemoryNotFoundError",
    "MemoryValidationError",
    "EmbeddingDimensionError",
    "BudgetExceededError",
    "StoreError",
    "PolicyError",
]
