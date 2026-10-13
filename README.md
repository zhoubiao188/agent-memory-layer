# agent-memory-layer

给 AI Agent 加一层**可运维的长期记忆**：写入策略 / 事实取代 / 混合检索 / token 预算组装 / 遗忘与归并。

> 3296 行实现 + 2225 行测试，**309 个测试用例全部通过**，零 API Key、离线可跑。

```text
$ python -m pytest tests/
309 passed in 0.28s

$ python scripts/demo.py
... 六个场景的端到端演示 ...
```

## 它解决什么问题

把 Agent 的记忆做成"每轮对话都塞进向量库"很容易，跑三天必炸：

| 症状 | 根因 |
|---|---|
| 同一个偏好存了 12 遍 | 没有去重 |
| 用户改用 PostgreSQL 后，MySQL 那条还在，检索看运气 | 没有事实取代 |
| prompt 里 90% 是"好的""谢谢" | 没有重要性筛选 |
| 召回 20 条塞进上下文，关键约束被稀释 | 没有预算与分层 |
| 三个月后什么都查不出来 | 没有遗忘与归并 |

这个库把上面五件事各做成一个可测试的模块。

## 快速开始

```bash
pip install -e ".[dev]"
python -m pytest tests/        # 309 passed
python scripts/demo.py         # 离线端到端演示
```

```python
from agent_memory import MemoryManager

mem = MemoryManager()                       # 默认 SQLite 内存库
mem.observe("记住，回答尽量简短，不要长篇大论。", role="user")
mem.observe("我们项目现在用 MySQL 8 做订单库。", role="user")
mem.observe("我们决定把订单库换成 PostgreSQL 16。", role="user")

bundle = mem.recall("订单库现在用什么？")
print(bundle.text)
# ## 关于用户的已知记忆
# ### 必须遵守的约定
# - （刚刚）记住，回答尽量简短，不要长篇大论 [m:9b47b4]
# ### 相关背景
# - （刚刚）我们决定把订单库换成 PostgreSQL 16 [m:85c17f]
#
# 注意：MySQL 那条已经是 SUPERSEDED，不会再出现在上下文里。
```

接进 Agent 循环：

```python
from agent_memory import MemoryAugmentedAgent, ScriptedResponder

agent = MemoryAugmentedAgent(mem, ScriptedResponder())
agent.chat("记住，回答尽量简短。")     # 回合后写入记忆
agent.chat("灰度发布怎么落地？")       # 回合前召回 → 回答风格已按偏好改变
```

## 架构

```text
              ┌──────────────────────── MemoryManager ────────────────────────┐
   用户发言 ─▶  observe()                                                    │
              │    └─▶ RuleBasedExtractor   抽取 (subject, predicate, object)│
              │    └─▶ WritePolicy          打分 → 去重 → 取代 → 合并/丢弃   │
              │    └─▶ MemoryStore (SQLite) 落库 + 事件审计 + 软删除          │
              │                                                              │
   提问    ─▶  recall()                                                      │
              │    └─▶ HybridRetriever      向量 + BM25 + 时效 + 频次 → MMR  │
              │    └─▶ ContextAssembler     token 预算内分层装箱              │
              │    └─▶ 刷新访问统计（反哺下一次排序）                        │
              │                                                              │
   定时任务 ▶  consolidate()   过期 / 合并 / 归纳 / 衰减                     │
              └──────────────────────────────────────────────────────────────┘
```

| 模块 | 职责 | 关键设计 |
|---|---|---|
| `models.py` | 数据契约 | `subject/predicate/object` 三元组 + 生命周期状态机 |
| `text.py` | 中文友好分词 | 汉字 uni/bi-gram + 拉丁词；`blake2b` 保证确定性 |
| `embedder.py` | 向量化 | `HashingEmbedder` 零依赖可复现；接口与真模型一致 |
| `bm25.py` | 词面检索 | 补上向量对低频专有名词的盲区 |
| `store.py` | 持久层 | **不物理删除**，每次写入留审计事件 |
| `policy.py` | 写入策略 | 重要性打分、去重、**单值事实才取代** |
| `extract.py` | 记忆抽取 | 规则打底、LLM 可插拔（`ExtractionStrategy`） |
| `retrieve.py` | 混合检索 | 五路信号加权 + RRF + MMR 去冗余 |
| `budget.py` | 上下文组装 | 硬约束优先，超预算整条丢弃 |
| `consolidate.py` | 记忆巩固 | 过期 / 合并 / 归纳 / 衰减 |
| `manager.py` | 门面 | 6 个动作够 Agent 循环用 |

## 三个值得单独说的设计决定

### 1. 只有单值事实才谈得上"取代"

约束和偏好是**可累积**的。用户说了 5 条规范就该记住 5 条；
只有"某个槽位当前等于什么"这种事实，新值才有资格让旧值失效。

```python
SINGLE_VALUED_KINDS = frozenset({MemoryKind.SEMANTIC})
```

早期实现对所有类型都做 `(subject, predicate)` 冲突判定，结果
"不要长篇大论"被后来的"必须有回滚开关"取代掉了——两条都被抽成 `(团队, 约束)`。

### 2. 失败与放不下都要"记账"，而不是静默截断

* 放不下的记忆进入 `dropped_ids`，**不截半句话**——半截记忆比没有记忆更危险；
* 每次写入/取代/过期都落 `memory_events`，能回答"它为什么记得这个"；
* 遗忘是软删除（`status=expired`），可复盘。

### 3. 阈值与权重都不是常数，是校准量

`HashingEmbedder` 是词面级别的，会压缩相似度分布：两条真正的同义表述只有 **0.813**，
而真实语义向量通常在 0.95 以上。所以 `scripts/demo.py` 把离线合并阈值从默认 0.93 调到 0.80。

同理，哈希 embedder 的相似度**区分度差**（"完全不相关"的两条也能到 0.5），
所以提供了 `RetrievalWeights.lexical_heavy()` 预设，把话语权交给 BM25 与重要性。

**换了 embedder 就要重新标定阈值和权重**，否则要么永远合并不了，
要么把不同的事合并成一条，要么排序退化成"看起来都差不多"。

## 可替换的两个接口

```python
class Embedder(Protocol):        # 换 OpenAI / bge / voyage：只换这一个类
    dimension: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

class ExtractionStrategy(Protocol):   # 换 LLM 抽取：只换这一个类
    def extract(self, text: str, *, turn_id: str | None = None) -> list[MemoryDraft]: ...
```

下游（策略 / 检索 / 组装 / 巩固）一行都不用改。

## 测试

```text
tests/test_text.py         24   分词、token 估算、全角半角
tests/test_models.py       24   契约校验、生命周期、派生属性
tests/test_embedder.py     17   确定性、归一化、sublinear tf
tests/test_bm25.py         16   IDF、长度归一化、RRF、min-max
tests/test_store.py        29   CRUD、软删除、事件审计、导出导入
tests/test_policy.py       42   打分、去重、取代、累积、衰减、遗忘
tests/test_extract.py      35   12 类句式、最左匹配、槽位派生
tests/test_retrieve.py     34   过滤、五路信号、MMR、RRF、降级、权重预设
tests/test_budget.py       18   年龄渲染、分层、预算、丢弃
tests/test_consolidate.py  18   过期、合并、归纳、衰减、dry-run
tests/test_manager.py      33   观察、召回、取代、遗忘、统计
tests/test_agent.py        19   prompt 组装、跨会话记忆、风格改变
                          ─────
                          309
```

`manual clock + 注入式 embedder` 让所有涉及时间与相似度的分支都能写出确定性断言。

## License

MIT
