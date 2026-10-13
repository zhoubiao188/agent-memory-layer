#!/usr/bin/env python3
"""端到端演示：一个"有长期记忆"的 Agent 的一天。

    python scripts/demo.py

全程离线运行（无需 API Key）：向量化用确定性的 HashingEmbedder，
"模型"用一个确定性 responder 代替。但每一步的**接口与真实系统一致**，
换成 OpenAI / 本地模型只需要替换 embedding 与 responder 两个实现。

六个场景，对应文章里的六个小节：

1. 一轮对话被抽成四类记忆
2. 写入策略：事实取代 vs 约束累积
3. 混合检索的分数拆解
4. token 预算内的上下文组装
5. 记忆如何改变下一轮回答
6. 六十天后的记忆巩固
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_memory import (  # noqa: E402
    DAY,
    HOUR,
    Consolidator,
    HashingEmbedder,
    ManualClock,
    MemoryAugmentedAgent,
    MemoryItem,
    MemoryKind,
    MemoryManager,
    MemoryStore,
    ScriptedResponder,
    WriteAction,
    WritePolicy,
)
from agent_memory.agent import kind_breakdown, summarize_writes  # noqa: E402

WIDTH = 78


def hr(title: str = "") -> None:
    if title:
        print(f"\n{'=' * WIDTH}\n  {title}\n{'=' * WIDTH}")
    else:
        print("-" * WIDTH)


def kv(label: str, value: object, pad: int = 22) -> None:
    print(f"  {label:<{pad}}{value}")


def new_runtime() -> tuple[ManualClock, MemoryManager]:
    clock = ManualClock(1_700_000_000.0)  # 2023-11-15 06:13:20 +08:00
    embedder = HashingEmbedder(dimension=512)
    store = MemoryStore(":memory:")
    policy = WritePolicy()

    # 阈值必须和 embedder 匹配。哈希 embedder 是"词面级别"的，
    # 会明显压缩真实语义的相似度分布：两条真正的同义表述只有 0.81，
    # 而真实语义向量通常在 0.95 以上。所以这里把离线合并阈值调到 0.80。
    # 这一段是全文最容易被忽略的工程细节——阈值不是常数，是校准量。
    consolidator = Consolidator(
        store, embedder, policy, merge_threshold=0.80, summarize_min_cluster=3
    )
    memory = MemoryManager(
        store=store,
        embedder=embedder,
        policy=policy,
        consolidator=consolidator,
        clock=clock,
    )
    return clock, memory


# ---------------------------------------------------------------- 场景 1
def act1_extraction(memory: MemoryManager) -> None:
    hr("场景 1 · 一轮对话被抽成四类记忆")
    turns = [
        "记住，回答尽量简短，不要长篇大论。",
        "我更喜欢用中文和我说话，专业术语可以保留英文。",
        "我叫周彪，是后端工程师，平时主要写 Java。",
        "我们项目现在用 MySQL 8 做订单库。",
        "以后都走灰度发布，上线前必须有回滚开关。",
        "今天上线了新的对账服务，遇到了连接池打满的问题。",
        "嗯嗯，好的。",
    ]
    print(f"  用户连续说了 {len(turns)} 句话，抽取与写入结果：\n")
    print(f"  {'ACTION':<10}{'KIND':<11}{'SCORE':<8}REASON / TEXT")
    print(f"  {'-' * (WIDTH - 4)}")
    for text in turns:
        for d in memory.observe(text):
            flag = "" if d.writes else "  (丢弃)"
            print(
                f"  {d.action.value:<10}{d.draft.kind.value:<11}"
                f"{d.importance:<8.2f}{d.reason}{flag}"
            )
            print(f"  {'':<29}└─ {d.draft.text}")
    print()
    kv("写入统计", summarize_writes(memory.decisions))
    kv("活跃记忆条数", len(memory.memories()))
    kv("类型分布", kind_breakdown(memory))


# ---------------------------------------------------------------- 场景 2
def act2_write_policy(memory: MemoryManager) -> None:
    hr("场景 2 · 写入策略：事实取代 vs 约束累积")
    print("  关键设计：只有**单值事实**才谈得上取代；约束和偏好是可累积的。\n")

    print("  (a) 技术选型变了 → 旧事实必须失效：")
    before = len(memory.memories())
    for text in ("我们决定把订单库换成 PostgreSQL 16。",):
        for d in memory.observe(text):
            print(f"      {d.action.value.upper():<10}{d.reason}")
    kv("    活跃记忆", f"{before} → {len(memory.memories())}（新增 1 条，旧条目转 SUPERSEDED）")
    stale = [i for i in memory.memories(include_inactive=True) if "MySQL" in i.text]
    for item in stale:
        print(f"      ↳ {item.status.value:<11}{item.text}")
    print()

    print("  (b) 又说了两条规范 → 与已有规范共存，而不是后者覆盖前者：")
    for text in (
        "不要使用表情符号，正式场合要严肃。",
        "另外，所有对外接口必须带幂等键。",
    ):
        for d in memory.observe(text):
            print(
                f"      {d.action.value.upper():<10}{d.draft.subject}/"
                f"{d.draft.predicate} = {d.draft.object}"
            )
    rules = [i for i in memory.memories() if i.kind is MemoryKind.PROCEDURAL]
    for item in rules:
        print(f"      ✓ active   {item.text}")
    kv("    规范条数", len(rules))
    print("      说明：三条规范落在同一个 (团队, 约束) 槽位上，但都活着——")
    print("      因为约束是**可累积**的，只有单值事实才允许被取代。")
    print()

    print("  (c) 用户重复说了同一件事 → 不当成新记忆：")
    for d in memory.observe("以后都走灰度发布，上线前必须有回滚开关。"):
        print(f"      {d.action.value.upper():<10}{d.reason}  sim={d.similarity:.4f}")
    print()
    kv("当前活跃记忆", len(memory.memories()))
    print()
    kv("追溯来源（审计事件）", "")
    for ev in memory.events(limit=6):
        detail = ev["detail"] or {}
        extra = detail.get("reason") or detail.get("kind") or ""
        print(f"      #{ev['seq']:<4}{ev['action']:<11}{extra}")


# ---------------------------------------------------------------- 场景 3
def act3_retrieval(memory: MemoryManager) -> None:
    hr("场景 3 · 混合检索的分数拆解")
    queries = ["订单库现在用什么？", "发布要注意什么？", "回答风格有什么要求？"]
    for query in queries:
        print(f"\n  query: 「{query}」")
        bundle = memory.recall(query, top_k=3)
        print(
            f"  {'SCORE':<8}{'SIM':<8}{'BM25':<8}{'RECENCY':<9}{'IMPORT':<8}TEXT"
        )
        print(f"  {'-' * (WIDTH - 4)}")
        for hit in bundle.hits:
            c = hit.components
            print(
                f"  {hit.score:<8.4f}{c['similarity']:<8.3f}{c['lexical']:<8.3f}"
                f"{c['recency']:<9.3f}{c['importance']:<8.3f}{hit.item.text[:30]}"
            )
        print(f"  → 召回 {len(bundle.hits)} 条，访问计数已 +1（访问频次会反哺后续排序）")


# ---------------------------------------------------------------- 场景 4
def act4_budget(memory: MemoryManager) -> None:
    hr("场景 4 · token 预算内的上下文组装")
    for budget in (200, 60):
        bundle = memory.recall("灰度发布要注意什么？", top_k=6, max_tokens=budget)
        ctx = bundle.context
        assert ctx is not None
        print(f"\n  ── 预算 {ctx.budget} tokens ──────────────────────────────")
        for line in ctx.text.splitlines():
            print(f"  │ {line}")
        kv("  占用", f"{ctx.tokens_used} / {ctx.budget}  ({ctx.utilization:.0%})")
        kv("  入块 / 丢弃", f"{len(ctx.included_ids)} / {len(ctx.dropped_ids)}")
        kv("  被截断", ctx.truncated)
        if ctx.dropped_ids:
            print("  说明：放不下的整条丢弃，而不是截半句话——半截记忆比没记忆更危险。")


# ---------------------------------------------------------------- 场景 5
def act5_agent(memory: MemoryManager, clock: ManualClock) -> None:
    hr("场景 5 · 记忆如何改变下一轮回答")
    agent = MemoryAugmentedAgent(
        memory, ScriptedResponder(), context_max_tokens=280, top_k=5
    )

    print("  会话 A（同一进程内继续）\n")
    reply = agent.chat("灰度发布具体怎么落地？", turn_id="t100")
    kv("  召回条数", len(reply.hit_ids))
    print("  ── 注入的上下文 ──")
    for line in reply.context.splitlines():
        print(f"  │ {line}")
    print("  ── 模型回答 ──")
    for line in reply.text.splitlines():
        print(f"  │ {line}")

    print("\n  会话 B（三天后，进程重启，记忆从 SQLite 重新加载）\n")
    clock.advance_days(3)
    reply2 = agent.chat("再讲讲发布流程。", turn_id="t101")
    print(f"  上下文里带上了「简短」偏好：{'简短' in reply2.context}")
    for line in reply2.text.splitlines():
        print(f"  │ {line}")


# ---------------------------------------------------------------- 场景 6
def act6_consolidation(memory: MemoryManager, clock: ManualClock) -> None:
    hr("场景 6 · 记忆巩固：遗忘、合并、归纳")
    print("  先塞进一批「写入策略管不到」的历史记忆。")
    print("  真实来源：旧系统迁移、批量导入、换过 embedder 后没重建索引的数据。\n")

    base = clock.now()
    legacy = [
        MemoryItem(
            kind=MemoryKind.SEMANTIC,
            text="订单库使用 PostgreSQL 16 作为主库",
            created_at=base - 2 * DAY,
            updated_at=base - 2 * DAY,
            importance=0.6,
        ),
        MemoryItem(
            kind=MemoryKind.SEMANTIC,
            text="订单库使用 PostgreSQL 16 做主库",
            created_at=base - 2 * DAY,
            updated_at=base - 2 * DAY,
            importance=0.6,
        ),
        *[
            MemoryItem(
                kind=MemoryKind.EPISODIC,
                text=text,
                subject="事件",
                predicate="经历",
                object=text,
                created_at=base - DAY,
                updated_at=base - DAY,
                importance=0.65,
            )
            for text in (
                "处理了一次连接池被打满的告警",
                "处理了一次慢查询引起的接口超时",
                "处理了一次磁盘写满导致的写入失败",
            )
        ],
        # 一条孤立的经历：没有同类可以归纳，只能自然衰减——用来演示"遗忘"
        MemoryItem(
            kind=MemoryKind.EPISODIC,
            text="凌晨三点被一次 P1 告警叫醒",
            subject="值班",
            predicate="经历",
            object="凌晨三点被一次 P1 告警叫醒",
            created_at=base - DAY,
            updated_at=base - DAY,
            importance=0.65,
        ),
    ]
    for item in legacy:
        memory.store.add(item)
    kv("  导入历史记忆", f"{len(legacy)} 条")
    from agent_memory import cosine  # noqa: PLC0415  — 仅为演示打印相似度

    pair = (legacy[0].text, legacy[1].text)
    sim = cosine(memory.embedder.embed_one(pair[0]), memory.embedder.embed_one(pair[1]))
    print(f"  合并检测：「{pair[0]}」 vs 「{pair[1]}」")
    print(f"            sim={sim:.3f}，离线合并阈值={memory.consolidator.merge_threshold}")

    before = memory.stats()
    kv("  巩固前 / 总条数", before.total)
    kv("  活跃 / 已取代", f"{before.active} / {before.superseded}")

    print("\n  ── 第 1 次巩固（5 天后）────────────────────────────")
    clock.advance_days(5)
    report1 = memory.consolidate()
    for key, value in report1.as_dict().items():
        kv(f"    {key}", value)
    print("    归纳产物：")
    for item in memory.memories(include_inactive=True):
        if "consolidated" in item.tags:
            print(f"      [{item.kind.value}] {item.text}")

    print("\n  ── 第 2 次巩固（再 55 天后）────────────────────────")
    clock.advance_days(55)
    report2 = memory.consolidate()
    for key, value in report2.as_dict().items():
        kv(f"    {key}", value)

    after = memory.stats()
    kv("  巩固后 / 总条数", after.total)
    kv("  活跃 / 已取代 / 已过期", f"{after.active} / {after.superseded} / {after.expired}")
    print("\n  今天仍然可用的记忆（偏好与规范几乎不衰减）：")
    for item in memory.memories():
        print(
            f"      [{item.kind.value:<10} imp={item.importance:.2f} "
            f"hit={item.access_count}] {item.text[:42]}"
        )


def main() -> None:
    clock, memory = new_runtime()
    print("#" * WIDTH)
    print("#  agent-memory-layer · 端到端演示")
    print("#  离线可跑：无 API Key、无模型权重、无网络")
    print("#" * WIDTH)

    act1_extraction(memory)
    act2_write_policy(memory)
    act3_retrieval(memory)
    act4_budget(memory)
    act5_agent(memory, clock)
    act6_consolidation(memory, clock)

    hr("完成")
    stats = memory.stats()
    print(f"  最终：{stats.total} 条记忆（{stats.active} 活跃 / "
          f"{stats.superseded} 被取代 / {stats.expired} 已过期），"
          f"累计占用 {stats.tokens_stored} tokens 原文")
    print(f"  写入决策共 {len(memory.decisions)} 次，其中最关键的几条：")
    for d in memory.decisions:
        if d.action in {WriteAction.SUPERSEDE, WriteAction.MERGE}:
            print(f"      {d.action.value:<10}{d.reason}")
    memory.close()


if __name__ == "__main__":
    main()
