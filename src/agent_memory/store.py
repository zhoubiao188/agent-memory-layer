"""持久层：SQLite 单文件存储。

几个刻意的取舍：

1. **不物理删除**。遗忘只是把 ``status`` 置为 ``expired``。
   记忆系统的正确性事故（"它把我上个月说的偏好忘了"）需要能复盘，
   物理删除等于销毁证据。
2. **一行一条**，不做 key-value blob。记忆需要按 ``subject/predicate`` 建索引、
   按 ``status`` 过滤、按 ``created_at`` 排序——这些查询交给 SQL 比在 Python 里全表扫描划算。
3. **事件表单独落**。每次写入/取代/过期都记一条 ``memory_events``，
   这样"为什么它记得这个"可以一行 SQL 查清楚。

线程安全用一把 ``RLock`` 兜住，连接用 ``check_same_thread=False``。
Agent 常跑在 ``ThreadPoolExecutor`` 里，工具的硬超时也依赖线程池（见上一篇文章），
所以这里必须能跨线程用。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import MemoryNotFoundError, StoreError
from .models import (
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    MemoryStats,
    new_id,
)
from .text import estimate_tokens

__all__ = ["MemoryStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    text             TEXT NOT NULL,
    subject          TEXT,
    predicate        TEXT,
    "object"         TEXT,
    importance       REAL NOT NULL,
    confidence       REAL NOT NULL,
    status           TEXT NOT NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    last_accessed_at REAL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    ttl_s            REAL,
    superseded_by    TEXT,
    tags             TEXT NOT NULL DEFAULT '[]',
    source_turns     TEXT NOT NULL DEFAULT '[]',
    metadata         TEXT NOT NULL DEFAULT '{}',
    embedding        TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

CREATE TABLE IF NOT EXISTS memory_events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    at        REAL NOT NULL,
    action    TEXT NOT NULL,
    memory_id TEXT,
    detail    TEXT
);
"""

_COLUMNS = (
    "id, kind, text, subject, predicate, \"object\", importance, confidence, status, "
    "created_at, updated_at, last_accessed_at, access_count, ttl_s, superseded_by, "
    "tags, source_turns, metadata, embedding"
)


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        kind=MemoryKind(row["kind"]),
        text=row["text"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        importance=row["importance"],
        confidence=row["confidence"],
        status=MemoryStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        ttl_s=row["ttl_s"],
        superseded_by=row["superseded_by"],
        tags=json.loads(row["tags"] or "[]"),
        source_turns=json.loads(row["source_turns"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
    )


class MemoryStore:
    """记忆仓储。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - 环境相关
            raise StoreError(f"failed to open store at {self.path}: {exc}") from exc
        self._lock = threading.RLock()

    # ---------- 内部 ----------

    def _log(self, action: str, memory_id: str | None, detail: Any, at: float) -> None:
        self._conn.execute(
            "INSERT INTO memory_events (at, action, memory_id, detail) VALUES (?,?,?,?)",
            (at, action, memory_id, json.dumps(detail, ensure_ascii=False)),
        )

    # ---------- 写入 ----------

    def _insert_tuple(self, item: MemoryItem) -> tuple[Any, ...]:
        # 显式列出顺序，避免 model_dump() 的 exclude/字段顺序变化悄悄错位。
        return (
            item.id,
            item.kind.value,
            item.text,
            item.subject,
            item.predicate,
            item.object,
            item.importance,
            item.confidence,
            item.status.value,
            item.created_at,
            item.updated_at,
            item.last_accessed_at,
            item.access_count,
            item.ttl_s,
            item.superseded_by,
            json.dumps(item.tags, ensure_ascii=False),
            json.dumps(item.source_turns, ensure_ascii=False),
            json.dumps(item.metadata, ensure_ascii=False, default=str),
            json.dumps(item.embedding) if item.embedding is not None else None,
        )

    def add(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO memories ({_COLUMNS}) VALUES ("
                    + ",".join(["?"] * 19)
                    + ")",
                    self._insert_tuple(item),
                )
                self._log("add", item.id, {"kind": item.kind.value}, item.created_at)
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"add failed: {exc}") from exc
        return item

    def update(self, item: MemoryItem, *, now: float | None = None) -> MemoryItem:
        with self._lock:
            if self._raw_get(item.id) is None:
                raise MemoryNotFoundError(item.id)
            item.updated_at = now if now is not None else time.time()
            self._conn.execute(
                """
                UPDATE memories SET kind=?, text=?, subject=?, predicate=?, "object"=?,
                    importance=?, confidence=?, status=?, updated_at=?, last_accessed_at=?,
                    access_count=?, ttl_s=?, superseded_by=?, tags=?, source_turns=?,
                    metadata=?, embedding=?
                WHERE id=?
                """,
                (
                    item.kind.value,
                    item.text,
                    item.subject,
                    item.predicate,
                    item.object,
                    item.importance,
                    item.confidence,
                    item.status.value,
                    item.updated_at,
                    item.last_accessed_at,
                    item.access_count,
                    item.ttl_s,
                    item.superseded_by,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.source_turns, ensure_ascii=False),
                    json.dumps(item.metadata, ensure_ascii=False, default=str),
                    json.dumps(item.embedding) if item.embedding is not None else None,
                    item.id,
                ),
            )
            self._log("update", item.id, None, item.updated_at)
            self._conn.commit()
        return item

    def record_access(self, memory_id: str, now: float, increment: int = 1) -> None:
        """批量刷新召回次数。检索时不写库，召回结果返回后由上层统一调用一次。"""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + ?, last_accessed_at = ? WHERE id = ?",
                (increment, now, memory_id),
            )
            self._conn.commit()

    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        superseded_by: str | None = None,
        now: float | None = None,
        reason: str = "",
    ) -> None:
        with self._lock:
            if self._raw_get(memory_id) is None:
                raise MemoryNotFoundError(memory_id)
            at = now if now is not None else time.time()
            self._conn.execute(
                "UPDATE memories SET status=?, superseded_by=?, updated_at=? WHERE id=?",
                (status.value, superseded_by, at, memory_id),
            )
            self._log(
                "supersede" if status is MemoryStatus.SUPERSEDED else "expire",
                memory_id,
                {"reason": reason, "superseded_by": superseded_by},
                at,
            )
            self._conn.commit()

    def delete(self, memory_id: str, *, hard: bool = False) -> bool:
        """默认软删除（置 expired）。``hard=True`` 才真的 DELETE。"""
        if not hard:
            try:
                self.set_status(memory_id, MemoryStatus.EXPIRED, reason="manual")
            except MemoryNotFoundError:
                return False
            return True
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._log("delete", memory_id, None, time.time())
            self._conn.commit()
            return cur.rowcount > 0

    # ---------- 读取 ----------

    def _raw_get(self, memory_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,))
        return cur.fetchone()

    def get(self, memory_id: str) -> MemoryItem | None:
        with self._lock:
            row = self._raw_get(memory_id)
        return _row_to_item(row) if row else None

    def require(self, memory_id: str) -> MemoryItem:
        item = self.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        return item

    def list(
        self,
        *,
        kinds: Sequence[MemoryKind] | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        tags: Sequence[str] | None = None,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND status=?"
            params.append(status.value)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(k.value for k in kinds)
        if ids is not None:
            if not ids:
                return []
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        sql += " ORDER BY created_at ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def all(self, **kwargs: Any) -> list[MemoryItem]:
        """不限 status 的全量查询（审计/统计用）。"""
        kwargs.setdefault("status", None)
        return self.list(**kwargs)

    def find_by_key(
        self, subject: str, predicate: str, *, include_inactive: bool = False
    ) -> list[MemoryItem]:
        sql = (
            "SELECT * FROM memories WHERE lower(subject)=? AND lower(predicate)=?"
        )
        params: list[Any] = [subject.strip().lower(), predicate.strip().lower()]
        if not include_inactive:
            sql += " AND status='active'"
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def filter_tags(self, items: Sequence[MemoryItem], tags: Sequence[str]) -> list[MemoryItem]:
        """标签是"或"语义：命中任一标签即保留。"""
        wanted = {t.lower() for t in tags}
        return [i for i in items if wanted & {t.lower() for t in i.tags}]

    def events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, at, action, memory_id, detail FROM memory_events "
                "ORDER BY seq DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "seq": r["seq"],
                    "at": r["at"],
                    "action": r["action"],
                    "memory_id": r["memory_id"],
                    "detail": json.loads(r["detail"]) if r["detail"] else None,
                }
            )
        return out

    # ---------- 统计 ----------

    def stats(self) -> MemoryStats:
        items = self.all()
        by_kind: dict[str, int] = {}
        for it in items:
            by_kind[it.kind.value] = by_kind.get(it.kind.value, 0) + 1
        return MemoryStats(
            total=len(items),
            active=sum(1 for i in items if i.status is MemoryStatus.ACTIVE),
            superseded=sum(1 for i in items if i.status is MemoryStatus.SUPERSEDED),
            expired=sum(1 for i in items if i.status is MemoryStatus.EXPIRED),
            by_kind=by_kind,
            tokens_stored=sum(estimate_tokens(i.text) for i in items),
        )

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # ---------- 批量 / 维护 ----------

    def add_many(self, items: Iterable[MemoryItem]) -> int:
        n = 0
        for it in items:
            self.add(it)
            n += 1
        return n

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM memory_events")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- 导入导出 ----------

    def export(self) -> list[dict[str, Any]]:
        return [i.model_dump(mode="json") for i in self.all()]

    def import_items(self, rows: Iterable[dict[str, Any]]) -> int:
        n = 0
        for row in rows:
            data = dict(row)
            data.pop("embedding", None)
            data.setdefault("id", new_id())
            self.add(MemoryItem(**data))
            n += 1
        return n
