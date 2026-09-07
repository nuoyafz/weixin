"""消息仓库：按联系人 / 时间范围 / 发送方查询、统计未回复数、查询最近 N 条。

遵循"每次操作新建连接"模式。只读查询基于现有 messages 表。
"""
from __future__ import annotations

from typing import List, Optional

from . import db

SENDER_CUSTOMER = "customer"
SENDER_US = "us"


def _to_rows(rows) -> List[dict]:
    return [dict(r) for r in rows]


def query_messages(
    contact_key: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    sender: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """组合条件查询消息：联系人 / 发送方精确匹配，时间按区间过滤。"""
    clauses: List[str] = []
    params: list = []
    if contact_key:
        clauses.append("contact_key=?")
        params.append(contact_key)
    if sender:
        clauses.append("sender=?")
        params.append(sender)
    if since:
        clauses.append("created_at >= ?")
        params.append(since)
    if until:
        clauses.append("created_at <= ?")
        params.append(until)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM messages {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return _to_rows(rows)


def recent_messages(contact_key: str, n: int = 10) -> List[dict]:
    """查询某联系人最近的 N 条消息（升序返回，越旧在前）。"""
    if n <= 0:
        return []
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE contact_key=? "
            "ORDER BY id DESC LIMIT ?",
            (contact_key or "unknown", n),
        ).fetchall()
        return list(reversed(_to_rows(rows)))


def count_unreplied(contact_key: str, since: Optional[str] = None) -> int:
    """统计某联系人尚未回复的消息数：用户(商家)最新消息之后仍有客户未读消息。

    判定规则：自用户最近一条发送消息之后，客户新发来的消息即计为"未回复"。
    """
    key = contact_key or "unknown"
    with db.get_conn() as conn:
        last_us = conn.execute(
            "SELECT MAX(id) AS mid FROM messages "
            "WHERE contact_key=? AND sender=?",
            (key, SENDER_US),
        ).fetchone()["mid"]
        cond = ""
        params: list = [key, SENDER_CUSTOMER]
        if since:
            cond += " AND created_at >= ?"
            params.append(since)
        if last_us is not None:
            cond += " AND id > ?"
            params.append(last_us)
        cnt = conn.execute(
            f"SELECT COUNT(*) AS n FROM messages "
            f"WHERE contact_key=? AND sender=?{cond}",
            params,
        ).fetchone()["n"]
        return int(cnt)


def count_unreplied_many(
    contact_keys: List[str], since: Optional[str] = None
) -> dict:
    """批量统计多个联系人未回复数，返回 {contact_key: count}。"""
    out: dict = {}
    keys = [k for k in (contact_keys or []) if k]
    if not keys:
        return out
    with db.get_conn() as conn:
        marks = ",".join("?" for _ in keys)
        # 每个联系人最近的商家消息 id
        lhs = conn.execute(
            f"SELECT contact_key, MAX(id) AS mid FROM messages "
            f"WHERE sender=? AND contact_key IN ({marks}) GROUP BY contact_key",
            (SENDER_US, *keys),
        ).fetchall()
        last_us = {r["contact_key"]: r["mid"] for r in lhs}
        if since:
            rows = conn.execute(
                f"SELECT contact_key, COUNT(*) AS nn FROM ("
                f"SELECT contact_key, id FROM messages "
                f"WHERE sender=? AND created_at >= ? AND contact_key IN ({marks})) "
                f"GROUP BY contact_key",
                (SENDER_CUSTOMER, since, *keys),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT contact_key, COUNT(*) AS nn FROM messages "
                f"WHERE sender=? AND contact_key IN ({marks}) GROUP BY contact_key",
                (SENDER_CUSTOMER, *keys),
            ).fetchall()
        base = {r["contact_key"]: int(r["nn"]) for r in rows}
        for k in keys:
            b = base.get(k, 0)
            lid = last_us.get(k)
            if lid is None:
                out[k] = b
                continue
            if since:
                mid = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE contact_key=? "
                    "AND sender=? AND id > ? AND created_at >= ?",
                    (k, SENDER_CUSTOMER, lid, since),
                ).fetchone()["n"]
            else:
                mid = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE contact_key=? "
                    "AND sender=? AND id > ?",
                    (k, SENDER_CUSTOMER, lid),
                ).fetchone()["n"]
            out[k] = int(mid)
    return out


def last_message(contact_key: str) -> Optional[dict]:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE contact_key=? ORDER BY id DESC LIMIT 1",
            (contact_key or "unknown",),
        ).fetchone()
        return dict(row) if row else None


def messages_since(since: str, limit: int = 500) -> List[dict]:
    """自某个时间点以来的全部客户消息（用于增量抓取）。"""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE created_at >= ? "
            "ORDER BY id ASC LIMIT ?",
            (since, limit),
        ).fetchall()
        return _to_rows(rows)


class MessagesRepo:
    """消息仓库：包装模块级函数为类接口。"""

    def __init__(self):
        pass

    def query_messages(self, contact_key: Optional[str] = None,
                       since: Optional[str] = None,
                       until: Optional[str] = None,
                       sender: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> List[dict]:
        return query_messages(contact_key=contact_key, since=since,
                              until=until, sender=sender,
                              limit=limit, offset=offset)

    def recent_messages(self, contact_key: str, n: int = 10) -> List[dict]:
        return recent_messages(contact_key, n=n)

    def count_unreplied(self, contact_key: str,
                        since: Optional[str] = None) -> int:
        return count_unreplied(contact_key, since=since)

    def count_unreplied_many(self, contact_keys: List[str],
                             since: Optional[str] = None) -> dict:
        return count_unreplied_many(contact_keys, since=since)

    def last_message(self, contact_key: str) -> Optional[dict]:
        return last_message(contact_key)

    def messages_since(self, since: str, limit: int = 500) -> List[dict]:
        return messages_since(since, limit=limit)


class CyclesRepo:
    """对话周期仓库：记录每次观察循环的日志。"""

    def __init__(self):
        pass

    def log_cycle(self, contact_key: str, status: str = "",
                  summary: str = "") -> None:
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO cycles(contact_key, status, summary, created_at) "
                "VALUES(?,?,?,?)",
                (contact_key, status, summary, db.now_str()),
            )

    def list_cycles(self, contact_key: Optional[str] = None,
                    limit: int = 50) -> List[dict]:
        with db.get_conn() as conn:
            if contact_key:
                rows = conn.execute(
                    "SELECT * FROM cycles WHERE contact_key=? "
                    "ORDER BY id DESC LIMIT ?",
                    (contact_key, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cycles ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]