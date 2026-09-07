"""线索仓库：把客户 + 最近消息 + 意图 + 阶段整理成线索，持久化到 leads 表。

支持按阶段 / 更新时间查询、标记跟进。遵循"每次操作新建连接"模式。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import db

DEFAULT_STAGE = "new"
VALID_STAGES = ("new", "contacted", "interested", "negotiating", "converted", "lost")


def _norm_stage(stage: Optional[str]) -> str:
    s = (stage or DEFAULT_STAGE).strip().lower()
    return s if s in VALID_STAGES else DEFAULT_STAGE


def _recent_message(conn, contact_key: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT content, created_at, intent FROM messages "
        "WHERE contact_key=? ORDER BY id DESC LIMIT 1",
        (contact_key,),
    ).fetchone()
    return dict(row) if row else None


def upsert_lead(
    contact_key: str,
    contact: str = "",
    name: str = "",
    intent: str = "",
    stage: Optional[str] = None,
    note: str = "",
) -> int:
    """根据联系人与最近消息整理成一条线索；已存在则更新阶段/意图等信息。"""
    stage = _norm_stage(stage)
    with db.get_conn() as conn:
        msg = _recent_message(conn, contact_key or "unknown")
        last_message = (msg or {}).get("content", "")
        last_at = (msg or {}).get("created_at", "")
        intent = intent or (msg or {}).get("intent", "") or ""
        c = conn.execute(
            "SELECT id FROM leads WHERE contact_key=?",
            (contact_key or "unknown",),
        ).fetchone()
        ts = db.now_str()
        if c:
            conn.execute(
                "UPDATE leads SET contact=?, name=?, intent=?, stage=?, "
                "last_message=?, last_message_at=?, updated_at=?, note=? "
                "WHERE id=?",
                (contact or "", name or "", intent, stage, last_message,
                 last_at or ts, ts, note or "", c["id"]),
            )
            return int(c["id"])
        cur = conn.execute(
            "INSERT INTO leads(contact_key,contact,name,intent,stage,last_message,"
            "last_message_at,updated_at,note) VALUES(?,?,?,?,?,?,?,?,?)",
            (contact_key or "unknown", contact or "", name or "", intent, stage,
             last_message, last_at or ts, ts, note or ""),
        )
        return int(cur.lastrowid)


def set_stage(lead_id: int, stage: str) -> None:
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE leads SET stage=?, updated_at=? WHERE id=?",
            (_norm_stage(stage), db.now_str(), lead_id),
        )


def mark_followed_up(lead_id: int, followed_up: bool = True) -> None:
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE leads SET is_followed_up=?, follow_up_at=?, updated_at=? WHERE id=?",
            (int(followed_up), db.now_str() if followed_up else "", db.now_str(), lead_id),
        )


def _query_leads(
    where: str, params: list, limit: int = 100, offset: int = 0
) -> List[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM leads {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def list_leads(stage: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[dict]:
    """按阶段查询线索；stage 为空则返回全部。"""
    if stage:
        where, params = "WHERE stage=?", [_norm_stage(stage)]
    else:
        where, params = "", []
    return _query_leads(where, params, limit, offset)


def list_leads_by_updated(
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """按更新时间范围查询线索（since/until 为 'YYYY-MM-DD HH:MM:SS'，含边界）。"""
    clauses: List[str] = []
    params: list = []
    if since:
        clauses.append("updated_at >= ?")
        params.append(since)
    if until:
        clauses.append("updated_at <= ?")
        params.append(until)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return _query_leads(where, params, limit, offset)


def get_lead(lead_id: int) -> Optional[dict]:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        return dict(row) if row else None


def get_lead_by_contact(contact_key: str) -> Optional[dict]:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM leads WHERE contact_key=?", (contact_key or "unknown",)
        ).fetchone()
        return dict(row) if row else None


def lead_export() -> str:
    """导出全部线索为 JSON 字符串（便于备份/迁移）。"""
    with db.get_conn() as conn:
        rows = conn.execute("SELECT * FROM leads").fetchall()
        data = [dict(r) for r in rows]
        return json.dumps(data, ensure_ascii=False, default=str)


def lead_summary() -> Dict[str, int]:
    """按阶段统计线索数量。"""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT stage, COUNT(*) AS n FROM leads GROUP BY stage"
        ).fetchall()
        return {r["stage"]: int(r["n"]) for r in rows}


class LeadsRepo:
    """线索仓库：包装模块级函数为类接口。"""

    def __init__(self):
        pass

    def upsert_lead(self, contact_key: str, contact: str = "",
                    name: str = "", intent: str = "",
                    stage: Optional[str] = None, note: str = "") -> int:
        return upsert_lead(contact_key, contact=contact, name=name,
                           intent=intent, stage=stage, note=note)

    def set_stage(self, lead_id: int, stage: str) -> None:
        return set_stage(lead_id, stage)

    def mark_followed_up(self, lead_id: int, followed_up: bool = True) -> None:
        return mark_followed_up(lead_id, followed_up)

    def list_leads(self, stage: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> List[dict]:
        return list_leads(stage=stage, limit=limit, offset=offset)

    def get_lead(self, lead_id: int) -> Optional[dict]:
        return get_lead(lead_id)

    def get_lead_by_contact(self, contact_key: str) -> Optional[dict]:
        return get_lead_by_contact(contact_key)

    def lead_summary(self) -> Dict[str, int]:
        return lead_summary()