"""联系人控制仓库：标记黑名单、移除、按状态查询活动联系人。

黑名单 / 移除状态持久化在 contacts 表新增的控制字段
（blacklist_reason / is_blacklisted / is_removed），由 db._migrate_contacts 幂等迁移。
遵循"每次操作新建连接"模式。
"""
from __future__ import annotations

from typing import List, Optional

from . import db


def mark_blacklist(contact_key: str, reason: str = "manual") -> bool:
    """将联系人标记为黑名单；返回是否发生更新。"""
    key = contact_key or "unknown"
    with db.get_conn() as conn:
        cur = conn.execute(
            "UPDATE contacts SET is_blacklisted=1, blacklist_reason=?, updated_at=? "
            "WHERE contact_key=?",
            (reason, db.now_str(), key),
        )
        if cur.rowcount:
            return True
        # 联系人不存在则补齐记录
        conn.execute(
            "INSERT INTO contacts(contact_key,name,first_seen_at,updated_at,"
            "is_blacklisted,blacklist_reason) VALUES(?,?,?,?,1,?) "
            "ON CONFLICT(contact_key) DO UPDATE SET is_blacklisted=1, "
            "blacklist_reason=excluded.blacklist_reason, updated_at=excluded.updated_at",
            (key, key, db.now_str(), db.now_str(), reason),
        )
        return True


def unmark_blacklist(contact_key: str) -> None:
    """取消联系人黑名单标记。"""
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE contacts SET is_blacklisted=0, blacklist_reason='', updated_at=? "
            "WHERE contact_key=?",
            (db.now_str(), contact_key or "unknown"),
        )


def is_blacklisted(contact_key: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT is_blacklisted FROM contacts WHERE contact_key=?",
            (contact_key or "unknown",),
        ).fetchone()
        return bool(row and row["is_blacklisted"])


def remove_contact(contact_key: str) -> bool:
    """软删除联系人（标记 is_removed）；返回是否存在该记录。"""
    with db.get_conn() as conn:
        cur = conn.execute(
            "UPDATE contacts SET is_removed=1, updated_at=? WHERE contact_key=? ",
            (db.now_str(), contact_key or "unknown"),
        )
        return cur.rowcount > 0


def restore_contact(contact_key: str) -> None:
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE contacts SET is_removed=0, updated_at=? WHERE contact_key=?",
            (db.now_str(), contact_key or "unknown"),
        )


def list_active_contacts(
    status: str = "active",
    include_banned: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> List[dict]:
    """按状态查询联系人。

    status 取值：active（正常）/ blacklisted（黑名单）/ removed（已移除）/ all。
    include_banned 为 True 时，active 查询也包含黑名单联系人（仍排除已移除）。
    """
    with db.get_conn() as conn:
        clauses: List[str] = []
        params: list = []
        if status == "active":
            clauses.append("is_removed=0")
            if not include_banned:
                clauses.append("COALESCE(is_blacklisted,0)=0")
        elif status == "blacklisted":
            clauses.append("COALESCE(is_blacklisted,0)=1")
        elif status == "removed":
            clauses.append("is_removed=1")
        # 'all' / 其它取值不做过滤
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM contacts {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def get_contact(contact_key: str) -> Optional[dict]:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM contacts WHERE contact_key=?", (contact_key or "unknown",)
        ).fetchone()
        return dict(row) if row else None


def blacklist_reasons() -> List[str]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT blacklist_reason FROM contacts "
            "WHERE COALESCE(is_blacklisted,0)=1 AND blacklist_reason!=''"
        ).fetchall()
        return [r["blacklist_reason"] for r in rows]


# =====================================================================
# ContactControlRepo —— 对齐原版 app.storage.contact_control.ContactControlRepo
# =====================================================================

class ContactControlRepo:
    """联系人控制仓库 —— 封装白名单/黑名单/回复计数等操作。

    对齐原版 app.storage.contact_control.ContactControlRepo。
    """

    def __init__(self, db_conn=None):
        self._db_conn = db_conn

    def _get_conn(self):
        if callable(self._db_conn):
            return self._db_conn()
        return self._db_conn

    def is_whitelisted(self, contact_key: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT is_whitelisted FROM contacts WHERE contact_key=?",
                (contact_key or "unknown",),
            ).fetchone()
            return bool(row and row["is_whitelisted"])

    def is_blacklisted(self, contact_key: str) -> bool:
        return is_blacklisted(contact_key)

    def get_reply_count(self, contact_key: str) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT reply_count FROM contacts WHERE contact_key=?",
                (contact_key or "unknown",),
            ).fetchone()
            return int(row["reply_count"]) if row and row["reply_count"] else 0

    def add_whitelist(self, contact_key: str) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE contacts SET is_whitelisted=1, updated_at=? "
                "WHERE contact_key=?",
                (db.now_str(), contact_key or "unknown"),
            )
            return True

    def add_blacklist(self, contact_key: str) -> bool:
        return mark_blacklist(contact_key, reason="send_guard")

    def get_status(self, contact_name: str) -> Optional[str]:
        key = contact_name or "unknown"
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT is_blacklisted, is_whitelisted, is_removed, reply_count "
                "FROM contacts WHERE contact_key=?",
                (key,),
            ).fetchone()
            if not row:
                return None
            if row["is_removed"]:
                return "removed"
            if row["is_blacklisted"]:
                return "blacklisted"
            if row["is_whitelisted"]:
                return "whitelisted"
            return "normal"

    def get_effective_min_confidence(self, contact_name: str,
                                     default_val: float = 0.6) -> float:
        key = contact_name or "unknown"
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT min_confidence_override FROM contacts WHERE contact_key=?",
                (key,),
            ).fetchone()
            if row and row["min_confidence_override"] is not None:
                return float(row["min_confidence_override"])
        return float(default_val)