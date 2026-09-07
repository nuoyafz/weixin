"""轻量 SQLite 存储：会话记录与处理日志（对齐原版的 contacts/messages/cycles 设计）。

采用"每次操作新建连接"模式，避免跨线程锁竞争问题。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _SCRIPTS_DIR / "data" / "agent.db"

_db_path: Path = _DEFAULT_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_key  TEXT UNIQUE,
    name         TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    last_message TEXT DEFAULT '',
    last_decision TEXT DEFAULT '',
    status       TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_key TEXT,
    contact     TEXT,
    created_at  TEXT NOT NULL,
    sender      TEXT NOT NULL,
    msg_type    TEXT DEFAULT 'text',
    content     TEXT NOT NULL,
    intent      TEXT DEFAULT '',
    decision    TEXT DEFAULT '',
    is_sent_by_us INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_contact ON messages(contact_key, created_at);

CREATE TABLE IF NOT EXISTS cycles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    contact     TEXT DEFAULT '',
    contact_key TEXT DEFAULT '',
    decision    TEXT DEFAULT '',
    intent      TEXT DEFAULT '',
    confidence  REAL DEFAULT 0,
    sent        INTEGER DEFAULT 0,
    skip_reason TEXT DEFAULT '',
    reply_text  TEXT DEFAULT '',
    error       TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_cycles_created_at ON cycles(created_at);

CREATE TABLE IF NOT EXISTS sent_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    contact     TEXT DEFAULT '',
    reply_text  TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    UNIQUE(contact_key, fingerprint)
);

CREATE TABLE IF NOT EXISTS leads (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_key    TEXT UNIQUE NOT NULL,
    contact        TEXT DEFAULT '',
    name           TEXT DEFAULT '',
    intent         TEXT DEFAULT '',
    stage          TEXT DEFAULT 'new',
    last_message   TEXT DEFAULT '',
    last_message_at TEXT DEFAULT '',
    updated_at     TEXT NOT NULL,
    note           TEXT DEFAULT '',
    is_followed_up INTEGER DEFAULT 0,
    follow_up_at   TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_leads_stage ON leads(stage);
CREATE INDEX IF NOT EXISTS idx_leads_updated ON leads(updated_at);

CREATE TABLE IF NOT EXISTS file_refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT DEFAULT '',
    stored_path TEXT NOT NULL,
    name        TEXT NOT NULL,
    size        INTEGER DEFAULT 0,
    kind        TEXT DEFAULT 'doc',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_file_refs_name ON file_refs(name);

-- 聊天历史快照（2026-09-06 方哥要求：每次识别到的聊天记录都存下来，
-- UI 里按真实微信会话窗口的样子回看）。一次识别 = 一个 session。
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    contact     TEXT NOT NULL,
    contact_key TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    msg_count   INTEGER DEFAULT 0,
    intent      TEXT DEFAULT '',
    reply_text  TEXT DEFAULT '',
    confidence  REAL DEFAULT 0,
    source      TEXT DEFAULT 'ocr'
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created ON chat_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_contact ON chat_sessions(contact_key, created_at);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    side       TEXT DEFAULT 'left',      -- left=对方 right=自己 center=系统提示
    sender     TEXT DEFAULT '',
    content    TEXT NOT NULL,
    msg_type   TEXT DEFAULT 'text',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);
"""

# contacts 表控制字段迁移（黑名单/白名单/回复计数/置信度覆盖），幂等，
# 安全忽略已存在的列。is_whitelisted / reply_count / min_confidence_override
# 由 ContactControlRepo（send_guard）使用，旧库缺列会导致
# "no such column: min_confidence_override" 而中断整轮发送。
_CONTROL_COLUMNS = (
    ("blacklist_reason", "TEXT DEFAULT ''"),
    ("is_blacklisted", "INTEGER DEFAULT 0"),
    ("is_removed", "INTEGER DEFAULT 0"),
    ("is_whitelisted", "INTEGER DEFAULT 0"),
    ("reply_count", "INTEGER DEFAULT 0"),
    ("min_confidence_override", "REAL"),
)


def _migrate_contacts(conn: sqlite3.Connection) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(contacts)").fetchall()}
    for name, ddl in _CONTROL_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE contacts ADD COLUMN {name} {ddl}")


def configure(path: Optional[str] = None) -> None:
    global _db_path
    if path:
        _db_path = Path(path)


@contextmanager
def _get_conn():
    """上下文管理器：创建并自动关闭 SQLite 连接。"""
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(_SCHEMA)
        _migrate_contacts(conn)
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_conn():
    """公开的连接上下文管理器（send_guard 等防重模块使用）。"""
    return _get_conn()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upsert_contact(name: str, contact_key: str = "") -> None:
    key = contact_key or name or "unknown"
    with _get_conn() as conn:
        ts = now_str()
        conn.execute(
            "INSERT INTO contacts(contact_key,name,first_seen_at,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(contact_key) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
            (key, name or key, ts, ts),
        )


def save_message(contact: str, content: str, sender: str = "customer",
                 msg_type: str = "text", contact_key: str = "",
                 intent: str = "", decision: str = "", is_sent_by_us: bool = False) -> None:
    if not content:
        return
    key = contact_key or contact or "unknown"
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages(contact_key,contact,created_at,sender,msg_type,content,intent,decision,is_sent_by_us) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (key, contact or key, now_str(), sender, msg_type, content[:2000], intent, decision,
             int(is_sent_by_us)),
        )


def record_cycle(contact: str, decision: str = "", intent: str = "", confidence: float = 0.0,
                 sent: bool = False, skip_reason: str = "", reply_text: str = "",
                 contact_key: str = "", error: str = "") -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO cycles(created_at,contact,contact_key,decision,intent,confidence,sent,skip_reason,reply_text,error) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (now_str(), contact or "", contact_key or contact or "", decision, intent,
             confidence, int(sent), skip_reason, reply_text[:2000], error or ""),
        )
        return int(cur.lastrowid)


def recent_cycles(limit: int = 50) -> List[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def recent_counters() -> dict:
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM cycles").fetchone()["c"]
        sent = conn.execute("SELECT COUNT(*) c FROM cycles WHERE sent=1").fetchone()["c"]
        contacts = conn.execute("SELECT COUNT(*) c FROM contacts").fetchone()["c"]
        return {"total_cycles": total, "sent": sent, "contacts": contacts}


# ================================================================
# 聊天历史记录
# 需求（方哥 2026-09-06）：每次识别到的聊天记录都存下来，且在 UI 里
# 按真实微信会话窗口的样子回看。旧表 save_message/record_cycle 定义了
# 却从未被调用，运行时并不落任何聊天记录，故此处新增专用存储。
# ================================================================

def _msg_field(item, name, default=""):
    """消息元素兼容两种结构：OcrMessage 对象（属性）或 dict（键）。"""
    v = item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)
    return default if v is None else v


def _msg_text(item) -> str:
    if isinstance(item, dict):
        raw = item.get("text") or item.get("content") or ""
    else:
        raw = getattr(item, "text", "") or getattr(item, "content", "") or ""
    return str(raw).strip()


def save_chat_session(contact: str, messages, contact_key: str = "",
                      intent: str = "", reply_text: str = "",
                      confidence: float = 0.0, source: str = "ocr") -> int:
    """把一轮识别到的整段会话存为历史快照，返回 session_id（空会话不落库）。

    messages：visible_conversation_messages / customer_turn_messages，
    元素可为 OcrMessage（side/sender/text）或 dict。
    """
    if not messages:
        return 0
    key = contact_key or contact or "unknown"
    ts = now_str()
    rows = []
    for m in messages:
        text = _msg_text(m)
        if not text:
            continue
        side = str(_msg_field(m, "side", "") or "").strip().lower()
        sender = str(_msg_field(m, "sender", "") or "").strip()
        if not side:
            # side 缺失时按 sender 推断：自己发的靠右（微信习惯）
            side = "right" if sender in ("我", "self", "me", "assistant") else "left"
        rows.append((side, sender, text[:2000], "text", ts))
    if not rows:
        return 0
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_sessions(contact,contact_key,created_at,msg_count,"
            "intent,reply_text,confidence,source) VALUES(?,?,?,?,?,?,?,?)",
            (contact or key, key, ts, len(rows), intent or "",
             (reply_text or "")[:2000], float(confidence or 0.0), source or "ocr"),
        )
        sid = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO chat_messages(session_id,side,sender,content,msg_type,created_at) "
            "VALUES(?,?,?,?,?,?)",
            [(sid, s, sd, c, t, ct) for (s, sd, c, t, ct) in rows],
        )
        return sid


def list_chat_sessions(limit: int = 100, offset: int = 0,
                       contact_key: str = "") -> List[dict]:
    """历史会话列表（倒序），附带最后一条消息预览。"""
    try:
        limit = max(1, min(int(limit or 100), 500))
    except Exception:
        limit = 100
    try:
        offset = max(0, int(offset or 0))
    except Exception:
        offset = 0
    sql = ("SELECT s.*, (SELECT m.content FROM chat_messages m "
           "WHERE m.session_id=s.id ORDER BY m.id DESC LIMIT 1) AS preview "
           "FROM chat_sessions s ")
    args: list = []
    if contact_key:
        sql += "WHERE s.contact_key=? "
        args.append(contact_key)
    sql += "ORDER BY s.id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def get_chat_session(session_id) -> Optional[dict]:
    """取单个历史会话及其全部消息（按原始顺序）。"""
    try:
        sid = int(session_id)
    except Exception:
        return None
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            return None
        msgs = conn.execute(
            "SELECT side,sender,content,msg_type,created_at FROM chat_messages "
            "WHERE session_id=? ORDER BY id", (sid,)).fetchall()
        return {"session": dict(row), "messages": [dict(m) for m in msgs]}


def clear_chat_history(contact_key: str = "") -> int:
    """清空历史记录（不传 contact_key 则全清），返回删除的会话数。"""
    with _get_conn() as conn:
        if contact_key:
            rows = conn.execute("SELECT id FROM chat_sessions WHERE contact_key=?",
                                (contact_key,)).fetchall()
        else:
            rows = conn.execute("SELECT id FROM chat_sessions").fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            return 0
        q = ",".join("?" * len(ids))
        conn.execute("DELETE FROM chat_messages WHERE session_id IN (%s)" % q, ids)
        conn.execute("DELETE FROM chat_sessions WHERE id IN (%s)" % q, ids)
        return len(ids)
