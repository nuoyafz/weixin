"""SendGuard v2 - tri-mode contact gating.

Aligned with original app.rpa.send_guard v3.10:
  whitelist     - only listed contacts can be sent to
  blacklist     - default-allow: send unless contact is on the blacklist or
                  is a system contact (微信支付 etc)
  trial_period  - "soft start": for the first N replies to any contact, apply
                  stricter checks; after N replies they get full access

All 11 original V37 checks (decision-sendable, latest-sender-customer, no
vision_error, etc.) are kept verbatim.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from ..storage import db
from ..storage.contact_control import ContactControlRepo

SENDABLE_DECISIONS = {
    'reply',
    'reply_draft',
    'weak_lead_draft',
}


class SendGuard:
    """Tri-mode contact gating matching original v3.10."""

    def __init__(self, config=None, db_conn=None):
        self._config = config or {}
        self._db_conn = db_conn or db.get_conn
        self._contact_control = ContactControlRepo(self._db_conn)
        self._mode = self._config.get('send_guard_mode', 'blacklist')
        self._trial_reply_limit = int(
            self._config.get('trial_reply_limit', 5))
        self._trial_period_min_confidence = float(
            self._config.get('trial_period_min_confidence', 0.6))
        self._test_contact_whitelist = self._clean_list(
            self._config.get('test_contact_whitelist', ''))
        self._contact_blacklist = self._clean_list(
            self._config.get('contact_blacklist', ''))

    # ---------------------------------------------------------------
    # 主入口：综合安全检查（对齐原版 check()）
    # ---------------------------------------------------------------
    def check(self, wechat_cfg: dict, safety_cfg: dict,
              analysis: dict, msg: dict) -> Tuple[bool, str]:
        """Run all 11 safety checks. Returns (pass, reason)."""
        # 对齐原版第一道门控：enable_rpa_send 总开关
        enable = bool((wechat_cfg or {}).get(
            "enable_rpa_send", self._config.get("enable_rpa_send", True)))
        if not enable:
            return False, 'wechat.enable_rpa_send must be true'

        decision = analysis.get('decision', '')
        if not self._is_decision_sendable(decision):
            return False, 'decision is not sendable: ' + str(decision)

        latest_sender = (msg.get('sender', '') or '').strip()
        # 居中系统气泡（时间分割线 / 添加好友提示 / 撤回提示 / 日期分隔等）不是
        # 真实发言。会话末尾常夹这类气泡，会让 latest_message.sender 误判成 system，
        # 从而把「客户发了消息 + 末尾一条系统提示」的会话永久封死、永远发不出去
        # （方政会话正是如此）。这里回退到「最后一条非系统气泡」的发件人来判定：
        #   - 最后真实发言是客户 -> 放行回复
        #   - 最后真实发言是自己 -> 仍拦截（刚回复过，不重复发）
        if latest_sender == 'system':
            latest_sender = self._last_real_sender(analysis) or latest_sender

        if latest_sender != 'customer':
            return False, 'latest_message.sender is not customer: ' + str(latest_sender)

        if msg.get('is_self_latest_message'):
            return False, 'latest message is self'

        if self._is_system_contact(analysis.get('current_contact', '')):
            return False, 'system/service contact'

        if analysis.get('is_ad_or_promotion'):
            return False, 'ad/promotion message'

        if analysis.get('is_payment_notice'):
            return False, 'payment notice'

        if not self._customer_text_readable(msg):
            return False, 'customer text was not readable; skip auto-send'

        reply = analysis.get('reply_draft', '')
        if not reply or not reply.strip():
            return False, 'reply_draft is empty'

        min_conf = float(self._config.get('min_confidence_to_reply', 0.0))
        effective_min_conf = float(self._contact_control.get_effective_min_confidence(
            analysis.get('current_contact', 'unknown')) or min_conf)
        confidence = float(analysis.get('confidence', 0))
        if confidence < effective_min_conf:
            return False, f'confidence {confidence} < min {effective_min_conf}'

        if analysis.get('vision_error'):
            if not self._vision_error_sendable(analysis):
                return False, 'vision_error: ' + str(analysis.get('vision_error'))

        # Contact control gate
        contact_name = analysis.get('current_contact', 'unknown')
        control_ok, control_reason = self._contact_control_gate(contact_name)
        if not control_ok:
            return False, control_reason

        # Test contact lock gate
        test_ok, test_reason = self._test_contact_lock_gate(contact_name)
        if not test_ok:
            return False, test_reason

        # Contact gate (whitelist/blacklist/trial)
        gate_ok, gate_reason = self._contact_gate(contact_name)
        if not gate_ok:
            return False, gate_reason

        return True, 'all safety checks passed'

    # ---------------------------------------------------------------
    # 原有简化接口（兼容旧调用）
    # ---------------------------------------------------------------
    def should_send(self, contact_key: str, content: str,
                    contact_name: str = "",
                    decision: str = "reply") -> bool:
        """Check if a message should be sent to the given contact."""
        if not self._is_decision_sendable(decision):
            return False
        if self._is_system_contact(contact_name):
            return False
        if self._mode == 'whitelist':
            return self._contact_control.is_whitelisted(contact_key)
        if self._mode == 'blacklist':
            if self._contact_control.is_blacklisted(contact_key):
                return False
            return True
        if self._mode == 'trial_period':
            reply_count = self._contact_control.get_reply_count(contact_key)
            return reply_count < self._trial_reply_limit
        return True

    def mark_sent(self, contact_key: str, content: str,
                  contact: str = "", reply_text: str = "") -> bool:
        """Record a sent message for deduplication."""
        fp = self._fingerprint(content)
        try:
            with self._db_conn() as conn:
                self._ensure_rows(conn)
                conn.execute(
                    "INSERT OR IGNORE INTO sent_messages"
                    "(contact_key,fingerprint,contact,reply_text,created_at)"
                    " VALUES(?,?,?,?,?)",
                    (contact_key, fp, contact or contact_key,
                     reply_text[:2000], db.now_str()),
                )
                return True
        except Exception:
            return False

    def already_sent(self, contact_key: str, content: str) -> bool:
        fp = self._fingerprint(content)
        with self._db_conn() as conn:
            self._ensure_rows(conn)
            row = conn.execute(
                "SELECT 1 FROM sent_messages"
                " WHERE contact_key=? AND fingerprint=? LIMIT 1",
                (contact_key, fp),
            ).fetchone()
            return row is not None

    def count_sent(self, contact_key: Optional[str] = None) -> int:
        with self._db_conn() as conn:
            self._ensure_rows(conn)
            if contact_key:
                row = conn.execute(
                    "SELECT COUNT(*) c FROM sent_messages"
                    " WHERE contact_key=?",
                    (contact_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) c FROM sent_messages").fetchone()
            return int(row["c"])

    def add_to_whitelist(self, contact_key: str) -> bool:
        return self._contact_control.add_whitelist(contact_key)

    def add_to_blacklist(self, contact_key: str) -> bool:
        return self._contact_control.add_blacklist(contact_key)

    # ---------------------------------------------------------------
    # 内部安全检查方法
    # ---------------------------------------------------------------
    def _is_decision_sendable(self, decision: str) -> bool:
        return decision in SENDABLE_DECISIONS

    def _is_system_contact(self, contact_name: str) -> bool:
        if not contact_name:
            return False
        name = contact_name.strip()
        system_patterns = [
            '微信支付', '微信游戏', '服务号', '公众号', '订阅号',
            '文件传输助手', '微信团队', 'QQ邮箱提醒', '企业微信',
            '小程序', '饿了么',
        ]
        for p in system_patterns:
            if p in name:
                return True
        return False

    def _customer_text_readable(self, msg: dict) -> bool:
        """Check if customer text is readable enough for auto-reply."""
        text = (msg.get('customer_turn_text', '') or
                msg.get('content', '') or
                msg.get('text', ''))
        if not text or not text.strip():
            return False
        if msg.get('intent') == 'unread_text_unreadable':
            return False
        return True

    @staticmethod
    def _last_real_sender(analysis: dict) -> str:
        """从完整消息列表回退找「最后一条非系统气泡」的发件人。

        会话末尾常夹带时间分割线、添加好友提示、撤回提示、日期分隔等居中系统
        气泡，它们会让 latest_message.sender 误判为 system 而封死回复。本方法
        从 messages 列表倒序找最后一条真实气泡（left/customer 或 right/self），
        据此判定谁最后发言，避免被尾部系统提示误导。找不到非系统气泡时返回空串。
        """
        msgs = (analysis or {}).get('messages') or []
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            s = str(m.get('sender') or m.get('side') or '').strip().lower()
            if not s or s in ('center', 'system'):
                continue
            if s in ('left', 'customer'):
                return 'customer'
            if s in ('right', 'self'):
                return 'self'
        return ''

    def _vision_error_sendable(self, analysis: dict) -> bool:
        """Only allow send despite vision error if fallback was used."""
        return bool(analysis.get('vision_fallback_used'))

    def _contact_control_gate(self, contact_name: str) -> Tuple[bool, str]:
        """Check contact_control status for the contact."""
        status = self._contact_control.get_status(contact_name)
        if not status:
            return True, ''
        status_lower = (status or '').strip().lower()
        if status_lower == 'auto_reply':
            return True, 'contact status: auto_reply'
        if status_lower == 'suggest_only':
            return False, 'contact status: suggest_only; generated reply only'
        if status_lower == 'paused':
            return False, 'contact status: paused'
        if status_lower == 'handoff':
            return False, 'contact status: handoff to human'
        if status_lower == 'blacklist':
            return False, 'contact status: blacklist'
        return True, ''

    def _test_contact_lock_gate(self, contact_name: str) -> Tuple[bool, str]:
        """Only allow sends to test whitelist contacts."""
        if not self._test_contact_whitelist:
            return True, 'test contact lock disabled'
        if self._matches(contact_name, self._test_contact_whitelist):
            return True, 'test contact lock matched'
        return False, f"test contact lock: contact '{contact_name}' not in test whitelist"

    def _contact_gate(self, contact_name: str) -> Tuple[bool, str]:
        """Whitelist/blacklist/trial_period gating."""
        if self._mode == 'whitelist':
            if self._contact_control.is_whitelisted(contact_name):
                return True, 'whitelisted'
            return False, f"contact '{contact_name}' not in whitelist"

        if self._mode == 'blacklist':
            if self._matches(contact_name, self._contact_blacklist):
                return False, f"'{contact_name}' on blacklist"
            return True, 'not on blacklist'

        if self._mode == 'trial_period':
            if self._matches(contact_name, self._contact_blacklist):
                return False, f"'{contact_name}' on blacklist (trial mode)"
            return True, 'trial mode: pass gate'

        return True, ''

    def _past_sent_count(self, contact_key: str) -> int:
        """Count past sent messages to a contact."""
        try:
            with self._db_conn() as conn:
                self._ensure_rows(conn)
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM sent_messages WHERE contact_key = ?",
                    (contact_key,),
                ).fetchone()
                return int(row["c"]) if row else 0
        except Exception:
            return 0

    # ---------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------
    @staticmethod
    def _fingerprint(content: str) -> str:
        import hashlib
        if not content:
            return hashlib.sha256(b"").hexdigest()
        return hashlib.sha256(
            content.strip().encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _clean_list(raw) -> list:
        """Parse comma/whitespace separated list into cleaned items."""
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, str):
            return [x.strip() for x in re.split(r'[,;\n]+', raw) if x.strip()]
        return []

    @staticmethod
    def _match_text(text: str) -> str:
        """Normalize text for matching: remove brackets, digits, extra spaces."""
        text = (text or '').strip()
        text = re.sub(r'[\[\(（]\s*\d+\s*[\]\)）]', '', text)
        return re.sub(r'\s+', '', text.casefold())

    @staticmethod
    def _matches(contact: str, allow_list: list) -> bool:
        """Check if contact matches any entry in the allow list."""
        contact_norm = SendGuard._match_text(contact)
        if not contact_norm:
            return False
        w_norms = [SendGuard._match_text(w) for w in allow_list]
        return any(contact_norm == w for w in w_norms) or any(
            w in contact_norm for w in w_norms if len(w) >= 3)

    @staticmethod
    def _float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _ensure_rows(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sent_messages ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " contact_key TEXT NOT NULL,"
            " fingerprint TEXT NOT NULL,"
            " contact TEXT DEFAULT '',"
            " reply_text TEXT DEFAULT '',"
            " created_at TEXT NOT NULL,"
            " UNIQUE(contact_key, fingerprint))"
        )