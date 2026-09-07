"""ConversationPolicy - conversation start/end and frequency policies.

Aligned with original app.agent.conversation_policy - "Extra gates for
group chats and voice-message transcripts." The original class gates
group-chat @-mentions and voice-message transcripts; the frequency
helpers (can_start_conversation / can_send_message / ...) are kept for
backwards compatibility with the earlier rewrite but are no longer used
by apply().
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional


class ConversationPolicy:
    """Group-chat mention gating, voice-message gating and frequency mgmt."""

    # ------------------------------------------------------------------
    # lifecycle / state
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        self._config = config or {}
        self.config = config or {}
        self._recent_group_mentions = {}
        # frequency-helper state (kept from the earlier rewrite)
        self._max_messages_per_session = self._config.get(
            "max_messages_per_session", 50)
        self._cooldown_seconds = self._config.get(
            "cooldown_seconds", 300)
        self._session_timeout = self._config.get(
            "session_timeout", 3600)
        self._sessions = {}

    # ------------------------------------------------------------------
    # apply (entry point) - aligned with original ConversationPolicy.apply
    #   original: apply(self, analysis, voice_result=None)
    #   observe_service calls apply(analysis) -> voice_result is optional.
    # ------------------------------------------------------------------
    def apply(self, analysis: dict, voice_result: Optional[dict] = None) -> dict:
        """应用会话策略（对齐原版 ConversationPolicy.apply）。

        原版签名 self, analysis, voice_result(=None)。先跑群聊回复门控，
        命中拦截即返回；否则再跑语音回复门控。返回 dict 可能含 blocked/reason。
        """
        out: dict = {}
        group = self._apply_group_reply_policy(analysis)
        if isinstance(group, dict):
            out.update(group)
        if out.get("blocked"):
            return out
        voice = self._apply_voice_reply_policy(analysis, voice_result)
        if isinstance(voice, dict):
            out.update(voice)
        return out

    # ------------------------------------------------------------------
    # group-chat reply gating
    # ------------------------------------------------------------------
    def _apply_group_reply_policy(self, analysis: dict) -> dict:
        cfg = self.config if isinstance(self.config, dict) else {}
        contact = str(analysis.get("current_contact", ""))
        model_chat_type = str(analysis.get("chat_type", "")).strip().lower()
        model_is_group = bool(analysis.get("is_group_chat", False))
        group_chat_types = frozenset(
            {"groupchat", "群聊", "group_chat", "group"})
        title_is_group = model_chat_type in group_chat_types
        is_group = title_is_group or model_is_group

        out: dict = {}
        if not is_group:
            return out

        group_name = self._group_name(contact)
        diagnostics = self._group_mention_diagnostics(
            cfg, analysis, contact=contact, group_name=group_name)

        # group reply disabled
        if not diagnostics.get("enabled"):
            analysis["group_reply_allowed"] = False
            out["blocked"] = True
            out["reason"] = "group_reply_disabled"
            out["policy_block_reason"] = "群聊回复未开启。"
            analysis["group_mention_diagnostics"] = diagnostics
            self._block(analysis, "group_reply_disabled", "群聊回复未开启。")
            return out

        # allowed-groups whitelist
        allowed_groups = diagnostics.get("allowed_groups") or []
        if allowed_groups:
            allowed = (self._matches(group_name, allowed_groups)
                       or self._matches(contact, allowed_groups))
            if not allowed:
                analysis["group_reply_allowed"] = False
                out["blocked"] = True
                out["reason"] = "group_not_allowed"
                msg = "群聊不在允许列表：" + (group_name or contact)
                out["policy_block_reason"] = msg
                analysis["group_mention_diagnostics"] = diagnostics
                self._block(analysis, "group_not_allowed", msg)
                return out

        mention_names = diagnostics.get("configured_mention_names") or []
        mention_match = self._latest_group_mention_match(analysis, mention_names)
        unconfigured_match = self._latest_unconfigured_group_mention_match(analysis)
        follow_up_context = self._recent_group_follow_up_context(
            cfg, analysis, contact=contact, group_name=group_name)
        allow_unconfigured = self._allow_unconfigured_group_mention(cfg, mention_names)
        require_mention = diagnostics.get("require_mention", False)

        mentioned_me = bool(mention_match.get("name")) or (
            bool(unconfigured_match) and allow_unconfigured)

        diagnostics["group_mentioned_me"] = mentioned_me
        diagnostics["matched"] = bool(mention_match.get("name"))
        diagnostics["match_source"] = (
            "configured_mention" if mention_match.get("name")
            else ("unconfigured_mention" if unconfigured_match else ""))
        diagnostics["matched_name"] = mention_match.get("name", "")
        diagnostics["matched_payload"] = mention_match.get("payload", "")

        if require_mention and not mentioned_me and not follow_up_context:
            analysis["group_reply_allowed"] = False
            out["blocked"] = True
            out["reason"] = "group_not_mentioned"
            out["policy_block_reason"] = "群聊里没有 @我，不回复。"
            analysis["group_mention_diagnostics"] = diagnostics
            self._block(analysis, "group_not_mentioned", "群聊里没有 @我，不回复。")
            return out

        # remember the mention so follow-up turns inherit context
        if mentioned_me:
            self._remember_group_mention(
                cfg, analysis, contact=contact, group_name=group_name,
                mention_name=mention_match.get("name", ""))

        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if isinstance(latest, dict):
            analysis["group_original_customer_turn_text"] = (
                latest.get("customer_turn_text") or latest.get("content") or "")

        analysis["group_reply_allowed"] = True
        analysis["group_mention_diagnostics"] = diagnostics
        out["group_reply_allowed"] = True
        out["group_mentioned_me"] = mentioned_me
        out["group_mention_diagnostics"] = diagnostics
        return out

    def _remember_group_mention(self, cfg, analysis, *, contact="",
                                group_name="", mention_name=""):
        cfg = cfg if isinstance(cfg, dict) else {}
        ttl = self._group_follow_up_seconds(cfg)
        sender_aliases = self._group_sender_aliases(analysis)
        sender_name = self._latest_group_sender_name(analysis)
        key = self._group_context_key(group_name or contact)
        now = time.monotonic()
        self._prune_group_mention_context(now)
        self._recent_group_mentions[key] = {
            "recorded_at": now,
            "sender_name": sender_name,
            "sender_aliases": sender_aliases,
            "mention_name": mention_name,
        }
        return ttl

    def _recent_group_follow_up_context(self, cfg, analysis, *, contact="",
                                        group_name=""):
        cfg = cfg if isinstance(cfg, dict) else {}
        ttl = self._group_follow_up_seconds(cfg)
        sender_aliases = self._group_sender_aliases(analysis)
        sender_name = self._latest_group_sender_name(analysis)
        key = self._group_context_key(group_name or contact)
        now = time.monotonic()
        self._prune_group_mention_context(now)
        previous = self._recent_group_mentions.get(key)
        if not isinstance(previous, dict):
            return {}
        age = max(0.0, now - float(previous.get("recorded_at", now)))
        if age > ttl:
            return {}
        previous_aliases = previous.get("sender_aliases", []) or []
        matched_alias = ""
        current_norm = {self._match_text(a) for a in sender_aliases}
        for alias in previous_aliases:
            if self._match_text(alias) in current_norm:
                matched_alias = alias
                break
        return {
            "age_seconds": age,
            "matched_sender_alias": matched_alias,
            "current_sender_aliases": sender_aliases,
        }

    def _prune_group_mention_context(self, now):
        expired = [
            key for key, value in self._recent_group_mentions.items()
            if float((value or {}).get("recorded_at", 0.0)) < now - 300.0
        ]
        for key in expired:
            self._recent_group_mentions.pop(key, None)

    @staticmethod
    def _group_follow_up_seconds(cfg) -> float:
        try:
            value = float(cfg.get("follow_up_context_seconds", 90))
        except (TypeError, ValueError, Exception):
            value = 90.0
        return max(0.0, min(value, 300.0))

    @classmethod
    def _group_context_key(cls, value) -> str:
        return cls._match_text(value)

    @staticmethod
    def _latest_is_customer(analysis) -> bool:
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if not isinstance(latest, dict):
            return False
        sender = latest.get("sender")
        if isinstance(sender, dict):
            side = str(sender.get("side", "")).strip().lower()
            return side == "customer"
        text = str(sender).strip().lower() if sender is not None else ""
        return text == "customer"

    @classmethod
    def _group_sender_aliases(cls, analysis):
        candidates = []
        v = analysis.get("vision_group_sender_name") if isinstance(analysis, dict) else None
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if isinstance(latest, dict):
            g = latest.get("group_sender_name")
            if isinstance(g, str) and g.strip():
                candidates.append(g.strip())
        turns = analysis.get("customer_turn_messages") if isinstance(analysis, dict) else None
        if isinstance(turns, list):
            for turn in reversed(turns):
                if isinstance(turn, dict):
                    name = turn.get("sender_name") or turn.get("group_sender_name")
                    if isinstance(name, str) and name.strip():
                        candidates.append(name.strip())
        return cls._clean_sender_aliases(candidates)

    @classmethod
    def _clean_sender_aliases(cls, values):
        result = []
        seen = set()
        for item in values:
            if isinstance(item, (list, tuple)):
                for v in item:
                    if isinstance(v, str):
                        value = v.strip()
                        if value:
                            norm = cls._match_text(value)
                            if norm not in seen:
                                seen.add(norm)
                                result.append(value)
            elif isinstance(item, str):
                value = item.strip()
                if value:
                    norm = cls._match_text(value)
                    if norm not in seen:
                        seen.add(norm)
                        result.append(value)
        if len(result) > 2:
            result = result[-2:]
        return result

    @classmethod
    def _latest_group_sender_name(cls, analysis) -> str:
        aliases = cls._group_sender_aliases(analysis)
        return aliases[0] if aliases else ""

    # ------------------------------------------------------------------
    # voice-message reply gating
    # ------------------------------------------------------------------
    def _apply_voice_reply_policy(self, analysis: dict, voice_result=None) -> dict:
        cfg = self.config if isinstance(self.config, dict) else {}
        voice_cfg = cfg.get("voice_messages", {}) if isinstance(cfg, dict) else {}
        out: dict = {}
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        is_voice = False
        if isinstance(latest, dict):
            t = str(latest.get("type", "")).strip().lower()
            is_voice = t in ("voice", "语音")

        converted = bool(voice_result.get("converted", False)) if isinstance(voice_result, dict) else False

        if not converted:
            # maybe the latest bubble only shows a voice-duration placeholder
            if self._latest_text_looks_like_voice_duration(analysis):
                out["voice_detected"] = True
                out["blocked"] = True
                out["reason"] = "voice_convert_failed"
                out["policy_block_reason"] = "识别到语音消息，但没有成功转成文字，本轮不自动回复。"
                self._block(analysis, "voice_convert_failed",
                            "识别到语音消息，但没有成功转成文字，本轮不自动回复。")
                return out
            return out

        transcript = ""
        if isinstance(voice_result, dict):
            transcript = str(voice_result.get("converted_text", "")).strip()
            if voice_result.get("ocr_text"):
                out["voice_ocr_text_used"] = True

        if transcript:
            analysis["voice_transcribed_text"] = transcript
            out["voice_transcribed"] = True
            if not analysis.get("customer_turn_text"):
                analysis["customer_turn_text"] = transcript
        else:
            out["voice_transcription_text_not_found"] = True
            out["blocked"] = True
            out["reason"] = "voice_convert_failed"
            out["policy_block_reason"] = "语音没有成功转成文字，本轮不自动回复。"
            self._block(analysis, "voice_convert_failed",
                        "语音没有成功转成文字，本轮不自动回复。")
            return out

        if not voice_cfg.get("auto_reply_after_transcription", False):
            out["blocked"] = True
            out["reason"] = "voice_auto_reply_disabled"
            out["policy_block_reason"] = "语音已转成文字，但语音自动回复默认关闭。"
            self._block(analysis, "voice_auto_reply_disabled",
                        "语音已转成文字，但语音自动回复默认关闭。")
            return out

        return out

    @staticmethod
    def _latest_text_looks_like_voice_duration(analysis) -> bool:
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if not isinstance(latest, dict):
            return False
        text = ""
        content = latest.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = latest.get("customer_turn_text", "")
        if not isinstance(text, str):
            text = ""
        text = text.replace(" ", "").replace("\n", "")
        if "转文字" in text:
            return True
        norm = (text.replace("’", "'").replace("′", "'").replace("＇", "'")
                    .replace("″", "\"").replace("”", "").replace("“", ""))
        return bool(re.fullmatch(r"\d{1,2}('{1,2}|秒|s|S)", norm))

    @staticmethod
    def _has_substantive_customer_text(analysis) -> bool:
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if not isinstance(latest, dict):
            return False
        text = latest.get("customer_turn_text") or latest.get("content") or ""
        if not isinstance(text, str):
            text = ""
        if not text.strip():
            return False
        compact = re.sub(
            r'[\s，,。！？!?；;：:、~～\-_（）()【】\[\]""'
            r"''/／|]+", "", text)
        if not compact:
            return False
        placeholders = frozenset({"语音转文字", "按下说话", "按住说话", "转文字"})
        if compact in placeholders:
            return False
        if re.fullmatch(r"\d{1,3}", compact):
            return False
        return True

    @staticmethod
    def _group_name(contact) -> str:
        if not isinstance(contact, str):
            return ""
        text = contact.strip()
        m = re.search(r"^(?P<name>.+?)[(（]\s*\d+\s*[)）]$", text)
        if m:
            return m.group("name")
        return text

    @staticmethod
    def _conversation_text(analysis) -> str:
        if not isinstance(analysis, dict):
            return ""
        parts = []
        latest = analysis.get("latest_message")
        if isinstance(latest, dict):
            c = latest.get("content") or latest.get("customer_turn_text") or ""
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
        conv = analysis.get("visible_conversation_text")
        if isinstance(conv, str) and conv.strip():
            parts.append(conv.strip())
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _mentioned_me(text, names) -> bool:
        if not isinstance(text, str):
            return False
        normalized = text.replace("＠", "@").replace(" ", "")
        if not names:
            return "@" in normalized
        for name in names:
            n = str(name).replace("＠", "@").replace(" ", "")
            if ("@" + n) in normalized:
                return True
        return False

    @classmethod
    def _latest_group_mention_text(cls, analysis, names) -> str:
        match = cls._latest_group_mention_match(analysis, names)
        if isinstance(match, dict):
            return match.get("payload", "")
        return ""

    @classmethod
    def _latest_group_mention_match(cls, analysis, names):
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if not isinstance(latest, dict):
            return {}
        latest_text = cls._latest_bubble_text(latest)
        matched_name = cls._matched_mention_name(latest_text, names)
        if matched_name:
            payload = cls._extract_last_mention_payload(latest_text, names)
            return {"payload": payload, "name": matched_name,
                    "latest_bubble_text": latest_text}
        return {"name": "", "latest_bubble_text": latest_text}

    @staticmethod
    def _matched_mention_name(text, names) -> str:
        if not isinstance(text, str):
            return ""
        normalized = text.replace("＠", "@").replace(" ", "")
        for name in names:
            n = str(name).replace("＠", "@").replace(" ", "")
            if ("@" + n) in normalized:
                return name
        return ""

    @classmethod
    def _latest_bubble_text(cls, latest) -> str:
        if not isinstance(latest, dict):
            return ""
        bubble = latest.get("bubble_text")
        if isinstance(bubble, str) and bubble.strip():
            return bubble.strip()
        content = latest.get("content")
        if not isinstance(content, str) or not content.strip():
            return ""
        sender_name = latest.get("group_sender_name")
        if isinstance(sender_name, str) and sender_name.strip():
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            sender_norm = cls._match_text(sender_name)
            kept = []
            for line in lines:
                content_norm = cls._match_text(line)
                if len(content_norm) > len(sender_norm) and content_norm.startswith(sender_norm):
                    stripped = line[len(sender_name):].lstrip(" \t\r\n:：,，、;；")
                    if stripped:
                        kept.append(stripped)
                else:
                    kept.append(line)
            return "\n".join(kept)
        return content.strip()

    @classmethod
    def _latest_unconfigured_group_mention_match(cls, analysis):
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        if not isinstance(latest, dict):
            return {}
        latest_text = cls._latest_bubble_text(latest)
        normalized = latest_text.replace("＠", "@")
        at_index = normalized.find("@")
        if at_index < 0 or at_index > 24:
            return {}
        prefix = normalized[:at_index]
        if prefix.strip(" \t\r\n:：，,。?？"):
            return {}
        payload = cls._strip_unconfigured_mention_token(latest_text)
        if cls._has_substantive_customer_text(analysis) or payload.strip():
            return {"payload": payload, "latest_bubble_text": latest_text}
        return {}

    @classmethod
    def _latest_unconfigured_group_mention_text(cls, analysis) -> str:
        match = cls._latest_unconfigured_group_mention_match(analysis)
        if isinstance(match, dict):
            return match.get("payload", "")
        return ""

    @staticmethod
    def _strip_unconfigured_mention_token(payload) -> str:
        if not isinstance(payload, str):
            return ""
        parts = payload.split("@", 1)
        if len(parts) < 2:
            return payload.strip()
        rest = parts[1]
        name_part = rest.split(None, 1)
        if len(name_part) > 1:
            return name_part[1].strip()
        return ""

    @classmethod
    def _group_mention_diagnostics(cls, cfg, analysis, *, contact="", group_name=""):
        if not isinstance(cfg, dict):
            cfg = {}
        latest = analysis.get("latest_message") if isinstance(analysis, dict) else None
        latest_bubble_text = cls._latest_bubble_text(latest) if isinstance(latest, dict) else ""
        mention_names = cls._clean_list(cfg.get("mention_names", []))
        return {
            "current_contact": contact,
            "group_name": group_name,
            "enabled": bool(cfg.get("group_reply", False)),
            "allowed_groups": cls._clean_list(cfg.get("allowed_groups", [])),
            "configured_mention_names": mention_names,
            "require_mention": bool(cfg.get("require_mention", False)),
            "auto_reply": bool(cfg.get("auto_reply", True)),
            "latest_bubble_text": latest_bubble_text,
            "allow_unconfigured_mention": bool(cfg.get("allow_unconfigured_mention", False)),
            "matched": False,
            "match_source": "",
            "matched_name": "",
            "matched_payload": "",
            "group_mentioned_me": False,
        }

    @classmethod
    def _allow_unconfigured_group_mention(cls, cfg, names) -> bool:
        if not isinstance(cfg, dict):
            cfg = {}
        if bool(cfg.get("allow_unconfigured_mention", False)):
            return True
        generic = frozenset({"客服助手", "小助手", "助手", "微信客服助手", "客服"})
        if names:
            return bool(names) and all(str(n).strip() in generic for n in names)
        return False

    @staticmethod
    def _extract_last_mention_payload(text, names) -> str:
        if not isinstance(text, str):
            return ""
        normalized = text.replace("＠", "@")
        lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
        for line in reversed(lines):
            for name in names:
                n = str(name).replace("＠", "@").replace(" ", "")
                if ("@" + n) in line:
                    m = re.search(r"@\s*" + re.escape(n) + r"\s*(.*)$", line)
                    if m:
                        return m.group(1).strip(" \t\r\n:：,，、;；")
        return ""

    @staticmethod
    def _matches(value, candidates) -> bool:
        if not isinstance(value, str) or not isinstance(candidates, (list, tuple, set)):
            return False
        value_norm = ConversationPolicy._match_text(value)
        value_ocr = ConversationPolicy._ocr_safe_match_text(value)
        for item in candidates:
            if not isinstance(item, str):
                continue
            item_norm = ConversationPolicy._match_text(item)
            item_ocr = ConversationPolicy._ocr_safe_match_text(item)
            if value_norm == item_norm or value_ocr == item_ocr:
                return True
        return False

    @staticmethod
    def _match_text(text) -> str:
        if not isinstance(text, str):
            return ""
        value = text.strip().casefold()
        value = re.sub(r"[(（]\s*\d+\s*[)）]$", "", value)
        value = re.sub(r"\s+", "", value)
        return value

    @staticmethod
    def _ocr_safe_match_text(text) -> str:
        value = ConversationPolicy._match_text(text)
        trans = str.maketrans({"i": "1", "l": "1", "|": "1", "o": "0", "O": "0"})
        return value.translate(trans)

    @staticmethod
    def _clean_list(raw):
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = raw.replace("，", ",").replace("、", ",")
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, (list, tuple, set)):
            result = []
            for item in raw:
                if isinstance(item, str):
                    result.extend(ConversationPolicy._clean_list(item))
                else:
                    result.append(item)
            return result
        return [raw]

    @staticmethod
    def _block(analysis, intent, reason):
        if not isinstance(analysis, dict):
            return None
        existing_reply = analysis.get("reply_draft", "")
        if isinstance(existing_reply, str) and existing_reply.strip():
            analysis["policy_blocked_reply_draft"] = existing_reply
        analysis["reply_draft"] = ""
        analysis["policy_block_intent"] = intent
        analysis["policy_block_reason"] = reason
        analysis["intent"] = intent
        analysis["no_reply"] = True
        analysis["decision"] = "no_reply"
        analysis["action"] = "block"
        analysis["should_reply"] = False
        analysis["reason"] = reason
        analysis["skip_text_model"] = True
        return None

    # ------------------------------------------------------------------
    # frequency helpers (kept from earlier rewrite; not used by apply)
    # ------------------------------------------------------------------
    def can_start_conversation(self, contact_key: str) -> bool:
        """Check if a new conversation can be started."""
        if contact_key in self._sessions:
            session = self._sessions[contact_key]
            if time.time() - session.get("last_activity", 0) < self._cooldown_seconds:
                return False
        return True

    def start_conversation(self, contact_key: str) -> bool:
        """Start a conversation for a contact."""
        if not self.can_start_conversation(contact_key):
            return False
        self._sessions[contact_key] = {
            "started_at": time.time(),
            "last_activity": time.time(),
            "message_count": 0,
        }
        return True

    def record_message(self, contact_key: str) -> None:
        """Record a message in the conversation."""
        if contact_key not in self._sessions:
            self.start_conversation(contact_key)
        session = self._sessions.get(contact_key, {})
        session["message_count"] = session.get("message_count", 0) + 1
        session["last_activity"] = time.time()

    def can_send_message(self, contact_key: str) -> bool:
        """Check if another message can be sent."""
        session = self._sessions.get(contact_key)
        if not session:
            return True
        if session.get("message_count", 0) >= self._max_messages_per_session:
            return False
        if time.time() - session.get("last_activity", 0) < 1.0:
            return False
        return True

    def end_conversation(self, contact_key: str) -> None:
        """End a conversation."""
        if contact_key in self._sessions:
            self._sessions[contact_key]["ended_at"] = time.time()

    def is_conversation_active(self, contact_key: str) -> bool:
        """Check if a conversation is active."""
        session = self._sessions.get(contact_key)
        if not session:
            return False
        if session.get("ended_at"):
            return False
        if time.time() - session.get("last_activity", 0) > self._session_timeout:
            return False
        return True

    def cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            k for k, v in self._sessions.items()
            if now - v.get("last_activity", 0) > self._session_timeout
        ]
        for k in expired:
            del self._sessions[k]
