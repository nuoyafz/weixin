"""FallbackReply - default-allow reply policy for real customers.

Aligned with original app.brain.fallback_reply v3.10:
  - can_reply: decide whether a non-blacklisted customer may enter the model fallback path
  - relax: let normal reply generation run for a non-blacklisted customer
  - apply_final: mark for model fallback when all evidence-based paths failed
  - sanitize_outgoing: block internal explanations before they reach WeChat
  - _looks_like_internal_no_reply: detect internal no-reply explanations
"""
from __future__ import annotations

import re
from typing import Any


class FallbackReply:
    """Default-allow reply policy for real customers.

    This class does not generate customer-visible sentences. It only decides
    whether a non-blacklisted customer may enter the model fallback path.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def enabled(self) -> bool:
        cfg = self.config.get("reply_fallback") or {}
        if "no_knowledge_chitchat_enabled" in cfg:
            return bool(cfg.get("no_knowledge_chitchat_enabled"))
        if "enabled" in cfg:
            return bool(cfg.get("enabled"))
        return True

    def can_reply(self, analysis: dict[str, Any]) -> bool:
        if not self.enabled():
            return False
        analysis = analysis or {}
        latest = (
            analysis.get("latest_message")
            if isinstance(analysis.get("latest_message"), dict)
            else {}
        )
        sender = str(latest.get("sender") or "").strip().lower()
        side = str(latest.get("side") or "").strip().lower()
        contact = str(analysis.get("current_contact") or "").strip()
        if contact and contact == "unknown":
            return False
        if bool(analysis.get("is_self_latest_message")) or sender == "self" or side == "right":
            return False
        if sender and sender not in frozenset({"unknown", "customer"}) and side != "left":
            return False
        from .skip_contacts import is_skipped_contact
        if is_skipped_contact(contact, self.config):
            return False
        if self._matches_contact_blacklist(contact):
            return False
        status = str(analysis.get("contact_control_status") or "").strip().lower()
        if status == "blacklist":
            return False
        return True

    def relax(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Let normal reply generation run for a non-blacklisted customer."""
        out = dict(analysis or {})
        if not self.can_reply(out):
            return out
        decision = str(out.get("decision") or out.get("action") or "")
        if decision not in frozenset({"handoff", "no_reply"}):
            return out
        if out.get("fallback_original_intent") is None:
            out["fallback_original_intent"] = out.get("intent", "")
        if out.get("fallback_original_reason") is None:
            out["fallback_original_reason"] = out.get("reason", "")
        out["fallback_policy_relaxed"] = True
        out["decision"] = "reply"
        out["action"] = "reply"
        out["should_reply"] = True
        out["needs_reply"] = True
        if (
            not str(out.get("reply_draft") or "").strip()
            and not str(out.get("model_reply_draft") or "").strip()
            and not str(out.get("policy_blocked_reply_draft") or "").strip()
        ):
            out["reply_draft"] = ""
        out["skip_text_model"] = False
        if str(out.get("intent") or "") in self._blocked_intents():
            out["intent"] = "customer_message"
        out["reason"] = "非黑名单客户消息，继续尝试自动回复。"
        out["is_ad_or_promotion"] = False
        out["is_payment_notice"] = False
        return out

    def apply_final(self, analysis: dict[str, Any], reply: str | None) -> tuple[dict[str, Any], str]:
        """Mark the message for model fallback when all evidence-based paths failed."""
        out = dict(analysis or {})
        current_reply = str(reply or out.get("reply_draft") or "").strip()
        if current_reply:
            if str(out.get("decision") or out.get("action") or "") != "no_reply":
                return (out, current_reply)
        if not self.can_reply(out):
            return (out, current_reply)
        out["fallback_reply_pending"] = True
        if out.get("fallback_original_intent") is None:
            out["fallback_original_intent"] = out.get("intent", "")
        out["intent"] = "fallback_reply"
        out["decision"] = "reply"
        out["action"] = "reply"
        out["should_reply"] = True
        out["reply_draft"] = ""
        out["confidence"] = max(float(out.get("confidence") or 0), 0.82)
        out["reason"] = "无资料/闲聊兜底已开启，交给模型生成自然回复。"
        out["is_ad_or_promotion"] = False
        out["is_payment_notice"] = False
        return (out, "")

    def sanitize_outgoing(self, analysis: dict[str, Any], reply: str | None) -> tuple[dict[str, Any], str]:
        """Block internal explanations and policy templates before they reach WeChat."""
        out = dict(analysis or {})
        current_reply = str(reply or out.get("reply_draft") or "").strip()
        if not current_reply:
            return (out, current_reply)
        if not self.can_reply(out):
            return (out, current_reply)
        if not self._looks_like_internal_no_reply(current_reply):
            return (out, current_reply)
        out["fallback_replaced_internal_reply"] = True
        out["unsafe_reply_draft"] = current_reply
        out["fallback_reply_pending"] = True
        if out.get("fallback_original_intent") is None:
            out["fallback_original_intent"] = out.get("intent", "")
        out["intent"] = "fallback_reply"
        out["decision"] = "reply"
        out["action"] = "reply"
        out["should_reply"] = True
        out["reply_draft"] = ""
        out["reason"] = "生成内容包含内部不回复原因，已交给模型重新生成客户可见回复。"
        out["is_ad_or_promotion"] = False
        out["is_payment_notice"] = False
        return (out, "")

    @staticmethod
    def _blocked_intents() -> set[str]:
        return frozenset(
            {
                "payment_notice",
                "ad_or_promotion",
                "group_not_allowed",
                "group_not_mentioned",
                "missing_reply_basis",
                "group_reply_disabled",
                "human_service_request",
                "image_sticker_no_reply",
                "group_allowed_list_empty",
                "group_auto_reply_disabled",
            }
        )

    def _matches_contact_blacklist(self, contact: str) -> bool:
        wechat = self.config if isinstance(self.config, dict) else {}
        wechat = wechat.get("wechat") or {}
        items = wechat.get("contact_blacklist") or []
        for raw in items:
            item = str(raw or "").strip()
            if item and (item == contact or item in contact or contact in item):
                return True
        return False

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(
            r'[\s，,。！？!?；;：:、~～\-_（）()【】\[\]\"\'<>《》]+',
            "",
            str(value or ""),
        ).lower()

    @classmethod
    def _looks_like_internal_no_reply(cls, reply: str) -> bool:
        text = str(reply or "").strip()
        if not text:
            return False
        raw_hits = [
            "不回复",
            "不建议回复",
            "本轮不回复",
            "对方是私人联系人",
            "私人联系人",
            "非客服场景",
            "不属于客服场景",
            "无需回复",
            "服务范围、上门范围、发货范围或交付范围以这里为准",
            "未写明的范围需要人工确认",
            "价格以「产品/服务/价格」中已写明的内容为准",
            "客户问到未写明的套餐",
            "按客户当前所问流程提供对应信息",
            "客户问到具体时间时需要人工确认",
            "不能直接承诺库存、档期",
            "不得直接承诺库存、档期",
            "资料不明确时先说明需要确认并转人工",
            "缺少关键信息时先登记",
        ]
        if any(hit in text for hit in raw_hits):
            return True
        compact = cls._compact(text)
        return any(cls._compact(hit) in compact for hit in raw_hits)