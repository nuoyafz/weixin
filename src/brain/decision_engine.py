"""决策引擎：对视觉/AI 给出的分析做标准化与过滤，输出最终 decision。

normalize() 是核心入口：接收一份「分析草稿」（来自视觉识别 + AI 文本模型），
补齐字段、做黑名单/系统联系人/支付通知/广告/被动图片等过滤，最终给出
统一的 decision / action / should_reply 等字段。

对齐原版 app.brain.decision_engine v3.10。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from .skip_contacts import is_skipped_contact, is_blacklisted_contact


@dataclass
class Decision:
    """决策结果数据类。"""
    should_reply: bool = False
    route: str = "skip"
    skip_reason: str = ""
    reply_text: str = ""
    intent: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "should_reply": self.should_reply,
            "route": self.route,
            "skip_reason": self.skip_reason,
            "reply_text": self.reply_text,
            "intent": self.intent,
            "confidence": self.confidence,
        }

AD_KEYWORDS = (
    "福利", "活动", "促销", "优惠", "领取", "充值", "满减", "红包", "点击", "立即",
    "推广", "广告", "免费", "抽奖", "下载", "小程序", "外卖", "游戏",
)

PAYMENT_KEYWORDS = (
    "赞赏到账通知", "收款金额", "已收款", "微信转账", "转账", "到账时间",
    "收款", "付款", "支付", "¥", "￥",
)

PASSIVE_IMAGE_KEYWORDS = (
    "表情包", "表情", "动图", "动画表情", "贴纸", "gif", "sticker", "emoji", "meme",
    "谢谢老板", "哈哈", "笑脸", "小孩", "小女孩", "小男孩", "人像", "无关图片",
)

IMAGE_QUESTION_KEYWORDS = (
    "帮我看", "帮看看", "帮我看看", "看看这个", "这是什么", "这个是什么",
    "识别", "怎么", "怎么弄", "怎么处理", "为什么", "哪里", "报错", "错误",
    "截图", "界面", "配置", "设置", "功能", "知识库", "能用吗", "可以吗",
    "多少钱", "价格", "付款", "支付", "订单",
)


class DecisionEngine:
    def __init__(self, config: "Dict[str, Any] | None" = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # normalize — 核心入口（对齐原版）
    # ------------------------------------------------------------------
    def normalize(self, analysis: "Dict[str, Any] | None") -> "Dict[str, Any]":
        """标准化分析结果，进行六级瀑布过滤，输出最终 decision。

        对齐原版 DecisionEngine.normalize()。
        """
        analysis = analysis or {}
        msg = analysis.get("latest_message") or {}
        msg.setdefault("sender", "unknown")
        msg.setdefault("side", "unknown")
        msg.setdefault("type", "unknown")
        msg.setdefault("content", "")
        analysis["latest_message"] = msg

        decision = analysis.get("decision") or analysis.get("action") or "no_reply"
        if decision == "reply_draft":
            decision = "reply"

        contact = str(analysis.get("current_contact") or "unknown")
        text = f"{contact} {msg.get('content', '')} {analysis.get('reason', '')} {analysis.get('intent', '')}"

        is_self = (
            bool(analysis.get("is_self_latest_message"))
            or msg.get("sender") == "self"
            or msg.get("side") == "right"
        )
        contact_status = str(analysis.get("contact_control_status") or "").strip().lower()
        is_blacklisted = (contact_status == "blacklist") or self._matches_contact_blacklist(contact)
        is_system = bool(analysis.get("is_system_contact")) or is_skipped_contact(contact, self.config)

        is_payment = self._looks_like_payment_notice(contact, msg, analysis)
        is_ad = self._looks_like_ad(text, msg)
        is_passive_image = self._looks_like_passive_image(msg, analysis)

        analysis["is_self_latest_message"] = is_self
        analysis["is_system_contact"] = is_system
        analysis["is_contact_blacklisted"] = is_blacklisted
        analysis["is_payment_notice"] = is_payment
        analysis["is_ad_or_promotion"] = is_ad

        # 红点未读优先：真实的未读徽标（red_dot_badge）说明确有未读客户消息，
        # 不应因 OCR 把末条误判为「自己发送」而漏回复（如「方舟」类真实客户）。
        # 仅当联系人非黑名单 / 非系统号 / 非支付通知时才放行，避免误回系统推送。
        unread_cnt = analysis.get("unread_count") or 0
        unread_src = str(analysis.get("unread_count_source") or "").lower()
        red_dot_unread = bool(
            unread_cnt and unread_cnt > 0
            and unread_src in ("red_dot_badge", "badge", "red_dot")
        )
        if red_dot_unread and not (is_blacklisted or is_system or is_payment):
            is_self = False
            analysis["is_self_latest_message"] = False

        if is_self:
            decision = "no_reply"
            analysis["intent"] = "self_latest_message"
            if not analysis.get("reason"):
                analysis["reason"] = "最后一条消息为自己发送，不回复。"
            analysis["reply_draft"] = ""
        elif is_blacklisted:
            decision = "no_reply"
            analysis["intent"] = "contact_blacklist"
            if not analysis.get("reason"):
                analysis["reason"] = "当前联系人在黑名单中，不回复。"
            analysis["reply_draft"] = ""
        elif is_system:
            decision = "no_reply"
            analysis["intent"] = "system_notice"
            if not analysis.get("reason"):
                analysis["reason"] = "系统联系人/服务号/公众号/文件传输助手，不回复。"
            analysis["reply_draft"] = ""
        elif is_payment:
            decision = "no_reply"
            analysis["intent"] = "payment_notice"
            if not analysis.get("reason"):
                analysis["reason"] = "支付/收款/转账通知，不回复。"
            analysis["reply_draft"] = ""
        elif is_ad:
            decision = "no_reply"
            analysis["intent"] = "ad_or_promotion"
            if not analysis.get("reason"):
                analysis["reason"] = "广告/营销/活动/福利类信息，不回复。"
            analysis["reply_draft"] = ""
        elif is_passive_image:
            decision = "no_reply"
            analysis["intent"] = "image_sticker_no_reply"
            if not analysis.get("reason"):
                analysis["reason"] = "客户只发了表情包或无明确业务问题的图片，本轮不回复。"
            analysis["reply_draft"] = ""
            analysis["reply_suggestion"] = "不建议回复：客户只发了表情包或无明确业务问题的图片。"

        # 红点未读优先（续）：确为未读客户消息时，强制进入回复流程，
        # 覆盖上方可能因 OCR 误判留下的 no_reply 结论。
        if red_dot_unread and not (is_blacklisted or is_system or is_payment):
            decision = "reply"
            analysis["decision"] = "reply"
            analysis["action"] = "reply"
            analysis["intent"] = analysis.get("intent") or "chat_message"
            analysis["no_reply"] = False
            analysis["should_reply"] = True
            analysis["needs_reply"] = True

        if decision not in frozenset({"reply", "handoff", "no_reply", "weak_lead_draft", "open_unread_chat"}):
            decision = "no_reply"

        try:
            conf = float(analysis.get("confidence") or 0)
        except Exception:
            conf = 0
        analysis["confidence"] = max(0, min(1, conf))

        for field in ("recognition_confidence", "reply_confidence"):
            if field not in analysis:
                continue
            try:
                value = float(analysis.get(field) or 0)
            except Exception:
                value = 0
            analysis[field] = max(0, min(1, value))

        analysis["decision"] = decision
        analysis["action"] = decision
        analysis["should_reply"] = decision in frozenset({"reply", "weak_lead_draft"})
        analysis["needs_reply"] = analysis["should_reply"]

        analysis.setdefault("platform", "wechat")
        analysis.setdefault("screen_state", "unknown")
        analysis.setdefault("input_box_hint", self._default_input_box_hint())
        analysis.setdefault("unread_message_hint", {
            "has_unread": False, "x_ratio": 0, "y_ratio": 0, "contact_name": "", "reason": "",
        })
        return analysis

    def _default_input_box_hint(self) -> "Dict[str, float]":
        # 兼容 dict 与 Settings 对象两种 config 形态
        c = self.config
        if isinstance(c, dict):
            wechat = c.get("wechat") or {}
        else:
            wechat = getattr(c, "wechat", None) or {}
        if isinstance(wechat, dict):
            x = float(wechat.get("input_box_click_ratio_x", 0.68) or 0.68)
            y = float(wechat.get("input_box_click_ratio_y", 0.9) or 0.9)
        else:
            x = float(getattr(wechat, "input_box_click_ratio_x", 0.68) or 0.68)
            y = float(getattr(wechat, "input_box_click_ratio_y", 0.9) or 0.9)
        return {"x_ratio": x, "y_ratio": y}

    def _contains_any(self, text: str, keywords) -> bool:
        return any(k in text for k in keywords)

    def _matches_contact_blacklist(self, contact: str) -> bool:
        return is_blacklisted_contact(contact, self.config)

    def _looks_like_passive_image(self, msg: "Dict[str, Any]", analysis: "Dict[str, Any]") -> bool:
        msg_type = str(msg.get("type") or "").strip().lower()
        if msg_type not in frozenset({"image", "text_or_image"}):
            return False
        if str(msg.get("sender") or "") != "customer":
            return False
        latest_text = str(msg.get("content") or analysis.get("recognized_text") or "").strip().lower()
        combined = " ".join(str(x) for x in (
            msg.get("content"), analysis.get("recognized_text"),
            analysis.get("customer_turn_text"), analysis.get("reason"),
        )).strip().lower()
        if not latest_text and not combined:
            return True
        if self._contains_any(latest_text, IMAGE_QUESTION_KEYWORDS):
            return False
        if self._contains_any(latest_text or combined, PASSIVE_IMAGE_KEYWORDS):
            return True
        if msg_type == "image" and len(latest_text or combined) <= 80:
            return True
        return False

    def _looks_like_payment_notice(self, contact: str, msg: "Dict[str, Any]", analysis: "Dict[str, Any]") -> bool:
        sender = str(msg.get("sender") or "").strip().lower()
        side = str(msg.get("side") or "").strip().lower()
        content = str(
            msg.get("content") or analysis.get("recognized_text") or analysis.get("reason") or ""
        ).strip().lower()
        if is_skipped_contact(contact, self.config):
            if any(k in contact for k in ("支付", "收款", "银行")):
                return True
        if sender == "customer" or side == "left":
            notice_terms = ["赞赏到账通知", "收款金额", "到账时间", "二维码到账", "已收款", "微信转账"]
            has_notice = any(term in content for term in notice_terms)
            has_amount = bool(re.search(r"[¥￥]\s*\d|\d+(?:\.\d+)?\s*元", content))
            return bool(has_notice) and bool(has_amount)
        return bool(analysis.get("is_payment_notice"))

    def _looks_like_ad(self, text: str, msg: "Dict[str, Any] | None") -> bool:
        msg = msg or {}
        sender = str(msg.get("sender") or "").strip().lower()
        side = str(msg.get("side") or "").strip().lower()
        if sender != "customer" and side != "left":
            # 只有客户侧(customer/left)发来的消息才可能判为广告；
            # 自己(self/right)或非客户侧直接排除，不误杀客户营销消息。
            return False
        hits = sum(1 for k in AD_KEYWORDS if k in text)
        if hits >= 2:
            return True
        if hits >= 1:
            return any(k in text for k in ("http", "www.", "扫码", "点击", "领取", "立即", "小程序"))
        return False