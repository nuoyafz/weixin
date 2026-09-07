"""报价策略 + 引用回复决策。

本文件包含两套独立策略：
1. ``PriceQuotePolicy``  —— 价格报价策略（my_agent 自有功能，原类名
   ``QuoteReplyPolicy`` 因与原版冲突已改名）。
2. ``QuoteReplyPolicy``  —— 原版 app.brain.quote_reply_policy 的引用回复
   决策策略（对齐反编译骨架逐方法重建）：

   - ``apply(analysis)``            -> ``{'quote_reply': self._decision(analysis)}``
   - ``_decision(analysis)``        -> 键为 ('enabled','should_quote','trigger',
      'reason','target_text','target_message','target_point') 的决策 dict
   - ``_target_message`` (classmethod)
   - ``_has_box`` / ``_message_text`` / ``_normalize_text`` /
     ``_is_user_keyword_rule`` / ``_target_point`` (staticmethod)
   - ``_matches_any_text`` / ``_is_explicit_keyword_rule_match`` (classmethod)
   - ``_quote_allowed_by_send_mode`` (self)

决策触发条件（对齐骨架常量）：
   mode=always                -> '引用回复模式为始终引用。'
   命中用户关键词回复规则       -> '命中关键词回复规则。'
   群聊里客户 @ 助手           -> '群聊里客户 @ 助手。'
   其余 smart 模式             -> 'smart_quote_not_triggered'
阻断条件：
   'quote_reply_disabled' / 'not_reply_decision' /
   'latest_message_not_customer' / 'quote_reply_requires_safe_or_force_mode'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .text_intents import detect_text_intents


@dataclass
class QuoteReply:
    matched: bool = False
    reply: str = ""
    price: str = ""
    price_units: str = ""
    bundles: list = field(default_factory=list)
    trigger_intent: str = ""

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "reply": self.reply,
            "price": self.price,
            "price_units": self.price_units,
            "bundles": self.bundles,
            "trigger_intent": self.trigger_intent,
        }


class PriceQuotePolicy:
    """基于价格关键词 + 配置的价格/套餐规则生成报价回复。

    config 结构（兼容 dict 或对象，取值路径见 _read_config）：
        quote:
          base_price: 199            # 基准价
          price_units: /月            # 价格单位
          bundle_rules:
            - {name: 基础版, price: 199, period: 月, desc: 满足日常使用}
    """

    _DEFAULT_PRICE = "199"
    _DEFAULT_UNITS = "/月"
    _PRICE_KW = ("价格", "多少钱", "报价", "费用", "收费", "价钱", "预算",
                 "贵不贵", "贵吗", "优惠", "折扣")

    def __init__(self, config: Any = None):
        self.config = config
        self.base_price = self._DEFAULT_PRICE
        self.price_units = self._DEFAULT_UNITS
        self.bundle_rules: List[dict] = []
        self._load_config()

    def _read_config(self) -> dict:
        """从 config 读出 quote 配置项（dict 或对象兼容）。"""
        if isinstance(self.config, dict):
            cfg = self.config.get("quote") or {}
            return cfg if isinstance(cfg, dict) else {}
        cfg = getattr(self.config, "quote", None) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _load_config(self) -> None:
        cfg = self._read_config()
        self.base_price = str(cfg.get("base_price", self._DEFAULT_PRICE))
        self.price_units = str(cfg.get("price_units", self._DEFAULT_UNITS))
        rules = cfg.get("bundle_rules")
        if isinstance(rules, list):
            self.bundle_rules = [r for r in rules if isinstance(r, dict)]
        elif isinstance(rules, dict):
            self.bundle_rules = [
                {"name": k, **v} for k, v in rules.items() if isinstance(v, dict)
            ]

    # ------------------------------------------------------------------ 匹配
    def _hit_price(self, text: str) -> bool:
        if not text:
            return False
        return any(k in text for k in self._PRICE_KW)

    def build_reply(self, bundles: Optional[List[dict]] = None) -> str:
        """按 base_price / price_units / bundle_rules 生成报价文本。"""
        price = self.base_price
        units = self.price_units
        bundles = bundles if bundles is not None else self.bundle_rules

        if not bundles:
            # 无套餐：直接给基准价 + 一句价值背书
            return f"这款现在是 {price}{units}，费用透明，没有隐藏收费。您先了解下功能，我可以帮您算哪种方案最合适。"

        # 有套餐：开头报基准价，再列举套餐
        lines = [f"这款基准价是 {price}{units}，我们分几档套餐，您按需要选："]
        for b in bundles:
            name = b.get("name", "套餐")
            bprice = b.get("price", price)
            bperiod = b.get("period", "")
            unit = f"/{bperiod}" if bperiod and not bperiod.startswith("/") else bperiod
            bdesc = b.get("desc", "")
            line = f"{name}：{bprice}{unit if unit else units}"
            if bdesc:
                line += f"，{bdesc}"
            lines.append(line)
        lines.append("您更看重哪类功能？我帮您选最合适的档位。")
        return "\n".join(lines)

    def match(self, text: str) -> QuoteReply:
        """命中价格意图即返回报价回复。"""
        intents = detect_text_intents(text)
        hit_hint = self._hit_price(text)
        if not intents and not hit_hint:
            return QuoteReply(matched=False)
        trigger = "price" if "price" in intents else ("price" if hit_hint else ",".join(intents))
        return QuoteReply(
            matched=True,
            reply=self.build_reply(),
            price=self.base_price,
            price_units=self.price_units,
            bundles=self.bundle_rules,
            trigger_intent=trigger,
        )


# =====================================================================
# 原版 QuoteReplyPolicy —— 引用回复决策（对齐反编译骨架重建）
# =====================================================================

class QuoteReplyPolicy:
    """Decide whether the next reply should use WeChat quote reply.

    config 键（dict 或对象兼容，路径 quote_reply）：
        quote_reply:
          enabled: true          # 总开关（默认 true）
          mode: smart            # smart | off | always
    """

    _MODES = frozenset({"smart", "off", "always"})

    def __init__(self, config: Any = None):
        self.config = config

    # ---------------------------------------------------------------- 公共
    def apply(self, analysis: Dict[str, Any],
              reply: str | None = None) -> Dict[str, Any]:
        """原版签名 apply(self, analysis)；reply 参数为兼容 my_agent 调用点保留。"""
        out: Dict[str, Any] = {}
        out["quote_reply"] = self._decision(
            analysis if isinstance(analysis, dict) else {}
        )
        return out

    # ---------------------------------------------------------------- 决策
    def _decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._quote_cfg()
        enabled = bool(cfg.get("enabled", True))
        mode = str(cfg.get("mode", "smart") or "smart").strip().lower()
        if mode not in self._MODES:
            mode = "smart"

        result: Dict[str, Any] = {
            "enabled": enabled,
            "should_quote": False,
            "trigger": "",
            "reason": "",
            "target_text": "",
            "target_message": {},
            "target_point": {},
        }

        if not enabled or mode == "off":
            result["reason"] = "quote_reply_disabled"
            return result

        decision = analysis.get("decision")
        if isinstance(decision, dict) and str(decision.get("action", "")) == "no_reply":
            result["reason"] = "not_reply_decision"
            return result

        latest = analysis.get("latest_message")
        if not isinstance(latest, dict) or str(latest.get("sender", "")) != "customer":
            result["reason"] = "latest_message_not_customer"
            return result

        if not self._quote_allowed_by_send_mode():
            result["reason"] = "quote_reply_requires_safe_or_force_mode"
            return result

        if mode == "always":
            result.update(
                should_quote=True, trigger="always",
                reason="引用回复模式为始终引用。",
            )
        elif self._is_explicit_keyword_rule_match(analysis):
            result.update(
                should_quote=True, trigger="keyword_rule",
                reason="命中关键词回复规则。",
            )
        elif bool(analysis.get("group_mentioned_me")) or bool(analysis.get("group_mention")):
            result.update(
                should_quote=True, trigger="group_mention",
                reason="群聊里客户 @ 助手。",
            )
        else:
            result["reason"] = "smart_quote_not_triggered"
            return result

        target = self._target_message(analysis, latest)
        result["target_text"] = str(target.get("target_text", "") or "")
        msg = target.get("target_message")
        result["target_message"] = msg if isinstance(msg, dict) else {}
        result["target_point"] = self._target_point(result["target_message"], analysis)
        return result

    def _quote_cfg(self) -> dict:
        if isinstance(self.config, dict):
            cfg = self.config.get("quote_reply") or {}
            return cfg if isinstance(cfg, dict) else {}
        cfg = getattr(self.config, "quote_reply", None) or {}
        return cfg if isinstance(cfg, dict) else {}

    # ------------------------------------------------------- 目标消息定位
    @classmethod
    def _target_message(cls, analysis: Dict[str, Any],
                        latest: Dict[str, Any]) -> Dict[str, Any]:
        """从历史轮次里找出被引用的目标消息（客户发的、且与 latest 文本相关）。"""
        target_texts: List[str] = []
        if isinstance(latest, dict):
            for key in ("content", "text", "bubble_text",
                        "customer_turn_text", "ocr_guard", "latest_message"):
                value = latest.get(key)
                if isinstance(value, dict):
                    continue
                text = cls._normalize_text(value)
                if text:
                    target_texts.append(text)

        candidates: List[Any] = []
        for key in ("customer_turn_messages", "visible_conversation_messages",
                    "conversation_context", "ocr_messages"):
            items = analysis.get(key)
            if isinstance(items, list):
                candidates.extend(items)

        for item in reversed(candidates):
            if not isinstance(item, dict):
                continue
            if str(item.get("sender", "")) != "customer":
                continue
            if cls._matches_any_text(item, target_texts):
                return {"target_text": cls._message_text(item),
                        "target_message": item}

        if isinstance(latest, dict):
            return {"target_text": cls._message_text(latest),
                    "target_message": latest}
        return {}

    @staticmethod
    def _has_box(item: Dict[str, Any]) -> bool:
        try:
            return all(float(item.get(k, 0) or 0) > 0
                       for k in ("left", "top", "right", "bottom"))
        except Exception:
            return False

    @classmethod
    def _matches_any_text(cls, item: Dict[str, Any],
                          target_texts: List[str]) -> bool:
        text = cls._message_text(item)
        if not text:
            return False
        for target in target_texts:
            if not target:
                continue
            if text == target or target in text or text in target:
                return True
        return False

    @staticmethod
    def _message_text(item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        raw = item.get("text") or item.get("content") or item.get("bubble_text") or ""
        return QuoteReplyPolicy._normalize_text(raw)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    # ------------------------------------------------------- 关键词规则命中
    @classmethod
    def _is_explicit_keyword_rule_match(cls, analysis: Dict[str, Any]) -> bool:
        matched = analysis.get("keyword_reply_matched")
        if isinstance(matched, bool) and matched:
            return True
        candidates: List[dict] = []
        rule = analysis.get("keyword_reply_rule")
        if isinstance(rule, dict):
            candidates.append(rule)
        elif isinstance(rule, list):
            candidates.extend(r for r in rule if isinstance(r, dict))
        rules = analysis.get("keyword_reply_rules")
        if isinstance(rules, list):
            candidates.extend(r for r in rules if isinstance(r, dict))
        return any(cls._is_user_keyword_rule(r) for r in candidates)

    @staticmethod
    def _is_user_keyword_rule(rule: Any) -> bool:
        if not isinstance(rule, dict):
            return False
        source = str(rule.get("source") or "").strip()
        if source == "keyword_reply":
            return True
        try:
            return int(rule.get("line_no") or 0) > 0
        except Exception:
            return False

    # ------------------------------------------------------- 目标坐标换算
    @staticmethod
    def _target_point(target: Dict[str, Any],
                      analysis: Dict[str, Any]) -> Dict[str, float]:
        """把目标消息的 bbox 换算为截图比例坐标 {'x_ratio','y_ratio'}。"""
        if not isinstance(target, dict) or not QuoteReplyPolicy._has_box(target):
            return {}
        try:
            left = float(target.get("left") or 0)
            top = float(target.get("top") or 0)
            right = float(target.get("right") or 0)
            bottom = float(target.get("bottom") or 0)
            size = analysis.get("ocr_size") if isinstance(analysis, dict) else None
            if not isinstance(size, dict):
                size = target.get("ocr_size")
            if not isinstance(size, dict):
                size = {}
            width = float(size.get("width") or 0)
            height = float(size.get("height") or 0)
            if width <= 0 or height <= 0:
                return {}
            x_ratio = min(max((left + right) / 2.0 / width, 0.0), 1.0)
            y_ratio = min(max((top + bottom) / 2.0 / height, 0.0), 1.0)
            return {"x_ratio": round(x_ratio, 4), "y_ratio": round(y_ratio, 4)}
        except Exception:
            return {}

    # ------------------------------------------------------- 发送模式检查
    def _quote_allowed_by_send_mode(self) -> bool:
        """键盘效率模式（Enter 直接发送）下引用菜单不可用。"""
        if isinstance(self.config, dict):
            wechat_cfg = self.config.get("wechat")
        else:
            wechat_cfg = getattr(self.config, "wechat", None)
        if not isinstance(wechat_cfg, dict):
            return True
        mode = str(wechat_cfg.get("foreground_keyboard_mode") or "").strip().lower()
        return mode not in ("efficiency", "efficient", "off")


# 向后兼容别名（旧代码里 WeChatQuoteReplyPolicy 即引用回复决策）
WeChatQuoteReplyPolicy = QuoteReplyPolicy
