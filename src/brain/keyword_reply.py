"""Full keyword → reply matching engine.

Rebuilt from disassembled app.brain.keyword_reply.pyc.
Original had ~200 keyword tokens across 8 intent categories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .text_intents import detect_text_intents


# ---- Intent keyword sets extracted from original bytecode ----
PRICE_KEYWORDS = (
    "价格", "多少钱", "收费", "费用", "贵吗", "套餐",
    "报价", "价钱", "几钱", "什么价", "怎么卖", "怎么算",
    "优惠", "便宜", "打折",
)
SERVICE_SCOPE_KEYWORDS = (
    "服务范围", "上门", "哪里", "能上门", "地区", "其他地区",
    "外地", "本地", "全国", "异地", "线上", "远程", "城市",
    "发货范围", "交付范围", "门店地址", "公司地址",
)
BOOKING_KEYWORDS = (
    "预约", "下单", "怎么约", "能约吗", "能约", "约吗",
    "能做", "可以做", "今天能约", "明天能约", "今天能做",
    "明天能做", "当天预约", "购买", "怎么买", "如何购买",
    "我想买", "我要买", "想购买", "要购买",
)
FEATURE_KEYWORDS = (
    "功能", "有什么功能", "有哪些功能", "能做什么", "支持什么",
    "可以做什么", "有什么用", "核心功能", "能干什么",
    "自动回复", "自动通过", "关键词", "知识库", "语音",
)
RISK_PROMISE_KEYWORDS = (
    "百分百", "保证", "绝对", "一定", "承诺", "封号",
    "不封号", "入侵", "安全", "恢复", "修好", "成功",
    "到账", "退款", "赔偿",
)
BUY_LINK_KEYWORDS = (
    "怎么买", "如何买", "哪里买", "购买入口", "购买链接",
    "购买地址", "下单入口", "下单链接", "付款链接", "官网",
    "网址", "链接", "在哪买", "怎么下单", "如何下单",
    "授权码", "定制",
)
COMPLAINT_KEYWORDS = (
    "垃圾", "骗子", "坑人", "不满意", "差评", "离谱",
    "有问题", "投诉", "纠纷",
)
AFTER_SALES_KEYWORDS = (
    "售后", "保修", "质保", "退费", "退货", "坏了",
    "不能用", "没效果", "无效", "泄露",
)
TIME_SLOT_KEYWORDS = (
    "今天", "明天", "后天", "几点", "什么时候", "时间",
    "时段", "安排", "改期",
)

# Compliance guard terms — never directly promise these in auto-replies
_NO_DIRECT_PROMISE_TERMS = ("百分百", "保证", "绝对", "一定", "承诺")
_HUMAN_CONFIRM_TERMS = ("人工确认", "人工核对", "检测后", "不能承诺")

_PUNCT_RE = re.compile(r"[吗呢啊呀吧的了么嘛?？!！。；;，,、]")
_PRICE_UNIT_RE = re.compile(
    r"[\d][\d.,]*(?:元|块|U|u|人民币|¥|￥)"
    r"|起[/／](?:月|年|周|天)|月卡|年卡|周卡|天卡"
)
_ITEM_PRICE_SPLIT_RE = re.compile(r"(?:价格|收费|费用|报价|套餐|多少钱|价钱)$")

# ---- Recovered real keyword/term sets (from app.brain.keyword_reply bytecode consts) ----
# These were lifted directly from the original app's disassembled constants.
POLICY_TERMS = (  # _policy_terms (app.brain.keyword_reply.py:565)
    "退款", "退费", "退货", "赔偿", "补偿", "售后", "保修", "质保",
    "不好用", "无效", "失败", "不成功", "修不好", "成功率", "100%", "百分百",
    "保证", "承诺", "绝对", "隐私", "泄露", "安全", "封号", "不封号",
    "hook", "入侵", "新号", "常用号", "经常使用", "稳定",
)
HANDOFF_TERMS = (  # _handoff_terms default (app.brain.keyword_reply.py:600)
    "退款", "投诉", "赔偿", "售后", "纠纷", "隐私", "泄露", "成功率",
    "100%", "百分百", "保证", "绝对", "修不好", "不成功", "失败", "封号", "不封号",
)
FEATURE_TERMS = (  # _feature_terms (app.brain.keyword_reply.py:686)
    "功能", "有什么功能", "有哪些功能", "能做什么", "支持什么", "可以做什么",
    "有什么用", "核心功能", "能干什么",
)
APPOINTMENT_TERMS = (  # _appointment_terms default (app.brain.keyword_reply.py:553)
    "购买", "怎么买", "如何购买", "哪里买", "买", "下单", "怎么下单", "开通",
    "怎么开通", "发货", "交付", "授权码", "预约", "怎么约", "能约吗", "能约", "约吗",
)
ADDRESS_TERMS = (  # _address_terms default (app.brain.keyword_reply.py:1010)
    "服务范围", "范围", "上门", "哪里", "地区", "其他地区", "外地", "本地",
    "全国", "异地", "线上", "远程", "城市",
)
APPOINTMENT_DETECT_TERMS = (  # _looks_like_appointment_question (app.brain.keyword_reply.py:4576)
    "预约", "下单", "购买", "今天", "明天", "后天", "几点", "什么时候",
    "时间", "时段", "安排", "改期",
)
PURCHASE_ENTRANCE_TERMS = (  # _looks_like_purchase_entrance_question (app.brain.keyword_reply.py:5660)
    "购买入口", "购买链接", "购买地址", "购买网址", "下单入口", "下单链接",
    "付款链接", "网站在哪", "官网", "网址", "链接", "哪里买", "在哪买",
)
PURCHASE_PROCESS_TERMS = (  # _looks_like_purchase_process_question (app.brain.keyword_reply.py:5747)
    "怎么买", "如何购买", "怎么下单", "如何下单", "怎么付款", "如何付款",
    "购买流程", "下单流程", "购买授权码", "授权码怎么买", "我想买", "我要买", "想购买", "要购买",
)
PRICE_GENERIC_TERMS = ("价格", "多少钱", "收费", "费用", "贵吗", "套餐")
PRICE_QUESTION_TERMS = ("价格", "多少钱", "收费", "费用", "贵吗", "套餐", "报价")
PRICE_SPECIFIC_TERMS = ("价格", "多少钱", "收费", "费用", "报价", "贵吗", "套餐", "优惠")
AREA_GENERIC_TERMS = ("服务范围", "上门", "哪里", "能上门")

COMPLAINT_WEAK_TERMS = ("垃圾", "骗子", "坑人", "不满意", "差评", "投诉", "离谱", "有问题")
FAILURE_TERMS = ("修不好", "不成功", "失败", "没效果", "不能用", "坏了")
FEE_TERMS = ("收费", "退费", "退款", "赔", "补偿", "售后")
COMMITMENT_GUARD_TERMS = ("人工确认", "人工核对", "检测后", "不能承诺", "不得承诺", "不保证")
COMMITMENT_DATA_TERMS = ("恢复", "修好", "成功", "隐私", "价格", "收费", "退款")
COMMITMENT_ACCOUNT_TERMS = ("封号", "不封号", "hook", "入侵", "新号", "常用号", "经常使用", "稳定")
ABSOLUTE_TERMS = frozenset({"保证", "百分百", "一定", "承诺", "100%", "绝对"})
ABSOLUTE_TERMS_LIST = ("100%", "百分百", "保证", "绝对", "一定")
RESULT_TERMS = ("恢复", "修好", "成功", "到账", "发货", "开通", "安全", "封号", "不封号")
FEATURE_CAPABILITY_TERMS = ("自动回复", "自动通过", "关键词", "引用", "语音", "群聊", "知识库", "截图", "OCR")
DURATION_ALIASES = {
    "天": ("天卡", "日卡", "一天", "按天", "体验"),
    "周": ("周卡", "一周", "按周"),
    "月": ("月卡", "一个月", "按月", "月套餐"),
    "年": ("年卡", "一年", "按年", "年套餐"),
}
GENERIC_FILLERS = frozenset({"", "大约", "你们", "都", "这个", "大概", "一般", "有", "那个", "全部"})
PRICE_VALUE_UNIT = r"(¥|￥|元|块|人民币|rmb|usd|usdt|u|U|起|月|年|周|天|小时|次|个|台|套|份|永久|终身|平方|平方米|平米|㎡|账号|号)"
PRICE_NAME_TAIL = r"(价格|收费|费用|报价|套餐|多少钱|价钱)$"

_COMPACT_RE = re.compile(r"[\s：:，,。！？!?；;、~～\-_（）()【】\[\]\"'“”‘’/／|]+")
_MEANINGFUL_RE = re.compile(r"[\w\u4e00-\u9fff]")
_PRICE_WORDS = ("多少钱", "多少", "价格", "价钱", "收费", "费用", "报价", "贵吗",
                "套餐", "优惠", "怎么卖", "怎么算", "什么价", "几钱")
_PUNCT2_RE = re.compile(r"[吗呢啊呀吧的了么嘛?？!！。；;，,、]")
_PRICE_SEG_RE = re.compile(
    r"(?P<price>[¥￥]?\s*\d+(?:\.\d+)?\s*(?:元|块|人民币|rmb|usd|usdt|u|U)?"
    r"(?:\s*(?:起|[/／]\s*(?:月|年|周|天|小时|次|个|台|套|份|永久|终身)"
    r"|每(?:月|年|周|天|次)|月|年|周|天|小时|次|个|台|套|份|永久|终身"
    r"|平方|平方米|平米|㎡|账号|号))*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KeywordReplyMatch:
    source: str = "keyword_reply"
    keywords: tuple[str, ...] = ()
    reply: str = ""
    line_no: int = 0
    intent_category: str = ""
    matched: bool = True

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "keywords": list(self.keywords),
            "reply": self.reply,
            "line_no": self.line_no,
            "intent_category": self.intent_category,
        }


def _contains_any(text_lower: str, words) -> bool:
    return any(w in text_lower for w in words if w)


class KeywordReplyMatcher:
    """Match keyword rules from config + built-in intent rules."""

    def __init__(self, config: Any = None):
        if hasattr(config, "keyword_reply"):
            raw_rules = config.keyword_reply or []
        elif isinstance(config, dict):
            rr = config.get("keyword_reply") or []
            raw_rules = list(rr) if isinstance(rr, list) else []
        else:
            raw_rules = []

        # Normalized config dict the rest of the engine reads uniformly.
        self.config = self._normalize_config(config)

        self._rules: list[tuple[tuple[str, ...], str, int]] = []
        for idx, entry in enumerate(raw_rules):
            if not isinstance(entry, str) or "=>" not in entry:
                continue
            kw_part, _, reply_part = entry.partition("=>")
            kws = tuple(k.strip() for k in kw_part.split(",") if k.strip())
            reply = reply_part.strip()
            if kws and reply:
                self._rules.append((kws, reply, idx + 1))

        # Built-in intent groups (from decompiled constants)
        self._intent_map: dict[str, tuple] = {
            "price": PRICE_KEYWORDS,
            "service_scope": SERVICE_SCOPE_KEYWORDS,
            "booking": BOOKING_KEYWORDS,
            "feature": FEATURE_KEYWORDS,
            "risk_promise": RISK_PROMISE_KEYWORDS,
            "buy_link": BUY_LINK_KEYWORDS,
            "complaint": COMPLAINT_KEYWORDS,
            "after_sales": AFTER_SALES_KEYWORDS,
            "time_slot": TIME_SLOT_KEYWORDS,
        }

    @property
    def rule_count(self) -> int:
        return len(self._rules) + len(self._intent_map)

    def match(self, text: str) -> Optional[KeywordReplyMatch]:
        """Return first matched rule or intent-based fallback."""
        if not text or not text.strip():
            return None
        low = text.lower().strip()
        core = _PUNCT_RE.sub("", low).strip()
        if not core:
            return None

        for kws, reply, line_no in self._rules:
            if _contains_any(low, kws) or _contains_any(core, kws):
                return KeywordReplyMatch(keywords=kws, reply=reply,
                                         line_no=line_no, source="config_rule")

        intents = detect_text_intents(text) or []
        for cat_name, words in self._intent_map.items():
            hits = [w for w in words if w in low or w in core]
            if not hits:
                continue
            if cat_name == "risk_promise":
                continue  # handled separately below as guard
            if cat_name == "complaint":
                continue  # requires human handling

            reply = self._build_intent_reply(cat_name, text)
            if reply:
                return KeywordReplyMatch(keywords=tuple(hits), reply=reply,
                                         line_no=len(self._rules)+1,
                                         intent_category=cat_name,
                                         source="builtin_intent")

        # Price-only short query without item context
        if any(w in low for w in ("多少钱", "什么价")) and len(core) <= 6:
            return KeywordReplyMatch(
                keywords=("价格",), line_no=0,
                reply="你想问哪个项目的价格？常见参考请看菜单或发我资料里的项目名。",
                intent_category="price_query_short",
                source="builtin_intent")

        return None

    def is_risky_claim(self, reply_text: str) -> bool:
        """Check if an AI-generated reply contains forbidden promises."""
        low = reply_text.lower()
        return any(t in low for t in _NO_DIRECT_PROMISE_TERMS)

    def sanitize_reply(self, reply_text: str) -> str:
        """Strip forbidden absolute commitments from replies."""
        sanitized = reply_text
        for term in _NO_DIRECT_PROMISE_TERMS:
            sanitized = sanitized.replace(term, "通常")
        if not any(h in sanitized for h in _HUMAN_CONFIRM_TERMS):
            sanitized += "（具体请以客服确认后为准）"
        return sanitized

    def _build_intent_reply(self, category: str, user_text: str) -> str:
        templates = {
            "price": "关于价格，我们会根据具体需求给您详细报价，请问您主要关注哪个项目？",
            "service_scope": "我们目前提供的服务覆盖范围请您看简介或直接告诉我您所在地区，我来确认。",
            "booking": "可以的，方便的话告诉我您的需求和时间，我帮您安排。",
            "feature": "我们的产品支持自动接待和知识库问答等功能，具体场景您可以简单描述一下。",
            "buy_link": "购买入口稍后发给您，请留一下联系方式，我会马上跟进。",
            "after_sales": "您好，遇到问题我们先帮您排查一下，稍后为您处理。",
            "time_slot": "好的，我帮您确认时间安排，稍后给您答复。",
        }
        return templates.get(category, "")

    # ------------------------------------------------------------------
    # Config helpers (not in the 65-list, used by recovered methods)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_config(config: Any) -> dict:
        """Coerce object/dict config into a plain dict the engine can read."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        cfg: dict = {}
        if hasattr(config, "keyword_reply"):
            cfg["keyword_reply"] = getattr(config, "keyword_reply", None)
        for attr in ("business_profile", "business", "reply_rules"):
            if hasattr(config, attr):
                cfg[attr] = getattr(config, attr)
        return cfg

    def _cfg(self) -> dict:
        return self.config if isinstance(self.config, dict) else {}

    def _filter_terms(self, name: str, default) -> list:
        cfg = self._cfg()
        fr = cfg.get("keyword_reply")
        if isinstance(fr, dict):
            flt = fr.get("filter")
            if isinstance(flt, dict):
                vals = flt.get(name)
                if isinstance(vals, str):
                    return [v.strip() for v in re.split(r"[,，、/|;\n]+", vals) if v.strip()]
                if isinstance(vals, (list, tuple)):
                    return [str(v).strip() for v in vals if str(v).strip()]
        return list(default)

    # ------------------------------------------------------------------
    # Shared string helpers (recovered consts)
    # ------------------------------------------------------------------
    @staticmethod
    def _compact(value) -> str:
        return _COMPACT_RE.sub("", str(value)).casefold()

    @staticmethod
    def _is_meaningful_char(value) -> bool:
        return bool(_MEANINGFUL_RE.match(str(value)))

    @staticmethod
    def _remove_price_words(value) -> str:
        text = str(value)
        for term in _PRICE_WORDS:
            text = text.replace(term, " ")
        return _PUNCT2_RE.sub("", text)

    @staticmethod
    def _parse_line(line) -> "Optional[tuple[list[str], str]]":
        text = str(line).strip()
        if not text or text.startswith("#"):
            return None
        if "=>" not in text:
            return None
        left, _, right = text.partition("=>")
        reply = right.strip()
        if not reply:
            return None
        keys = [k.strip() for k in re.split(r"[,，、/|]", left) if k.strip()]
        if not keys:
            return None
        return (keys, reply)

    @staticmethod
    def _normalize_rules_text(value) -> str:
        text = str(value)
        text = text.replace("\r\n", "\n").replace("\n", "\n").replace("\r", "\n")
        return text

    @staticmethod
    def _key_matches_query(query: str, key: str) -> bool:
        q_intents = detect_text_intents(query) or []
        k_intents = detect_text_intents(key) or []
        if q_intents and k_intents and set(q_intents) & set(k_intents):
            return True
        return key in query or KeywordReplyMatcher._compact(key) in KeywordReplyMatcher._compact(query)

    # ------------------------------------------------------------------
    # Top-level matching entry points
    # ------------------------------------------------------------------
    def match_all(self, text: str) -> "list[KeywordReplyMatch]":
        """Return every matched rule/intent for the query (unfiltered)."""
        if not text or not text.strip():
            return []
        matches: list[KeywordReplyMatch] = []

        rules_text = self._cfg().get("keyword_reply", "")
        if isinstance(rules_text, (list, tuple)):
            rules_text = "\n".join(str(r) for r in rules_text)
        else:
            rules_text = str(rules_text or "")
        for line_no, line in enumerate(
            self._normalize_rules_text(rules_text).splitlines(), start=1
        ):
            parsed = self._parse_line(line)
            if not parsed:
                continue
            keys, reply = parsed
            if any(self._key_matches_query(text, k) for k in keys):
                matches.append(KeywordReplyMatch(
                    keywords=tuple(keys), reply=reply,
                    line_no=line_no, source="config_rule"))

        matches.extend(self._dynamic_policy_matches(text))
        matches.extend(self._dynamic_feature_matches(text))
        matches.extend(self._dynamic_handoff_matches(text))
        matches.extend(self._dynamic_appointment_matches(text))
        matches.extend(self._dynamic_address_matches(text))
        matches.extend(self._dynamic_price_matches(text))
        matches.extend(self._dynamic_price_overview_matches(text))
        return matches

    def filter_matches(self, query: str, matches: "list[KeywordReplyMatch]") -> "list[KeywordReplyMatch]":
        """Prefer profile-specific rules over broad generic rules.

        The actual industry words live in data/knowledge/profile.yaml under
        keyword_reply.filter. This keeps the matcher generic while still letting
        each customer profile tune "price vs appointment vs area" priority.
        """
        if not matches:
            return []
        dynamic_policy = [m for m in matches if m.source == "dynamic_policy"]
        dynamic_handoff = [m for m in matches if m.source == "dynamic_handoff"]
        dynamic_price_overview = [m for m in matches if m.source == "dynamic_price_overview"]
        dynamic_feature = [m for m in matches if m.source == "dynamic_feature"]
        dynamic_appointment = [m for m in matches if m.source == "dynamic_appointment"]
        dynamic_address = [m for m in matches if m.source == "dynamic_address"]
        dynamic_price_item = [m for m in matches if m.source == "dynamic_price_item"]

        price_generic_terms = set(PRICE_GENERIC_TERMS)
        area_generic_terms = set(AREA_GENERIC_TERMS)

        def is_generic_price(m: KeywordReplyMatch) -> bool:
            return bool(m.keywords) and all(k in price_generic_terms for k in m.keywords)

        def is_generic_area(m: KeywordReplyMatch) -> bool:
            return bool(m.keywords) and all(k in area_generic_terms for k in m.keywords)

        specific_price = [m for m in dynamic_price_item if not is_generic_price(m)]
        generic_price = [m for m in dynamic_price_item if is_generic_price(m)]
        specific_area = [m for m in dynamic_address if not is_generic_area(m)]
        generic_area = [m for m in dynamic_address if is_generic_area(m)]

        preferred: list[KeywordReplyMatch] = []
        preferred += dynamic_policy
        preferred += dynamic_handoff
        preferred += specific_price
        preferred += dynamic_feature
        preferred += dynamic_appointment
        preferred += specific_area
        preferred += dynamic_price_overview
        preferred += generic_area
        preferred += generic_price
        preferred += [m for m in matches if m.source == "config_rule"]

        seen = set()
        out: list[KeywordReplyMatch] = []
        for m in preferred:
            if id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        return out

    @classmethod
    def _sort_user_keyword_matches(cls, query: str, matches: "list[KeywordReplyMatch]") -> "list[KeywordReplyMatch]":
        compact_query = cls._compact(query)
        price_generic = set(PRICE_GENERIC_TERMS)

        def sort_key(match: KeywordReplyMatch):
            keys = [cls._compact(k) for k in match.keywords]
            exact = any(k == compact_query for k in keys)
            best_len = max((len(k) for k in keys), default=0)
            generic_only = all(k in price_generic for k in keys)
            return (0 if exact else 1, best_len, 0 if generic_only else 1, -match.line_no)

        return sorted(matches, key=sort_key)

    # ------------------------------------------------------------------
    # Policy intent
    # ------------------------------------------------------------------
    def _dynamic_policy_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_policy_question(query):
            return []
        scored = []
        for segment in self._policy_segments(query):
            score = self._score_policy_segment(query, segment)
            scored.append((self._policy_detail_score(len(segment), score), segment))
        scored = [(s, seg) for s, seg in scored if s >= 12]
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [seg for _, seg in scored[:3]]
        reply = self._policy_reply(query, selected)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=self._policy_terms(), reply=reply,
                                  line_no=-1002, source="dynamic_policy")]

    @classmethod
    def _segment_has_policy_signal(cls, segment: str, policy_terms=None) -> bool:
        if policy_terms is None:
            policy_terms = cls._policy_terms()
        comp = cls._compact(segment)
        if any(t in comp for t in policy_terms):
            return True
        if any(t in comp for t in COMMITMENT_DATA_TERMS):
            return True
        if any(t in comp for t in COMMITMENT_ACCOUNT_TERMS):
            return True
        return False

    @classmethod
    def _score_policy_segment(cls, query: str, segment: str) -> int:
        score = 0
        seg_comp = cls._compact(segment)
        query_chars = set(ch for ch in cls._compact(query) if cls._is_meaningful_char(ch))
        seg_chars = set(ch for ch in seg_comp if cls._is_meaningful_char(ch))
        if query_chars & seg_chars:
            score += 12
        if any(t in seg_comp for t in COMMITMENT_GUARD_TERMS):
            score += 18
        if any(t in seg_comp for t in ABSOLUTE_TERMS):
            score += 10
        if any(t in seg_comp for t in RESULT_TERMS):
            score += 8
        if cls._looks_like_commitment_question(query):
            score += 8
        return score

    def _policy_detail_score(self, text_len: int, score: int) -> int:
        if text_len >= 2:
            return score
        return score - 100

    def _policy_reply(self, query: str, segments) -> str:
        parts = []
        for seg in segments:
            clean = self._clean_policy_segment(seg)
            if clean:
                parts.append(clean)
        if not parts:
            return ""
        text = "；".join(parts)
        comp = self._compact(query)
        if "封号" in comp or "不封号" in comp:
            text = text.rstrip("。") + "。不能承诺100%不封号，"
        if any(t in text for t in ABSOLUTE_TERMS_LIST) and "客服" not in text:
            text += "（具体以客服确认为准）"
        return text

    def _clean_policy_segment(self, segment: str) -> str:
        text = re.sub(r"\s+", " ", str(segment)).strip()
        text = text.strip(" -，,。；;")
        text = re.sub(r"^\d*[\.、]?\s*问[:：]", "", text)
        text = re.sub(r"^\d+[\.、]\s*", "", text)
        text = re.sub(r"^问[:：].*?(?:答[:：])", "", text)
        text = re.sub(r"^答[:：]", "", text)
        return text.strip()

    def _policy_segments(self, query: str) -> "list[str]":
        profile = self._cfg().get("business_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        business = profile.get("business") or {}
        if not isinstance(business, dict):
            business = {}
        chunks = []
        for key in ("products_or_services", "company_intro", "contact_info",
                    "service_hours", "reply_style", "industry",
                    "forbidden", "handoff", "store_address"):
            val = business.get(key) or profile.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        segments = []
        seen = set()
        for raw in chunks:
            for seg in re.split(r"[\r\n。；;]+", raw):
                seg = seg.strip()
                comp = self._compact(seg)
                if not comp or comp in seen:
                    continue
                if self._segment_has_policy_signal(seg):
                    seen.add(comp)
                    segments.append(seg)
        return segments

    @classmethod
    def _looks_like_policy_question(cls, query: str) -> bool:
        comp = cls._compact(query)
        if any(t in comp for t in cls._policy_terms()):
            return True
        return cls._looks_like_commitment_question(query)

    @staticmethod
    def _looks_like_commitment_question(query: str) -> bool:
        comp = KeywordReplyMatcher._compact(query)
        if any(t in comp for t in COMMITMENT_GUARD_TERMS):
            return True
        if any(t in comp for t in COMMITMENT_DATA_TERMS):
            return True
        if any(t in comp for t in COMMITMENT_ACCOUNT_TERMS):
            return True
        return False

    @staticmethod
    def _policy_terms() -> tuple:
        return POLICY_TERMS

    # ------------------------------------------------------------------
    # Feature intent
    # ------------------------------------------------------------------
    def _dynamic_feature_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_feature_question(query):
            return []
        reply = self._feature_reply(query)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=self._feature_terms(), reply=reply,
                                  line_no=-1007, source="dynamic_feature")]

    @classmethod
    def _looks_like_feature_question(cls, query: str) -> bool:
        comp = cls._compact(query)
        return any(t in comp for t in cls._feature_terms())

    @staticmethod
    def _feature_terms() -> tuple:
        return FEATURE_TERMS

    def _feature_reply(self, query: str) -> str:
        profile = self._cfg().get("business_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        chunks = []
        for key in ("products_or_services", "company_intro", "identity_text", "reply_style"):
            val = profile.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        candidates = []
        for raw in chunks:
            for seg in re.split(r"[\r\n。；;]+", raw):
                seg = self._clean_feature_segment(seg)
                if seg:
                    candidates.append(seg)
        if not candidates:
            return ""
        candidates.sort(key=lambda c: self._score_feature_reply(c), reverse=True)
        best = candidates[0]
        if self._is_low_information_feature_reply(best):
            return ""
        return best[:160].rstrip(" ，,。；;") + "。"

    @classmethod
    def _clean_feature_segment(cls, raw: str) -> str:
        t = re.sub(r"\s+", " ", str(raw)).strip()
        t = t.strip(" -，,。；;")
        t = re.sub(r"^\d*[\.、]?\s*问[:：]", "", t)
        t = re.sub(r"^\d+[\.、]\s*", "", t)
        t = re.sub(r"^问[:：].*?(?:答[:：])", "", t)
        t = re.sub(r"^答[:：]", "", t)
        parts = [p.strip() for p in re.split(r"[:：]", t, maxsplit=1)]
        t = parts[-1] if parts else t
        if cls._looks_like_feature_list(t):
            return t
        if any(term in t for term in ("支持", "可以", "可")):
            return t
        return t

    def _looks_like_feature_list(self, text: str) -> bool:
        value = text or ""
        if not value:
            return False
        marks = ("、", "，", ",", "/", "／")
        if sum(value.count(m) for m in marks) >= 8:
            return True
        return "等" in value

    @classmethod
    def _score_feature_reply(cls, text: str) -> int:
        score = 0
        comp = cls._compact(text)
        if "功能" in comp:
            score += 12
        score += min(len(text), 18)
        score += sum(text.count(m) for m in ("、", "，", ",", "/", "／")) * 3
        if any(t in text for t in FEATURE_CAPABILITY_TERMS):
            score += 4
        if "一句话定位" in text or "帮助用户" in text:
            score += 8
        return score

    @classmethod
    def _is_low_information_feature_reply(cls, reply: str) -> bool:
        comp = cls._compact(reply)
        if len(comp) <= 8 and not any(t in comp for t in FEATURE_CAPABILITY_TERMS):
            return True
        return False

    # ------------------------------------------------------------------
    # Handoff (转人工) intent
    # ------------------------------------------------------------------
    def _dynamic_handoff_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_handoff_question(query):
            return []
        reply = self._handoff_reply(query)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=tuple(self._handoff_terms()), reply=reply,
                                  line_no=-1001, source="dynamic_handoff")]

    def _looks_like_handoff_question(self, query: str, terms=None) -> bool:
        if terms is None:
            terms = self._handoff_terms()
        comp = self._compact(query)
        if any(t in comp for t in terms):
            return True
        if self._looks_like_unknown_data_question(query):
            return True
        if self._looks_like_complaint(query):
            return True
        if self._looks_like_commitment_question(query):
            return True
        return False

    def _handoff_terms(self, filter_cfg=None) -> "list":
        return self._filter_terms("handoff_terms", HANDOFF_TERMS)

    def _handoff_reply(self, query: str) -> str:
        rules = self._reply_rule_text("handoff")
        if isinstance(rules, str) and rules.strip():
            return rules.strip()[:50]
        # TODO-RECOVER: 原版无配置时返回空串，此处兜底为转人工话术（推断）
        return "您的问题需要人工处理，我为您转接客服，请稍候。"

    @staticmethod
    def _looks_like_unknown_data_question(compact: str) -> bool:
        comp = KeywordReplyMatcher._compact(compact)
        data_terms = ("资料", "没写", "未写", "没有写", "没提", "未提", "资料外", "其他项目")
        ability_terms = ("能做", "可以做", "能不能", "可不可以", "可以吗", "能吗")
        if any(t in comp for t in data_terms):
            return True
        if any(t in comp for t in ability_terms):
            return True
        return False

    @staticmethod
    def _looks_like_complaint(query: str) -> bool:
        comp = KeywordReplyMatcher._compact(query)
        if any(t in comp for t in COMPLAINT_WEAK_TERMS):
            return True
        if any(t in comp for t in FAILURE_TERMS):
            return True
        if any(t in comp for t in FEE_TERMS):
            return True
        return False

    # ------------------------------------------------------------------
    # Appointment intent
    # ------------------------------------------------------------------
    def _dynamic_appointment_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_appointment_question(query):
            return []
        reply = self._appointment_reply(query)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=tuple(self._appointment_terms()), reply=reply,
                                  line_no=-1004, source="dynamic_appointment")]

    def _looks_like_appointment_question(self, query: str) -> bool:
        comp = self._compact(query)
        if any(t in comp for t in APPOINTMENT_DETECT_TERMS):
            return True
        if any(t in comp for t in TIME_SLOT_KEYWORDS):
            return True
        return False

    def _appointment_terms(self, filter_cfg=None) -> "list":
        return self._filter_terms("appointment_terms", APPOINTMENT_TERMS)

    def _appointment_reply(self, query: str) -> str:
        if self._looks_like_purchase_process_question(query):
            return self._purchase_process_reply(None)
        rules = self._reply_rule_text("appointment")
        if isinstance(rules, str) and rules.strip():
            return self._clean_business_rule_reply(rules)
        profile = self._cfg().get("business_profile") or {}
        business = profile.get("business") or {} if isinstance(profile, dict) else {}
        chunks = []
        for key in ("purchase_flow", "delivery_flow", "service_hours", "contact_info"):
            val = business.get(key) if isinstance(business, dict) else None
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        if chunks:
            return self._clean_business_rule_reply("；".join(chunks))
        # TODO-RECOVER: 原版无配置时回复推断
        return "好的，方便告诉我您想预约的项目和时间吗？我帮您安排。"

    # ------------------------------------------------------------------
    # Purchase (购买入口/购买流程) intent
    # ------------------------------------------------------------------
    @classmethod
    def _looks_like_purchase_entrance_question(cls, query: str, direct_terms=None) -> bool:
        if direct_terms is None:
            direct_terms = PURCHASE_ENTRANCE_TERMS
        comp = cls._compact(query)
        return any(t in comp for t in direct_terms)

    @classmethod
    def _looks_like_purchase_process_question(cls, query: str, process_terms=None) -> bool:
        if process_terms is None:
            process_terms = PURCHASE_PROCESS_TERMS
        comp = cls._compact(query)
        if any(t in comp for t in process_terms):
            return True
        return cls._looks_like_purchase_entrance_question(query)

    def _purchase_process_reply(self, entrance_reply=None) -> str:
        profile = self._cfg().get("business_profile") or {}
        business = profile.get("business") or {} if isinstance(profile, dict) else {}
        rules = self._reply_rule_text("purchase_flow")
        line_scores = (
            ("购买入口", 24), ("下单渠道", 22), ("购买渠道", 22), ("付款方式", 18),
            ("支付方式", 18), ("交付方式", 14), ("购买后", 10), ("授权码", 8),
        )
        seen = set()
        scored = []
        chunks = []
        for key in ("purchase_flow", "delivery_flow", "products_or_services", "appointment", "lead_capture"):
            val = business.get(key) if isinstance(business, dict) else None
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        for raw in chunks:
            for seg in re.split(r"[\r\n；;]+", raw):
                seg = self._clean_purchase_process_segment(seg)
                if seg and seg not in seen:
                    seen.add(seg)
                    score = 0
                    for marker, sc in line_scores:
                        if marker in seg:
                            score = max(score, sc)
                    scored.append((score, seg))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return "；".join(s for _, s in scored[:6])
        if isinstance(rules, str) and rules.strip():
            return rules.strip()
        # TODO-RECOVER: 原版无配置时流程话术推断
        return "您可以这样购买：先确认套餐，付款后我们会发送授权码并开通，随后按交付流程使用。"

    def _clean_purchase_process_segment(self, raw: str) -> str:
        text = re.sub(r"\s+", " ", str(raw)).strip()
        text = text.strip(" -，,。；;")
        text = re.sub(r"^客户要[^：:]{0,40}[：:]", "", text)
        return text.strip()

    def _purchase_entrance_reply(self, query=None) -> str:
        profile = self._cfg().get("business_profile") or {}
        business = profile.get("business") or {} if isinstance(profile, dict) else {}
        rules = self._reply_rule_text("purchase_flow")
        chunks = []
        for key in ("products_or_services", "purchase_flow", "appointment", "lead_capture"):
            val = business.get(key) if isinstance(business, dict) else None
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        for raw in chunks:
            for seg in re.split(r"[\r\n；;]+", raw):
                seg = self._clean_purchase_entrance_segment(seg)
                if seg:
                    return seg
        if isinstance(rules, str) and rules.strip():
            return rules.strip()
        # TODO-RECOVER: 原版无配置时入口话术推断
        return "购买入口稍后发您，您也可以留个联系方式，我马上跟进。"

    def _clean_purchase_entrance_segment(self, raw: str) -> str:
        text = re.sub(r"\s+", " ", str(raw)).strip()
        text = text.strip(" -，,。；;")
        text = re.sub(r"https?://[^\s，,。；;]+", "", text)
        text = re.sub(r"^客户要[^：:]{0,50}[：:]", "", text)
        if any(k in text for k in ("购买入口", "下单入口", "购买链接", "下单链接",
                                   "购买", "下单", "入口", "链接", "网址", "网站")):
            if len(text) > 120:
                text = text[:117] + "..."
            return text
        return ""

    # ------------------------------------------------------------------
    # Address / service scope intent
    # ------------------------------------------------------------------
    def _dynamic_address_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_address_question(query):
            return []
        reply = self._address_reply(query)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=tuple(self._address_terms()), reply=reply,
                                  line_no=-1005, source="dynamic_address")]

    def _looks_like_address_question(self, query: str, terms=None) -> bool:
        if terms is None:
            terms = self._address_terms()
        comp = self._compact(query)
        return any(t in comp for t in terms)

    def _address_terms(self, filter_cfg=None) -> "list":
        return self._filter_terms("area_generic_terms", ADDRESS_TERMS)

    def _address_reply(self, query: str) -> str:
        profile = self._cfg().get("business_profile") or {}
        business = profile.get("business") or {} if isinstance(profile, dict) else {}
        service_area = business.get("service_area") if isinstance(business, dict) else None
        delivery_area = business.get("delivery_area") if isinstance(business, dict) else None
        service_scope = business.get("service_scope") if isinstance(business, dict) else None
        blob = " ".join(str(x) for x in (service_area, delivery_area, service_scope) if x)
        online_markers = {"仅线上服务", "线上服务", "纯线上服务"}
        if any(m in blob for m in online_markers):
            return "我们是纯线上服务。"
        parts = []
        if service_area:
            parts.append("服务范围：" + str(service_area).strip(" 。；;"))
        if delivery_area:
            parts.append(str(delivery_area).strip(" 。；;"))
        scope_facts = self._customer_visible_scope_facts([service_area, delivery_area, service_scope])
        for fact in scope_facts:
            txt = fact.get("text")
            if txt:
                parts.append(str(txt))
        store_address = business.get("store_address") if isinstance(business, dict) else None
        if store_address:
            parts.append("地址：" + str(store_address).strip(" 。；;"))
        if not parts:
            return ""
        return "；".join(parts) + "。"

    @classmethod
    def _customer_visible_scope_facts(cls, values) -> "list[dict[str, str]]":
        facts: list[dict[str, str]] = []
        seen = set()
        internal_markers = ("以这里为准", "未写明", "未填写", "客户问到", "人工确认",
                            "不能直接承诺", "不得直接承诺", "资料不明确", "缺少关键信息")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            for raw in re.split(r"[\r\n；;]+", value):
                raw = raw.strip()
                if not raw or any(m in raw for m in internal_markers):
                    continue
                fact = re.sub(r"\s+", " ", raw).strip().strip(" -，,。；;")
                fact = re.sub(
                    r"^(?:服务范围|服务地区|服务区域|上门范围|发货范围|交付范围|门店地址|公司地址|地址)[:：]\s*",
                    "", fact).strip()
                comp = cls._compact(fact)
                if comp and comp not in seen:
                    seen.add(comp)
                    facts.append({"text": fact})
        return facts

    def _reply_rule_text(self, key: str) -> str:
        rules = self._cfg().get("reply_rules")
        if isinstance(rules, dict):
            val = rules.get(key)
            if isinstance(val, str):
                return val.strip()
            if isinstance(val, list):
                return "；".join(str(v).strip() for v in val if str(v).strip())
        return ""

    def _first_config_text(self, paths) -> str:
        cfg = self._cfg()
        for path in paths:
            if isinstance(path, (list, tuple)):
                section = cfg
                ok = True
                for part in path:
                    if isinstance(section, dict) and part in section:
                        section = section[part]
                    else:
                        ok = False
                        break
                if ok and isinstance(section, str):
                    return section.strip()
            elif isinstance(path, str) and path in cfg:
                val = cfg[path]
                if isinstance(val, str):
                    return val.strip()
        return ""

    def _clean_business_rule_reply(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text)).strip()
        clean = clean.strip(" -，,。；;")
        clean = re.sub(r"^客户要[^：:]{0,40}[：:]", "", clean)
        for marker in ("不能直接承诺", "需要人工确认", "不得直接承诺"):
            if marker in clean:
                clean += "（需人工确认）"
                break
        if len(clean) > 220:
            clean = clean[:217] + "..."
        return clean

    # ------------------------------------------------------------------
    # Price intent (item-level + overview + catalog)
    # ------------------------------------------------------------------
    def _dynamic_price_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_price_question(query):
            return []
        items = self._extract_price_items(query)
        if not items:
            return []
        scored = [(self._score_price_item(query, it.get("name", "")), it) for it in items]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] < 2:
            return []
        selected = [it for s, it in scored if s >= 2][:8]
        reply = self._price_items_reply(selected)
        if not reply:
            return []
        return [KeywordReplyMatch(keywords=tuple(self._price_question_terms()), reply=reply,
                                  line_no=-1000, source="dynamic_price_item")]

    def _dynamic_price_overview_matches(self, query: str) -> "list[KeywordReplyMatch]":
        if not self._looks_like_generic_price_question(query):
            return []
        if self._looks_like_catalog_price_question(query):
            return []
        items = self._extract_price_items(query)
        if not items:
            return []
        parts = [self._price_items_reply([it]) for it in items[:5]]
        parts = [p for p in parts if p]
        reply = "你想问哪个项目的价格？常见参考：" + "；".join(parts) + "。"
        return [KeywordReplyMatch(keywords=tuple(self._price_question_terms()), reply=reply,
                                  line_no=-1003, source="dynamic_price_overview")]

    def _looks_like_price_question(self, query: str, terms=None) -> bool:
        comp = self._compact(query)
        if terms is None:
            terms = set(self._price_question_terms())
        else:
            terms = set(terms)
        terms.update({"什么价", "价钱", "多少", "几钱"})
        return any(t in comp for t in terms)

    def _looks_like_generic_price_question(self, query: str) -> bool:
        if not self._looks_like_price_question(query):
            return False
        stripped = self._remove_price_words(self._compact(query))
        stripped = re.sub(r"[吗呢啊呀吧的了么嘛?？!！。；;，,、]", "", stripped).strip()
        return stripped in GENERIC_FILLERS

    def _looks_like_catalog_price_question(self, query: str) -> bool:
        if not self._looks_like_price_question(query):
            return False
        if self._looks_like_generic_price_question(query):
            return True
        comp = self._compact(query)
        items = self._extract_price_items(query)
        scored = [self._score_price_item(query, it.get("name", "")) for it in items]
        if scored and max(scored) >= 8:
            return False
        catalog_terms = self._price_catalog_terms()
        return any(t in comp for t in catalog_terms)

    def _price_catalog_terms(self) -> "set":
        cfg = self._cfg()
        out = set()
        fr = cfg.get("keyword_reply")
        if isinstance(fr, dict):
            flt = fr.get("filter")
            if isinstance(flt, dict):
                vals = flt.get("price_specific_terms")
                if isinstance(vals, str):
                    out.update(v.strip() for v in re.split(r"[,，、/|;\n]+", vals) if v.strip())
                elif isinstance(vals, (list, tuple)):
                    out.update(str(v).strip() for v in vals if str(v).strip())
        profile = cfg.get("business_profile") or {}
        business = profile.get("business") or {} if isinstance(profile, dict) else {}
        name_keys = ("brand_name", "business_name", "store_name", "company_name",
                     "product_name", "service_name", "assistant_name", "name")
        for k in name_keys:
            v = business.get(k) if isinstance(business, dict) else None
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
        out.update(self._price_question_terms())
        return out

    def _price_question_terms(self, filter_cfg=None) -> "list":
        return self._filter_terms("price_question_terms", PRICE_QUESTION_TERMS)

    def _extract_price_items(self, query=None) -> "list[dict[str, str]]":
        profile = self._cfg().get("business_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        business = profile.get("business") or {}
        if not isinstance(business, dict):
            business = {}
        chunks = []
        for key in ("products_or_services", "company_intro", "industry"):
            val = business.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val)
        items = []
        seen = set()
        for raw in chunks:
            for seg in re.split(r"[\r\n，,。；;]+", raw):
                seg = seg.strip()
                if not seg:
                    continue
                parsed = self._parse_price_segment(seg)
                if parsed:
                    name = parsed.get("name", "")
                    comp = self._compact(name)
                    if comp and comp not in seen:
                        seen.add(comp)
                        items.append(parsed)
        return items

    @classmethod
    def _parse_price_segment(cls, segment: str) -> "Optional[dict[str, str]]":
        text = str(segment).strip().strip(" \t\r\n-—")
        match = _PRICE_SEG_RE.search(text)
        if not match:
            return None
        raw_name = text[:match.start()].strip(" \t\r\n:：-—")
        name = cls._clean_price_item_name(raw_name)
        price = match.group("price").strip()
        return {"name": name, "price": price}

    @classmethod
    def _looks_like_price_value(cls, price: str, prefix: str) -> bool:
        prefix_compact = cls._compact(prefix) if prefix else ""
        if re.search(PRICE_VALUE_UNIT, str(price), re.IGNORECASE):
            return True
        if re.search(PRICE_NAME_TAIL, prefix_compact):
            return True
        return False

    def _clean_price_item_name(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value)).strip()
        text = re.sub(
            r"^(价格以.*?为准|我们|主要|提供|包含|支持|可以|可做|可卖|卖|做|产品|服务|项目|报价|价格)\s*",
            "", text)
        text = re.sub(r"^(和|及|以及|还有|另外|另有|其中)\s*", "", text)
        text = text.strip(" \t\r\n:：,，、-—")
        return text

    @classmethod
    def _score_price_item(cls, query: str, name: str) -> int:
        if not name:
            return 0
        compact_query = cls._compact(query)
        compact_name = cls._compact(name)
        if not compact_name:
            return 0
        query_focus = cls._remove_price_words(compact_query)
        score = 0
        aliases = cls._item_aliases(name)
        item_chars = set(ch for ch in compact_name if cls._is_meaningful_char(ch))
        query_chars = set(ch for ch in query_focus if cls._is_meaningful_char(ch))
        overlap = item_chars & query_chars
        coverage = len(overlap) / max(len(query_chars), 1)
        score += min(len(overlap) * 12, 30)
        if any(a in compact_query for a in aliases):
            score += 18
        if coverage >= 0.5:
            score += 14
        elif coverage >= 0.3:
            score += 10
        return score

    @classmethod
    def _item_aliases(cls, name: str) -> "set":
        aliases = set()
        comp = cls._compact(name)
        if not comp:
            return aliases
        for part in re.split(r"[/／|、,，]+", comp):
            part = part.strip()
            if part:
                aliases.add(part)
        for dur, dur_aliases in DURATION_ALIASES.items():
            if dur in comp:
                aliases.update(dur_aliases)
        if "永久" in comp or "终身" in comp:
            aliases.update(("永久", "永久版", "终身", "终身版"))
        return aliases

    def _price_items_reply(self, items) -> str:
        if not items:
            return ""
        parts = []
        for item in items:
            name = (item.get("name") or "").strip()
            price = (item.get("price") or "").strip()
            if name and price:
                parts.append(f"{name} {price}")
            elif name:
                parts.append(name)
        if not parts:
            return ""
        return "；".join(parts) + "。"