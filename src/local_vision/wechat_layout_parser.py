"""微信窗口固定布局解析 + 消息分类引擎 —— 对齐原版 app.local_vision.wechat_layout_parser。

原版 WeChatLayoutParser 是核心消息分析引擎，包含：
  1. 布局区域分类（搜索栏 / 联系人列表 / 聊天区）
  2. 消息候选分类（25+ 类别：支付/广告/投诉/售后/问价/问地址/拒绝给电话等）
  3. 联系人名称提取（含名称提示词匹配）
  4. 系统联系人识别
  5. 消息分析结果生成（含决策/意图/风险）
  6. 消息相似度判定（过滤非消息文本）
  7. 视觉分隔线检测（"以下为新消息"）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..message.parser import ContactInfo, ChatMessage, MessageSide, MessageType
from .wechat_ocr_parser import WeChatOCRParser


# =====================================================================
# 数据类
# =====================================================================

class LayoutRegion(Enum):
    SEARCH = "search"
    CONTACTS = "contacts"
    CHAT = "chat"
    UNKNOWN = "unknown"


class MessageSideLabel(Enum):
    SELF = "self"
    OTHER = "other"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class MessageCandidate:
    """消息候选 —— 对齐原版 MessageCandidate dataclass。"""
    text: str = ""
    side: str = ""                     # "left" | "right" | "center"
    sender: str = ""
    left: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    confidence: float = 0.0

    def fingerprint(self) -> str:
        return f"{self.side}_{self.sender}_{self.text[:30]}"


@dataclass
class ChunkMessage:
    """聊天区中一个被归类后的消息块。"""
    sender: str = ""
    content: str = ""
    timestamp: str = ""
    side: str = MessageSideLabel.UNKNOWN.value
    region: str = LayoutRegion.CHAT.value
    group_sender: str = ""
    box: list = field(default_factory=list)
    y_center: float = 0.0
    confidence: float = 0.0
    raw_text: str = ""

    def fingerprint(self) -> str:
        return f"{self.side}_{self.sender}_{self.content[:30]}_{self.timestamp}"


@dataclass
class WechatLayoutResult:
    region_lines: dict = field(default_factory=dict)   # region -> List[dict]
    contacts: List[ContactInfo] = field(default_factory=list)
    messages: List[ChunkMessage] = field(default_factory=list)


# =====================================================================
# 微信默认布局比例（可配置）
# =====================================================================
DEFAULT_SEARCH_HEIGHT = 0.08
DEFAULT_LIST_WIDTH_RATIO = 0.28

# =====================================================================
# 关键词常量（对齐原版全量关键词列表）
# =====================================================================

# 系统联系人 / 服务号
SYSTEM_CONTACT_KEYWORDS = [
    "微信支付", "微信游戏", "服务号", "公众号", "订阅号", "文件传输助手",
    "微信团队", "QQ邮箱提醒", "企业微信",
]

# 系统消息
SYSTEM_MESSAGE_KEYWORDS = [
    "语音通话", "视频通话", "已在其它设备", "已在其他设备", "微信电脑版",
    "对方撤回了一条消息", "你撤回了一条消息", "拍了拍", "开启了朋友验证",
    "同意了好友验证",
]

# 广告关键词（对齐原版 AD_KEYWORDS）
AD_KEYWORDS = [
    "福利", "活动", "促销", "优惠", "领取", "充值", "满减", "红包", "点击",
    "立即", "推广", "广告", "免费", "抽奖", "下载", "小程序", "外卖", "美团",
    "饿了么", "特价", "限时", "扫码",
]

# 支付通知（对齐原版 PAYMENT_KEYWORDS）
PAYMENT_KEYWORDS = [
    "赞赏到账通知", "收款金额", "已收款", "微信转账", "转账", "到账时间",
    "收款", "付款", "¥", "￥",
]

# 投诉/售后关键词（对齐原版 COMPLAINT_KEYWORDS）
COMPLAINT_KEYWORDS = [
    "投诉", "退款", "退货", "差评", "举报", "维权", "客服", "投诉电话",
    "315", "消协", "工商", "不满意", "退钱", "赔偿", "质量问题",
    "坏了", "不能用", "不好用", "有问题", "不好使", "不管用",
]

# 拒绝给电话关键词（对齐原版 REFUSE_PHONE_KEYWORDS）
REFUSE_PHONE_KEYWORDS = [
    "不方便", "不给", "不提供", "没有电话", "不方便给", "不太方便",
    "不能给", "不告诉", "隐私", "保密", "个人信息", "不想给",
    "不方便留电话", "不想留手机号", "不留电话", "电话就不留了",
    "直接给我地址", "我自己去", "不用留电话", "不提供手机号",
    "不想给电话", "不想留电话", "不方便给手机号", "不留电话了",
    "电话不方便给", "不要电话", "不要电话了", "不打电话", "别打电话",
    "不方便接电话", "手机号算了", "先不加微信", "不加微信",
]

# 地址询问关键词（对齐原版 ADDRESS_KEYWORDS）
ADDRESS_KEYWORDS = [
    "地址", "位置", "在哪里", "怎么走", "怎么去", "导航", "定位",
    "门店", "店铺", "店面", "实体店", "线下", "门面", "哪条路",
    "几楼", "几层", "哪个区", "什么路", "什么街道",
]

# 价格询问关键词（对齐原版 PRICE_KEYWORDS）
PRICE_KEYWORDS = [
    "多少钱", "价格", "报价", "费用", "收费", "怎么收费", "怎么算",
    "贵不贵", "便宜", "折扣", "打几折", "优惠价", "最低价", "原价",
    "现价", "单价", "总价", "预算", "多少钱一", "收费标准",
]

# 联系人名称提示词（对齐原版 _CONTACT_NAME_HINTS）
CONTACT_NAME_HINTS = [
    "微信", "WeChat", "新消息", "通讯录", "聊天", "联系人",
]

# 新消息分隔线
UNREAD_DIVIDER_KEYWORDS = ["以下为新消息", "以下为未读消息", "你以上是新消息"]

# 链接关键词（用于广告判定）
LINK_KEYWORDS = ["http", "www.", "扫码", "点击", "领取", "立即", "小程序"]


# =====================================================================
# 独立函数：跳过联系人判定（对齐原版 is_skipped_contact）
# =====================================================================

def is_skipped_contact(contact: Any, config: Any) -> bool:
    """判定联系人是否在跳过列表中（对齐原版 is_skipped_contact）。

    从 config.wechat.system_contacts 读取跳过列表，支持：
    1. 精确 substring 匹配
    2. 拼音模糊匹配（兜 OCR 同音错读）
    """
    from ..brain.skip_contacts import is_skipped_contact as _impl
    return _impl(contact, config)


# =====================================================================
# WeChatLayoutParser —— 对齐原版完整实现
# =====================================================================

class WechatLayoutParser:
    """微信布局解析 + 消息分类引擎。

    对齐原版 app.local_vision.wechat_layout_parser，包含：
    - 布局区域分类
    - 消息候选分类（25+ 类别）
    - 联系人名称提取
    - 系统联系人识别
    - 消息分析结果生成
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 window_info: Optional[Dict[str, Any]] = None,
                 screenshot_path: str = "",
                 list_width_ratio: float = DEFAULT_LIST_WIDTH_RATIO,
                 search_height_ratio: float = DEFAULT_SEARCH_HEIGHT,
                 ocr_parser: Optional[WeChatOCRParser] = None,
                 stable_frames: int = 1):
        self.config = config or {}
        self.window_info = window_info or {}
        self.screenshot_path = screenshot_path
        self.list_width_ratio = list_width_ratio
        self.search_height_ratio = search_height_ratio
        self._ocr = ocr_parser or WeChatOCRParser()

        self._contact_title_frames: dict = {}
        self._last_contact_title: str = ""
        self._message_stability: dict = {}
        self._stable_frames_required = max(1, int(stable_frames))

    # ==================================================================
    # 主入口：parse() —— 对齐原版 parse()
    # ==================================================================

    def parse(self, source,
              window_width: Optional[int] = None,
              window_height: Optional[int] = None) -> WechatLayoutResult:
        """解析 OCR 结果，产出生结构化的布局 + 联系人 + 消息列表。

        source 可为 OCRResult 或 line dict 列表。
        """
        lines = self._ocr.process(source)

        width = window_width or self._infer_width(lines)
        height = window_height or self._infer_height(lines)

        region_lines = {r.value: [] for r in LayoutRegion}
        for line in lines:
            region = self._classify_region(line, width, height)
            region_lines[region.value].append(line)

        contacts = self.parse_contacts(region_lines[LayoutRegion.CONTACTS.value])
        messages = self.parse_contact_list_chunks(
            region_lines[LayoutRegion.CHAT.value],
            contacts=contacts,
        )

        return WechatLayoutResult(
            region_lines=region_lines,
            contacts=contacts,
            messages=messages,
        )

    # ==================================================================
    # 消息候选分类 —— 对齐原版 _message_candidates()
    # ==================================================================

    def _message_candidates(self,
                            ocr_messages: list,
                            contact_name: str = "",
                            window_width: int = 0,
                            window_height: int = 0) -> List[Dict[str, Any]]:
        """对 OCR 消息进行 25+ 类别分类（对齐原版 _message_candidates）。

        返回每个候选的分类标注：
        - system / notice / system_notice / no_reply
        - payment_notice / ad_or_promotion / customer
        - text_or_image / unknown / self_latest_message
        - complaint_or_after_sales / handoff / refuse_full_phone
        - weak_lead_draft / address_question / reply_draft / price_question
        """
        candidates: List[Dict[str, Any]] = []
        if not ocr_messages:
            return candidates

        contact_lower = (contact_name or "").lower()
        is_system_contact = self._is_system_contact(contact_name)

        # 排序：按 y 坐标（bottom）升序
        sorted_msgs = sorted(
            ocr_messages,
            key=lambda m: getattr(m, "bottom", getattr(m, "y_center", 0))
            if hasattr(m, "bottom") or hasattr(m, "y_center") else 0
        )

        for msg in sorted_msgs:
            if hasattr(msg, "as_chat_line"):
                m = msg.as_chat_line()
            elif isinstance(msg, dict):
                m = msg
            else:
                continue

            text = (m.get("text", "") or "").strip()
            side = m.get("side", "")
            sender = m.get("sender", "") or m.get("group_sender", "")

            candidate = {
                "title": contact_name,
                "visible_text": text,
                "controls": m,
                "rect": m.get("box", []),
                "system": False,
                "notice": False,
                "system_notice": False,
                "no_reply": False,
                "payment_notice": False,
                "customer": False,
                "text_or_image": "text",
                "ad_or_promotion": False,
                "unknown": False,
                "self_latest_message": False,
                "complaint_or_after_sales": False,
                "handoff": False,
                "refuse_full_phone": False,
                "weak_lead_draft": False,
                "address_question": False,
                "reply_draft": "",
                "price_question": False,
            }

            if not text:
                candidate["unknown"] = True
                candidates.append(candidate)
                continue

            # 系统联系人消息
            if is_system_contact or self._is_system_contact(sender):
                candidate["system"] = True
                candidate["system_notice"] = True
                candidate["no_reply"] = True
                candidates.append(candidate)
                continue

            # 自身消息
            if side == "self" or side == "right":
                candidate["self_latest_message"] = True
                candidate["no_reply"] = True
                candidates.append(candidate)
                continue

            # 客户消息分类
            candidate["customer"] = True
            text_lower = text.lower()

            # 支付通知
            if self._contains_any(text, PAYMENT_KEYWORDS):
                candidate["payment_notice"] = True
                candidate["no_reply"] = True
                candidates.append(candidate)
                continue

            # 广告/推广
            if self._looks_like_ad(text):
                candidate["ad_or_promotion"] = True
                candidate["no_reply"] = True
                candidates.append(candidate)
                continue

            # 投诉/售后
            if self._contains_any(text, COMPLAINT_KEYWORDS):
                candidate["complaint_or_after_sales"] = True

            # 拒绝给电话
            if self._contains_any(text, REFUSE_PHONE_KEYWORDS):
                candidate["refuse_full_phone"] = True
                candidate["weak_lead_draft"] = True

            # 地址询问
            if self._contains_any(text, ADDRESS_KEYWORDS):
                candidate["address_question"] = True

            # 价格询问
            if self._contains_any(text, PRICE_KEYWORDS):
                candidate["price_question"] = True

            # 转人工
            if self._contains_any(text, ["转人工", "人工客服", "人工服务", "找人工"]):
                candidate["handoff"] = True

            candidates.append(candidate)

        return candidates

    # ==================================================================
    # 结果生成 —— 对齐原版 _result()
    # ==================================================================

    def _result(self,
                contact_name: str = "",
                candidates: Optional[List[Dict[str, Any]]] = None,
                msg_type: str = "",
                intent: str = "",
                decision: str = "",
                reply_draft: str = "",
                confidence: float = 0.0,
                platform: str = "wechat",
                screen_state: str = "chat",
                ) -> Dict[str, Any]:
        """生成完整的分析结果（对齐原版 _result() schema）。

        返回字段：
        - wechat / chat_or_list / type / content / high
        - platform / screen_state / current_contact / latest_message
        - intent / decision / action / risk_level / should_reply / reason
        - parser / contact / msg_type / conf
        """
        candidates = candidates or []
        latest_candidate = candidates[-1] if candidates else {}
        latest_text = latest_candidate.get("visible_text", "")

        # 从候选列表中提取最新客户消息
        customer_candidates = [c for c in candidates if c.get("customer")]
        customer_text = ""
        if customer_candidates:
            customer_text = customer_candidates[-1].get("visible_text", "")

        # 判定是否应回复
        should_reply = bool(
            decision in ("reply", "reply_draft", "weak_lead_draft", "ai")
            and reply_draft
            and not latest_candidate.get("no_reply", False)
        )

        # 风险等级
        risk_level = "low"
        if latest_candidate.get("complaint_or_after_sales"):
            risk_level = "high"
        elif latest_candidate.get("handoff"):
            risk_level = "medium"

        return {
            "wechat": True,
            "chat_or_list": "chat",
            "type": msg_type or "text",
            "content": latest_text,
            "high": bool(latest_candidate.get("complaint_or_after_sales")),
            "platform": platform,
            "screen_state": screen_state,
            "current_contact": contact_name,
            "latest_message": {
                "text": latest_text,
                "sender": "customer" if latest_candidate.get("customer") else "system",
                "side": "left" if latest_candidate.get("customer") else "",
            },
            "intent": intent or "unknown",
            "decision": decision or "no_reply",
            "action": "",
            "risk_level": risk_level,
            "should_reply": should_reply,
            "reason": latest_candidate.get("reason", ""),
            "parser": "wechat_layout_parser",
            "contact": contact_name,
            "msg_type": msg_type or "text",
            "conf": confidence,
        }

    # ==================================================================
    # 聊天区结构化解析（被 assistant 主循环调用）
    # ==================================================================

    def parse_chat_messages(self, bgr: np.ndarray, ocr_lines: Optional[list] = None,
                            window_width: Optional[int] = None,
                            window_height: Optional[int] = None,
                            list_width_ratio: Optional[float] = None,
                            search_height_ratio: Optional[float] = None,
                            ):
        """结构化解析右侧聊天区 → 返回 (all_messages, new_messages, unread_anchor)。

        - 只取右侧聊天区文本（剔除左侧列表 + 顶部搜索栏）
        - 按气泡/行聚合，判定 side（self/other）
        - 前置过滤系统消息、服务号支付通知、广告
        """
        lines = self._ocr.process(ocr_lines if ocr_lines is not None else bgr)
        h, w = bgr.shape[:2]
        lwr = list_width_ratio if list_width_ratio is not None else self.list_width_ratio
        shr = search_height_ratio if search_height_ratio is not None else self.search_height_ratio

        list_end = max(300, min(int(w * lwr), 420))
        chat_lines = [
            l for l in lines
            if (l["x_min"] + l["x_max"]) / 2 > list_end and l["y_center"] > h * shr
        ]

        groups = self._segment_chat(chat_lines)

        entries: List[tuple] = []
        anchor_seen = False
        divider_y = self._detect_divider_visual(bgr, list_end, int(h * shr), w, h)

        for grp in groups:
            text = "".join(l.get("text", "") for l in grp).strip()
            if self._is_divider(text):
                anchor_seen = True
                entries.append((True, None))
                continue
            if self._is_timestamp(text):
                continue
            if any(k in text for k in SYSTEM_MESSAGE_KEYWORDS):
                continue
            if any(k in text for k in SYSTEM_CONTACT_KEYWORDS):
                continue
            if divider_y > 0 and not anchor_seen:
                grp_y = float(np.mean([l.get("y_center", 0) for l in grp]))
                if grp_y < divider_y:
                    continue
            m = self._group_to_chat_message(grp, w)
            if m is None:
                continue
            if self._is_ignorable_message(m):
                continue
            entries.append((False, m))

        all_messages = [m for flag, m in entries if m is not None]

        if anchor_seen:
            new_messages: List[ChatMessage] = []
            on = False
            for flag, m in entries:
                if flag:
                    on = True
                    continue
                if on and m is not None and m.side == MessageSide.OTHER and m.content:
                    new_messages.append(m)
        else:
            if divider_y > 0:
                chat_bottom_y = float(divider_y)
            else:
                chat_bottom_y = h * 0.55
            new_messages = [m for m in all_messages
                            if m.side == MessageSide.OTHER and m.content
                            and m.y_center > chat_bottom_y]

        if self._stable_frames_required > 1 and new_messages:
            gated = [m for m in new_messages
                     if m.side != MessageSide.OTHER or not m.content
                     or self.check_message_stability("current_chat", m.content)]
            new_messages = gated

        return all_messages, new_messages, anchor_seen

    def analyze_chat_screen(self, bgr: np.ndarray,
                            ocr_lines: Optional[List[dict]] = None) -> Dict[str, Any]:
        """本地无模型分析入口：截图 + OCR 行 → 原版视觉分析 schema。

        串起 OCREngine 输出 → parse_chat_messages（结构化消息）→
        _message_candidates（业务分类）→ _result（原版 JSON schema），
        供 observe_service 在视觉模型未配置时作为主读取路径。
        """
        lines = list(ocr_lines or [])
        if lines:
            lines = self._ocr.process(lines)

        h, w = bgr.shape[:2]

        # 0. 列表画面守卫：顶部搜索栏 + 左侧列表区大量时间戳 + 右侧无聊天内容
        #    → 当前是真正会话列表，不能把列表预览当聊天消息（防误回复）。
        #    分栏模式下左侧列表始终存在，必须同时满足"右侧无聊天内容"才判 list。
        import re as _re
        lwr = self.list_width_ratio
        list_end = max(300, min(int(w * lwr), 420))
        time_like = [l for l in lines
                     if _re.fullmatch(r"\d{1,2}:\d{2}", (l.get("text", "") or "").strip())]
        list_time_like = [l for l in time_like
                          if (l.get("x_max") or 0) < list_end]
        has_search = any("搜索" in (l.get("text", "") or "") for l in lines)
        chat_lines = [l for l in lines
                      if (l.get("x_min") or 0) > list_end + 20
                      and not _re.fullmatch(r"\d{1,2}:\d{2}",
                                             (l.get("text", "") or "").strip())]
        is_list_view = (
            has_search
            and len(list_time_like) >= 3
            and len(chat_lines) < 2
        )
        if is_list_view:
            result = self._result(contact_name="", candidates=[],
                                  msg_type="text", intent="contact_list",
                                  decision="no_reply", confidence=0.6,
                                  screen_state="list")
            result["chat_or_list"] = "list"
            result["parser"] = "wechat_layout_parser"
            return result

        # 1. 联系人名（限定聊天窗口顶部标题区，防止把长消息行当标题）
        title_lines = [l for l in lines if l.get("y_center", 10 ** 9) < h * 0.09]
        contact = self._extract_contact_name(title_lines)
        contact = _re.sub(r"\s*\d{1,2}:\d{2}\s*$", "", contact or "").strip()

        # 2. 结构化消息（assistant 主循环同款解析：气泡切分 + side 判定）
        all_msgs, new_msgs, anchor = self.parse_chat_messages(bgr, ocr_lines=lines)

        # 3. ChatMessage → dict 交给业务分类
        msg_dicts: List[Dict[str, Any]] = []
        for m in all_msgs:
            msg_dicts.append({
                "text": m.content or "",
                "side": "right" if m.side == MessageSide.SELF else "left",
                "sender": getattr(m, "sender", "") or "",
                "box": list(getattr(m, "box", []) or []),
                "y_center": getattr(m, "y_center", 0),
            })
        candidates = self._message_candidates(
            msg_dicts, contact_name=contact,
            window_width=w, window_height=h)

        # 4. 汇总决策：有新客户消息 → 交下游（关键词/FAQ/文本模型）生成回复
        if new_msgs:
            decision = "reply_draft"
            intent = "chat_message"
            confidence = min(0.9, 0.55 + 0.05 * len(new_msgs))
        elif candidates:
            decision = "no_reply"
            intent = "chat_no_new_message"
            confidence = 0.5
        else:
            decision = "no_reply"
            intent = "chat_empty"
            confidence = 0.3

        result = self._result(contact_name=contact, candidates=candidates,
                              msg_type="text", intent=intent,
                              decision=decision, confidence=confidence)

        # 5. 注入新消息明细（下游管线读 latest_message.content）
        if new_msgs:
            latest = new_msgs[-1]
            is_other = latest.side == MessageSide.OTHER
            result["latest_message"].update({
                "text": latest.content or "",
                "content": latest.content or "",
                "sender": "customer" if is_other else "self",
                "side": "left" if is_other else "right",
            })
            result["content"] = latest.content or ""
            result["new_messages"] = [
                {"content": m.content or "",
                 "side": "left" if m.side == MessageSide.OTHER else "right"}
                for m in new_msgs
            ]
            result["unread_anchor"] = bool(anchor)
            # 有新客户消息时放行下游回复生成（_result 的 should_reply
            # 要求 reply_draft 非空，本地路径的回复由文本模型生成）
            if is_other:
                result["should_reply"] = True

        result["parser"] = "wechat_layout_parser"
        return result

    # ==================================================================
    # 消息稳定性追踪
    # ==================================================================

    def check_message_stability(self, contact_key: str, content: str) -> bool:
        """Track repeated signature across frames; only stable messages pass."""
        import re as _re
        sig = _re.sub(r"[\s\W_]+", "", (content or "").lower())
        if not sig:
            return False
        slot = self._message_stability.setdefault(contact_key, {})
        if slot.get("signature") == sig:
            slot["frames"] = int(slot.get("frames", 0)) + 1
        else:
            slot["signature"] = sig
            slot["frames"] = 1
        return slot["frames"] >= self._stable_frames_required

    def reset_message_stability(self, contact_key: str = "") -> None:
        if contact_key:
            self._message_stability.pop(contact_key, None)
        else:
            self._message_stability.clear()

    def stabilize_contact_title(self, new_title: str) -> str:
        """Smooth chat title across frames; short flickers fall back."""
        t = (new_title or "").strip()
        key = "".join(t.split())[:24].lower()
        if key == "".join(self._last_contact_title.split())[:24].lower():
            return self._last_contact_title
        n = self._contact_title_frames.get(key, 0) + 1
        self._contact_title_frames[key] = n
        if n >= 2 or len(t) > len(self._last_contact_title):
            self._last_contact_title = t
            self._contact_title_frames.clear()
            return t
        return self._last_contact_title or t

    # ==================================================================
    # 联系人名称提取 —— 对齐原版 _extract_contact_name()
    # ==================================================================

    def _extract_contact_name(self, lines: List[dict]) -> str:
        """从 OCR 文本行中提取当前聊天联系人名称（对齐原版）。"""
        candidates = []
        for line in lines:
            text = (line.get("text", "") or "").strip()
            if not text:
                continue

            # 过滤掉明显不是联系人名称的行
            if self._is_timestamp(text):
                continue
            if self._is_divider(text):
                continue
            if any(k in text for k in SYSTEM_MESSAGE_KEYWORDS):
                continue

            # 清理文本
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            # 检查是否看起来像联系人标题
            if self._looks_like_contact_title(cleaned):
                candidates.append(cleaned)

        if candidates:
            # 取最长且最稳定的候选
            candidates.sort(key=len, reverse=True)
            return self.stabilize_contact_title(candidates[0])

        return ""

    def _clean_text(self, text: str) -> str:
        """清理文本：去除明显不是联系人名称的噪声。"""
        if not text:
            return ""
        cleaned = text.strip()
        # 去除 "微信" 等前缀
        for hint in CONTACT_NAME_HINTS:
            if cleaned.startswith(hint):
                cleaned = cleaned[len(hint):].strip()
        # 去除尾部的未读数标记，如 "张三(3)"
        import re
        cleaned = re.sub(r'[（(]\s*\d+\s*[）)]\s*$', '', cleaned)
        # 去除尾部空格和标点
        cleaned = cleaned.strip().rstrip("，。！？、：；")
        return cleaned

    def _looks_like_contact_title(self, text: str) -> bool:
        """判断文本是否看起来像联系人标题（对齐原版）。"""
        if not text:
            return False
        # 太短的不像联系人名
        if len(text) < 2:
            return False
        # 太长的可能不是
        if len(text) > 30:
            return False
        # 含关键字的不是联系人
        if any(k in text for k in SYSTEM_MESSAGE_KEYWORDS):
            return False
        if self._is_timestamp(text):
            return False
        if self._is_divider(text):
            return False
        return True

    # ==================================================================
    # 系统联系人判定 —— 对齐原版 _is_system_contact()
    # ==================================================================

    def _is_system_contact(self, name: str) -> bool:
        """判定是否为系统联系人/服务号（对齐原版）。"""
        if not name:
            return False
        name_lower = name.strip().lower()
        if not name_lower:
            return False
        return any(k.lower() in name_lower for k in SYSTEM_CONTACT_KEYWORDS)

    # ==================================================================
    # 广告判定 —— 对齐原版 _looks_like_ad()
    # ==================================================================

    def _looks_like_ad(self, text: str) -> bool:
        """判定文本是否像广告（对齐原版 _looks_like_ad）。"""
        if not text:
            return False
        low = text.lower()
        ad_hits = sum(1 for k in AD_KEYWORDS if k in low)
        # 命中 >= 2 个广告关键词，或命中 1 个且带链接词
        if ad_hits >= 2:
            return True
        if ad_hits >= 1 and self._contains_any(text, LINK_KEYWORDS):
            return True
        # 检查 URL 模式
        import re
        if re.search(r'(http|www\.)', low):
            return True
        return False

    # ==================================================================
    # 消息相似度判定 —— 对齐原版 _is_message_like()
    # ==================================================================

    def _is_message_like(self, text: str) -> bool:
        """判定文本是否像一条有效消息（对齐原版 _is_message_like）。"""
        if not text or not text.strip():
            return False
        stripped = text.strip()
        # 黑名单：不是消息的文本
        blacklists = [
            "微信", "WeChat", "通讯录", "发现", "我", "设置",
            "搜索", "取消", "确定", "发送", "输入", "消息",
            "小程序", "视频号", "看一看", "搜一搜",
        ]
        if stripped in blacklists:
            return False
        if len(stripped) < 2:
            return False
        if self._is_timestamp(stripped):
            return False
        return True

    # ==================================================================
    # 文本归一化 —— 对齐原版 _normalize()
    # ==================================================================

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化文本：去空白、全角转半角（对齐原版 _normalize）。"""
        if not text:
            return ""
        import re
        # 去多余空白
        result = re.sub(r'\s+', '', text)
        # 全角 ASCII 转半角
        out = []
        for ch in result:
            code = ord(ch)
            if 65281 <= code <= 65374:
                out.append(chr(code - 65248))
            else:
                out.append(ch)
        return "".join(out).strip()

    # ==================================================================
    # 文本截断 —— 对齐原版 _short()
    # ==================================================================

    @staticmethod
    def _short(text: str, limit: int = 200) -> str:
        """截断文本到指定长度（对齐原版 _short）。"""
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    # ==================================================================
    # 包含判定 —— 对齐原版 _contains_any()
    # ==================================================================

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        """检查文本是否包含任意关键词（对齐原版 _contains_any）。"""
        if not text:
            return False
        low = text.lower()
        return any(k.lower() in low for k in keywords)

    # ==================================================================
    # 工具方法
    # ==================================================================

    def _segment_chat(self, lines: List[dict]) -> List[List[dict]]:
        """把聊天区文本行按气泡聚合为组。"""
        ordered = self._ocr.sort_by_y(lines)
        groups: List[List[dict]] = []
        for line in ordered:
            if groups and self._same_chat_message(groups[-1][-1], line):
                groups[-1].append(line)
            else:
                groups.append([line])
        return groups

    def _same_chat_message(self, prev: dict, cur: dict) -> bool:
        """判定 prev、cur 是否属于同一条消息气泡。"""
        prev_text = prev.get("text", "")
        if self._is_timestamp(prev_text) or self._is_divider(prev_text):
            return False
        gap = abs(cur["y_center"] - prev["y_center"])
        if gap > 26.0:
            return False
        prev_x = (prev["x_min"] + prev["x_max"]) / 2
        cur_x = (cur["x_min"] + cur["x_max"]) / 2
        if abs(prev_x - cur_x) > 90.0:
            return False
        return True

    def _group_to_chat_message(self, group: List[dict], window_width: int) -> Optional[ChatMessage]:
        """把一个气泡文本组转成 ChatMessage，并判定 side。"""
        if not group:
            return None
        box_lines = [np.array(g["box"]) for g in group if g.get("box")]
        if box_lines:
            box_all = np.vstack(box_lines)
            bubble_center_x = float(np.mean(box_all[:, 0]))
            y_center = float(np.mean(box_all[:, 1]))
        else:
            bubble_center_x = float(np.mean(
                [g["x_min"] + (g["x_max"] - g["x_min"]) / 2 for g in group]))
            y_center = float(np.mean([g["y_center"] for g in group]))

        full_text = "".join(g.get("text", "") for g in group).strip()
        if not full_text and not box_lines:
            return None

        msg = ChatMessage()
        msg.content = full_text
        msg.raw_text = full_text
        msg.confidence = float(np.mean([g.get("score", 0.5) for g in group]))
        msg.box = box_all.tolist() if box_lines else np.array([])
        msg.y_center = y_center

        if bubble_center_x > window_width * 0.5:
            msg.side = MessageSide.SELF
        else:
            msg.side = MessageSide.OTHER

        if full_text.startswith("[图片]") or full_text.startswith("[表情]"):
            msg.msg_type = MessageType.IMAGE
        elif full_text.startswith("[语音]"):
            msg.msg_type = MessageType.VOICE
        elif full_text.startswith("[") and full_text.endswith("]"):
            msg.msg_type = MessageType.EMOJI

        return msg

    def _is_ignorable_message(self, m: ChatMessage) -> bool:
        text = (m.content or "").strip()
        if not text:
            return False
        if len(text) < 2:
            return True
        if text in ("...", "…", "发送", "输入", "消息", ""):
            return True
        if any(k in text for k in SYSTEM_MESSAGE_KEYWORDS):
            return True
        if any(k in text for k in SYSTEM_CONTACT_KEYWORDS):
            return True
        if m.side == MessageSide.OTHER:
            if any(k in text[:20] for k in PAYMENT_KEYWORDS):
                return True
            if any(k in text for k in AD_KEYWORDS):
                return True
        return False

    @staticmethod
    def _is_divider(text: str) -> bool:
        return any(k in text for k in UNREAD_DIVIDER_KEYWORDS)

    @staticmethod
    def _is_timestamp(text: str) -> bool:
        import re
        patterns = [
            r'^\d{1,2}:\d{2}$',
            r'^\d{1,2}:\d{2}:\d{2}$',
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'昨天\s*\d{1,2}:\d{2}',
            r'今天\s*\d{1,2}:\d{2}',
            r'上午\s*\d{1,2}:\d{2}',
            r'下午\s*\d{1,2}:\d{2}',
            r'晚上\s*\d{1,2}:\d{2}',
        ]
        return any(re.match(p, text.strip()) for p in patterns)

    def _detect_divider_visual(self, bgr: np.ndarray,
                                list_end: int, search_top: int,
                                window_w: int, window_h: int) -> int:
        """视觉检测微信"以下为新消息"分隔线，返回 y 坐标，未找到返回 -1。"""
        import cv2
        try:
            chat_h = window_h - search_top
            chat_w = window_w - list_end
            if chat_h < 60 or chat_w < 60:
                return -1
            chat = bgr[search_top:window_h, list_end:window_w]
            if chat.size == 0:
                return -1
            gray = cv2.cvtColor(chat, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 90)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                    minLineLength=int(chat_w * 0.3),
                                    maxLineGap=10)
            if lines is None:
                return -1
            candidates = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) > 2:
                    continue
                y_mid = (y1 + y2) // 2
                if y_mid < 10 or y_mid > chat_h - 10:
                    continue
                x1_c = max(0, x1)
                x2_c = min(chat_w - 1, x2)
                if x2_c <= x1_c:
                    continue
                roi = chat[y_mid:y_mid + 2, x1_c:x2_c]
                if roi.size == 0:
                    continue
                mean_color = roi.mean(axis=(0, 1))
                if 100 < mean_color[0] < 210 and abs(
                        float(mean_color[0]) - float(mean_color[1])) < 15:
                    candidates.append((y_mid, x2_c - x1_c))
            if not candidates:
                return -1
            candidates.sort(key=lambda x: -x[1])
            return search_top + candidates[0][0]
        except Exception:
            return -1

    # ==================================================================
    # 区域归类
    # ==================================================================

    def _classify_region(self, line: dict, width: float, height: float) -> LayoutRegion:
        x_center = (line["x_min"] + line["x_max"]) / 2
        if height > 0 and line["y_center"] <= height * self.search_height_ratio:
            return LayoutRegion.SEARCH
        if width > 0 and x_center <= width * self.list_width_ratio:
            return LayoutRegion.CONTACTS
        return LayoutRegion.CHAT

    def _infer_width(self, lines: List[dict]) -> int:
        if not lines:
            return 800
        return int(max(line["x_max"] for line in lines))

    def _infer_height(self, lines: List[dict]) -> int:
        if not lines:
            return 600
        return int(max(line["y_center"] for line in lines) * 1.1 + 50)

    # ==================================================================
    # 联系人识别
    # ==================================================================

    def parse_contacts(self, list_lines: List[dict]) -> List[ContactInfo]:
        """从左侧联系人列表行解析出联系人（含未读、最近消息）。"""
        contacts: List[ContactInfo] = []
        for line in self._ocr.sort_by_y(list_lines):
            text = line.get("text", "").strip()
            if not text:
                continue
            contact = ContactInfo(name=text)
            contacts.append(contact)
        return contacts

    # ==================================================================
    # 聊天区消息归属
    # ==================================================================

    def parse_contact_list_chunks(self, chat_lines: List[dict],
                                  contacts: Optional[List[ContactInfo]] = None,
                                  image_width: Optional[int] = None
                                  ) -> List[ChunkMessage]:
        """把右侧聊天区文本行按 y 聚合成消息块，判定 side 与 sender 归属。"""
        if not chat_lines:
            return []

        width = image_width or self._infer_width(chat_lines)
        center_x = width / 2.0
        contact_names = [c.name for c in contacts] if contacts else []

        chunks: List[List[dict]] = []
        for line in self._ocr.sort_by_y(chat_lines):
            placed = False
            for group in chunks:
                if self._same_chunk(group[-1], line):
                    group.append(line)
                    placed = True
                    break
            if not placed:
                chunks.append([line])

        messages = []
        for group in chunks:
            msg = self._chunk_to_message(group, center_x, contact_names)
            if msg and (msg.content or msg.sender):
                messages.append(msg)
        return messages

    def _same_chunk(self, prev: dict, cur: dict) -> bool:
        if prev.get("text") and self._is_timestamp(prev["text"]):
            return False
        gap = abs(cur["y_center"] - prev["y_center"])
        return gap <= 24.0

    def _chunk_to_message(self, group: List[dict], center_x: float,
                          contact_names: List[str]) -> ChunkMessage:
        box_lines = [np.array(g["box"]) for g in group if g.get("box")]
        if box_lines:
            box_all = np.vstack(box_lines)
            box = box_all.tolist()
            bubble_center_x = float(np.mean(box_all[:, 0]))
            y_center = float(np.mean(box_all[:, 1]))
        else:
            box, bubble_center_x, y_center = [], center_x, float(
                np.mean([g["y_center"] for g in group]))

        full_text = "".join(g.get("text", "") for g in group).strip()

        msg = ChunkMessage(
            content=full_text,
            raw_text=full_text,
            confidence=float(np.mean([g.get("score", 0.5) for g in group])),
            box=box,
            y_center=y_center,
        )

        if self._is_timestamp(full_text):
            msg.side = MessageSideLabel.SYSTEM.value
            msg.timestamp = full_text
            msg.content = ""
            return msg

        msg.side = (MessageSideLabel.SELF.value if bubble_center_x > center_x
                    else MessageSideLabel.OTHER.value)
        if msg.side == MessageSideLabel.OTHER.value:
            msg.sender = self._match_sender(bubble_center_x, contact_names, group)
        else:
            msg.sender = ""
        return msg

    def _match_sender(self, bubble_center_x: float, contact_names: List[str],
                      group: List[dict]) -> str:
        if not contact_names:
            return ""
        return contact_names[0]