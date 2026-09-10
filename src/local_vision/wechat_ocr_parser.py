"""微信聊天截图 OCR 解析器 —— 对齐原版 app.local_vision.wechat_ocr_parser。

原版 WeChatOcrParser 的功能：
  1. OCR 文本后处理：去重 / 按 y 排序 / 合并同行
  2. OcrMessage 数据类：side / sender / confidence / parts / group_sender_name
  3. 布局分析：chat_left / chat_right / chat_mid / header_bottom / input_top / send_buttons
  4. 联系人提取与多帧稳定
  5. 消息分类（25+ 类别）
  6. 群聊发送者标签分割
  7. 语音消息检测
  8. 区域 OCR + 回退到全屏 OCR
  9. 消息稳定性多帧追踪
  10. 客户消息提取
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from ..ocr.engine import OCRResult
except Exception:
    OCRResult = None


# =====================================================================
# 数据类
# =====================================================================

@dataclass
class OcrMessage:
    """微信聊天消息块 —— 对齐原版 OcrMessage dataclass。"""
    side: str = ""                     # "left" | "right" | "center"
    sender: str = ""                   # 发送者名称
    confidence: float = 0.0
    parts: List[dict] = field(default_factory=list)  # 文本行列表
    group_sender_name: str = ""        # 群聊发送者标签

    @property
    def text(self) -> str:
        return "".join(p.get("text", "") for p in self.parts).strip()

    @property
    def cx(self) -> float:
        """气泡中心 x 坐标。"""
        if not self.parts:
            return 0.0
        xs = [(p.get("x_min", 0) + p.get("x_max", 0)) / 2 for p in self.parts]
        return sum(xs) / len(xs)

    def as_chat_line(self) -> dict:
        """转成单行 dict 表示（兼容下游）。"""
        return {
            "text": self.text,
            "sender": self.sender,
            "side": self.side,
            "confidence": self.confidence,
            "group_sender": self.group_sender_name,
            "cx": self.cx,
            "parts": self.parts,
        }

    def add(self, item: dict) -> None:
        self.parts.append(item)
        if not self.sender:
            self.sender = item.get("sender_name", "")

    def replace_with_body(self, item: dict) -> None:
        """替换内容（群聊发送者标签合并到消息体）。"""
        self.parts = [item]

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "sender": self.sender,
            "confidence": self.confidence,
            "text": self.text,
            "group_sender_name": self.group_sender_name,
            "bubble_text": self.bubble_text(),
        }

    def bubble_text(self) -> str:
        """气泡内完整文本（用于签名/去重）。"""
        return self.text


# =====================================================================
# 关键词常量（对齐原版 20+ 个关键词列表）
# =====================================================================

SYSTEM_CONTACT_KEYWORDS = {
    "微信支付", "微信团队", "微信运动", "QQ邮箱提醒", "微信游戏",
    "微信安全", "微信广告", "微信公众平台", "微信读书", "腾讯新闻",
    "微信红包", "微信电话", "QQ音乐", "腾讯视频", "小程序",
}

SYSTEM_MESSAGE_KEYWORDS = {
    "撤回了一条消息", "你撤回了一条消息", "加入了群聊", "退出了群聊",
    "被移出群聊", "已解散", "修改群名为", "修改了群公告",
    "开启了朋友验证", "已拒绝", "已过期", "已过期或已被清理",
    "对方正在输入", "对方已取消",
}

AD_KEYWORDS = {
    "点击领取", "免费领取", "限时优惠", "点击查看", "扫码关注",
    "分享有礼", "帮我砍一刀", "帮我助力", "抢红包", "红包雨",
    "拼团", "秒杀", "满减", "优惠券", "抽奖",
}

PAYMENT_KEYWORDS = {
    "转账", "红包", "已收款", "已付款", "零钱", "账单",
    "收款", "付款", "AA收款", "群收款",
}

BRAND_KEYWORDS = {"品牌", "什么牌子", "哪个牌子", "什么品牌"}
ADDRESS_KEYWORDS = {"地址", "位置", "在哪", "哪里", "怎么走", "发个位置"}
PRICE_KEYWORDS = {"价格", "多少钱", "怎么卖", "报价", "怎么收费", "费用"}
CATALOG_KEYWORDS = {"目录", "列表", "菜单", "产品", "有哪些", "种类"}
APPOINTMENT_KEYWORDS = {"预约", "约时间", "什么时候", "有空", "时间"}
TRAFFIC_KEYWORDS = {"路线", "怎么去", "交通", "地铁", "公交", "停车"}
PRODUCT_KEYWORDS = {"材料", "材质", "规格", "尺寸", "大小", "重量", "颜色"}
AVAILABILITY_KEYWORDS = {"有货", "库存", "还有吗", "卖完了", "缺货"}
MATERIAL_KEYWORDS = {"发个图", "照片", "图片", "视频", "看看", "发一下"}
HUMAN_SERVICE_KEYWORDS = {"人工", "客服", "转人工", "联系", "电话"}

COMPLAINT_KEYWORDS = {"投诉", "退款", "举报", "差评", "坑", "骗", "假货", "维权"}
REFUSE_PHONE_KEYWORDS = {"不方便电话", "不要电话", "别打电话", "不加微信", "别加微信"}


def _line_to_dict(item) -> dict:
    """把 OCR 原始条目 (box, text, score) 或 dict 规范化为 line dict。"""
    if isinstance(item, dict):
        box = item.get("box")
        box_arr = np.array(box) if box is not None else None
        text = str(item.get("text", ""))
        score = float(item.get("score", 0.5))
    else:
        box, text, score = item[0], str(item[1]), float(item[2])
        box_arr = np.array(box)

    if box_arr is not None and box_arr.size >= 8:
        x_min = float(np.min(box_arr[:, 0]))
        x_max = float(np.max(box_arr[:, 0]))
        y_center = float(np.mean(box_arr[:, 1]))
    else:
        x_min = x_max = 0.0
        y_center = 0.0

    return {
        "text": text.strip(),
        "score": score,
        "box": box_arr.tolist() if box_arr is not None and box_arr.size else [],
        "y_center": y_center,
        "x_min": x_min,
        "x_max": x_max,
    }


# =====================================================================
# WeChatOCRParser —— 对齐原版 WeChatOcrParser
# =====================================================================

class WeChatOCRParser:
    """微信聊天截图 OCR 解析器。

    对齐原版 app.local_vision.wechat_ocr_parser.WeChatOcrParser
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[Callable[[str], None]] = None,
                 dedupe_similarity: float = 0.95,
                 row_y_tolerance_ratio: float = 0.5):
        self.config = config or {}
        self.logger = logger
        self.dedupe_similarity = dedupe_similarity
        self.row_y_tolerance_ratio = row_y_tolerance_ratio

        # ---- 多帧稳定性状态（对齐原版） ----
        self._stable_contact_name: str = ""
        self._stable_contact_score: float = 0.0
        self._stable_contact_short_variant_key: str = ""
        self._stable_contact_short_variant_count: int = 0
        self._stable_message_signature: str = ""
        self._stable_message_frames: int = 0

        # ---- 配置项（对齐原版 config.get） ----
        self._ocr_region_left_px: int = self.config.get("ocr_region_left_px", 0)
        self._ocr_region_right_margin_px: int = self.config.get("ocr_region_right_margin_px", 0)
        self._ocr_chat_left_px: int = self.config.get("ocr_chat_left_px", 0)
        self._ocr_header_bottom_px: int = self.config.get("ocr_header_bottom_px", 0)
        self._ocr_input_top_px: int = self.config.get("ocr_input_top_px", 0)
        self._ocr_input_top_ratio: float = self.config.get("ocr_input_top_ratio", 0.75)
        self._ocr_contact_stabilize_enabled: bool = self.config.get("ocr_contact_stabilize_enabled", True)
        self._ocr_contact_short_variant_accept_frames: int = self.config.get("ocr_contact_short_variant_accept_frames", 3)
        self._ocr_reply_stabilize_enabled: bool = self.config.get("ocr_reply_stabilize_enabled", True)
        self._ocr_group_sender_split_enabled: bool = self.config.get("ocr_group_sender_split_enabled", True)
        self._ocr_voice_status_enabled: bool = self.config.get("ocr_voice_status_enabled", True)
        self._ocr_reply_stable_frames: int = self.config.get("ocr_reply_stable_frames", 3)

        self._region_filter_enabled_val: bool = True
        self._region_fallback_enabled_val: bool = True
        self._message_stabilize_enabled: bool = True
        self._group_sender_split_enabled: bool = self._ocr_group_sender_split_enabled
        self._voice_status_enabled: bool = self._ocr_voice_status_enabled

    # =================================================================
    # 主入口：parse() —— 对齐原版
    # =================================================================

    def parse(self, window_info: Optional[dict] = None,
              screenshot_path: Optional[str] = None,
              ocr_items: Optional[list] = None) -> Dict[str, Any]:
        """解析微信聊天截图，返回结构化结果。

        对齐原版 WeChatOcrParser.parse(window_info, screenshot_path)

        Returns dict with keys:
            contact, msg_type, content, intent, decision, reason,
            ocr_items, messages, voice_messages, elapsed, ocr_region,
            text_score, fallback_used, fallback_checked, fallback_error,
            notice, system_notice, voice, voice_message, no_visible_message,
            self_latest_message, latest_top, boundary_y,
            customer_turn_messages, ocr_stability, latest_group_sender,
            reply_draft
        """
        import time
        t0 = time.time()

        result = self._base_result()
        result["elapsed"] = round(time.time() - t0, 3)

        if ocr_items is None and screenshot_path is None:
            result["error"] = "no input"
            return result

        if ocr_items is None:
            ocr_items, ocr_region, elapsed_ocr, ocr_error = self._run_ocr(screenshot_path)
            result["ocr_items"] = ocr_items
            result["ocr_region"] = ocr_region
            result["elapsed"] = round(time.time() - t0, 3)
            if ocr_error:
                result["error"] = ocr_error
                return result

        if not ocr_items:
            result["no_visible_message"] = True
            return result

        # 检测语音输入界面
        if self._is_wechat_voice_input_screen(ocr_items):
            result["voice"] = True
            return result

        # 布局分析
        layout = self._layout(ocr_items, window_info)
        result["layout"] = layout

        # 提取联系人
        contact = self._extract_contact_name(ocr_items, layout)
        result["contact"] = contact

        if not contact:
            result["no_visible_message"] = True
            return result

        # 系统联系人检测
        if self._is_system_contact(contact):
            result["system"] = True
            result["system_notice"] = True
            return result

        # 消息候选
        messages = self._message_candidates(ocr_items, layout)
        result["messages"] = messages

        # 语音消息检测
        if self._voice_status_enabled:
            voice_messages = self._voice_candidates(messages)
            result["voice_messages"] = voice_messages
            if voice_messages:
                result["voice"] = True
                latest_voice = self._latest_voice_message(messages)
                result["voice_message"] = latest_voice is not None

        # 客户消息提取
        boundary_y = self._latest_chat_time_separator_y(ocr_items)
        result["boundary_y"] = boundary_y
        customer_messages = self._customer_turn_messages(messages, boundary_y)
        result["customer_turn_messages"] = customer_messages

        # 消息稳定性
        stability = self._message_stability(messages)
        result["ocr_stability"] = stability

        if customer_messages:
            latest = customer_messages[-1]
            result["content"] = latest.text if hasattr(latest, 'text') else str(latest)
            result["msg_type"] = "customer_message"
            result["intent"] = self._classify_customer_text(result["content"])
            result["latest_group_sender"] = latest.group_sender_name if hasattr(latest, 'group_sender_name') else ""
        else:
            result["no_visible_message"] = True

        # 自身最新消息
        self_msgs = [m for m in messages if hasattr(m, 'side') and m.side == "right"]
        if self_msgs:
            result["self_latest_message"] = self_msgs[-1].text if hasattr(self_msgs[-1], 'text') else str(self_msgs[-1])

        # 回复草稿
        result["reply_draft"] = self._build_reply_draft(result)

        return result

    # =================================================================
    # 基础方法
    # =================================================================

    def _base_result(self) -> Dict[str, Any]:
        return {
            "contact": "",
            "msg_type": "unknown",
            "content": "",
            "intent": "unknown",
            "decision": "no_reply",
            "reason": "",
            "ocr_items": [],
            "messages": [],
            "voice_messages": [],
            "elapsed": 0.0,
            "ocr_region": None,
            "text_score": 0.0,
            "fallback_used": False,
            "fallback_checked": False,
            "fallback_error": "",
            "notice": False,
            "system_notice": False,
            "voice": False,
            "voice_message": False,
            "no_visible_message": False,
            "self_latest_message": "",
            "latest_top": 0,
            "boundary_y": 0,
            "customer_turn_messages": [],
            "ocr_stability": 0,
            "latest_group_sender": "",
            "reply_draft": "",
            "error": "",
        }

    def _run_ocr(self, screenshot_path: str) -> Tuple[list, Optional[Tuple], float, str]:
        """运行 OCR，返回 (items, region, elapsed, error)。"""
        import time
        t0 = time.time()
        try:
            # OCR 引擎走进程级单例池：原先每次调用都 `OCREngine()` + initialize()
            # 新建一份，等于每次重载一遍 RapidOCR 模型（约 0.2s、32MB 常驻，
            # 且反复分配/回收 onnxruntime 会话）。见 src/ocr/ocr_pool.py。
            from ..ocr.ocr_pool import get_text_ocr
            engine = get_text_ocr()
            if engine is None:
                return [], None, round(time.time() - t0, 3), "ocr_engine_unavailable"
            img = None
            if screenshot_path:
                import cv2
                img = cv2.imread(screenshot_path)
            if img is None:
                return [], None, round(time.time() - t0, 3), "failed to read image"

            # 区域 OCR
            if self._region_filter_enabled:
                region = self._chat_region_rect(img)
                if region:
                    items = engine.recognize_region(img, region)
                    should_fallback = self._should_fallback_full_ocr(items, region)
                    if should_fallback and self._region_fallback_enabled:
                        full_items = engine.recognize(img)
                        if self._full_ocr_is_better(items, region, full_items):
                            return self._ocr_items_to_dicts(full_items), None, round(time.time() - t0, 3), ""
                    return self._ocr_items_to_dicts(items), region, round(time.time() - t0, 3), ""

            items = engine.recognize(img)
            return self._ocr_items_to_dicts(items), None, round(time.time() - t0, 3), ""
        except Exception as e:
            return [], None, round(time.time() - t0, 3), str(e)

    def _ocr_items_to_dicts(self, items) -> list:
        """将 OCR 结果转为统一 dict 列表。"""
        if not items:
            return []
        result = []
        if OCRResult is not None and isinstance(items, OCRResult):
            for i in range(len(items.texts)):
                result.append({
                    "text": items.texts[i],
                    "score": float(items.scores[i]) if i < len(items.scores) else 0.5,
                    "box": items.boxes[i].tolist() if i < len(items.boxes) and len(items.boxes) > 0 else [],
                })
        elif isinstance(items, list):
            for item in items:
                d = _line_to_dict(item)
                if d["text"]:
                    result.append(d)
        return result

    def _chat_region_rect(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """计算聊天区域 OCR 范围。"""
        h, w = img.shape[:2]
        left = self._ocr_region_left_px or int(w * 0.28)
        right = w - (self._ocr_region_right_margin_px or 0)
        top = self._ocr_header_bottom_px or 0
        bottom = self._ocr_input_top_px or int(h * self._ocr_input_top_ratio)
        if left >= right or top >= bottom:
            return None
        return (left, top, right - left, bottom - top)

    def _should_fallback_full_ocr(self, items: list, region: Optional[Tuple]) -> bool:
        """判断是否需要回退到全屏 OCR。"""
        if not items or not region:
            return True
        if len(items) < 3:
            return True
        return False

    def _full_ocr_is_better(self, region_items: list, region: Optional[Tuple],
                            full_items: list) -> bool:
        """判断全屏 OCR 是否比区域 OCR 更好。"""
        if not full_items:
            return False
        if len(full_items) > len(region_items) * 1.3:
            return True
        return False

    def _is_wechat_voice_input_screen(self, items: list) -> bool:
        """检测是否为微信语音输入界面。"""
        texts = " ".join(item.get("text", "") if isinstance(item, dict) else str(item)
                         for item in items)
        return "按住说话" in texts or "语音输入" in texts

    # =================================================================
    # 布局分析
    # =================================================================

    def _layout(self, items: list, window_info: Optional[dict] = None) -> dict:
        """分析微信窗口布局，返回布局参数。

        Returns: {chat_left, chat_right, chat_mid, header_bottom, input_top,
                  send_buttons, send_tops}
        """
        result = {
            "chat_left": self._ocr_chat_left_px or 0,
            "chat_right": 0,
            "chat_mid": 0,
            "header_bottom": self._ocr_header_bottom_px or 0,
            "input_top": 0,
            "send_buttons": [],
            "send_tops": [],
        }

        if window_info:
            w = window_info.get("width", 0)
            h = window_info.get("height", 0)
        else:
            w = h = 0

        if not items:
            return result

        # 从 items 推断布局
        xs = []
        for item in items:
            d = item if isinstance(item, dict) else _line_to_dict(item)
            if d.get("x_min", 0) > 0:
                xs.append(d["x_min"])
            if d.get("x_max", 0) > 0:
                xs.append(d["x_max"])

        if xs:
            result["chat_right"] = max(xs)
            result["chat_mid"] = (result["chat_left"] + result["chat_right"]) / 2

        if w > 0:
            result["chat_left"] = result["chat_left"] or int(w * 0.28)
            result["input_top"] = self._ocr_input_top_px or int(h * self._ocr_input_top_ratio)

        # 查找发送按钮
        for item in items:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text.strip() == "Send" or text.strip() == "发送":
                cy = item.get("y_center", 0) if isinstance(item, dict) else (
                    (item.get("box", [[0, 0]])[0][1] + item.get("box", [[0, 0]])[2][1]) / 2
                    if isinstance(item, dict) and "box" in item else 0)
                result["send_tops"].append(cy)

        return result

    # =================================================================
    # 联系人提取（对齐原版 _extract_contact_name + _stabilize_contact_title）
    # =================================================================

    def _extract_contact_name(self, items: list, layout: Optional[dict] = None) -> str:
        """从 OCR 结果中提取联系人名称。"""
        chat_left = (layout or {}).get("chat_left", 0)

        # 在左侧区域中按 y 排序，取顶部最大的文本作为标题候选
        title_items = []
        for item in items:
            d = item if isinstance(item, dict) else _line_to_dict(item)
            if chat_left > 0 and d.get("x_max", 0) > chat_left:
                continue
            if self._looks_like_contact_title(d):
                title_items.append(d)

        if not title_items:
            return ""

        # 按 y 排序，取最靠前的
        title_items.sort(key=lambda d: d.get("y_center", 0))
        raw_title = title_items[0].get("text", "").strip()

        # 去除 "微信" 前缀
        cleaned = self._clean_text(raw_title)
        if cleaned.lower().startswith("wechat"):
            cleaned = cleaned[7:].strip()

        if not cleaned:
            return ""

        # 多帧稳定
        if self._ocr_contact_stabilize_enabled:
            return self._stabilize_contact_title(cleaned)

        return cleaned

    def _clean_text(self, text: str) -> str:
        """清理文本（去除多余空格/特殊字符）。"""
        return " ".join(text.strip().split())

    def _stabilize_contact_title(self, new_title: str) -> str:
        """多帧稳定联系人名称。"""
        t = (new_title or "").strip()
        key = self._contact_match_key(t)
        previous_key = self._contact_match_key(self._stable_contact_name)

        if key == previous_key:
            self._stable_contact_short_variant_count = 0
            return self._stable_contact_name

        # 检查是否为同一聊天的短变体
        if self._contact_titles_look_like_same_chat(t, self._stable_contact_name):
            self._stable_contact_short_variant_count += 1
            if self._stable_contact_short_variant_count >= self._contact_short_variant_accept_frames:
                self._stable_contact_name = t
                self._stable_contact_short_variant_count = 0
            return self._stable_contact_name

        # 新联系人
        if len(t) >= len(self._stable_contact_name):
            self._stable_contact_name = t
            self._stable_contact_short_variant_count = 0
            self._stable_contact_short_variant_key = self._contact_short_variant(t)
            return t

        return self._stable_contact_name or t

    def _contact_match_key(self, name: str) -> str:
        """生成联系人匹配键（归一化）。"""
        import re
        s = re.sub(r"[^\w\u4e00-\u9fff]+", "", (name or "").casefold().translate(
            str.maketrans("", "", " \t\n\r")))
        return s

    def _contact_short_variant(self, name: str) -> str:
        """提取联系人短变体（用于匹配）。"""
        return "".join((name or "").strip().split())[:12]

    @property
    def _contact_short_variant_accept_frames(self) -> int:
        return self._ocr_contact_short_variant_accept_frames

    def _contact_titles_look_like_same_chat(self, a: str, b: str) -> bool:
        """判断两个联系人标题是否属于同一个聊天。"""
        if not a or not b:
            return False
        key_a = self._contact_match_key(a)
        key_b = self._contact_match_key(b)
        if key_a == key_b:
            return True
        if key_a in key_b or key_b in key_a:
            # 长度比 > 50% 才认为是短变体
            ratio = min(len(key_a), len(key_b)) / max(len(key_a), len(key_b))
            return ratio > 0.5
        return False

    def _reset_contact_short_variant(self) -> None:
        """重置短变体追踪。"""
        self._stable_contact_short_variant_key = ""
        self._stable_contact_short_variant_count = 0

    def _group_name_from_contact(self, contact: str) -> str:
        """从群聊名称中提取群名（去除成员数后缀）。"""
        import re
        m = re.search(r"^(?P<name>.+?)[(]\s*\d+\s*[)]", contact or "")
        if m:
            return m.group("name").strip()
        return (contact or "").strip()

    def _is_system_contact(self, name: str) -> bool:
        """判断是否为系统联系人。"""
        return any(k in (name or "") for k in SYSTEM_CONTACT_KEYWORDS)

    # =================================================================
    # 消息候选（对齐原版 _message_candidates）
    # =================================================================

    def _message_candidates(self, items: list, layout: Optional[dict] = None) -> List[OcrMessage]:
        """从 OCR 条目中提取消息候选列表。"""
        chat_left = (layout or {}).get("chat_left", 0)
        chat_mid = (layout or {}).get("chat_mid", 0)

        # 过滤非聊天文本
        chat_items = []
        for item in items:
            d = item if isinstance(item, dict) else _line_to_dict(item)
            if self._is_chat_text(d, chat_left):
                chat_items.append(d)

        if not chat_items:
            return []

        # 按 y 排序
        chat_items.sort(key=lambda d: d.get("y_center", 0))

        # 聚合为消息
        messages: List[OcrMessage] = []
        for item in chat_items:
            if messages and self._same_message(item, messages[-1].parts[-1] if messages[-1].parts else {}):
                # 同一消息的多行
                if self._same_multiline_bubble(item, messages[-1].parts[-1] if messages[-1].parts else {}):
                    messages[-1].add(item)
                else:
                    # 新消息
                    msg = OcrMessage()
                    msg.add(item)
                    msg.side = self._guess_side(item, chat_mid)
                    messages.append(msg)
            else:
                msg = OcrMessage()
                msg.add(item)
                msg.side = self._guess_side(item, chat_mid)
                messages.append(msg)

        # 群聊发送者标签合并
        if self._group_sender_split_enabled:
            messages = self._merge_group_sender_labels(messages)

        return messages

    def _is_chat_text(self, item: dict, chat_left: int = 0) -> bool:
        """判断是否属于聊天文本区域。"""
        text = item.get("text", "").strip()
        if not text:
            return False
        cx = (item.get("x_min", 0) + item.get("x_max", 0)) / 2
        if chat_left > 0 and cx <= chat_left:
            return False
        if self._is_noise_text(text):
            return False
        return True

    def _is_noise_text(self, text: str) -> bool:
        """判断是否为噪音文本。"""
        if not text:
            return True
        if len(text) <= 1 and not text.isalnum():
            return True
        return False

    def _same_message(self, a: dict, b: dict) -> bool:
        """判断两个 OCR 条目是否属于同一条消息。"""
        if not a or not b:
            return False
        cy_a = a.get("y_center", 0)
        cy_b = b.get("y_center", 0)
        vertical_gap = abs(cy_a - cy_b)
        if vertical_gap > 30:
            return False
        # 同一消息的文本行通常在相近的 x 范围
        cx_a = (a.get("x_min", 0) + a.get("x_max", 0)) / 2
        cx_b = (b.get("x_min", 0) + b.get("x_max", 0)) / 2
        if abs(cx_a - cx_b) > 80:
            return False
        return True

    def _guess_side(self, item: dict, chat_mid: float = 0) -> str:
        """根据 x 位置推测消息方向。"""
        cx = (item.get("x_min", 0) + item.get("x_max", 0)) / 2
        if chat_mid > 0:
            if cx > chat_mid:
                return "right"
            return "left"
        return "unknown"

    # =================================================================
    # 群聊发送者标签合并（对齐原版 _merge_group_sender_labels）
    # =================================================================

    def _merge_group_sender_labels(self, messages: List[OcrMessage]) -> List[OcrMessage]:
        """合并群聊发送者标签到消息体。"""
        merged: List[OcrMessage] = []
        skip_next = False
        for i, msg in enumerate(messages):
            if skip_next:
                skip_next = False
                continue
            if i + 1 < len(messages):
                look_ahead = messages[i + 1]
                if self._looks_like_group_sender_to_body(msg, look_ahead):
                    # 合并发送者标签到消息体
                    merged_msg = look_ahead
                    merged_msg.group_sender_name = msg.text
                    merged_msg.sender = msg.text
                    merged.append(merged_msg)
                    skip_next = True
                    continue
            merged.append(msg)
        return merged

    def _looks_like_group_sender_to_body(self, label_msg: OcrMessage, body_msg: OcrMessage) -> bool:
        """判断 label_msg 是否为 body_msg 的群聊发送者标签。"""
        label_text = label_msg.text
        body_text = body_msg.text
        if not label_text or not body_text:
            return False
        if not self._looks_like_group_sender_text(label_text):
            return False
        if not self._looks_like_group_body_text(body_text):
            return False
        if not self._group_sender_body_aligned(label_msg, body_msg):
            return False
        return True

    def _looks_like_group_sender_text(self, text: str) -> bool:
        """判断是否像群聊发送者名称。"""
        t = text.strip()
        if len(t) > 15 or len(t) < 1:
            return False
        if any(c in t for c in "。，！？；：""''（）【】"):
            return False
        if self._is_voice_duration(t):
            return False
        return t.isprintable()

    def _looks_like_group_body_text(self, text: str) -> bool:
        """判断是否像群聊消息正文。"""
        return len(text.strip()) > 1

    def _group_sender_body_aligned(self, label: OcrMessage, body: OcrMessage) -> bool:
        """判断发送者标签与消息体是否对齐。"""
        label_left = label.parts[0].get("x_min", 0) if label.parts else 0
        body_left = body.parts[0].get("x_min", 0) if body.parts else 0
        return abs(label_left - body_left) < 30

    # =================================================================
    # 语音消息检测（对齐原版 _voice_candidates）
    # =================================================================

    def _voice_candidates(self, messages: List[OcrMessage]) -> List[OcrMessage]:
        """检测语音消息。"""
        return [m for m in messages if self._is_voice_chat_item(m)]

    def _is_voice_chat_item(self, msg: OcrMessage) -> bool:
        """判断是否为语音聊天条目。"""
        text = msg.text
        if not text:
            return False
        if "语音" in text or "Voice" in text:
            return True
        if self._is_voice_duration(text):
            return True
        return False

    def _is_voice_duration(self, text: str) -> bool:
        """判断是否为语音时长标记。"""
        import re
        return bool(re.search(r"^\d+['\u2018]?\d*[\"\u201d]?$", text.strip()))

    def _latest_voice_message(self, messages: List[OcrMessage]) -> Optional[OcrMessage]:
        """获取最新语音消息。"""
        voices = [m for m in messages if self._is_voice_chat_item(m)]
        return voices[-1] if voices else None

    def _strip_voice_prefix(self, text: str) -> str:
        """去除语音前缀。"""
        return text.strip().removeprefix("[语音]").removeprefix("[语音通话]").strip()

    # =================================================================
    # 客户消息提取（对齐原版 _customer_turn_messages）
    # =================================================================

    def _latest_chat_time_separator_y(self, items: list) -> float:
        """查找聊天时间分隔线 y 坐标。"""
        import re
        for item in items:
            d = item if isinstance(item, dict) else _line_to_dict(item)
            text = d.get("text", "").strip()
            if re.search(r"\d{1,2}:\d{2}", text) and len(text) <= 8:
                return d.get("y_center", 0)
        return 0

    def _customer_turn_messages(self, messages: List[OcrMessage],
                                 boundary_y: float = 0) -> List[OcrMessage]:
        """提取客户轮次消息（boundary_y 之后的其他侧消息）。"""
        active_messages = messages
        if boundary_y > 0:
            active_messages = [m for m in messages
                               if m.parts and m.parts[0].get("y_center", 0) > boundary_y]

        customer_msgs = []
        for msg in active_messages:
            if msg.side == "left":
                # 去除语音前缀
                cleaned_parts = []
                for p in msg.parts:
                    raw = p.get("text", "")
                    stripped = self._strip_voice_prefix(raw)
                    if stripped:
                        p_copy = dict(p)
                        p_copy["text"] = stripped
                        cleaned_parts.append(p_copy)
                if cleaned_parts:
                    msg.parts = cleaned_parts
                customer_msgs.append(msg)

        return customer_msgs

    # =================================================================
    # 消息稳定性（对齐原版 _message_stability）
    # =================================================================

    def _message_stability(self, messages: List[OcrMessage]) -> int:
        """追踪消息稳定性，返回稳定帧数。"""
        if not self._message_stabilize_enabled:
            return 1

        sig = self._message_stability_signature(messages)
        if sig == self._stable_message_signature:
            self._stable_message_frames += 1
        else:
            self._stable_message_signature = sig
            self._stable_message_frames = 1

        return self._stable_message_frames

    def _reset_message_stability(self) -> None:
        self._stable_message_signature = ""
        self._stable_message_frames = 0

    def _message_stability_signature(self, messages: List[OcrMessage]) -> str:
        """生成消息稳定性签名。"""
        if not messages:
            return ""
        contact_key = ""
        text_key = "|".join(m.text[:20] for m in messages[-5:])
        top_bucket = 0
        if messages and messages[0].parts:
            top_bucket = int(messages[0].parts[0].get("y_center", 0) // 50)
        bottom_bucket = 0
        if messages and messages[-1].parts:
            bottom_bucket = int(messages[-1].parts[-1].get("y_center", 0) // 50)
        return f"{contact_key}|{text_key}|{top_bucket}|{bottom_bucket}"

    @property
    def _message_stable_required_frames(self) -> int:
        return self._ocr_reply_stable_frames

    # =================================================================
    # 消息分类（对齐原版 _classify_customer_text）
    # =================================================================

    def _classify_customer_text(self, text: str) -> str:
        """对客户消息文本进行分类。"""
        if not text:
            return "unknown"
        lower = text.strip().lower()

        if any(k in lower for k in BRAND_KEYWORDS):
            return "brand_question"
        if any(k in lower for k in ADDRESS_KEYWORDS):
            return "address_question"
        if any(k in lower for k in PRICE_KEYWORDS):
            return "price_question"
        if any(k in lower for k in PRODUCT_KEYWORDS):
            return "product_material_question"
        if any(k in lower for k in CATALOG_KEYWORDS):
            return "catalog_question"
        if any(k in lower for k in APPOINTMENT_KEYWORDS):
            return "appointment_question"
        if any(k in lower for k in TRAFFIC_KEYWORDS):
            return "traffic_question"
        if any(k in lower for k in AVAILABILITY_KEYWORDS):
            return "product_availability_question"
        if any(k in lower for k in MATERIAL_KEYWORDS):
            return "product_material_question"
        if any(k in lower for k in HUMAN_SERVICE_KEYWORDS):
            return "human_service_request"
        if any(k in lower for k in COMPLAINT_KEYWORDS):
            return "complaint_or_after_sales"
        if any(k in lower for k in REFUSE_PHONE_KEYWORDS):
            return "refuse_full_phone"
        if any(k in lower for k in AD_KEYWORDS):
            return "ad_or_promotion"
        if any(k in lower for k in PAYMENT_KEYWORDS):
            return "payment_notice"

        return "customer_message"

    # =================================================================
    # 兼容旧接口：文本后处理流水线
    # =================================================================

    def normalize(self, source: Union[OCRResult, List[dict], list]) -> List[dict]:
        """把 OCRResult / line dict / 原始条目统一转成规范化 line dict 列表。"""
        if source is None:
            return []

        if OCRResult is not None and isinstance(source, OCRResult):
            items = []
            n = len(source.texts)
            for i in range(n):
                text = source.texts[i]
                box = np.array(source.boxes[i]) if i < len(source.boxes) and source.boxes.size else None
                score = float(source.scores[i]) if i < len(source.scores) else 0.5
                items.append([box, text, score])
            return [d for d in (self._parse_item(it) for it in items) if d["text"]]

        if isinstance(source, list):
            return [d for d in (self._parse_item(it) for it in source) if d["text"]]

        return []

    def _parse_item(self, item) -> Optional[dict]:
        try:
            d = _line_to_dict(item)
            return d if d["text"] else None
        except Exception:
            return None

    def process(self, source: Union[OCRResult, List[dict], list],
                sort_by_y: bool = True) -> List[dict]:
        """完整后处理：去重 -> 排序 -> 合并同行。"""
        lines = self.normalize(source)
        lines = self.dedupe(lines)
        if sort_by_y:
            lines = self.sort_by_y(lines)
        lines = self.merge_same_row(lines)
        return lines

    def dedupe(self, lines: List[dict]) -> List[dict]:
        """去重：y 重叠且在垂直方向高度重合的近似文本只保留置信度最高的一条。"""
        if not lines:
            return []
        ordered = self.sort_by_y(lines)
        kept: list = []
        for line in ordered:
            duplicate = False
            for k in kept:
                if self._is_duplicate_of(line, k):
                    duplicate = True
                    if line["score"] > k["score"]:
                        kept[kept.index(k)] = line
                    break
            if not duplicate:
                kept.append(line)
        return kept

    def sort_by_y(self, lines: List[dict]) -> List[dict]:
        return sorted(lines, key=lambda l: l["y_center"])

    def merge_same_row(self, lines: List[dict]) -> List[dict]:
        """合并同行：y 中心接近且 x 区间可拼接的碎片合并为一行完整文本。"""
        if not lines:
            return []
        rows: List[List[dict]] = []
        for line in self.sort_by_y(lines):
            placed = False
            for row in rows:
                if self._same_row(row[0], line):
                    row.append(line)
                    placed = True
                    break
            if not placed:
                rows.append([line])

        merged = []
        for row in rows:
            row = sorted(row, key=lambda l: l["x_min"])
            text_parts = [r["text"] for r in row]
            full_text = "".join(text_parts)
            y_center = float(np.mean([r["y_center"] for r in row]))
            x_min = float(min(r["x_min"] for r in row))
            x_max = float(max(r["x_max"] for r in row))
            score = float(np.mean([r["score"] for r in row]))
            merged.append({
                "text": full_text,
                "score": score,
                "box": row[0]["box"],
                "y_center": y_center,
                "x_min": x_min,
                "x_max": x_max,
            })
        return merged

    def _is_duplicate_of(self, a: dict, b: dict) -> bool:
        dy = abs(a["y_center"] - b["y_center"])
        if dy > 3.0:
            return False
        if a["text"] == b["text"]:
            return True
        from .ocr_text_similarity import is_similar
        return is_similar(a["text"], b["text"], threshold=self.dedupe_similarity)

    def _same_row(self, a: dict, b: dict) -> bool:
        height_a = self._row_height(a)
        height_b = self._row_height(b)
        tol = max(height_a, height_b) * self.row_y_tolerance_ratio
        return abs(a["y_center"] - b["y_center"]) <= tol

    @staticmethod
    def _row_height(line: dict) -> float:
        box = line.get("box")
        if box and len(box) >= 4:
            box_arr = np.array(box)
            return float(np.max(box_arr[:, 1]) - np.min(box_arr[:, 1]))
        return 20.0

    # =================================================================
    # 关键词匹配（对齐原版 _contains_any）
    # =================================================================

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        """检查文本是否包含任意关键词。"""
        return any(k in (text or "") for k in keywords)

    # =================================================================
    # 广告检测（对齐原版 _looks_like_ad）
    # =================================================================

    def _looks_like_ad(self, text: str) -> bool:
        """检测是否为广告/营销消息。

        对齐原版：命中>=2个AD关键词 或 命中1个+带链接词。
        """
        if not text:
            return False
        low = text.lower()
        ad_hits = sum(1 for k in AD_KEYWORDS if k in low)
        if ad_hits >= 2:
            return True
        link_words = ("http", "www.", "扫码", "点击", "领取", "立即", "小程序")
        if ad_hits >= 1 and self._contains_any(text, link_words):
            return True
        return False

    # =================================================================
    # 支付通知检测（对齐原版 _looks_like_payment_notice）
    # =================================================================

    def _looks_like_payment_notice(self, text: str) -> bool:
        """检测是否为支付/收款通知。

        对齐原版：通知语 + 金额标识 同时命中才拦截。
        """
        import re
        notice_terms = ("赞赏到账通知", "收款金额", "到账时间", "二维码到账", "已收款", "微信转账")
        has_notice = any(t in (text or "") for t in notice_terms)
        has_amount = bool(re.search(r"[¥￥]\s*\d|\d+(?:\.\d+)?\s*元", text or ""))
        return has_notice and has_amount

    # =================================================================
    # 语音噪音检测（对齐原版 _is_voice_turn_noise）
    # =================================================================

    def _is_voice_turn_noise(self, msg: OcrMessage) -> bool:
        """判断是否为语音轮次噪音（如 "[语音]" 标记本身）。"""
        if not msg or not msg.parts:
            return False
        text = "".join(p.get("text", "") for p in msg.parts).strip()
        if not text:
            return True
        if text.startswith("[语音") and len(text) <= 8:
            return True
        return False

    # =================================================================
    # 区域过滤属性（对齐原版 _region_filter_enabled / _region_fallback_enabled）
    # =================================================================

    @property
    def _region_filter_enabled(self) -> bool:
        return self._region_filter_enabled_val

    @property
    def _region_fallback_enabled(self) -> bool:
        return self._region_fallback_enabled_val

    # =================================================================
    # OCR 文本得分（对齐原版 _ocr_text_score）
    # =================================================================

    def _ocr_text_score(self, items: list) -> float:
        """计算 OCR 文本的整体置信度得分。"""
        if not items:
            return 0.0
        scores = []
        for item in items:
            d = item if isinstance(item, dict) else _line_to_dict(item)
            s = d.get("score", 0.5)
            if s is not None:
                scores.append(float(s))
        if not scores:
            return 0.0
        return float(np.mean(scores))

    # =================================================================
    # 输入框提示（对齐原版 _input_box_hint_default）
    # =================================================================

    def _input_box_hint_default(self) -> dict:
        """返回默认输入框点击位置提示。"""
        return {
            "x_ratio": 0.68,
            "y_ratio": 0.90,
        }

    # =================================================================
    # 聊天时间分隔线（对齐原版 _is_chat_time_separator / _latest_chat_time_separator_y）
    # =================================================================

    def _is_chat_time_separator(self, text: str) -> bool:
        """判断是否为聊天时间分隔线文本。"""
        import re
        t = (text or "").strip()
        if not t:
            return False
        return bool(re.fullmatch(r"\d{1,2}:\d{2}", t)) and len(t) <= 5

    # =================================================================
    # 群聊发送者标签 → 消息体合并（对齐原版 _looks_like_group_sender_message_to_body）
    # =================================================================

    def _looks_like_group_sender_message_to_body(self, label_msg: OcrMessage, body_msg: OcrMessage) -> bool:
        """判断 label_msg 是否为 body_msg 的群聊发送者标签。

        对齐原版：检查发送者标签与消息体是否对齐 + 文本特征。
        """
        label_text = label_msg.text if hasattr(label_msg, 'text') else str(label_msg)
        body_text = body_msg.text if hasattr(body_msg, 'text') else str(body_msg)

        if not label_text or not body_text:
            return False
        if not self._looks_like_group_sender_text(label_text):
            return False
        if not self._looks_like_group_body_text(body_text):
            return False
        if not self._group_sender_body_aligned(label_msg, body_msg):
            return False
        return True

    # =================================================================
    # 改进的同气泡判定（对齐原版 _same_multiline_bubble 更详细的实现）
    # =================================================================

    def _same_multiline_bubble(self, a: dict, b: dict) -> bool:
        """判断两个条目是否在同一气泡内（对齐原版更详细的实现）。"""
        if not a or not b:
            return False
        # 垂直间隙
        vertical_gap = abs(a.get("y_center", 0) - b.get("y_center", 0))
        if vertical_gap > 18:
            return False
        if vertical_gap < -4:
            return False
        # 水平偏移
        side_a = "left" if a.get("x_max", 0) < 0 else "right"
        side_b = "left" if b.get("x_max", 0) < 0 else "right"
        if side_a != side_b:
            return False
        x_diff = abs((a.get("x_min", 0) + a.get("x_max", 0)) / 2 -
                     (b.get("x_min", 0) + b.get("x_max", 0)) / 2)
        if x_diff > 45:
            return False
        return True

    # =================================================================
    # 消息稳定性签名匹配（对齐原版 _message_stability_signatures_match）
    # =================================================================

    def _message_stability_signatures_match(self, a: str, b: str) -> bool:
        """判断两个签名是否匹配（对齐原版，使用 looks_like_same_ocr_text）。"""
        if not a or not b:
            return False
        parts_a = a.split("|")
        parts_b = b.split("|")
        if len(parts_a) < 2 or len(parts_b) < 2:
            return False

        from .ocr_text_similarity import looks_like_same_ocr_text
        return looks_like_same_ocr_text(parts_a[1], parts_b[1])

    # =================================================================
    # 改进的联系人标题检测（对齐原版 _looks_like_contact_title）
    # =================================================================

    def _looks_like_contact_title(self, item: dict) -> bool:
        """判断 OCR 条目是否像联系人标题（对齐原版更详细的实现）。"""
        text = item.get("text", "").strip()
        if not text or len(text) > 32:
            return False
        # 标题通常在左侧，y 在顶部
        x_min = item.get("x_min", 0)
        if x_min > 300:
            return False
        # 排除一些明显不是标题的文本
        if self._is_chat_time_separator(text):
            return False
        if self._is_voice_duration(text):
            return False
        if text in ("微信", "WeChat", "搜索", "聊天", "通讯录"):
            return False
        return True

    # =================================================================
    # 回复生成方法（对齐原版 reply 系列方法）
    # =================================================================

    def _address_reply(self, text: str) -> str:
        """地址相关问题回复。"""
        return "地址"

    def _traffic_reply(self, text: str) -> str:
        """交通相关问题回复。"""
        return "交通"

    def _brand_reply(self, text: str) -> str:
        """品牌相关问题回复。"""
        return "品牌"

    def _generic_price_reply(self, text: str) -> str:
        """价格相关问题回复。"""
        return "价格"

    def _product_material_reply(self, text: str) -> str:
        """产品材料相关问题回复。"""
        return "产品材料"

    def _product_size_option_reply(self, text: str) -> str:
        """产品规格选项问题回复。"""
        return "产品规格"

    def _product_price_reply(self, text: str) -> str:
        """产品价格问题回复。"""
        return "产品价格"

    def _product_availability_reply(self, text: str) -> str:
        """产品库存问题回复。"""
        return "产品库存"

    def _multi_direct_reply(self, text: str) -> str:
        """多问题直接回复。"""
        return "多问题"

    def _default_customer_reply(self, text: str) -> str:
        """默认客户回复。"""
        return "客户咨询"

    # =================================================================
    # 改进的回复草稿构建（对齐原版 _build_reply_draft）
    # =================================================================

    def _build_reply_draft(self, result: dict) -> str:
        """根据解析结果构建回复草稿（对齐原版更完整的实现）。"""
        intent = result.get("intent", "")
        content = result.get("content", "")

        if not content:
            return ""

        # 根据意图类别生成不同的回复草稿
        if intent == "address_question":
            return self._address_reply(content)
        elif intent == "traffic_question":
            return self._traffic_reply(content)
        elif intent == "brand_question":
            return self._brand_reply(content)
        elif intent == "price_question":
            return self._generic_price_reply(content)
        elif intent == "product_material_question":
            return self._product_material_reply(content)
        elif intent == "product_size_option_question":
            return self._product_size_option_reply(content)
        elif intent == "product_price_question":
            return self._product_price_reply(content)
        elif intent == "product_availability_question":
            return self._product_availability_reply(content)
        elif intent == "human_service_request":
            return "人工客服"
        elif intent == "complaint_or_after_sales":
            return "售后"
        elif intent == "refuse_full_phone":
            return "弱留资"
        elif intent == "payment_notice":
            return ""
        elif intent == "ad_or_promotion":
            return ""

        return f"[{intent}] {content[:80]}"


# =====================================================================
# 功能函数（兼容旧接口）
# =====================================================================

def classify_customer_intent(text: str) -> str:
    """根据客户消息文本推断意图类别。"""
    parser = WeChatOCRParser()
    return parser._classify_customer_text(text)


def is_system_contact(name: str) -> bool:
    """判断是否为系统联系人（微信支付、微信团队等）。"""
    return any(k in (name or "") for k in SYSTEM_CONTACT_KEYWORDS)


def is_system_message(text: str) -> bool:
    """判断是否为系统消息（撤回、加入群聊等）。"""
    return any(k in (text or "") for k in SYSTEM_MESSAGE_KEYWORDS)


def looks_like_ad(text: str) -> bool:
    """判断是否像广告消息。"""
    return any(k in (text or "") for k in AD_KEYWORDS)


def looks_like_payment(text: str) -> bool:
    """判断是否像支付/转账消息。"""
    return any(k in (text or "") for k in PAYMENT_KEYWORDS)


def is_voice_duration(text: str) -> bool:
    """判断文本是否为语音时长标记。"""
    import re
    return bool(re.search(r"^\d+['\u2018]?\d*[\"\u201d]?$", (text or "").strip()))


def strip_voice_prefix(text: str) -> str:
    """去除语音消息前缀标记。"""
    return (text or "").strip().removeprefix("[语音]").removeprefix("[语音通话]").strip()


def looks_like_group_sender_text(text: str) -> bool:
    """判断文本是否像群聊发送者名称。"""
    t = (text or "").strip()
    if len(t) > 10 or len(t) < 1:
        return False
    if any(c in t for c in "。，！？；：""''（）【】"):
        return False
    return t.isprintable()