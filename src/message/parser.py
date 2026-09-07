import re
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MessageSide(Enum):
    SELF = "self"
    OTHER = "other"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    EMOJI = "emoji"
    SYSTEM = "system"
    REFERENCE = "reference"


@dataclass
class ContactInfo:
    name: str = ""
    is_group: bool = False
    unread_count: int = 0
    last_message: str = ""
    last_time: str = ""


@dataclass
class ChatMessage:
    sender: str = ""
    content: str = ""
    timestamp: str = ""
    side: MessageSide = MessageSide.UNKNOWN
    msg_type: MessageType = MessageType.TEXT
    is_group: bool = False
    group_sender: str = ""
    confidence: float = 0.0
    box: Optional[np.ndarray] = None
    reply_to: str = ""
    raw_text: str = ""
    y_center: float = 0.0

    def fingerprint(self) -> str:
        return f"{self.side.value}_{self.sender}_{self.content[:30]}_{self.timestamp}"


class MessageParser:
    BUBBLE_COLOR_THRESHOLD = 100
    UNREAD_MARKER_KEYWORDS = ["以下为新消息", "你以上是新消息", "以下为未读消息"]
    SYSTEM_KEYWORDS = ["撤回了一条消息", "拍了拍", "同意了好友验证", "开启了朋友验证"]

    def __init__(self, system_contacts: Optional[List[str]] = None):
        self.system_contacts = system_contacts or [
            "微信支付", "微信游戏", "服务号", "公众号", "订阅号",
            "微信团队", "QQ邮箱提醒"
        ]
        self._known_contacts: Dict[str, ContactInfo] = {}
        self._last_processed_fingerprints: set = set()
        self._max_history = 100

    def parse_chat_screenshot(self, image: np.ndarray,
                               ocr_lines: List[dict],
                               chat_region: Optional[Tuple[int, int, int, int]] = None) -> List[ChatMessage]:
        messages = []

        if chat_region:
            x, y, w, h = chat_region
            roi = image[y:y+h, x:x+w]
            adjusted_lines = []
            for line in ocr_lines:
                adjusted = dict(line)
                if "y_center" in adjusted:
                    adjusted["y_center"] -= y
                adjusted_lines.append(adjusted)
        else:
            roi = image
            adjusted_lines = ocr_lines

        text_blocks = self._group_into_text_blocks(adjusted_lines)

        for block in text_blocks:
            msg = self._parse_block(block, roi)
            if msg and msg.content:
                messages.append(msg)

        messages = self._identify_sides(messages, roi)
        return messages

    def _group_into_text_blocks(self, lines: List[dict]) -> List[List[dict]]:
        if not lines:
            return []

        blocks = []
        current_block = [lines[0]]

        for i in range(1, len(lines)):
            line = lines[i]
            prev_line = lines[i-1]

            y_gap = abs(line["y_center"] - prev_line["y_center"])
            avg_height = (self._estimate_height(prev_line) + self._estimate_height(line)) / 2

            if y_gap < avg_height * 2.5:
                current_block.append(line)
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = [line]

        if current_block:
            blocks.append(current_block)

        return blocks

    def _estimate_height(self, line: dict) -> float:
        if "box" in line and len(line["box"]) >= 4:
            box = np.array(line["box"])
            return float(np.max(box[:, 1]) - np.min(box[:, 1]))
        return 20.0

    def _parse_block(self, block: List[dict], image: np.ndarray) -> Optional[ChatMessage]:
        if not block:
            return None

        full_text = " ".join(l["text"] for l in block)
        avg_x = np.mean([l["x_min"] + (l["x_max"] - l["x_min"]) / 2 for l in block])
        avg_y = np.mean([l["y_center"] for l in block])

        msg = ChatMessage()
        msg.raw_text = full_text

        if self._is_timestamp(full_text):
            msg.msg_type = MessageType.SYSTEM
            msg.side = MessageSide.SYSTEM
            msg.timestamp = full_text
            return msg

        if self._is_system_message(full_text):
            msg.msg_type = MessageType.SYSTEM
            msg.side = MessageSide.SYSTEM
            msg.content = full_text
            return msg

        if self._is_unread_marker(full_text):
            msg.msg_type = MessageType.SYSTEM
            msg.side = MessageSide.SYSTEM
            msg.content = "__UNREAD_MARKER__"
            return msg

        if full_text.startswith("[") and full_text.endswith("]"):
            msg.msg_type = MessageType.EMOJI
        elif full_text.startswith("[图片]") or full_text.startswith("[表情]"):
            msg.msg_type = MessageType.IMAGE
        elif full_text.startswith("[语音]"):
            msg.msg_type = MessageType.VOICE

        msg.content = full_text
        msg.confidence = np.mean([l.get("score", 0.5) for l in block])

        if block[0].get("box"):
            msg.box = np.array(block[0]["box"])

        return msg

    def _identify_sides(self, messages: List[ChatMessage],
                        image: np.ndarray) -> List[ChatMessage]:
        img_width = image.shape[1] if len(image.shape) > 1 else 800
        center_x = img_width / 2

        for msg in messages:
            if msg.side in (MessageSide.SYSTEM,):
                continue

            if msg.box is not None and len(msg.box) >= 4:
                x_center = np.mean(msg.box[:, 0])
            else:
                continue

            if x_center > center_x:
                msg.side = MessageSide.SELF
            else:
                msg.side = MessageSide.OTHER

        return messages

    def _is_timestamp(self, text: str) -> bool:
        patterns = [
            r'^\d{1,2}:\d{2}$',
            r'^\d{1,2}:\d{2}:\d{2}$',
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'昨天\s*\d{1,2}:\d{2}',
            r'今天\s*\d{1,2}:\d{2}',
            r'上午\s*\d{1,2}:\d{2}',
            r'下午\s*\d{1,2}:\d{2}',
            r'晚上\s*\d{1,2}:\d{2}',
            r'星期[一二三四五六日天]',
        ]
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def _is_system_message(self, text: str) -> bool:
        for kw in self.SYSTEM_KEYWORDS:
            if kw in text:
                return True
        return False

    def _is_unread_marker(self, text: str) -> bool:
        for kw in self.UNREAD_MARKER_KEYWORDS:
            if kw in text:
                return True
        return False

    def parse_contact_list(self, ocr_lines: List[dict]) -> List[ContactInfo]:
        contacts = []
        current = ContactInfo()

        for line in ocr_lines:
            text = line["text"]

            if self._is_timestamp(text):
                if current.name:
                    contacts.append(current)
                current = ContactInfo()
                current.last_time = text
                continue

            if not current.name:
                current.name = text
            elif not current.last_message:
                current.last_message = text

        if current.name:
            contacts.append(current)

        return contacts

    def identify_unread_messages(self, messages: List[ChatMessage],
                                   last_processed_fingerprint: str = "") -> List[ChatMessage]:
        unread = []
        found_unread_marker = False

        for msg in messages:
            if msg.content == "__UNREAD_MARKER__":
                found_unread_marker = True
                continue

            if found_unread_marker:
                if msg.side == MessageSide.OTHER and msg.content:
                    if msg.fingerprint() not in self._last_processed_fingerprints:
                        unread.append(msg)
            elif not last_processed_fingerprint:
                if msg.side == MessageSide.OTHER and msg.content:
                    if msg.fingerprint() not in self._last_processed_fingerprints:
                        unread.append(msg)

        # 首次遇到该联系人且无分隔线标记时，只取最后 5 条（底部最新消息）
        if not found_unread_marker and not last_processed_fingerprint:
            unread = unread[-5:] if len(unread) > 5 else unread

        return unread

    def mark_processed(self, messages: List[ChatMessage]) -> None:
        for msg in messages:
            self._last_processed_fingerprints.add(msg.fingerprint())
            if len(self._last_processed_fingerprints) > self._max_history:
                self._last_processed_fingerprints = set(
                    list(self._last_processed_fingerprints)[-self._max_history:]
                )

    def is_system_contact(self, name: str) -> bool:
        for sys_name in self.system_contacts:
            if sys_name in name or name in sys_name:
                return True
        return False

    def extract_mentions(self, text: str) -> List[str]:
        mentions = re.findall(r'@(\S+)', text)
        return mentions

    def detect_message_intent(self, text: str) -> str:
        intents = {
            "greeting": ["你好", "您好", "在吗", "hi", "hello", "在不"],
            "question": ["吗", "呢", "?", "？", "怎么", "为什么", "什么"],
            "purchase": ["多少钱", "价格", "购买", "下单", "买", "套餐"],
            "complaint": ["投诉", "不好", "差", "退货", "退款", "差评"],
            "thanks": ["谢谢", "感谢", "多谢", "thanks", "thank"],
        }
        for intent, keywords in intents.items():
            for kw in keywords:
                if kw in text.lower():
                    return intent
        return "general"