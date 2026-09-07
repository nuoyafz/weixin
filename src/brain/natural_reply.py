"""自然回复润色：让 AI 回复更像真人、去除 Markdown、适配小程序场景、可选"你/您"迭代。"""
from __future__ import annotations

import random
import re
from typing import Any, List


class NaturalReplyPolisher:
    def __init__(self, settings: Any = None):
        self.settings = settings
        self.use_casual_you = bool(getattr(settings, "use_casual_you", True)) if settings else True

    def polish(self, text: str, analysis: dict = None) -> str:
        if not text:
            return text
        text = text.strip()
        # 去掉 markdown 标记
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = text.replace("```", "")
        # 去掉过度空行
        text = re.sub(r"\n{2,}", "\n", text)
        # "祝你""为您"等过于正式的敬语适度柔和（仅当开启 casual）
        return text.strip()

    def make_casual(self, text: str) -> str:
        if not text or self.use_casual_you:
            return text
        return text.replace("您", "你")

    def split_segments(self, text: str, max_chars: int = 64,
                       max_segments: int = 3) -> List[str]:
        """按原版自然回复分段规则拆分多条消息。"""
        if not text:
            return []
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        segments = []
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        for para in paragraphs:
            while len(para) > max_chars and len(segments) < max_segments:
                pos = self._find_split_point(para, max_chars)
                segments.append(para[:pos].strip())
                para = para[pos:].strip()
            if para:
                segments.append(para)
            if len(segments) >= max_segments:
                break
        return segments[:max_segments]

    @staticmethod
    def _find_split_point(text: str, max_pos: int) -> int:
        if len(text) <= max_pos:
            return len(text)
        for sep in ["。", "！", "？", ".", "!", "?", ",", "，"]:
            pos = text.rfind(sep, max_pos - 20, max_pos + 10)
            if pos > 0:
                return pos + 1
        return max_pos