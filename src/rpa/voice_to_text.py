"""VoiceToTextConverter - 微信语音转文字，对齐原版 app.rpa.voice_to_text v3.10。

核心流程：
1. convert_visible_voice() - 多轮循环转换（最多 8 条）
2. _convert_one_visible_voice() - 4 级按钮查找回退
3. 支持右键菜单转文字、语音时长检测、去重、虚拟屏偏移
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from ..config import settings as config


def _textbox_center(item) -> tuple:
    box = getattr(item, "box", None)
    if box is not None and hasattr(box, "size") and box.size >= 8:
        pts = box.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return 0.0, 0.0


class VoiceToTextConverter:
    """使用微信内置「转文字」按钮将语音消息转为文本。

    对齐原版 app.rpa.voice_to_text.VoiceToTextConverter v3.10。
    支持：
    - 多轮转换（最多 8 条语音）
    - 4 级按钮查找回退（可见按钮 → 匹配按钮 → 估算位置 → 右键菜单）
    - 语音时长 OCR 识别
    - 去重
    - 虚拟屏坐标偏移
    """

    def __init__(self, config=None, capture=None, logger=None):
        from ..capture.screen_capture import ScreenCapture
        from ..ocr.ocr_pool import get_vision_ocr

        self.config = config or {}
        self.capture = capture or ScreenCapture()
        # OCR 走进程级单例池：原先这里自建一份 OCREngine，一旦启用语音转文字
        # （voice_messages.auto_convert_to_text 打开）就会再多加载一份 RapidOCR
        # 模型（约 32MB + 独立 onnxruntime 线程池），见 src/ocr/ocr_pool.py。
        # 取不到时为 None，下面各处 self.ocr.run(...) 均已包在 try/except 内，
        # 会安全降级为空结果。
        self.ocr = get_vision_ocr()
        self.logger = logger or (lambda msg: None)

    # ---------------------------------------------------------------
    # 主入口：多轮语音转换
    # ---------------------------------------------------------------
    def convert_visible_voice(self, window_info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """转换当前聊天窗口中的可见语音消息。

        Args:
            window_info: 窗口信息字典，包含 rect, handle 等。

        Returns:
            dict with keys: ok, converted, voice_detected, reason,
                           before_image, after_image, converted_text,
                           converted_texts, attempts
        """
        cfg = self.config.get("voice_messages", {})
        if not cfg.get("auto_convert_to_text", True):
            return {"ok": False, "converted": False, "voice_detected": False,
                    "reason": "voice_auto_convert_disabled"}

        info = dict(window_info or {})
        rect = info.get("rect", {})
        hwnd = info.get("handle", 0) or info.get("hwnd", 0)
        if not hwnd or not rect:
            return {"ok": False, "converted": False, "voice_detected": False,
                    "reason": "wechat_window_missing"}

        max_items = min(cfg.get("max_voice_convert_per_cycle", 5), 8)
        wait_seconds = float(cfg.get("convert_wait_seconds", 2.0))

        converted_texts = []
        attempts = 0
        first_before_path = ""
        last_after_path = ""
        voice_detected = False

        for index in range(max_items):
            one = self._convert_one_visible_voice(rect, hwnd, wait_seconds, index)
            if not one.get("voice_detected", False):
                if index == 0:
                    first_before_path = one.get("before_image", "")
                break

            voice_detected = True
            attempts = index + 1
            if index == 0:
                first_before_path = one.get("before_image", "")

            text = str(one.get("converted_text", "")).strip()
            if text:
                converted_texts.append(text)
                last_after_path = one.get("after_image", "")
            else:
                last_after_path = one.get("after_image", one.get("before_image", ""))

            time.sleep(0.12)

        if converted_texts:
            merged = "\n".join(converted_texts)
            merged = self._dedupe_texts(merged)
            result = {
                "ok": True, "converted": True, "voice_detected": voice_detected,
                "reason": "voice_converted_by_wechat",
                "before_image": first_before_path, "after_image": last_after_path,
                "converted_text": merged, "converted_texts": converted_texts,
                "attempts": attempts,
            }
            self._log(f"voice_to_text converted_count={len(converted_texts)} text={merged[:120]}")
            return result

        self._log(f"voice_to_text not_converted reason={one.get('reason', '')} attempts={attempts}")
        return {
            "ok": one.get("ok", False), "converted": False,
            "voice_detected": voice_detected,
            "reason": one.get("reason", "voice_convert_button_not_found"),
            "before_image": first_before_path, "after_image": last_after_path,
            "converted_text": "", "converted_texts": converted_texts,
            "attempts": attempts,
        }

    # ---------------------------------------------------------------
    # 单条语音转换（4 级按钮查找回退）
    # ---------------------------------------------------------------
    def _convert_one_visible_voice(self, rect: dict[str, int], hwnd: int,
                                   wait_seconds: float, index: int) -> dict[str, Any]:
        """转换单条语音消息。

        4 级回退策略：
        1. visible_convert_button - 直接可见的「转文字」按钮
        2. matched_convert_button - 与语音时长匹配的按钮
        3. estimated_convert_button - 根据语音条估算按钮位置
        4. fallback_convert_button - 回退到任意按钮，右键菜单
        """
        before_path, items = self._capture_ocr(rect, hwnd, f"voice_before_convert_{index}")

        pair = self._find_next_voice_convert_pair(items, rect)
        click_method = ""
        click_point = None
        voice_item = None
        target = None
        anchor = None
        menu_result = None

        if pair:
            target, anchor = pair
            click_point = self._convert_button_click_point(target, anchor)
            if click_point:
                click_method = "visible_convert_button"
                voice_item = anchor
        else:
            voice_item = self._find_next_unconverted_voice_duration_item(items, rect)
            if voice_item:
                target = self._find_convert_button(items, rect, voice_item)
                if target:
                    click_method = "matched_convert_button"
                else:
                    click_point = self._estimated_convert_button_click_point(voice_item, rect)
                    if click_point:
                        click_method = "estimated_convert_button"
                    else:
                        target = self._find_convert_button(items, rect, None)
                        if target:
                            click_method = "fallback_convert_button"

        if click_method == "fallback_convert_button" and target:
            click_point = {"x": int(target.cx), "y": int(target.cy)}

        if click_point:
            self._click(rect, int(click_point["x"]), int(click_point["y"]), hwnd)
        elif voice_item and click_method != "fallback_convert_button":
            menu_result = self._select_convert_from_context_menu(rect, voice_item)
            if not menu_result.get("ok", False):
                return {"ok": False, "converted": False, "voice_detected": True,
                        "reason": "voice_convert_button_not_found",
                        "before_image": before_path, "click_method": click_method}
            click_method = "context_menu"

        if not click_point and not menu_result:
            return {"ok": False, "converted": False, "voice_detected": False,
                    "reason": "voice_convert_button_not_found",
                    "before_image": before_path, "click_method": click_method}

        if menu_result and not menu_result.get("ok", False):
            return {"ok": False, "converted": False, "voice_detected": True,
                    "reason": "voice_context_menu_convert_not_found",
                    "before_image": before_path, "menu_image": menu_result.get("image", ""),
                    "click_method": click_method}

        after_path, converted_text = self._wait_for_converted_text(
            rect, hwnd, anchor or voice_item or target, wait_seconds)

        if converted_text and converted_text[0] and converted_text[1]:
            return {"ok": True, "converted": True, "voice_detected": True,
                    "reason": "voice_converted_by_wechat",
                    "before_image": before_path, "after_image": after_path,
                    "converted_text": f"{converted_text[0]}: {converted_text[1]}",
                    "click_method": click_method}

        return {"ok": False, "converted": False, "voice_detected": True,
                "reason": "voice_transcription_text_not_found",
                "before_image": before_path, "after_image": after_path,
                "converted_text": "", "click_method": click_method}

    # ---------------------------------------------------------------
    # OCR 截图
    # ---------------------------------------------------------------
    def _capture_ocr(self, rect: dict[str, int], hwnd: int, prefix: str) -> tuple[str, list]:
        """截图并 OCR。

        Returns:
            (image_path, ocr_items)
        """
        path = ""
        result = None
        try:
            result = self.capture.capture_window(hwnd=hwnd, subdir="voice_to_text", prefix=prefix)
            if result and result.success and result.image is not None:
                path = getattr(result, "path", getattr(result, "image_path", ""))
                items = self.ocr.run(result.image, text_score=0.45)
                return path, list(items.items) if hasattr(items, "items") else list(items)
        except Exception as e:
            self._log(f"voice_to_text_ocr_failed prefix={prefix} error={e}")
        return path, []

    # ---------------------------------------------------------------
    # 按钮查找
    # ---------------------------------------------------------------
    @staticmethod
    def _find_convert_button(items: list, rect: dict[str, int],
                             anchor=None) -> Optional[Any]:
        """在 OCR 结果中查找「转文字」按钮。"""
        candidates = []
        for item in items:
            if VoiceToTextConverter._looks_like_convert_button(item.text):
                if rect:
                    width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
                    cx = float(getattr(item, "cx", 0))
                    cy = float(getattr(item, "cy", 0))
                    if width > 0 and cx > width * 0.3 and cy > rect.get("top", 0) + (rect.get("bottom", 0) - rect.get("top", 0)) * 0.82:
                        continue
                    if item.text == "转文字" and cx > 260:
                        continue
                    candidates.append(item)

        if not candidates:
            return None

        if anchor is not None:
            candidates.sort(key=lambda item: abs(float(getattr(item, "cy", 0)) - float(getattr(anchor, "cy", 0))))
            return candidates[0] if abs(float(getattr(candidates[0], "cy", 0)) - float(getattr(anchor, "cy", 0))) < 55 else None

        candidates.sort(key=lambda item: float(getattr(item, "cy", 0)))
        return candidates[0] if candidates else None

    @staticmethod
    def _find_next_voice_convert_pair(items: list, rect: dict[str, int]) -> Optional[tuple]:
        """查找最近的语音时长+转文字按钮配对。"""
        buttons = []
        for item in items:
            if VoiceToTextConverter._looks_like_convert_button(item.text):
                if rect:
                    width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
                    cx = float(getattr(item, "cx", 0))
                    if width > 0 and cx < width * 0.3:
                        continue
                buttons.append(item)

        voices = VoiceToTextConverter._voice_duration_items(items, rect)
        if not buttons or not voices:
            return None

        voices.sort(key=lambda voice: abs(float(getattr(voice, "cy", 0)) - float(getattr(voice, "cx", 0))))
        pairs = []
        for button in buttons:
            for voice in voices:
                dy = abs(float(getattr(button, "cy", 0)) - float(getattr(voice, "cy", 0)))
                if dy < 34:
                    pairs.append((button, voice, dy))

        if pairs:
            pairs.sort(key=lambda pair: (pair[2], pair[1].cy, pair[1].cx))
            return pairs[0][0], pairs[0][1]

        return None

    @staticmethod
    def _find_voice_duration_item(items: list, rect: dict[str, int]) -> Optional[Any]:
        """查找语音时长文本项。"""
        candidates = []
        for item in items:
            if VoiceToTextConverter._looks_like_voice_duration(item.text):
                if rect:
                    width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
                    cx = float(getattr(item, "cx", 0))
                    cy = float(getattr(item, "cy", 0))
                    if width > 0 and (cx < width * 0.1 or cx > width * 0.72):
                        continue
                    if width > 0 and (cy < rect.get("top", 0) + (rect.get("bottom", 0) - rect.get("top", 0)) * 0.3 or
                                      cy > rect.get("top", 0) + (rect.get("bottom", 0) - rect.get("top", 0)) * 0.82):
                        continue
                candidates.append(item)

        if not candidates:
            return None

        candidates.sort(key=lambda item: (float(getattr(item, "cy", 0)), float(getattr(item, "cx", 0))))
        return candidates[-1] if candidates else None

    @staticmethod
    def _find_next_voice_duration_item(items: list, rect: dict[str, int]) -> Optional[Any]:
        """查找第一条未转换的语音时长。"""
        candidates = VoiceToTextConverter._voice_duration_items(items, rect)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (float(getattr(item, "cy", 0)), float(getattr(item, "cx", 0))))
        return candidates[0]

    @staticmethod
    def _find_next_unconverted_voice_duration_item(items: list, rect: dict[str, int]) -> Optional[Any]:
        """查找下一条尚未转换的语音时长。"""
        candidates = VoiceToTextConverter._voice_duration_items(items, rect)
        candidates = [item for item in candidates
                      if not VoiceToTextConverter._voice_has_visible_transcript(item, items)]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (float(getattr(item, "cy", 0)), float(getattr(item, "cx", 0))))
        return candidates[0]

    @staticmethod
    def _voice_duration_items(items: list, rect: dict[str, int]) -> list:
        """获取所有语音时长文本项。"""
        result = []
        for item in items:
            if VoiceToTextConverter._looks_like_voice_duration(item.text):
                if rect:
                    width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
                    cx = float(getattr(item, "cx", 0))
                    cy = float(getattr(item, "cy", 0))
                    if width > 0 and (cx < width * 0.1 or cx > width * 0.72):
                        continue
                    if width > 0 and (cy < rect.get("top", 0) + (rect.get("bottom", 0) - rect.get("top", 0)) * 0.3 or
                                      cy > rect.get("top", 0) + (rect.get("bottom", 0) - rect.get("top", 0)) * 0.82):
                        continue
                result.append(item)
        return result

    # ---------------------------------------------------------------
    # 外观判断
    # ---------------------------------------------------------------
    @staticmethod
    def _looks_like_convert_button(text: str) -> bool:
        """判断是不是真的「转文字」按钮，排除 bot 消息中的子串。"""
        value = str(text or "").strip()
        if not value:
            return False
        for ch in "[]【】「」":
            value = value.replace(ch, "")
        if value == "转文字" and len(str(text or "").strip()) <= 6:
            return True
        return False

    @staticmethod
    def _looks_like_voice_duration(text: str) -> bool:
        """判断是否是语音时长文本（如 '15''、'30秒'）。"""
        value = str(text or "").strip()
        if not value:
            return False
        value = value.replace(" ", "").replace("\n", "")
        if value == "转文字":
            return False
        for ch in "'′＇″''\"”“":
            value = value.replace(ch, "'")
        return bool(re.fullmatch(r"\d{1,2}('{1,2}|秒|s|S)", value))

    @staticmethod
    def _text_after_voice(anchor: Any, items: list) -> str:
        """获取语音消息后的文本（转文字结果）。"""
        texts = []
        anchor_y = float(getattr(anchor, "cy", 0))
        for item in items:
            item_y = float(getattr(item, "cy", 0))
            if item_y > anchor_y - 10:
                t = str(item.text or "").strip()
                if t and t not in ("转文字", "语音", "播放"):
                    texts.append(t)
        return "\n".join(texts)

    @staticmethod
    def _voice_has_visible_transcript(anchor: Any, items: list) -> bool:
        """检查语音是否已有可见的转文字结果。"""
        anchor_y = float(getattr(anchor, "cy", 0))
        for item in items:
            item_y = float(getattr(item, "cy", 0))
            if item_y > anchor_y - 10 and item_y < anchor_y + 60:
                if VoiceToTextConverter._looks_like_transcript_text(item.text):
                    return True
        return False

    @staticmethod
    def _looks_like_transcript_text(text: str) -> bool:
        """判断文本是否看起来像转文字结果。"""
        value = str(text or "").strip()
        if not value:
            return False
        if len(value) < 2:
            return False
        if value in ("转文字", "语音", "播放"):
            return False
        return True

    # ---------------------------------------------------------------
    # 坐标计算
    # ---------------------------------------------------------------
    @staticmethod
    def _convert_button_click_point(target: Any, anchor: Any) -> Optional[dict[str, int]]:
        """计算「转文字」按钮的点击坐标。"""
        text = str(getattr(target, "text", "") or "").strip()
        if text == "转文字":
            anchor_text = str(getattr(anchor, "text", "") or "").strip()
            if VoiceToTextConverter._looks_like_voice_duration(anchor_text):
                button_right = int(getattr(target, "right", 0))
                voice_cy = int(getattr(anchor, "cy", 0))
                if button_right > 0:
                    return {"x": button_right + 18, "y": voice_cy}
        return {
            "x": int(getattr(target, "cx", 0)),
            "y": int(getattr(target, "cy", 0)),
        } if getattr(target, "cx", 0) else None

    @staticmethod
    def _estimated_convert_button_click_point(voice_item: Any,
                                               rect: dict[str, int]) -> Optional[dict[str, int]]:
        """根据语音时长条估算「转文字」按钮位置。"""
        width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
        if width <= 0:
            return None
        voice_right = int(getattr(voice_item, "right", 0))
        x = max(voice_right + 86, rect.get("left", 0) + width - 100)
        x = min(x, rect.get("right", 0) - 24)
        return {"x": x, "y": int(getattr(voice_item, "cy", 0))}

    # ---------------------------------------------------------------
    # 右键菜单转文字
    # ---------------------------------------------------------------
    def _select_convert_from_context_menu(self, rect: dict[str, int],
                                           voice_item: Any) -> dict[str, Any]:
        """右键点击语音消息，在弹出的上下文菜单中选择「转文字」。"""
        screen_x = int(getattr(voice_item, "cx", 0))
        screen_y = int(getattr(voice_item, "cy", 0))

        from .background_clicker import background_right_click_screen
        if not background_right_click_screen(screen_x, screen_y):
            return {"ok": False, "reason": "right_click_failed"}
        time.sleep(0.3)

        origin = self._virtual_screen_origin()
        menu_path, menu_items = self._capture_screen_ocr("voice_context_menu")

        for item in menu_items:
            if item.text == "转文字":
                image_x = int(getattr(item, "cx", 0))
                image_y = int(getattr(item, "cy", 0))
                self._click_screen(origin["x"] + image_x, origin["y"] + image_y)
                return {"ok": True, "image": menu_path}

        return {"ok": False, "reason": "menu_convert_item_not_found", "image": menu_path}

    def _capture_screen_ocr(self, prefix: str) -> tuple[str, list]:
        """全屏截图并 OCR。"""
        path = ""
        try:
            result = self.capture._capture_imagegrab_rect({})
            if result is not None and hasattr(result, "image") and result.image is not None:
                path = getattr(result, "path", "")
                items = self.ocr.run(result.image, text_score=0.35)
                return path, list(items.items) if hasattr(items, "items") else list(items)
        except Exception:
            pass
        return path, []

    def _find_menu_convert_item(self, items: list, screen_x: int, screen_y: int) -> Optional[Any]:
        """在 OCR 结果中查找「转文字」菜单项。"""
        for item in items:
            if item.text == "转文字":
                cx = float(getattr(item, "cx", 0))
                cy = float(getattr(item, "cy", 0))
                if abs(cx - screen_x) < 200 and abs(cy - screen_y) < 200:
                    return item
        return None

    # ---------------------------------------------------------------
    # 等待转文字结果
    # ---------------------------------------------------------------
    def _wait_for_converted_text(self, rect: dict[str, int], hwnd: int,
                                  anchor: Any, wait_seconds: float) -> tuple[str, tuple[str, str]]:
        """等待转文字完成并获取结果。"""
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            time.sleep(0.3)
            after_path, items = self._capture_ocr(rect, hwnd, "voice_after_convert")
            text = self._text_after_voice(anchor, items)
            if text and len(text.strip()) > 1:
                return after_path, ("", text.strip())

        after_path, items = self._capture_ocr(rect, hwnd, "voice_after_convert_final")
        text = self._text_after_voice(anchor, items)
        return after_path, ("", text.strip() if text else "")

    # ---------------------------------------------------------------
    # 虚拟屏幕原点
    # ---------------------------------------------------------------
    @staticmethod
    def _virtual_screen_origin() -> dict[str, int]:
        """获取虚拟屏幕原点坐标。"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sm_x = user32.GetSystemMetrics(76)
            sm_y = user32.GetSystemMetrics(77)
            return {"x": sm_x, "y": sm_y}
        except Exception:
            return {"x": 0, "y": 0}

    # ---------------------------------------------------------------
    # 去重
    # ---------------------------------------------------------------
    @staticmethod
    def _dedupe_texts(values: list) -> str:
        """去重转文字结果。"""
        result = []
        seen = set()
        for v in values:
            v = str(v).strip()
            if v and v not in seen:
                result.append(v)
                seen.add(v)
        return "\n".join(result)

    # ---------------------------------------------------------------
    # 点击
    # ---------------------------------------------------------------
    def _click(self, rect: dict[str, int], local_x: int, local_y: int,
               hwnd: int) -> None:
        """点击窗口内的坐标（后台 PostMessage，不移动真实鼠标）。"""
        try:
            from .background_clicker import background_click_screen
            screen_x = rect.get("left", 0) + local_x
            screen_y = rect.get("top", 0) + local_y
            background_click_screen(screen_x, screen_y)
        except Exception:
            pass

    def _click_screen(self, x: int, y: int) -> None:
        """点击屏幕坐标（后台 PostMessage）。"""
        try:
            from .background_clicker import background_click_screen
            background_click_screen(x, y)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 日志
    # ---------------------------------------------------------------
    def _log(self, message: str) -> None:
        """记录日志。"""
        try:
            if self.logger:
                self.logger(message)
        except Exception:
            pass

    # =================================================================
    # 兼容旧接口（简化版单条转换）
    # =================================================================
    def convert_voice(self, hwnd: int, voice_region=None) -> str | None:
        """Click '转文字' button and OCR the transcript. (compat)"""
        if not hwnd:
            return None

        frame = self.capture.capture_window(hwnd=hwnd)
        if not frame or not frame.success:
            return None

        image = frame.image
        if image is None:
            return None

        try:
            ocr_result = self.ocr.run(image)
        except Exception:
            return None

        if not ocr_result:
            return None

        items = list(ocr_result.items) if hasattr(ocr_result, "items") else list(ocr_result)

        for item in items:
            text = str(getattr(item, "text", "") or "")
            if "转文字" in text:
                self._click_transcribe(hwnd, item)
                time.sleep(1.5)
                return self._read_transcript(hwnd)

        return None

    def _click_transcribe(self, hwnd: int, ocr_item) -> None:
        cx, cy = _textbox_center(ocr_item)
        if cx and cy:
            self._click_screen(int(cx), int(cy))

    def _read_transcript(self, hwnd: int) -> str | None:
        frame = self.capture.capture_window(hwnd=hwnd)
        if not frame or not frame.success or frame.image is None:
            return None

        try:
            ocr_result = self.ocr.run(frame.image)
        except Exception:
            return None

        if not ocr_result:
            return None

        texts = []
        items = list(ocr_result.items) if hasattr(ocr_result, "items") else list(ocr_result)
        for item in items:
            text = str(getattr(item, "text", "") or "")
            if text and text not in ("转文字", "语音", "播放"):
                texts.append(text)

        return "\n".join(texts) if texts else None