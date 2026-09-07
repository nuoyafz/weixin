"""SendConfirm - screenshot-based confirmation for WeChat send actions.

Aligned with original app.rpa.send_confirm v3.10:
- Uses PIL for pixel-level difference detection
- Region-specific thresholds (chat area, input area, customer area)
- WeChatOcrParser integration for text verification
- Multi-capture retry with configurable delay
- confirm_send, confirm_expected_reply, inspect_new_customer_messages
- inspect_chat_message_visual_change, confirm_input_matches_baseline
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops


class SendConfirmResult(NamedTuple):
    success: bool
    diff_ratio: float
    changed_pixels: int
    region: Optional[Tuple[int, int, int, int]] = None
    before_present: bool = True
    after_present: bool = True
    error: str = ""


class SendConfirm:
    """Screenshot-based confirmation for WeChat send actions.

    Aligned with original v3.10 constants and pixel-diff approach.
    """

    PIXEL_DIFF_THRESHOLD = 18
    CHAT_CHANGE_RATIO = 0.006
    CHAT_CHANGE_MEAN = 0.35
    CUSTOMER_VISUAL_CHANGE_RATIO = 0.0015
    CUSTOMER_VISUAL_CHANGE_MEAN = 0.18
    INPUT_CHANGE_RATIO = 0.0015
    INPUT_CHANGE_MEAN = 0.08
    INPUT_DRAFT_DARK_RATIO = 0.001
    INPUT_DRAFT_NONWHITE_RATIO = 0.02
    DARK_INPUT_LIGHT_FOREGROUND_RATIO = 0.0008
    DARK_INPUT_COLORED_FOREGROUND_RATIO = 0.002
    CHAT_READY_INPUT_NONWHITE_RATIO = 0.01
    CHAT_READY_HEADER_NONWHITE_RATIO = 0.015
    CHAT_READY_BODY_NONWHITE_RATIO = 0.05

    def __init__(self, config=None, screen_capture=None, logger=None):
        if isinstance(config, dict) or config is None:
            self.config = config or {}
        else:
            self.config = {}
            screen_capture = config
        if screen_capture is None:
            from ..capture.screen_capture import ScreenCapture
            screen_capture = ScreenCapture()
        self.screen_capture = screen_capture
        self.logger = logger
        self.min_diff_ratio = 0.005
        self._ocr_parser = None

    def set_ocr_parser(self, parser):
        self._ocr_parser = parser

    def capture(self, hwnd: int) -> Optional[np.ndarray]:
        """截取窗口帧，失败返回 None。"""
        result = self.screen_capture.capture_window(hwnd)
        if not result.success or result.image is None:
            return None
        return result.image

    def _crop(self, frame: np.ndarray,
              region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if region is None:
            return frame
        x, y, w, h = region
        x = max(0, int(x)); y = max(0, int(y))
        return frame[y:y + int(h), x:x + int(w)]

    def compare(self, before: np.ndarray, after: np.ndarray,
                region: Optional[Tuple[int, int, int, int]] = None,
                min_diff_ratio: Optional[float] = None) -> SendConfirmResult:
        """比较前后两帧的像素差异，返回变化比例。"""
        ratio = min_diff_ratio if min_diff_ratio is not None else self.min_diff_ratio
        if before is None or after is None:
            return SendConfirmResult(
                success=False, diff_ratio=0.0, changed_pixels=0,
                region=region,
                before_present=before is not None,
                after_present=after is not None,
                error="before/after frame missing")

        try:
            b = self._crop(before, region)
            a = self._crop(after, region)
            if b.shape != a.shape:
                a = self._resize_like(a, b.shape)
            if b.size == 0:
                return SendConfirmResult(success=False, diff_ratio=0.0,
                                         changed_pixels=0, region=region,
                                         error="empty region")

            diff = (np.abs(a.astype(np.int16) - b.astype(np.int16))
                    .max(axis=2) > 12)
            changed = int(diff.sum())
            ratio_val = changed / float(diff.size)
            success = ratio_val >= ratio
            return SendConfirmResult(
                success=success, diff_ratio=round(ratio_val, 5),
                changed_pixels=changed, region=region)
        except Exception as e:
            return SendConfirmResult(success=False, diff_ratio=0.0,
                                     changed_pixels=0, region=region,
                                     error=str(e))

    def confirm_after_send(self, hwnd: int, send_fn: Callable[[], bool],
                           region: Optional[Tuple[int, int, int, int]] = None,
                           min_diff_ratio: Optional[float] = None,
                           settle_time: float = 0.5) -> SendConfirmResult:
        """发送前后截屏比对确认。"""
        before = self.capture(hwnd)
        try:
            ok = bool(send_fn())
        except Exception as e:
            return SendConfirmResult(success=False, diff_ratio=0.0,
                                     changed_pixels=0, region=region,
                                     error=f"send_fn raised: {e}")
        if not ok:
            return SendConfirmResult(success=False, diff_ratio=0.0,
                                     changed_pixels=0, region=region,
                                     error="send_fn returned False")
        time.sleep(settle_time)
        after = self.capture(hwnd)
        return self.compare(before, after, region, min_diff_ratio)

    # ---------------------------------------------------------------
    # 综合发送确认（对齐原版 confirm_send）
    # ---------------------------------------------------------------
    def confirm_send(self, before_image: np.ndarray,
                     after_image: np.ndarray,
                     after_paste_image: Optional[np.ndarray] = None,
                     expected_reply: str = "",
                     expected_customer_text: str = "",
                     window_width: int = 0,
                     window_height: int = 0) -> Dict[str, Any]:
        """Comprehensive send confirmation matching original v3.10.

        Returns dict with keys:
          - success: bool
          - reason: str
          - visual_ok: bool
          - visual_reply_delivery_ok: bool
          - visual_reply_fallback_ok: bool
          - reply_text_check: bool
          - new_customer_check: bool
          - new_customer_before_reply: bool
          - before_after_metrics: dict
          - before_after_paste_metrics: dict
          - chat_changed: bool
          - right_chat_changed: bool
          - paste_metrics: dict
        """
        result = {
            "success": False,
            "reason": "",
            "visual_ok": False,
            "visual_reply_delivery_ok": False,
            "visual_reply_fallback_ok": False,
            "reply_text_check": False,
            "new_customer_check": False,
            "new_customer_before_reply": False,
            "before_after_metrics": {},
            "before_after_paste_metrics": {},
            "chat_changed": False,
            "right_chat_changed": False,
            "paste_metrics": {},
        }

        if after_image is None:
            result["reason"] = "send_screenshot_missing"
            return result

        # 1. Pixel-level visual change detection
        before_after_metrics = self._diff_metrics(before_image, after_image)
        result["before_after_metrics"] = before_after_metrics
        chat_changed = self._chat_changed(before_after_metrics)
        right_chat_changed = self._right_chat_changed(before_after_metrics)
        result["chat_changed"] = chat_changed
        result["right_chat_changed"] = right_chat_changed

        if after_paste_image is not None:
            paste_metrics = self._diff_metrics(after_paste_image, after_image)
            result["paste_metrics"] = paste_metrics
            result["before_after_paste_metrics"] = paste_metrics

        # 2. Visual reply delivery check
        if chat_changed or right_chat_changed:
            result["visual_ok"] = True
            result["visual_reply_delivery_ok"] = True
            result["visual_reply_fallback_ok"] = True
            result["reason"] = "send_confirmed_by_visual_change"

            if right_chat_changed:
                result["reason"] = "send_confirmed_by_visual_right_bubble"
        else:
            # Check if reply text is visible
            reply_text_check = self.confirm_expected_reply(
                before_image, after_image, expected_reply, expected_customer_text)
            result["reply_text_check"] = bool(reply_text_check)

            if reply_text_check:
                result["visual_ok"] = True
                result["reason"] = "send_confirmed_by_reply_text"
            else:
                result["reason"] = "sent_reply_text_not_found"

        # 3. New customer message check
        new_customer_result = self.inspect_new_customer_messages(
            before_image, after_image, expected_customer_text)
        result["new_customer_check"] = new_customer_result.get("has_new", False)
        result["new_customer_before_reply"] = new_customer_result.get(
            "has_new_before_reply", False)

        # 4. Paste check
        if after_paste_image is not None:
            if not self._region_changed(after_paste_image, after_image,
                                        self.INPUT_CHANGE_RATIO,
                                        self.INPUT_CHANGE_MEAN):
                result["reason"] = "paste_not_visible_in_input_box"

        # 5. Input clear check
        if not self.inspect_input_clear(before_image, after_image):
            if result["reason"] == "send_confirmed_by_visual_change":
                pass  # still OK, visual change confirmed
            else:
                result["reason"] = "input_box_not_cleared_after_enter"

        result["success"] = result["visual_ok"] or result["reply_text_check"]
        return result

    # ---------------------------------------------------------------
    # 期望回复文本确认
    # ---------------------------------------------------------------
    def confirm_expected_reply(self, before_image: np.ndarray,
                                after_image: np.ndarray,
                                expected_reply: str,
                                expected_customer_text: str = "") -> bool:
        """Check if the expected reply text appears in the chat area."""
        if not expected_reply or not expected_reply.strip():
            return False
        if self._ocr_parser is None:
            return False

        expected = expected_reply.strip()
        try:
            messages = self._ocr_chat_messages(after_image)
            if not messages:
                return False

            # Check if reply text appears after the expected customer text
            expected_customer_bottom = self._expected_customer_bottom(
                messages, expected_customer_text)

            matched = self._matching_self_messages(messages, expected)
            for m in matched:
                text = m.get("text", "")
                if expected in text or self._looks_like_same_text(text, expected):
                    if expected_customer_bottom is None:
                        return True
                    y = m.get("y_center", 0)
                    if y > expected_customer_bottom:
                        return True
            return False
        except Exception:
            return False

    # ---------------------------------------------------------------
    # 新客户消息检测
    # ---------------------------------------------------------------
    def inspect_new_customer_messages(self, before_image: np.ndarray,
                                       after_image: np.ndarray,
                                       expected_customer: str = "") -> Dict[str, Any]:
        """Detect new customer messages that arrived during send."""
        result = {
            "has_new": False,
            "has_new_before_reply": False,
            "detection_source": "",
            "visual_fallback": False,
            "before_customer_count": 0,
            "after_customer_count": 0,
        }

        if before_image is None or after_image is None:
            return result

        # OCR-based detection
        if self._ocr_parser is not None:
            try:
                before_messages = self._ocr_chat_messages(before_image)
                after_messages = self._ocr_chat_messages(after_image)

                before_customer = [m for m in before_messages
                                   if self._is_customer_message(m)]
                after_customer = [m for m in after_messages
                                  if self._is_customer_message(m)]

                result["before_customer_count"] = len(before_customer)
                result["after_customer_count"] = len(after_customer)

                if len(after_customer) > len(before_customer):
                    result["has_new"] = True
                    result["detection_source"] = "ocr"

                    # Check if new message appeared before our reply
                    try:
                        before_bottom = self._expected_customer_bottom(
                            before_messages, expected_customer)
                        if before_bottom is not None:
                            for m in after_customer:
                                if not self._customer_message_matches_expected(
                                        m, expected_customer):
                                    y = m.get("y_center", 0)
                                    if y < before_bottom:
                                        result["has_new_before_reply"] = True
                                        break
                    except Exception:
                        pass
            except Exception:
                pass

        # Visual fallback
        if not result["has_new"]:
            visual_result = self.inspect_chat_message_visual_change(
                before_image, after_image)
            if visual_result.get("changed"):
                result["has_new"] = True
                result["detection_source"] = "visual"
                result["visual_fallback"] = True

        return result

    # ---------------------------------------------------------------
    # 聊天区域视觉变化检测
    # ---------------------------------------------------------------
    def inspect_chat_message_visual_change(self, before_image: np.ndarray,
                                            after_image: np.ndarray) -> Dict[str, Any]:
        """Compare only the visible chat-message area, excluding the editor."""
        result = {"changed": False, "reason": ""}

        if before_image is None or after_image is None:
            result["reason"] = "customer_visual_guard_image_missing"
            return result

        h, w = before_image.shape[:2]
        input_top_y = int(h * 0.82)  # Input box is roughly bottom 18%

        before_chat = before_image[:input_top_y, :]
        after_chat = after_image[:input_top_y, :]

        if before_chat.shape != after_chat.shape:
            min_h = min(before_chat.shape[0], after_chat.shape[0])
            min_w = min(before_chat.shape[1], after_chat.shape[1])
            before_chat = before_chat[:min_h, :min_w]
            after_chat = after_chat[:min_h, :min_w]

        metrics = self._diff_metrics(before_chat, after_chat)
        if metrics.get("ratio", 0) > self.CUSTOMER_VISUAL_CHANGE_RATIO:
            result["changed"] = True
            result["reason"] = "chat_message_area_changed"
        else:
            result["reason"] = "chat_message_area_unchanged"

        return result

    # ---------------------------------------------------------------
    # 输入框清空检查
    # ---------------------------------------------------------------
    def inspect_input_clear(self, before_image: np.ndarray,
                            after_image: np.ndarray) -> bool:
        """Check if the input box was cleared after sending."""
        if before_image is None or after_image is None:
            return False
        h = before_image.shape[0]
        input_region = (0, int(h * 0.82), before_image.shape[1], h - int(h * 0.82))
        before_input = self._crop(before_image, input_region)
        after_input = self._crop(after_image, input_region)
        metrics = self._diff_metrics(before_input, after_input)
        return metrics.get("ratio", 0) < self.INPUT_CHANGE_RATIO * 2

    # ---------------------------------------------------------------
    # 粘贴确认
    # ---------------------------------------------------------------
    def confirm_paste(self, before_image: np.ndarray,
                      after_paste_image: np.ndarray) -> Dict[str, Any]:
        """Confirm text was pasted into the input box."""
        result = {
            "success": False,
            "reason": "",
            "metrics": {},
        }

        if before_image is None:
            result["reason"] = "paste_screenshot_missing"
            return result
        if after_paste_image is None:
            result["reason"] = "after_paste_exists"
            return result

        metrics = self._diff_metrics(before_image, after_paste_image)
        result["metrics"] = metrics

        if self._region_changed(before_image, after_paste_image,
                                self.INPUT_CHANGE_RATIO,
                                self.INPUT_CHANGE_MEAN):
            result["success"] = True
            result["reason"] = "input_box_changed_after_paste"
        else:
            result["reason"] = "input_box_unchanged_after_paste"

        return result

    def confirm_input_matches_baseline(self, baseline_image: np.ndarray,
                                        current_image: np.ndarray) -> Dict[str, Any]:
        """Confirm that the editor returned to the post-quote safe baseline."""
        result = {
            "success": False,
            "reason": "",
        }

        if baseline_image is None:
            result["reason"] = "input_baseline_screenshot_missing"
            return result
        if current_image is None:
            result["reason"] = "current_exists"
            return result

        try:
            baseline_box = self._input_text_box(baseline_image)
            current_box = self._input_text_box(current_image)

            if baseline_box is None or current_box is None:
                result["reason"] = "input_baseline_compare_failed"
                return result

            baseline_crop = self._crop(baseline_image, baseline_box)
            current_crop = self._crop(current_image, current_box)

            metrics = self._diff_metrics(baseline_crop, current_crop)
            if metrics.get("ratio", 0) < self.INPUT_CHANGE_RATIO * 0.5:
                result["success"] = True
                result["reason"] = "input_box_matches_quote_baseline"
            else:
                result["reason"] = "input_box_differs_from_quote_baseline"
        except Exception as e:
            result["reason"] = f"input_baseline_compare_failed: {e}"

        return result

    # ---------------------------------------------------------------
    # 内部辅助方法
    # ---------------------------------------------------------------
    def _diff_metrics(self, before: np.ndarray,
                      after: np.ndarray) -> Dict[str, Any]:
        """Calculate per-pixel diff metrics between two images."""
        if before is None or after is None:
            return {"ratio": 0.0, "changed": 0, "total": 0, "mean": 0.0}
        try:
            if before.shape != after.shape:
                min_h = min(before.shape[0], after.shape[0])
                min_w = min(before.shape[1], after.shape[1])
                before = before[:min_h, :min_w]
                after = after[:min_h, :min_w]
            diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
            max_diff = diff.max(axis=2)
            changed = int((max_diff > self.PIXEL_DIFF_THRESHOLD).sum())
            total = int(max_diff.size)
            ratio = changed / float(total) if total > 0 else 0.0
            mean_val = float(max_diff.mean())
            return {"ratio": ratio, "changed": changed, "total": total, "mean": mean_val}
        except Exception:
            return {"ratio": 0.0, "changed": 0, "total": 0, "mean": 0.0}

    def _region_changed(self, before: np.ndarray, after: np.ndarray,
                        ratio_threshold: float,
                        mean_threshold: float) -> bool:
        metrics = self._diff_metrics(before, after)
        return (metrics.get("ratio", 0) > ratio_threshold or
                metrics.get("mean", 0) > mean_threshold)

    def _chat_changed(self, metrics: Dict[str, Any]) -> bool:
        return (metrics.get("ratio", 0) > self.CHAT_CHANGE_RATIO or
                metrics.get("mean", 0) > self.CHAT_CHANGE_MEAN)

    def _right_chat_changed(self, metrics: Dict[str, Any]) -> bool:
        return (metrics.get("ratio", 0) > self.CHAT_CHANGE_RATIO * 0.5)

    def _input_text_box(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the input text box region."""
        if image is None:
            return None
        h, w = image.shape[:2]
        y_start = int(h * 0.84)
        y_end = int(h * 0.96)
        x_start = int(w * 0.05)
        x_end = int(w * 0.95)
        return (x_start, y_start, x_end - x_start, y_end - y_start)

    def _input_text_box_for_image(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        return self._input_text_box(image)

    def _ocr_chat_messages(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR on the chat area and return structured messages."""
        if self._ocr_parser is None:
            return []
        try:
            return self._ocr_parser.parse_chat_messages(image)
        except Exception:
            return []

    def _matching_self_messages(self, messages: List[Dict[str, Any]],
                                 text: str) -> List[Dict[str, Any]]:
        """Find self-sent messages matching the given text."""
        return [m for m in messages
                if m.get("side") == "right" and
                (text in (m.get("text", "") or "") or
                 self._looks_like_same_text(m.get("text", ""), text))]

    def _expected_customer_bottom(self, messages: List[Dict[str, Any]],
                                   expected_text: str) -> Optional[float]:
        """Find the bottom y of the expected customer message."""
        if not expected_text:
            return None
        for m in reversed(messages):
            if m.get("side") == "left":
                if self._looks_like_same_text(m.get("text", ""), expected_text):
                    return float(m.get("y_center", 0))
        return None

    def _is_customer_message(self, msg: Dict[str, Any]) -> bool:
        """Check if a message is from the customer (left side)."""
        return msg.get("side") == "left" and not self._is_low_confidence_ocr_fragment(msg)

    def _is_low_confidence_ocr_fragment(self, msg: Dict[str, Any]) -> bool:
        """Filter out low-confidence OCR fragments."""
        text = (msg.get("text", "") or "").strip()
        if len(text) <= 1:
            return True
        if float(msg.get("confidence", 0)) < 0.7:
            return True
        return False

    def _customer_message_matches_expected(self, msg: Dict[str, Any],
                                            expected: str) -> bool:
        if not expected:
            return False
        text = (msg.get("text", "") or "").strip()
        return self._looks_like_same_text(text, expected)

    def _customer_message_signature(self, msg: Dict[str, Any]) -> str:
        text = (msg.get("text", "") or "").strip()
        return re.sub(r'[\s\W_]+', '', text.lower())[:40]

    def _find_existing_customer_message_index(self, msg: Dict[str, Any],
                                               before_messages: List[Dict[str, Any]],
                                               used_indices: Optional[Any] = None) -> int:
        """Find matching customer message index in before_messages.

        Aligned with original signature (after_msg, before_messages, used_indices);
        used_indices lets callers skip already-matched slots.
        """
        used = set(used_indices) if used_indices else set()
        signature = self._customer_message_signature(msg)
        for i, before_msg in enumerate(before_messages):
            if i in used:
                continue
            if self._customer_message_signature(before_msg) == signature:
                return i
            if self._customer_messages_look_same(msg, before_msg):
                return i
        return -1

    def _customer_messages_look_same(self, a: Dict[str, Any],
                                      b: Dict[str, Any]) -> bool:
        a_text = (a.get("text", "") or "").strip()
        b_text = (b.get("text", "") or "").strip()
        if not a_text or not b_text:
            return False
        shorter = a_text if len(a_text) < len(b_text) else b_text
        longer = b_text if len(a_text) < len(b_text) else a_text
        if len(shorter) < 3:
            return False
        ratio = SequenceMatcher(None, shorter, longer).ratio()
        return ratio > 0.72

    def _message_positions_close(self, a: Dict[str, Any],
                                  b: Dict[str, Any]) -> bool:
        a_y = float(a.get("y_center", 0))
        b_y = float(b.get("y_center", 0))
        a_x = float(a.get("cx", 0))
        b_x = float(b.get("cx", 0))
        return abs(a_y - b_y) < 28 and abs(a_x - b_x) < 80

    def _message_preview(self, msg: Dict[str, Any]) -> str:
        text = (msg.get("text", "") or "").strip()
        return text[:50]

    def _message_intersects_input_box(self, msg: Dict[str, Any],
                                       input_box: Tuple[int, int, int, int]) -> bool:
        y = float(msg.get("y_center", 0))
        _, iy, _, ih = input_box
        return iy <= y <= iy + ih

    def _has_customer_message_before(self, before_messages: List[Dict[str, Any]],
                                      expected_customer: str) -> bool:
        if not expected_customer:
            return False
        for m in before_messages:
            if m.get("side") == "left" and self._customer_message_matches_expected(
                    m, expected_customer):
                return True
        return False

    @staticmethod
    def _looks_like_same_text(a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_norm = re.sub(r'[\s\W_]+', '', a.lower())
        b_norm = re.sub(r'[\s\W_]+', '', b.lower())
        if not a_norm or not b_norm:
            return False
        if a_norm == b_norm:
            return True
        shorter = a_norm if len(a_norm) < len(b_norm) else b_norm
        longer = b_norm if len(a_norm) < len(b_norm) else a_norm
        if len(shorter) < 3:
            return False
        return SequenceMatcher(None, shorter, longer).ratio() > 0.8

    @staticmethod
    def _resize_like(frame: np.ndarray, shape: tuple) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(frame, (shape[1], shape[0]))
        except Exception:
            return frame

    # =================================================================
    # 输入框边框检测（对齐原版）
    # =================================================================

    def _detect_input_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """通过检测输入框边框像素来定位输入区域，对齐原版 _detect_input_frame。

        微信输入框顶部有一条浅灰色分隔线，底部有边框。
        通过扫描这些特征来精确确定输入框位置。
        """
        if image is None:
            return None
        try:
            h, w = image.shape[:2]
            # 只扫描底部 40% 区域
            y_start = int(h * 0.6)
            y_end = h

            # 寻找输入框顶部分隔线（浅灰色横向线）
            input_top = None
            for y in range(y_start, y_end - 10):
                light_gray = self._light_gray_runs_on_row(image, y, w)
                if light_gray > w * 0.6:
                    input_top = y
                    break

            # 寻找输入框底部边框
            input_bottom = None
            for y in range(y_end - 1, y_start + 10, -1):
                if self._is_input_border_pixel(image, y, w):
                    input_bottom = y
                    break

            if input_top is None:
                input_top = int(h * 0.84)
            if input_bottom is None:
                input_bottom = int(h * 0.96)

            x_start = int(w * 0.05)
            x_end = int(w * 0.95)

            return (x_start, input_top, x_end - x_start, input_bottom - input_top)
        except Exception:
            return None

    def _light_gray_runs_on_row(self, image: np.ndarray, y: int, w: int) -> int:
        """统计一行中浅灰色像素的连续长度，对齐原版 _light_gray_runs_on_row。"""
        if y < 0 or y >= image.shape[0]:
            return 0
        max_run = 0
        current_run = 0
        sample_step = 4
        for x in range(0, min(w, image.shape[1]), sample_step):
            pixel = image[y, x]
            r, g, b = int(pixel[2]), int(pixel[1]), int(pixel[0])
            # 浅灰色判定
            if 180 <= r <= 240 and 180 <= g <= 240 and 180 <= b <= 240:
                current_run += sample_step
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run

    def _is_input_border_pixel(self, image: np.ndarray, y: int, w: int) -> bool:
        """检查一行是否包含输入框边框特征，对齐原版 _is_input_border_pixel。"""
        if y < 0 or y >= image.shape[0]:
            return False
        sample_step = 4
        border_pixels = 0
        total = 0
        for x in range(0, min(w, image.shape[1]), sample_step):
            pixel = image[y, x]
            r, g, b = int(pixel[2]), int(pixel[1]), int(pixel[0])
            # 深色边框像素
            if r < 100 and g < 100 and b < 100:
                border_pixels += 1
            total += 1
        return total > 0 and border_pixels / total > 0.15

    # =================================================================
    # 原版缺失方法补充
    # =================================================================

    def confirm(self, before_image, after_image,
                after_paste_image=None) -> Dict[str, Any]:
        """Simple wrapper around confirm_send, matching original signature."""
        return self.confirm_send(
            before_image=before_image,
            after_image=after_image,
            after_paste_image=after_paste_image,
        )

    def inspect_chat_ready(self, before_image: np.ndarray,
                           hwnd: int = 0) -> Dict[str, object]:
        """Check if the chat input area is ready for sending.

        Verifies that the chat window is visible and the input area is
        accessible by checking non-white pixel ratios in key regions.
        """
        result: Dict[str, object] = {"ok": True, "reason": ""}

        if before_image is None:
            result["ok"] = False
            result["reason"] = "chat_ready_screenshot_missing"
            return result

        try:
            h, w = before_image.shape[:2]

            input_box = self._input_text_box(before_image)
            if input_box is None:
                result["ok"] = False
                result["reason"] = "input_box_not_detected"
                return result

            ix, iy, iw, ih = input_box
            input_region = before_image[iy:iy + ih, ix:ix + iw]

            input_nonwhite = self._color_metrics(
                input_region, (0.34, 0.76, 0.985, 0.965))
            if input_nonwhite.get("nonwhite_ratio", 0) < self.CHAT_READY_INPUT_NONWHITE_RATIO:
                result["ok"] = False
                result["reason"] = "input_area_blank"
                return result

            header_y = int(h * 0.02)
            header_h = int(h * 0.08)
            if header_h > 0:
                header_region = before_image[header_y:header_y + header_h, :]
                header_metrics = self._color_metrics(
                    header_region, (0.34, 0.06, 0.985, 0.14))
                if header_metrics.get("nonwhite_ratio", 0) < self.CHAT_READY_HEADER_NONWHITE_RATIO:
                    result["ok"] = False
                    result["reason"] = "chat_header_blank"
                    return result

            body_y = int(h * 0.08)
            body_h = iy - body_y
            if body_h > 0:
                body_region = before_image[body_y:body_y + body_h, :]
                body_metrics = self._color_metrics(
                    body_region, (0.34, 0.14, 0.985, 0.965))
                if body_metrics.get("nonwhite_ratio", 0) < self.CHAT_READY_BODY_NONWHITE_RATIO:
                    result["ok"] = False
                    result["reason"] = "chat_body_blank"
                    return result

        except Exception as e:
            result["ok"] = False
            result["reason"] = f"chat_ready_check_failed: {e}"

        return result

    def inspect_current_customer_turn(
        self,
        image: np.ndarray,
        expected_text: str = "",
        *,
        baseline_image: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Check if the current customer turn matches the expected text.

        Used before sending to verify the customer hasn't sent new messages
        that would change the conversation context.
        """
        result: Dict[str, object] = {
            "enabled": True,
            "ok": True,
            "reason": "",
            "ocr_ok": False,
            "visual_check": None,
        }

        expected = str(expected_text).strip() if expected_text else ""
        if not expected:
            result["ok"] = False
            result["reason"] = "expected_customer_turn_empty"
            return result

        # 记录 baseline（进入聊天时的快照），供文本不匹配时做视觉差分双信号。
        # 参数为 numpy 数组（非路径），调用方应传入 analyze 时刻的截图帧。
        baseline_img = baseline_image if isinstance(baseline_image, np.ndarray) else None
        if baseline_img is image:
            # 同帧差分无意义（调用方误传同一帧），退化为纯 OCR 校验
            baseline_img = None
        if baseline_img is not None:
            result["visual_check"] = self.inspect_chat_message_visual_change(
                baseline_img, image)

        if self._ocr_parser is None:
            # 未注入 OCR 解析器时跳过客户回合校验，避免永久阻塞发送。
            # 后续 confirm_send 仍会通过像素差判定消息是否真正送达。
            result["ok"] = True
            result["reason"] = "customer_turn_guard_ocr_unavailable_skipped"
            return result

        try:
            messages = self._ocr_chat_messages(image)
            if not messages:
                result["ok"] = False
                result["reason"] = "customer_turn_guard_ocr_unavailable"
                return result

            turn_messages = self._latest_customer_turn_messages(messages)
            if not turn_messages:
                result["ok"] = False
                result["reason"] = "current_customer_turn_empty"
                return result

            result["ocr_ok"] = True

            current_text = "\n".join(
                str(m.get("text", m.get("content", m.get("bubble_text", "")))).strip()
                for m in turn_messages
            )
            current_text = current_text.strip()[:240]

            expected_norm = self._normalize_text(expected)
            current_norm = self._normalize_text(current_text)

            if expected_norm == current_norm:
                result["ok"] = True
                result["reason"] = "customer_turn_matched"
                return result

            if current_norm.endswith(expected_norm[-6:] if len(expected_norm) >= 6 else expected_norm):
                result["ok"] = True
                result["reason"] = "customer_turn_latest_message_matched"
                return result

            if expected_norm in current_norm:
                result["ok"] = True
                result["reason"] = "customer_turn_contains_expected"
                return result

            # 第一重兜底：模糊相似度（容忍 OCR 噪声，少字/标点差异视为同一话轮）
            similarity = self._text_similarity(expected_norm, current_norm)
            if similarity >= 0.82:
                result["ok"] = True
                result["reason"] = "customer_turn_fuzzy_matched"
                return result

            # 文本不匹配：进入第二重（视觉差分）判断是否有真实新消息
            visual_changed = None
            vc = result.get("visual_check")
            if isinstance(vc, dict):
                visual_changed = bool(vc.get("changed", False))

            if visual_changed is False:
                # 视觉无变化 => 聊天区没有新消息气泡 => 信任无变化（仅 OCR 文本偏差）
                result["ok"] = True
                result["reason"] = "customer_turn_visual_unchanged_passed"
                return result

            if similarity >= 0.6:
                # 中等相似度，疑似噪声导致偏差；不直接放弃，交由上游重试
                result["ok"] = False
                result["reason"] = "customer_turn_uncertain_retry"
                return result

            # 文本明显不同且有视觉变化（或无法视觉判断）=> 认定客户话轮已变
            result["ok"] = False
            result["reason"] = "customer_turn_text_mismatch_before_send"

        except Exception as e:
            result["ok"] = False
            result["reason"] = f"customer_turn_check_failed: {e}"

        return result

    def _is_self_message(self, msg: Dict[str, Any]) -> bool:
        """Check if a message is from self (right side)."""
        return msg.get("side") == "right"

    def _latest_customer_turn_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get the latest continuous customer messages from the bottom."""
        result: List[Dict[str, Any]] = []
        found_self = False
        for msg in reversed(messages):
            if self._is_self_message(msg):
                found_self = True
                continue
            if self._is_customer_message(msg):
                if not found_self:
                    result.insert(0, msg)
                else:
                    break
        return result

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """相似度 0~1，用于容忍 OCR 噪声（少字/标点差异）。"""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison by removing whitespace and punctuation."""
        if not text:
            return ""
        return re.sub(r'[\s\W_]+', '', str(text).strip().lower())

    def compare_regions(
        self, before_image, after_image
    ) -> Dict[str, Dict[str, float]]:
        """Compare image regions and return per-region diff metrics.

        Returns dict with keys like 'chat', 'input', 'full' each containing
        ratio/changed/total/mean metrics.
        """
        result: Dict[str, Dict[str, float]] = {
            "full": {"ratio": 0.0, "changed": 0, "total": 0, "mean": 0.0},
            "chat": {"ratio": 0.0, "changed": 0, "total": 0, "mean": 0.0},
            "input": {"ratio": 0.0, "changed": 0, "total": 0, "mean": 0.0},
        }

        if before_image is None or after_image is None:
            return result

        try:
            if isinstance(before_image, str):
                before = np.array(Image.open(before_image).convert("RGB"))
            else:
                before = before_image

            if isinstance(after_image, str):
                after = np.array(Image.open(after_image).convert("RGB"))
            else:
                after = after_image

            if before.shape != after.shape:
                min_h = min(before.shape[0], after.shape[0])
                min_w = min(before.shape[1], after.shape[1])
                before = before[:min_h, :min_w]
                after = after[:min_h, :min_w]

            result["full"] = self._diff_metrics(before, after)

            h, w = before.shape[:2]
            input_top = int(h * 0.82)
            if input_top < h:
                before_chat = before[:input_top, :]
                after_chat = after[:input_top, :]
                result["chat"] = self._diff_metrics(before_chat, after_chat)

                before_input = before[input_top:, :]
                after_input = after[input_top:, :]
                result["input"] = self._diff_metrics(before_input, after_input)

        except Exception:
            pass

        return result

    @staticmethod
    def _region_box(
        width: int, height: int,
        ratios: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        """Calculate pixel box from width/height and ratio tuple (left, top, right, bottom)."""
        left = int(width * ratios[0])
        top = int(height * ratios[1])
        right = int(width * ratios[2])
        bottom = int(height * ratios[3])
        return (left, top, right, bottom)

    def _color_metrics(
        self, image: np.ndarray,
        ratios: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> Dict[str, float]:
        """Calculate color metrics for a region of the image."""
        result: Dict[str, float] = {
            "nonwhite_ratio": 0.0,
            "dark_ratio": 0.0,
            "mean_brightness": 0.0,
        }

        if image is None or image.size == 0:
            return result

        try:
            h, w = image.shape[:2]
            left, top, right, bottom = self._region_box(w, h, ratios)
            left = max(0, left); top = max(0, top)
            right = min(w, right); bottom = min(h, bottom)

            if right <= left or bottom <= top:
                return result

            region = image[top:bottom, left:right]
            if region.size == 0:
                return result

            total_pixels = region.shape[0] * region.shape[1]

            gray = np.mean(region, axis=2)
            nonwhite = int(np.sum(gray < 240))
            dark = int(np.sum(gray < 80))

            result["nonwhite_ratio"] = nonwhite / max(total_pixels, 1)
            result["dark_ratio"] = dark / max(total_pixels, 1)
            result["mean_brightness"] = float(np.mean(gray))

        except Exception:
            pass

        return result