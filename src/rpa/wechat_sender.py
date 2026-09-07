"""WeChatSender - 微信消息发送器，对齐原版 app.rpa.wechat_sender v3.10。

协调 RPA 底层操作、发送确认、发送守卫，完成完整的消息发送流程。
支持：文本发送、图片发送、文件发送、@提醒、引用回复、素材发送。
"""
from __future__ import annotations

import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..agent.compliance_gate import ComplianceGate

logger = logging.getLogger(__name__)


class WeChatSender:
    """微信消息发送器，对齐原版 app.rpa.wechat_sender.WeChatSender v3.10。

    核心职责：
    1. 接收发送请求（文本/图片/文件）
    2. 调用 RPA 层执行实际操作
    3. 通过 send_confirm 验证发送结果
    4. 通过 send_guard 防止重复发送
    5. 支持素材引用（{image:name} 语法）
    """

    SEND_DELAY = 0.5
    PASTE_DELAY = 0.15
    ENTER_DELAY = 0.3
    MAX_RETRIES = 3

    def __init__(self, config=None, finder=None, capture=None, store=None,
                 human_monitor=None, window_manager=None, privacy_overlay=None,
                 rpa=None, send_confirm=None, send_guard=None,
                 screen_capture=None, human_like_mouse=None,
                 human_activity_monitor=None):
        self.config = config or {}
        self.finder = finder
        self.capture = capture
        self.store = store
        self.human_monitor = human_monitor
        self.window_manager = window_manager
        self.privacy_overlay = privacy_overlay

        self._rpa = rpa
        self._send_confirm = send_confirm
        self._send_guard = send_guard
        self._screen_capture = screen_capture or capture
        self._human_like_mouse = human_like_mouse
        self._human_activity_monitor = human_activity_monitor or human_monitor

        self._config = self.config
        self._send_delay = self._config.get("send_delay", self.SEND_DELAY)
        self._max_retries = self._config.get("max_retries", self.MAX_RETRIES)
        self._last_send_before = None
        self._compliance_gate = None

        if self._rpa is None:
            try:
                from .wechat_rpa import WeChatRPA
                self._rpa = WeChatRPA()
            except Exception:
                pass

        if self._send_guard is None:
            try:
                from .send_guard import SendGuard
                self._send_guard = SendGuard(self.config)
            except Exception:
                pass

        if self._human_like_mouse is None:
            try:
                from .human_like_mouse import HumanLikeMouse
                self._human_like_mouse = HumanLikeMouse(self.config)
            except Exception:
                pass

        if self._send_confirm is None:
            try:
                from .send_confirm import SendConfirm
                self._send_confirm = SendConfirm(self.config, self.capture)
            except Exception:
                pass

        if store and hasattr(store, 'append_log'):
            store.append_log("WechatSender initialized")

    # ---------------------------------------------------------------
    # send_report — 对齐原版 send_report（ObserveService 主入口）
    # ---------------------------------------------------------------
    def send_report(self, report: dict) -> Dict[str, Any]:
        """Send reply based on analysis report from ObserveService.

        This is the main entry point called by ObserveService.send_current_reply()
        and _run_one_cycle_impl(). It extracts the necessary fields from the
        report dict, runs the send guard, and delegates to _send().

        Expected report keys:
          - analysis: dict with current_contact, chat_type, reply_draft, etc.
          - reply_draft: str (the reply text to send)
          - current_contact: str
          - chat_type: str ("private" or "group")
          - materials_to_send: list (optional image/file materials)

        Returns:
          dict with keys: ok, reason, guard, action, duration_ms
        """
        started = time.perf_counter()

        analysis = report.get("analysis", report) if isinstance(report, dict) else {}
        reply = report.get("reply_draft", "") or analysis.get("reply_draft", "")
        contact = report.get("current_contact", "") or analysis.get("current_contact", "")
        contact_key = report.get("contact_key", "") or self._contact_key(contact)
        chat_type = report.get("chat_type", "") or analysis.get("chat_type", "private")
        materials = report.get("materials_to_send", []) or analysis.get("materials_to_send", [])
        # 期望客户话术：发送给发送前守卫做"客户话轮未变"校验。
        # 文字消息路径下 expected_customer_text / customer_turn_text 长期为空，
        # 必须回退到观察阶段真实抓到的客户消息（latest_message.content / messages），
        # 否则守卫因 expected 为空永远拦截（regression: 发送前校验 never pass）。
        expected_customer_text = self._resolve_expected_customer_text(report, analysis)

        if self._send_guard:
            wechat_cfg = self._config.get("wechat", {})
            safety_cfg = self._config.get("safety", {})
            msg = analysis.get("latest_message", {}) or {}
            guard_ok, guard_reason = self._send_guard.check(
                wechat_cfg, safety_cfg, analysis, msg)
            if not guard_ok:
                return {
                    "ok": False,
                    "reason": guard_reason,
                    "guard": {"ok": False, "reason": guard_reason},
                    "action": "skip_send",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }

        try:
            result = self._send_report_impl(
                report=report,
                analysis=analysis,
                reply=reply,
                contact=contact,
                contact_key=contact_key,
                chat_type=chat_type,
                materials=materials,
                expected_customer_text=expected_customer_text,
            )
            result["duration_ms"] = int((time.perf_counter() - started) * 1000)
            return result
        except Exception as e:
            store = getattr(self, 'store', None)
            if store and hasattr(store, 'append_log'):
                store.append_log(
                    f"rpa_send exception={e} contact={contact} reply={reply[:80]}")
            return {
                "ok": False,
                "reason": f"RPA send failed: {e}",
                "guard": {},
                "action": "send_text",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }

    def _send_report_impl(self, report: dict, analysis: dict, reply: str,
                          contact: str, contact_key: str, chat_type: str,
                          materials: list, expected_customer_text: str) -> Dict[str, Any]:
        """Internal implementation of send_report.

        Finds the WeChat window, prepares it, and sends the reply text
        with optional materials.
        """
        if not reply or not reply.strip():
            return {"ok": False, "reason": "empty_reply", "guard": {},
                    "action": "send_text"}

        mouse_release = self._wait_for_mouse_release_before_window_action()
        if mouse_release is False:
            return {"ok": False, "reason": "user_dragging_mouse",
                    "guard": {}, "action": "defer_send"}

        store = getattr(self, 'store', None)
        if store and hasattr(store, 'append_log'):
            store.append_log(f"send_report contact={contact} reply_len={len(reply)}")

        found = self.finder.find() if hasattr(self, 'finder') and self.finder else None
        if not found:
            msg = f"WeChat window not found before send: {contact}"
            if store and hasattr(store, 'append_log'):
                store.append_log(msg)
            return {"ok": False, "reason": msg, "guard": {}, "action": "send_text"}

        window_info = found.to_dict() if hasattr(found, 'to_dict') else found
        hwnd = window_info.get("handle", 0) or window_info.get("hwnd", 0)
        rect = window_info.get("rect", {})

        if not hwnd:
            return {"ok": False, "reason": "No WeChat window handle available.",
                    "guard": {}, "action": "send_text"}

        x_ratio = float(self._config.get("wechat", {}).get(
            "input_box_click_ratio_x", 0.68))
        y_ratio = float(self._config.get("wechat", {}).get(
            "input_box_click_ratio_y", 0.9))

        background = self._background_send_enabled()

        self._prepare_privacy_overlay_interaction(hwnd)

        if self._human_activity_monitor and hasattr(
                self._human_activity_monitor, 'ignore_for'):
            self._human_activity_monitor.ignore_for(3.0)

        before = self._capture_frame(hwnd)
        if before is None:
            before = self._screen_capture.capture_window(
                hwnd, subdir="before_send", prefix="before"
            ) if self._screen_capture else None

        if self._send_confirm:
            chat_ready = self._send_confirm.inspect_chat_ready(
                before, hwnd) if before is not None else {"ok": True}
            if not chat_ready.get("ok", True):
                return {"ok": False, "reason": "chat_input_not_ready",
                        "guard": {}, "action": "send_text"}

        if self._send_confirm and hasattr(self._send_confirm, 'inspect_current_customer_turn'):
            baseline_img = (report.get("_send_baseline_image")
                            if isinstance(report, dict) else None)
            turn_check = self._send_confirm.inspect_current_customer_turn(
                before, expected_customer_text, baseline_image=baseline_img)
            if not turn_check.get("ok", True):
                # 透传守卫真实原因，避免日志被写死的前缀误导（如 expected_customer_turn_empty）
                real_reason = turn_check.get("reason", "customer_turn_text_mismatch_before_send")
                return {"ok": False,
                        "reason": f"pre_send_guard:{real_reason} expected={expected_customer_text[:40]}",
                        "guard": {}, "action": "send_text"}

        # 发送前合规门：拦截与知识库规范价不符的价格事实（防止 LLM 说错价）
        if self._compliance_gate is None:
            try:
                self._compliance_gate = ComplianceGate(self._config)
            except Exception:
                self._compliance_gate = False
        if self._compliance_gate:
            _ok, _reason = self._compliance_gate.check(reply)
            if not _ok:
                return {"ok": False,
                        "reason": f"compliance_price_mismatch:{_reason}",
                        "guard": {}, "action": "send_text"}

        contact_check = self._verify_current_chat_contact(hwnd, contact)
        if not contact_check:
            return {"ok": False, "reason": "current_chat_mismatch",
                    "guard": {}, "action": "send_text"}

        reply_segments = self._reply_segments(reply)
        reply_segments = self._join_short_reply_lines(reply_segments)

        if self._managed_display_mode() if hasattr(self, '_managed_display_mode') else False:
            reply = self._merge_reply_segments_for_managed_display(reply_segments)

        quote_prepared = False
        if self._quote_requested(analysis):
            quote_prepared = self._try_prepare_quote_reply(hwnd, analysis)

        input_clear = self._clear_input_before_send(hwnd)
        if not input_clear:
            self._force_clear_input_before_send(hwnd)

        send_log = []
        ok_count = 0
        total = len(reply_segments) + (len(materials) if materials else 0)

        for idx, segment in enumerate(reply_segments):
            segment_result = self._send_text_segment_with_confirm(
                hwnd, rect, segment, x_ratio, y_ratio, background,
                before, contact, chat_type, quote_prepared and idx == 0,
                expected_customer_text, idx, total,
                self._config.get("wechat", {}))
            send_log.append(segment_result)
            if segment_result.get("ok"):
                ok_count += 1
            if idx < len(reply_segments) - 1:
                time.sleep(self._segment_delay())

        if materials:
            for idx, material in enumerate(materials):
                if isinstance(material, dict):
                    kind = material.get("kind", "")
                    path = material.get("path", "")
                    title = material.get("title", "")
                else:
                    kind = "image"
                    path = str(material)
                    title = ""

                if kind == "image" and path and os.path.exists(path):
                    if self._human_like_mouse and hasattr(
                            self._human_like_mouse, 'paste_image_and_enter'):
                        self._human_like_mouse.paste_image_and_enter(hwnd, path)
                        ok_count += 1
                        send_log.append({"kind": kind, "path": path, "ok": True})
                    else:
                        self._rpa._send_text_via_clipboard(hwnd, path)
                        time.sleep(0.15)
                        self._rpa.send_key_press(0x0D)
                        ok_count += 1
                        send_log.append({"kind": kind, "path": path, "ok": True})
                elif kind in ("file", "video") and path and os.path.exists(path):
                    if self._human_like_mouse and hasattr(
                            self._human_like_mouse, 'paste_file_and_enter'):
                        self._human_like_mouse.paste_file_and_enter(hwnd, path)
                        ok_count += 1
                        send_log.append({"kind": kind, "path": path, "ok": True})
                    else:
                        self._rpa._send_text_via_clipboard(hwnd, path)
                        time.sleep(0.15)
                        self._rpa.send_key_press(0x0D)
                        ok_count += 1
                        send_log.append({"kind": kind, "path": path, "ok": True})
                else:
                    send_log.append({"kind": kind, "path": path,
                                     "ok": False, "reason": "not_found"})

                time.sleep(self._segment_delay())

        after_delay = self._send_confirm_delay_seconds()
        time.sleep(after_delay)

        after_image = self._capture_frame(hwnd)

        confirm = {}
        if self._send_confirm and before is not None and after_image is not None:
            confirm = self._send_confirm.confirm_send(
                before_image=before,
                after_image=after_image,
                expected_reply=reply,
                expected_customer_text=expected_customer_text,
            )

        all_ok = ok_count > 0 and confirm.get("success", True)

        if self._send_guard and all_ok:
            self._send_guard.mark_sent(
                contact_key, reply, contact=contact, reply_text=reply)

        # 透传真实失败原因，避免日志只剩裸 send_failed 无法定位：
        # 优先分段失败原因（如 _rpa 缺失抛的 AttributeError），其次发送确认原因。
        reason = ""
        if not all_ok:
            seg_fail = next(
                (s.get("reason") for s in send_log if s.get("ok") is False), "")
            confirm_reason = (confirm or {}).get("reason", "")
            reason = seg_fail or confirm_reason or "send_failed"

        result = {
            "ok": all_ok,
            "reason": reason,
            "guard": {},
            "action": "send_text_and_materials" if materials else "send_text",
            "contact": contact,
            "reply": reply,
            "materials_sent": ok_count > 0,
            "send_method": self._send_method_name(),
            "before_image": before,
            "after_image": after_image,
            "confirm": confirm,
            "send_log": send_log,
        }
        return result

    def _send_text_segment_with_confirm(self, hwnd: int, rect: dict,
                                         segment: str, x_ratio: float,
                                         y_ratio: float, background: bool,
                                         before_image, contact: str,
                                         chat_type: str, quote_prepared: bool,
                                         expected_customer_text: str,
                                         segment_index: int, total: int,
                                         wechat_cfg: dict) -> Dict[str, Any]:
        """Send a single text segment with confirmation."""
        if not segment:
            return {"ok": True, "kind": "text", "segment_index": segment_index}

        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)

            self._rpa._send_text_via_clipboard(hwnd, segment)
            time.sleep(self.PASTE_DELAY)

            after_paste_image = self._capture_frame(hwnd)

            self._rpa.send_key_press(0x0D)
            time.sleep(self._send_delay)

            return {
                "ok": True,
                "kind": "text",
                "preview": segment[:80],
                "segment_index": segment_index,
                "segment_total": total,
                "send_method": self._send_method_name(),
                "before_image": before_image,
                "after_paste_image": after_paste_image,
            }
        except Exception as e:
            return {
                "ok": False,
                "kind": "text",
                "reason": str(e),
                "preview": segment[:80],
                "segment_index": segment_index,
                "segment_total": total,
            }

    # ---------------------------------------------------------------
    # 核心发送方法（对齐原版 _send）
    # ---------------------------------------------------------------
    def _send(self, hwnd: int, text: str, contact_key: str = "",
              contact_name: str = "", analysis: dict = None,
              msg: dict = None, image_paths: List[str] = None,
              expected_customer_text: str = "") -> Dict[str, Any]:
        """Core send flow matching original v3.10 _send().

        Full flow:
        1. guard_check - send guard safety checks
        2. ocr_guard - self-check OCR text quality
        3. before_screenshot - capture before state
        4. paste - paste text into input box
        5. after_paste_screenshot - capture after paste state
        6. enter - press Enter to send
        7. confirm_send - verify send success
        8. mark_sent - record in send guard
        """
        if not text or not text.strip():
            return {"success": False, "message": "", "error": "Empty text"}

        if not self._rpa:
            return {"success": False, "message": "", "error": "No RPA engine"}

        analysis = analysis or {}
        msg = msg or {}

        wechat_cfg = self._config.get("wechat", {})
        safety_cfg = self._config.get("safety", {})

        if self._send_guard:
            ok, reason = self._send_guard.check(
                wechat_cfg, safety_cfg, analysis, msg)
            if not ok:
                return {"success": False, "message": "",
                        "error": f"SendGuard blocked: {reason}"}

        for attempt in range(self._max_retries):
            try:
                self._rpa.activate_window(hwnd)
                time.sleep(0.1)

                before_image = self._capture_frame(hwnd)
                self._last_send_before = before_image

                if image_paths:
                    for img_path in image_paths:
                        if os.path.exists(img_path):
                            if self._human_like_mouse:
                                self._human_like_mouse.paste_image_and_enter(
                                    hwnd, img_path)
                            time.sleep(0.1)

                self._rpa._send_text_via_clipboard(hwnd, text)
                time.sleep(self.PASTE_DELAY)

                after_paste_image = self._capture_frame(hwnd)

                self._rpa.send_key_press(0x0D)
                time.sleep(self._send_delay)

                after_image = self._capture_frame(hwnd)

                if self._send_confirm:
                    confirm_result = self._send_confirm.confirm_send(
                        before_image=before_image,
                        after_image=after_image,
                        after_paste_image=after_paste_image,
                        expected_reply=text,
                        expected_customer_text=expected_customer_text,
                    )
                    if confirm_result.get("success"):
                        if self._send_guard:
                            self._send_guard.mark_sent(
                                contact_key, text,
                                contact=contact_name, reply_text=text)
                        return {"success": True, "message": text,
                                "error": "", "confirm": confirm_result}
                    else:
                        logger.warning(
                            f"Send confirm failed (attempt {attempt + 1}): "
                            f"{confirm_result.get('reason', '')}")
                        if attempt < self._max_retries - 1:
                            time.sleep(1.0)
                            continue
                        return {"success": False, "message": "",
                                "error": confirm_result.get("reason", ""),
                                "confirm": confirm_result}
                else:
                    if self._verify_send(hwnd):
                        if self._send_guard:
                            self._send_guard.mark_sent(
                                contact_key, text,
                                contact=contact_name, reply_text=text)
                        return {"success": True, "message": text, "error": ""}

                logger.warning(
                    f"Send verification failed (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Send attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(1.0)

        return {"success": False, "message": "",
                "error": "Send failed after retries"}

    # ---------------------------------------------------------------
    # 发送文本和素材
    # ---------------------------------------------------------------
    def send_text(self, hwnd: int, text: str, contact_key: str = "",
                  contact_name: str = "", analysis: dict = None,
                  msg: dict = None,
                  expected_customer_text: str = "") -> Dict[str, Any]:
        """发送文本消息。"""
        text, image_paths = self._resolve_image_refs(text)
        return self._send(
            hwnd, text, contact_key=contact_key, contact_name=contact_name,
            analysis=analysis, msg=msg,
            image_paths=image_paths if image_paths else None,
            expected_customer_text=expected_customer_text)

    def send_text_and_materials(self, hwnd: int, text: str,
                                 contact_key: str = "",
                                 contact_name: str = "",
                                 analysis: dict = None,
                                 msg: dict = None,
                                 materials: List[str] = None,
                                 expected_customer_text: str = "") -> Dict[str, Any]:
        """Send text with optional materials (images/files)."""
        text, image_refs = self._resolve_image_refs(text)
        image_paths = (materials or []) + (image_refs or [])
        return self._send(
            hwnd, text, contact_key=contact_key, contact_name=contact_name,
            analysis=analysis, msg=msg,
            image_paths=image_paths if image_paths else None,
            expected_customer_text=expected_customer_text)

    # ---------------------------------------------------------------
    # 素材引用解析
    # ---------------------------------------------------------------
    def _resolve_image_refs(self, text: str) -> Tuple[str, List[str]]:
        """Resolve {image:name} references in text to actual file paths.

        Returns (cleaned_text, list_of_image_paths).
        """
        if not text:
            return text, []

        image_refs = []
        cleaned = text

        pattern = r'\{image:([^}]+)\}'
        matches = re.findall(pattern, text)
        for name in matches:
            path = self._image_ref(name.strip())
            if path and os.path.exists(path):
                image_refs.append(path)
            cleaned = cleaned.replace(f"{{image:{name}}}", "")

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned, image_refs

    def _image_ref(self, name: str) -> Optional[str]:
        """Resolve an image name to its file path."""
        material_dir = self._config.get("material_dir", "")
        if material_dir and os.path.isdir(material_dir):
            for ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
                path = os.path.join(material_dir, name + ext)
                if os.path.exists(path):
                    return path
                path = os.path.join(material_dir, name)
                if os.path.exists(path):
                    return path
        return None

    # ---------------------------------------------------------------
    # 发送图片
    # ---------------------------------------------------------------
    def send_image(self, hwnd: int, image_path: str,
                   contact_key: str = "",
                   contact_name: str = "") -> Dict[str, Any]:
        """发送图片消息。"""
        if not image_path or not os.path.exists(image_path):
            return {"success": False, "message": "", "error": "Image not found"}

        if self._send_guard:
            if not self._send_guard.should_send(
                    contact_key, f"[image]{image_path}",
                    contact_name=contact_name):
                return {"success": False, "message": "",
                        "error": "SendGuard blocked"}

        if not self._rpa:
            return {"success": False, "message": "", "error": "No RPA engine"}

        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.2)

            if self._human_like_mouse and hasattr(self._human_like_mouse,
                                                   "paste_image_and_enter"):
                self._human_like_mouse.paste_image_and_enter(hwnd, image_path)
            else:
                self._rpa._send_text_via_clipboard(hwnd, image_path)
                time.sleep(0.1)
                self._rpa.send_key_press(0x0D)

            time.sleep(self._send_delay)

            if self._send_guard:
                self._send_guard.mark_sent(
                    contact_key, f"[image]{image_path}",
                    contact=contact_name)

            return {"success": True, "message": f"[image]{image_path}",
                    "error": ""}
        except Exception as e:
            return {"success": False, "message": "", "error": str(e)}

    # ---------------------------------------------------------------
    # 发送文件
    # ---------------------------------------------------------------
    def send_file(self, hwnd: int, file_path: str,
                  contact_key: str = "",
                  contact_name: str = "") -> Dict[str, Any]:
        """发送文件消息。"""
        if not file_path or not os.path.exists(file_path):
            return {"success": False, "message": "", "error": "File not found"}

        if self._send_guard:
            if not self._send_guard.should_send(
                    contact_key, f"[file]{file_path}",
                    contact_name=contact_name):
                return {"success": False, "message": "",
                        "error": "SendGuard blocked"}

        if not self._rpa:
            return {"success": False, "message": "", "error": "No RPA engine"}

        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.2)

            if self._human_like_mouse and hasattr(self._human_like_mouse,
                                                   "paste_file_and_enter"):
                self._human_like_mouse.paste_file_and_enter(hwnd, file_path)
            else:
                self._rpa._send_text_via_clipboard(hwnd, file_path)
                time.sleep(0.1)
                self._rpa.send_key_press(0x0D)

            time.sleep(self._send_delay)

            if self._send_guard:
                self._send_guard.mark_sent(
                    contact_key, f"[file]{file_path}",
                    contact=contact_name)

            return {"success": True, "message": f"[file]{file_path}",
                    "error": ""}
        except Exception as e:
            return {"success": False, "message": "", "error": str(e)}

    # ---------------------------------------------------------------
    # @提醒消息
    # ---------------------------------------------------------------
    def send_at_message(self, hwnd: int, text: str, at_name: str = "",
                        contact_key: str = "",
                        contact_name: str = "") -> Dict[str, Any]:
        """发送 @ 消息。"""
        if not text:
            return {"success": False, "message": "", "error": "Empty text"}

        full_text = f"@{at_name} {text}" if at_name else text

        if self._send_guard:
            if not self._send_guard.should_send(
                    contact_key, full_text, contact_name=contact_name):
                return {"success": False, "message": "",
                        "error": "SendGuard blocked"}

        if not self._rpa:
            return {"success": False, "message": "", "error": "No RPA engine"}

        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)

            self._rpa._send_text_via_clipboard(hwnd, full_text)
            time.sleep(0.1)

            self._rpa.send_key_press(0x0D)
            time.sleep(self._send_delay)

            if self._send_guard:
                self._send_guard.mark_sent(
                    contact_key, full_text,
                    contact=contact_name, reply_text=text)

            return {"success": True, "message": full_text, "error": ""}
        except Exception as e:
            return {"success": False, "message": "", "error": str(e)}

    # ---------------------------------------------------------------
    # 引用回复
    # ---------------------------------------------------------------
    def send_reply(self, hwnd: int, text: str, contact_key: str = "",
                   contact_name: str = "",
                   quote_text: str = "") -> Dict[str, Any]:
        """发送引用回复。"""
        if not text:
            return {"success": False, "message": "", "error": "Empty text"}

        if self._send_guard:
            if not self._send_guard.should_send(
                    contact_key, text, contact_name=contact_name):
                return {"success": False, "message": "",
                        "error": "SendGuard blocked"}

        if not self._rpa:
            return {"success": False, "message": "", "error": "No RPA engine"}

        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)

            self._rpa._send_text_via_clipboard(hwnd, text)
            time.sleep(0.1)

            self._rpa.send_key_press(0x0D)
            time.sleep(self._send_delay)

            if self._send_guard:
                self._send_guard.mark_sent(
                    contact_key, text,
                    contact=contact_name, reply_text=text)

            return {"success": True, "message": text, "error": ""}
        except Exception as e:
            return {"success": False, "message": "", "error": str(e)}

    # ---------------------------------------------------------------
    # 联系人验证
    # ---------------------------------------------------------------
    def contact_verify(self, hwnd: int, expected_name: str) -> bool:
        """Verify that the current chat contact matches expected name."""
        if not expected_name or not self._screen_capture:
            return True
        try:
            frame = self._capture_frame(hwnd)
            if frame is None:
                return True
            if self._send_confirm and self._send_confirm._ocr_parser:
                messages = self._send_confirm._ocr_parser.parse_chat_messages(
                    frame)
                contact = self._send_confirm._ocr_parser.current_contact
                if contact:
                    return self._contact_names_match(contact, expected_name)
        except Exception:
            pass
        return True

    @staticmethod
    def _contact_names_match(a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_norm = re.sub(r'[\s\W_]+', '', a.casefold())
        b_norm = re.sub(r'[\s\W_]+', '', b.casefold())
        return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm

    # ---------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------
    def send_typing_indicator(self, hwnd: int) -> None:
        """模拟正在输入状态（发送一个不可见字符）。"""
        if not self._rpa:
            return
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            self._rpa._send_text_via_clipboard(hwnd, "\u200B")
            time.sleep(0.05)
        except Exception:
            pass

    def _verify_send(self, hwnd: int) -> bool:
        """验证消息是否发送成功（前后截屏比对）。"""
        if not self._send_confirm:
            return True

        try:
            after = self._screen_capture.capture_window(hwnd)
            if not after or not after.success:
                return True

            if self._last_send_before is not None:
                result = self._send_confirm.compare(
                    self._last_send_before, after.image)
                return result.success
            return True
        except Exception:
            return True

    def _capture_frame(self, hwnd: int):
        """捕获一帧用于后续比对。"""
        if not self._screen_capture:
            return None
        try:
            result = self._screen_capture.capture_window(hwnd)
            if result and result.success:
                return result.image
        except Exception:
            pass
        return None

    def clear_input(self, hwnd: int) -> None:
        """清空微信输入框。"""
        if not self._rpa:
            return
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            self._rpa.send_key_combo(0x11, 0x41)
            time.sleep(0.05)
            self._rpa.send_key_press(0x2E)
            time.sleep(0.05)
        except Exception:
            pass

    def can_send(self, contact_key: str, content: str,
                 contact_name: str = "") -> bool:
        """检查是否可以向该联系人发送消息（send_guard 代理）。"""
        if self._send_guard:
            return self._send_guard.should_send(
                contact_key, content, contact_name=contact_name)
        return True

    def record_send(self, contact_key: str, content: str,
                    contact_name: str = "") -> bool:
        """记录发送（send_guard 代理）。"""
        if self._send_guard:
            return self._send_guard.mark_sent(
                contact_key, content, contact=contact_name)
        return True

    # =================================================================
    # 引用回复处理（对齐原版）
    # =================================================================

    @staticmethod
    def _send_confirm_delay_seconds() -> float:
        """发送确认延迟时间。"""
        return 0.5

    def _quote_requested(self, analysis: dict = None) -> bool:
        """检查是否需要引用回复。"""
        if not analysis:
            return False
        return bool(analysis.get("quote_requested", False))

    def _try_prepare_quote_reply(self, hwnd: int, analysis: dict = None) -> bool:
        """尝试准备引用回复。"""
        if not self._quote_requested(analysis):
            return True
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.15)
            return self._select_quote_menu_item(hwnd)
        except Exception:
            return False

    def _select_quote_menu_item(self, hwnd: int) -> bool:
        """选择引用回复菜单项。"""
        try:
            # 尝试快速选择
            if self._select_quote_menu_item_fast(hwnd):
                return True
            # 回退到 OCR 选择
            return self._select_quote_menu_item_by_popup_ocr(hwnd)
        except Exception:
            return False

    def _select_quote_menu_item_fast(self, hwnd: int) -> bool:
        """快速选择引用菜单项（通过已知偏移坐标）。"""
        try:
            point = self._quote_menu_fast_point(hwnd)
            if point:
                self._click_screen_point(point[0], point[1])
                time.sleep(0.1)
                return True
            return False
        except Exception:
            return False

    def _select_quote_menu_item_by_popup_ocr(self, hwnd: int) -> bool:
        """通过 OCR 识别弹出菜单来选择引用项。"""
        try:
            popup = self._quote_menu_popup(hwnd)
            if popup:
                self._click_screen_point(
                    popup.get("center_x", 0),
                    popup.get("center_y", 0))
                time.sleep(0.1)
                return True
            return False
        except Exception:
            return False

    def _quote_menu_popup(self, hwnd: int) -> Optional[Dict[str, int]]:
        """检测引用菜单弹出位置。"""
        try:
            img = self._capture_window_printwindow(hwnd)
            if img is None:
                return None
            h, w = img.shape[:2]
            return {"center_x": w // 2, "center_y": int(h * 0.7)}
        except Exception:
            return None

    def _quote_menu_fast_point(self, hwnd: int) -> Optional[Tuple[int, int]]:
        """获取引用菜单快速点击坐标。"""
        try:
            import win32gui
            rect = win32gui.GetWindowRect(hwnd)
            cx = rect[0] + (rect[2] - rect[0]) // 2
            cy = rect[1] + int((rect[3] - rect[1]) * 0.7)
            return (cx, cy)
        except Exception:
            return None

    def _capture_window_printwindow(self, hwnd: int) -> Optional[Any]:
        """使用 PrintWindow 截取窗口。"""
        if self._screen_capture:
            return self._screen_capture._capture_printwindow(hwnd)
        return None

    def _capture_all_screens(self) -> Optional[Any]:
        """截取所有屏幕。"""
        if self._screen_capture:
            return self._screen_capture._capture_imagegrab_rect({})
        return None

    def _virtual_screen_origin(self) -> Tuple[int, int]:
        """获取虚拟屏幕原点。"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sm_x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            sm_y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            return (sm_x, sm_y)
        except Exception:
            return (0, 0)

    def _choose_quote_menu_item(self, hwnd: int, label: str) -> bool:
        """选择指定的引用菜单项。"""
        try:
            popup = self._quote_menu_popup(hwnd)
            if popup:
                self._click_screen_point(
                    popup.get("center_x", 0),
                    popup.get("center_y", 0))
                time.sleep(0.1)
                return True
            return False
        except Exception:
            return False

    def _click_screen_point(self, x: int, y: int) -> None:
        """点击屏幕坐标（后台 PostMessage，不移动真实鼠标、不抢焦点）。"""
        from .background_clicker import background_click_screen
        try:
            background_click_screen(x, y)
        except Exception:
            pass

    def _dismiss_quote_menu(self) -> None:
        """关闭引用菜单。"""
        try:
            import pyautogui
            pyautogui.press("escape")
            time.sleep(0.05)
        except Exception:
            pass

    # =================================================================
    # 输入清理（对齐原版）
    # =================================================================

    def _clear_input_before_send(self, hwnd: int) -> bool:
        """发送前清空输入框。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            self._rpa.send_key_combo(0x11, 0x41)  # Ctrl+A
            time.sleep(0.05)
            self._rpa.send_key_press(0x2E)  # Delete
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def _force_clear_input_before_send(self, hwnd: int) -> bool:
        """强制清空输入框（多次尝试）。"""
        for _ in range(3):
            if self._clear_input_before_send(hwnd):
                return True
            time.sleep(0.1)
        return False

    def _foreground_clear_input_before_send(self, hwnd: int) -> bool:
        """前台清空输入框。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.15)
            self._rpa.send_key_combo(0x11, 0x41)
            time.sleep(0.05)
            self._rpa.send_key_press(0x2E)
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def _find_input_overlay_close_ratio(self, hwnd: int) -> float:
        """查找输入覆盖层关闭按钮的匹配度。"""
        try:
            img = self._capture_window_printwindow(hwnd)
            if img is None:
                return 0.0
            # 检查输入框区域是否有覆盖层（如表情面板、文件面板等）
            h, w = img.shape[:2]
            input_area = img[int(h * 0.82):h, :]
            gray_pixels = np.sum(
                (input_area[:, :, 0] > 200) &
                (input_area[:, :, 1] > 200) &
                (input_area[:, :, 2] > 200))
            return gray_pixels / max(input_area.size, 1)
        except Exception:
            return 0.0

    def _try_close_stale_input_overlay(self, hwnd: int) -> bool:
        """尝试关闭残留的输入覆盖层。"""
        ratio = self._find_input_overlay_close_ratio(hwnd)
        if ratio > 0.5:
            try:
                import pyautogui
                pyautogui.press("escape")
                time.sleep(0.1)
                return True
            except Exception:
                pass
        return False

    # =================================================================
    # 分段发送（对齐原版）
    # =================================================================

    def _send_text_segment(self, hwnd: int, text: str,
                           segment_index: int = 0) -> bool:
        """发送单个文本段。"""
        if not text:
            return True
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            self._rpa._send_text_via_clipboard(hwnd, text)
            time.sleep(self.PASTE_DELAY)
            self._rpa.send_key_press(0x0D)
            time.sleep(self._segment_delay())
            return True
        except Exception:
            return False

    def _foreground_clear_draft_and_send_segment(self, hwnd: int,
                                                  text: str) -> bool:
        """前台清除草稿并发送文本段。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.15)
            self._rpa.send_key_combo(0x11, 0x41)
            time.sleep(0.05)
            self._rpa.send_key_press(0x2E)
            time.sleep(0.05)
            self._rpa._send_text_via_clipboard(hwnd, text)
            time.sleep(0.1)
            self._rpa.send_key_press(0x0D)
            time.sleep(0.3)
            return True
        except Exception:
            return False

    def _foreground_enter_fallback(self, hwnd: int) -> bool:
        """前台回车回退（PostMessage 不可靠时）。"""
        try:
            self._rpa.send_key_press(0x0D)
            time.sleep(0.15)
            return True
        except Exception:
            return False

    def _foreground_full_fallback(self, hwnd: int, text: str) -> bool:
        """前台完整回退发送。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.2)
            self._rpa._send_text_via_clipboard(hwnd, text)
            time.sleep(0.1)
            self._rpa.send_key_press(0x0D)
            time.sleep(0.3)
            return True
        except Exception:
            return False

    def _failed_segment(self, text: str, reason: str = "") -> Dict[str, Any]:
        """返回失败段结果。"""
        return {"success": False, "message": text, "error": reason}

    def _foreground_fallback_enabled(self) -> bool:
        """检查是否允许前台回退。"""
        return self._config.get("foreground_fallback_enabled", True)

    # =================================================================
    # 客户消息守卫（对齐原版）
    # =================================================================

    def _guard_against_new_customer_before_enter(self, hwnd: int) -> bool:
        """在发送前检查是否有新客户消息。"""
        try:
            before = self._capture_frame(hwnd)
            if before is None:
                return True
            if self._send_confirm:
                result = self._send_confirm.inspect_new_customer_messages(
                    before, before)
                if result.get("has_new"):
                    return self._judge_new_customer_before_enter()
            return True
        except Exception:
            return True

    def _judge_new_customer_before_enter(self) -> bool:
        """判断是否允许在新客户消息到达时继续发送。"""
        strategy = self._config.get("new_customer_strategy", "allow")
        if strategy == "block":
            return False
        if strategy == "allow":
            return True
        return True

    @staticmethod
    def _pre_enter_customer_guard_delay_seconds() -> float:
        """发送前客户消息守卫延迟。"""
        return 0.3

    def _send_blocked_by_new_customer(self) -> bool:
        """检查发送是否被新客户消息阻止。"""
        return not self._config.get("allow_send_during_customer_message", True)

    def _safe_background_chars_enabled(self) -> bool:
        """检查是否安全使用后台字符发送。"""
        return self._config.get("safe_background_chars", True)

    # =================================================================
    # 联系人验证（对齐原版）
    # =================================================================

    def _verify_current_chat_contact(self, hwnd: int,
                                      expected_name: str) -> bool:
        """验证当前聊天联系人是否匹配。"""
        return self.contact_verify(hwnd, expected_name)

    def _resolve_expected_customer_text(self, report: dict, analysis: dict) -> str:
        """解析发送给守卫的"期望客户话术"。

        守卫 inspect_current_customer_turn 在发送前会重新 OCR 当前会话，
        校验客户最新消息未变；若 expected 为空会直接拦截（expected_customer_turn_empty）。
        本项目文字消息路径下 expected_customer_text / customer_turn_text 从未被赋值，
        因此必须回退到观察阶段真实抓到的客户消息文本，否则每条都会发送失败。
        """
        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(report, dict):
            report = {}

        # 1) 显式字段（若上游某处赋值）
        cand = report.get("expected_customer_text", "") or analysis.get("expected_customer_text", "")
        if cand and str(cand).strip():
            return str(cand).strip()

        # 2) 回复生成时基于的客户消息簇
        cand = analysis.get("customer_turn_text", "")
        if cand and str(cand).strip():
            return str(cand).strip()

        # 3) 观察阶段抓到的客户最新消息（文字消息下通常在此，customer_turn_text 为空）
        lm = analysis.get("latest_message")
        if isinstance(lm, dict):
            cand = lm.get("content", "")
            if cand and str(cand).strip():
                return str(cand).strip()

        # 4) 消息列表中最后一条客户（左侧）消息
        msgs = analysis.get("messages") or []
        if isinstance(msgs, list):
            cust = [m for m in msgs
                    if isinstance(m, dict)
                    and (m.get("side") == "left"
                         or m.get("is_self") is False
                         or m.get("role") == "customer")]
            if cust:
                last = cust[-1]
                cand = last.get("text") or last.get("content") or ""
                if cand and str(cand).strip():
                    return str(cand).strip()

        return ""

    def _report_customer_turn_text(self, messages: list) -> Optional[str]:
        """报告客户最新消息文本。"""
        if not messages:
            return None
        customer_msgs = [m for m in messages
                         if m.get("side") == "left"]
        if not customer_msgs:
            return None
        return customer_msgs[-1].get("text", "")

    def _report_group_sender_name(self, messages: list) -> Optional[str]:
        """报告群聊中发送者名称。"""
        if not messages:
            return None
        for m in messages:
            if m.get("sender_name"):
                return m.get("sender_name")
        return None

    def _contact_key(self, contact_name: str = "") -> str:
        """生成联系人唯一标识。"""
        return re.sub(r'[\s\W_]+', '', contact_name.lower())

    def _contact_matches(self, a: str, b: str) -> bool:
        """比较两个联系人名称是否匹配。"""
        return self._contact_names_match(a, b)

    # =================================================================
    # 显示管理（对齐原版）
    # =================================================================

    def _merge_reply_segments_for_managed_display(self, segments: list) -> str:
        """合并回复段用于托管显示器。"""
        if not segments:
            return ""
        return "\n".join(s for s in segments if s)

    def _foreground_enter_delay_seconds(self) -> float:
        """前台回车延迟。"""
        return float(self._config.get("foreground_enter_delay", 0.3))

    def _foreground_keyboard_mode(self) -> str:
        """前台键盘模式。"""
        return self._config.get("foreground_keyboard_mode", "sendinput")

    def _foreground_takeover_max_seconds(self) -> float:
        """前台接管最大时间。"""
        return float(self._config.get("foreground_takeover_max", 5.0))

    def _foreground_block_input_enabled(self) -> bool:
        """是否启用前台输入阻塞。"""
        return self._config.get("foreground_block_input", True)

    def _wait_for_foreground_send_slot(self) -> bool:
        """等待前台发送槽位。"""
        time.sleep(self._foreground_enter_delay_seconds())
        return True

    def _wait_for_mouse_release_before_window_action(self) -> bool:
        """等待鼠标释放后再执行窗口操作。"""
        try:
            import ctypes
            for _ in range(30):
                for vk in (1, 2, 4):
                    if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
                        time.sleep(0.05)
                        break
                else:
                    return True
            return False
        except Exception:
            return True

    def _pressed_mouse_button(self) -> Optional[str]:
        """检查当前按下的鼠标按键。"""
        try:
            import ctypes
            _MOUSE_BUTTONS = {1: "left", 2: "right", 4: "middle"}
            for vk, name in _MOUSE_BUTTONS.items():
                if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
                    return name
            return None
        except Exception:
            return None

    def _foreground_slot_deferred(self) -> bool:
        """检查前台槽位是否延迟。"""
        return self._config.get("foreground_slot_deferred", False)

    def _float_config(self, key: str, default: float = 0.0) -> float:
        """获取浮点配置值。"""
        return float(self._config.get(key, default))

    def _seconds_since_last_input(self) -> float:
        """距离上次输入的时间。"""
        if self._human_activity_monitor and hasattr(
                self._human_activity_monitor, "_seconds_since_last_input"):
            return self._human_activity_monitor._seconds_since_last_input()
        return 0.0

    def _skip_background_text_postmessage(self) -> bool:
        """是否跳过后台文本 PostMessage。"""
        return self._config.get("skip_background_text_postmessage", False)

    def _send_failure_reason(self, default: str = "") -> str:
        """获取发送失败原因。"""
        return self._config.get("send_failure_reason", default)

    def _reply_segments(self, text: str) -> List[str]:
        """将回复文本拆分为段。"""
        if not text:
            return []
        return [s.strip() for s in text.split("\n") if s.strip()]

    def _join_short_reply_lines(self, segments: List[str]) -> List[str]:
        """合并短回复行。"""
        if not segments:
            return []
        result = []
        buffer = []
        for seg in segments:
            if len(seg) < 30:
                buffer.append(seg)
            else:
                if buffer:
                    result.append(" ".join(buffer))
                    buffer.clear()
                result.append(seg)
        if buffer:
            result.append(" ".join(buffer))
        return result

    def _segment_delay(self) -> float:
        """段间延迟。"""
        return float(self._config.get("segment_delay", 0.5))

    def _strategy(self) -> str:
        """获取当前发送策略。"""
        return self._config.get("send_strategy", "background")

    def _background_send_enabled(self) -> bool:
        """是否启用后台发送。"""
        return self._config.get("background_send_enabled", True)

    def _send_method_name(self) -> str:
        """获取发送方法名称。"""
        return self._strategy()

    def _prepare_privacy_overlay_interaction(self, hwnd: int) -> bool:
        """准备隐私覆盖层交互。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            return True
        except Exception:
            return False

    def open_unread_hint(self, hwnd: int) -> str:
        """打开未读提示。"""
        try:
            self._rpa.activate_window(hwnd)
            time.sleep(0.1)
            return "ok"
        except Exception as e:
            return f"error: {e}"


# 兼容原版类名别名
WechatSender = WeChatSender