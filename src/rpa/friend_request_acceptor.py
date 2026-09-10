"""FriendRequestAcceptor - 微信好友申请自动通过，对齐原版 app.rpa.friend_request_acceptor v3.10。

完整流程：
1. has_friend_request_badge() - 检查通讯录红点
2. _open_new_friends() - 通讯录 → 新的朋友
3. _click_first_waiting_request() - 点击第一个等待验证
4. _click_go_verify() - 前往验证
5. _confirm_verify_popup() - 确认通过验证弹窗
6. _apply_friend_permission() - 设置朋友圈权限
7. _send_welcome_if_enabled() - 发送欢迎语
8. _restore_chats_tab() - 回到聊天标签页
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np


def _textbox_center(item) -> tuple:
    box = getattr(item, "box", None)
    if box is not None and hasattr(box, "size") and box.size >= 8:
        pts = box.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return 0.0, 0.0


class FriendRequestAcceptor:
    """自动通过微信好友请求，使用截图/OCR 确认。

    对齐原版 app.rpa.friend_request_acceptor.FriendRequestAcceptor v3.10。
    WeChat 4.x 不暴露 UIA 控件，因此使用截图引导的 RPA。
    """

    CONTACTS_TAB_X = 38
    CONTACTS_TAB_Y = 158
    CHATS_TAB_X = 112
    CHATS_TAB_Y = 285
    NEW_FRIENDS_RIGHT_X = 136
    NEW_FRIENDS_Y = 184
    REQUEST_ROW_X = 0.62
    GO_VERIFY_FALLBACK_X_RATIO = 0.62
    GO_VERIFY_FALLBACK_Y_RATIO = 0.545
    POPUP_WAIT_SECONDS = 2.2

    ACCEPT_KEYWORDS = ["通过验证", "接受", "添加到通讯录", "发消息"]

    def __init__(self, config=None, window_manager=None, capture=None,
                 logger=None, red_dot=None):
        from ..desktop.wechat_window_manager import WeChatWindowManager
        from ..capture.screen_capture import ScreenCapture
        from ..ocr.ocr_pool import get_vision_ocr
        from ..rpa.red_dot_detector import RedDotDetector
        from ..rpa.human_like_mouse import HumanLikeMouse

        self.config = config or {}
        self.logger = logger or (lambda msg: None)
        self.window_manager = window_manager or WeChatWindowManager()
        self.capture = capture or ScreenCapture()
        # OCR 走进程级单例池（原先这里自建一份 OCREngine，与主感知/红点检测
        # 各一份互不共享 → RapidOCR 模型被重复加载，见 src/ocr/ocr_pool.py）。
        # 取不到时保持 None：下面两处 OCR 调用点都包在 try/except 内，
        # 会安全降级（返回空结果/False），不会 AttributeError 崩链路。
        self.ocr = get_vision_ocr()
        # red_dot 允许外部注入：observe_service 会传入它那一个已绑好
        # config/screen_capture/logger 的 detector，避免再 new 一份。
        self.red_dot = red_dot or RedDotDetector()
        self.mouse = HumanLikeMouse()
        self._allow_foreground_fallback = True

    # ---------------------------------------------------------------
    # 配置
    # ---------------------------------------------------------------
    @property
    def friend_config(self) -> dict:
        if isinstance(self.config, dict):
            return self.config.get("friend_requests", {})
        return {}

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------
    def accept_all(self, window_info: Optional[dict[str, Any]] = None,
                   only_when_badge: bool = True,
                   max_requests: int = 3,
                   restore_chat: bool = True,
                   allow_foreground_fallback: bool = True) -> dict[str, Any]:
        """接受所有好友请求。

        Args:
            window_info: 微信窗口信息
            only_when_badge: 仅在检测到红点时执行
            max_requests: 最大接受数量
            restore_chat: 完成后是否回到聊天标签
            allow_foreground_fallback: 是否允许前台窗口回退

        Returns:
            dict with keys: ok, accepted_count, reason, screenshots
        """
        previous_foreground_policy = self._allow_foreground_fallback
        self._allow_foreground_fallback = bool(allow_foreground_fallback)

        info = dict(window_info or {})
        rect = info.get("rect", {})
        hwnd = info.get("handle", 0) or info.get("hwnd", 0)
        screenshots = []
        badge = None
        probe_path = ""
        probe_items = []
        page_state = "unknown"

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            if restore_chat:
                self._restore_chats_tab(hwnd, rect)
            if not self._is_quiet_foreground_reason(str(payload.get("reason", ""))):
                self._allow_foreground_fallback = previous_foreground_policy
            return payload

        if not hwnd:
            return finish({"ok": False, "accepted_count": 0, "reason": "wechat_window_not_found"})

        if only_when_badge:
            badge = self.has_friend_request_badge(
                rect, hwnd, "friend_no_badge_page_probe")
            if not badge.get("found", False):
                # —— 无红点：到此为止，绝不再往下走 ——
                # 历史 bug：此前这里只给 page_state 赋值就继续往下，流程照样
                # 落到 _open_new_friends()，于是每 check_interval_seconds(默认30s)
                # 无差别打开一次「新的朋友」页面再切回 —— 这就是窗口抖动的来源；
                # 且 page_state=unknown 时还会 _focus() 抢走前台焦点打断用户。
                # 未切换过页面，因此不需要 _restore_chats_tab（其坐标同样硬编码、
                # 在当前分辨率下并不可靠，能不点就不点）。
                self._allow_foreground_fallback = previous_foreground_policy
                self._log(
                    f"friend_request_no_badge_skip reason={badge.get('reason', '')}")
                return {"ok": True, "accepted_count": 0,
                        "reason": "no_friend_request_badge",
                        "screenshots": screenshots}
            page_state = "new_friends"

        opened = self._open_new_friends(rect, hwnd, screenshots)
        if not opened.get("ok"):
            return finish({"ok": False, "accepted_count": 0,
                           "reason": "open_new_friends_failed",
                           "screenshots": screenshots,
                           "state": opened.get("state", "unknown")})

        accepted = 0
        reasons = []

        for _ in range(max(1, max_requests)):
            if page_state == "new_friends":
                reopened = self._open_new_friends(rect, hwnd, screenshots)
                if not reopened.get("ok"):
                    self._log("new_friends_reopen_failed")
                    break
                time.sleep(0.35)

            request = self._click_first_waiting_request(rect, hwnd, screenshots)
            if not request.get("ok"):
                reasons.append(request.get("reason", "no_waiting_request"))
                break

            verify = self._click_go_verify(rect, hwnd, screenshots)
            if not verify.get("ok"):
                reasons.append(verify.get("reason", "go_verify_not_found"))
                break

            confirmed = self._confirm_verify_popup(screenshots)
            if not confirmed.get("ok"):
                reasons.append(confirmed.get("reason", "verify_popup_confirm_failed"))
                break

            welcome = self._send_welcome_if_enabled(rect, hwnd, screenshots)
            if not welcome.get("ok"):
                self._log(f"friend_request_welcome_failed reason={welcome.get('reason', '')}")

            accepted += 1
            time.sleep(0.8)

        self._log(f"friend_request_accept ok={accepted > 0} accepted={accepted} reason={'; '.join(reasons)}")
        return finish({
            "ok": accepted > 0,
            "accepted_count": accepted,
            "reason": "; ".join(reasons) if reasons else "friend_request_accepted",
            "screenshots": screenshots,
        })

    # ---------------------------------------------------------------
    # 红点检测
    # ---------------------------------------------------------------
    def has_friend_request_badge(self, rect: dict[str, int], hwnd: int,
                                  prefix: str = "") -> dict[str, Any]:
        """检测通讯录红点（好友申请入口的徽章）。

        检测区域按「截图真实像素」的比例计算。
        历史 bug：原为硬编码绝对坐标 x=35~62 / y=138~170，只在当初校准的那一个
        窗口尺寸下有效；换分辨率后区域整体错位，表现为永远检测不到红点。
        比例可在 config.yaml 的 friend_requests 段微调，日志会打印实际使用的
        像素区域，便于对照真机截图校准。
        """
        img = None
        try:
            img = self.red_dot._take_screenshot_with_retry(hwnd, rect)
        except Exception:
            return {"found": False, "reason": "screenshot_failed"}

        if img is None:
            return {"found": False, "reason": "screenshot_failed"}

        # 以截图像素为准 —— _find_dots 就工作在这个坐标系上，不能用 rect 推算
        try:
            h, w = img.shape[:2]
        except Exception:
            return {"found": False, "reason": "screenshot_shape_unreadable"}

        cfg = self.friend_config
        try:
            x_start = int(w * float(cfg.get("badge_x_start_ratio", 0.020)))
            x_end = int(w * float(cfg.get("badge_x_end_ratio", 0.048)))
            y_start = int(h * float(cfg.get("badge_y_start_ratio", 0.125)))
            y_end = int(h * float(cfg.get("badge_y_end_ratio", 0.180)))
        except Exception:
            x_start, x_end, y_start, y_end = int(w * 0.020), int(w * 0.048), \
                int(h * 0.125), int(h * 0.180)

        try:
            dots = self.red_dot._find_dots(img, x_start=x_start, x_end=x_end,
                                           y_start=y_start, y_end=y_end)
        except Exception:
            dots = None

        count = len(dots) if dots else 0
        # 打印实际区域，便于拿真机截图对账微调（红点在通讯录图标右上角）
        self._log(
            f"friend_badge_probe img={w}x{h} "
            f"region=({x_start},{y_start})-({x_end},{y_end}) dots={count}")

        if not dots:
            return {"found": False, "reason": "friend_request_badge_not_found", "badge": None}

        dots.sort(key=lambda item: float(getattr(item, "score", 0)), reverse=True)
        return {"found": True, "badge": dots[0] if dots else None,
                "reason": "friend_request_badge_found"}

    # ---------------------------------------------------------------
    # 导航到「新的朋友」
    # ---------------------------------------------------------------
    def _open_new_friends(self, rect: dict[str, int], hwnd: int,
                          screenshots: list) -> dict[str, Any]:
        """打开「新的朋友」页面。"""
        path, items = self._capture_ocr(rect, hwnd, "friend_new_friends_before_open", screenshots)
        state = self._friend_page_state(items)
        if state in ("detail_verify", "new_friends"):
            return {"ok": True, "reason": f"already_on_{state}", "state": state}

        target = self._find_text(items, ("新的朋友",))
        if target:
            y = int(getattr(target, "cy", 0))
            x = max(120, int(getattr(target, "cx", 0)))
            self._click_window(rect, x, y, hwnd)
            time.sleep(0.65)
            after_path, after_items = self._capture_ocr(
                rect, hwnd, "friend_new_friends_after_open", screenshots)
            new_state = self._friend_page_state(after_items)
            if new_state in ("new_friends", "detail_verify"):
                return {"ok": True, "reason": "new_friends_opened", "screenshot": after_path}

        return {"ok": False, "reason": "new_friends_page_not_opened"}

    # ---------------------------------------------------------------
    # 点击第一个等待验证
    # ---------------------------------------------------------------
    def _click_first_waiting_request(self, rect: dict[str, int], hwnd: int,
                                      screenshots: list) -> dict[str, Any]:
        """点击第一个等待验证的好友请求。"""
        _, items = self._capture_ocr(rect, hwnd, "friend_waiting_request_scan", screenshots)
        width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))

        waiting = []
        for item in items:
            text = str(getattr(item, "text", "") or "").replace(" ", "")
            if any(kw in text for kw in ("等待验证", "待验证")):
                if width > 0 and getattr(item, "left", 0) < 825:
                    waiting.append(item)

        if not waiting:
            return {"ok": False, "reason": "no_waiting_request"}

        waiting.sort(key=lambda item: getattr(item, "top", 0))
        item = waiting[0]
        row_y = int(getattr(item, "cy", 0))
        source = "ocr_waiting"

        self._click_window(rect, int(self.REQUEST_ROW_X * width), row_y, hwnd)
        time.sleep(0.18)

        after_path, after_items = self._capture_ocr(
            rect, hwnd, "friend_waiting_request_after_click", screenshots)
        new_state = self._friend_page_state(after_items)

        if new_state == "detail_verify":
            return {"ok": True, "reason": "waiting_request_clicked",
                    "source": source, "click_method": "ocr"}

        if self._allow_foreground_fallback:
            self._foreground_click_window(hwnd, int(self.REQUEST_ROW_X * width), row_y)
            time.sleep(0.75)
            fallback_after, fallback_items = self._capture_ocr(
                rect, hwnd, "friend_waiting_request_after_fallback", screenshots)
            fallback_state = self._friend_page_state(fallback_items)
            if fallback_state == "detail_verify":
                return {"ok": True, "reason": "waiting_request_clicked",
                        "source": "_foreground", "click_method": "_foreground"}

        return {"ok": False, "reason": "waiting_request_card_not_opened"}

    # ---------------------------------------------------------------
    # 点击「前往验证」
    # ---------------------------------------------------------------
    def _click_go_verify(self, rect: dict[str, int], hwnd: int,
                          screenshots: list) -> dict[str, Any]:
        """点击「前往验证」按钮。"""
        _, items = self._capture_ocr(rect, hwnd, "friend_detail_before_verify", screenshots)

        if not self._request_allowed(items):
            return {"ok": False, "reason": "friend_request_keyword_not_matched"}

        target = self._find_verify_button(items, rect)
        if target:
            local_x = int(getattr(target, "cx", 0))
            local_y = int(getattr(target, "cy", 0))
            source = "ocr"
        else:
            width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
            height = int(max(rect.get("bottom", 0) - rect.get("top", 0), 0))
            local_x = int(self.GO_VERIFY_FALLBACK_X_RATIO * width)
            local_y = int(self.GO_VERIFY_FALLBACK_Y_RATIO * height)
            source = "geometry_fallback"

        if not local_x or not local_y:
            return {"ok": False, "reason": "verify_button_not_found"}

        self._click_window(rect, local_x, local_y, hwnd)
        self._log(f"friend_request_go_verify_click source={source} method=ocr local=({local_x},{local_y})")

        popup = self._wait_for_popup("通过朋友验证", self.POPUP_WAIT_SECONDS)
        if popup:
            return {"ok": True, "reason": "verify_button_clicked", "click_method": "ocr"}

        if self._allow_foreground_fallback:
            self._foreground_click_window(hwnd, local_x, local_y)
            self._log(f"friend_request_go_verify_foreground_fallback method={source}")
            popup = self._wait_for_popup("通过朋友验证", self.POPUP_WAIT_SECONDS + 1.2)
            if popup:
                return {"ok": True, "reason": "verify_button_clicked", "click_method": "_foreground"}

        return {"ok": False, "reason": "verify_popup_not_opened_after_click"}

    # ---------------------------------------------------------------
    # 确认通过验证弹窗
    # ---------------------------------------------------------------
    def _confirm_verify_popup(self, screenshots: list) -> dict[str, Any]:
        """确认「通过朋友验证」弹窗。"""
        popup = self._find_popup("通过朋友验证")
        if not popup:
            return {"ok": False, "reason": "verify_popup_not_found"}

        hwnd = popup.get("handle", 0)
        rect = popup.get("rect", {})

        self._focus(hwnd)
        _, items = self._capture_ocr(rect, hwnd, "friend_verify_popup_before_confirm", screenshots)

        self._apply_friend_permission(hwnd, rect, items)

        target = self._find_text(items, ("确定",))
        if target:
            self._click_window(rect, int(getattr(target, "cx", 0)),
                               int(getattr(target, "cy", 0)), hwnd)
            self._log(f"friend_request_confirm_popup_click method=ocr")
            closed = self._wait_until_popup_closed("通过朋友验证", 2.4)
            if closed:
                return {"ok": True, "reason": "verify_popup_confirmed"}
        else:
            if self._allow_foreground_fallback:
                self._foreground_click_window(hwnd, int(0.365 * rect.get("width", 0)),
                                              int(0.929 * rect.get("height", 0)))
                self._log(f"friend_request_confirm_popup_fallback method=_foreground")
                time.sleep(3.0)
                closed = self._wait_until_popup_closed("通过朋友验证", 0.6)
                if closed:
                    return {"ok": True, "reason": "verify_popup_confirmed"}

        return {"ok": False, "reason": "verify_popup_still_visible"}

    # ---------------------------------------------------------------
    # 请求是否允许
    # ---------------------------------------------------------------
    def _request_allowed(self, items: list) -> bool:
        """检查好友请求是否允许通过。"""
        mode = self.friend_config.get("accept_mode", "accept_all")
        if mode == "accept_all":
            return True

        keywords = self.friend_config.get("keyword_rules", [])
        if isinstance(keywords, str):
            keywords = [p.strip() for p in keywords.replace("，", ",").replace("\n", ",").replace("；", ";").split(",") if p.strip()]

        normalized = [str(item.text).strip() for item in items]
        return any(kw in item for item in normalized for kw in keywords)

    # ---------------------------------------------------------------
    # 设置朋友圈权限
    # ---------------------------------------------------------------
    def _apply_friend_permission(self, hwnd: int, rect: dict[str, int],
                                  items: list) -> None:
        """设置好友权限（如仅聊天）。"""
        permission = self.friend_config.get("friend_permission", "default")
        if permission == "default":
            return

        if permission == "chat_only":
            target = self._find_text(items, ("仅聊天",))
            if target:
                self._click_window(rect, int(getattr(target, "cx", 0)),
                                   int(getattr(target, "cy", 0)), hwnd)
                time.sleep(0.25)

    # ---------------------------------------------------------------
    # 发送欢迎语
    # ---------------------------------------------------------------
    def _send_welcome_if_enabled(self, rect: dict[str, int], hwnd: int,
                                  screenshots: list) -> dict[str, Any]:
        """发送欢迎语。"""
        cfg = self.friend_config
        if not cfg.get("send_welcome", True):
            return {"ok": False, "reason": "welcome_disabled"}

        text = str(cfg.get("welcome_text", "")).strip()
        if not text:
            return {"ok": False, "reason": "welcome_text_empty"}

        _, items = self._capture_ocr(rect, hwnd, "friend_after_accept_before_welcome", screenshots)

        target = self._find_text(items, ("发消息",))
        if not target:
            return {"ok": False, "reason": "send_message_button_not_found"}

        if self._allow_foreground_fallback:
            self._click_window(rect, int(getattr(target, "cx", 0)),
                               int(getattr(target, "cy", 0)), hwnd)
            time.sleep(0.7)

            width = int(max(rect.get("right", 0) - rect.get("left", 0), 0))
            height = int(max(rect.get("bottom", 0) - rect.get("top", 0), 0))
            self.mouse.foreground_paste_enter_once(
                hwnd, text, enter_delay=0.15)
            return {"ok": True, "reason": "welcome_sent"}

        return {"ok": False, "reason": "welcome_skipped_user_active"}

    # ---------------------------------------------------------------
    # 恢复到聊天标签页
    # ---------------------------------------------------------------
    def _restore_chats_tab(self, hwnd: int, rect: dict[str, int]) -> None:
        """恢复到聊天标签页。"""
        try:
            self._click_window(rect, self.CHATS_TAB_X, self.CHATS_TAB_Y, hwnd)
            time.sleep(0.15)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # OCR 截图
    # ---------------------------------------------------------------
    def _capture_ocr(self, rect: dict[str, int], hwnd: int, prefix: str,
                     screenshots: list) -> tuple[str, list]:
        """截图并 OCR。"""
        path = ""
        try:
            result = self.capture.capture_window(hwnd=hwnd, subdir="friend_requests", prefix=prefix)
            if result and result.success and result.image is not None:
                path = getattr(result, "path", getattr(result, "image_path", ""))
                if screenshots is not None:
                    screenshots.append(path)
                items = self.ocr.run(result.image, text_score=0.35)
                return path, list(items.items) if hasattr(items, "items") else list(items)
        except Exception:
            pass
        return path, []

    # ---------------------------------------------------------------
    # 文本查找
    # ---------------------------------------------------------------
    @staticmethod
    def _find_text(items: list, keywords: tuple[str, ...]) -> Optional[Any]:
        """在 OCR 结果中查找指定文本。"""
        for item in items:
            text = str(getattr(item, "text", "") or "")
            if any(kw in text for kw in keywords):
                return item
        return None

    # ---------------------------------------------------------------
    # 验证按钮查找
    # ---------------------------------------------------------------
    @staticmethod
    def _find_verify_button(items: list, rect: dict[str, int]) -> Optional[Any]:
        """查找「前往验证」按钮。"""
        for item in items:
            text = str(getattr(item, "text", "") or "")
            if text in ("前往验证", "通过验证"):
                return item
        return None

    # ---------------------------------------------------------------
    # 页面状态识别
    # ---------------------------------------------------------------
    @staticmethod
    def _friend_page_state(items: list) -> str:
        """识别当前好友页面状态。"""
        texts = [str(getattr(item, "text", "") or "") for item in items]
        if any("前往验证" in t for t in texts):
            return "detail_verify"
        if any("等待验证" in t for t in texts):
            return "new_friends"
        if any("新的朋友" in t for t in texts):
            return "new_friends"
        if any("通讯录" in t for t in texts):
            return "contacts"
        return "unknown"

    @staticmethod
    def _looks_like_friend_detail(items: list) -> bool:
        texts = [str(getattr(item, "text", "") or "") for item in items]
        return any("前往验证" in t for t in texts) or any("通过验证" in t for t in texts)

    # ---------------------------------------------------------------
    # 弹窗检测
    # ---------------------------------------------------------------
    def _find_popup(self, title_keyword: str) -> Optional[dict[str, Any]]:
        """查找指定标题的弹窗。"""
        try:
            import win32gui
            def enum_callback(hwnd, result):
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if title_keyword in text:
                        rect = win32gui.GetWindowRect(hwnd)
                        result.append({
                            "handle": hwnd,
                            "rect": {
                                "left": rect[0], "top": rect[1],
                                "right": rect[2], "bottom": rect[3],
                                "width": rect[2] - rect[0],
                                "height": rect[3] - rect[1],
                            }
                        })
            result = []
            win32gui.EnumWindows(enum_callback, result)
            return result[0] if result else None
        except Exception:
            return None

    def _wait_for_popup(self, title_keyword: str, timeout: float) -> Optional[dict[str, Any]]:
        """等待弹窗出现。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            popup = self._find_popup(title_keyword)
            if popup:
                return popup
            time.sleep(0.15)
        return None

    def _wait_until_popup_closed(self, title_keyword: str, timeout: float) -> bool:
        """等待弹窗关闭。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            popup = self._find_popup(title_keyword)
            if not popup:
                return True
            time.sleep(0.2)
        return False

    # ---------------------------------------------------------------
    # 窗口操作
    # ---------------------------------------------------------------
    @staticmethod
    def _rect(hwnd: int) -> Optional[dict[str, int]]:
        """获取窗口矩形。"""
        try:
            import win32gui
            r = win32gui.GetWindowRect(hwnd)
            return {
                "left": r[0], "top": r[1], "right": r[2], "bottom": r[3],
                "width": r[2] - r[0], "height": r[3] - r[1],
            }
        except Exception:
            return None

    @staticmethod
    def _focus(hwnd: int) -> None:
        """聚焦窗口。"""
        try:
            import win32gui
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _real_click(self, hwnd: int, local_x: int, local_y: int) -> bool:
        """真实鼠标点击（窗口相对坐标），统一经 window_manager.foreground_click 派发。

        微信好友申请页（接受/前往验证/确定 等按钮）同样是 CEF 自绘，后台合成
        PostMessage 点击不会被 Chromium 命中派发 —— 必须走真实鼠标输入，与
        red_dot_detector._do_click 主链路保持一致。极少数情况下真实点击失败
        时，降级回原后台合成点击。

        坐标约定：local_x/local_y 是相对窗口左上角(含边框)的偏移，与 PrintWindow
        截图像素同源。
        """
        wm = self.window_manager
        if wm is not None and hasattr(wm, "foreground_click"):
            try:
                return bool(wm.foreground_click(int(local_x), int(local_y), hwnd=hwnd))
            except Exception:
                pass
        # 降级：后台合成点击
        try:
            from .background_clicker import background_click_screen
            rect = wm.get_rect(hwnd) if wm is not None else None
            if isinstance(rect, dict):
                left = rect.get("left", 0); top = rect.get("top", 0)
            elif rect is not None:
                left = getattr(rect, "x", 0); top = getattr(rect, "y", 0)
            else:
                left = 0; top = 0
            background_click_screen(int(left) + int(local_x), int(top) + int(local_y))
        except Exception:
            pass
        return False

    def _click_window(self, rect: dict[str, int], local_x: int, local_y: int,
                      hwnd: int) -> None:
        """点击窗口内坐标（真实鼠标，走 foreground_click 主链路）。"""
        self._real_click(hwnd, local_x, local_y)

    def _click_screen(self, rect: dict[str, int], rel_x: int, rel_y: int) -> None:
        """点击屏幕相对坐标（真实鼠标）。"""
        hwnd = self.window_manager.find_wechat_window() if self.window_manager else 0
        self._real_click(hwnd, rel_x, rel_y)

    def _foreground_click_window(self, hwnd: int, x: int, y: int) -> bool:
        """真实鼠标点击（统一走 foreground_click 主链路）。"""
        return self._real_click(hwnd, x, y)

    def _background_click(self, x: int, y: int) -> None:
        """后台点击（窗口相对坐标）。"""
        hwnd = self.window_manager.find_wechat_window() if self.window_manager else 0
        self._real_click(hwnd, x, y)

    def _click_abs(self, x: int, y: int) -> None:
        """绝对坐标点击（转换后走 foreground_click）。"""
        hwnd = self.window_manager.find_wechat_window() if self.window_manager else 0
        rect = self.window_manager.get_rect(hwnd) if self.window_manager else None
        if isinstance(rect, dict):
            left = rect.get("left", 0); top = rect.get("top", 0)
        elif rect is not None:
            left = getattr(rect, "x", 0); top = getattr(rect, "y", 0)
        else:
            left = 0; top = 0
        self._real_click(hwnd, int(x) - int(left), int(y) - int(top))

    def _click_point(self, x: int, y: int) -> None:
        """点击指定坐标（窗口相对，走真实鼠标）。"""
        hwnd = self.window_manager.find_wechat_window() if self.window_manager else 0
        self._real_click(hwnd, x, y)

    # ---------------------------------------------------------------
    # 日志
    # ---------------------------------------------------------------
    def _log(self, message: str) -> None:
        try:
            if self.logger:
                self.logger(message)
        except Exception:
            pass

    @staticmethod
    def _is_quiet_foreground_reason(reason: str) -> bool:
        return "quiet" in reason.lower() or "deferred" in reason.lower()

    # =================================================================
    # 兼容旧接口
    # =================================================================
    def accept(self, hwnd: int = None) -> bool:
        """Try to accept friend requests on the current WeChat window. (compat)"""
        if hwnd is None:
            hwnd = self.window_manager.find_wechat_window()
        if not hwnd:
            return False

        frame = self.capture.capture_window(hwnd=hwnd)
        if not frame or not frame.success or frame.image is None:
            return False

        try:
            ocr_result = self.ocr.run(frame.image)
        except Exception:
            return False

        if not ocr_result:
            return False

        items = list(ocr_result.items) if hasattr(ocr_result, "items") else list(ocr_result)
        for item in items:
            text = str(getattr(item, "text", "") or "")
            if any(kw in text for kw in self.ACCEPT_KEYWORDS):
                cx, cy = _textbox_center(item)
                if cx and cy:
                    self._click_point(int(cx), int(cy))
                    time.sleep(0.5)
                    return True

        return False

    def accept_all_compat(self, max_attempts: int = 10) -> int:
        """Accept all friend requests, up to max_attempts. (compat)"""
        accepted = 0
        for _ in range(max_attempts):
            if self.accept():
                accepted += 1
                time.sleep(1.0)
            else:
                break
        return accepted