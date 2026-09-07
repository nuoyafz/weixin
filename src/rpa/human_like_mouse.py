"""HumanLikeMouse v2 - extends V37 with foreground management, user-drag detection,
image/file paste, typewriter mode, and mouse guard.

Aligned with original app.rpa.human_like_mouse v3.10:
  - force_foreground_window / restore_previous_foreground / is_target_foreground
  - _foreground_mouse_allowed / _focus_window_for_privacy_overlay
  - _foreground_fallback_click / user_dragging_mouse
  - click_absolute_win32 / send_ctrl_v_win32
  - mouse_guard / defer_mouse / foreground_lock
  - paste_image_and_enter / paste_file_and_enter / typewriter
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import io
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import win32con
import win32gui
import win32process


class HumanLikeMouse:
    """Human-like mouse movement with foreground management and paste support.

    Aligned with original v3.10. Uses Windows native clipboard for image/file
    paste via win32clipboard, and full foreground window management.
    """

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040

    def __init__(self, config=None, window_manager=None):
        self._config = config or {}
        self._window_manager = window_manager
        self._user32 = ctypes.windll.user32
        self.step_interval = (0.008, 0.02)
        self.jitter = 3.0
        self.min_steps = 12
        self._foreground_lock = False
        self._mouse_guard_enabled = False
        self._mouse_guard_defer_ms = 500
        self._last_foreground_hwnd = None
        self._last_mouse_move_time = 0.0
        self._foreground_fallback_enabled = True

    def get_position(self):
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        self._user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def _set_position(self, x, y):
        self._user32.SetCursorPos(int(round(x)), int(round(y)))

    def _mouse_event(self, flags):
        self._user32.mouse_event(flags, 0, 0, 0, 0)

    @staticmethod
    def _smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _build_points(self, sx, sy, ex, ey):
        dx, dy = ex - sx, ey - sy
        dist = math.hypot(dx, dy)
        steps = max(self.min_steps, int(dist / 6.0))
        t = np.linspace(0.0, 1.0, steps + 1)
        ease = self._smoothstep(t)
        xs = sx + dx * ease
        ys = sy + dy * ease
        if self.jitter > 0 and steps > 2:
            jx = np.random.normal(0, self.jitter, steps + 1)
            jy = np.random.normal(0, self.jitter, steps + 1)
            jx[0] = jy[0] = 0.0
            jx[-1] = jy[-1] = 0.0
            xs = xs + jx
            ys = ys + jy
        return np.stack([xs, ys], axis=1)

    def move_to(self, x, y, duration=None, block=True):
        sx, sy = self.get_position()
        if (sx == int(x) and sy == int(y)):
            return
        points = self._build_points(sx, sy, int(x), int(y))
        n = len(points)
        if duration is None:
            dist = math.hypot(int(x) - sx, int(y) - sy)
            duration = max(0.05, dist / 1400.0)
        avg_interval = max(0.004, duration / n)
        lo = max(2.0, avg_interval * random.uniform(0.4, 0.6))
        hi = max(lo + 1.0, avg_interval * random.uniform(1.4, 1.8))
        for i in range(1, n):
            self._set_position(points[i][0], points[i][1])
            time.sleep(random.uniform(lo, hi) * 0.001)
        self._set_position(int(x), int(y))
        self._last_mouse_move_time = time.time()
        if block:
            time.sleep(random.uniform(0.15, 0.45))

    def _press_release(self, down_flag, up_flag):
        self._mouse_event(down_flag)
        time.sleep(random.uniform(0.06, 0.14))
        self._mouse_event(up_flag)

    def click(self, button="left"):
        if button == "right":
            self._press_release(self.MOUSEEVENTF_RIGHTDOWN, self.MOUSEEVENTF_RIGHTUP)
        else:
            self._press_release(self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP)

    def click_at(self, x, y, button="left", duration=None):
        self.move_to(x, y, duration=duration)
        self.click(button)

    def click_at_with_human(self, x, y, button="left"):
        offset_x = int(random.uniform(-3, 3))
        offset_y = int(random.uniform(-3, 3))
        self.click_at(x + offset_x, y + offset_y, button=button)

    def double_click(self, x, y):
        self.move_to(x, y)
        self._press_release(self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP)
        time.sleep(random.uniform(0.08, 0.16))
        self._press_release(self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP)

    def drag(self, x1, y1, x2, y2):
        self.move_to(x1, y1)
        self._mouse_event(self.MOUSEEVENTF_LEFTDOWN)
        try:
            pts = self._build_points(x1, y1, x2, y2)
            lo, hi = self.step_interval
            for i in range(1, len(pts)):
                self._set_position(pts[i][0], pts[i][1])
                time.sleep(random.uniform(lo, hi))
        finally:
            self._mouse_event(self.MOUSEEVENTF_LEFTUP)

    def move_click_ratio(self, hwnd, x_ratio, y_ratio, button="left", *,
                         click=True, background=False, rect=None, duration=0.05):
        """Click at a position relative to window client area."""
        try:
            user32 = ctypes.windll.user32
            rect = ctypes.wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            cw = rect.right - rect.left
            ch = rect.bottom - rect.top
            if cw <= 0 or ch <= 0:
                return False
            cx = int(cw * x_ratio)
            cy = int(ch * y_ratio)
            pt = ctypes.wintypes.POINT(cx, cy)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))
            self.click_at(pt.x, pt.y, button=button)
            return True
        except Exception:
            return False

    def click_absolute_win32(self, x: int, y: int, button: str = "left",
                              hwnd: int = 0) -> bool:
        """Send mouse click directly to window at absolute screen coords."""
        try:
            if hwnd:
                pt = ctypes.wintypes.POINT(x, y)
                ctypes.windll.user32.ScreenToClient(hwnd, ctypes.byref(pt))
                client_x, client_y = pt.x, pt.y
            else:
                client_x, client_y = x, y

            WM_LBUTTONDOWN = 0x0201
            WM_LBUTTONUP = 0x0202
            WM_RBUTTONDOWN = 0x0204
            WM_RBUTTONUP = 0x0205

            if button == "right":
                down, up = WM_RBUTTONDOWN, WM_RBUTTONUP
            else:
                down, up = WM_LBUTTONDOWN, WM_LBUTTONUP

            lparam = (client_y << 16) | (client_x & 0xFFFF)
            if hwnd:
                ctypes.windll.user32.PostMessageW(hwnd, down, 0, lparam)
                time.sleep(random.uniform(0.05, 0.1))
                ctypes.windll.user32.PostMessageW(hwnd, up, 0, lparam)
            else:
                self._mouse_event(down)
                time.sleep(random.uniform(0.05, 0.1))
                self._mouse_event(up)
            return True
        except Exception:
            return False

    def send_ctrl_v_win32(self, hwnd: int) -> bool:
        """Send Ctrl+V to a window using PostMessage."""
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_CONTROL = 0x11
        try:
            user32 = ctypes.windll.user32
            time.sleep(0.05)
            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_CONTROL, 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYDOWN, ord('V'), 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYUP, ord('V'), 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_CONTROL, 0)
            return True
        except Exception:
            return False

    def send_enter_win32(self, hwnd: int) -> bool:
        """Send Enter key to a window using PostMessage."""
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_RETURN = 0x0D
        try:
            user32 = ctypes.windll.user32
            time.sleep(0.05)
            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------
    # 前台窗口管理
    # ---------------------------------------------------------------
    def force_foreground_window(self, hwnd: int) -> bool:
        """Bring window to foreground with aggressive restore/show."""
        if not hwnd:
            return False
        try:
            SW_SHOW = 5
            SW_RESTORE = 9
            SW_SHOWNOACTIVATE = 4
            user32 = ctypes.windll.user32

            current_fg = user32.GetForegroundWindow()
            self._last_foreground_hwnd = current_fg

            if current_fg == hwnd:
                return True

            user32.ShowWindow(hwnd, SW_RESTORE)
            time.sleep(0.05)
            user32.ShowWindow(hwnd, SW_SHOW)
            time.sleep(0.05)

            user32.SetForegroundWindow(hwnd)
            time.sleep(0.15)

            result = user32.GetForegroundWindow() == hwnd
            if not result:
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                time.sleep(0.05)
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.15)
                result = user32.GetForegroundWindow() == hwnd

            return result
        except Exception:
            return False

    def restore_previous_foreground(self, hwnd: Optional[int] = None) -> bool:
        """Restore previously saved foreground window."""
        try:
            target = hwnd or self._last_foreground_hwnd
            if target and target != ctypes.windll.user32.GetForegroundWindow():
                ctypes.windll.user32.SetForegroundWindow(target)
                time.sleep(0.1)
                return True
            return False
        except Exception:
            return False

    def is_target_foreground(self, hwnd: int) -> bool:
        """Check if the given window is currently the foreground window."""
        try:
            return ctypes.windll.user32.GetForegroundWindow() == hwnd
        except Exception:
            return False

    def _foreground_mouse_allowed(self) -> bool:
        """Check if foreground mouse input is safe (no user is dragging)."""
        if self.user_dragging_mouse():
            return False
        if self._foreground_lock:
            return False
        return True

    def _focus_window_for_privacy_overlay(self) -> bool:
        """Focus the window to ensure privacy overlay covers it."""
        if self._window_manager is None:
            return False
        try:
            hwnd = self._window_manager.target_hwnd
            if not hwnd:
                return False
            if not self.is_target_foreground(hwnd):
                return self.force_foreground_window(hwnd)
            return True
        except Exception:
            return False

    def _foreground_fallback_click(self, hwnd: int, x: int, y: int) -> bool:
        """Fallback click when foreground is needed for a click action."""
        if not self._foreground_mouse_allowed():
            return False
        if not self._foreground_fallback_enabled:
            return False

        try:
            was_foreground = self.is_target_foreground(hwnd)
            if not was_foreground:
                if not self.force_foreground_window(hwnd):
                    return False
            self.click_at(x, y)
            if not was_foreground:
                self.restore_previous_foreground()
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------
    # 用户拖拽检测
    # ---------------------------------------------------------------
    def user_dragging_mouse(self) -> bool:
        """Detect if the user is currently dragging the mouse."""
        try:
            LEFT_BUTTON = 0x01
            async_state = ctypes.windll.user32.GetAsyncKeyState(LEFT_BUTTON)
            if not (async_state & 0x8000):
                return False
            elapsed = time.time() - self._last_mouse_move_time
            if elapsed < 0.05:
                return False
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------
    # 鼠标护盾
    # ---------------------------------------------------------------
    def enable_mouse_guard(self, defer_ms: int = 500):
        """Enable mouse guard: defer automation if user is active."""
        self._mouse_guard_enabled = True
        self._mouse_guard_defer_ms = defer_ms

    def disable_mouse_guard(self):
        self._mouse_guard_enabled = False

    def mouse_guard(self) -> bool:
        """Check if mouse guard allows automation to proceed."""
        if not self._mouse_guard_enabled:
            return True
        if self.user_dragging_mouse():
            time.sleep(self._mouse_guard_defer_ms / 1000.0)
            if self.user_dragging_mouse():
                return False
        return True

    def defer_mouse(self, ms: int = 200):
        """Defer mouse operations for a short period."""
        time.sleep(ms / 1000.0)

    def acquire_foreground_lock(self) -> bool:
        """Acquire foreground lock to prevent concurrent foreground changes."""
        if self._foreground_lock:
            return False
        self._foreground_lock = True
        return True

    def release_foreground_lock(self):
        self._foreground_lock = False

    # ---------------------------------------------------------------
    # 剪贴板操作
    # ---------------------------------------------------------------
    def _set_clipboard_text(self, text):
        CF_UNICODETEXT = 13
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        block = text.encode("utf-16-le") + b"\x00\x00"
        size = len(block)
        hmem = kernel32.GlobalAlloc(0x0002, size)
        ptr = kernel32.GlobalLock(hmem)
        ctypes.memmove(ptr, block, size)
        kernel32.GlobalUnlock(hmem)
        user32.OpenClipboard(0)
        user32.EmptyClipboard()
        user32.SetClipboardData(CF_UNICODETEXT, hmem)
        user32.CloseClipboard()

    def _set_clipboard_image(self, image_path):
        try:
            import win32clipboard
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            output = io.BytesIO()
            img.convert("RGB").save(output, format="BMP")
            data = output.getvalue()[14:]
            output.close()
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _post_ctrl_v_enter(self, hwnd):
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_CONTROL = 0x11
        VK_RETURN = 0x0D
        try:
            user32 = ctypes.windll.user32
            time.sleep(0.05)
            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_CONTROL, 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYDOWN, ord('V'), 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYUP, ord('V'), 0)
            time.sleep(0.02)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_CONTROL, 0)
            time.sleep(0.1)
            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0)
            return True
        except Exception:
            return False

    def paste_and_enter(self, hwnd, text, background=False, compose_delay=None):
        """Paste text into WeChat and press Enter to send."""
        if not text:
            return False
        self._set_clipboard_text(text)
        return self._post_ctrl_v_enter(hwnd)

    def paste_image_and_enter(self, hwnd, image_path, background=False):
        """Paste image into WeChat and press Enter to send."""
        if not image_path or not os.path.exists(image_path):
            return False
        if not self._set_clipboard_image(image_path):
            return False
        return self._post_ctrl_v_enter(hwnd)

    def paste_file_and_enter(self, hwnd, file_path, background=False):
        """Paste file into WeChat via clipboard and press Enter."""
        if not file_path or not os.path.exists(file_path):
            return False
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(
                win32clipboard.CF_HDROP,
                (file_path,))
            win32clipboard.CloseClipboard()
            return self._post_ctrl_v_enter(hwnd)
        except ImportError:
            return False
        except Exception:
            return False

    def typewriter(self, text, cps_min=15, cps_max=28):
        """Emulate human typing by sending characters one at a time."""
        if not text:
            return
        KEYEVENTF_UNICODE = 0x0004
        KEYEVENTF_KEYUP = 0x0002
        extra = ctypes.POINTER(ctypes.wintypes.ULONG)()
        user32 = ctypes.windll.user32

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.wintypes.WORD),
                ("wScan", ctypes.wintypes.WORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
            ]

        class INPUT(ctypes.Structure):
            class _INPUTunion(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]
            _anonymous_ = ("u",)
            _fields_ = [("type", ctypes.wintypes.DWORD), ("u", _INPUTunion)]

        for char in text:
            code = ord(char)
            inputs = [
                INPUT(type=1, ki=KEYBDINPUT(
                    wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE,
                    time=0, dwExtraInfo=extra)),
                INPUT(type=1, ki=KEYBDINPUT(
                    wVk=0, wScan=code,
                    dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                    time=0, dwExtraInfo=extra)),
            ]
            arr = (INPUT * len(inputs))(*inputs)
            user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
            lo = 1.0 / max(cps_min, cps_max)
            hi = 1.0 / min(cps_min, cps_max)
            time.sleep(random.uniform(lo, hi))

    def block_physical_input(self):
        try:
            return bool(ctypes.windll.user32.BlockInput(True))
        except Exception:
            return False

    def unblock_physical_input(self):
        try:
            return bool(ctypes.windll.user32.BlockInput(False))
        except Exception:
            return False

    def release_stuck_modifiers(self):
        VK_CTRL = 0x11
        VK_ALT = 0x12
        VK_SHIFT = 0x10
        VK_WIN = 0x5B
        KEYEVENTF_KEYUP = 0x0002
        modifiers = [VK_CTRL, VK_ALT, VK_SHIFT, VK_WIN]
        try:
            for vk in modifiers:
                async_key = ctypes.windll.user32.GetAsyncKeyState(vk)
                if async_key & 0x8000:
                    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        except Exception:
            pass

    # ===============================================================
    # 比例坐标换算 / 比例点击（前台+后台两条路径）
    # ===============================================================
    def point_from_ratio(self, rect, x_ratio, y_ratio):
        """把 0~1 比例换算成绝对屏幕坐标（基于 rect 的 left/top/width/height）。"""
        rect = rect or {}
        left = int(rect.get("left", 0))
        top = int(rect.get("top", 0))
        width = int(rect.get("width", 0)) or (int(rect.get("right", 0)) - left)
        height = int(rect.get("height", 0)) or (int(rect.get("bottom", 0)) - top)
        if width <= 0:
            width = 1
        if height <= 0:
            height = 1
        px = left + int(round(x_ratio * width))
        py = top + int(round(y_ratio * height))
        return (px, py)

    def post_click_ratio(self, hwnd, rect, x_ratio, y_ratio, *, double=False):
        """后台（不抢焦点）点击窗口内比例坐标。"""
        x, y = self.point_from_ratio(rect, x_ratio, y_ratio)
        rel_x = int(x) - int(rect.get("left", 0)) if rect else int(x)
        rel_y = int(y) - int(rect.get("top", 0)) if rect else int(y)
        try:
            HumanLikeMouse._post_click_no_activate(int(hwnd), rel_x, rel_y, double=double)
        except Exception:
            pass
        return (x, y)

    def right_click_ratio(self, rect, x_ratio, y_ratio, *, hwnd=0, background=False):
        """右键点击窗口内比例坐标（默认前台，hwnd 给定时走后台）。"""
        x, y = self.point_from_ratio(rect, x_ratio, y_ratio)
        rel_x = int(x) - int(rect.get("left", 0)) if rect else int(x)
        rel_y = int(y) - int(rect.get("top", 0)) if rect else int(y)
        if hwnd:
            try:
                HumanLikeMouse._post_right_click_no_activate(int(hwnd), rel_x, rel_y)
            except Exception:
                pass
        else:
            self.click_at_with_human(x, y, button="right")
            time.sleep(random.uniform(0.1, 0.22))
        return (x, y)

    # ===============================================================
    # 后台输入（PostMessage / WM_CHAR）
    # ===============================================================
    def post_text_chars(self, text, *, hwnd=0, background=False):
        """后台逐字符发送（GBK 兼容路径）。"""
        try:
            if os.name != "nt" or not hwnd:
                return {"ok": False, "method": "background_postmessage_chars",
                        "reason": "windows_hwnd_required"}
            HumanLikeMouse._post_text_chars(int(hwnd), text)
            return {"ok": True, "method": "background_postmessage_chars",
                    "reason": "background_chars_sent"}
        except Exception as exc:
            return {"ok": False, "method": "background_postmessage_chars",
                    "reason": "background chars failed: " + str(exc)}

    def post_clipboard_text(self, text, *, hwnd=0, background=False):
        """后台写剪贴板后发 WM_PASTE。"""
        try:
            if os.name != "nt" or not hwnd:
                return {"ok": False, "method": "background_wm_paste",
                        "reason": "windows_hwnd_required"}
            HumanLikeMouse._set_text_clipboard(text)
            time.sleep(random.uniform(0.05, 0.12))
            HumanLikeMouse._post_paste(int(hwnd))
            return {"ok": True, "method": "background_wm_paste",
                    "reason": "background_wm_paste_sent"}
        except Exception as exc:
            return {"ok": False, "method": "background_wm_paste",
                    "reason": "background wm_paste failed: " + str(exc)}

    def paste_only(self, hwnd, text, background=False, compose_delay=None):
        """只粘贴不回车。"""
        if not text:
            return False
        HumanLikeMouse._set_text_clipboard(text)
        if compose_delay is None:
            compose_delay = random.uniform(0.08, 0.2)
        try:
            if hwnd:
                time.sleep(compose_delay)
                HumanLikeMouse._post_paste(int(hwnd))
            else:
                HumanLikeMouse._send_ctrl_v_win32()
        except Exception:
            return False
        return True

    def enter_only(self, hwnd=0, background=False):
        """只回车。"""
        try:
            if background and hwnd:
                HumanLikeMouse._post_enter(int(hwnd))
            else:
                HumanLikeMouse._send_enter_win32()
        except Exception:
            return False
        return True

    def clear_input_background(self, hwnd=0):
        """后台清空输入框（多发 Backspace）。"""
        try:
            if os.name != "nt" or not hwnd:
                return {"ok": False, "method": "background_postmessage",
                        "reason": "windows_hwnd_required"}
            HumanLikeMouse._post_select_all_backspace(int(hwnd))
            return {"ok": True, "method": "background_postmessage",
                    "reason": "background_backspace_clear_sent"}
        except Exception as exc:
            return {"ok": False, "method": "background_postmessage",
                    "reason": "background clear failed: " + str(exc)}

    # ===============================================================
    # 前台窗口管理（私有静态/实例，对齐原版 v3.10）
    # ===============================================================
    @staticmethod
    def _is_target_foreground(hwnd, active_hwnd):
        """判断 hwnd 与 active_hwnd 是否为同一前台窗口（按根窗口/进程）。"""
        try:
            if not hwnd or not active_hwnd:
                return False
            if int(hwnd) == int(active_hwnd):
                return True
            target_root = win32gui.GetAncestor(int(hwnd), win32con.GA_ROOT)
            active_root = win32gui.GetAncestor(int(active_hwnd), win32con.GA_ROOT)
            if target_root and active_root and target_root == active_root:
                return True
            _, target_pid = win32process.GetWindowThreadProcessId(int(hwnd))
            _, active_pid = win32process.GetWindowThreadProcessId(int(active_hwnd))
            return target_pid == active_pid
        except Exception:
            return False

    @staticmethod
    def _force_foreground_window(hwnd):
        """用 AttachThreadInput 强制把窗口置前台。"""
        try:
            hwnd = int(hwnd)
            if not hwnd or not win32gui.IsWindow(hwnd):
                return False
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            current_thread = kernel32.GetCurrentThreadId()
            foreground = user32.GetForegroundWindow()
            target_thread = user32.GetWindowThreadProcessId(hwnd)[0]
            foreground_thread = user32.GetWindowThreadProcessId(foreground)[0]
            attached = []
            try:
                if foreground_thread and foreground_thread != current_thread:
                    if user32.AttachThreadInput(current_thread, foreground_thread, True):
                        attached.append((current_thread, foreground_thread))
                if target_thread and target_thread != current_thread:
                    if user32.AttachThreadInput(current_thread, target_thread, True):
                        attached.append((current_thread, target_thread))
                user32.AllowSetForegroundWindow(0xFFFFFFFF)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetActiveWindow(hwnd)
                win32gui.SetForegroundWindow(hwnd)
                user32.SwitchToThisWindow(hwnd, True)
                user32.SetFocus(hwnd)
            finally:
                for a, b in reversed(attached):
                    try:
                        user32.AttachThreadInput(a, b, False)
                    except Exception:
                        pass
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            return False

    def _foreground_focus_wait_seconds(self):
        """前台聚焦等待秒数（来自 config.wechat.foreground_fallback_focus_wait_seconds）。"""
        cfg = self._config
        if isinstance(cfg, dict):
            sub = cfg.get("wechat", cfg)
            if isinstance(sub, dict):
                return max(0.02, float(sub.get("foreground_fallback_focus_wait_seconds", 0.05)))
        return 0.02

    def _foreground_clipboard_wait_seconds(self):
        """剪贴板就绪等待秒数（来自 config.wechat.foreground_fallback_clipboard_wait_seconds）。"""
        cfg = self._config
        if isinstance(cfg, dict):
            sub = cfg.get("wechat", cfg)
            if isinstance(sub, dict):
                return max(0.0, float(sub.get("foreground_fallback_clipboard_wait_seconds", 0.02)))
        return 0.0

    def _restore_previous_foreground(self, previous_hwnd):
        """恢复之前的前台窗口。"""
        try:
            cfg = self._config
            enabled = True
            if isinstance(cfg, dict):
                sub = cfg.get("wechat", cfg)
                if isinstance(sub, dict):
                    enabled = bool(sub.get("background_send_restore_foreground", True))
            if not enabled:
                return
            if previous_hwnd and win32gui.IsWindow(int(previous_hwnd)):
                win32gui.SetForegroundWindow(int(previous_hwnd))
        except Exception:
            pass

    @staticmethod
    def _click_absolute_win32(x, y):
        """在物理桌面坐标点击（含虚拟屏，走 Win32 光标路径）。"""
        try:
            user32 = ctypes.windll.user32
            user32.SetCursorPos(int(x), int(y))
            time.sleep(random.uniform(0.025, 0.035))
            user32.mouse_event(HumanLikeMouse.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(random.uniform(0.025, 0.035))
            user32.mouse_event(HumanLikeMouse.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except Exception:
            pass

    @staticmethod
    def _release_stuck_modifiers_win32():
        """前台键注入前释放卡住的 Win/Ctrl/Alt/Shift（含左右变体）。"""
        try:
            user32 = ctypes.windll.user32
            keys = [0x5B, 0x5C, 0x11, 0xA2, 0xA3, 0x12, 0xA4, 0xA5, 0x10, 0xA0, 0xA1]
            for key in keys:
                if user32.GetAsyncKeyState(key) & 0x8000:
                    user32.keybd_event(key, 0, 0x0002, 0)
                    time.sleep(0.01)
        except Exception:
            pass

    @staticmethod
    def _send_ctrl_v_win32():
        """前台 keybd_event 发送 Ctrl+V。"""
        try:
            HumanLikeMouse._release_stuck_modifiers_win32()
            user32 = ctypes.windll.user32
            VK_CONTROL = 0x11
            time.sleep(0.006)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            time.sleep(0.006)
            user32.keybd_event(ord("V"), 0, 0, 0)
            time.sleep(0.006)
            user32.keybd_event(ord("V"), 0, 0x0002, 0)
            time.sleep(0.006)
            user32.keybd_event(VK_CONTROL, 0, 0x0002, 0)
        except Exception:
            pass

    @staticmethod
    def _send_enter_win32():
        """前台 keybd_event 发送 Enter。"""
        try:
            HumanLikeMouse._release_stuck_modifiers_win32()
            user32 = ctypes.windll.user32
            VK_RETURN = 0x0D
            time.sleep(0.006)
            user32.keybd_event(VK_RETURN, 0, 0, 0)
            time.sleep(0.006)
            user32.keybd_event(VK_RETURN, 0, 0x0002, 0)
        except Exception:
            pass

    @staticmethod
    def _send_escape_win32():
        """前台 keybd_event 发送 Esc。"""
        try:
            HumanLikeMouse._release_stuck_modifiers_win32()
            user32 = ctypes.windll.user32
            VK_ESCAPE = 0x1B
            time.sleep(0.006)
            user32.keybd_event(VK_ESCAPE, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(VK_ESCAPE, 0, 0x0002, 0)
        except Exception:
            pass

    @staticmethod
    def _send_ctrl_a_delete_backspace_win32():
        """前台 Ctrl+A 全选后 Delete 清空。"""
        try:
            HumanLikeMouse._release_stuck_modifiers_win32()
            user32 = ctypes.windll.user32
            VK_CONTROL = 0x11
            time.sleep(0.006)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            time.sleep(0.006)
            user32.keybd_event(ord("A"), 0, 0, 0)
            time.sleep(0.006)
            user32.keybd_event(ord("A"), 0, 0x0002, 0)
            time.sleep(0.006)
            user32.keybd_event(VK_CONTROL, 0, 0x0002, 0)
            time.sleep(0.025)
            VK_DELETE = 0x2E
            user32.keybd_event(VK_DELETE, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(VK_DELETE, 0, 0x0002, 0)
        except Exception:
            pass

    @staticmethod
    def _block_physical_input(enabled):
        """屏蔽/恢复物理输入（BlockInput）。"""
        try:
            if not enabled:
                ctypes.windll.user32.BlockInput(False)
                return False
            return bool(ctypes.windll.user32.BlockInput(True))
        except Exception:
            return False

    @staticmethod
    def _set_text_clipboard(text):
        """写文本到系统剪贴板（win32clipboard CF_UNICODETEXT）。"""
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text))
            win32clipboard.CloseClipboard()
        except Exception:
            pass

    def _compose_pause(self, text):
        """根据字数/配置生成拟人的“构思停顿”延迟。"""
        cfg = self._config
        enabled = True
        low, high = 0.15, 0.35
        cap = 1.5
        per_char = 0.025
        if isinstance(cfg, dict):
            sub = cfg.get("natural_reply", cfg)
            if isinstance(sub, dict):
                enabled = bool(sub.get("compose_pause_enabled", True))
                low = float(sub.get("compose_pause_min_seconds", 0.15))
                high = float(sub.get("compose_pause_max_seconds", 0.35))
                per_char = float(sub.get("compose_pause_per_char_seconds", 0.025))
        if not enabled:
            return 0.0
        typed = str(text)
        delay = random.uniform(low, high) + len(typed) * per_char * random.uniform(0.65, 1.35)
        return max(low, min(delay, cap))

    @staticmethod
    def _send_ctrl_v_via_foreground_keybd(wc_hwnd):
        """SetForegroundWindow + 真实 keybd_event Ctrl+V，随后恢复前台。"""
        try:
            wc_hwnd = int(wc_hwnd)
            user32 = ctypes.windll.user32
            prev_fg = user32.GetForegroundWindow()
            user32.SetForegroundWindow(wc_hwnd)
            time.sleep(0.05)
            HumanLikeMouse._release_stuck_modifiers_win32()
            VK_CONTROL = 17
            VK_V = 86
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(VK_V, 0, 0, 0)
            time.sleep(0.03)
            user32.keybd_event(VK_V, 0, 0x0002, 0)
            time.sleep(0.04)
            user32.keybd_event(VK_CONTROL, 0, 0x0002, 0)
            if prev_fg:
                try:
                    user32.SetForegroundWindow(prev_fg)
                except Exception:
                    pass
            return True
        except Exception:
            return False

    # ===============================================================
    # 后台 PostMessage 原语（私有静态）
    # ===============================================================
    @staticmethod
    def _post_click(hwnd, x, y, *, double=False):
        """后台 PostMessage 鼠标左键点击（client 坐标）。"""
        try:
            hwnd = int(hwnd)
            user32 = ctypes.windll.user32
            lparam = (int(y) << 16) | (int(x) & 0xFFFF)
            clicks = 2 if double else 1
            WM_MOUSEMOVE = 0x0200
            WM_LBUTTONDOWN = 0x0201
            WM_LBUTTONUP = 0x0202
            MK_LBUTTON = 0x0001
            for _ in range(clicks):
                user32.PostMessageW(hwnd, WM_MOUSEMOVE, MK_LBUTTON, lparam)
                time.sleep(0.03)
                user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                time.sleep(0.05)
                user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)
                time.sleep(0.08)
        except Exception:
            pass

    @staticmethod
    def _post_click_no_activate(hwnd, x, y, *, double=False):
        """后台点击且不抢占前台（临时加 WS_EX_NOACTIVATE）。"""
        hwnd = int(hwnd)
        old_exstyle = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               old_exstyle | win32con.WS_EX_NOACTIVATE)
        try:
            win32gui.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)
            HumanLikeMouse._post_click(hwnd, x, y, double=double)
        finally:
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, old_exstyle)
            win32gui.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)

    @staticmethod
    def _post_right_click_no_activate(hwnd, x, y):
        """后台右键点击且不抢占前台。"""
        hwnd = int(hwnd)
        old_exstyle = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               old_exstyle | win32con.WS_EX_NOACTIVATE)
        try:
            win32gui.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)
            user32 = ctypes.windll.user32
            lparam = (int(y) << 16) | (int(x) & 0xFFFF)
            WM_MOUSEMOVE = 0x0200
            WM_RBUTTONDOWN = 0x0204
            WM_RBUTTONUP = 0x0205
            MK_RBUTTON = 0x0002
            user32.PostMessageW(hwnd, WM_MOUSEMOVE, MK_RBUTTON, lparam)
            time.sleep(0.03)
            user32.PostMessageW(hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)
            time.sleep(0.05)
            user32.PostMessageW(hwnd, WM_RBUTTONUP, 0, lparam)
        finally:
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, old_exstyle)
            win32gui.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)

    @staticmethod
    def _post_text_chars(hwnd, text):
        """后台 WM_CHAR 逐字符（剥掉补充平面/GBK 不支持字符）。"""
        try:
            hwnd = int(hwnd)
            user32 = ctypes.windll.user32
            WM_CHAR = 0x0102
            cleaned = []
            for ch in str(text):
                if ord(ch) > 0xFFFF:
                    continue
                try:
                    ch.encode("gbk")
                except UnicodeEncodeError:
                    continue
                cleaned.append(ch)
            for ch in "".join(cleaned):
                user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)
                time.sleep(0.015)
        except Exception:
            pass

    @staticmethod
    def _post_paste_and_enter(hwnd, *, compose_delay=0.05):
        """后台 WM_PASTE + Enter。"""
        try:
            hwnd = int(hwnd)
            HumanLikeMouse._post_paste(hwnd)
            time.sleep(max(0.05, float(compose_delay)))
            HumanLikeMouse._post_enter(hwnd)
        except Exception:
            pass

    @staticmethod
    def _post_paste(hwnd):
        """后台 WM_PASTE（best-effort）。"""
        try:
            ctypes.windll.user32.PostMessageW(int(hwnd), 0x0302, 0, 0)
        except Exception:
            pass

    @staticmethod
    def _post_enter(hwnd):
        """后台 WM_KEYDOWN/UP VK_RETURN。"""
        try:
            user32 = ctypes.windll.user32
            VK_RETURN = 0x0D
            WM_KEYDOWN = 0x0100
            WM_KEYUP = 0x0101
            user32.PostMessageW(int(hwnd), WM_KEYDOWN, VK_RETURN, 0)
            time.sleep(0.03)
            user32.PostMessageW(int(hwnd), WM_KEYUP, VK_RETURN, 0)
        except Exception:
            pass

    @staticmethod
    def _post_escape(hwnd):
        """后台 WM_KEYDOWN/UP VK_ESCAPE。"""
        try:
            user32 = ctypes.windll.user32
            VK_ESCAPE = 0x1B
            WM_KEYDOWN = 0x0100
            WM_KEYUP = 0x0101
            user32.PostMessageW(int(hwnd), WM_KEYDOWN, VK_ESCAPE, 0)
            time.sleep(0.02)
            user32.PostMessageW(int(hwnd), WM_KEYUP, VK_ESCAPE, 0)
        except Exception:
            pass

    @staticmethod
    def _post_backspace(hwnd, *, count=1):
        """后台多次 VK_BACK。"""
        try:
            user32 = ctypes.windll.user32
            VK_BACK = 0x08
            WM_KEYDOWN = 0x0100
            WM_KEYUP = 0x0101
            for _ in range(max(1, int(count))):
                user32.PostMessageW(int(hwnd), WM_KEYDOWN, VK_BACK, 0)
                time.sleep(0.008)
                user32.PostMessageW(int(hwnd), WM_KEYUP, VK_BACK, 0)
        except Exception:
            pass

    @staticmethod
    def _post_select_all_backspace(hwnd):
        """后台清空输入框：连发 60 次 Backspace（PostMessage 无 Ctrl 同步）。"""
        HumanLikeMouse._post_backspace(int(hwnd), count=60)

    # ===============================================================
    # 前台一次性动作（对齐原版 foreground_*_once 系列）
    # ===============================================================
    def focus_window(self, hwnd, *, activation_point=None):
        """聚焦 WeChat 窗口，返回 dict 含 previous_foreground。"""
        try:
            if not hwnd or not win32gui.IsWindow(int(hwnd)):
                return {"ok": False, "deferred": False, "method": "foreground_keyboard",
                        "reason": "wechat_window_handle_invalid"}
            previous_foreground = win32gui.GetForegroundWindow()
            if self.user_dragging_mouse():
                return {"ok": False, "deferred": True, "method": "foreground_keyboard",
                        "reason": "user_dragging_mouse", "mouse_button": None,
                        "previous_foreground": previous_foreground}
            self.force_foreground_window(int(hwnd))
            time.sleep(self._foreground_focus_wait_seconds())
            if not self.is_target_foreground(int(hwnd)):
                HumanLikeMouse._force_foreground_window(int(hwnd))
                time.sleep(self._foreground_focus_wait_seconds())
                if not self.is_target_foreground(int(hwnd)):
                    return {"ok": False, "deferred": False, "method": "foreground_keyboard",
                            "reason": "wechat_window_did_not_become_foreground",
                            "previous_foreground": previous_foreground,
                            "active_foreground": win32gui.GetForegroundWindow()}
            self._last_foreground_hwnd = previous_foreground
            return {"ok": True, "deferred": False, "method": "foreground_keyboard",
                    "reason": "window_focused", "previous_foreground": previous_foreground}
        except Exception as exc:
            return {"ok": False, "deferred": False, "method": "foreground_keyboard",
                    "reason": "focus failed: " + str(exc), "previous_foreground": None}

    def foreground_clear_input_once(self, hwnd, rect, x_ratio, y_ratio, *,
                                    block_input=False, preserve_quote=False):
        """前台清空输入框（点输入框→Ctrl+A→Delete），用完恢复焦点/鼠标。"""
        result = {"ok": False, "method": "foreground_keyboard_once",
                  "reason": "windows_hwnd_and_rect_required"}
        try:
            if os.name != "nt" or not hwnd or not rect:
                return result
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            blocked = False
            if block_input:
                blocked = HumanLikeMouse._block_physical_input(True)
            try:
                x, y = self.point_from_ratio(rect, x_ratio, y_ratio)
                self.move_to(x, y)
                time.sleep(0.025)
                HumanLikeMouse._click_absolute_win32(x, y)
                HumanLikeMouse._send_ctrl_a_delete_backspace_win32()
                result = {"ok": True, "method": "foreground_keyboard_once",
                          "reason": "foreground_ctrl_a_delete_backspace_sent",
                          "previous_foreground": prev}
            finally:
                if blocked:
                    HumanLikeMouse._block_physical_input(False)
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard_once",
                      "reason": "foreground clear failed: " + str(exc)}
        return result

    def foreground_click_once(self, hwnd, rect, x_ratio, y_ratio, *, block_input=False):
        """前台点一次控件，立即恢复焦点与鼠标。"""
        result = {"ok": False, "method": "foreground_click_once",
                  "reason": "windows_hwnd_and_rect_required"}
        try:
            if os.name != "nt" or not hwnd or not rect:
                return result
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            blocked = False
            if block_input:
                blocked = HumanLikeMouse._block_physical_input(True)
            try:
                x, y = self.point_from_ratio(rect, x_ratio, y_ratio)
                self.move_to(x, y)
                time.sleep(0.025)
                HumanLikeMouse._click_absolute_win32(x, y)
                result = {"ok": True, "method": "foreground_click_once",
                          "reason": "foreground_control_clicked", "x": x, "y": y,
                          "previous_foreground": prev}
            finally:
                if blocked:
                    HumanLikeMouse._block_physical_input(False)
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_click_once",
                      "reason": "foreground click failed: " + str(exc)}
        return result

    def foreground_paste_and_enter(self, hwnd, text, *, compose_delay=None):
        """前台聚焦→写剪贴板→Ctrl+V→Enter，用完恢复焦点。"""
        result = {"ok": False, "method": "foreground_keyboard",
                  "reason": "windows_hwnd_required"}
        try:
            if os.name != "nt" or not hwnd:
                return result
            if compose_delay is None:
                compose_delay = random.uniform(0.08, 0.18)
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            try:
                HumanLikeMouse._set_text_clipboard(text)
                time.sleep(compose_delay)
                HumanLikeMouse._send_ctrl_v_win32()
                time.sleep(0.05)
                HumanLikeMouse._send_enter_win32()
                result = {"ok": True, "method": "foreground_keyboard",
                          "reason": "foreground_ctrl_v_enter_sent", "previous_foreground": prev}
            finally:
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard",
                      "reason": "foreground keyboard send failed: " + str(exc)}
        return result

    def foreground_paste_enter_once(self, hwnd, text, *, enter_delay=0.01):
        """前台一次性 Ctrl+V+Enter（被 friend_request_acceptor 调用）。"""
        result = {"ok": False, "method": "foreground_keyboard_once",
                  "reason": "windows_hwnd_required"}
        try:
            if os.name != "nt" or not hwnd:
                return result
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            try:
                HumanLikeMouse._set_text_clipboard(text)
                time.sleep(self._foreground_clipboard_wait_seconds())
                HumanLikeMouse._send_ctrl_v_win32()
                time.sleep(max(enter_delay, 0.01))
                HumanLikeMouse._send_enter_win32()
                result = {"ok": True, "method": "foreground_keyboard_once",
                          "reason": "foreground_ctrl_v_enter_once_sent", "previous_foreground": prev}
            finally:
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard_once",
                      "reason": "foreground one-shot send failed: " + str(exc)}
        return result

    def foreground_clear_paste_enter_once(self, hwnd, text, rect, x_ratio, y_ratio, *,
                                          enter_delay=0.025, max_takeover_seconds=1.0,
                                          block_input=False, preserve_quote=False):
        """前台清空→粘贴→发送，整套接管窗体后恢复。"""
        result = {"ok": False, "method": "foreground_keyboard_takeover",
                  "reason": "windows_hwnd_required"}
        try:
            if os.name != "nt" or not hwnd:
                return result
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            blocked = False
            if block_input:
                blocked = HumanLikeMouse._block_physical_input(True)
            try:
                x, y = self.point_from_ratio(rect, x_ratio, y_ratio)
                self.move_to(x, y)
                time.sleep(0.025)
                HumanLikeMouse._click_absolute_win32(x, y)
                HumanLikeMouse._send_ctrl_a_delete_backspace_win32()
                time.sleep(0.02)
                HumanLikeMouse._set_text_clipboard(text)
                time.sleep(self._foreground_clipboard_wait_seconds())
                HumanLikeMouse._send_ctrl_v_win32()
                time.sleep(max(enter_delay, 0.01))
                HumanLikeMouse._send_enter_win32()
                result = {"ok": True, "method": "foreground_keyboard_takeover",
                          "reason": "foreground_clear_paste_enter_sent", "previous_foreground": prev}
            finally:
                if blocked:
                    HumanLikeMouse._block_physical_input(False)
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard_takeover",
                      "reason": "foreground takeover send failed: " + str(exc)}
        return result

    def foreground_paste_only(self, hwnd, text, *, compose_delay=None):
        """前台只粘贴不回车。"""
        result = {"ok": False, "method": "foreground_keyboard",
                  "reason": "windows_hwnd_required"}
        try:
            if os.name != "nt" or not hwnd:
                return result
            if compose_delay is None:
                compose_delay = random.uniform(0.08, 0.18)
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            try:
                HumanLikeMouse._set_text_clipboard(text)
                time.sleep(compose_delay)
                HumanLikeMouse._send_ctrl_v_win32()
                result = {"ok": True, "method": "foreground_keyboard",
                          "reason": "foreground_ctrl_v_sent", "previous_foreground": prev}
            finally:
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard",
                      "reason": "foreground paste failed: " + str(exc)}
        return result

    def foreground_enter_only(self, hwnd=0):
        """前台只回车。"""
        result = {"ok": False, "method": "foreground_keyboard",
                  "reason": "windows_hwnd_required"}
        try:
            if os.name != "nt" or not hwnd:
                return result
            focused = self.focus_window(int(hwnd))
            prev = focused.get("previous_foreground") if isinstance(focused, dict) else None
            try:
                HumanLikeMouse._send_enter_win32()
                result = {"ok": True, "method": "foreground_keyboard",
                          "reason": "foreground_enter_sent", "previous_foreground": prev}
            finally:
                self._restore_previous_foreground(prev)
        except Exception as exc:
            result = {"ok": False, "method": "foreground_keyboard",
                      "reason": "foreground enter failed: " + str(exc)}
        return result