import ctypes
from ctypes import wintypes
import time
import random
import threading
import logging
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEMOVE = 0x0200
WM_SETTEXT = 0x000C
WM_GETTEXT = 0x000D
WM_USER = 0x0400

VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_ALT = 0x12

MK_LBUTTON = 0x0001
MK_CONTROL = 0x0008

HC_ACTION = 0

# ---- SendInput structures (hardware-level input injection) ----
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class INPUT(ctypes.Structure):
    class _INPUTunion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


@dataclass
class RPASendResult:
    success: bool = False
    message: str = ""
    error: str = ""


class WeChatRPA:

    # ---- 对齐原版 RedDotDetector 类常量（从 disasm 反编译还原）----
    LIST_X_END = 400                 # 列表列宽（用于绿色高亮判定区域右界）
    LIST_CLICK_X = 100               # 会话行的固定点击 X（客户区像素）
    LIST_CLICK_Y_OFFSET = -8         # 相对徽章中心 Y 的偏移（对准头像圆心）
    VERIFY_DELAY_SECONDS = 0.35      # 后台点击后等待 UI 刷新时间

    def __init__(self):
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._gdi32 = ctypes.windll.gdi32
        self._hwnd_wechat: Optional[int] = None
        self._clipboard_seq: int = 0
        self._typing_active: bool = False
        self._stop_event = threading.Event()
        # ---- DPI 感知（对齐原版）----
        # 未声明 DPI-aware 时，GetClientRect/ClientToScreen 返回逻辑像素，
        # 而 SendInput 需要物理像素 —— 150% 缩放下所有点击都会整体偏移。
        try:
            PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
            ok = self._user32.SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)
            if not ok:
                raise OSError("SetProcessDpiAwarenessContext failed")
        except Exception:
            try:
                self._user32.SetProcessDPIAware()   # Win7/8 fallback
            except Exception:
                pass

    def set_wechat_hwnd(self, hwnd: int) -> None:
        self._hwnd_wechat = hwnd

    def find_wechat_window(self, keywords: Optional[list] = None) -> Optional[int]:
        if keywords is None:
            keywords = ["微信", "WeChat"]

        found_hwnd = [0]

        def _enum_callback(hwnd, _):
            if not self._user32.IsWindowVisible(hwnd):
                return True
            title_buf = ctypes.create_unicode_buffer(512)
            self._user32.GetWindowTextW(hwnd, title_buf, 512)
            title = title_buf.value
            for kw in keywords:
                if kw in title:
                    found_hwnd[0] = hwnd
                    return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self._user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)
        self._hwnd_wechat = found_hwnd[0] if found_hwnd[0] else None
        return self._hwnd_wechat

    def activate_window(self, hwnd: Optional[int] = None) -> bool:
        target = hwnd or self._hwnd_wechat
        if not target:
            return False
        SW_RESTORE = 9
        self._user32.ShowWindow(target, SW_RESTORE)
        self._user32.SetForegroundWindow(target)
        return True

    def move_window(self, x: int, y: int, width: int, height: int,
                    hwnd: Optional[int] = None) -> bool:
        target = hwnd or self._hwnd_wechat
        if not target:
            return False
        return bool(self._user32.MoveWindow(target, x, y, width, height, True))

    def minimize_window(self, hwnd: Optional[int] = None) -> bool:
        target = hwnd or self._hwnd_wechat
        if not target:
            return False
        SW_MINIMIZE = 6
        self._user32.ShowWindow(target, SW_MINIMIZE)
        return True

    def restore_window(self, hwnd: Optional[int] = None) -> bool:
        target = hwnd or self._hwnd_wechat
        if not target:
            return False
        SW_RESTORE = 9
        self._user32.ShowWindow(target, SW_RESTORE)
        return True

    def _post_message(self, hwnd: int, msg: int, wparam: int = 0,
                       lparam: int = 0) -> bool:
        return bool(self._user32.PostMessageW(hwnd, msg, wparam, lparam))

    def _send_text_via_clipboard(self, hwnd: int, text: str) -> bool:
        if not text:
            return False
        try:
            import win32clipboard
            import win32con
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            win32clipboard.CloseClipboard()
        except ImportError:
            self._set_clipboard_python(text)
            time.sleep(0.05)

        self.send_key_combo(VK_CONTROL, ord('V'))
        return True

    def _send_text_via_clipboard_post(self, hwnd: int, text: str) -> bool:
        """后台粘贴：先设剪贴板，再用 PostMessage 发 Ctrl+V 到指定窗口。

        优点：不依赖窗口处于前台，不干扰用户操作。
        """
        if not text:
            return False
        try:
            import win32clipboard
            import win32con
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            win32clipboard.CloseClipboard()
        except ImportError:
            self._set_clipboard_python(text)
            time.sleep(0.05)

        self._post_message(hwnd, WM_KEYDOWN, VK_CONTROL, 0)
        time.sleep(0.02)
        self._post_message(hwnd, WM_KEYDOWN, ord('V'), 0)
        time.sleep(0.02)
        self._post_message(hwnd, WM_KEYUP, ord('V'), 0)
        time.sleep(0.02)
        self._post_message(hwnd, WM_KEYUP, VK_CONTROL, 0)
        return True

    def _send_input(self, inputs: list) -> int:
        """Call user32.SendInputW with a list of INPUT structs."""
        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        return self._user32.SendInput(n, arr, ctypes.sizeof(INPUT))

    def send_mouse_click(self, x: int, y: int,
                         button: str = "left") -> bool:
        """后台点击（兼容旧调用）：屏幕坐标 → 后台 PostMessage。
        不移动真实鼠标、不抢焦点。
        """
        from .background_clicker import background_click_screen as _bg_click_screen
        return _bg_click_screen(x, y, double=(button == "double"))

    def send_key_press(self, vk_code: int, extended: bool = False) -> bool:
        """Send a single key press-down and release via SendInput."""
        scan = self._user32.MapVirtualKeyW(vk_code, 0)
        flags_down = 0
        flags_up = KEYEVENTF_KEYUP
        if extended:
            flags_down |= 0x0001  # KEYEVENTF_EXTENDEDKEY
            flags_up |= 0x0001

        inputs = [
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_code, wScan=scan, dwFlags=flags_down,
                time=0, dwExtraInfo=None)),
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_code, wScan=scan, dwFlags=flags_up,
                time=0, dwExtraInfo=None)),
        ]
        result = self._send_input(inputs)
        return result == len(inputs)

    def send_key_combo(self, vk_mod: int, vk_key: int) -> bool:
        """Send modifier + key combination (e.g. Ctrl+V)."""
        scan_mod = self._user32.MapVirtualKeyW(vk_mod, 0)
        scan_key = self._user32.MapVirtualKeyW(vk_key, 0)

        inputs = [
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_mod, wScan=scan_mod, dwFlags=0,
                time=0, dwExtraInfo=None)),
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_key, wScan=scan_key, dwFlags=0,
                time=0, dwExtraInfo=None)),
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_key, wScan=scan_key, dwFlags=KEYEVENTF_KEYUP,
                time=0, dwExtraInfo=None)),
            INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                wVk=vk_mod, wScan=scan_mod, dwFlags=KEYEVENTF_KEYUP,
                time=0, dwExtraInfo=None)),
        ]
        result = self._send_input(inputs)
        return result == len(inputs)

    def send_unicode_text(self, text: str, delay_range=(0.02, 0.08)) -> None:
        """Type unicode characters via KEYEVENTF_UNICODE for CJK support."""
        extra = ctypes.POINTER(wintypes.ULONG)()
        for char in text:
            code = ord(char)
            inputs = [
                INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                    wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE,
                    time=0, dwExtraInfo=extra)),
                INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(
                    wVk=0, wScan=code,
                    dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                    time=0, dwExtraInfo=extra)),
            ]
            self._send_input(inputs)
            time.sleep(random.uniform(*delay_range))

    def _set_clipboard_python(self, text: str) -> None:
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, (len(text) + 1) * 2)
        ptr = kernel32.GlobalLock(handle)
        ctypes.cdll.msvcrt.wcscpy(ctypes.c_wchar_p(ptr), text)
        kernel32.GlobalUnlock(handle)

        if not user32.OpenClipboard(0):
            return
        user32.EmptyClipboard()
        user32.SetClipboardData(CF_UNICODETEXT, handle)
        user32.CloseClipboard()

    def send_message(self, contact: str, text: str,
                     hwnd: Optional[int] = None,
                     use_clipboard: bool = True) -> RPASendResult:
        target = hwnd or self._hwnd_wechat
        if not target:
            return RPASendResult(success=False, error="No WeChat window")

        try:
            self.activate_window(target)
            time.sleep(0.1)

            if not self._open_chat_by_search(target, contact):
                return RPASendResult(success=False, error=f"Cannot find contact: {contact}")

            time.sleep(0.3)

            if use_clipboard:
                self._send_text_via_clipboard(target, text)
            else:
                self._send_text_typing(target, text)

            time.sleep(0.15)

            self.send_key_press(VK_RETURN)

            time.sleep(0.15)
            return RPASendResult(success=True, message=f"Sent to {contact}: {text[:50]}...")

        except Exception as e:
            return RPASendResult(success=False, error=str(e))

    def _open_chat_by_search(self, hwnd: int, contact: str) -> bool:
        self.send_key_combo(VK_CONTROL, ord('F'))

        time.sleep(0.3)

        try:
            self._send_text_via_clipboard(hwnd, contact)
        except Exception:
            return False

        time.sleep(0.8)

        self.send_key_press(VK_RETURN)

        time.sleep(0.2)
        return True

    def open_chat_by_contact(self, contact: str,
                             hwnd: Optional[int] = None) -> bool:
        """打开与指定联系人的会话。

        与 send_message 的 _open_chat_by_search 共用逻辑，但只在已打开该会话后返回
        （供「未读识别→进入会话→读取消息」步骤使用，不执行发送）。
        """
        target = hwnd or self._hwnd_wechat
        if not target:
            return False

        # 虚拟外屏模式下窗口仍处于正常渲染状态，无需置前台；仅为保险还原一次
        try:
            SW_RESTORE = 9
            self._user32.ShowWindow(target, SW_RESTORE)
        except Exception:
            pass
        return self._open_chat_by_search(target, contact)

    def send_in_current(self, hwnd: int, text: str,
                        use_clipboard: bool = True,
                        confirm_send: bool = True,
                        prefer_background: bool = True) -> RPASendResult:
        """在当前已打开的会话里直接发送消息（不搜索联系人）。

        流程：后台 PostMessage 点击输入框 → 后台粘贴 → 后台回车 → 截屏 diff 确认。
        prefer_background=True 时用 PostMessage 全链路后台操作，不干扰用户；
        失败时自动降级到前台 SendInput。
        confirm_send=True 时用 SendConfirm 截屏比对，只有画面真变了才算成功。
        """
        if not hwnd or not text:
            return RPASendResult(success=False, error="No window or empty text")

        try:
            # -- 准备 SendConfirm（失败也不阻塞发送）--
            confirmer = None
            before_frame = None
            if confirm_send:
                try:
                    from .send_confirm import SendConfirm
                    from ..capture.screen_capture import ScreenCapture
                    confirmer = SendConfirm(ScreenCapture())
                    before_frame = confirmer.capture(hwnd)
                except Exception:
                    confirmer = None

            # 使用客户区尺寸（PostMessage 用客户区坐标）
            client_rect = wintypes.RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
            cw = client_rect.right - client_rect.left
            ch = client_rect.bottom - client_rect.top
            if cw <= 0 or ch <= 0:
                return RPASendResult(success=False, error="Invalid window size")

            ix = int(cw * 0.55)
            iy = int(ch * 0.955)

            if prefer_background:
                # 后台 PostMessage 链路：不依赖窗口前台
                self.post_message_click(hwnd, ix, iy)
                time.sleep(0.25)
                if use_clipboard:
                    self._send_text_via_clipboard_post(hwnd, text)
                else:
                    self._send_text_typing(hwnd, text)
                time.sleep(0.15)
                self._post_message(hwnd, WM_KEYDOWN, VK_RETURN, 0)
                self._post_message(hwnd, WM_KEYUP, VK_RETURN, 0)
                time.sleep(0.25)
            else:
                # 前台 SendInput 链路（兼容旧行为）
                self.background_click(hwnd, ix, iy)
                time.sleep(0.35)
                if use_clipboard:
                    self._send_text_via_clipboard(hwnd, text)
                else:
                    self._send_text_typing(hwnd, text)
                time.sleep(0.15)
                self.send_key_press(VK_RETURN)
                time.sleep(0.15)

            # ---- 截屏 diff 确认真正上屏 ----
            if confirmer is not None and before_frame is not None:
                confirm = confirmer.compare(
                    before=before_frame,
                    after=confirmer.capture(hwnd),
                    region=None,
                    min_diff_ratio=0.003)
                if not confirm.success:
                    if prefer_background:
                        # 后台未确认 → 降级前台重试一次
                        logger = logging.getLogger("WeChatRPA")
                        logger.info("Background send not confirmed, fallback to foreground")
                        return self.send_in_current(
                            hwnd, text, use_clipboard=use_clipboard,
                            confirm_send=True, prefer_background=False)
                    return RPASendResult(
                        success=False,
                        error=(f"send_not_confirmed diff_ratio="
                               f"{confirm.diff_ratio} "
                               f"({confirm.error or '画面无明显变化'})"))

            return RPASendResult(success=True,
                                 message=f"Confirmed sent: {text[:50]}...")
        except Exception as e:
            return RPASendResult(success=False, error=str(e))

    def send_message_confirmed(self, contact: str, text: str,
                               hwnd: Optional[int] = None) -> RPASendResult:
        """Send with clipboard paste + screenshot-diff confirmation.

        Used by ObserveService fallback path where search-by-name is needed.
        """
        target = hwnd or self._hwnd_wechat
        if not target:
            return RPASendResult(success=False, error="No WeChat window")
        return self.send_in_current(target, text, use_clipboard=True,
                                     confirm_send=True)

    # =====================================================================
    # 未读会话点击链路 —— 对齐原版 RedDotDetector._execute_click 流程：
    #   1. 首选：PostMessage 后台点击（不动光标、不打扰用户）
    #   2. 验证：0.35s 后截屏检查该行是否变绿（选中态）
    #   3. 兜底：仅当配置允许时才使用前台 SendInput 人性化点击
    # =====================================================================

    def post_message_click(self, hwnd: int, client_x: int, client_y: int,
                            double: bool = False) -> bool:
        """后台点击：WM_MOUSEMOVE → LBUTTONDOWN → LBUTTONUP，不动用户鼠标。"""
        try:
            lparam = (int(client_y) << 16) | (int(client_x) & 0xFFFF)
            self._post_message(hwnd, WM_MOUSEMOVE, 0, lparam)
            time.sleep(random.uniform(0.03, 0.05))
            self._post_message(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
            time.sleep(random.uniform(0.02, 0.04))
            self._post_message(hwnd, WM_LBUTTONUP, 0, lparam)
            if double:
                time.sleep(random.uniform(
                    getattr(type(self), "POST_DBL_MIN", 0.12),
                    getattr(type(self), "POST_DBL_MAX", 0.28)))
                self._post_message(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                time.sleep(0.03)
                self._post_message(hwnd, WM_LBUTTONUP, 0, lparam)
            return True
        except Exception:
            return False

    @staticmethod
    def is_list_row_selected(bgr_frame, cy_client: int) -> bool:
        """Check if a sidebar row shows the selected-state green tint.

        Restored from original `_is_list_row_selected`:
          Scan pixels in region [x∈(45, LIST_X_END-50), y∈(cy-24, cy+32)].
          A "green" pixel satisfies:
             g >= 120, r <= 90, b <= 160, g-r >= 45, g-b >= 15
          Returns True when green_ratio >= 0.25.
        """
        try:
            h, w = bgr_frame.shape[:2]
            x_start = 45
            x_end = min(w, WeChatRPA.LIST_X_END - 50)
            y_start = max(0, int(cy_client) - 24)
            y_end = min(h, int(cy_client) + 32)
            if x_end <= x_start or y_end <= y_start:
                return False
            crop = bgr_frame[y_start:y_end, x_start:x_end]
            b = crop[:, :, 0].astype(np.int16)
            g = crop[:, :, 1].astype(np.int16)
            r = crop[:, :, 2].astype(np.int16)
            total = crop.shape[0] * crop.shape[1]
            if total == 0:
                return False
            green_mask = ((g >= 120) & (r <= 90) &
                          (b <= 160) &
                          ((g - r) >= 45) & ((g - b) >= 15))
            green_count = int(green_mask.sum())
            return (green_count / float(total)) >= 0.25
        except Exception:
            return False

    def _capture_fresh_frame(self, hwnd: Optional[int]):
        """Take a screenshot of hwnd's full window (BGR numpy). None on fail."""
        try:
            from ..capture.screen_capture import ScreenCapture
            cap = ScreenCapture()
            result = cap.capture_window(int(hwnd))
            if result.success and result.image is not None:
                return result.image
        except Exception:
            pass
        return None

    def open_chat_at(self, hwnd: int, client_x: int, client_y: int,
                      verify_enabled: bool = True) -> bool:
        """Open unread chat at sidebar coordinates using original strategy.

        全程【后台 PostMessage】点击，绝不降级到前台真实鼠标（SendInput）。
        后台点击后等待验证；验证失败也仅返回后台结果，不移动真实光标、
        不抢焦点（前台 / 后台 / 虚拟屏均可点）。
        """
        if not hwnd:
            return False
        old_hwnd = self._hwnd_wechat
        self._hwnd_wechat = hwnd
        try:
            # 转换：client_y 已是红点中心；应用原版的 Y_OFFSET。
            cy_target = int(client_y) + self.LIST_CLICK_Y_OFFSET
            x_target = self.LIST_CLICK_X     # 固定客户区 X=100

            # ---- 后台 PostMessage 点击（唯一点击方式）----
            ok_bg = self.post_message_click(hwnd, x_target, cy_target)
            if ok_bg and verify_enabled:
                time.sleep(self.VERIFY_DELAY_SECONDS)
                frame = self._capture_fresh_frame(hwnd)
                if frame is not None and \
                   self.is_list_row_selected(frame, int(client_y)):
                    return True      # 该行已高亮 → 成功
            return bool(ok_bg)
        except Exception:
            return False
        finally:
            self._hwnd_wechat = old_hwnd

    def send_mouse_click_abs_from_client(self, hwnd: int,
                                          client_x: int,
                                          client_y: int) -> bool:
        """后台 PostMessage 点击（原前台 SendInput 已改为后台，不移动真实鼠标）。

        前台 / 后台 / 虚拟屏均可点。
        """
        from .background_clicker import background_click as _bg_click
        return _bg_click(hwnd, client_x, client_y)

    def _send_text_typing(self, hwnd: int, text: str) -> None:
        """Hardware-level unicode typing using SendInput KEYEVENTF_UNICODE."""
        old_hwnd = self._hwnd_wechat
        self._hwnd_wechat = hwnd
        try:
            self.send_unicode_text(text)
        finally:
            self._hwnd_wechat = old_hwnd

    def _char_to_vk(self, char: str) -> Optional[int]:
        vk = self._user32.VkKeyScanW(ord(char))
        if vk == -1:
            return None
        return vk & 0xFF

    def send_segmented(self, contact: str, segments: List[str],
                      hwnd: Optional[int] = None) -> List[RPASendResult]:
        results = []
        target = hwnd or self._hwnd_wechat

        for i, segment in enumerate(segments):
            if self._stop_event.is_set():
                break

            result = self.send_message(contact, segment, target, use_clipboard=True)
            results.append(result)

            if i < len(segments) - 1:
                delay = random.uniform(0.8, 1.8)
                time.sleep(delay)

        return results

    def background_click(self, hwnd: int, x: int, y: int,
                          button: str = "left") -> bool:
        """后台 PostMessage 点击（不移动真实鼠标、不抢焦点）。

        向客户区坐标直接 PostMessage，前台 / 后台 / 虚拟屏均可点。
        注：此方法名 historical 上曾用 SendInput 硬件级点击（会移动真实光标），
        现已统一为后台 PostMessage，符合全局「不碰真实鼠标」策略。
        """
        from .background_clicker import background_click as _bg_click
        return _bg_click(hwnd, x, y, double=(button == "double"))

    def send_mouse_click_abs(self, sm_x: int, sm_y: int,
                              button: str = "left") -> bool:
        """后台点击（兼容旧调用）。

        把归一化 [0,65535] 虚拟桌面坐标还原为屏幕坐标，再走后台 PostMessage。
        不移动真实鼠标、不抢焦点。
        """
        from .background_clicker import background_click_screen as _bg_click_screen
        try:
            SM_XVIRT, SM_YVIRT = 76, 77
            SM_CXVIRT, SM_CYVIRT = 78, 79
            user32 = ctypes.windll.user32
            vx = user32.GetSystemMetrics(SM_XVIRT)
            vy = user32.GetSystemMetrics(SM_YVIRT)
            cw = user32.GetSystemMetrics(SM_CXVIRT)
            chh = user32.GetSystemMetrics(SM_CYVIRT)
            if cw <= 0 or chh <= 0:
                return False
            screen_x = int(sm_x / 65535.0 * cw) + vx
            screen_y = int(sm_y / 65535.0 * chh) + vy
            return _bg_click_screen(screen_x, screen_y, double=(button == "double"))
        except Exception:
            return False

    def stop(self) -> None:
        self._stop_event.set()
        self._typing_active = False

    def reset(self) -> None:
        self._stop_event.clear()

    # =====================================================================
    # 发送链安全守卫
    # =====================================================================

    def seconds_since_last_input(self) -> float:
        """检查用户最后输入时间（键盘/鼠标），返回距上次输入的秒数。"""
        try:
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if self._user32.GetLastInputInfo(ctypes.byref(lii)):
                tick = self._user32.GetTickCount()
                return max(0.0, (tick - lii.dwTime) / 1000.0)
        except Exception:
            pass
        return 999.0

    def is_user_idle(self, min_idle_seconds: float = 1.0) -> bool:
        """用户是否空闲（未操作键盘鼠标）。"""
        return self.seconds_since_last_input() >= min_idle_seconds

    def _clear_input_before_send(self, hwnd: int) -> bool:
        """清理输入框中的残留内容（Ctrl+A → Backspace）。"""
        try:
            self._post_message(hwnd, WM_KEYDOWN, VK_CONTROL, 0)
            self._post_message(hwnd, WM_KEYDOWN, ord('A'), 0)
            self._post_message(hwnd, WM_KEYUP, ord('A'), 0)
            self._post_message(hwnd, WM_KEYUP, VK_CONTROL, 0)
            time.sleep(0.05)
            self._post_message(hwnd, WM_KEYDOWN, VK_BACK, 0)
            self._post_message(hwnd, WM_KEYUP, VK_BACK, 0)
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def _try_close_stale_input_overlay(self, hwnd: int) -> bool:
        """尝试关闭输入框上残留的浮层（表情面板、草稿提示等）。"""
        try:
            self._post_message(hwnd, WM_KEYDOWN, VK_ESCAPE, 0)
            self._post_message(hwnd, WM_KEYUP, VK_ESCAPE, 0)
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def _foreground_clear_draft_and_send(self, hwnd: int, text: str) -> RPASendResult:
        """前台模式：清理草稿 → 粘贴 → 回车（完整前台降级路径）。"""
        try:
            self.activate_window(hwnd)
            time.sleep(0.15)
            rect = wintypes.RECT()
            self._user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                return RPASendResult(success=False, error="Invalid window size")
            ix = int(w * 0.55)
            iy = int(h * 0.955)
            self.send_mouse_click_abs_from_client(hwnd, ix, iy)
            time.sleep(0.2)
            self._clear_input_before_send(hwnd)
            time.sleep(0.1)
            self._send_text_via_clipboard(hwnd, text)
            time.sleep(0.15)
            self.send_key_press(VK_RETURN)
            time.sleep(0.15)
            return RPASendResult(success=True, message=f"Sent foreground: {text[:50]}...")
        except Exception as e:
            return RPASendResult(success=False, error=str(e))

    def _background_send_segment(self, hwnd: int, text: str) -> RPASendResult:
        """后台 PostMessage 发送：清理 → 粘贴 → 回车，不干扰用户。"""
        try:
            self._try_close_stale_input_overlay(hwnd)
            self._clear_input_before_send(hwnd)
            time.sleep(0.1)
            self._send_text_via_clipboard_post(hwnd, text)
            time.sleep(0.15)
            self._post_message(hwnd, WM_KEYDOWN, VK_RETURN, 0)
            self._post_message(hwnd, WM_KEYUP, VK_RETURN, 0)
            time.sleep(0.15)
            return RPASendResult(success=True, message=f"Sent background: {text[:50]}...")
        except Exception as e:
            return RPASendResult(success=False, error=str(e))

    def send_with_strategy(
        self, hwnd: int, text: str,
        prefer_background: bool = True,
        confirm_send: bool = True,
    ) -> RPASendResult:
        """多级发送策略：后台 PostMessage 优先 → 前台 SendInput 降级 → 截屏确认。

        现在统一走 send_in_current，内部已含后台/前台降级和截屏确认逻辑，
        避免旧版"先发再确认"造成的双重发送问题。
        """
        if not hwnd or not text:
            return RPASendResult(success=False, error="No window or empty text")

        segments = self._split_long_message(text)
        results: list[RPASendResult] = []

        for i, segment in enumerate(segments):
            if i > 0:
                time.sleep(random.uniform(0.8, 1.5))

            result = self.send_in_current(
                hwnd, segment,
                use_clipboard=True,
                confirm_send=confirm_send,
                prefer_background=prefer_background)

            if not result.success:
                return result

            results.append(result)

        if not results:
            return RPASendResult(success=False, error="No segments sent")
        return RPASendResult(success=True, message=f"Sent {len(results)} segments")

    def _split_long_message(self, text: str) -> list[str]:
        """将长消息按换行分段，超长段再按句号拆分。"""
        if len(text) <= 600:
            return [text]

        segments: list[str] = []
        lines = text.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) > 500:
                if current:
                    segments.append(current.strip())
                current = line
            else:
                current = current + "\n" + line if current else line

        if current:
            final: list[str] = []
            for seg in segments + [current]:
                if len(seg) <= 600:
                    final.append(seg.strip())
                else:
                    parts = seg.split("。")
                    buf = ""
                    for p in parts:
                        if len(buf) + len(p) > 500:
                            if buf:
                                final.append(buf.strip() + "。")
                            buf = p
                        else:
                            buf = buf + "。" + p if buf else p
                    if buf:
                        final.append(buf.strip() + "。")
            segments = final

        return [s for s in segments if s.strip()]


# ---------------------------------------------------------------------------
# 模块级兼容包装函数
# observe_service.py 按模块级函数名调用以下接口，下方封装 WeChatRPA 实例方法，
# 保持旧接口可用，避免 ImportError。如需精确坐标（红点检测），可直接用
# WeChatRPA().open_chat_at(hwnd, x, y)。
# ---------------------------------------------------------------------------

def click_unread_chat(hwnd: Optional[int] = None) -> dict:
    """点击微信聊天列表中第一个未读会话。

    默认布局下未读会话位于聊天列表首行；若有红点检测坐标可传入 hwnd 后改用
    WeChatRPA.open_chat_at 精确定位。
    """
    rpa = WeChatRPA()
    target = hwnd or rpa.find_wechat_window()
    if not target:
        return {"ok": False, "error": "no wechat window"}
    rpa.set_wechat_hwnd(target)
    rpa.activate_window(target)
    # 聊天列表首行（窗口左侧，顶部偏下）
    rpa.background_click(target, 110, 140)
    return {"ok": True, "hwnd": target}


def click_file_transfer(hwnd: Optional[int] = None) -> dict:
    """打开「文件传输助手」会话。"""
    rpa = WeChatRPA()
    target = hwnd or rpa.find_wechat_window()
    if not target:
        return {"ok": False, "error": "no wechat window"}
    rpa.set_wechat_hwnd(target)
    ok = rpa.open_chat_by_contact("文件传输助手", target)
    return {"ok": bool(ok), "hwnd": target}


def send_text(text: str, hwnd: Optional[int] = None) -> dict:
    """在当前已打开的会话中发送文本（带多级降级策略）。"""
    if not text:
        return {"ok": False, "sent": False, "error": "empty text"}
    rpa = WeChatRPA()
    target = hwnd or rpa.find_wechat_window()
    if not target:
        return {"ok": False, "sent": False, "error": "no wechat window"}
    rpa.set_wechat_hwnd(target)
    res = rpa.send_with_strategy(target, text, prefer_background=True, confirm_send=True)
    return {"ok": res.success, "sent": res.success, "error": res.error}