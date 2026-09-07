"""微信窗口控制：激活 / 置顶 / 移动 / 隐藏。

全部基于 win32 API（ctypes 直接调用 user32），不依赖额外第三方库。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Optional

from ..capture.screen_capture import WindowInfo
from .window_finder import WindowFinder


# ShowWindow nCmdShow 常量
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_MINIMIZE = 6
SW_RESTORE = 9

# SetWindowPos flags
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_NOOWNERZORDER = 0x0200
SWP_FRAMECHANGED = 0x0020
SWP_NOZORDER = 0x0004

# 窗口样式 / 状态常量
GWL_STYLE = -16
WS_MINIMIZE = 0x20000000

# 屏幕外保活模式（移植自 wechat-ai-reply-main）：窗口移到虚拟屏外坐标，
# 保持「可见但屏外」(SW_SHOWNOACTIVATE)，全程不回物理桌面，用户零感知。
OFFSCREEN_X = -10000
OFFSCREEN_Y = 0

# 最小化恢复所需的窗口位置结构（ctypes.wintypes 未导出，这里自定义）
class _WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("showCmd", ctypes.c_uint),
        ("ptMinPosition", ctypes.wintypes.POINT),
        ("ptMaxPosition", ctypes.wintypes.POINT),
        ("rcNormalPosition", ctypes.wintypes.RECT),
    ]


# GetSystemMetrics 索引
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class WeChatWindowManager:
    def __init__(self, store=None, keywords=None,
                 finder: Optional[WindowFinder] = None):
        self._user32 = ctypes.windll.user32
        self._store = store
        self._keywords = keywords or ["微信"]
        self._finder = finder or WindowFinder()
        self._hwnd_wechat: Optional[int] = None
        self._saved_rect: Optional[tuple] = None   # (x, y, w, h) park 前的原始位置
        self._native_size: Optional[tuple] = None  # (w, h) 用户原始正常尺寸（只增不减）
        self._parked: bool = False
        self._offscreen_original: dict = {}        # hwnd -> (x, y, w, h) 用户原可见位置
        self._ctypes_ready: bool = False
        self.OFFSCREEN_X = -10000                    # 屏外坐标（实例属性，供方法 self.OFFSCREEN_X 访问）
        self.OFFSCREEN_Y = 0

    def set_wechat_hwnd(self, hwnd: Optional[int]) -> None:
        self._hwnd_wechat = hwnd

    def find_wechat_window(self) -> Optional[int]:
        """查找微信窗口，返回 hwnd（对齐原版 find_wechat_window）。"""
        win = self._finder.find_wechat()
        if win and win.hwnd:
            self._hwnd_wechat = win.hwnd
            return win.hwnd
        return None

    def _resolve(self, hwnd: Optional[int]) -> Optional[int]:
        target = hwnd or self._hwnd_wechat
        if target:
            return target
        win = self._finder.find_wechat()
        if win and win.hwnd:
            self._hwnd_wechat = win.hwnd
            return win.hwnd
        return None

    # -------------------------------------------------- 状态
    def get_rect(self, hwnd: Optional[int] = None) -> Optional[WindowInfo]:
        target = self._resolve(hwnd)
        if not target:
            return None
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(target, ctypes.byref(rect)):
            return None
        return WindowInfo(
            hwnd=target,
            title="",
            width=rect.right - rect.left,
            height=rect.bottom - rect.top,
            x=rect.left,
            y=rect.top,
        )

    def is_visible(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        return bool(target and self._user32.IsWindowVisible(target))

    # -------------------------------------------------- 激活
    def activate(self, hwnd: Optional[int] = None) -> bool:
        """还原并置为前台。"""
        target = self._resolve(hwnd)
        if not target:
            return False
        self._user32.ShowWindow(target, SW_RESTORE)
        return bool(self._user32.SetForegroundWindow(target))

    def focus(self, hwnd: Optional[int] = None) -> bool:
        return self.activate(hwnd)

    # -------------------------------------------------- 置顶
    def topmost(self, hwnd: Optional[int] = None, enable: bool = True) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        insert_after = HWND_TOPMOST if enable else HWND_NOTOPMOST
        return bool(self._user32.SetWindowPos(
            target, insert_after, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        ))

    def untopmost(self, hwnd: Optional[int] = None) -> bool:
        return self.topmost(hwnd, enable=False)

    # -------------------------------------------------- 移动
    def move(self, x: int, y: int, width: int, height: int,
             hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.MoveWindow(target, x, y, width, height, True))

    # -------------------------------------------------- 显隐
    def hide(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.ShowWindow(target, SW_HIDE))

    def show(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.ShowWindow(target, SW_SHOW))

    def minimize(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.ShowWindow(target, SW_MINIMIZE))

    def restore(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.ShowWindow(target, SW_RESTORE))

    def maximize(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.ShowWindow(target, SW_SHOWMAXIMIZED))

    # -------------------------------------------------- 虚拟外屏 (park/unpark)
    def get_virtual_screen(self) -> tuple:
        """虚拟桌面边界 (x, y, w, h)。"""
        x = self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        y = self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        return (x, y, w, h)

    def is_minimized(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve(hwnd)
        if not target:
            return False
        return bool(self._user32.IsIconic(target))

    def _remember_restore_position(self, target: int) -> None:
        """用 GetWindowPlacement 记录「还原到桌面」时的正常位置矩形。

        最小化时 GetWindowRect 拿到的是任务栏槽位矩形，不能作为还原位置，
        所以优先用 WM_PLACEMENT.rcNormalPosition（最小化前/后的正常窗口矩形）。
        """
        if self._saved_rect is not None:
            return
        try:
            placement = _WINDOWPLACEMENT()
            placement.length = ctypes.sizeof(_WINDOWPLACEMENT)
            if self._user32.GetWindowPlacement(target, ctypes.byref(placement)):
                r = placement.rcNormalPosition
                if r.right - r.left > 0 and r.bottom - r.top > 0:
                    self._saved_rect = (r.left, r.top,
                                        r.right - r.left, r.bottom - r.top)
                    return
        except Exception:
            pass
        rect = self.get_rect(target)
        if rect:
            self._saved_rect = (rect.x, rect.y, rect.width, rect.height)

    def _remember_native_size(self, target: int) -> tuple:
        """记住用户窗口的『原始正常尺寸』(w, h)，防止被瞬态小尺寸永久缩小。

        规则（只增不减）：
          - 仅当窗口当前处于『正常可见态』(非最小化、尺寸合理) 才用其尺寸更新；
          - 取历史最大值，因此即便某瞬间读到最小化槽位(276x45)或上次被缩小的
            小窗(867x824)，也不会把 _native_size 越改越小；
          - 同时用 OS 的 rcNormalPosition 作种子（用户拖动调整过的正常尺寸），
            二者取大，覆盖绝大多数场景。
        任何 park/unpark/click/restore 需要尺寸时都走这里，保证窗口永不被
        bot 主动缩成小窗。
        """
        # 1) 种子：OS 记住的正常尺寸
        try:
            wp = _WINDOWPLACEMENT()
            wp.length = ctypes.sizeof(_WINDOWPLACEMENT)
            if self._user32.GetWindowPlacement(target, ctypes.byref(wp)):
                r = wp.rcNormalPosition
                rw, rh = r.right - r.left, r.bottom - r.top
                if rw > 300 and rh > 200:
                    if self._native_size is None:
                        self._native_size = (rw, rh)
                    else:
                        self._native_size = (max(self._native_size[0], rw),
                                             max(self._native_size[1], rh))
        except Exception:
            pass
        # 2) 当前正常可见态：取较大值（避免小窗污染）
        try:
            if not self.is_minimized(target):
                rect = self.get_rect(target)
                if rect and rect.width > 300 and rect.height > 200:
                    if self._native_size is None:
                        self._native_size = (rect.width, rect.height)
                    else:
                        self._native_size = (max(self._native_size[0], rect.width),
                                             max(self._native_size[1], rect.height))
        except Exception:
            pass
        return self._native_size or (900, 680)

    def _virtual_screen(self) -> tuple:
        return self.get_virtual_screen()

    def _is_on_screen(self, rect: WindowInfo) -> bool:
        """判断窗口是否在可见屏幕区域内（至少 30% 面积可见）。"""
        vs_x, vs_y, vs_w, vs_h = self.get_virtual_screen()
        # 窗口与屏幕的交集
        overlap_x = max(0, min(rect.x + rect.width, vs_x + vs_w) - max(rect.x, vs_x))
        overlap_y = max(0, min(rect.y + rect.height, vs_y + vs_h) - max(rect.y, vs_y))
        overlap_area = overlap_x * overlap_y
        window_area = rect.width * rect.height
        if window_area <= 0:
            return False
        return overlap_area >= window_area * 0.3

    def park(self, x: int = None, y: int = 60,
             keep_size: bool = True, hwnd: Optional[int] = None) -> bool:
        """把微信移动到虚拟屏幕外的后台区（负坐标/越界），保持正常渲染后台化。

        与最小化的本质区别：最小化后客户区不再绘制，PrintWindow 抓到的是空白；
        而把窗口移到屏幕外，窗口仍保持前台绘制状态，PrintWindow/OCR 可读到真实画面，
        同时用户桌面上看不见微信窗口。这就是原版"虚拟外屏"思路。
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        self._remember_restore_position(target)

        # 若在最小化状态：转为「屏外可见」态（不闪桌面）。
        # 直接 SetWindowPos 到屏外 + SW_SHOWNOACTIVATE，避免在桌面上 SW_RESTORE 闪现。
        if self.is_minimized(target):
            return self.move_window_offscreen(target)

        w, h = self._remember_native_size(target)
        width, height = w, h

        # 目标位置：放在虚拟屏最左侧越界处（默认单屏 x 起点=0，取负坐标即完全移出屏幕）
        vs_x, vs_y, vs_w, vs_h = self.get_virtual_screen()
        off_x = x if x is not None else vs_x - width - 120    # e.g. -1020
        off_y = y                                             # 顶部，保留一点边缘

        ok = bool(self._user32.SetWindowPos(
            target, 0, off_x, off_y, width, height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW))
        if ok:
            self._parked = True
        return ok

    def ensure_capturable(self, hwnd: Optional[int] = None) -> bool:
        """状态感知：保证能读到微信真实画面，同时尊重用户当前的窗口状态。

        - 微信正常显示在桌面 → 原地识别，不移动、不隐藏（用户说别动就别动）。
        - 微信被最小化 → 借助虚拟外屏：还原后移到屏幕外(负坐标)后台化渲染，
          用户桌面看不见、但 PrintWindow 仍然能读到实时画面，即"最小化识别"。
        - 用户已手动把窗口恢复到桌面 → 视为用户要用，放弃外屏状态，专心原地识别。
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        rect = self.get_rect(target)
        on_screen = rect is not None and self._is_on_screen(rect)

        if on_screen and not self.is_minimized(target):
            # 桌面正常显示：原地识别，绝不移动
            self._parked = False
            self._saved_rect = None
            return True

        if self.is_minimized(target):
            # 最小化：进入虚拟外屏后台化渲染
            self._parked = False
            return self.park(hwnd=target)

        # 已在外屏(负坐标)后台：保持现状
        self._parked = True
        return True

    def ensure_parked(self, hwnd: Optional[int] = None,
                      attempts: int = 3) -> bool:
        """确保微信处于虚拟外屏(屏幕外)后台；若被还原到桌面则重新移出去。

        应对微信自身或系统把窗口拉回桌面/还原的情况：每次截图前调用，
        若发现仍在屏幕上或已最小化，则重新 park 并校验位置。
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        for _ in range(attempts):
            if self.is_minimized(target) or not self._parked:
                if not self.park(hwnd=target):
                    return False
            rect = self.get_rect(target)
            if rect is None:
                break
            if not self._is_on_screen(rect):
                self._parked = True
                return True
            # 仍在屏幕上：重新移出去
            self._parked = False
            if not self.park(hwnd=target):
                return False
            time.sleep(0.2)
        # 最后一次校验
        rect = self.get_rect(target)
        if rect is None:
            return self._parked
        if not self._is_on_screen(rect):
            self._parked = True
            return True
        return False

    def unpark(self, hwnd: Optional[int] = None) -> bool:
        """把微信从虚拟外屏恢复到原始屏幕位置。

        若 _saved_rect 丢失（如 ensure_capturable 在 park 前清空了记录），
        则回退到主屏居中放置，确保窗口不会卡在屏幕外无法恢复。
        """
        target = self._resolve(hwnd)
        if not target:
            return False
        self._user32.ShowWindow(target, SW_RESTORE)
        if self._saved_rect is not None:
            sx, sy, sw, sh = self._saved_rect
            self._saved_rect = None
            self._user32.MoveWindow(target, sx, sy, sw, sh, True)
        else:
            # 兜底：主屏居中放置，防止窗口永远卡在负坐标外屏。
            # 必须用主屏指标 SM_CXSCREEN/SM_CYSCREEN（恒为非负），
            # 不能用虚拟屏指标（多屏起点可能为负，会把窗口居中到屏外）。
            try:
                pw = self._user32.GetSystemMetrics(0)   # SM_CXSCREEN
                ph = self._user32.GetSystemMetrics(1)   # SM_CYSCREEN
                rect = self.get_rect(target)
                w, h = self._remember_native_size(target)
                new_x = max(0, (pw - w) // 2)
                new_y = max(0, (ph - h) // 2)
                self._user32.MoveWindow(target, new_x, new_y, w, h, True)
            except Exception:
                pass
        self._parked = False
        return True

    def show_window(self, hwnd: Optional[int] = None) -> Optional[dict]:
        """将微信恢复到桌面可见区域（供 UI「显示微信」按钮调用）。

        优先恢复到 park 前记录的原始位置；记录丢失时回退到主显示器居中，
        用主屏指标 SM_CXSCREEN/SM_CYSCREEN（恒为非负）计算，避免多屏虚拟屏
        起点为负导致居中落到屏外（这是此前「微信回不到桌面」的根因）。

        Returns: 最终窗口矩形 dict（含 ok/x/y/width/height），失败返回 None。
        """
        target = self._resolve(hwnd)
        if not target:
            return None

        # 先确保窗口可见（最小化/隐藏都拉起）
        self._user32.ShowWindow(target, SW_RESTORE)

        try:
            rect = self.get_rect(target)
            w, h = self._remember_native_size(target)
        except Exception:
            w, h = 900, 680

        placed = False
        new_x = new_y = 0
        if self._saved_rect is not None:
            sx, sy, sw, sh = self._saved_rect
            try:
                self._user32.MoveWindow(target, sx, sy, sw, sh, True)
                placed = True
            except Exception:
                placed = False
            self._saved_rect = None

        if not placed:
            # 主屏居中（主屏指标 SM_CXSCREEN=0 / SM_CYSCREEN=1 恒为非负，
            # 不会因多屏虚拟屏负坐标而落到屏外）
            try:
                pw = self._user32.GetSystemMetrics(0)
                ph = self._user32.GetSystemMetrics(1)
                new_x = max(0, (pw - w) // 2)
                new_y = max(0, (ph - h) // 2)
                self._user32.MoveWindow(target, new_x, new_y, w, h, True)
                placed = True
            except Exception:
                placed = False

        # 激活到前台，确保用户能看见
        try:
            self._user32.SetForegroundWindow(target)
        except Exception:
            pass

        self._parked = False
        final = self.get_rect(target)
        return {
            "ok": placed,
            "x": final.x if final else new_x,
            "y": final.y if final else new_y,
            "width": final.width if final else w,
            "height": final.height if final else h,
        }

    def restore_offscreen(self, hwnd: Optional[int] = None) -> bool:
        """把最小化窗口恢复到「屏外(负坐标)」渲染态，专供 PrintWindow 捕获。

        要点（参考原版 VDD 虚拟显示思路，但实现更干净）：
        - PrintWindow 捕获的是窗口自身绘制表面，窗口即使位于负坐标屏外也能
          抓到真实画面，因此「截图」永远不需要把窗口拉回物理桌面中央。
        - 唯一抓不到的情况是「最小化」（客户区停止绘制），这时才需要恢复。
          本方法把最小化窗口直接恢复到屏外负坐标，全程不进入物理桌面可见区，
          用户看不到任何移动/闪烁；恢复后 PrintWindow 即可读到真实画面。
        - 若窗口本就非最小化（含已屏外后台渲染中），直接返回 True，不做任何操作。
        """
        target = self._resolve(hwnd)
        if not target:
            return False
        if not self.is_minimized(target):
            return True

        # 最小化时 GetWindowRect 返回任务栏槽位尺寸(135x22)，不可用；
        # 用 _remember_native_size 取用户原始正常尺寸（只增不减，杜绝缩成小窗）。
        w, h = self._remember_native_size(target)
        if w <= 0 or h <= 0:
            w, h = 900, 680
        vs_x, vs_y, vs_w, vs_h = self.get_virtual_screen()
        # 屏外恢复时若窗口完全离开虚拟屏幕，DWM/CEF 可能跳过徽章等动态元素渲染。
        # 改为只让窗口右下角露出 1 像素在屏幕内，肉眼不可见却能强制完整渲染。
        off_x = vs_x + 1 - w
        off_y = vs_y + 1 - h

        done = False
        try:
            placement = _WINDOWPLACEMENT()
            placement.length = ctypes.sizeof(_WINDOWPLACEMENT)
            if self._user32.GetWindowPlacement(target, ctypes.byref(placement)):
                r = placement.rcNormalPosition
                r.left = off_x
                r.top = off_y
                r.right = off_x + w
                r.bottom = off_y + h
                # SW_SHOWNORMAL=1：恢复到刚设定的屏外正常位置，绝不闪现桌面中央
                placement.showCmd = 1
                if self._user32.SetWindowPlacement(target, ctypes.byref(placement)):
                    done = True
        except Exception:
            done = False

        if not done:
            # 兜底：SetWindowPos 直接到屏外 + ShowWindow(SW_SHOW)
            try:
                self._user32.SetWindowPos(
                    target, 0, off_x, off_y, w, h,
                    SWP_NOACTIVATE | SWP_SHOWWINDOW)
                self._user32.ShowWindow(target, SW_SHOW)
                done = True
            except Exception:
                done = False

        if done:
            self._parked = False
        return done

    def is_parked(self) -> bool:
        return self._parked

    def park_reset(self) -> None:
        """清理 park 状态记录（不移动窗口）。"""
        self._saved_rect = None
        self._parked = False

    # ---------------------------------------------------------------
    # 运行时检查与准备
    # ---------------------------------------------------------------
    def runtime_check(self, hwnd: Optional[int] = None) -> bool:
        """Runtime check: ensure window is capturable and ready for automation.

        Returns True if the window is ready for automation operations.
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        if not self._user32.IsWindow(target):
            return False

        if self.is_minimized(target):
            if not self.restore(target):
                return False
            time.sleep(0.2)

        if not self._user32.IsWindowVisible(target):
            self._user32.ShowWindow(target, SW_SHOWNOACTIVATE)
            time.sleep(0.1)

        return True

    def prepare_window(self, hwnd: Optional[int] = None) -> bool:
        """Prepare window for automation.

        屏幕外模式：若微信被最小化，不还原到「桌面可见」（会闪现/恢复桌面），
        而是转为「屏外可见」态（move_window_offscreen），窗口留在屏外、PrintWindow
        仍可读到画面。仅当窗口已正常显示在桌面时，才保持原样不动。
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        if not self._user32.IsWindow(target):
            return False

        if self.is_minimized(target):
            self.move_window_offscreen(target)
            return True

        if not self._user32.IsWindowVisible(target):
            self._user32.ShowWindow(target, SW_SHOWNOACTIVATE)
            time.sleep(0.1)

        return True

    def prepare(self, hwnd: int) -> bool:
        """prepare_window 的别名（对齐原版 WeChatWindowManager.prepare）。"""
        return self.prepare_window(hwnd)

    def find_window(self, hwnd: int) -> bool:
        """查找并确认窗口存在（对齐原版 WeChatWindowManager.find_window）。"""
        if not hwnd:
            return False
        return bool(self._user32.IsWindow(hwnd))

    # ---------------------------------------------------------------
    # 主屏移动
    # ---------------------------------------------------------------
    def move_to_primary(self, hwnd: Optional[int] = None) -> bool:
        """Move window to the primary display."""
        target = self._resolve(hwnd)
        if not target:
            return False

        rx, ry, rw, rh = self._primary_target_rect()
        if rw <= 0 or rh <= 0:
            return False

        rect = self.get_rect(target)
        if rect is None:
            return False

        w, h = self._remember_native_size(target)
        if w <= 0 or h <= 0:
            w, h = 900, 680

        self._user32.ShowWindow(target, SW_RESTORE)
        time.sleep(0.1)

        return bool(self._user32.MoveWindow(target, rx, ry, w, h, True))

    def _primary_target_rect(self) -> tuple:
        """Get the primary display work area as (x, y, w, h)."""
        try:
            vs_x = self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vs_y = self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vs_w = self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vs_h = self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
            return (vs_x, vs_y, vs_w, vs_h)
        except Exception:
            return (0, 0, 1920, 1080)

    def _hit_test_is_self(self, target: int, screen_x: int, screen_y: int) -> bool:
        """屏幕坐标点当前最上层窗口是否属于 target（自身 / 子窗口根 / 属主）。"""
        try:
            import win32gui
            hit = win32gui.WindowFromPoint((int(screen_x), int(screen_y)))
            if not hit:
                return False
            if hit == target:
                return True
            GA_ROOT = 2
            root = win32gui.GetAncestor(hit, GA_ROOT)
            if root == target:
                return True
            try:
                owner = win32gui.GetWindow(hit, 4)  # GW_OWNER
            except Exception:
                owner = 0
            return owner == target
        except Exception:
            return True  # 无法判定时不阻塞，按原方案点击

    def foreground_click(self, window_x: int, window_y: int,
                         hwnd: Optional[int] = None,
                         double: bool = False,
                         repark: bool = False) -> bool:
        """用【真实鼠标】点击窗口坐标（CEF 命中必须真实输入）。

        为什么是真实鼠标（根因）：
            微信是 CEF/Chromium 自绘窗口，SendMessageW/PostMessage 这类合成鼠标消息
            不经过系统命中测试，Chromium 不会把点击派发到具体列表项 —— 合成点击永远
            打不开会话（红点不消失、截图一直是列表）。必须用 SetCursorPos + mouse_event
            真实输入才能驱动 CEF 命中。

        坐标说明：
            window_x/window_y 来自 PrintWindow 整窗截图（物理像素，含标题栏），与窗口
            左上角同源。按下述策略点击：
              - 窗口已在主屏完整可见：原地点击 (rect.left+window_x, rect.top+window_y)，
                不移动窗口、不闪烁；
              - 窗口屏外/最小化/超出主屏：临时拉回主屏 (0,0) 真实点击，再还原到原状态
                （屏外则推回屏外，否则还原原始位置）。把窗口挪到 (0,0) 还能让原本落在
                屏幕外的目标项重新可见、可点击。

        Returns: 是否成功派发了真实点击。
        """
        target = self._resolve(hwnd)
        if not target:
            return False
        try:
            import win32api
            import win32con
        except Exception:
            return False

        saved_pos = None
        try:
            saved_pos = win32api.GetCursorPos()
        except Exception:
            saved_pos = None

        pw = self._user32.GetSystemMetrics(0)   # SM_CXSCREEN
        ph = self._user32.GetSystemMetrics(1)   # SM_CYSCREEN

        rect = self.get_rect(target)
        if rect is None:
            return False
        w, h = self._remember_native_size(target)
        if not w or not h:
            w, h = rect.width, rect.height

        was_offscreen = self.is_window_offscreen(target) or self.is_minimized(target)

        # 目标点是否已在主屏可见：是则原地点击（不移动、不闪烁）；
        # 否则（屏外/最小化/目标点超出主屏）临时拉回主屏 (0,0) 真实点击，再还原。
        # 用「目标点」而非「整窗」判可见，避免大窗略微超出屏幕时每次点击都抖动。
        target_sx = rect.x + int(window_x)
        target_sy = rect.y + int(window_y)
        point_on_screen = (0 <= target_sx <= pw) and (0 <= target_sy <= ph)
        in_place = (
            rect.x >= 0 and rect.y >= 0
            and rect.width > 0 and rect.height > 0
            and point_on_screen
        )

        moved = False
        topmost_applied = False
        reused_sticky = False
        if in_place:
            # 原地点击前短暂置顶微信：真实鼠标只会命中「该屏幕点最上层的窗口」，
            # 若微信被其他窗口（本助手 GUI、浏览器等）遮挡，点击会打在别人身上。
            try:
                self.topmost(target, enable=True)
                topmost_applied = True
            except Exception:
                topmost_applied = False
            # 命中测试：置顶后该点必须仍属于微信，否则降级为重定位路径
            if not self._hit_test_is_self(target, target_sx, target_sy):
                if topmost_applied:
                    try:
                        self.topmost(target, enable=False)
                    except Exception:
                        pass
                    topmost_applied = False
                in_place = False
                self._log_offscreen(
                    f"点({target_sx},{target_sy})未命中微信 -> 改走重定位路径")

        if in_place:
            base_x, base_y = rect.x, rect.y
            self._log_offscreen(
                f"real_click in_place at ({target_sx},{target_sy}) "
                f"rect=({rect.x},{rect.y}) size={rect.width}x{rect.height}")
        else:
            # —— 抖动修复：sticky relocate 复用 ——
            # 一轮流程（回列表单击→双击置顶→点未读行单击）有多次真实点击，
            # 旧逻辑每次点击都「拉回(0,0)→点→还原」，窗口往返搬移 = 用户看到的
            # 严重抖动。改为：第一次拉回后 8s 内的后续点击直接复用 (0,0) 位置
            # 不再搬移；过期后由下一次点击先把窗口还原到原位再走正常流程。
            # 抖动从「每次点击一次往返」降到「每轮最多一次往返」。
            now = time.time()
            sticky_until = getattr(self, "_reloc_sticky_until", 0.0)
            if now < sticky_until:
                r2 = self.get_rect(target)
                if r2 and r2.width > 0 and abs(r2.x) <= 10 and abs(r2.y) <= 10:
                    base_x, base_y = r2.x, r2.y
                    reused_sticky = True
                    try:
                        self.topmost(target, enable=True)
                        topmost_applied = True
                    except Exception:
                        topmost_applied = False
                    self._log_offscreen(
                        f"real_click reuse sticky relocate at ({base_x},{base_y}) "
                        f"remain={sticky_until - now:.1f}s")
            if not reused_sticky:
                # 过期 sticky：先把窗口还原到上次搬移前的位置，再走正常流程
                if getattr(self, "_reloc_active", False):
                    origin = getattr(self, "_reloc_origin", None)
                    if origin:
                        try:
                            self._user32.MoveWindow(
                                target, origin[0], origin[1],
                                origin[2], origin[3], True)
                        except Exception:
                            pass
                    self._reloc_active = False
                    rect = self.get_rect(target) or rect
                    time.sleep(0.05)
                self._remember_restore_position(target)
                base_x, base_y = 0, 0
                try:
                    self._user32.ShowWindow(target, SW_RESTORE)
                    time.sleep(0.08)
                    self.topmost(target, enable=True)     # 暂时置顶，确保点击命中微信
                    self._user32.MoveWindow(target, base_x, base_y, w, h, True)
                    time.sleep(0.15)
                    moved = True
                except Exception:
                    moved = False
                    base_x, base_y = rect.x, rect.y
            self._log_offscreen(
                f"real_click relocate at "
                f"({int(base_x) + int(window_x)},{int(base_y) + int(window_y)})")

        screen_x = int(base_x) + int(window_x)
        screen_y = int(base_y) + int(window_y)
        try:
            # —— 双击必须先激活窗口（根因）——
            # 微信屏外挂机时不是前台窗口。点击「非激活窗口」时，第一次点击会被系统
            # 用于激活窗口（应用收不到，或只收到一个孤立单击），两次点击无法被配对
            # 成双击，于是双击退化成单击 —— 表现就是「聊天图标只是被选中，未读却
            # 没有置顶」。点击前先激活，保证两次点击都完整送达并被系统合成为双击。
            if double and was_offscreen:
                # 仅屏外/最小化窗口需要激活：屏外挂机时微信不是前台窗口，第一次点击
                # 会被系统用于激活，导致双击退化成单击。窗口本身在屏幕可见时无需抢前台，
                # 否则会把焦点从用户当前窗口抢走（影响体验）。
                try:
                    self._activate_offscreen_window(target)
                except Exception:
                    pass
            # 两次点击的间隔必须落在系统双击时限内（默认 500ms），否则仍是两次单击。
            # 注意：mouse_event 没有 DBLCLK 标志，双击只能靠间隔 + 光标不移动来合成。
            try:
                dct_ms = int(self._user32.GetDoubleClickTime())
            except Exception:
                dct_ms = 500
            if dct_ms <= 0:
                dct_ms = 500
            # 取双击时限的 1/4 且封顶 100ms，远离阈值边界
            gap = max(0.02, min(0.10, dct_ms / 4000.0))
            if double:
                self._log_offscreen(
                    f"double_click at ({screen_x},{screen_y}) gap={gap:.3f}s "
                    f"dblclick_time={dct_ms}ms")
            win32api.SetCursorPos((screen_x, screen_y))
            time.sleep(0.03)
            clicks = 2 if double else 1
            for i in range(clicks):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.03)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                if i < clicks - 1:
                    # 期间绝不移动光标：超出双击位移阈值同样会被判成两次单击
                    time.sleep(gap)
            time.sleep(0.15)
        except Exception:
            pass
        finally:
            # 还原光标位置，避免打扰用户
            if saved_pos:
                try:
                    win32api.SetCursorPos(saved_pos)
                except Exception:
                    pass
            if topmost_applied and not moved:
                try:
                    self.topmost(target, enable=False)
                except Exception:
                    pass
            if moved:
                # 抖动修复：不立即还原。登记 sticky（8s），窗口保持在 (0,0)
                # 供本轮后续点击复用；下一次点击发现过期时统一还原到原位。
                # 旧逻辑每次点击后立刻推回屏外/还原位置 —— 一轮 3 次点击 =
                # 3 次窗口搬移 = 严重抖动。
                self._reloc_sticky_until = time.time() + 8.0
                self._reloc_active = True
                self._reloc_origin = (rect.x, rect.y, rect.width, rect.height)
                try:
                    self.topmost(target, enable=False)
                except Exception:
                    pass
        return True

    def is_foreground(self, hwnd: Optional[int] = None) -> bool:
        """微信窗口是否在前台（用户正在使用）。"""
        target = self._resolve(hwnd)
        if not target:
            return False
        try:
            return self._user32.GetForegroundWindow() == target
        except Exception:
            return False

    def move_offscreen(self, hwnd: Optional[int] = None) -> bool:
        """Move window completely off-screen without minimizing.

        Keeps window rendering but hidden from user view.
        """
        target = self._resolve(hwnd)
        if not target:
            return False

        self._remember_restore_position(target)

        # 最小化时转为屏外可见态（不 SW_RESTORE 到桌面闪现）
        if self.is_minimized(target):
            return self.move_window_offscreen(target)

        rect = self.get_rect(target)
        if rect is None:
            return False

        w, h = self._remember_native_size(target)

        vs_x, vs_y, vs_w, vs_h = self.get_virtual_screen()
        off_x = vs_x - w - 200
        off_y = vs_y

        return bool(self._user32.SetWindowPos(
            target, 0, off_x, off_y, w, h,
            SWP_NOACTIVATE | SWP_SHOWWINDOW))

    # ================================================================
    # 屏幕外保活模式（移植自 wechat-ai-reply-main 的 window_manager.py）
    # 设计要点：
    #   - 微信最小化时，不还原到「桌面可见」，而是 SetWindowPos 到屏外坐标 +
    #     SW_SHOWNOACTIVATE，使窗口变为「屏外可见」态（不激活、不抢前台）。
    #   - 屏外可见窗口仍被系统绘制，PrintWindow 能读到真实画面（最小化读不到）。
    #   - 点击用 SendMessageW 把鼠标消息注入客户区坐标（光标到不了屏外，
    #     物理点击不可行，必须消息注入），窗口全程留在屏外。
    #   - 仅当用户主动 Alt+Tab / 点任务栏激活窗口时，watcher 才把它移回桌面。
    #   这样彻底消除「最小化挂机时窗口闪现/恢复桌面」的问题。
    # ================================================================
    def _ensure_ctypes_signatures(self) -> None:
        """惰性设置 ctypes 函数签名（仅首次调用时设置一次）。"""
        if self._ctypes_ready:
            return
        u32 = self._user32
        try:
            u32.SendMessageW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            u32.SendMessageW.restype = ctypes.c_long
            u32.SetForegroundWindow.argtypes = [wintypes.HWND]
            u32.SetForegroundWindow.restype = wintypes.BOOL
            u32.GetForegroundWindow.argtypes = []
            u32.GetForegroundWindow.restype = wintypes.HWND
            u32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            u32.GetWindowLongW.restype = ctypes.c_long
            u32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
            u32.AllowSetForegroundWindow.restype = wintypes.BOOL
            u32.SystemParametersInfoW.argtypes = [
                wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
            u32.SystemParametersInfoW.restype = wintypes.BOOL
            u32.RedrawWindow.argtypes = [
                wintypes.HWND, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
            u32.RedrawWindow.restype = wintypes.BOOL
        except Exception:
            pass
        self._ctypes_ready = True

    def _get_style(self, target: int) -> int:
        try:
            return int(self._user32.GetWindowLongW(target, GWL_STYLE))
        except Exception:
            return 0

    def _remember_window_rect(self, target: int) -> None:
        """记录窗口原始可见位置（仅正常可见且在屏幕上时记录）。"""
        if self.is_minimized(target):
            return  # 最小化时坐标是 -32000，无效
        rect = self.get_rect(target)
        if rect is None:
            return
        if rect.x < -1000:
            return  # 已在屏外，不记录
        self._offscreen_original[target] = (rect.x, rect.y, rect.width, rect.height)

    def is_window_offscreen(self, hwnd: Optional[int] = None) -> bool:
        """窗口是否处于屏幕外模式（左侧坐标 < -1000）。"""
        target = self._resolve(hwnd)
        if not target:
            return False
        rect = self.get_rect(target)
        if rect is None:
            return False
        return rect.x < -1000

    def move_window_offscreen(self, hwnd: Optional[int] = None) -> bool:
        """把最小化窗口转为「屏外可见」态（不闪桌面）。

        关键顺序（移植自参考项目）：
          1) 窗口仍最小化时先 SetWindowPos 到屏外 —— 同时设置其「还原位置」到屏外；
          2) 再 SW_SHOWNOACTIVATE 取消最小化，窗口在屏外位置变为可见
             （不激活、不抢前台，用户看不到）；
          3) 校验 Qt 是否把窗口拖回屏内，是则强制重设 + SWP_FRAMECHANGED 重绘。
        """
        target = self._resolve(hwnd)
        if not target:
            return False
        self._ensure_ctypes_signatures()

        self._remember_window_rect(target)
        orig = self._offscreen_original.get(target)
        if orig:
            w, h = orig[2], orig[3]
        else:
            w, h = self._remember_native_size(target)
        if w <= 200 or h <= 200:
            w, h = 900, 680

        is_min = bool(self._get_style(target) & WS_MINIMIZE)

        # 1) 仍最小化时先设屏外位置（同时设置还原位置）
        self._user32.SetWindowPos(
            target, 0, self.OFFSCREEN_X, self.OFFSCREEN_Y, w, h,
            SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER)
        time.sleep(0.1)
        # 2) 取消最小化，窗口在屏外变为可见（不激活）
        if is_min:
            self._user32.ShowWindow(target, SW_SHOWNOACTIVATE)
            time.sleep(0.2)
        # 3) 校验 Qt 对抗
        rect = self.get_rect(target)
        if rect is None or rect.x > -1000:
            self._user32.SetWindowPos(
                target, 0, self.OFFSCREEN_X, self.OFFSCREEN_Y, w, h,
                SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_FRAMECHANGED | SWP_NOZORDER)
            time.sleep(0.15)
        # 强制 Qt 重绘
        try:
            self._user32.RedrawWindow(
                target, None, None, 0x0001 | 0x0100 | 0x0080)  # INVALIDATE|UPDATENOW|ALLCHILDREN
            time.sleep(0.1)
        except Exception:
            pass

        self._parked = True
        self.start_offscreen_watcher(target)
        return True

    def _ensure_offscreen_visible(self, hwnd: int,
                                  window: Optional[object] = None) -> bool:
        """确保窗口在屏外且可见（取消最小化到屏外，不弹回桌面）。

        关键技巧：最小化窗口先用 SetWindowPlacement 预置「还原位置」到屏外，
        再 SW_RESTORE → 窗口恢复到屏外（参考项目验证 SW_SHOWNOACTIVATE 对最小化窗口
        无效，无法取消最小化，会导致「弹回桌面」）。
        """
        self._ensure_ctypes_signatures()
        rect = self.get_rect(hwnd)
        if rect is None:
            return False
        w = max(rect.width, 100)
        h = max(rect.height, 100)

        if self.is_minimized(hwnd):
            try:
                placement = _WINDOWPLACEMENT()
                placement.length = ctypes.sizeof(_WINDOWPLACEMENT)
                if self._user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
                    r = placement.rcNormalPosition
                    r.left = self.OFFSCREEN_X
                    r.top = self.OFFSCREEN_Y
                    r.right = self.OFFSCREEN_X + w
                    r.bottom = self.OFFSCREEN_Y + h
                    placement.showCmd = 1  # SW_SHOWNORMAL
                    self._user32.SetWindowPlacement(hwnd, ctypes.byref(placement))
                    time.sleep(0.05)
            except Exception:
                pass
            self._user32.ShowWindow(hwnd, SW_RESTORE)
            time.sleep(0.15)

        # 若被 Qt 拖回屏内 → 重新放回屏外
        rect2 = self.get_rect(hwnd)
        if rect2 is not None and rect2.x > -1000:
            self._user32.SetWindowPos(
                hwnd, 0, self.OFFSCREEN_X, self.OFFSCREEN_Y, w, h,
                SWP_NOACTIVATE | SWP_NOZORDER)
            time.sleep(0.1)
        try:
            if window is not None:
                window.left = self.OFFSCREEN_X
                window.top = self.OFFSCREEN_Y
                window.width = w
                window.height = h
        except Exception:
            pass
        return True

    def _activate_offscreen_window(self, hwnd: int) -> bool:
        """在屏外激活窗口（绕过 Windows 前台锁定）。

        Qt/CEF 只处理「前台激活窗口」的物理/合成输入，所以点击/滚动前必须先激活。
        SetForegroundWindow 不要求窗口在可见区域：窗口保持在屏外，用户看不到，
        但 Qt 认为它已激活 → 注入的消息会被处理。
        """
        self._ensure_ctypes_signatures()
        u32 = self._user32
        try:
            u32.AllowSetForegroundWindow(-1)  # ASF_ANY
        except Exception:
            pass
        try:
            u32.SystemParametersInfoW(0x2001, 0, 0, 0)  # SPI_SETFOREGROUNDLOCKTIMEOUT=0
        except Exception:
            pass
        # ALT 键兜底：让当前线程获得设置前台窗口的权限
        try:
            import win32api
            import win32con
            win32api.keybd_event(0x12, 0, 0, 0)        # VK_MENU down
            win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.02)
        except Exception:
            pass
        try:
            u32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            time.sleep(0.05)
            u32.SetForegroundWindow(hwnd)
            time.sleep(0.08)
        except Exception:
            pass
        return u32.GetForegroundWindow() == hwnd

    def _rehide_if_qt_moved(self, hwnd: int) -> bool:
        """Qt 可能在激活后把窗口弹回屏内 → 立即重新藏回屏外。"""
        rect = self.get_rect(hwnd)
        if rect is not None and rect.x > -1000:
            w = rect.width
            h = rect.height
            self._user32.SetWindowPos(
                hwnd, 0, self.OFFSCREEN_X, self.OFFSCREEN_Y, w, h,
                SWP_NOACTIVATE | SWP_NOZORDER)
            time.sleep(0.1)
            return True
        return False

    def click_offscreen(self, hwnd: int, client_x: int, client_y: int,
                        double: bool = False,
                        verify_func=None) -> bool:
        """屏外点击：SendMessageW 注入鼠标消息（窗口全程屏外不可见）。

        光标无法到达屏外坐标（Windows 钳制在虚拟屏内），所以物理点击不可行；
        改为：窗口保持屏外 → 真实激活(Qt 需要前台才处理输入) →
        SendMessageW 把 WM_LBUTTON 序列同步注入到客户区坐标。
        """
        self._ensure_ctypes_signatures()
        u32 = self._user32
        try:
            self._ensure_offscreen_visible(hwnd)
            self._acquire_interaction()
            time.sleep(0.05)

            prev_fg = None
            try:
                prev_fg = u32.GetForegroundWindow()
            except Exception:
                pass

            self._activate_offscreen_window(hwnd)
            time.sleep(0.05)

            bl, bt = self._border_offset(hwnd)
            cx = max(0, int(client_x) - int(bl))
            cy = max(0, int(client_y) - int(bt))
            self._send_mouse_messages(hwnd, cx, cy, double)

            self._rehide_if_qt_moved(hwnd)

            if prev_fg and u32.IsWindow(prev_fg):
                try:
                    u32.SetForegroundWindow(prev_fg)
                except Exception:
                    pass

            if verify_func:
                try:
                    if not verify_func():
                        return False
                except Exception:
                    pass

            self._release_interaction()
            return True
        except Exception:
            self._release_interaction()
            return False

    # =================================================================
    # SendMessageW 注入点击（屏内/屏外通用，不依赖真实鼠标）
    # =================================================================

    def _border_offset(self, hwnd: Optional[int]):
        """整窗截图坐标 → 客户区坐标的边框偏移（标题栏 + 边框）。

        PrintWindow(PW_RENDERFULLCONTENT) 截图含标题栏，原点是窗口左上角；
        而 SendMessageW 点击用客户区坐标，原点在标题栏下方。注入前需减去该
        偏移才能精准命中目标项。差值与原窗口是否被 park 到屏外无关（恒定）。
        """
        target = self._resolve(hwnd)
        if not target:
            return (0, 0)
        try:
            import win32gui
            wr = win32gui.GetWindowRect(target)
            origin = win32gui.ClientToScreen(target, (0, 0))
            return (int(origin[0]) - int(wr[0]), int(origin[1]) - int(wr[1]))
        except Exception:
            return (0, 0)

    def _send_mouse_messages(self, hwnd: int, client_x: int, client_y: int,
                             double: bool = False) -> None:
        """向窗口客户区坐标同步注入 WM_LBUTTON 序列（SendMessageW）。"""
        self._ensure_ctypes_signatures()
        u32 = self._user32
        cx = int(client_x) & 0xFFFF
        cy = int(client_y) & 0xFFFF
        lparam = (cy << 16) | cx
        MK_LBUTTON = 0x0001
        u32.SendMessageW(hwnd, 0x0006, 1, 0)      # WM_ACTIVATE
        u32.SendMessageW(hwnd, 0x0007, 0, 0)      # WM_SETFOCUS
        u32.SendMessageW(hwnd, 0x0200, 0, lparam)  # WM_MOUSEMOVE
        clicks = 2 if double else 1
        for _ in range(clicks):
            u32.SendMessageW(hwnd, 0x0201, MK_LBUTTON, lparam)  # WM_LBUTTONDOWN
            time.sleep(0.05)
            u32.SendMessageW(hwnd, 0x0202, 0, lparam)          # WM_LBUTTONUP
            if _ < clicks - 1:
                time.sleep(0.12)
        time.sleep(0.6)

    def click_inject(self, hwnd: Optional[int], x: int, y: int,
                     double: bool = False, verify_func=None) -> bool:
        """SendMessageW 注入点击（【不移动窗口】，屏内/屏外通用）。

        统一点击方案：无论 GUI 屏内还是挂机屏外，都优先用 SendMessageW 注入客户区
        坐标，规避真实鼠标带来的 DPI 缩放与窗口跳动问题。坐标 x/y 来自 PrintWindow
        整窗截图（物理像素，含标题栏），本方法内部减去边框偏移换算到客户区。

        与 click_offscreen 的区别：本方法【不】把窗口 park 到屏外/解除最小化，
        适合 GUI 屏内窗口 —— 保持原位置，不闪动、不隐藏到屏外。

        实测：屏外注入可稳定打开微信 CEF 会话（_verify_offscreen_mode 验证通过）；
        屏内同样有效，且无需把光标移到目标位置。
        """
        target = self._resolve(hwnd)
        if not target:
            return False
        self._ensure_ctypes_signatures()
        u32 = self._user32
        try:
            self._acquire_interaction()
            time.sleep(0.05)
            prev_fg = None
            try:
                prev_fg = u32.GetForegroundWindow()
            except Exception:
                pass
            self._activate_offscreen_window(target)
            time.sleep(0.05)

            bl, bt = self._border_offset(target)
            cx = max(0, int(x) - int(bl))
            cy = max(0, int(y) - int(bt))
            self._send_mouse_messages(target, cx, cy, double)

            ok = True
            if verify_func:
                try:
                    ok = bool(verify_func())
                except Exception:
                    ok = True
            if prev_fg and u32.IsWindow(prev_fg):
                try:
                    u32.SetForegroundWindow(prev_fg)
                except Exception:
                    pass
            self._release_interaction()
            return ok
        except Exception:
            self._release_interaction()
            return False

    def bring_window_back(self, hwnd: Optional[int] = None,
                          force_center: bool = False) -> bool:
        """从屏外恢复到原始可见位置（用户主动激活时调用）。"""
        target = self._resolve(hwnd)
        if not target:
            return False
        self.stop_offscreen_watcher()

        orig = self._offscreen_original.get(target)
        if force_center or not orig:
            pw = self._user32.GetSystemMetrics(0)
            ph = self._user32.GetSystemMetrics(1)
            w, h = self._remember_native_size(target)
            tx = max(0, (pw - w) // 4)
            ty = max(0, (ph - h) // 4)
            if not orig:
                self._log_offscreen("无原始位置记录，强制居中到(%d,%d)" % (tx, ty))
        else:
            tx, ty, w, h = orig[0], orig[1], orig[2], orig[3]

        placement = _WINDOWPLACEMENT()
        placement.length = ctypes.sizeof(_WINDOWPLACEMENT)
        placement.showCmd = 1  # SW_SHOWNORMAL
        r = placement.rcNormalPosition
        r.left = tx
        r.top = ty
        r.right = tx + w
        r.bottom = ty + h
        try:
            self._user32.SetWindowPlacement(target, ctypes.byref(placement))
            time.sleep(0.1)
            self._user32.SetForegroundWindow(target)
        except Exception:
            pass
        self._parked = False
        return True

    def _log_offscreen(self, msg: str) -> None:
        try:
            import logging
            logging.getLogger(__name__).info("[屏幕外] " + msg)
        except Exception:
            pass

    # ---- 交互锁 + watcher（移植自参考项目，防止点击/截图期间被误恢复） ----
    _watcher_thread = None
    _watcher_stop = None
    _interaction_lock = None  # 延迟为 threading.Event

    def _get_interaction_lock(self):
        if self._interaction_lock is None:
            import threading
            self._interaction_lock = threading.Event()
        return self._interaction_lock

    def _acquire_interaction(self) -> None:
        self.stop_offscreen_watcher()
        self._get_interaction_lock().set()

    def _release_interaction(self, window=None) -> None:
        self._get_interaction_lock().clear()
        if window is not None:
            self.start_offscreen_watcher(window if hasattr(window, "_hWnd") else self._resolve_for_window(window))

    def _resolve_for_window(self, window):
        hwnd = getattr(window, "_hWnd", None)
        if hwnd:
            return hwnd
        return self._hwnd_wechat

    def stop_offscreen_watcher(self) -> None:
        if self._watcher_stop is not None:
            try:
                self._watcher_stop.set()
            except Exception:
                pass

    def start_offscreen_watcher(self, window) -> None:
        """启动独立 watcher 线程：用户激活屏外窗口时立即移回桌面。"""
        import threading
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return
        self._watcher_stop = threading.Event()
        hwnd = self._resolve_for_window(window)
        if not hwnd:
            return

        def _watch():
            self._log_offscreen("屏幕外恢复监视已启动 hwnd=%s" % hwnd)
            while not self._watcher_stop.is_set():
                try:
                    if self._get_interaction_lock().is_set():
                        self._watcher_stop.wait(0.3)
                        continue
                    if not self._user32.IsWindow(hwnd):
                        break
                    fg = self._user32.GetForegroundWindow()
                    if fg == hwnd:
                        rect = self.get_rect(hwnd)
                        if rect is not None and rect.x < -1000:
                            self._log_offscreen("检测到用户激活微信 → 恢复窗口")
                            self.bring_window_back(hwnd)
                            return
                except Exception:
                    pass
                self._watcher_stop.wait(0.3)
            self._log_offscreen("屏幕外监视线程退出")

        self._watcher_thread = threading.Thread(
            target=_watch, daemon=True, name="offscreen-watcher")
        self._watcher_thread.start()

    def keep_alive_offscreen(self, window) -> None:
        """每轮调用：维护屏外态。

        - 用户最小化 → 移到屏外后台；
        - 屏外 + 用户激活 → 移回可见；
        - 屏外 + 未激活 → 保持后台；
        - 正常可见 → 记录原始位置。
        """
        target = self._resolve_for_window(window)
        if not target:
            return
        if self._get_interaction_lock().is_set():
            return
        try:
            style = self._get_style(target)
            is_min = bool(style & WS_MINIMIZE)
            is_vis = bool(style & 0x10000000)  # WS_VISIBLE
        except Exception:
            return
        rect = self.get_rect(target)
        offscreen = rect is not None and (rect.x < -1000 or (rect.x + rect.width) < 0)

        user_activated = False
        try:
            if self._user32.GetForegroundWindow() == target:
                user_activated = True
        except Exception:
            pass

        if is_min:
            self.move_window_offscreen(target)
            return
        if not is_vis:
            self.move_window_offscreen(target)
            return
        if offscreen and user_activated:
            self.bring_window_back(target)
            return
        if offscreen:
            return
        if not offscreen and not is_min:
            self._remember_window_rect(target)


__all__ = [
    "WeChatWindowManager",
    "SW_HIDE", "SW_SHOWNORMAL", "SW_SHOWMINIMIZED", "SW_SHOWMAXIMIZED",
    "SW_SHOWNOACTIVATE", "SW_SHOW", "SW_MINIMIZE", "SW_RESTORE",
]