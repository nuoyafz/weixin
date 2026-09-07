"""Screen capture —— 对齐原版 app.vision.screen_capture。

原版支持：
  - ImageGrab 全屏截图（默认）/ PrintWindow 备选通道
  - 截图可用性检查（_looks_usable，使用直方图分析）
  - 截图重试 + 窗口重绘恢复（ShowWindow/SetWindowPos/RedrawWindow/DwmFlush）
  - 区域截图 + 窗口截图裁剪
  - 截图保存（带时间戳）
  - capture_backend 参数控制
  - background_mode_enabled 标志
  - _prefer_printwindow 属性
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, NamedTuple

import numpy as np
from PIL import Image, ImageGrab


# =====================================================================
# DPI 感知（参考 wechat-ai-reply-main screenshot.py:26）
# =====================================================================
# Windows 在高 DPI 缩放下，未开启 DPI 感知的进程拿到的 GetWindowRect 是逻辑坐标，
# 而 PrintWindow 实际渲染的是物理像素。两者不匹配会导致位图只装下窗口左上角，
# 从而出现"截图只有一半 / 右下被截断"。必须在进程最早期（任何 DPI 相关调用前）
# 声明 Per-Monitor DPI Aware。
def _ensure_dpi_aware() -> None:
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        PM_V2 = ctypes.c_int(-4)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(PM_V2)
    except Exception:
        pass


_ensure_dpi_aware()


# =====================================================================
# 数据结构
# =====================================================================

class CaptureResult(NamedTuple):
    success: bool
    image: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    error: str = ""


class WindowInfo(NamedTuple):
    """窗口信息 —— 对齐原版 find_wechat_window 返回值。"""
    hwnd: int = 0
    title: str = ""
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0

    def to_dict(self) -> dict:
        """转为 dict 格式，兼容 observe_service 中的 handle/rect/found 字段。"""
        return {
            "found": True,
            "handle": self.hwnd,
            "hwnd": self.hwnd,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "x": self.x,
            "y": self.y,
            "rect": {
                "left": self.x,
                "top": self.y,
                "right": self.x + self.width,
                "bottom": self.y + self.height,
            },
        }


# =====================================================================
# ScreenCapture —— 对齐原版
# =====================================================================

class ScreenCapture:
    """双通道截图（ImageGrab + PrintWindow）+ 重试 + 完整性校验。

    对齐原版 app.vision.screen_capture.ScreenCapture。
    """

    SCREENSHOT_MAX_RETRIES = 3
    SCREENSHOT_RETRY_DELAY = 0.08
    SCREENSHOT_MIN_NONBLACK_NAV = 30
    DATA_DIR = ""

    def __init__(self, store=None, store_address=None, window_manager=None,
                 config=None, logger=None, backend: str = "auto"):
        self.store = store
        self.store_address = store_address
        self._window_manager = window_manager
        self.config = config or {}
        self.logger = logger

        self._user32 = ctypes.windll.user32
        self._hwnd_wechat = None

        if self.config:
            self.SCREENSHOT_MIN_NONBLACK_NAV = int(
                self.config.get("screenshot_min_nonblack_nav", 30))

        # 原版 capture_backend 参数：auto / imagegrab / printwindow
        self._backend = backend
        if backend == "printwindow":
            self._prefer_printwindow = True
        elif backend == "imagegrab":
            self._prefer_printwindow = False
        else:
            self._prefer_printwindow: bool = self.config.get("prefer_printwindow", False)

        self._printwindow_capture_unusable: bool = False
        self._imagegrab_capture_unusable: bool = False
        self._background_mode_enabled: bool = self.config.get("background_mode_enabled", False)

        self._data_dir = self.config.get("data_dir", self.DATA_DIR)
        self._last_screenshot: Optional[np.ndarray] = None
        # 截图回调钩子：每次成功截图后调用 on_captured(image)，
        # 供 observe_service 推送实时预览（覆盖红点扫描/点击重扫/分析截图/手动刷新）。
        self.on_captured: Optional[callable] = None
        self._last_screenshot_attempts: int = 0
        # 标记最近一次 PrintWindow 是否使用 PW_CLIENTONLY（影响坐标换算）
        self._last_pw_client_only: bool = False

    # =================================================================
    # _prefer_printwindow 属性（对齐原版）
    # =================================================================

    @property
    def prefer_printwindow(self) -> bool:
        return self._prefer_printwindow

    @prefer_printwindow.setter
    def prefer_printwindow(self, value: bool) -> None:
        self._prefer_printwindow = value

    @property
    def background_mode_enabled(self) -> bool:
        return self._background_mode_enabled

    @background_mode_enabled.setter
    def background_mode_enabled(self, value: bool) -> None:
        self._background_mode_enabled = value

    # =================================================================
    # 窗口查找
    # =================================================================

    def find_wechat_window(self, keywords=None):
        """查找微信主窗口，返回 WindowInfo 或 None。

        对齐原版：委托给 WindowFinder 执行严格四步验证。
        """
        from ..desktop.window_finder import WindowFinder as _WF
        wf = _WF()
        return wf.find(keywords)

    # =================================================================
    # 全屏截图（对齐原版 _capture_imagegrab_full）
    # =================================================================

    def _capture_imagegrab_full(self) -> Optional[np.ndarray]:
        try:
            img = ImageGrab.grab(all_screens=True)
            if img is None:
                return None
            return np.array(img.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            return None

    # =================================================================
    # 区域截图（对齐原版 _capture_imagegrab_rect）
    # =================================================================

    def _capture_imagegrab_rect(self, bbox: Dict[str, int]) -> Optional[np.ndarray]:
        try:
            left, top = bbox["left"], bbox["top"]
            right, bottom = bbox["right"], bbox["bottom"]
            img = ImageGrab.grab(bbox=(left, top, right, bottom))
            if img is None:
                return None
            return np.array(img.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            return None

    # =================================================================
    # PrintWindow 截图（对齐原版 _capture_printwindow）
    # =================================================================

    def _capture_printwindow(self, hwnd: int, rect=None) -> Optional[np.ndarray]:
        """使用 PrintWindow 截取窗口（参考 wechat-ai-reply-main 后台截图技术）。

        流程：
          1. 先 RedrawWindow 强制 CEF/Qt 提交渲染（解决 park 态 GPU 停绘）。
          2. 依次尝试 flags=2/3/0，拒绝黑图/白壳/半渲染帧（渲染健康校验）。
          3. 返回可用图像；记录本次是否使用 PW_CLIENTONLY。

        Args:
            hwnd: 窗口句柄
            rect: 截取区域 Dict[str, int] | None（None=全窗口）
        """
        try:
            import win32gui
            import win32ui
            import win32con
            from PIL import Image as PILImage

            # 检查窗口是否最小化 — 最小化时 PrintWindow 返回空白
            if self._user32.IsIconic(hwnd):
                self._log(f"printwindow_skip_minimized hwnd={hwnd}")
                return None

            # 优先使用传入的 rect；否则使用整窗尺寸
            if rect and isinstance(rect, dict):
                left = int(rect.get("left", 0))
                top = int(rect.get("top", 0))
                right = int(rect.get("right", 0))
                bottom = int(rect.get("bottom", 0))
                capture_w = right - left
                capture_h = bottom - top
            else:
                win_rect = win32gui.GetWindowRect(hwnd)
                capture_w = win_rect[2] - win_rect[0]
                capture_h = win_rect[3] - win_rect[1]

            if capture_w <= 0 or capture_h <= 0:
                self._log(f"printwindow_invalid_rect hwnd={hwnd} w={capture_w} h={capture_h}")
                return None

            # 防御：如果进程仍是 DPI-unaware，GetWindowRect 返回逻辑坐标，
            # 必须乘以缩放系数扩到物理像素，否则位图太小只能截到左上角。
            try:
                aware = ctypes.c_int()
                ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(aware))
                if aware.value == 0:  # PROCESS_DPI_UNAWARE
                    dpi = self._user32.GetDpiForWindow(hwnd) or 96
                    scale = dpi / 96.0
                    if scale > 1.01:
                        capture_w = int(round(capture_w * scale))
                        capture_h = int(round(capture_h * scale))
                        self._log(f"printwindow_dpi_scale hwnd={hwnd} scale={scale:.2f} physical={capture_w}x{capture_h}")
                else:
                    dpi = self._user32.GetDpiForWindow(hwnd) or 96
                    self._log(f"printwindow_dpi_aware hwnd={hwnd} dpi={dpi} size={capture_w}x{capture_h}")
            except Exception:
                pass

            # 强制重绘：park 态 GPU 可能停绘，必须先让窗口提交一帧
            RDW_INVALIDATE = 0x0001
            RDW_UPDATENOW = 0x0100
            RDW_ALLCHILDREN = 0x0080
            for _ in range(2):
                try:
                    self._user32.RedrawWindow(
                        hwnd, None, None,
                        RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN)
                    self._user32.UpdateWindow(hwnd)
                    time.sleep(0.2)
                except Exception:
                    pass

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            save_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            bitmap_dc = save_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(save_dc, capture_w, capture_h)
            bitmap_dc.SelectObject(bitmap)

            PW_RENDERFULLCONTENT = 0x00000002
            PW_CLIENTONLY = 0x00000001

            result = None
            self._last_pw_client_only = False

            try:
                for flags, method in [
                    (PW_RENDERFULLCONTENT, "FLAGS2"),
                    (PW_CLIENTONLY | PW_RENDERFULLCONTENT, "FLAGS3"),
                    (0, "FLAGS0"),
                ]:
                    try:
                        bitmap_dc.PatBlt((0, 0, capture_w, capture_h), win32con.BLACKNESS)
                    except Exception:
                        pass

                    pw_ok = self._user32.PrintWindow(hwnd, bitmap_dc.GetSafeHdc(), flags)
                    if not pw_ok:
                        self._log(f"printwindow_flag_failed hwnd={hwnd} flag={method}")
                        continue

                    bmpinfo = bitmap.GetInfo()
                    bmpstr = bitmap.GetBitmapBits(True)
                    img = PILImage.frombuffer(
                        "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                        bmpstr, "raw", "BGRX", 0, 1)
                    img_arr = np.array(img.convert("RGB"))[:, :, ::-1].copy()

                    mean_val = float(img_arr.mean())
                    std_val = float(img_arr.std())
                    healthy = self._is_render_healthy(img_arr)

                    self._log(
                        f"printwindow_attempt hwnd={hwnd} flag={method} "
                        f"size={img_arr.shape[1]}x{img_arr.shape[0]} "
                        f"mean={mean_val:.1f} std={std_val:.1f} healthy={healthy}")

                    # 拒绝：全黑 / 纯色 / 白壳（微信4.x GPU 未渲染典型特征）/ 半渲染
                    if (mean_val < 1 or std_val < 5 or
                        (mean_val > 230.0 and std_val < 20.0) or not healthy):
                        self._log(
                            f"printwindow_unhealthy hwnd={hwnd} flag={method} "
                            f"mean={mean_val:.1f} std={std_val:.1f}")
                        continue

                    result = img_arr
                    self._last_pw_client_only = bool(flags & PW_CLIENTONLY)
                    self._log(f"printwindow_success hwnd={hwnd} flag={method}")
                    break
            finally:
                try:
                    bitmap_dc.DeleteDC()
                except Exception:
                    pass
                try:
                    save_dc.DeleteDC()
                except Exception:
                    pass
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                try:
                    win32gui.DeleteObject(bitmap.GetHandle())
                except Exception:
                    pass

            return result
        except Exception as e:
            self._log(f"printwindow_capture_failed hwnd={hwnd} error={e}")
            return None

    def _capture_via_temporary_restore(self, hwnd: int) -> Optional[np.ndarray]:
        """PrintWindow 全失败时的兜底：短暂移到屏幕边缘，mss 抓屏，再移回屏外。

        参考 wechat-ai-reply-main：对 Qt/Chromium 微信，GPU 在屏外可能彻底停绘，
        唯一可靠方式是让它有一帧真正可见，截完再藏回去。
        """
        try:
            import win32gui
            import win32con
            import mss
            from PIL import Image as PILImage

            rect = win32gui.GetWindowRect(hwnd)
            orig_x, orig_y = rect[0], rect[1]
            orig_w = rect[2] - rect[0]
            orig_h = rect[3] - rect[1]
            if orig_w <= 0 or orig_h <= 0:
                return None

            screen_w = self._user32.GetSystemMetrics(0)
            screen_h = self._user32.GetSystemMetrics(1)
            # 屏幕右下角，大部分在屏外，只露少量像素，尽量减少干扰
            flash_x = max(0, screen_w - 10)
            flash_y = max(0, screen_h - orig_h - 5)

            was_min = win32gui.IsIconic(hwnd)
            img = None
            try:
                win32gui.SetWindowPos(
                    hwnd, win32con.HWND_TOPMOST,
                    flash_x, flash_y, orig_w, orig_h,
                    win32con.SWP_NOACTIVATE | win32con.SWP_NOZORDER)
                time.sleep(0.1)

                if was_min:
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
                    time.sleep(0.2)
                else:
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_TOPMOST,
                        flash_x, flash_y, orig_w, orig_h,
                        win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW)
                    time.sleep(0.2)

                RDW_INVALIDATE = 0x0001
                RDW_UPDATENOW = 0x0100
                RDW_ALLCHILDREN = 0x0080
                self._user32.RedrawWindow(
                    hwnd, None, None,
                    RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN)
                self._user32.UpdateWindow(hwnd)
                time.sleep(0.5)

                frame = win32gui.GetWindowRect(hwnd)
                with mss.mss() as sct:
                    raw = sct.grab({
                        "left": frame[0], "top": frame[1],
                        "width": frame[2] - frame[0],
                        "height": frame[3] - frame[1]})
                    arr = np.array(raw, dtype=np.uint8)
                    img = arr[..., :3][..., ::-1].copy()

                # 移回原位
                win32gui.SetWindowPos(
                    hwnd, win32con.HWND_NOTOPMOST,
                    orig_x, orig_y, orig_w, orig_h,
                    win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW |
                    win32con.SWP_NOZORDER)
            except Exception as e:
                self._log(f"temporary_restore_inner_failed hwnd={hwnd} error={e}")
                # 尽力复原
                try:
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_NOTOPMOST,
                        orig_x, orig_y, orig_w, orig_h,
                        win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW |
                        win32con.SWP_NOZORDER)
                except Exception:
                    pass
                return None

            if img is not None and self._is_render_healthy(img):
                self._last_pw_client_only = False
                self._log(f"temporary_restore_success hwnd={hwnd} size={img.shape[1]}x{img.shape[0]}")
                return img
            return None
        except Exception as e:
            self._log(f"temporary_restore_failed hwnd={hwnd} error={e}")
            return None

    # =================================================================
    # 截图可用性检查（对齐原版 _looks_usable）
    # =================================================================

    def _looks_usable(self, image: np.ndarray) -> bool:
        """检查截图是否可用（使用直方图分析）。

        对齐原版：拒绝全灰/全黑/全白等 GPU 渲染占位帧。
        使用 min_luma / max_luma / buckets / dominants 分析。
        """
        if image is None or image.size == 0:
            return False

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return False

        # 裁剪到导航栏区域（顶部 55px，左侧 55px）
        nav_h = min(55, h)
        nav_w = min(55, w)
        if nav_h <= 0 or nav_w <= 0:
            return False

        nav = image[:nav_h, :nav_w]

        # 步进采样
        step_x = max(1, nav_w // 10)
        step_y = max(1, nav_h // 10)
        pixels = []
        for y in range(0, nav_h, step_y):
            for x in range(0, nav_w, step_x):
                pixel = nav[y, x]
                luma = int(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0])
                pixels.append(luma)

        if not pixels:
            return False

        min_luma = min(pixels)
        max_luma = max(pixels)

        # 亮度直方图分桶
        buckets = [0] * 10
        for p in pixels:
            idx = min(9, p * 10 // 256)
            buckets[idx] += 1

        total = len(pixels)
        if total == 0:
            return False

        # 主导桶占比超过 90% → 几乎单色
        dominants = sorted(buckets, reverse=True)
        dominant_ratio = dominants[0] / total

        # 灰度范围太小 → 全黑/全白/全灰（GPU 渲染占位帧）
        luma_range = max_luma - min_luma

        # 只有范围极小且主导桶占比极高才判定为 GPU 占位帧
        if luma_range < 3 and dominant_ratio > 0.95:
            return False

        # 非黑像素数
        non_black = sum(1 for p in pixels if p > 10)
        if non_black < self.SCREENSHOT_MIN_NONBLACK_NAV:
            return False

        # 额外拒绝"白壳"占位帧：微信 4.x 在 park 态 GPU 未渲染时，
        # PrintWindow 可能返回几乎纯白且标准差极低的壳子图。
        if max_luma > 230 and luma_range < 20:
            return False

        return True

    def _is_render_healthy(self, image: np.ndarray,
                           local_var_min: float = 2.0,
                           blank_threshold: float = 10.0) -> bool:
        """渲染健康度校验（参考 wechat-ai-reply-main）。

        微信 4.x / CEF 在 park/最小化/刚还原时 GPU 可能只提交部分内容（半渲染），
        此时 mean/std 可能正常，但局部方差极低。本函数用边缘/纹理密度判断是否"真的画完了"。
        """
        if image is None or image.size == 0:
            return False
        try:
            arr = np.asarray(image)
            if arr.size == 0:
                return False
            if arr.ndim == 3:
                gray = arr.mean(axis=2).astype(np.float32)
            else:
                gray = arr.astype(np.float32)

            if not np.isfinite(float(gray.mean())):
                return False
            if float(gray.mean()) < blank_threshold:
                return False
            if float(gray.std()) < 5:
                return False

            h, w = gray.shape
            if h < 8 or w < 8:
                return False

            step = max(1, min(h, w) // 200)
            small = gray[::step, ::step]
            dx = np.abs(np.diff(small, axis=1)).mean()
            dy = np.abs(np.diff(small, axis=0)).mean()
            local_var = float((dx + dy) * 0.5)
            return local_var >= local_var_min
        except Exception:
            # 校验本身出错时保守放行，不阻断主流程
            return True

    # =================================================================
    # 请求窗口重绘（对齐原版 _request_window_redraw）
    # =================================================================

    def _request_window_redraw(self, hwnd: int) -> None:
        """请求窗口重绘而不激活或移动鼠标。

        对齐原版：使用 ShowWindow/SWP_NOREDRAW 位，
        RedrawWindow(RDW_INVALIDATE|RDW_UPDATENOW|RDW_ERASE)，
        以及 DwmFlush 强制合成。
        """
        try:
            SW_SHOWNOACTIVATE = 4
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_NOREDRAW = 0x0008

            if self._user32.IsWindow(hwnd):
                # 先 ShowWindow 确保窗口可见
                self._user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                # 强制重绘位置
                self._user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                          SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                                          SWP_NOACTIVATE | SWP_NOREDRAW)
                # RedrawWindow
                RDW_INVALIDATE = 0x0001
                RDW_UPDATENOW = 0x0100
                RDW_ERASE = 0x0004
                self._user32.RedrawWindow(hwnd, None, None,
                                          RDW_INVALIDATE | RDW_UPDATENOW | RDW_ERASE)
                self._user32.UpdateWindow(hwnd)
                # DwmFlush
                try:
                    ctypes.windll.dwmapi.DwmFlush()
                except Exception:
                    pass
        except Exception as e:
            self._log(f"window_redraw_failed hwnd={hwnd} error={e}")

    # =================================================================
    # 窗口截图（带重试 + 完整性校验，对齐原版）
    # =================================================================

    def capture_window(self, rect=None, *, hwnd=None, subdir="screenshots",
                       prefix="capture", capture_backend: Optional[str] = None,
                       force_refresh: bool = False):
        """截取窗口客户区，对齐原版 ScreenCapture.capture_window。

        签名：capture_window(self, rect=None, *, hwnd, subdir, prefix)

        流程：
          1. PrintWindow（优先）→ ImageGrab → 请求重绘后重试
          2. 每步都做 _looks_usable 校验
          3. 完成后保存截图，返回文件路径（str）或失败 CaptureResult

        Args:
            rect: 截取区域 Dict[str, int] | None（None=全窗口）
            hwnd: 窗口句柄
            subdir: 保存子目录
            prefix: 文件名前缀
            capture_backend: "imagegrab" / "printwindow" / None(自动)
            force_refresh: 截图前先强制窗口重绘并等待合成器刷新。

                微信是 CEF/GPU 自绘窗口，PrintWindow 存在**帧滞后**：界面刚
                切换（如点开某个会话）时立刻截图，拿到的往往是切换前的旧帧。
                实测：点击未读后隔 4 秒重截，两张图 md5 完全相同。
                在"刚点击过、需要最新画面"的场景必须置 True。
        """
        # 兼容旧调用：capture_window(hwnd) 把 hwnd 当位置参数传入
        if isinstance(rect, int) and rect > 0:
            hwnd = hwnd or rect
            rect = None

        target_hwnd = hwnd or self._hwnd_wechat
        if not target_hwnd:
            return CaptureResult(success=False, error="No window handle")

        # —— 强制刷新：先重绘，再等合成器 flush，避免拿到滞后旧帧 ——
        if force_refresh:
            try:
                self._request_window_redraw(target_hwnd)
                # 点击/切换视图后 CEF 需更充分时间提交新帧，0.12s 太短易截到旧帧
                time.sleep(0.35)
            except Exception:
                pass

        # —— 屏外/最小化感知捕获（参考原版虚拟外屏思路，但不再把窗口拉回桌面）——
        # PrintWindow 捕获的是窗口自身绘制表面，窗口位于负坐标屏外也能抓到真实画面，
        # 因此截图永远不需要 unpark 到桌面中央（那会造成明显可见的窗口跳动）。
        # 仅当窗口最小化（客户区停止绘制）时，用 restore_offscreen 恢复为屏外渲染态，
        # 全程不进入物理桌面，用户无感；恢复后窗口仍处于屏外，无需再 park 回去。
        wm = self._window_manager
        if wm is not None:
            try:
                if wm.is_minimized(target_hwnd):
                    wm.restore_offscreen(target_hwnd)
            except Exception:
                pass

        if True:  # 截图主流程（窗口已确保在屏外可用，无需挪动；原 try/finally 已移除）
            # 确定后端
            if capture_backend is None:
                backend = self._resolve_backend()
            else:
                backend = capture_backend.lower()

            # 若窗口当前位于屏外（负坐标），强制 PrintWindow：
            # ImageGrab 抓取的是物理屏幕缓冲，屏外窗口不在其中；而 PrintWindow
            # 走窗口自身绘制表面，屏外也能抓到真实画面，且完全不需要移动窗口。
            forced_printwindow = False
            if backend != "printwindow" and wm is not None:
                try:
                    wrect = wm.get_rect(target_hwnd)
                    if wrect is not None:
                        # 只要微信由 WeChatWindowManager 托管，就优先 PrintWindow。
                        # PW_RENDERFULLCONTENT 会要求 CEF/Chromium 把完整内容渲染到
                        # 目标 DC，无视窗口是否被其它窗口遮挡，解决"桌面被挡住就截错"的问题。
                        backend = "printwindow"
                        forced_printwindow = True
                except Exception:
                    pass

            for attempt in range(self.SCREENSHOT_MAX_RETRIES):
                if backend == "printwindow":
                    img = self._capture_printwindow(target_hwnd, rect)
                    if img is not None and self._looks_usable(img):
                        h, w = img.shape[:2]
                        self._last_screenshot = img
                        self._last_screenshot_attempts = attempt + 1
                        result = CaptureResult(success=True, image=img, width=w, height=h)
                        self._save_capture(img, subdir, prefix)
                        self._fire_captured(img)
                        return result
                    else:
                        self._log(f"printwindow_capture_unusable hwnd={target_hwnd}; fallback=imagegrab")
                        backend = "imagegrab"
                        continue

                elif backend == "imagegrab":
                    rect = ctypes.wintypes.RECT()
                    self._user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
                    ww = rect.right - rect.left
                    wh = rect.bottom - rect.top
                    bbox = {
                        "left": rect.left, "top": rect.top,
                        "right": rect.right, "bottom": rect.bottom,
                    }
                    img = self._capture_imagegrab_rect(bbox)
                    usable = self._looks_usable(img) if img is not None else False
                    if img is not None and usable:
                        h, w = img.shape[:2]
                        self._last_screenshot = img
                        self._last_screenshot_attempts = attempt + 1
                        if self._imagegrab_capture_unusable and attempt > 0:
                            self._log(f"capture_recovered_after_redraw hwnd={target_hwnd}")
                            self._imagegrab_capture_unusable = False
                        result = CaptureResult(success=True, image=img, width=w, height=h)
                        self._save_capture(img, subdir, prefix)
                        self._fire_captured(img)
                        return result
                    else:
                        if attempt == 0:
                            self._log(f"capture_unusable_flat_image hwnd={target_hwnd}")
                            self._imagegrab_capture_unusable = True
                        self._request_window_redraw(target_hwnd)
                        time.sleep(self.SCREENSHOT_RETRY_DELAY)
                        continue

                if attempt < self.SCREENSHOT_MAX_RETRIES - 1:
                    time.sleep(self.SCREENSHOT_RETRY_DELAY)

            # 最后尝试 PrintWindow
            if self._imagegrab_capture_unusable:
                pw_result = self._capture_printwindow(target_hwnd)
                if pw_result is not None and self._looks_usable(pw_result):
                    h, w = pw_result.shape[:2]
                    self._last_screenshot = pw_result
                    self._log(f"printwindow_capture_used fallback hwnd={target_hwnd}")
                    result = CaptureResult(success=True, image=pw_result, width=w, height=h)
                    self._save_capture(pw_result, subdir, prefix)
                    self._fire_captured(pw_result)
                    return result

            # 终级兜底：PrintWindow 始终失败且窗口在屏外/强制 PrintWindow 时，
            # 短暂恢复窗口到屏幕边缘，mss 抓屏后再移回原位。
            if backend == "printwindow" or forced_printwindow:
                restore_img = self._capture_via_temporary_restore(target_hwnd)
                if restore_img is not None and self._looks_usable(restore_img):
                    h, w = restore_img.shape[:2]
                    self._last_screenshot = restore_img
                    self._log(f"temporary_restore_capture_used hwnd={target_hwnd}")
                    result = CaptureResult(success=True, image=restore_img, width=w, height=h)
                    self._save_capture(restore_img, subdir, prefix)
                    self._fire_captured(restore_img)
                    return result

            return CaptureResult(success=False, error="capture failed after retries")

    def _fire_captured(self, img) -> None:
        """截图成功后触发回调（若有），用于实时预览推送。"""
        cb = getattr(self, "on_captured", None)
        if callable(cb) and img is not None:
            try:
                cb(img)
            except Exception:
                pass

    def _resolve_backend(self) -> str:
        """根据配置解析截图后端。"""
        if self._prefer_printwindow:
            return "printwindow"
        backend = self.config.get("capture_backend", "imagegrab")
        if isinstance(backend, str):
            return backend.lower()
        return "imagegrab"

    def _save_capture(self, img: np.ndarray, subdir: str = "screenshots",
                      prefix: str = "capture") -> Optional[str]:
        """保存截图到文件，对齐原版内部保存逻辑。"""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            parent = Path(self._data_dir) if self._data_dir else Path(".")
            base = parent / subdir
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"{prefix}_{ts}.png"

            import cv2
            cv2.imwrite(str(path), img)

            # 保存 latest.png
            latest = base / "latest.png"
            cv2.imwrite(str(latest), img)

            return str(path)
        except Exception as e:
            self._log(f"save_capture failed: {e}")
            return None

    def save(self, rect=None, *, hwnd=None, subdir="screenshots",
             prefix="capture") -> Optional[str]:
        """截图并保存到文件，对齐原版 save 方法。

        Returns: 保存的文件路径，失败返回 None。
        """
        target_hwnd = hwnd or self._hwnd_wechat
        if not target_hwnd:
            return None

        # 最小化时先恢复到屏外渲染态（不闪现桌面），保证 PrintWindow 可读到画面
        wm = self._window_manager
        if wm is not None:
            try:
                if wm.is_minimized(target_hwnd):
                    wm.restore_offscreen(target_hwnd)
            except Exception:
                pass

        img = self._capture_printwindow(target_hwnd, rect)
        if img is not None and self._looks_usable(img):
            return self._save_capture(img, subdir, prefix)

        # 回退到 ImageGrab
        if rect and isinstance(rect, dict):
            img = self._capture_imagegrab_rect(rect)
        else:
            img = self._capture_imagegrab_rect({})
        if img is not None and self._looks_usable(img):
            return self._save_capture(img, subdir, prefix)

        return None

    # =================================================================
    # 截图保存（对齐原版）
    # =================================================================

    def capture_to_file(self, filepath, hwnd=None):
        result = self.capture_window(hwnd)
        if not result.success or result.image is None:
            return False
        try:
            import cv2
            cv2.imwrite(filepath, result.image)
            return True
        except Exception:
            return False

    def save_screenshot(self, hwnd=None, subdir: str = "screenshots",
                        prefix: str = "screenshot") -> Optional[str]:
        """保存截图（带时间戳），对齐原版 save 方法。"""
        try:
            result = self.capture_window(hwnd)
            if not result.success or result.image is None:
                return None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            parent = Path(self._data_dir) if self._data_dir else Path(".")
            base = parent / subdir
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"{prefix}_{ts}.png"

            import cv2
            cv2.imwrite(str(path), result.image)

            # 保存 latest.png
            latest = base / "latest.png"
            cv2.imwrite(str(latest), result.image)

            return str(path)
        except Exception as e:
            self._log(f"save_screenshot failed: {e}")
            return None

    # =================================================================
    # 聊天区截图
    # =================================================================

    def capture_chat_area(self, hwnd=None):
        result = self.capture_window(hwnd)
        if not result.success or result.image is None:
            return result
        h, w = result.image.shape[:2]
        chat_x = int(w * 0.28)
        chat_crop = result.image[:, chat_x:].copy()
        return CaptureResult(success=True, image=chat_crop,
                             width=chat_crop.shape[1], height=chat_crop.shape[0])

    # =================================================================
    # 区域截图
    # =================================================================

    def capture_region(self, x: int, y: int, w: int, h: int) -> CaptureResult:
        try:
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            if img is None:
                return CaptureResult(success=False, error="ImageGrab returned None")
            bgr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
            return CaptureResult(success=True, image=bgr, width=bgr.shape[1], height=bgr.shape[0])
        except Exception as e:
            return CaptureResult(success=False, error=str(e))

    # =================================================================
    # 窗口列表
    # =================================================================

    def get_window_list(self):
        windows = []

        def _enum_callback(hwnd, _):
            if self._user32.IsWindowVisible(hwnd):
                title_buf = ctypes.create_unicode_buffer(512)
                self._user32.GetWindowTextW(hwnd, title_buf, 512)
                title = title_buf.value
                if title:
                    rect = ctypes.wintypes.RECT()
                    self._user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    windows.append({
                        "hwnd": hwnd, "title": title,
                        "width": rect.right - rect.left,
                        "height": rect.bottom - rect.top,
                    })
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND,
                                          ctypes.wintypes.LPARAM)
        self._user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)
        return windows

    # =================================================================
    # 工具方法
    # =================================================================

    def _log(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger(msg)
            except Exception:
                pass

    @property
    def last_screenshot(self) -> Optional[np.ndarray]:
        return self._last_screenshot

    @property
    def last_screenshot_attempts(self) -> int:
        return self._last_screenshot_attempts