"""微信运行环境检测：版本 / 焦点 / 虚拟屏坐标。

依赖 win32 API，用于在执行自动化前确认微信窗口可用、定位并归一化坐标。
"""

from __future__ import annotations

import fnmatch
import os

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

from .wechat_window_manager import WeChatWindowManager
from .window_finder import WindowFinder


# GetSystemMetrics 索引
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_CXSCREEN = 0
SM_CYSCREEN = 1


@dataclass
class VirtualScreen:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass
class WeChatWindowState:
    found: bool = False
    hwnd: Optional[int] = None
    focused: bool = False
    visible: bool = False
    version: str = ""
    rect: Optional[tuple] = None   # (x, y, w, h)


class WeChatEnvironment:
    def __init__(self, finder: Optional[WindowFinder] = None,
                 manager: Optional[WeChatWindowManager] = None):
        self._user32 = ctypes.windll.user32
        self._finder = finder or WindowFinder()
        self._manager = manager or WeChatWindowManager(self._finder)
        self._hwnd_wechat: Optional[int] = None

    # -------------------------------------------------- 虚拟屏坐标
    def get_virtual_screen(self) -> VirtualScreen:
        return VirtualScreen(
            x=self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            y=self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            width=self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            height=self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    def get_primary_screen(self) -> tuple:
        """主屏 (width, height)。"""
        return (self._user32.GetSystemMetrics(SM_CXSCREEN),
                self._user32.GetSystemMetrics(SM_CYSCREEN))

    def hwnd_to_virtual_coord(self, hwnd: int) -> tuple:
        """把窗口屏幕坐标换算到虚拟屏坐标系（一般一致，多屏时使用）。"""
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (0, 0)
        vs = self.get_virtual_screen()
        return (rect.left - vs.x, rect.top - vs.y)

    # -------------------------------------------------- 焦点
    def is_focused(self, hwnd: Optional[int] = None) -> bool:
        target = self._resolve_hwnd(hwnd)
        if not target:
            return False
        return self._user32.GetForegroundWindow() == target

    # -------------------------------------------------- 版本检测
    def check_version(self, hwnd: Optional[int] = None) -> str:
        """检测微信版本：窗口标题中若有“微信数字”则提取，否则返回空串。"""
        target = self._resolve_hwnd(hwnd)
        if not target:
            return ""
        title = WindowFinder._get_title(target)
        return self._extract_version(title)

    @staticmethod
    def _extract_version(text: str) -> str:
        import re
        match = re.search(r"(\d+(?:\.\d+){2,})", text)
        return match.group(1) if match else ""

    # -------------------------------------------------- 综合状态
    def inspect(self) -> WeChatWindowState:
        win = self._finder.find_wechat()
        if not win or not win.hwnd:
            return WeChatWindowState(found=False)
        self._hwnd_wechat = win.hwnd
        return WeChatWindowState(
            found=True,
            hwnd=win.hwnd,
            focused=self.is_focused(win.hwnd),
            visible=self._manager.is_visible(win.hwnd),
            version=self.check_version(win.hwnd),
            rect=(win.x, win.y, win.width, win.height),
        )

    # -------------------------------------------------- 内部
    def _resolve_hwnd(self, hwnd: Optional[int]) -> Optional[int]:
        target = hwnd or self._hwnd_wechat
        if target:
            return target
        win = self._finder.find_wechat()
        if win and win.hwnd:
            self._hwnd_wechat = win.hwnd
            return win.hwnd
        return None


# ---------------------------------------------------------------------------
# 模块级环境检测函数（兼容原项目 app.desktop.wechat_environment 顶层函数）
# ---------------------------------------------------------------------------

PROCESS_QUERY_LIMITED_INFORMATION = 4096
MAIN_WINDOW_MIN_WIDTH = 600
MAIN_WINDOW_MIN_HEIGHT = 450
WECHAT_MAIN_EXE_NAMES = {"wechat.exe", "weixin.exe"}


@dataclass
class WeChatWindowCandidate:
    """单个微信主窗口候选（供环境报告使用）。"""

    handle: int = 0
    title: str = ""
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {
            "handle": self.handle,
            "title": self.title,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


def _looks_like_wechat_main_rect(win) -> bool:
    """按最小尺寸判断窗口是否像微信主窗口。"""
    if win is None:
        return False
    width = int(getattr(win, "width", 0) or 0)
    height = int(getattr(win, "height", 0) or 0)
    return width >= MAIN_WINDOW_MIN_WIDTH and height >= MAIN_WINDOW_MIN_HEIGHT


def _normalize_allowed_versions(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    versions = []
    for v in raw:
        v = str(v).strip()
        if v:
            versions.append(v)
    return versions


def _selected_window(windows, selected_handle=None):
    if not windows:
        return None
    if selected_handle:
        for w in windows:
            if getattr(w, "handle", None) == selected_handle:
                return w
    return windows[0]


def _query_process_image_path(pid) -> str:
    """返回进程 pid 的可执行文件路径（ctypes，无 pywin32 依赖）。"""
    if not pid:
        return ""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        h = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid)
        )
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            needed = ctypes.c_uint(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(needed)):
                return buf.value
            return ""
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return ""


def file_version(path=None) -> str:
    """读取 PE 文件的版本字符串 (x.y.z.w)。"""
    if not path:
        return ""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        size = kernel32.GetFileVersionInfoSizeW(ctypes.c_wchar_p(path), None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        if not kernel32.GetFileVersionInfoW(ctypes.c_wchar_p(path), 0, size, buf):
            return ""
        transl = ctypes.c_void_p()
        transl_len = ctypes.c_uint()
        if not kernel32.VerQueryValueW(
            buf, ctypes.c_wchar_p("\\VarFileInfo\\Translation"),
            ctypes.byref(transl), ctypes.byref(transl_len),
        ):
            return ""
        lang = ctypes.cast(transl, ctypes.POINTER(ctypes.c_uint16 * 2))[0]
        lang_hex = f"{lang[0]:04x}{lang[1]:04x}"
        sub_block = ctypes.c_wchar_p(f"\\StringFileInfo\\{lang_hex}\\FileVersion")
        value_ptr = ctypes.c_void_p()
        value_len = ctypes.c_uint()
        if not kernel32.VerQueryValueW(
            buf, sub_block, ctypes.byref(value_ptr), ctypes.byref(value_len),
        ):
            return ""
        return ctypes.wstring_at(value_ptr, value_len.value - 1)
    except Exception:
        return ""


def wechat_process_version(hwnd=None) -> dict:
    if os.name != "nt":
        return {"ok": False, "reason": "runtime_not_windows",
                "pid": 0, "exe_path": "", "version": ""}
    if not hwnd:
        return {"ok": False, "reason": "wechat_window_handle_missing",
                "pid": 0, "exe_path": "", "version": ""}
    try:
        import ctypes
        user32 = ctypes.windll.user32
        pid = ctypes.c_uint()
        user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        exe_path = _query_process_image_path(int(pid.value))
        version = file_version(exe_path) if exe_path else ""
        return {
            "ok": bool(version),
            "reason": "" if version else "wechat_version_unknown",
            "pid": int(pid.value),
            "exe_path": exe_path,
            "version": version,
            "exe_name": os.path.basename(exe_path) if exe_path else "",
        }
    except Exception:
        return None


def is_version_allowed(version=None, allowed_versions=None) -> bool:
    if version is None:
        version = ""
    version = str(version).strip()
    if not version:
        return False
    for item in (allowed_versions or []):
        if not item:
            continue
        allowed = str(item).strip()
        if not allowed:
            continue
        if allowed == "*":
            return True
        if fnmatch.fnmatch(version, allowed):
            return True
        if version == allowed:
            return True
        if version.startswith(allowed.rstrip(".") + "."):
            return True
    return False


def list_wechat_main_windows() -> list:
    """枚举所有候选的微信主窗口，返回 WeChatWindowCandidate 列表。"""
    try:
        finder = WindowFinder()
        wins = finder.find_all_wechat(["微信", "WeChat", "Weixin"])
        candidates = []
        for w in wins:
            if not _looks_like_wechat_main_rect(w):
                continue
            candidates.append(WeChatWindowCandidate(
                handle=w.hwnd,
                title=w.title,
                left=w.x,
                top=w.y,
                right=w.x + w.width,
                bottom=w.y + w.height,
                width=w.width,
                height=w.height,
            ))
        return candidates
    except Exception:
        return []


def required_wechat_theme(config=None) -> dict:
    """检查微信主题是否被要求且受支持。"""
    cfg = {}
    if isinstance(config, dict):
        cfg = config.get("wechat", {}) or {}
    required = bool(cfg.get("require_supported_theme", False))
    if not required:
        return {"required": False, "theme": "", "ok": True, "reason": ""}
    from .wechat_theme import validate_wechat_theme
    res = validate_wechat_theme(cfg)
    res["required"] = True
    return res


def wechat_environment_report(config=None, selected_handle=None) -> dict:
    cfg = {}
    if isinstance(config, dict):
        cfg = config.get("wechat", {}) or {}
    if not cfg and isinstance(config, dict):
        cfg = config

    single_required = bool(cfg.get("single_wechat_window_required", True))
    version_policy = str(cfg.get("wechat_version_policy", "warn")).strip().lower()
    if version_policy not in ("off", "warn", "strict"):
        version_policy = "warn"
    allowed_versions = _normalize_allowed_versions(
        cfg.get("allowed_wechat_versions") or cfg.get("wechat_allowed_versions")
    )

    windows = list_wechat_main_windows()
    selected = _selected_window(windows, selected_handle)
    version_info = (
        wechat_process_version(selected.handle)
        if selected else {
            "ok": False, "reason": "wechat_window_not_found",
            "pid": 0, "exe_path": "", "version": "",
        }
    )
    version = (
        version_info.get("version", "") if isinstance(version_info, dict) else ""
    )

    version_required = version_policy == "strict"
    version_ok = True
    version_reason = ""
    if version_policy != "off" and allowed_versions:
        version_ok = is_version_allowed(version, allowed_versions)
        if not version_ok:
            version_reason = "wechat_version_not_verified"
    elif version_required and not version:
        version_ok = False
        version_reason = "wechat_version_unknown"

    single_ok = len(windows) <= 1 if single_required else True

    ok = True
    reason = ""
    if not selected:
        ok = False
        reason = "wechat_window_not_found"
    elif not single_ok:
        ok = False
        reason = "wechat_multiple_main_windows"
    elif version_required and not version_ok:
        ok = False
        reason = version_reason or "wechat_version_not_verified"
    elif not version_required and not version_ok and version_policy != "off":
        ok = False
        reason = "wechat_version_not_verified"

    return {
        "ok": ok,
        "reason": reason,
        "single_wechat_window_required": single_required,
        "window_count": len(windows),
        "windows": [w.to_dict() for w in windows],
        "selected_window": selected.to_dict() if selected else None,
        "version_policy": version_policy,
        "version_required": version_required,
        "allowed_wechat_versions": allowed_versions,
        "version_ok": bool(version_ok),
        "version_reason": version_reason,
        "version": version,
        "version_info": version_info,
    }


def wechat_environment_summary(report=None) -> dict:
    if report is None:
        report = wechat_environment_report()
    if not isinstance(report, dict):
        return {"ok": False, "summary": "invalid_report"}
    parts = []
    if report.get("ok"):
        parts.append("微信环境正常")
    else:
        parts.append(f"环境异常: {report.get('reason', 'unknown')}")
    parts.append(f"窗口数: {report.get('window_count', 0)}")
    if report.get("version"):
        parts.append(f"版本: {report.get('version')}")
    else:
        parts.append("版本: 未知")
    return {"ok": bool(report.get("ok", False)), "summary": "; ".join(parts)}


__all__ = [
    "WeChatEnvironment",
    "WeChatWindowState",
    "VirtualScreen",
    "WeChatWindowCandidate",
    "list_wechat_main_windows",
    "wechat_environment_report",
    "wechat_environment_summary",
    "wechat_process_version",
    "file_version",
    "is_version_allowed",
    "required_wechat_theme",
    "MAIN_WINDOW_MIN_WIDTH",
    "MAIN_WINDOW_MIN_HEIGHT",
    "WECHAT_MAIN_EXE_NAMES",
    "SM_XVIRTUALSCREEN", "SM_YVIRTUALSCREEN",
    "SM_CXVIRTUALSCREEN", "SM_CYVIRTUALSCREEN",
]