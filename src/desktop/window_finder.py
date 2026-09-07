"""窗口查找 —— 严格对齐原版 app.desktop.window_finder.WeChatWindowFinder v3。

原版 v3 规则：
  1. 标题必须含关键词（保留 v2 行为）
  2. 标题不能含明显是别的应用的特征词（浏览器、编辑器等）
  3. 窗口尺寸必须正常（width >= 600, height >= 450）
  4. 多候选时挑尺寸最大的（真微信主窗口通常最大）

核心流程：
  find() → _find_win32_exact_main_window() （严格 Win32 枚举）
         → 失败则回退到枚举 + 关键词/负向词/尺寸过滤
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
from typing import List, Optional, Tuple

from ..capture.screen_capture import WindowInfo


# =====================================================================
# 常量（对齐原版 app.desktop.wechat_environment）
# =====================================================================

MAIN_WINDOW_MIN_WIDTH = 600
MAIN_WINDOW_MIN_HEIGHT = 450
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WECHAT_MAIN_EXE_NAMES = frozenset({"wechat.exe", "weixin.exe"})

# v3 负向关键词：标题含这些特征则应排除
_NEGATIVE_KEYWORDS = [
    "客服助手", "Chrome", "Edge", "Firefox", "Safari", "Opera", "Brave",
    "浏览器", "Browser", "Visual Studio", "VS Code", "PyCharm", "Sublime",
    "Notepad", "记事本", "Cursor", "Claude", "ChatGPT", "Gemini",
    "命令提示符", "PowerShell", "Terminal", "cmd.exe", "资源管理器",
    "Explorer", ".py - ", ".md - ", ".txt - ", " - 知乎", " - 百度",
    " - Google", "Agent",
]


# =====================================================================
# WindowFinder
# =====================================================================

class WindowFinder:
    """微信窗口查找器 —— 对齐原版 WeChatWindowFinder v3。"""

    MIN_WIDTH = MAIN_WINDOW_MIN_WIDTH
    MIN_HEIGHT = MAIN_WINDOW_MIN_HEIGHT

    def __init__(self, user32=None):
        self._user32 = user32 or ctypes.windll.user32
        self._pid_self = os.getpid()
        self._user32.GetWindowTextW.argtypes = [
            ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]
        self._user32.IsWindowVisible.restype = ctypes.wintypes.BOOL
        self._user32.GetWindowRect.argtypes = [
            ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.RECT)]
        self._user32.GetWindowRect.restype = ctypes.wintypes.BOOL

    # ==============================================================
    # 主入口：find()
    # ==============================================================

    def find(self, keywords: Optional[list] = None) -> Optional[WindowInfo]:
        """对齐原版 WeChatWindowFinder.find()：
        先尝试 _find_win32_exact_main_window，失败则回退。
        """
        if keywords is None:
            keywords = ["微信", "WeChat", "Weixin"]

        # Step 1: 严格 Win32 枚举（进程验证 + 尺寸过滤）
        exact = self._find_win32_exact_main_window(keywords)
        if exact is not None:
            return exact

        # Step 2: 回退 —— 枚举所有可见窗口，按关键词/负向词/尺寸过滤
        all_windows = self.list_windows()
        candidates: List[WindowInfo] = []
        for win in all_windows:
            title = win.title or ""
            if not any(k in title for k in keywords):
                continue
            if any(neg in title for neg in _NEGATIVE_KEYWORDS):
                continue
            # 铁门槛（与严格枚举路径一致）：非微信进程必须标题精确匹配，
            # 防"微信.txt - 记事本"这类窗口在回退路径里被当成微信
            title_exact = title in ("微信", "WeChat", "Weixin")
            exe_path = _query_process_image_path(
                self._get_window_pid(win.hwnd))
            if not _looks_like_wechat_main_process(exe_path) and not title_exact:
                continue
            if win.width < self.MIN_WIDTH or win.height < self.MIN_HEIGHT:
                # 尺寸太小 → 验证进程，可能是最小化的微信窗口
                if not _looks_like_wechat_main_process(exe_path):
                    continue
            candidates.append(win)

        if not candidates:
            import logging, sys
            _log = logging.getLogger("WindowFinder")
            kw_wins = [(w.title, w.width, w.height) for w in all_windows
                       if any(k in (w.title or "") for k in keywords)]
            neg_wins = [(w.title, w.width, w.height) for w in all_windows
                        if any(k in (w.title or "") for k in keywords)
                        and any(neg in (w.title or "") for neg in _NEGATIVE_KEYWORDS)]
            size_wins = [(w.title, w.width, w.height) for w in all_windows
                         if any(k in (w.title or "") for k in keywords)
                         and not any(neg in (w.title or "") for neg in _NEGATIVE_KEYWORDS)
                         and (w.width < self.MIN_WIDTH or w.height < self.MIN_HEIGHT)]
            msg = (
                "find: no candidate. step1 failed, "
                "step2 scanned %d windows, "
                "kw_wins=%s, neg_wins=%s, size_wins=%s"
            )
            _log.warning(msg, len(all_windows), kw_wins, neg_wins, size_wins)
            print(msg % (len(all_windows), kw_wins, neg_wins, size_wins),
                  file=sys.stderr, flush=True)
            extra = self.dump_all_windows()
            kw_all = [(d["title"], d["w"], d["h"], d["exe"]) for d in extra
                      if any(k in (d["title"] or "") for k in keywords)]
            print("find: all kw windows (any size): %s" % kw_all,
                  file=sys.stderr, flush=True)
            return None

        candidates.sort(key=lambda w: w.width * w.height, reverse=True)
        return candidates[0]

    # ==============================================================
    # 严格 Win32 枚举（对齐原版 _find_win32_exact_main_window）
    # ==============================================================

    def _find_win32_exact_main_window(
        self, keywords: Optional[list] = None
    ) -> Optional[WindowInfo]:
        """使用 EnumWindows 回调严格匹配微信主窗口。

        对齐原版 v3：
        1. 标题含关键词 + 不含负向词 → 直接通过
        2. 标题不含关键词 → 验证进程是否 wechat.exe/weixin.exe
        3. 窗口尺寸 >= 600x450
        4. 收集所有候选，按面积排序取最大
        """
        if keywords is None:
            keywords = ["微信", "WeChat", "Weixin"]

        candidates: List[WindowInfo] = []
        debug_info: List[dict] = []  # 调试用

        _GetWindowThreadProcessId = self._user32.GetWindowThreadProcessId
        _GetWindowThreadProcessId.argtypes = [
            ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.DWORD)]
        _GetWindowThreadProcessId.restype = ctypes.wintypes.DWORD

        def _callback(hwnd, _):
            try:
                buf = ctypes.create_unicode_buffer(512)
                self._user32.GetWindowTextW(hwnd, buf, 512)
                title = buf.value.strip()

                visible = bool(self._user32.IsWindowVisible(hwnd))

                r = ctypes.wintypes.RECT()
                self._user32.GetWindowRect(hwnd, ctypes.byref(r))
                width = r.right - r.left
                height = r.bottom - r.top

                kw_match = any(k in title for k in keywords)
                neg_match = any(neg in title for neg in _NEGATIVE_KEYWORDS)
                title_ok = kw_match and not neg_match

                if not title or not visible:
                    return True

                # 获取进程信息（用于进程验证 + 调试）
                pid = ctypes.c_ulong()
                _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                exe_path = _query_process_image_path(pid.value)
                is_wechat_proc = _looks_like_wechat_main_process(exe_path)

                # 铁门槛（修复：用户开了"微信.txt"记事本后窗口被误认成微信）：
                # 1) 负向词（记事本/Notepad/.txt - 等）一票否决——即便标题含"微信"
                # 2) 非微信进程时，标题必须**精确**等于 微信/WeChat/Weixin 才算候选，
                #    杜绝"微信.txt""微信备份.docx"这类含关键词的别家窗口靠面积大上位
                title_exact = title in ("微信", "WeChat", "Weixin")
                if neg_match:
                    return True
                if not is_wechat_proc and not title_exact:
                    return True

                # 调试：记录大窗口
                if width >= MAIN_WINDOW_MIN_WIDTH and height >= MAIN_WINDOW_MIN_HEIGHT:
                    debug_info.append({
                        "title": title, "w": width, "h": height,
                        "kw_match": kw_match, "neg_match": neg_match,
                        "visible": visible, "exe": exe_path,
                    })
                elif is_wechat_proc:
                    debug_info.append({
                        "title": title, "w": width, "h": height,
                        "kw_match": kw_match, "neg_match": neg_match,
                        "visible": visible, "exe": exe_path,
                        "small": True,
                    })

                candidates.append(WindowInfo(
                    hwnd=hwnd, title=title, width=width, height=height,
                    x=r.left, y=r.top,
                ))
            except Exception:
                pass
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        self._user32.EnumWindows(WNDENUMPROC(_callback), 0)

        if not candidates:
            import logging, sys
            _log = logging.getLogger("WindowFinder")
            msg = (
                "_find_win32_exact_main_window: no candidate. "
                "debug (kw_match windows with size>=600x450): %s"
            )
            _log.warning(msg, debug_info)
            print(msg % debug_info, file=sys.stderr, flush=True)
            return None

        candidates.sort(key=lambda w: w.width * w.height, reverse=True)
        return candidates[0]

    # ==============================================================
    # 辅助方法
    # ==============================================================

    def find_wechat(self, keywords: Optional[list] = None) -> Optional[WindowInfo]:
        """便捷方法，等同于 find()。"""
        return self.find(keywords)

    def _get_window_pid(self, hwnd: int) -> int:
        """获取窗口所属进程 PID。"""
        pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def dump_all_windows(self) -> List[dict]:
        """调试：列出所有可见窗口（标题、尺寸、PID）"""
        result: List[dict] = []
        _GetWindowThreadProcessId = self._user32.GetWindowThreadProcessId
        _GetWindowThreadProcessId.argtypes = [
            ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.DWORD)]
        _GetWindowThreadProcessId.restype = ctypes.wintypes.DWORD

        def _cb(hwnd, _):
            if not self._user32.IsWindowVisible(hwnd):
                return True
            buf = ctypes.create_unicode_buffer(512)
            self._user32.GetWindowTextW(hwnd, buf, 512)
            title = buf.value.strip()
            r = ctypes.wintypes.RECT()
            self._user32.GetWindowRect(hwnd, ctypes.byref(r))
            w = r.right - r.left
            h = r.bottom - r.top
            pid = ctypes.c_ulong()
            _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            exe = _query_process_image_path(pid.value)
            result.append({
                "hwnd": hwnd, "title": title, "w": w, "h": h,
                "pid": pid.value, "exe": exe,
            })
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        self._user32.EnumWindows(WNDENUMPROC(_cb), 0)
        return result

    def find_all_wechat(self, keywords: Optional[list] = None) -> List[WindowInfo]:
        """返回所有匹配标题的窗口（不限于第一个）。"""
        if keywords is None:
            keywords = ["微信", "WeChat"]
        return self._find_by_keywords(keywords)

    def _collect_visible_text(self, win) -> Tuple[str, list]:
        """收集窗口子控件的可见文本，返回 (text_blob, controls)。"""
        hwnd = getattr(win, "hwnd", win)
        texts: List[str] = []
        controls: List[dict] = []
        if not hwnd:
            return ("", [])
        user32 = ctypes.windll.user32

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND,
                            ctypes.wintypes.LPARAM)
        def _enum(hwnd_child, _):
            try:
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd_child, buf, 512)
                txt = buf.value.strip()
                if txt:
                    texts.append(txt)
                    controls.append({"text": txt})
            except Exception:
                pass
            return True

        try:
            user32.EnumChildWindows(int(hwnd), _enum, 0)
        except Exception:
            pass
        return ("\n".join(texts[:300]), controls[:300])

    def list_windows(self) -> List[WindowInfo]:
        """枚举所有可见窗口并结构化。"""
        windows: List[WindowInfo] = []
        cb = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        def _cb(hwnd, _):
            if not self._user32.IsWindowVisible(hwnd):
                return True
            title = self._get_title(hwnd)
            if not title:
                return True
            windows.append(self._make_window_info(hwnd, title))
            return True

        self._user32.EnumWindows(cb(_cb), 0)
        return windows

    def _find_by_keywords(self, keywords: List[str]) -> List[WindowInfo]:
        matches: List[WindowInfo] = []
        for win in self.list_windows():
            for kw in keywords:
                if kw in win.title:
                    matches.append(win)
                    break
        return matches

    def _make_window_info(self, hwnd: int, title: str) -> WindowInfo:
        rect = ctypes.wintypes.RECT()
        self._user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return WindowInfo(
            hwnd=hwnd,
            title=title,
            width=rect.right - rect.left,
            height=rect.bottom - rect.top,
            x=rect.left,
            y=rect.top,
        )

    @staticmethod
    def _get_title(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value

    def focus_target(self, hwnd: int) -> bool:
        """聚焦指定窗口（对齐原版 WeChatWindowFinder.focus_target）。"""
        if not hwnd:
            return False
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False


# =====================================================================
# 辅助函数（对齐原版 app.desktop.wechat_environment）
# =====================================================================

def _looks_like_wechat_main_process(exe_path: str) -> bool:
    """检查进程路径是否属于微信主程序。

    对齐原版 _looks_like_wechat_main_process。
    """
    normalized = str(exe_path or "").strip().replace("/", "\\")
    try:
        exe_name = normalized.rsplit("\\", 1)[-1].lower()
    except Exception:
        return False
    return exe_name in WECHAT_MAIN_EXE_NAMES


def _query_process_image_path(pid: int) -> str:
    """通过进程 PID 查询可执行文件路径。

    对齐原版 _query_process_image_path（使用 QueryFullProcessImageNameW）。
    """
    if not pid or os.name != "nt":
        return ""

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return ""

        size = ctypes.c_uint(32768)
        buffer = ctypes.create_unicode_buffer(size.value)

        kernel32.QueryFullProcessImageNameW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_uint)]
        kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int

        if kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)):
            kernel32.CloseHandle(handle)
            return buffer.value

        kernel32.CloseHandle(handle)
    except Exception:
        pass
    return ""


# =====================================================================
# 兼容别名
# =====================================================================

WeChatWindowFinder = WindowFinder