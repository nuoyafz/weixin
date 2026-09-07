#!/usr/bin/env python3
"""一次性：截图当前微信窗口 + 本地 RapidOCR 识别，分步计时并打印结果。

微信可能最小化到托盘（WindowFinder 只认可见窗口找不到），本脚本改为：
  1) 按进程名(weixin.exe/wechat.exe) + 标题定位真正的微信主窗口
  2) 屏外恢复渲染（用户无感）
  3) 截图 -> 本地 OCR + 布局解析
  4) 分步计时并打印

用法：
    python tools/capture_ocr_now.py
"""
from __future__ import annotations

import sys
import time
import os
import cv2
import ctypes
import ctypes.wintypes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.desktop.wechat_window_manager import WeChatWindowManager
from src.capture.screen_capture import ScreenCapture, CaptureResult
from src.clean_perception.reader import WechatScreenReader


def _fmt_dt(seconds: float) -> str:
    ms = seconds * 1000.0
    return f"{ms:.0f}ms" if ms < 1000 else f"{seconds:.2f}s"


def find_wechat_hwnd():
    """枚举全部顶层窗口，按 weixin.exe/wechat.exe 进程 + 标题含'微信/WeChat'定位主窗口。"""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
    GetWindowTextW.restype = ctypes.c_int

    def pid_name(pid):
        try:
            h = kernel32.OpenProcess(0x0410, False, pid)
            if not h:
                return "?"
            buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleFileNameExW(h, 0, buf, 260)
            kernel32.CloseHandle(h)
            return buf.value.split("\\")[-1].lower()
        except Exception:
            return "?"

    found = []
    def enum_cb(hwnd, _):
        buf = ctypes.create_unicode_buffer(512)
        GetWindowTextW(hwnd, buf, 512)
        title = buf.value
        pid = ctypes.wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = pid_name(pid.value)
        if ("weixin.exe" in name or "wechat.exe" in name) and ("微信" in title or "WeChat" in title):
            found.append((hwnd, title, name, pid.value))
        return True

    EnumWindows = user32.EnumWindows
    ENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    EnumWindows(ENUMPROC(enum_cb), 0)
    # 优先选标题正好是“微信”的主窗口
    for hwnd, title, name, _ in found:
        if title.strip() == "微信":
            return hwnd, title, name
    return (found[0][0], found[0][1], found[0][2]) if found else (None, None, None)


def load_image_from_capture(result):
    if isinstance(result, str):
        if not os.path.exists(result):
            return None
        return cv2.imread(result)
    if isinstance(result, CaptureResult):
        if getattr(result, "success", False) and result.image is not None:
            return result.image
    return None


def main():
    t_start = time.time()

    # 1) 定位窗口
    t0 = time.time()
    hwnd, title, name = find_wechat_hwnd()
    t1 = time.time()
    if hwnd is None:
        print("❌ 未找到微信主窗口（请确认微信已启动）")
        return
    print(f"[1] 定位窗口            : {_fmt_dt(t1 - t0)}  hwnd={hwnd} title={title!r} exe={name}")

    # 2) 把微信恢复到桌面可见区（当前微信在托盘，不恢复抓不到真实画面）
    wm = WeChatWindowManager()
    wm._hwnd_wechat = hwnd
    t2 = time.time()
    pos = wm.show_window(hwnd)
    t3 = time.time()
    print(f"[2] 恢复微信到桌面      : {_fmt_dt(t3 - t2)}  pos={pos}")
    # 给窗口一点绘制时间，避免 PrintWindow 拿到旧帧
    time.sleep(0.5)

    # 3) 截图
    sc = ScreenCapture(window_manager=wm, config={"data_dir": os.path.join(PROJECT_ROOT, "data")})
    t4 = time.time()
    result = sc.capture_window(None, hwnd=hwnd, subdir="wechat", prefix="ocr_now")
    t5 = time.time()
    img = load_image_from_capture(result)
    if img is None:
        print(f"❌ 截图失败: {result!r}")
        return
    h, w = img.shape[:2]
    print(f"[3] 截图                : {_fmt_dt(t5 - t4)}  size={w}x{h}")
    # 保存一份便于人工查看
    out_path = os.path.join(PROJECT_ROOT, "data", "wechat", "ocr_now_latest.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

    # 4) 本地 OCR + 布局解析
    reader = WechatScreenReader()
    t6 = time.time()
    analysis = reader.analyze(img)
    t7 = time.time()
    print(f"[4] 本地 OCR + 解析     : {_fmt_dt(t7 - t6)}")

    # 5) 汇总
    print("-" * 60)
    print(f"视图类型    : {analysis.view} (conf={analysis.view_confidence:.2f})")
    print(f"当前联系人  : {analysis.current_contact!r}")
    print(f"草稿文本    : {analysis.draft_text!r}")
    print(f"消息条数    : {len(analysis.messages)}")
    print(f"感知置信度  : {analysis.perception_confidence:.2f}")
    if analysis.unread_count is not None:
        print(f"未读数      : {analysis.unread_count} (source={analysis.unread_count_source})")
    if analysis.intent:
        print(f"意图        : {analysis.intent}")
    if analysis.low_conf_flags:
        print(f"低置信标记  : {analysis.low_conf_flags}")
    print("-" * 60)
    print("识别文本行（前 30 行）:")
    for i, ln in enumerate(analysis.raw_text_lines[:30]):
        print(f"  {i + 1:2d}. {ln}")
    if len(analysis.raw_text_lines) > 30:
        print(f"  ... 共 {len(analysis.raw_text_lines)} 行")

    print("-" * 60)
    print(f"截图已保存  : {out_path}")
    print(f"总耗时      : {_fmt_dt(time.time() - t_start)}")
    print(f"纯 OCR 耗时 : {_fmt_dt(t7 - t6)}")


if __name__ == "__main__":
    main()
