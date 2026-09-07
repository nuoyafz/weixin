#!/usr/bin/env python3
"""调试：看每条 OCR 行的坐标 / 置信度 / 是否被 _extract_contact 与聊天区过滤命中。

复用 capture_ocr_now 的窗口定位 + 截图，把 anchors 与逐行细节打印出来。
"""
from __future__ import annotations

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import cv2
import ctypes
import ctypes.wintypes

from src.desktop.wechat_window_manager import WeChatWindowManager
from src.capture.screen_capture import ScreenCapture
from src.clean_perception.reader import WechatScreenReader, _normalize_ocr


def find_wechat_hwnd():
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
            found.append((hwnd, title, name))
        return True
    ENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows(ENUMPROC(enum_cb), 0)
    for hwnd, title, name in found:
        if title.strip() == "微信":
            return hwnd
    return found[0][0] if found else None


def main():
    hwnd = find_wechat_hwnd()
    if hwnd is None:
        print("未找到微信")
        return
    wm = WeChatWindowManager()
    wm._hwnd_wechat = hwnd
    wm.show_window(hwnd)
    time.sleep(0.5)

    sc = ScreenCapture(window_manager=wm, config={"data_dir": os.path.join(PROJECT_ROOT, "data")})
    result = sc.capture_window(None, hwnd=hwnd, subdir="wechat", prefix="dbg")
    if isinstance(result, str):
        img = cv2.imread(result)
    elif getattr(result, "image", None) is not None:
        img = result.image
    else:
        print("截图失败", result)
        return
    h, w = img.shape[:2]

    reader = WechatScreenReader()
    arr = reader._to_array(img)
    ocr = reader._get_ocr()
    enhanced = reader._enhance(arr)
    raw = ocr.run(enhanced) if hasattr(ocr, "run") else None
    lines = _normalize_ocr(raw)
    from src.clean_perception.reader import ChatAnalysis
    analysis = ChatAnalysis(image_size=(w, h))
    lines = reader._quality_filter(lines, analysis)

    # 拿到 anchors
    anchors = reader.layout.estimate(arr, lines)

    print(f"窗口尺寸     : {w}x{h}")
    print(f"anchors      : chat_left={anchors.chat_left} chat_right={anchors.chat_right} "
          f"header_bottom={anchors.header_bottom} message_top={anchors.message_top} "
          f"message_bottom={anchors.message_bottom} input_top={anchors.input_top} source={anchors.source}")
    print(f"top_bar_y    : {h*0.045:.1f}")
    print(f"far_right_x  : {w*0.90:.1f}")
    print("=" * 100)
    print(f"{'#':>2} {'x_min':>5} {'x_max':>5} {'y_min':>5} {'y_max':>5} {'cy':>5} {'score':>5} {'low':>3}  text")
    print("-" * 100)
    for i, ln in enumerate(lines):
        # 复算 _extract_contact 的过滤条件
        in_header = ln.cy <= anchors.header_bottom
        below_topbar = ln.y_min >= h * 0.045
        right_of_chat = ln.x_min >= anchors.chat_left
        left_of_farright = ln.x_max <= w * 0.90
        is_contact_candidate = in_header and below_topbar and right_of_chat and left_of_farright and not ln.low_conf and ln.text
        # 聊天区消息条件
        in_chat = (ln.x_min >= anchors.chat_left and ln.y_min >= anchors.message_top
                   and ln.y_max <= anchors.message_bottom)
        flag = "C" if is_contact_candidate else (" M" if in_chat else "  ")
        print(f"{i:>2} {ln.x_min:>5.0f} {ln.x_max:>5.0f} {ln.y_min:>5.0f} {ln.y_max:>5.0f} {ln.cy:>5.0f} {ln.score:>5.2f} {str(ln.low_conf):>3}  {flag} {ln.text}")

    print("=" * 100)
    contact = reader._extract_contact(lines, anchors, h, w)
    print(f"→ _extract_contact 结果: {contact!r}")


if __name__ == "__main__":
    main()
