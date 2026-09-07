"""统一后台点击工具（全局唯一真相源）。

技术：向目标窗口句柄 PostMessage 发送
    WM_MOUSEMOVE -> WM_LBUTTONDOWN(MK_LBUTTON) -> WM_LBUTTONUP
（双击则重复一次，间隔随机）
前台 / 后台 / 虚拟屏均可点，不抢焦点、不移动真实鼠标。
带随机时间间隔模拟真实操作。

全项目的点击都应走这里，确保：
  - 绝不移动 / 抢真实鼠标
  - 前台 / 后台 / 虚拟显示器下都能点
  - 默认不回退到任何真实鼠标方案（如 pyautogui / SendInput）

用法：
    from .background_clicker import (
        background_click, background_click_screen, background_right_click_screen,
    )

    # 已知 HWND + 客户区坐标
    background_click(hwnd, cx, cy, double=False)

    # 只有屏幕坐标（自动 WindowFromPoint 解析 HWND + ScreenToClient 换算）
    background_click_screen(x, y, double=False)
    background_right_click_screen(x, y)   # 右键，用于上下文菜单
"""
import time
import random
from typing import Optional

import win32api
import win32con
import win32gui


def background_click(hwnd: int, client_x: int, client_y: int,
                     double: bool = False,
                     method: str = "auto") -> bool:
    """向窗口【客户区坐标】发送后台左键点击（不抢焦点、不移动真实鼠标）。

    顺序：① 直接发给顶层窗口(SendMessage) → ② 落空时枚举子窗口，把点击派发给
    真正包含该坐标的子控件(CEF 渲染窗/WebView)，对微信这类自绘窗口更可靠 →
    ③ 最终回退 PostMessage。
    method: "auto"(默认) / "send" / "post" / "child"。
    """
    if not hwnd:
        return False
    try:
        rect = win32gui.GetWindowRect(hwnd)
        origin_x, origin_y = rect[0], rect[1]
        screen_x, screen_y = origin_x + int(client_x), origin_y + int(client_y)

        # ① 直接发给顶层窗口
        if method in ("auto", "send"):
            if _post_click(hwnd, client_x, client_y, double, send=True):
                return True

        # ② 发给包含该点的子窗口（自绘/CEF 窗口的关键命中路径）
        child = _child_at_point(screen_x, screen_y)
        if child and child != hwnd:
            if _post_click(child, client_x, client_y, double, send=True):
                return True

        # ③ 兜底：顶层 PostMessage
        return _post_click(hwnd, client_x, client_y, double, send=False)
    except Exception:
        return False


def _post_click(target: int, client_x: int, client_y: int,
               double: bool, send: bool) -> bool:
    """对单个 HWND 投放一次（或双击）左键消息。send=True 用 SendMessage。"""
    try:
        lparam = win32api.MAKELONG(int(client_x), int(client_y))
        clicks = 2 if double else 1
        fn = win32gui.SendMessage if send else win32gui.PostMessage
        for i in range(clicks):
            fn(target, win32con.WM_MOUSEMOVE, 0, lparam)
            time.sleep(random.uniform(0.02, 0.05))
            fn(target, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
            time.sleep(random.uniform(0.05, 0.09))
            fn(target, win32con.WM_LBUTTONUP, 0, lparam)
            if i < clicks - 1:
                time.sleep(random.uniform(0.25, 0.45))
        return True
    except Exception:
        return False


def _child_at_point(screen_x: int, screen_y: int) -> Optional[int]:
    """返回屏幕坐标处的最深子窗口 HWND（用于把点击派发给 CEF 渲染控件）。"""
    try:
        return int(win32gui.WindowFromPoint((int(screen_x), int(screen_y))))
    except Exception:
        return None


def background_click_screen(x: int, y: int, double: bool = False) -> bool:
    """向【屏幕坐标】发送后台左键点击。

    通过 WindowFromPoint 找到该坐标下的窗口，ScreenToClient 换算为客户区
    坐标，再 PostMessage。用于只有屏幕坐标、没有 HWND 的场景
    （引用菜单项、好友申请按钮、语音转文字菜单等）。
    """
    try:
        hwnd = win32gui.WindowFromPoint((int(x), int(y)))
        if not hwnd:
            return False
        client = win32gui.ScreenToClient(hwnd, (int(x), int(y)))
        return background_click(hwnd, client[0], client[1], double=double)
    except Exception:
        return False


def background_right_click_screen(x: int, y: int) -> bool:
    """向【屏幕坐标】发送后台右键点击（用于触发上下文菜单）。"""
    try:
        hwnd = win32gui.WindowFromPoint((int(x), int(y)))
        if not hwnd:
            return False
        client = win32gui.ScreenToClient(hwnd, (int(x), int(y)))
        lparam = win32api.MAKELONG(int(client[0]), int(client[1]))
        win32gui.PostMessage(
            hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
        time.sleep(random.uniform(0.05, 0.09))
        win32gui.PostMessage(hwnd, win32con.WM_RBUTTONUP, 0, lparam)
        return True
    except Exception:
        return False
