# -*- coding: utf-8 -*-
"""边缘缩放真机自检：不依赖人手，直接验证注入层与 Python 端 resize 链路。

用法: venv/Scripts/python.exe tools/_selftest_resize.py
流程: 启动 GUI -> 等页面加载 -> 探针检查 __edgeAt / api.resizeWindow ->
      模拟右缘拖拽(+200px) -> 对比窗口宽度 -> destroy。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import webview  # noqa: E402
from src.config import Settings  # noqa: E402
from src.ui.webview_window import create_app, HTML_FILE  # noqa: E402

app = create_app(Settings())


def probe():
    win = webview.windows[0]

    import ctypes
    import ctypes.wintypes as _wt

    _gwr = ctypes.windll.user32.GetWindowRect
    _gwr.argtypes = [_wt.HWND, _wt.LPRECT]
    _gwr.restype = _wt.BOOL

    def real_size():
        """GetWindowRect 实测物理像素（pywebview 的 win.width 是静态初始值）。"""
        try:
            r = _wt.RECT()
            _gwr(int(win.native.Handle.ToInt64()), ctypes.byref(r))
            return r.right - r.left, r.bottom - r.top
        except Exception:
            return -1, -1

    try:
        win.evaluate_js("true")
        time.sleep(3.0)  # 等页面加载 + INJECT_JS 注入 + pywebview api 就绪

        r1 = win.evaluate_js("typeof window.__edgeAt")
        r2 = win.evaluate_js(
            "(function(){ try { return typeof window.pywebview.api.resizeWindow; }"
            "(catch(e){}) })()" if False else
            "(window.pywebview && window.pywebview.api) ? "
            "typeof window.pywebview.api.resizeWindow : 'api-missing'")
        print(f"[PROBE] __edgeAt = {r1!r}")
        print(f"[PROBE] api.resizeWindow = {r2!r}")

        w0, h0 = real_size()
        print(f"[PROBE] before: {w0}x{h0}")

        # 模拟右缘拖拽：一次性 +200px（等价 40ms 合并后的净位移）
        win.evaluate_js(
            "window.pywebview.api.resizeWindow("
            "JSON.stringify({edge:'e', dx:200, dy:0, dpr: window.devicePixelRatio||1})); 'sent'")
        time.sleep(1.2)
        w1, _ = real_size()
        print(f"[PROBE] after edge=e dx=200: width {w0} -> {w1} "
              f"({'OK' if w1 >= w0 + 150 else 'FAIL'})")

        # 模拟底缘
        win.evaluate_js(
            "window.pywebview.api.resizeWindow("
            "JSON.stringify({edge:'s', dx:0, dy:120, dpr: window.devicePixelRatio||1})); 'sent'")
        time.sleep(1.2)
        _, h1 = real_size()
        print(f"[PROBE] after edge=s dy=120: height {h0} -> {h1} "
              f"({'OK' if h1 >= h0 + 80 else 'FAIL'})")
    except Exception as e:  # noqa: BLE001
        print(f"[PROBE] exception: {e}")
    finally:
        try:
            win.destroy()
        except Exception:
            pass


window = webview.create_window(
    "resize-selftest",
    url=HTML_FILE.as_uri(),
    js_api=app.api,
    width=1240, height=820,
    frameless=True,
    easy_drag=False,          # 与修复一致：关掉整窗原生拖拽
    background_color="#F5F5F7",
)
app.bind(window)
webview.start(func=probe, gui="edgechromium")
