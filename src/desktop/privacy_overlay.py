"""PrivacyOverlay - screen privacy protection during automation.

Aligned with original app.desktop.privacy_overlay - manages a semi-transparent
overlay window to hide WeChat automation from physical observation.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
from typing import Any


class PrivacyOverlay:
    """Screen privacy overlay for WeChat automation."""

    def __init__(self, config=None):
        self._config = config or {}
        self._overlay_hwnd = None
        self._enabled = self._config.get("privacy_overlay", False)
        self._opacity = self._config.get("privacy_opacity", 0.85)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def show(self, x=0, y=0, width=1920, height=1080, opacity=None):
        """Show the privacy overlay."""
        if opacity is None:
            opacity = self._opacity

        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            WS_EX_LAYERED = 0x00080000
            WS_EX_TOPMOST = 0x00000008
            WS_EX_TRANSPARENT = 0x00000020
            WS_POPUP = 0x80000000

            hinst = ctypes.windll.kernel32.GetModuleHandleW(None)

            wnd_class = ctypes.wintypes.WNDCLASSW()
            wnd_class.lpfnWndProc = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
                ctypes.c_void_p, ctypes.c_void_p
            )(self._wnd_proc)
            wnd_class.hInstance = hinst
            wnd_class.lpszClassName = "PrivacyOverlayClass"
            user32.RegisterClassW(ctypes.byref(wnd_class))

            ex_style = WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TRANSPARENT
            hwnd = user32.CreateWindowExW(
                ex_style, "PrivacyOverlayClass", "PrivacyOverlay",
                WS_POPUP, x, y, width, height,
                None, None, hinst, None)

            if hwnd:
                alpha = int(255 * opacity)
                user32.SetLayeredWindowAttributes(hwnd, 0, alpha, 2)
                user32.ShowWindow(hwnd, 5)
                self._overlay_hwnd = hwnd
                return True
        except Exception:
            pass
        return False

    def hide(self):
        """Hide the privacy overlay."""
        if self._overlay_hwnd:
            try:
                ctypes.windll.user32.DestroyWindow(self._overlay_hwnd)
            except Exception:
                pass
            self._overlay_hwnd = None

    @property
    def is_visible(self):
        return self._overlay_hwnd is not None

    def toggle(self):
        """Toggle overlay visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()

    @staticmethod
    def _wnd_proc(hwnd, msg, wparam, lparam):
        if msg == 0x0010:
            return 0
        return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)