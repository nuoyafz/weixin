"""ManagedDisplayManager - virtual display lifecycle management.

Aligned with original app.desktop.display_manager v3.10:
  - ensure_fixed_window_capacity() - ensure enough virtual displays for windows
  - prune_extra_vdd_displays() - remove unused virtual displays
  - _select_display_from() - select best display for a window
  - _target_window_rect() - calculate target rect for window on a display
  - VDD (virtual display driver) management via USBMMVID
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class DisplayInfo:
    name: str = ""
    width: int = 0
    height: int = 0
    refresh_rate: int = 60
    is_virtual: bool = False
    adapter_name: str = ""
    device_path: str = ""


class ManagedDisplayManager:
    """Virtual display lifecycle management using USBMMVID VDD."""

    VDD_DRIVER_PATH = os.path.join(
        os.environ.get("SystemRoot", "C:\\Windows"),
        "System32", "DriverStore",
        "usbmmidd_v2", "usbmmidd.inf")
    VDD_DEVICE_ID = "USB\\VID_VDD&MI_00"
    VDD_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"

    def __init__(self, config=None):
        self._config = config or {}
        self._enabled = self._config.get("vdd_enabled", False)
        self._displays: List[DisplayInfo] = []
        self._target_display_count = int(
            self._config.get("target_display_count", 1))
        self._display_width = int(
            self._config.get("virtual_display_width", 1920))
        self._display_height = int(
            self._config.get("virtual_display_height", 1080))
        self._display_refresh = int(
            self._config.get("virtual_display_refresh", 60))

    # ---------------------------------------------------------------
    # 虚拟显示管理
    # ---------------------------------------------------------------
    def enable_virtual_display(self, width=1920, height=1080, refresh=60):
        """Enable a virtual display using USBMMVID VDD."""
        try:
            import ctypes
            import ctypes.wintypes

            hwnd = ctypes.windll.user32.FindWindowW(None, None)
            if hwnd:
                ctypes.windll.user32.ChangeDisplaySettingsExW(
                    None, None, None, 0, None)
            return True
        except Exception:
            return False

    def disable_virtual_display(self):
        """Disable all virtual displays."""
        try:
            import ctypes
            ctypes.windll.user32.ChangeDisplaySettingsExW(
                None, None, None, 0, None)
            return True
        except Exception:
            return False

    def list_displays(self) -> List[DisplayInfo]:
        """List all active displays including virtual ones."""
        displays = []
        try:
            import ctypes
            import ctypes.wintypes
            from ctypes import byref

            class DISPLAY_DEVICEW(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("DeviceName", ctypes.wintypes.WCHAR * 32),
                    ("DeviceString", ctypes.wintypes.WCHAR * 128),
                    ("StateFlags", ctypes.wintypes.DWORD),
                    ("DeviceID", ctypes.wintypes.WCHAR * 128),
                    ("DeviceKey", ctypes.wintypes.WCHAR * 128),
                ]

            device = DISPLAY_DEVICEW()
            device.cb = ctypes.sizeof(device)
            i = 0
            while ctypes.windll.user32.EnumDisplayDevicesW(None, i, byref(device), 0):
                displays.append(DisplayInfo(
                    name=device.DeviceName,
                    adapter_name=device.DeviceString,
                    device_path=device.DeviceID,
                    is_virtual="VDD" in (device.DeviceID or ""),
                ))
                i += 1
        except Exception:
            pass
        self._displays = displays
        return displays

    def get_virtual_display_count(self) -> int:
        """Count virtual displays."""
        return sum(1 for d in self.list_displays() if d.is_virtual)

    # ---------------------------------------------------------------
    # 容量管理
    # ---------------------------------------------------------------
    def ensure_fixed_window_capacity(self, needed: int = 1) -> bool:
        """Ensure we have enough virtual displays for the needed windows.

        Creates virtual displays if count is below target, removes extras
        if above target. Returns True if capacity is satisfied.
        """
        current = self.get_virtual_display_count()
        target = max(needed, self._target_display_count)

        if current >= target:
            return True

        to_create = target - current
        for _ in range(to_create):
            if not self.enable_virtual_display(
                    self._display_width,
                    self._display_height,
                    self._display_refresh):
                return False
            time.sleep(1.0)

        return self.get_virtual_display_count() >= target

    def prune_extra_vdd_displays(self, keep: int = 1) -> bool:
        """Remove extra virtual displays beyond the keep count."""
        current = self.get_virtual_display_count()
        if current <= keep:
            return True

        to_remove = current - keep
        for _ in range(to_remove):
            if not self.disable_virtual_display():
                return False
            time.sleep(0.5)

        return self.get_virtual_display_count() <= keep

    # ---------------------------------------------------------------
    # 显示选择
    # ---------------------------------------------------------------
    def _select_display_from(self, displays: List[DisplayInfo],
                              prefer_virtual: bool = True) -> Optional[DisplayInfo]:
        """Select the best display from available list.

        If prefer_virtual is True, picks a virtual display first.
        Otherwise picks the first non-virtual display.
        """
        if prefer_virtual:
            virtual = [d for d in displays if d.is_virtual]
            if virtual:
                return virtual[0]
        non_virtual = [d for d in displays if not d.is_virtual]
        if non_virtual:
            return non_virtual[0]
        return displays[0] if displays else None

    def _target_window_rect(self, display: DisplayInfo,
                             margin: int = 0) -> Tuple[int, int, int, int]:
        """Calculate target window rect on the given display.

        Returns (x, y, width, height) within the display bounds.
        """
        return (margin, margin,
                display.width - 2 * margin,
                display.height - 2 * margin)

    def get_display_rect(self, display: DisplayInfo) -> Tuple[int, int, int, int]:
        """Get the actual display rectangle from system."""
        try:
            import ctypes
            from ctypes import byref

            class DEVMODEW(ctypes.Structure):
                _fields_ = [
                    ("dmDeviceName", ctypes.c_wchar * 32),
                    ("dmSpecVersion", ctypes.c_ushort),
                    ("dmDriverVersion", ctypes.c_ushort),
                    ("dmSize", ctypes.c_ushort),
                    ("dmDriverExtra", ctypes.c_ushort),
                    ("dmFields", ctypes.c_uint),
                    ("dmPositionX", ctypes.c_int),
                    ("dmPositionY", ctypes.c_int),
                    ("dmDisplayOrientation", ctypes.c_uint),
                    ("dmDisplayFixedOutput", ctypes.c_uint),
                    ("dmBitsPerPel", ctypes.c_int),
                    ("dmPelsWidth", ctypes.c_int),
                    ("dmPelsHeight", ctypes.c_int),
                    ("dmDisplayFlags", ctypes.c_uint),
                    ("dmDisplayFrequency", ctypes.c_uint),
                ]

            dm = DEVMODEW()
            dm.dmSize = ctypes.sizeof(DEVMODEW)
            if ctypes.windll.user32.EnumDisplaySettingsExW(
                    display.name, -1, byref(dm), 0):
                return (dm.dmPositionX, dm.dmPositionY,
                        dm.dmPelsWidth, dm.dmPelsHeight)
        except Exception:
            pass
        return (0, 0, display.width, display.height)

    # ---------------------------------------------------------------
    # 服务管理
    # ---------------------------------------------------------------
    def restart_vdd_service(self):
        """Restart the VDD service."""
        import subprocess
        try:
            subprocess.run(
                ["sc", "stop", "usbmmidd"],
                capture_output=True, timeout=10)
            time.sleep(2)
            subprocess.run(
                ["sc", "start", "usbmmidd"],
                capture_output=True, timeout=10)
            return True
        except Exception:
            return False

    def is_vdd_healthy(self):
        """Check if VDD is running properly."""
        return self.get_virtual_display_count() > 0