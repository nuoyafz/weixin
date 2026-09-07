"""DisplayScale - DPI scaling detection and reporting.

Aligned with original app.desktop.display_scale - detects Windows
display scaling per monitor and validates against expected values.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DisplayScaleInfo:
    monitor_name: str = ""
    scale_percent: int = 0
    dpi: int = 0
    is_primary: bool = False
    width: int = 0
    height: int = 0

    @property
    def expected_percent(self) -> int:
        return 100

    @property
    def tolerance_percent(self) -> int:
        return 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def display_scale_report() -> dict[str, Any]:
    """Generate a full display scale report."""
    monitors = list_display_scales()
    matching = all(
        abs(s.scale_percent - s.expected_percent) <= s.tolerance_percent
        for s in monitors)
    return {
        "monitors": [asdict(m) for m in monitors],
        "all_match": matching,
        "count": len(monitors),
    }


def list_display_scales() -> list[DisplayScaleInfo]:
    """List all monitors with their scale factors."""
    result = []
    try:
        user32 = ctypes.windll.user32
        shcore = ctypes.windll.shcore

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.wintypes.DWORD),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("szDevice", ctypes.wintypes.WCHAR * 32),
            ]

        monitors = []

        def _monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
            mi = MONITORINFOEXW()
            mi.cbSize = ctypes.sizeof(mi)
            user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
            x = mi.rcMonitor.left
            y = mi.rcMonitor.top
            dpi_x = ctypes.c_uint(0)
            dpi_y = ctypes.c_uint(0)
            try:
                shcore.GetDpiForMonitor(
                    hMonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
            except Exception:
                dpi_x.value = 96
                dpi_y.value = 96
            w = mi.rcMonitor.right - mi.rcMonitor.left
            h = mi.rcMonitor.bottom - mi.rcMonitor.top
            is_primary = bool(mi.dwFlags & 0x00000001)
            monitors.append(DisplayScaleInfo(
                monitor_name=mi.szDevice,
                scale_percent=scale_percent_from_dpi(dpi_x.value),
                dpi=dpi_x.value,
                is_primary=is_primary,
                width=w,
                height=h,
            ))
            return True

        MonitorEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(RECT), ctypes.c_void_p)
        enum_proc = MonitorEnumProc(_monitor_enum_proc)
        user32.EnumDisplayMonitors(None, None, enum_proc, 0)
        result = monitors
    except Exception:
        pass
    return result


def display_scale_summary() -> dict[str, Any] | None:
    """Return a human-readable scale summary."""
    report = display_scale_report()
    if not report or not report.get("monitors"):
        return None
    return {
        "all_match": report["all_match"],
        "scales": [m["scale_percent"] for m in report["monitors"]],
    }


def scale_percent_from_dpi(dpi: int | float) -> int:
    """Convert DPI value to scale percentage."""
    return int(round(float(dpi) / 96.0 * 100))


def _scale_matches(dpi: int, expected: int = 100) -> bool:
    """Check if DPI matches expected scale."""
    actual = scale_percent_from_dpi(dpi)
    return abs(actual - expected) <= 5


def _get_monitor_dpi(monitor_handle) -> tuple[int, int] | None:
    """Get DPI for a specific monitor handle."""
    try:
        shcore = ctypes.windll.shcore
        dpi_x = ctypes.c_uint(0)
        dpi_y = ctypes.c_uint(0)
        shcore.GetDpiForMonitor(
            monitor_handle, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        return dpi_x.value, dpi_y.value
    except Exception:
        return None


def _fallback_monitor_dpi() -> tuple[int, int]:
    """Fallback DPI using system DPI."""
    sys_dpi = _get_system_dpi()
    return sys_dpi, sys_dpi


def _get_system_dpi() -> int | None:
    """Get system DPI from HDC."""
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        if hdc:
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return dpi
    except Exception:
        pass
    return 96