"""Screen coordinate helpers for primary, extended and virtual displays.

Aligned with original app.desktop.screen_bounds - provides virtual screen
bounds, point containment checks, and rect clamping.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
from typing import NamedTuple


class ScreenBounds(NamedTuple):
    """Virtual screen bounding rectangle."""
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def virtual_screen_bounds() -> ScreenBounds | None:
    """Get the virtual screen bounds (union of all monitors)."""
    try:
        sm_cxvirtualscreen = 78
        sm_cyvirtualscreen = 79
        sm_xvirtualscreen = 76
        sm_yvirtualscreen = 77
        user32 = ctypes.windll.user32
        x = user32.GetSystemMetrics(sm_xvirtualscreen)
        y = user32.GetSystemMetrics(sm_yvirtualscreen)
        w = user32.GetSystemMetrics(sm_cxvirtualscreen)
        h = user32.GetSystemMetrics(sm_cyvirtualscreen)
        if w <= 0 or h <= 0:
            return None
        return ScreenBounds(
            left=x, top=y, right=x + w, bottom=y + h)
    except Exception:
        return None


def is_point_on_virtual_screen(x: int, y: int, margin: int = 30) -> bool:
    """Check if a point is within the virtual screen bounds."""
    bounds = virtual_screen_bounds()
    if bounds is None:
        return False
    return (
        bounds.left + margin <= x <= bounds.right - margin
        and bounds.top + margin <= y <= bounds.bottom - margin
    )


def rect_intersects_virtual_screen(
    rect: dict, margin: int = 0
) -> bool:
    """Check if a rectangle intersects the virtual screen."""
    bounds = virtual_screen_bounds()
    if bounds is None:
        return False
    rx = rect.get("x", 0)
    ry = rect.get("y", 0)
    rw = rect.get("width", 0)
    rh = rect.get("height", 0)
    return not (
        rx + rw < bounds.left + margin
        or rx > bounds.right - margin
        or ry + rh < bounds.top + margin
        or ry > bounds.bottom - margin
    )


def clamp_rect_origin_to_virtual_screen(
    x: int, y: int, width: int, height: int
) -> tuple[int, int]:
    """Clamp a rectangle origin to stay within the virtual screen."""
    bounds = virtual_screen_bounds()
    if bounds is None:
        return x, y
    clamped_x = max(bounds.left, min(x, bounds.right - width))
    clamped_y = max(bounds.top, min(y, bounds.bottom - height))
    return clamped_x, clamped_y