"""HumanActivityMonitor - detect physical mouse/keyboard activity.

Aligned with original app.rpa.human_activity_monitor - monitors for
physical user input during automation to pause/resume operations.
"""
from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple


_MOUSE_BUTTONS = {
    1: "left",
    2: "right",
    4: "middle",
    5: "x1",
    6: "x2",
}


def pressed_mouse_button() -> str | None:
    """Check which mouse button is currently pressed."""
    for vk, name in _MOUSE_BUTTONS.items():
        if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
            return name
    return None


@dataclass
class HumanActivityResult:
    active: bool = False
    mouse_moved: bool = False
    mouse_clicked: bool = False
    key_pressed: bool = False
    button: Optional[str] = None
    last_position: Optional[Tuple[int, int]] = None


class HumanActivityMonitor:
    """Monitor for physical human activity during automation."""

    POLL_HZ = 10

    def __init__(self, window_manager=None, privacy_overlay=None,
                 config=None, logger=None):
        self._config = config or {}
        self._window_manager = window_manager
        self._privacy_overlay = privacy_overlay
        self._logger = logger
        self._last_x = 0
        self._last_y = 0
        self._last_check = time.time()
        self._active = False
        self._last_input_time = time.time()
        self._ignore_until = 0.0
        self._snapshot = HumanActivityResult()

    def check(self) -> HumanActivityResult:
        """Check for human activity since last poll."""
        now = time.time()

        class POINT(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        x, y = int(pt.x), int(pt.y)

        moved = (x != self._last_x or y != self._last_y)
        clicked = pressed_mouse_button() is not None
        key_pressed = self._any_key_pressed()

        result = HumanActivityResult(
            active=moved or clicked or key_pressed,
            mouse_moved=moved,
            mouse_clicked=clicked,
            key_pressed=key_pressed,
            button=pressed_mouse_button(),
            last_position=(x, y),
        )

        self._last_x = x
        self._last_y = y
        self._last_check = now
        self._active = result.active

        if moved or clicked or key_pressed:
            self._last_input_time = now

        return result

    @property
    def is_active(self) -> bool:
        return self._active

    def detected(self) -> bool:
        """Check and return whether human activity is detected."""
        self.check()
        return self._active

    def wait_until_idle(self, timeout: float = 5.0, idle_seconds: float = 1.0):
        """Wait until no human activity is detected."""
        start = time.time()
        idle_since = 0.0
        while time.time() - start < timeout:
            result = self.check()
            if not result.active:
                if idle_since == 0.0:
                    idle_since = time.time()
                elif time.time() - idle_since >= idle_seconds:
                    return True
            else:
                idle_since = 0.0
            time.sleep(1.0 / self.POLL_HZ)
        return False

    # =================================================================
    # 缺失方法补全（对齐原版）
    # =================================================================

    def snapshot(self) -> HumanActivityResult:
        """获取当前活动快照，对齐原版 snapshot。"""
        return self._snapshot

    def ignore_for(self, seconds: float) -> None:
        """忽略接下来 seconds 秒内的人类活动检测。"""
        self._ignore_until = time.time() + seconds

    def wait_for_mouse_release(self, timeout: float = 3.0) -> bool:
        """等待鼠标按键释放。"""
        start = time.time()
        while time.time() - start < timeout:
            if pressed_mouse_button() is None:
                return True
            time.sleep(0.05)
        return False

    def _mouse_pos(self) -> Tuple[int, int]:
        """获取当前鼠标位置。"""
        class POINT(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))

    def _seconds_since_last_input(self) -> float:
        """距离上次人类输入的时间（秒）。"""
        return time.time() - self._last_input_time

    @staticmethod
    def _any_key_pressed() -> bool:
        """Check if any common key is pressed."""
        for vk in range(0x08, 0xFF):
            if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
                return True
        return False