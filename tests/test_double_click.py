"""双击路径单元测试（真机 bug 回归锁）。

覆盖此前「双击退化成单击」的根因：
1. double=True 时必须先激活窗口 —— 屏外挂机的微信不是前台窗口，点击非激活窗口
   的第一次点击会被系统用于激活，导致两次点击无法配对成双击。
2. 必须发出 2 组 DOWN/UP，且两次 DOWN 的间隔小于系统双击时限（默认 500ms）
   —— mouse_event 没有 DBLCLK 标志，双击只能靠间隔 + 光标不移动来合成。
3. 两次点击之间不能移动光标（超出双击位移阈值同样会被判成两次单击）。
"""
import sys
import time

sys.path.insert(0, ".")

import win32api
import win32con

from src.desktop.wechat_window_manager import WeChatWindowManager


class FakeRect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class FakeUser32:
    """只提供 foreground_click 用到的 user32 接口。"""

    def __init__(self, dblclick_ms=500):
        self._dblclick_ms = dblclick_ms

    def GetDoubleClickTime(self):
        return self._dblclick_ms

    def GetSystemMetrics(self, idx):
        return 1920 if idx == 0 else 1080


def run_case(dblclick_ms=500):
    events = []
    activated = []
    real_mouse_event = win32api.mouse_event
    real_set_cursor = win32api.SetCursorPos

    def fake_mouse_event(flags, dx=0, dy=0, data=0, extra=0):
        name = {
            win32con.MOUSEEVENTF_LEFTDOWN: "DOWN",
            win32con.MOUSEEVENTF_LEFTUP: "UP",
        }.get(flags, hex(flags))
        events.append((name, time.perf_counter()))

    def fake_set_cursor(pos):
        events.append(("MOVE", pos))

    win32api.mouse_event = fake_mouse_event
    win32api.SetCursorPos = fake_set_cursor
    try:
        wm = WeChatWindowManager.__new__(WeChatWindowManager)
        wm._user32 = FakeUser32(dblclick_ms)
        wm._log_offscreen = lambda m: None
        wm._resolve = lambda h: h
        wm.get_rect = lambda t: FakeRect(100, 100, 1600, 1200)
        wm.is_window_offscreen = lambda t: True
        wm.is_minimized = lambda t: False
        wm.topmost = lambda t, enable=True: True
        wm._hit_test_is_self = lambda t, x, y: True
        wm._remember_native_size = lambda t: (1600, 1200)
        wm._activate_offscreen_window = lambda t: activated.append(t) or True

        ok = wm.foreground_click(31, 192, hwnd=12345, double=True, repark=False)
    finally:
        win32api.mouse_event = real_mouse_event
        win32api.SetCursorPos = real_set_cursor

    seq = [e[0] for e in events]
    downs = [e[1] for e in events if e[0] == "DOWN"]
    moves_after_first_down = seq.index("DOWN") < len(seq) and \
        "MOVE" in seq[seq.index("DOWN") + 1:]

    print(f"  事件序列: {seq}")
    print(f"  激活窗口: {activated}")
    assert activated, (
        "double=True 必须先激活窗口：否则第一次点击被系统用于激活，"
        "双击会退化成单击（未读不置顶）")
    assert seq.count("DOWN") == 2, f"双击应发 2 次 LEFTDOWN，实际 {seq}"
    assert seq.count("UP") == 2, f"双击应发 2 次 LEFTUP，实际 {seq}"
    # 只检查「第一次 DOWN 到最后一次 UP」之间是否插入了光标移动；
    # 末尾那次 MOVE 是 finally 里还原光标位置，发生在两次点击之后，不算。
    first_down = seq.index("DOWN")
    last_up = max(i for i, s in enumerate(seq) if s == "UP")
    assert "MOVE" not in seq[first_down:last_up], "两次点击之间不能移动光标"
    gap_ms = (downs[1] - downs[0]) * 1000
    print(f"  两次 DOWN 间隔: {gap_ms:.1f}ms / 系统双击时限 {dblclick_ms}ms")
    assert gap_ms < dblclick_ms, (
        f"两次点击间隔 {gap_ms:.1f}ms 必须小于系统双击时限 {dblclick_ms}ms")
    assert ok, "foreground_click 应返回 True"
    return gap_ms


def main():
    print("用例1: 系统双击时限 500ms（默认）")
    run_case(500)
    print("PASSED: 双击先激活 + 两次点击间隔在时限内\n")

    print("用例2: 系统双击时限被改小到 200ms（间隔需自适应）")
    gap = run_case(200)
    assert gap < 200, f"间隔 {gap:.1f}ms 未随双击时限自适应"
    print("PASSED: 间隔随系统双击时限自适应\n")

    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
