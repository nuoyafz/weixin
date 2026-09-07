"""抗干扰点击 + 虚拟显示器隔离 的回归测试。

覆盖：
  - 轻量方案：foreground_click 改用 SendInput 绝对坐标（落点不依赖光标当前位置）
              + 空闲门控（用户活跃时等待）。
  - 重度方案：虚拟显示器真实启用（驱动缺失时安全返回）、停到虚拟屏、降级兜底。
"""
import sys
import types

import pytest

ROOT = "C:/VisionLeadAgent-wx/my_agent"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.desktop.wechat_window_manager import WeChatWindowManager
from src.desktop.display_manager import ManagedDisplayManager


# ---------------------------------------------------------------------------
# 轻量方案：SendInput 绝对坐标
# ---------------------------------------------------------------------------
class _FakeUser32:
    """只实现点击测试用到的两个 API。"""
    def __init__(self, vx=0, vy=0, vw=1920, vh=1080):
        self.vx, self.vy, self.vw, self.vh = vx, vy, vw, vh
        self.calls = []          # (flags, dx, dy)
        self.setwindowpos = 0
        self.movewindow = 0

    def GetSystemMetrics(self, idx):
        m = {0: self.vw, 1: self.vh,
             76: self.vx, 77: self.vy, 78: self.vw, 79: self.vh}
        return m.get(idx, 0)

    def SendInput(self, n, arr, size):
        mi = arr.contents.u.mi
        self.calls.append((mi.dwFlags, mi.dx, mi.dy))
        return n

    def SetWindowPos(self, *a, **k):
        self.setwindowpos += 1
        return 1

    def MoveWindow(self, *a, **k):
        self.movewindow += 1
        return 1


ABS = 0x8000
VDESK = 0x4000
LDOWN = 0x0002
LUP = 0x0004


def _make_wm():
    wm = WeChatWindowManager()
    wm._user32 = _FakeUser32()
    wm._resolve = lambda h: h
    wm._remember_restore_position = lambda t: None
    wm._remember_native_size = lambda t: (900, 680)
    wm.get_rect = lambda t: types.SimpleNamespace(x=0, y=0, width=900, height=680)
    wm.activity_monitor = None
    wm.display_manager = None
    return wm


def test_send_input_absolute_single_click_normalizes():
    wm = _make_wm()
    ok = wm._send_input_click_absolute(100, 200, double=False, gap=0.05)
    assert ok is True
    # 2 次调用：DOWN / UP
    assert len(wm._user32.calls) == 2
    down, up = wm._user32.calls
    # 绝对坐标标志（ABS | VIRTUALDESKTOP | LEFTxxxx）
    assert down[0] == (ABS | VDESK | LDOWN)
    assert up[0] == (ABS | VDESK | LUP)
    # 坐标已归一化到虚拟屏 (0,0)-(1920,1080)
    nx = int((100 - 0) * 65535 / 1920)
    ny = int((200 - 0) * 65535 / 1080)
    assert down[1] == nx and down[2] == ny
    # DOWN 与 UP 落点一致（双击配对前提）
    assert down[1] == up[1] and down[2] == up[2]


def test_send_input_absolute_double_click_fires_four_events():
    wm = _make_wm()
    ok = wm._send_input_click_absolute(100, 200, double=True, gap=0.05)
    assert ok is True
    assert len(wm._user32.calls) == 4
    # 序列：DOWN UP DOWN UP，四次同一坐标
    coords = [(c[1], c[2]) for c in wm._user32.calls]
    assert all(c == coords[0] for c in coords)


def test_send_input_absolute_uses_virtual_screen_origin():
    # 虚拟屏原点不在 (0,0) 时也能正确归一化（多屏左移场景）
    wm = _make_wm()
    wm._user32 = _FakeUser32(vx=-1920, vy=0, vw=1920, vh=1080)
    wm._send_input_click_absolute(-1820, 100, double=False, gap=0.05)
    nx = int((-1820 - (-1920)) * 65535 / 1920)
    ny = int((100 - 0) * 65535 / 1080)
    assert wm._user32.calls[0][1] == nx
    assert wm._user32.calls[0][2] == ny


# ---------------------------------------------------------------------------
# 轻量方案：空闲门控
# ---------------------------------------------------------------------------
class _FakeMonitor:
    def __init__(self):
        self.release = False
        self.idle = False

    def wait_for_mouse_release(self, timeout=2.0):
        self.release = True

    def wait_until_idle(self, timeout=5.0, idle_seconds=1.0):
        self.idle = True


def test_idle_gate_invokes_monitor_when_active():
    wm = _make_wm()
    mon = _FakeMonitor()
    wm.activity_monitor = mon
    wm._wait_user_idle_before_click()
    assert mon.release is True
    assert mon.idle is True


def test_idle_gate_no_monitor_does_not_crash():
    wm = _make_wm()
    wm.activity_monitor = None
    # 无监视器时惰性创建并轮询真实光标，不应抛异常
    wm._wait_user_idle_before_click()


# ---------------------------------------------------------------------------
# 重度方案：虚拟显示器
# ---------------------------------------------------------------------------
def _fake_display(is_virtual):
    return types.SimpleNamespace(is_virtual=is_virtual,
                                name="DISPLAY",
                                DeviceName="\\\\.\\DISPLAY1")


class _FakeDM:
    def __init__(self, vdisplays=None, rect=(1920, 0, 1920, 1080)):
        self._v = list(vdisplays or [])
        self._rect = rect

    def list_displays(self):
        return self._v

    def get_display_rect(self, d):
        return self._rect

    def get_virtual_display_count(self):
        return len(self._v)

    def is_on_managed_display(self, rect):
        return any(getattr(d, "is_virtual", False) for d in self._v)

    def display_for_rect(self, rect):
        for d in self._v:
            if getattr(d, "is_virtual", False):
                return d
        return None


def test_park_on_virtual_display_no_manager_returns_false():
    wm = _make_wm()
    wm.display_manager = None
    assert wm.park_on_virtual_display(123) is False


def test_park_on_virtual_display_no_virtual_returns_false():
    wm = _make_wm()
    wm.display_manager = _FakeDM(vdisplays=[_fake_display(False)])
    assert wm.park_on_virtual_display(123) is False


def test_park_on_virtual_display_moves_to_virtual():
    wm = _make_wm()
    wm.display_manager = _FakeDM(vdisplays=[_fake_display(True)],
                                rect=(1920, 0, 1920, 1080))
    ok = wm.park_on_virtual_display(123)
    assert ok is True
    # 调用了 MoveWindow（虚拟屏左上角 +10 边距），落点在虚拟屏内
    assert wm._user32.movewindow == 1


def test_move_offscreen_prefers_virtual_display():
    wm = _make_wm()
    wm.display_manager = _FakeDM(vdisplays=[_fake_display(True)],
                                rect=(1920, 0, 1920, 1080))
    ok = wm.move_offscreen(123)
    assert ok is True
    # 走虚拟屏停放：MoveWindow 被调用，普通屏外 SetWindowPos 不被调用
    assert wm._user32.movewindow == 1
    assert wm._user32.setwindowpos == 0


def test_move_offscreen_falls_back_when_no_manager():
    wm = _make_wm()
    wm.display_manager = None
    wm.is_minimized = lambda t: False
    ok = wm.move_offscreen(123)
    assert ok is True
    # 无虚拟显示器时走普通屏外保活：SetWindowPos 被调用
    assert wm._user32.setwindowpos == 1


def test_enable_virtual_display_safe_when_driver_missing():
    # 驱动工具不存在时返回 False（不崩溃、不抛异常）
    mgr = ManagedDisplayManager({})
    mgr.VDD_DRIVER_PATH = "C:/__no_such_dir__/usbmmidd_v2/usbmmidd.inf"
    assert mgr.enable_virtual_display() is False
    assert mgr.disable_virtual_display() is False


def test_ensure_capacity_false_when_no_driver():
    mgr = ManagedDisplayManager({"target_display_count": 1})
    mgr.VDD_DRIVER_PATH = "C:/__no_such_dir__/usbmmidd_v2/usbmmidd.inf"
    # 强制当前虚拟屏数量为 0，逼出「驱动缺失 -> 启用失败」分支
    mgr.get_virtual_display_count = lambda: 0
    # 工具缺失 -> 容量无法满足 -> 返回 False（run.py 据此降级）
    assert mgr.ensure_fixed_window_capacity() is False


# ---------------------------------------------------------------------------
# 主路径：虚拟屏隔离下的 foreground_click（模拟点击 + 用户鼠标脱离）
# ---------------------------------------------------------------------------
def test_foreground_click_on_vdd_skips_idle_gate_and_clicks_in_place():
    # 微信停在右侧虚拟屏(1920,0)-(3840,1080)：主路径应原地模拟点击，
    # 不调空闲门控、不把窗口拉回主屏 (0,0)。
    wm = _make_wm()
    wm.display_manager = _FakeDM(vdisplays=[_fake_display(True)],
                                rect=(1920, 0, 1920, 1080))
    wm.get_rect = lambda t: types.SimpleNamespace(
        x=1920, y=0, width=900, height=680)
    wm.is_window_offscreen = lambda t: False
    wm.is_minimized = lambda t: False
    wm.topmost = lambda *a, **k: None
    wm._hit_test_is_self = lambda *a, **k: True
    wm._activate_offscreen_window = lambda *a, **k: None
    wm.move_offscreen = lambda *a, **k: True
    wm.get_virtual_screen = lambda: (0, 0, 1920, 1080)

    idle_calls = []
    wm._wait_user_idle_before_click = lambda *a, **k: idle_calls.append(1)
    abs_calls = []
    wm._send_input_click_absolute = (
        lambda x, y, d, g: (abs_calls.append((x, y, d)), True)[1])

    ok = wm.foreground_click(window_x=100, window_y=50, hwnd=123, double=False)
    assert ok is True
    # 主路径：跳过空闲门控（用户光标在另一块屏，天然隔离）
    assert idle_calls == []
    # 按虚拟屏坐标原地点击（1920+100, 0+50）
    assert abs_calls == [(2020, 50, False)]
    # 未把窗口拉回主屏 (0,0) 做重定位
    assert wm._user32.movewindow == 0


def test_foreground_click_on_vdd_never_relocates_to_main_screen():
    # 即便命中测试失败、走到 relocate 分支，虚拟屏场景下也只按原虚拟坐标原地
    # 点击，绝不 MoveWindow 到 (0,0) 主屏（否则破坏隔离、主屏闪现）。
    wm = _make_wm()
    wm.display_manager = _FakeDM(vdisplays=[_fake_display(True)],
                                rect=(1920, 0, 1920, 1080))
    wm.get_rect = lambda t: types.SimpleNamespace(
        x=1920, y=0, width=900, height=680)
    wm.is_window_offscreen = lambda t: False
    wm.is_minimized = lambda t: False
    wm.topmost = lambda *a, **k: None
    wm._hit_test_is_self = lambda *a, **k: False  # 命中测试失败也会走 relocate 分支
    wm._activate_offscreen_window = lambda *a, **k: None
    wm.move_offscreen = lambda *a, **k: True
    wm.get_virtual_screen = lambda: (0, 0, 1920, 1080)
    wm._wait_user_idle_before_click = lambda *a, **k: None
    abs_calls = []
    wm._send_input_click_absolute = (
        lambda x, y, d, g: (abs_calls.append((x, y)), True)[1])

    ok = wm.foreground_click(window_x=5000, window_y=9000, hwnd=123, double=False)
    assert ok is True
    # 没有把窗口拉回主屏 (0,0)：movewindow 不被 relocate 调用
    assert wm._user32.movewindow == 0
    # 仍按原虚拟坐标(1920+5000)原地点击
    assert abs_calls and abs_calls[0][0] == 1920 + 5000
