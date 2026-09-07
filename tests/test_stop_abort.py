"""验证「点停即停鼠标」：RedDotDetector 的 should_abort 注入与热路径插桩。

背景：stop_loop 只发「正在停止」并置 _stop_requested，但 find_and_click_unread
一轮扫描（截图→双击置顶→点头行）可达 ~19s，若不插桩，停止后还会继续点聊天。
本套用例覆盖 _aborted 判定 + 各热路径守卫，确保收到停止后立即收尾、不再点击。
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from src.rpa.red_dot_detector import RedDotDetector


def make_detector(should_abort=None, mode="double_click_pin"):
    return RedDotDetector(
        config={"wechat": {"recognition_mode": mode}},
        should_abort=should_abort,
    )


# =================================================================
# 1) _aborted 判定三态
# =================================================================
def test_aborted_none_default():
    # 未注入 should_abort（默认 None）→ 永不中止，且不崩
    det = RedDotDetector()
    assert det._aborted() is False
    print("PASSED: 默认 should_abort=None → _aborted=False")


def test_aborted_true():
    det = make_detector(should_abort=lambda: True)
    assert det._aborted() is True
    print("PASSED: should_abort 返回 True → _aborted=True")


def test_aborted_false():
    det = make_detector(should_abort=lambda: False)
    assert det._aborted() is False
    print("PASSED: should_abort 返回 False → _aborted=False")


def test_aborted_exception_safe():
    def boom():
        raise RuntimeError("unexpected")
    det = make_detector(should_abort=boom)
    assert det._aborted() is False, "回调抛异常时应安全降级为 False（不中止也不崩）"
    print("PASSED: should_abort 抛异常 → _aborted=False 不崩")


# =================================================================
# 2) _do_click 统一入口守卫
# =================================================================
def test_do_click_aborted_returns_none():
    det = make_detector(should_abort=lambda: True)
    # 收到停止后，所有点击统一入口应直接返回 None（不落任何点击）
    assert det._do_click(12345, 100, 100, double=False) is None
    assert det._do_click(12345, 100, 100, double=True) is None
    print("PASSED: _do_click 收到停止 → 返回 None 不点击")


# =================================================================
# 3) find_and_click_unread 截图后守卫
# =================================================================
def test_find_and_click_aborted_after_screenshot():
    flag = {"abort": False}
    det = make_detector(should_abort=lambda: flag["abort"])
    det._auto_archive = False
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    det._live_rect = lambda hw: {"width": 1000, "height": 600, "left": 0, "top": 0}
    calls = {"pin": 0}

    def _capture(hw):
        flag["abort"] = True  # 模拟截图期间收到停止请求
        return img.copy()

    det._capture = _capture
    det._scan_nav_badge = lambda im: {"unread_count": 3}
    det._scan_contact_dots = lambda im: []
    det._pin_unread_to_top = (
        lambda hw, w, h, wm: (calls.__setitem__("pin", calls["pin"] + 1) or True)
    )

    res = det.find_and_click_unread(window_handle=12345, wm=None)
    assert res.get("reason") == "assistant_stop_requested", res
    assert res.get("clicked") is False, res
    assert calls["pin"] == 0, "截图后收到停止，绝不应双击置顶"
    print("PASSED: 截图后收到停止 → reason=assistant_stop_requested 且不置顶")


# =================================================================
# 4) pin 循环迭代守卫（ensure_chat_list 期间收到停止）
# =================================================================
def test_pin_loop_aborted_during_ensure():
    flag = {"abort": False}
    det = make_detector(should_abort=lambda: flag["abort"])
    det._auto_archive = False
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    det._live_rect = lambda hw: {"width": 1000, "height": 600, "left": 0, "top": 0}
    det._capture = lambda hw: img.copy()
    det._scan_nav_badge = lambda im: {"unread_count": 3}
    det._scan_contact_dots = lambda im: []
    det._dot_clickable = lambda d: True
    calls = {"pin": 0}
    det._pin_unread_to_top = (
        lambda hw, w, h, wm: (calls.__setitem__("pin", calls["pin"] + 1) or True)
    )

    def _ensure(hwnd, w, h, wm=None):
        flag["abort"] = True  # 模拟确保聊天列表期间收到停止请求
        return True

    det._ensure_chat_list = _ensure

    res = det.find_and_click_unread(window_handle=12345, wm=None)
    assert res.get("reason") == "assistant_stop_requested", res
    assert calls["pin"] == 0, "进入置顶循环前收到停止，绝不应双击置顶"
    print("PASSED: ensure_chat_list 期间收到停止 → pin 循环守卫立即收尾不置顶")


if __name__ == "__main__":
    test_aborted_none_default()
    test_aborted_true()
    test_aborted_false()
    test_aborted_exception_safe()
    test_do_click_aborted_returns_none()
    test_find_and_click_aborted_after_screenshot()
    test_pin_loop_aborted_during_ensure()
    print("ALL_TESTS_PASSED")
