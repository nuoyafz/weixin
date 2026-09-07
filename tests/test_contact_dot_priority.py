"""验证 Q1 修复：列表里只要扫到红点（即使没读出数字/面积没过严格阈值），
就优先点最顶红点，而不是退回「处理当前打开的会话」。"""
import sys, types
sys.path.insert(0, ".")

import numpy as np
import cv2

from src.rpa.red_dot_detector import RedDotDetector


def make_dot(cy, unread=None, area=40):
    return {
        "x": 180, "y": cy - 7, "w": 14, "h": 14, "area": area,
        "center_x": 194, "center_y": cy,
        "unread_count": unread, "_win_w": 2002, "_win_h": 1317,
        "label": f"c{cy}",
    }


def test_contact_dot_priority_over_current_chat():
    d = RedDotDetector.__new__(RedDotDetector)
    # 最小可用初始化：仅填充决策分支所需字段
    d._consecutive_empty = 0
    d._nav_badge = None
    d._contact_dots = []
    d._current_hwnd = 0
    d._window_rect = {"width": 2002, "height": 1317}
    d._last_screenshot = None
    d._nick_ocr = None
    d._blacklisted_hits = []
    d._last_contact_y = -1
    d._consecutive_contact_clicks = 0
    d._click_round = 0
    d._failed_contact_ys = set()
    d._failed_contact_y_frames = {}
    d._cfg_cache = {}
    d.FAILED_Y_TTL_ROUNDS = 3
    d.MAX_CONSECUTIVE_CONTACT_CLICKS = 3
    d._is_blacklisted_cache = {}
    d._avatar_anchor_x = None
    d._avatar_anchor_frames = 0
    d._window_state_before_cycle = None
    d._auto_archive = False

    def fake_archive(img):
        pass
    d._archive_debug = fake_archive

    calls = {}

    def fake_live_rect(hw):
        return {"x": 0, "y": 0, "width": 2002, "height": 1317}
    def fake_capture(hw):
        return np.zeros((1317, 2002, 3), dtype=np.uint8)
    def fake_scan_nav(img):
        return None  # 无 nav 未读数字
    def fake_scan_contact(img):
        return [make_dot(167)]  # 方舟：没数字、面积 40（低于严格阈值 60）
    def fake_pick(hw, dots, img, win_w, wm):
        calls["picked"] = [dd["center_y"] for dd in dots]
        # 模拟成功点击最顶红点
        return {"found": True, "clicked": True, "kind": "contact_dot",
                "contact": "方舟", "entered_conversation": True,
                "reason": "clicked contact dot (blacklist filtered)",
                "unread_count": None, "click_x": 194, "click_y": 167}
    def fake_ensure(hw, w, h, wm=None):
        return True
    def fake_pin(hw, w, h, wm):
        return True
    def fake_top(hw, w, h, wm, nav_num=None):
        return {"found": True, "clicked": False, "kind": "nav_badge",
                "reason": "unused"}

    d._live_rect = fake_live_rect
    d._capture = fake_capture
    d._scan_nav_badge = fake_scan_nav
    d._scan_contact_dots = fake_scan_contact
    d._pick_and_click_contact_dot = fake_pick
    d._ensure_chat_list = fake_ensure
    d._pin_unread_to_top = fake_pin
    d._click_top_conversation_row = fake_top

    # 直接调用决策主体（_run_one_cycle_impl 在 observe_service，这里只验 detector）
    # 为复用 find_and_click_unread，它内部会调 _live_rect/_capture 等桩。
    res = d.find_and_click_unread(0, wm=None)

    assert calls.get("picked") == [167], f"应把最顶红点(167)交给点击：{calls}"
    assert res.get("clicked") is True, f"应点击未读：{res}"
    assert res.get("contact") == "方舟", f"应进入方舟：{res}"
    print("PASS -> 扫到红点即点最顶（不退回当前会话）", res.get("reason"))


if __name__ == "__main__":
    test_contact_dot_priority_over_current_chat()
    print("ALL PASS")
