"""测试：双击置顶分支的「顶行无红点则跳过点击」门限（用户 9/8 要求）。

核心场景：双击置顶后，若列表里确有红点但都不在顶行（顶行非未读），
则不点进非未读会话、立即重双击置顶，耗尽后降级到列表红点扫描兜底。
安全约束：T1 漏检（无任何红点）时保守照常点击，不引入新跳过。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

H, W = 1317, 1521
FAKE_IMG = np.zeros((H, W, 3), dtype=np.uint8)

# 顶行 y：搜索框底边 100 + 首头像带 60 -> 顶行约 160
SB_Y = 100
TOP_Y = 160
# 红点：刻意放在远离顶行的位置（y=500），模拟「置顶未把未读顶上来/顶行非未读」
FAR_DOT = [{"center_x": 199, "center_y": 500, "area": 171, "unread_count": 1,
            "x": 190, "y": 492, "w": 15, "h": 15, "_win_w": W, "_win_h": H}]
# 红点：放在顶行附近（y=170），模拟「顶行确为未读」
NEAR_DOT = [{"center_x": 199, "center_y": 170, "area": 171, "unread_count": 1,
             "x": 190, "y": 162, "w": 15, "h": 15, "_win_w": W, "_win_h": H}]


def build(dots, top_click_recorder=None):
    d = RedDotDetector()
    d._capture = lambda hwnd: FAKE_IMG
    d._live_rect = lambda hwnd: {"width": W, "height": H, "left": 0, "top": 0}
    d._nav_chat_icon_y = lambda img: 200
    d._do_click = lambda hwnd, x, y, double=False, wm=None: True
    d._ensure_chat_list = lambda *a, **k: True
    d._pin_unread_to_top = lambda *a, **k: True
    d._scan_nav_badge = lambda img: {"unread_count": 2, "center_x": 71, "center_y": 193}
    d._scan_contact_dots = lambda img: list(dots)
    d._detect_search_box_bottom = lambda img, w, h: SB_Y
    d._first_avatar_below = lambda img, w, h, from_y: TOP_Y
    if top_click_recorder is not None:
        orig = d._click_top_conversation_row
        def wrapped(*a, **k):
            top_click_recorder["called"] = True
            return orig(*a, **k)
        d._click_top_conversation_row = wrapped
    # 兜底红点点击桩
    d._pick_and_click_contact_dot = lambda *a, **k: {
        "found": True, "clicked": True, "kind": "contact_dot",
        "contact": "兜底会话", "entered_conversation": True,
        "click_method": "contact_dot"}
    return d


def test_helper_no_dots_is_conservative():
    """T1 漏检（无红点）→ 保守返回 True，不跳过。"""
    d = build([])
    assert d._top_row_has_reddot(FAKE_IMG, W, H) is True


def test_helper_dot_far_from_top_is_false():
    """列表有红点但都不在顶行 → 返回 False（应跳过点）。"""
    d = build(FAR_DOT)
    assert d._top_row_has_reddot(FAKE_IMG, W, H) is False


def test_helper_dot_near_top_is_true():
    """顶行附近有红点 → 返回 True（正常点击）。"""
    d = build(NEAR_DOT)
    assert d._top_row_has_reddot(FAKE_IMG, W, H) is True


def test_branch_skips_top_and_falls_back():
    """集成：顶行无红点 → 不点顶行 → 重置顶 → 列表红点兜底点击。"""
    rec = {"called": False}
    d = build(FAR_DOT, top_click_recorder=rec)
    # 验证桩：本用例不应进入 verify（因为跳过点）
    d._verify_pin_success = lambda *a, **k: ("success", 0)

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[GATE] result:", res["kind"], "clicked=", res["clicked"],
          "pin_verify=", res.get("pin_verify"), "contact=", res.get("contact"))
    # 顶行点击绝不应发生
    assert rec["called"] is False, "顶行无红点时应跳过点，_click_top_conversation_row 不应被调用"
    # 应走列表红点兜底
    assert res["clicked"] is True, res
    assert res.get("pin_verify") == "fallback_red_dot_no_top_dot", res
    assert res.get("contact") == "兜底会话", res


if __name__ == "__main__":
    test_helper_no_dots_is_conservative()
    test_helper_dot_far_from_top_is_false()
    test_helper_dot_near_top_is_true()
    test_branch_skips_top_and_falls_back()
    print("\nALL PASS")
