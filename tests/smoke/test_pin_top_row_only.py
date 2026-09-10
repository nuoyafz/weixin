# -*- coding: utf-8 -*-
"""双击置顶后「只看第一行」——回归锁定（真机 2026-09-10 12:29 bug）。

真机症状（方哥反馈 + 日志）：
    12:28:52 双击置顶执行 点击=(30,200)
    12:28:56 顶行:搜索框底边=121 -> 第一行 y=130      ← 顶行坐标已算对
    12:29:00 未读行: 列表未读徽章 数字=1 名字='代取快递外卖' 红点位置=(188,783)
    12:29:00 顶行:未读行优先 ... click=(293,783)      ← 被全列表扫描覆盖成错行
    12:29:00 点击顶行 坐标=(293,783) 来源=未读行        ← 点错行

期望行为（方哥原话）：双击置顶后应判断「第一个会话是否未读」——
未读则进，否则继续双击置顶；而不是直接去点列表里的未读红点（识别率太低导致点错）。

本文件锁定三件事：
  1. 点击目标恒为第一行，全列表「未读行优先」分支已被删除；
  2. 顶行未读判定三态（unread / clean / unknown）与安全兜底语义；
  3. pin 分支在 clean 时不点击、继续双击置顶，耗尽后走列表红点兜底。

纯逻辑 + 桩，无网络、无 cv2、不读真实截图，可稳定在 CI 运行。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DETECTOR_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "rpa", "red_dot_detector.py")

H, W = 1317, 1521
SB_Y = 100
TOP_Y = 170
FAR_Y = 500


def _src():
    with open(DETECTOR_SRC, encoding="utf-8") as f:
        return f.read()


# ---------- 1. 源码守卫：全列表扫描分支必须消失，点击目标恒为第一行 ----------

def test_whole_list_unread_row_scan_removed():
    """`_find_unread_row`（全列表扫红点并覆盖点击点）必须已被删除。"""
    src = _src()
    assert "def _find_unread_row" not in src, \
        "全列表「未读行优先」分支仍在：会覆盖顶行坐标导致点错行（真机 12:29 bug）"
    assert 'click_src = "unread_row"' not in src, "仍存在 unread_row 点击来源"


def test_pin_branch_declares_top_only():
    """pin 分支必须显式声明 top_only=True，并改用三态顶行判定。"""
    src = _src()
    i = src.index("def _pin_doubleclick_branch")
    j = src.index("def _contact_click_reduced_unread")
    seg = src[i:j]
    assert "_top_row_badge_state" in seg, "pin 分支未做「只看第一行」判定"
    assert "top_only=True" in seg, "pin 分支未把点击范围限制在第一行"
    assert "_top_row_has_reddot" not in seg, "pin 分支仍在用旧的整列表红点门限"


# ---------- 2. 顶行三态判定 ----------

def _detector(band_has_badge, low_dots):
    from src.rpa.red_dot_detector import RedDotDetector
    d = RedDotDetector()
    d._detect_search_box_bottom = lambda img, w, h: SB_Y
    d._avatar_bands = lambda img, w, h, from_y, limit=3: [(150, 190), (240, 280)]
    d._band_has_badge = lambda band: band_has_badge
    d._find_dots = lambda region, max_size=None: list(low_dots)
    return d


IMG = np.zeros((H, W, 3), dtype=np.uint8)


def test_top_row_with_badge_is_unread():
    d = _detector(True, [])
    state, top_y = d._top_row_badge_state(IMG, W, H)
    assert state == "unread", state
    assert top_y is not None


def test_far_dot_only_is_clean():
    d = _detector(False, [object()])
    assert d._top_row_badge_state(IMG, W, H)[0] == "clean"


def test_no_dot_anywhere_is_unknown():
    """整列表都扫不到红点（T1 整体漏检）→ unknown：保守点第一行，不引入新跳过。"""
    d = _detector(False, [])
    assert d._top_row_badge_state(IMG, W, H)[0] == "unknown"


def test_search_box_unlocatable_is_unknown():
    d = _detector(False, [object()])
    d._detect_search_box_bottom = lambda *a, **k: None
    assert d._top_row_badge_state(IMG, W, H)[0] == "unknown"


def test_row_unlocatable_but_dots_below_is_clean():
    """第一行定位失败但列表下方有红点 → clean：不盲点第一行，继续双击置顶，
    耗尽后由列表红点扫描兜底（比盲点非未读行更安全）。"""
    d = _detector(False, [object()])
    d._avatar_bands = lambda *a, **k: []
    assert d._top_row_badge_state(IMG, W, H)[0] == "clean"


# ---------- 3. 集成：pin 分支行为 ----------

def _build(state, click_rec, img=None):
    from src.rpa.red_dot_detector import RedDotDetector
    d = RedDotDetector()
    image = IMG if img is None else img
    d._capture = lambda hwnd: image
    d._live_rect = lambda hwnd: {"width": W, "height": H, "left": 0, "top": 0}
    d._do_click = lambda hwnd, x, y, double=False, wm=None: True
    d._ensure_chat_list = lambda *a, **k: True
    d._pin_unread_to_top = lambda *a, **k: True
    d._scan_nav_badge = lambda img: {"unread_count": 2}
    d._scan_contact_dots = lambda img: [
        {"center_x": 199, "center_y": FAR_Y, "area": 171, "unread_count": 1}]
    d._detect_search_box_bottom = lambda img, w, h: SB_Y
    d._first_avatar_below = lambda img, w, h, from_y: TOP_Y
    d._top_row_badge_state = lambda img, w, h: (state, TOP_Y)
    d._ocr_first_contact_row = lambda img, w, h, sb: None
    d._resolve_contact_name = lambda *a, **k: ""
    d._verify_pin_success = lambda *a, **k: ("success", 0)

    orig = d._click_top_conversation_row

    def wrapped(*a, **k):
        click_rec["called"] = True
        click_rec["top_only"] = k.get("top_only")
        res = orig(*a, **k)
        click_rec["click_y"] = res.get("click_y")
        return res

    d._click_top_conversation_row = wrapped
    d._pick_and_click_contact_dot = lambda *a, **k: {
        "found": True, "clicked": True, "kind": "contact_dot",
        "contact": "兜底会话", "entered_conversation": True,
        "click_method": "contact_dot", "click_y": FAR_Y}
    return d


def test_clean_top_row_skips_click_and_repins():
    """顶行非未读 → 不点、反复重双击置顶 → 耗尽后列表红点兜底。"""
    rec = {"called": False}
    d = _build("clean", rec)
    res = d.find_and_click_unread(window_handle=1, wm=None)
    assert rec["called"] is False, "顶行非未读时绝不能点第一行"
    assert res["clicked"] is True, res
    assert res.get("pin_verify") == "fallback_red_dot_no_top_dot", res


def test_unread_top_row_clicks_first_row_only():
    """顶行未读 → 点第一行（top_only=True），点击 y 落在第一行。"""
    rec = {"called": False}
    d = _build("unread", rec)
    res = d.find_and_click_unread(window_handle=1, wm=None)
    assert rec["called"] is True
    assert rec["top_only"] is True
    assert res["clicked"] is True, res
    assert abs(int(res["click_y"]) - TOP_Y) <= 20, res
    assert int(res["click_y"]) != FAR_Y


def test_click_target_never_from_whole_list_scan():
    """直接调 _click_top_conversation_row：列表深处有红点也不得改变点击 y。"""
    rec = {"called": False}
    d = _build("unread", rec)
    d._do_click = lambda hwnd, x, y, double=False, wm=None: rec.update(
        x=x, click_y=y) or True
    res = d._click_top_conversation_row(1, W, H, None, nav_num=2, img=IMG,
                                        top_only=True)
    assert res["clicked"] is True, res
    assert abs(int(rec["click_y"]) - TOP_Y) <= 20, rec
    assert int(rec["click_y"]) != FAR_Y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
