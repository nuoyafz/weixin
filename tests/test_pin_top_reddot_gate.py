"""测试：双击置顶分支的「只看第一行」门限（用户 2026-09-10 要求）。

核心场景：双击置顶后，只判定列表**第一行**是否未读 ——
  · 顶行未读   → 点这一行进会话；
  · 顶行非未读 → 不点，立即重双击置顶取下一个未读（耗尽后降级列表红点兜底）；
  · 无法判定   → 保守照常点顶行。

回归锁定（真机 2026-09-10 12:29 日志的 bug）：旧实现里 `_click_top_conversation_row`
的「未读行优先」分支会在**全列表**扫红点并覆盖点击点，把已经算对的顶行 y=130
覆盖成 y=783 的错行。该分支已彻底删除，本文件锁定"点击目标恒为第一行"。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

H, W = 1317, 1521
SB_Y = 100                      # 搜索框底边
BANDS = [(150, 190), (240, 280)]  # 头像带：首行 150~190 -> top_y=170，行高 90
TOP_Y = 170
FAR_Y = 500                     # 列表深处的红点（旧 bug 会点到这里）


def make_img(dot_ys=()):
    """构造纯黑截图，按需画红色方块（17x17，可通过 _find_dots 全部校验）。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for dy in dot_ys:
        img[int(dy) - 8:int(dy) + 9, 192:209] = (0, 0, 220)  # BGR 红
    return img


def build(dot_ys=(), state_override=None, click_recorder=None):
    d = RedDotDetector()
    img = make_img(dot_ys)
    d._capture = lambda hwnd: img
    d._live_rect = lambda hwnd: {"width": W, "height": H, "left": 0, "top": 0}
    d._nav_chat_icon_y = lambda img: 200
    d._do_click = lambda hwnd, x, y, double=False, wm=None: True
    d._ensure_chat_list = lambda *a, **k: True
    d._pin_unread_to_top = lambda *a, **k: True
    d._scan_nav_badge = lambda img: {"unread_count": 2, "center_x": 71, "center_y": 193}
    # 兜底路径用的整列表红点（_dot_clickable 需 unread_count 非空）
    d._scan_contact_dots = lambda img: [
        {"center_x": 199, "center_y": FAR_Y, "area": 171, "unread_count": 1,
         "x": 192, "y": FAR_Y - 8, "w": 17, "h": 17, "_win_w": W, "_win_h": H}]
    d._detect_search_box_bottom = lambda img, w, h: SB_Y
    d._avatar_bands = lambda img, w, h, from_y, limit=3: list(BANDS)
    d._ocr_first_contact_row = lambda img, w, h, sb: None
    d._resolve_contact_name = lambda *a, **k: ""
    if state_override is not None:
        d._top_row_badge_state = lambda img, w, h: state_override(img, w, h)
    if click_recorder is not None:
        orig = d._click_top_conversation_row

        def wrapped(*a, **k):
            click_recorder["called"] = True
            click_recorder["top_only"] = k.get("top_only")
            return orig(*a, **k)

        d._click_top_conversation_row = wrapped
    d._pick_and_click_contact_dot = lambda *a, **k: {
        "found": True, "clicked": True, "kind": "contact_dot",
        "contact": "兜底会话", "entered_conversation": True,
        "click_method": "contact_dot"}
    return d


# ---------- 1. 顶行未读判定（_top_row_badge_state） ----------

def test_top_row_with_badge_is_unread():
    """顶行徽章区有红块 → unread（应点这一行）。"""
    d = build(dot_ys=(TOP_Y,))
    state, top_y = d._top_row_badge_state(make_img((TOP_Y,)), W, H)
    assert state == "unread", (state, top_y)
    assert abs(top_y - TOP_Y) <= 20


def test_far_dot_only_is_clean():
    """红点只在列表深处（顶行无徽章）→ clean（不点，继续双击置顶）。"""
    d = build(dot_ys=(FAR_Y,))
    state, top_y = d._top_row_badge_state(make_img((FAR_Y,)), W, H)
    assert state == "clean", (state, top_y)


def test_no_dot_anywhere_is_unknown():
    """整列表都扫不到红点（T1 整体漏检）→ unknown（保守点顶行，不引入新跳过）。"""
    d = build()
    state, _ = d._top_row_badge_state(make_img(()), W, H)
    assert state == "unknown", state


def test_search_box_unlocatable_is_unknown():
    """搜索框检不出 → unknown（保守），不得误判 clean。"""
    d = build()
    d._detect_search_box_bottom = lambda *a, **k: None
    assert d._top_row_badge_state(make_img((FAR_Y,)), W, H)[0] == "unknown"


def test_row_unlocatable_but_dots_below_is_clean():
    """第一行定位失败但列表下方有红点 → clean（不盲点第一行，继续双击置顶）。"""
    d = build()
    d._avatar_bands = lambda *a, **k: []
    assert d._top_row_badge_state(make_img((FAR_Y,)), W, H)[0] == "clean"


# ---------- 2. 点击目标恒为第一行（回归锁定：不再被全列表扫描覆盖） ----------

def test_click_target_is_top_row_not_far_dot():
    """顶行未读 + 列表深处也有红点 → 点击必须落在顶行，绝不点 y=500。"""
    d = build(dot_ys=(TOP_Y, FAR_Y))
    clicked = {}
    d._do_click = lambda hwnd, x, y, double=False, wm=None: clicked.update(x=x, y=y) or True
    d._verify_pin_success = lambda *a, **k: ("success", 0)
    res = d._click_top_conversation_row(1, W, H, None, nav_num=2,
                                        img=make_img((TOP_Y, FAR_Y)),
                                        top_only=True)
    assert res.get("clicked") is True, res
    assert abs(clicked["y"] - TOP_Y) <= 20, clicked
    assert clicked["y"] != FAR_Y


# ---------- 3. pin 分支：顶行非未读 → 跳过点击 → 列表红点兜底 ----------

def test_branch_skips_top_and_falls_back():
    """集成：顶行非未读 → 不点顶行 → 重双击置顶 → 耗尽后列表红点兜底点击。"""
    rec = {"called": False}
    d = build(dot_ys=(FAR_Y,), click_recorder=rec)
    d._verify_pin_success = lambda *a, **k: ("success", 0)

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[GATE] result:", res["kind"], "clicked=", res["clicked"],
          "pin_verify=", res.get("pin_verify"), "contact=", res.get("contact"))
    assert rec["called"] is False, "顶行非未读时应跳过点，不应调用 _click_top_conversation_row"
    assert res["clicked"] is True, res
    assert res.get("pin_verify") == "fallback_red_dot_no_top_dot", res
    assert res.get("contact") == "兜底会话", res


def test_branch_clicks_top_row_when_unread():
    """集成：顶行未读 → 点第一行（top_only=True）并验证成功返回。"""
    rec = {"called": False}
    d = build(dot_ys=(TOP_Y,), click_recorder=rec)
    d._verify_pin_success = lambda *a, **k: ("success", 0)

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[PIN-OK] result:", res["kind"], "clicked=", res["clicked"],
          "y=", res.get("click_y"), "method=", res.get("click_method"))
    assert rec["called"] is True
    assert rec["top_only"] is True, "pin 分支必须显式声明 top_only=True"
    assert res["clicked"] is True, res
    assert abs(int(res["click_y"]) - TOP_Y) <= 20, res


if __name__ == "__main__":
    test_top_row_with_badge_is_unread()
    test_far_dot_only_is_clean()
    test_no_dot_anywhere_is_unknown()
    test_search_box_unlocatable_is_unknown()
    test_row_unlocatable_but_dots_below_is_clean()
    test_click_target_is_top_row_not_far_dot()
    test_branch_skips_top_and_falls_back()
    test_branch_clicks_top_row_when_unread()
    print("\nALL PASS")
