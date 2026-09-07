"""测试：nav 有未读数字时的「双击置顶 + 点顶行」路由。

模拟：
  场景 A：双击置顶后红点仍没扫到 -> 走坐标点最顶行兜底
  场景 B：双击置顶后红点扫到了 -> 走 _pick_and_click_contact_dot 点最顶红点
  场景 C：nav 无数字（头像噪声）-> 不双击，走原 _ensure_chat_list 单点恢复
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

H, W = 1317, 1521
FAKE_IMG = np.zeros((H, W, 3), dtype=np.uint8)


def make_detector():
    d = RedDotDetector()
    # 禁用实时截图/窗口交互：全部打桩
    d._capture = lambda hwnd: FAKE_IMG
    d._live_rect = lambda hwnd: {"width": W, "height": H, "left": 0, "top": 0}
    d._nav_chat_icon_y = lambda img: 200
    d._do_click = lambda hwnd, x, y, double=False, wm=None: (
        clicks.append((x, y, double)) or True)
    return d


calls = {}
clicks = []


def test_scenario_a():
    """双击置顶策略：红点漏扫时，先单击回聊天列表 → 双击置顶 → 坐标点顶行兜底。

    注：red_dot_detector 于 9/2 重写为「双击置顶 + 坐标点顶行」统一策略，
    红点是否扫到不再影响主路径（都走坐标点顶行），故这里只校验行为序列
    与最终进入会话，不依赖黑图桩无法提供的真实几何坐标。
    """
    global clicks
    clicks = []
    d = make_detector()
    d._scan_nav_badge = lambda img: {"unread_count": 4, "center_x": 71, "center_y": 193}
    d._scan_contact_dots = lambda img: []  # 红点漏扫

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[A] result:", res["kind"], "clicked=", res["clicked"],
          "entered=", res.get("entered_conversation"), "reason=", res["reason"])
    # 行为序列：①单击聊天图标回列表 ②双击置顶 ③坐标点顶行
    assert clicks, "应当产生点击"
    assert clicks[0] == (28, 200, False), f"首个应为单击回列表(28,200): {clicks[0]}"
    assert clicks[1] == (28, 200, True), f"其次应为双击置顶(28,200): {clicks[1]}"
    assert any(c[2] for c in clicks), "应当有双击置顶动作"
    # 最终进入未读会话
    assert res["clicked"] is True, res
    assert res.get("entered_conversation") is True, res
    assert res["kind"] == "contact_dot", res
    print("[A] PASS -> 单击(28,200) + 双击置顶(28,200) + 坐标点顶行")


def test_scenario_b():
    """双击置顶策略：红点扫到时同样走坐标点顶行（不再靠红点 OCR 定位）。

    与 A 的区别仅在于红点是否被扫到；当前统一策略下两者均进入未读会话。
    """
    global clicks
    clicks = []
    d = make_detector()

    def fake_scan(img):
        # 置顶后，亚磊(顶) + 方舟 都出现红点
        return [
            {"center_x": 199, "center_y": 167, "area": 171, "unread_count": 1,
             "x": 190, "y": 159, "w": 15, "h": 15, "_win_w": W, "_win_h": H},
            {"center_x": 199, "center_y": 493, "area": 171, "unread_count": 1,
             "x": 190, "y": 485, "w": 15, "h": 15, "_win_w": W, "_win_h": H},
        ]

    d._scan_nav_badge = lambda img: {"unread_count": 2, "center_x": 71, "center_y": 193}
    d._scan_contact_dots = fake_scan

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[B] result:", res["kind"], "clicked=", res["clicked"],
          "entered=", res.get("entered_conversation"), "unread=", res.get("unread_count"))
    # 同样：单击回列表 → 双击置顶 → 坐标点顶行，最终进入会话
    assert clicks[0] == (28, 200, False), f"首个应为单击回列表(28,200): {clicks[0]}"
    assert clicks[1] == (28, 200, True), f"其次应为双击置顶(28,200): {clicks[1]}"
    assert res["clicked"] is True, res
    assert res.get("entered_conversation") is True, res
    assert res["kind"] == "contact_dot", res
    print("[B] PASS -> 单击(28,200) + 双击置顶(28,200) + 坐标点顶行")


def test_scenario_c():
    """nav 无数字（头像噪声）-> 不双击，走原 _ensure_chat_list 单点。"""
    global clicks
    clicks = []
    d = make_detector()
    d._scan_nav_badge = lambda img: {"unread_count": None, "center_x": 71, "center_y": 60}
    d._scan_contact_dots = lambda img: []
    # _ensure_chat_list 内部会再 _capture + _scan_contact_dots，需维持桩
    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[C] result:", res["kind"], "reason=", res["reason"])
    assert not any(c[2] for c in clicks), "无数字 nav 不应双击"
    # _ensure_chat_list 用单点，且坐标应落在聊天图标(29,200)
    single = [c for c in clicks if not c[2]]
    assert single and single[0] == (28, 200, False), f"应单点聊天图标: {single}"
    print("[C] PASS -> 不双击，单点聊天图标(29,200) 走原恢复逻辑")


if __name__ == "__main__":
    test_scenario_a()
    test_scenario_b()
    test_scenario_c()
    print("\nALL PASS")
