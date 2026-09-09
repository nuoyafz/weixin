"""测试：nav 有未读数字时的「双击置顶 + 点顶行」路由。

模拟：
  场景 A：双击置顶后红点仍没扫到 -> 走坐标点最顶行兜底
  场景 B：双击置顶后红点扫到了 -> 走 _pick_and_click_contact_dot 点最顶红点
  场景 C：nav 无数字（仅徽章，数字被红包围白校验误杀）-> 仍双击置顶（先执行双击置顶）
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
    # 桩：置顶生效（验证成功）。本用例只校验「路由 + 点击序列」；
    # 「验证失败→重试3次→降级红点」策略由 test_recognition_mode 覆盖
    # （方哥要求：验证失败继续双击置顶重试）。
    d._verify_pin_success = (lambda hw, w, h, wm, nav_num,
                             before_img=None: ("success", 0))

    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[A] result:", res["kind"], "clicked=", res["clicked"],
          "entered=", res.get("entered_conversation"), "reason=", res["reason"])
    # 行为序列：①单击聊天图标回列表 ②双击置顶 ③坐标点顶行
    assert clicks, "应当产生点击"
    assert clicks[0] == (30, 200, False), f"首个应为单击回列表(30,200): {clicks[0]}"
    assert clicks[1] == (30, 200, True), f"其次应为双击置顶(30,200): {clicks[1]}"
    assert any(c[2] for c in clicks), "应当有双击置顶动作"
    # 最终进入未读会话
    assert res["clicked"] is True, res
    assert res.get("entered_conversation") is True, res
    assert res["kind"] == "contact_dot", res
    print("[A] PASS -> 单击(30,200) + 双击置顶(30,200) + 坐标点顶行")


def test_scenario_b():
    """双击置顶策略：红点扫到时同样走坐标点顶行（不再靠红点 OCR 定位）。

    与 A 的区别仅在于红点是否被扫到；当前统一策略下两者均进入未读会话。
    """
    global clicks
    clicks = []
    d = make_detector()

    _scan_n = {"n": 0}
    def fake_scan(img):
        _scan_n["n"] += 1
        if _scan_n["n"] == 1:
            # 预检时刻（双击置顶前）列表尚无新红点，不触发「已在列表」跳过
            return []
        # 置顶后，亚磊(顶) + 方舟 都出现红点
        return [
            {"center_x": 199, "center_y": 167, "area": 171, "unread_count": 1,
             "x": 190, "y": 159, "w": 15, "h": 15, "_win_w": W, "_win_h": H},
            {"center_x": 199, "center_y": 493, "area": 171, "unread_count": 1,
             "x": 190, "y": 485, "w": 15, "h": 15, "_win_w": W, "_win_h": H},
        ]

    d._scan_nav_badge = lambda img: {"unread_count": 2, "center_x": 71, "center_y": 193}
    d._scan_contact_dots = fake_scan
    # 桩：置顶生效（验证成功）。本用例只校验「路由 + 点击序列」；
    # 「验证失败→重试3次→降级红点」策略由 test_recognition_mode 覆盖
    # （方哥要求：验证失败继续双击置顶重试）。
    d._verify_pin_success = (lambda hw, w, h, wm, nav_num,
                             before_img=None: ("success", 0))


    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[B] result:", res["kind"], "clicked=", res["clicked"],
          "entered=", res.get("entered_conversation"), "unread=", res.get("unread_count"))
    # P1 后：红点已就位时 _ensure_chat_list 预检会跳过单击，直接进入双击置顶。
    # 核心契约：无论是否经单击回列表，最终都「双击置顶(30,200) + 坐标点顶行」进入会话。
    assert any(c[2] for c in clicks), "应当有双击置顶动作"
    assert (30, 200, True) in clicks, f"应有双击置顶(30,200): {clicks}"
    assert res["clicked"] is True, res
    assert res.get("entered_conversation") is True, res
    assert res["kind"] == "contact_dot", res
    print("[B] PASS -> 双击置顶(30,200) + 坐标点顶行 (P1: 红点就位跳过单击)")


def test_scenario_c():
    """nav 无数字（仅徽章，数字被红包围白校验误杀）-> 仍走双击置顶主路径。

    用户要求「先执行双击置顶」：检测到 nav 未读徽章即双击置顶，
    不再因 unread_count=None 退化到红点直点；顶行红点门限(:419)兜底。
    """
    global clicks
    clicks = []
    d = make_detector()
    d._scan_nav_badge = lambda img: {"unread_count": None, "center_x": 71, "center_y": 60}
    d._scan_contact_dots = lambda img: []
    # 桩验证成功，使主路径走通可预测；重点校验「无数字 nav 仍进入双击置顶」
    d._verify_pin_success = (lambda hw, w, h, wm, nav_num,
                             before_img=None: ("success", 0))
    res = d.find_and_click_unread(window_handle=1, wm=None)
    print("[C] result:", res["kind"], "clicked=", res["clicked"],
          "reason=", res["reason"])
    # 关键：无数字 nav 仍产生双击置顶动作（而非退化为红点直点）
    assert any(c[2] for c in clicks), "无数字 nav 仍应双击置顶"
    assert (30, 200, True) in clicks, f"应有双击置顶(30,200): {clicks}"
    assert res["clicked"] is True, res
    assert res.get("entered_conversation") is True, res
    print("[C] PASS -> 无数字 nav 仍双击置顶(30,200) + 点顶行")


if __name__ == "__main__":
    test_scenario_a()
    test_scenario_b()
    test_scenario_c()
    print("\nALL PASS")
