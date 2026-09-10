"""验证 recognition_mode 两种模式行为 + 双击置顶未读数量校验/降级。

- double_click_pin: 走双击置顶路径，双击后直接点顶行；点击后校验未读数量是否减少。
  - 减少 → success，回填 pin_echo.verify="success"
  - 未减少 → fail，降级红点扫描兜底（fallback_red_dot）或返回失败
  - before 读不出 → unknown（不强行判失败）
- red_dot: 跳过置顶，直接走红点/主页恢复分支（不调用 _pin_unread_to_top）。
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from src.rpa.red_dot_detector import RedDotDetector


def make_detector(mode: str) -> RedDotDetector:
    return RedDotDetector(config={"wechat": {"recognition_mode": mode}})


# =================================================================
# 1) 配置读取
# =================================================================
def test_read_mode():
    assert make_detector("red_dot")._recognition_mode == "red_dot"
    assert make_detector("double_click_pin")._recognition_mode == "double_click_pin"
    assert RedDotDetector(config={})._recognition_mode == "double_click_pin"
    print("PASSED: recognition_mode 配置读取正确")


# =================================================================
# 2) _verify_pin_success 纯逻辑（before/after 判定）
# =================================================================
def test_verify_pin_success_logic():
    det = make_detector("double_click_pin")
    det._auto_archive = False
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    det._capture = lambda hw: img.copy()

    # after < before → success
    det._scan_nav_badge = lambda im: {"unread_count": 2}
    assert det._verify_pin_success(1, 1000, 600, None, 3) == ("success", 2)
    # 徽章消失(after=None) → success（未读清零）
    det._scan_nav_badge = lambda im: None
    assert det._verify_pin_success(1, 1000, 600, None, 3) == ("success", 0)
    # 未变化 → fail
    det._scan_nav_badge = lambda im: {"unread_count": 3}
    assert det._verify_pin_success(1, 1000, 600, None, 3) == ("fail", 3)
    # before 读不出 → unknown
    det._scan_nav_badge = lambda im: {"unread_count": 1}
    assert det._verify_pin_success(1, 1000, 600, None, None) == ("unknown", 1)
    print("PASSED: _verify_pin_success 三种判定正确 (success/fail/unknown)")


# =================================================================
# 3) find_and_click_unread 端到端场景
# =================================================================
def run_pin_scenario(verify_return, nav_count=3, contacts_after_pin=None,
                     pick_result=None, entered_after_click=True):
    """构造一个双击置顶场景，由 verify_return 控制校验结果。

    entered_after_click: 模拟 _click_top_conversation_row 是否成功进入会话
    （真实实现中 clicked=True 时 entered_conversation=True）。用于区分
    "双击真失败（未进会话）" 与 "验证误判（已进会话但 nav 读数滞后）" 两种 fail。
    """
    det = make_detector("double_click_pin")
    det._auto_archive = False
    calls = {"pin": 0, "picked": 0}
    img = np.zeros((600, 1000, 3), dtype=np.uint8)

    det._live_rect = lambda hw: {"width": 1000, "height": 600, "left": 0, "top": 0}
    det._capture = lambda hw: img.copy()
    det._scan_nav_badge = lambda im: {"unread_count": nav_count}
    det._scan_contact_dots = lambda im: (contacts_after_pin or [])
    det._pin_unread_to_top = (
        lambda hw, w, h, wm: (calls.__setitem__("pin", calls["pin"] + 1) or True)
    )
    # 旧逻辑哨兵：若点最顶红点被误调用，计数
    det._pick_and_click_contact_dot = (
        lambda hw, dots, im, w, wm: (
            calls.__setitem__("picked", calls["picked"] + 1)
            or (pick_result if pick_result is not None else
                {"found": True, "clicked": True, "contact": "PICKED",
                 "unread_count": 1, "entered_conversation": True,
                 "kind": "contact_dot"})
        )
    )
    det._click_top_conversation_row = (
        lambda hw, w, h, wm, nav_num=None, img=None, top_only=False: {
            "found": True, "clicked": True, "kind": "contact_dot", "contact": "",
            "entered_conversation": entered_after_click,
            "click_method": "top_row_coordinate",
            "reason": "clicked top row", "unread_count": nav_num,
            "click_y": int(h * 0.105),
        }
    )
    det._dot_clickable = lambda d: True
    det._ensure_chat_list = lambda *a, **k: True
    # 校验结果由调用方注入
    det._verify_pin_success = lambda *a, **k: verify_return

    res = det.find_and_click_unread(window_handle=12345, wm=None)
    return res, calls


def test_double_click_pin_verify_success():
    res, calls = run_pin_scenario(("success", 2))
    assert calls["pin"] == 1
    assert calls["picked"] == 0, "success 不应触发降级红点扫描"
    assert res.get("pin_echo"), "应回填 pin_echo"
    pe = res["pin_echo"]
    assert pe["method"] == "top_row"
    assert pe["verify"] == "success", pe
    assert pe["before"] == 3 and pe["after"] == 2, pe
    print("PASSED: double_click_pin 校验成功 -> success, 不降级 ->", pe)


def test_double_click_pin_fail_fallback():
    # 真实失败（未进入会话）→ 降级红点扫描，且红点可点中
    contacts = [{"center_y": 300, "unread_count": 1}]
    res, calls = run_pin_scenario(
        ("fail", 3), contacts_after_pin=contacts, entered_after_click=False)
    assert calls["picked"] == 1, "fail 且未进会话时应降级调用 _pick_and_click_contact_dot"
    assert res.get("pin_verify") == "fallback_red_dot", res
    assert res.get("contact") == "PICKED", res
    print("PASSED: double_click_pin 真实失败(未进会话) -> 降级红点成功 ->", res.get("contact"))


def test_double_click_pin_fail_entered_retry_then_fallback():
    # 用户明确要求（red_dot_detector.py 「用户要求——继续双击置顶重试」注释）：
    # fail 且已进会话 → 继续双击置顶重试（共 MAX_PIN_RETRY=3 次尝试），
    # 耗尽后降级红点扫描兜底。真机日志 run_20260908_215104 实测同构流程
    # （重试第 2 次命中 success）。
    res, calls = run_pin_scenario(
        ("fail", 3), contacts_after_pin=[{"center_y": 300, "unread_count": 1}],
        entered_after_click=True)
    assert calls["pin"] == 3, f"fail+已进会话应重试满 3 次尝试，实际 {calls['pin']}"
    assert calls["picked"] == 1, "重试耗尽后应降级红点扫描"
    assert res.get("pin_verify") == "fallback_red_dot", res
    assert res.get("contact") == "PICKED", res
    print("PASSED: double_click_pin fail+已进会话 -> 重试3次 -> 降级红点 ->",
          res.get("pin_verify"))


def test_double_click_pin_fail_no_fallback():
    # 真实失败（未进入会话）→ 降级红点扫描，但无红点可点 → 返回失败结果
    res, calls = run_pin_scenario(
        ("fail", 3), contacts_after_pin=[], entered_after_click=False)
    assert calls["picked"] == 0
    assert res.get("clicked") is False, res
    assert res.get("pin_verify") == "fail", res
    print("PASSED: double_click_pin 真实失败(未进会话)且无红点 -> 返回失败结果 ->", res.get("reason"))


def test_red_dot_mode():
    det = make_detector("red_dot")
    det._auto_archive = False
    calls = {"pin": 0}
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    det._live_rect = lambda hw: {"width": 1000, "height": 600, "left": 0, "top": 0}
    det._capture = lambda hw: img.copy()
    det._scan_nav_badge = lambda im: {"unread_count": 3}
    det._scan_contact_dots = lambda im: []
    det._pin_unread_to_top = (
        lambda hw, w, h, wm: (calls.__setitem__("pin", calls["pin"] + 1) or True)
    )
    det._ensure_chat_list = lambda *a, **k: False
    res = det.find_and_click_unread(window_handle=12345, wm=None)
    assert calls["pin"] == 0, "red_dot 模式不应双击置顶"
    assert "pin_echo" not in res, "red_dot 模式不应有 pin_echo"
    print("PASSED: red_dot 模式跳过置顶，走红点/恢复分支 -> kind=", res.get("kind"))


def test_ensure_chat_list_guard_fires():
    """修复回归：_ensure_chat_list 返回 False 时，pin 分支必须真正 early-return
    （曾因 nav_num 未定义在 early-return 里抛 NameError，被 except 吞掉后静默继续
    置顶，导致守卫被架空）。"""
    det = make_detector("double_click_pin")
    det._auto_archive = False
    calls = {"pin": 0, "picked": 0}
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    det._live_rect = lambda hw: {"width": 1000, "height": 600, "left": 0, "top": 0}
    det._capture = lambda hw: img.copy()
    det._scan_nav_badge = lambda im: {"unread_count": 3}
    det._scan_contact_dots = lambda im: []
    det._pin_unread_to_top = (
        lambda hw, w, h, wm: (calls.__setitem__("pin", calls["pin"] + 1) or True)
    )
    det._dot_clickable = lambda d: True
    # 关键：聊天列表未就绪 → 守卫应触发，绝不走置顶
    det._ensure_chat_list = lambda *a, **k: False
    res = det.find_and_click_unread(window_handle=12345, wm=None)
    assert calls["pin"] == 0, "守卫触发后不应双击置顶"
    assert calls["picked"] == 0, "守卫触发后不应点红点"
    assert res.get("found") is False, res
    assert res.get("reason") == "ensure_chat_list failed before pin", res
    print("PASSED: _ensure_chat_list 守卫触发时真正 early-return，不静默继续置顶")


def test_detect_first_row_center_resolution_agnostic():
    """_detect_first_row_center 应随分辨率浮动（不写死比例）。"""
    import cv2
    from pathlib import Path
    det = RedDotDetector()
    base = Path("data/wechat")
    samples = [
        (base / "auto_open_unread_20260902_095433_492242.png", 1213),  # 1637x1213
        (base / "auto_open_unread_20260902_090348_574770.png", 1130),  # 1717x1130
    ]
    ys = []
    for fp, expected_h in samples:
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        assert h == expected_h, f"帧高不符: {fp.name} {h} != {expected_h}"
        y = det._detect_first_row_center(img, w, h)
        assert y is not None, f"未能检测第一行: {fp.name}"
        # 第一行应在搜索框(0.055H)下方、且明显低于写死的 0.075H 搜索框区
        assert y > h * 0.06 and y < h * 0.15, f"第一行 y 越界: {fp.name} y={y}"
        ys.append((fp.name, h, y, round(y / h, 4)))
    # 两个不同高度帧的检测 y/H 比例应接近（证明随分辨率浮动自适应）
    if len(ys) == 2:
        ratios = [r[3] for r in ys]
        assert abs(ratios[0] - ratios[1]) < 0.03, f"比例不随分辨率收敛: {ys}"
        print("PASSED: _detect_first_row_center 分辨率无关 ->", ys)
    else:
        print("SKIPPED: 样本不足", ys)


if __name__ == "__main__":
    test_read_mode()
    test_verify_pin_success_logic()
    test_double_click_pin_verify_success()
    test_double_click_pin_fail_fallback()
    test_double_click_pin_fail_entered_retry_then_fallback()
    test_double_click_pin_fail_no_fallback()
    test_red_dot_mode()
    test_ensure_chat_list_guard_fires()
    test_detect_first_row_center_resolution_agnostic()
    print("ALL_TESTS_PASSED")
