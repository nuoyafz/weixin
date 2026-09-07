"""测试：融合形态徽章下的「置顶重试 + 放行」策略。

背景（真机 1638x1190，日志 16:14）：导航栏徽章与图标融合，_find_dots 检测不到
独立红块，只能靠 cluster-OCR 读数字（单次约 20~30s），像素快速路径 100% 失效。

策略（find_and_click_unread）：
  - 融合形态 -> 禁用读数字 OCR，判定不了就放行（交给下游 analyze_once 兜底）；
  - 点击成功时不再二次置顶（否则会把刚打开的会话切回列表再点一次顶行）；
  - 点击失败时才重试置顶，至多 2 轮；
  - 正常形态 -> 行为完全不变（像素优先，判不了仍照旧降级 OCR）。

用 Harness 打桩，统计 pin / OCR 次数以区分各分支。
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from src.rpa.red_dot_detector import RedDotDetector

H, W = 1190, 1638
FAKE_IMG = np.zeros((H, W, 3), dtype=np.uint8)


class Harness:
    """记录 pin / click / OCR 次数，便于断言策略分支。"""

    def __init__(self, metric=(0, 0), click_results=None):
        self.clicks = []
        self.pin_calls = 0
        self.ocr_calls = 0
        # _click_top_conversation_row 的返回序列（用于模拟"第 1 次点失败"）
        self.click_results = click_results or [True]
        self.click_seq_i = 0
        self.metric = metric
        self.det = self._build()

    def _build(self):
        d = RedDotDetector()
        d._capture = lambda hwnd: FAKE_IMG
        d._live_rect = lambda hwnd: {
            "width": W, "height": H, "left": 0, "top": 0}
        d._nav_chat_icon_y = lambda img: 200
        d._do_click = lambda hwnd, x, y, double=False, wm=None: (
            self.clicks.append((x, y, double)) or True)
        d._ensure_chat_list = lambda hwnd, w, h, wm=None: True
        d._scan_contact_dots = lambda img: []
        # 开头会调 1 次（首轮扫描）；verify 阶段若降级 OCR 会再调 3 次
        d._scan_nav_badge = self._fake_scan_nav_badge
        d._nav_badge_metric = lambda img: self.metric
        d._pin_unread_to_top = self._fake_pin
        d._click_top_conversation_row = self._fake_click_top
        return d

    def _fake_scan_nav_badge(self, img):
        self.ocr_calls += 1
        return {"unread_count": 1, "center_x": 71, "center_y": 193}

    def _fake_pin(self, hwnd, w, h, wm):
        self.pin_calls += 1
        return True

    def _fake_click_top(self, hwnd, w, h, wm, nav_num=None, img=None):
        idx = min(self.click_seq_i, len(self.click_results) - 1)
        self.click_seq_i += 1
        if not self.click_results[idx]:
            return {"found": True, "clicked": False, "kind": "nav_badge",
                    "entered_conversation": False, "reason": "stub fail"}
        return {"found": True, "clicked": True, "kind": "contact_dot",
                "entered_conversation": True,
                "click_method": "top_row_ocr", "contact": "方舟",
                "click_x": 266, "click_y": 215, "unread_count": nav_num}


def test_fused_badge_skips_ocr():
    """融合形态：verify 阶段不应调用读数字 OCR（否则每轮空耗 20~30s）。"""
    hr = Harness(metric=(0, 0))          # (0,0) = 检测不到徽章 = 融合形态
    res = hr.det.find_and_click_unread(window_handle=1, wm=None)
    print(f"  OCR 调用={hr.ocr_calls} pin 次数={hr.pin_calls} "
          f"pin_verify={res.get('pin_verify')}")
    # 只允许开头的 1 次首轮扫描；verify 阶段（原会再调 3 次）必须为 0
    assert hr.ocr_calls == 1, f"融合形态不应在 verify 阶段读数字，实际 {hr.ocr_calls}"
    assert res["clicked"] is True, res
    assert res.get("pin_verify") == "pass_unknown", res.get("pin_verify")
    print("  融合形态 -> 零 OCR，放行交下游")


def test_fused_badge_pins_once_when_click_ok():
    """点击已成功时不再二次置顶：否则会把刚打开的会话切回列表再点一次顶行。"""
    hr = Harness(metric=(0, 0))
    hr.det.find_and_click_unread(window_handle=1, wm=None)
    print(f"  pin 次数={hr.pin_calls}")
    assert hr.pin_calls == 1, f"点击成功时应只置顶 1 轮，实际 {hr.pin_calls}"
    print("  点击成功 -> 只置顶 1 轮")


def test_retry_pin_when_click_failed():
    """点击失败（点空/无输入方式）-> 重试置顶，至多 2 轮。这是重试的容错价值。"""
    hr = Harness(metric=(0, 0), click_results=[False, True])
    res = hr.det.find_and_click_unread(window_handle=1, wm=None)
    print(f"  pin 次数={hr.pin_calls} clicked={res['clicked']}")
    assert hr.pin_calls == 2, f"点击失败应重试第 2 轮置顶，实际 {hr.pin_calls}"
    assert res["clicked"] is True, res
    print("  点击失败 -> 重试第 2 轮并成功")


def test_normal_badge_keeps_ocr_fallback():
    """正常形态（能检出徽章）行为不变：像素判不了时照旧降级 OCR。"""
    hr = Harness(metric=(2, 100))        # 有徽章，且点击后"仍在" -> 触发降级
    res = hr.det.find_and_click_unread(window_handle=1, wm=None)
    print(f"  OCR 调用={hr.ocr_calls} pin_verify={res.get('pin_verify')}")
    assert hr.ocr_calls > 1, f"正常形态应保留 OCR 兜底，实际 {hr.ocr_calls}"
    print("  正常形态 -> 保留原 OCR 兜底路径（行为不变）")
