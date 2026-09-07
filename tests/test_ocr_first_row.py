"""双击置顶后「OCR 识别第一行联系人」单元测试。

用 fake OCR 引擎锁定行为（真实 RapidOCR 需模型、且不便构造确定文本）：
  1. OCR 命中 → 取最靠上文本框，点击坐标 = 裁剪原点 + 框中心
  2. 裁剪区必须从 sb+5 开始（搜索框整块排除，"搜索"占位符不可能被识别/点到）
  3. OCR 不可用 / 无文本 / 低分 → 返回 None，调用方回退坐标法
  4. 设计保证：OCR 结果恒在搜索框下方（裁剪区最顶端也不越过 sb）
  5. 端到端：命中时 click_x/click_y 来自 OCR，click_method=top_row_ocr，
     contact 为识别到的名字

用法:
  ./venv/Scripts/python.exe tools/test_ocr_first_row.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from src.rpa.red_dot_detector import RedDotDetector

H, W = 1213, 1637
NAV_W = max(40, int(W * 0.045))  # 与实现一致
SB = 75                          # 实测搜索框底边


class FakeOCRResult:
    def __init__(self, items):
        self._items = items

    def to_dict_list(self):
        return [{"text": t, "score": s, "box": b} for t, s, b in self._items]


class FakeOCR:
    """记录最近一次 run 的裁剪区，便于断言搜索框被排除。"""

    def __init__(self, items):
        self._items = items
        self.last_strip = None

    def run(self, strip, **kw):
        self.last_strip = strip
        return FakeOCRResult(self._items)


def box(cx, cy, w=80, h=20):
    """以 (cx,cy) 为中心的 4 点多边形（模拟 RapidOCR 的 box）。"""
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def blank():
    return np.full((H, W, 3), 255, np.uint8)


def new_det(ocr_items):
    det = RedDotDetector()
    det._nick_ocr = FakeOCR(ocr_items)
    return det


def test_ocr_hit():
    det = new_det([
        ("兰双", 0.90, box(100, 30)),    # 第一行
        ("[语音通话]", 0.80, box(100, 55)),
        ("方珂珂", 0.85, box(100, 100)),
    ])
    got = det._ocr_first_contact_row(blank(), W, H, SB)
    assert got is not None, "应识别到第一行联系人"
    name, cx, cy = got
    print(f"  OCR 命中: name={name!r} click=({cx},{cy})")
    assert name == "兰双", f"应取最靠上文本，实际 {name!r}"
    assert cx == NAV_W + 50 + 100, f"x 换算错误: {cx}"
    assert cy == SB + 5 + 30, f"y 换算错误: {cy}"


def test_crop_excludes_search_box():
    """裁剪区必须从 sb+5 起，且只覆盖列表名字区（避开聊天区）。"""
    det = new_det([("兰双", 0.9, box(100, 30))])
    det._ocr_first_contact_row(blank(), W, H, SB)
    strip = det._nick_ocr.last_strip
    exp_h = int(H * 0.97) - (SB + 5)
    exp_w = (NAV_W + 360) - (NAV_W + 50)
    print(f"  裁剪区 {strip.shape[1]}x{strip.shape[0]} "
          f"期望 {exp_w}x{exp_h} (起点 sb+5={SB+5})")
    assert strip.shape[0] == exp_h, "裁剪高度必须以 sb+5 为起点"
    assert strip.shape[1] == exp_w, "裁剪宽度必须是列表名字区"


def test_ocr_unavailable():
    img = blank()
    det = RedDotDetector()
    det._nick_ocr = False          # 哨兵：初始化失败
    assert det._ocr_first_contact_row(img, W, H, SB) is None
    print("  OCR 不可用 -> None")

    assert new_det([])._ocr_first_contact_row(img, W, H, SB) is None
    print("  无文本 -> None")

    assert new_det([("噪声", 0.2, box(100, 30))])._ocr_first_contact_row(
        img, W, H, SB) is None
    print("  低分(0.2<0.5)被过滤 -> None")


def test_skip_set_skips_truncated_filehelper():
    """回归：OCR 把"文件传输助手"切成"文件传"时，跳过集仍应生效，
    取下一个真实未读行，而不是空点内置置顶项。"""
    det = new_det([
        ("文件传", 0.92, box(100, 30)),   # 被截断的内置置顶项（首行）
        ("方政", 0.88, box(100, 60)),      # 真实未读行（应被返回）
    ])
    got = det._ocr_first_contact_row(blank(), W, H, SB)
    assert got is not None, "跳过内置项后应返回真实未读行"
    name, cx, cy = got
    print(f"  跳过集抗截断: 命中={name!r} click=({cx},{cy})")
    assert name == "方政", f"应跳过被截断的'文件传'取真实未读，实际 {name!r}"
    assert cy > SB + 4, "点击点必须严格在搜索框下方"


def test_ocr_always_below_searchbox():
    """设计保证：即使文本在裁剪区最顶端，点击点也恒在搜索框下方。"""
    det = new_det([("极端", 0.9, box(100, 0))])
    _, _, cy = det._ocr_first_contact_row(blank(), W, H, SB)
    print(f"  裁剪区最顶端文本 -> click_y={cy} (sb={SB})")
    assert cy > SB + 4, "OCR 点击点必须严格在搜索框下方"


def test_click_uses_ocr():
    """端到端：命中时点击坐标来自 OCR，而非按比例推算。"""
    det = new_det([("兰双", 0.90, box(100, 30))])
    clicked = []
    det._do_click = lambda hwnd, x, y, double=False, wm=None: \
        clicked.append((x, y)) or "ok"
    det._capture = lambda hwnd: blank()
    det._detect_search_box_bottom = lambda i, ww, hh: SB
    det._first_avatar_below = lambda i, ww, hh, fy: 108  # 坐标法兜底值

    res = det._click_top_conversation_row(12345, W, H, None,
                                          nav_num=2, img=blank())
    exp = (NAV_W + 50 + 100, SB + 5 + 30)
    print(f"  实际点击={clicked[-1]} OCR 期望={exp} "
          f"(坐标法会给 ({int(W*0.20)},108))")
    assert clicked[-1] == exp, "应优先采用 OCR 坐标"
    assert res["click_method"] == "top_row_ocr", res["click_method"]
    assert res["contact"] == "兰双", res["contact"]
    print(f"  click_method={res['click_method']} contact={res['contact']!r}")


def test_click_falls_back_without_ocr():
    """OCR 不可用 → 回退坐标法（sb + 头像带），行为与改动前一致。"""
    det = RedDotDetector()
    det._nick_ocr = False
    clicked = []
    det._do_click = lambda hwnd, x, y, double=False, wm=None: \
        clicked.append((x, y)) or "ok"
    det._capture = lambda hwnd: blank()
    det._detect_search_box_bottom = lambda i, ww, hh: SB
    det._first_avatar_below = lambda i, ww, hh, fy: 108

    res = det._click_top_conversation_row(12345, W, H, None,
                                          nav_num=2, img=blank())
    exp = (int(W * 0.20), 108)
    print(f"  实际点击={clicked[-1]} 坐标法期望={exp}")
    assert clicked[-1] == exp, "OCR 不可用时应回退坐标法"
    assert res["click_method"] == "top_row_coordinate", res["click_method"]
    print(f"  click_method={res['click_method']}（已回退）")


def main():
    print("用例1: OCR 命中取最靠上文本")
    test_ocr_hit()
    print("PASSED\n")

    print("用例2: 裁剪区排除搜索框 + 只覆盖列表名字区")
    test_crop_excludes_search_box()
    print("PASSED\n")

    print("用例3: OCR 不可用/无文本/低分 -> None")
    test_ocr_unavailable()
    print("PASSED\n")

    print("用例4: 设计保证 OCR 恒在搜索框下方")
    test_ocr_always_below_searchbox()
    print("PASSED\n")

    print("用例5: 端到端优先采用 OCR 坐标")
    test_click_uses_ocr()
    print("PASSED\n")

    print("用例6: OCR 不可用回退坐标法")
    test_click_falls_back_without_ocr()
    print("PASSED\n")

    print("用例7: 跳过集抗截断（'文件传' 跳过取真实未读）")
    test_skip_set_skips_truncated_filehelper()
    print("PASSED\n")

    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
