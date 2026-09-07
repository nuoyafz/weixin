"""验证 RedDotDetector 在标记第一个点失败后能点到第二个点（方舟）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.rpa.red_dot_detector import RedDotDetector


def main():
    img_path = PROJECT_ROOT / "data" / "wechat" / "wait_conv_20260901_204140_819487.png"
    bgr = cv2.imread(str(img_path))
    d = RedDotDetector(config={"auto_archive": False})

    # 第一次：应该点亚磊 (y=167)
    dots = d._scan_contact_dots(bgr)
    clickable = [dd for dd in dots if d._dot_clickable(dd)]
    res1 = d._pick_and_click_contact_dot(0, clickable, bgr, 1521, None)
    print("round 1:", res1.get("contact"), "click_y=", res1.get("click_y"))
    assert res1.get("click_y") == 167, "第一轮应点亚磊 (y=167)"

    # 模拟发送失败：标记亚磊失败
    d.mark_contact_click_failed(167)

    # 第二次：应该点方舟 (y=493)
    res2 = d._pick_and_click_contact_dot(0, clickable, bgr, 1521, None)
    print("round 2:", res2.get("contact"), "click_y=", res2.get("click_y"))
    assert res2.get("click_y") == 493, "第二轮应跳过亚磊，点方舟 (y=493)"

    print("PASS: 失败标记能让后续红点被处理")


if __name__ == "__main__":
    main()
