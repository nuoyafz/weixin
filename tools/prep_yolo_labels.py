"""T2 数据标注引导脚本（可选，留待后期使用）。

用现有红点检测给 YOLO 训练集自动生成「种子框」，大幅减少人工标框量：
  - red_dot (class 0)      由 RedDotDetector._find_dots 的红/紫点自动生成
  - unread_badge (class 1) 由 _scan_nav_badge 的导航栏徽章生成（含 OCR，较慢）
  - avatar / nav_chat_icon / title_bar 不自动生成，需在 X-AnyLabeling 里人工补

输出 YOLO 格式： data/labels_yolo/<原图名>.txt
   每行: <class_id> <cx> <cy> <w> <h>   (均归一化 0~1)

用法：
    python tools/prep_yolo_labels.py                 # 默认扫 screenshots/ 前 200 张
    python tools/prep_yolo_labels.py --limit 500
    python tools/prep_yolo_labels.py --src screenshots/ --out data/labels_yolo

注意：种子框只为省人工，生成后务必在 X-AnyLabeling 里核对/修正，尤其
红衣头像可能被误当红点，需删除错误框、补全其余 3 类。
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rpa.red_dot_detector import RedDotDetector  # noqa: E402

# 与 yolo_locator.CLASSES 顺序一致
CLASS_RED_DOT = 0
CLASS_UNREAD_BADGE = 1


def _to_yolo(box, w, h):
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="screenshots")
    ap.add_argument("--out", default="data/labels_yolo")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--nav-badge", action="store_true",
                    help="同时用 _scan_nav_badge 生成 unread_badge 种子（较慢，含 OCR）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.png")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[prep] 未找到图片: {args.src}")
        return

    os.makedirs(args.out, exist_ok=True)
    det = RedDotDetector()
    det._debug = False

    total_red = 0
    total_badge = 0
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        # 红点种子（列表区 + 全图）
        try:
            dots = det._find_dots(img, max_size=80)
        except Exception:
            dots = []
        for d in dots:
            if d.kind in ("red", "purple"):
                box = {"x": d.x, "y": d.y, "w": d.w, "h": d.h}
                lines.append(f"{CLASS_RED_DOT} " + _to_yolo(box, w, h))
                total_red += 1
        # 导航栏徽章种子（可选，较慢）
        if args.nav_badge:
            try:
                badge = det._scan_nav_badge(img)
            except Exception:
                badge = None
            if badge and "x" in badge:
                box = {"x": badge["x"], "y": badge["y"],
                       "w": badge["w"], "h": badge["h"]}
                lines.append(f"{CLASS_UNREAD_BADGE} " + _to_yolo(box, w, h))
                total_badge += 1
        if lines:
            stem = os.path.splitext(os.path.basename(p))[0]
            with open(os.path.join(args.out, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    print(f"[prep] 处理 {len(files)} 张 -> 输出 {args.out}")
    print(f"[prep] red_dot 种子 {total_red} 个"
          + (f", unread_badge 种子 {total_badge} 个" if args.nav_badge else "")
          + "；其余 3 类请人工标注。")
    print("[prep] 下一步：用 X-AnyLabeling 打开 screenshots/ 导入 data/labels_yolo/ 修正并补全。")


if __name__ == "__main__":
    main()
