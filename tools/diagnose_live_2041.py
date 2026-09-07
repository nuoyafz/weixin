"""诊断 20:41 真机截图的红点检测。

输出：
  - 当前代码在 819487.png 上的实际检测结果
  - 亚磊、方舟等关键区域的像素级分析
  - 被过滤掉的候选及其原因
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rpa.red_dot_detector import RedDotDetector, RED, GREEN


def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


def save_crop(bgr, x, y, w, h, path, label=""):
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    pil = pil_from_bgr(crop)
    big = pil.resize((pil.width * 8, pil.height * 8), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 4), label, fill=(255, 0, 0), font=font)
    big.save(path)
    return crop


def analyze_crop(bgr, x, y, w, h):
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return {}
    r = crop[:, :, 2].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    b = crop[:, :, 0].astype(np.float32)
    rr, gg, bb = r.mean(), g.mean(), b.mean()
    red_mask = (r >= RED["r_min"]) & (g <= RED["g_max"]) & (b <= RED["b_max"]) & ((r - g) >= RED["rg_diff"]) & ((g + b) > 0) & (r / ((g + b)) > RED["r_ratio"])
    green_mask = (g >= GREEN["g_min"]) & (r <= GREEN["r_max"]) & (b <= GREEN["b_max"]) & ((g - r) >= GREEN["gb_diff"]) & ((g - b) >= GREEN["gb_diff"]) & ((r + b) > 0) & (g / ((r + b)) > GREEN["g_ratio"])
    return {
        "shape": crop.shape[:2],
        "mean_rgb": [round(rr), round(gg), round(bb)],
        "red_pixels": int(red_mask.sum()),
        "green_pixels": int(green_mask.sum()),
    }


def find_all_candidates(detector, bgr):
    """返回 _find_dots 在联系人列表区的所有原始候选（含被后续过滤的）。"""
    import cv2
    h, w = bgr.shape[:2]
    x_start = max(0, int(w * 0.02))
    x_end = min(w, int(w * 0.24))
    top_skip = max(0, int(h * 0.10))
    region = bgr[top_skip:h, x_start:x_end]
    dots = detector._find_dots(region)
    return [
        {
            "x": d.x + x_start,
            "y": d.y + top_skip,
            "w": d.w, "h": d.h,
            "center_x": d.center_x + x_start,
            "center_y": d.center_y + top_skip,
            "area": d.area,
        }
        for d in dots
    ]


def main():
    img_path = PROJECT_ROOT / "data" / "wechat" / "wait_conv_20260901_204140_819487.png"
    out_dir = PROJECT_ROOT / "data" / "debug" / "diagnose_2041"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"截图不存在: {img_path}")
        return

    try:
        import cv2
        bgr = cv2.imread(str(img_path))
    except Exception:
        bgr = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
    h, w = bgr.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    detector = RedDotDetector(config={"data_dir": str(out_dir)})

    # 1) 当前代码的实际检测结果
    contact_dots = detector._scan_contact_dots(bgr)
    nav_badge = detector._scan_nav_badge(bgr)

    print(f"\n联系人红点数量: {len(contact_dots)}")
    for i, d in enumerate(contact_dots):
        print(f"  #{i+1}: x={d['x']} y={d['y']} w={d['w']} h={d['h']} "
              f"center=({d['center_x']},{d['center_y']}) area={d['area']} "
              f"unread={d.get('unread_count')}")
    print(f"\nNav badge: {nav_badge}")

    # 2) 所有原始候选（被过滤前）
    raw_candidates = find_all_candidates(detector, bgr)
    print(f"\n_find_dots 原始候选数量: {len(raw_candidates)}")
    for i, d in enumerate(raw_candidates):
        print(f"  raw #{i+1}: center=({d['center_x']},{d['center_y']}) area={d['area']} "
              f"w={d['w']} h={d['h']}")

    # 3) 标注图：绿色=最终检测，黄色=原始候选被过滤，蓝色=nav
    annotated = pil_from_bgr(bgr)
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for d in raw_candidates:
        # 判断是否在最终结果中
        is_final = any(abs(d["center_x"] - f["center_x"]) < 5 and abs(d["center_y"] - f["center_y"]) < 5 for f in contact_dots)
        color = (0, 255, 0) if is_final else (255, 255, 0)
        draw.rectangle([d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]], outline=color, width=2)

    for i, d in enumerate(contact_dots):
        draw.text((d["x"] + d["w"] + 2, d["y"]), f"C{i+1} U={d.get('unread_count')}", fill=(0, 255, 0), font=font)

    if nav_badge:
        nb = nav_badge
        draw.rectangle([nb["x"], nb["y"], nb["x"] + nb["w"], nb["y"] + nb["h"]], outline=(0, 0, 255), width=2)
        draw.text((nb["x"] + nb["w"] + 2, nb["y"]), f"nav U={nb.get('unread_count')}", fill=(0, 0, 255), font=font)

    annotated.save(out_dir / "annotated_live.png")

    # 4) 关键区域像素分析
    regions = [
        ("亚磊_头像右上", 143, 135),
        ("方舟_头像右上", 143, 350),
        ("Hero_头像右上", 143, 270),
        ("L_头像右上", 143, 205),
        ("nav_聊天图标", 50, 175),
    ]
    print("\n关键区域 30x30 分析:")
    region_report = []
    for label, x, y in regions:
        stats = analyze_crop(bgr, x, y, 30, 30)
        crop_path = out_dir / f"region_{label}.png"
        save_crop(bgr, x, y, 30, 30, crop_path, f"{label} {stats.get('mean_rgb')}")
        print(f"  {label} @({x},{y}) {stats}")
        region_report.append({"label": label, "x": x, "y": y, "stats": stats})

    # 5) 方舟区域详细连通块分析
    print("\n方舟区域 (120x80) 详细连通块:")
    fang_region = bgr[310:390, 120:200]
    import cv2
    r = fang_region[:, :, 2].astype(np.int16)
    g = fang_region[:, :, 1].astype(np.int16)
    b = fang_region[:, :, 0].astype(np.int16)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r.astype(np.float64) / ((g + b).astype(np.float64))
        red_mask = (r >= detector._r_min) & (g <= detector._g_max) & (b <= detector._b_max) & ((r - g) >= detector._rg_diff) & ((g + b) > 0) & (ratio > detector._r_ratio)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8), connectivity=8)
    fang_components = []
    for i in range(1, int(num)):
        sx, sy, bw, bh, area = [int(v) for v in stats[i][:5]]
        comp = (labels == i)
        mr, mg, mb = int(r[comp].mean()), int(g[comp].mean()), int(b[comp].mean())
        is_blob = detector._is_red_blob(mr, mg, mb, area, bw, bh)
        info = {
            "i": i, "x": sx + 120, "y": sy + 310,
            "w": bw, "h": bh, "area": area,
            "mean_rgb": [mr, mg, mb],
            "is_red_blob": is_blob,
        }
        fang_components.append(info)
        print(f"  comp {i}: x={sx+120} y={sy+310} w={bw} h={bh} area={area} mean=({mr},{mg},{mb}) is_blob={is_blob}")

    # 6) 保存报告
    report = {
        "image_size": [w, h],
        "contact_dots": contact_dots,
        "nav_badge": nav_badge,
        "raw_candidates": raw_candidates,
        "regions": region_report,
        "fang_components": fang_components,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n报告已保存: {out_dir / 'report.json'}")
    print(f"标注图: {out_dir / 'annotated_live.png'}")


if __name__ == "__main__":
    main()
