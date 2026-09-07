"""离线诊断 19:38 截图的红点检测与 nav badge OCR。

输出：
  - data/debug/diagnose_1938/ 下每个检测候选的裁剪图与 RGB 统计
  - nav badge 多路 OCR 结果
  - 用户指定的 #1~#6 区域像素级分析
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

from src.rpa.red_dot_detector import RedDotDetector


def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


def save_crop(bgr: np.ndarray, x: int, y: int, w: int, h: int, path: Path, label: str = ""):
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    crop = bgr[y0:y1, x0:x1]
    pil = pil_from_bgr(crop)
    # 放大 8 倍便于查看
    big = pil.resize((pil.width * 8, pil.height * 8), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 4), label, fill=(255, 0, 0), font=font)
    big.save(path)
    return crop


def analyze_crop(bgr: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return {}
    r = crop[:, :, 2].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    b = crop[:, :, 0].astype(np.float32)
    rr = r.mean(); gg = g.mean(); bb = b.mean()
    red_mask = (r >= 190) & (g <= 165) & (b <= 165) & ((r - g) >= 55) & ((g + b) > 0) & (r / ((g + b)) > 0.90)
    green_mask = (g >= 180) & (r <= 140) & (b <= 140) & ((g - r) >= 50) & ((g - b) >= 50) & ((r + b) > 0) & (g / ((r + b)) > 1.15)
    return {
        "shape": crop.shape[:2],
        "mean_rgb": [round(rr), round(gg), round(bb)],
        "red_pixels": int(red_mask.sum()),
        "green_pixels": int(green_mask.sum()),
    }


def ocr_variants(detector: RedDotDetector, bgr: np.ndarray, box: dict) -> list[dict]:
    try:
        import cv2
    except Exception:
        return []
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    pad = max(8, int(w * 0.6))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ocr = detector._get_ocr()
    results = []
    variants = [
        ("gray6", gray, 6, False),
        ("gray8", gray, 8, False),
        ("inv140_6", cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1], 6, False),
        ("otsu_inv_8", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 8, False),
        ("otsu_8", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 8, False),
    ]
    try:
        eq = cv2.equalizeHist(gray)
        variants.append(("eq_otsu_inv_8", cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 8, False))
    except Exception:
        pass
    # 额外：彩色原图放大，看 RapidOCR 是否能直接读白字
    variants.append(("color6", crop, 6, True))
    variants.append(("color8", crop, 8, True))

    for name, src, scale, color in variants:
        if color:
            big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        try:
            res = ocr.run(big)
            texts = []
            try:
                texts = [str(t).strip() for t in (getattr(res, "texts", None) or []) if str(t).strip()]
            except Exception:
                pass
            if not texts:
                try:
                    texts = [ln.text.strip() for ln in res.iter_items() if getattr(ln, "text", "").strip()]
                except Exception:
                    pass
            results.append({"name": name, "texts": texts})
        except Exception as e:
            results.append({"name": name, "texts": [], "error": str(e)})
    return results


def main():
    img_path = PROJECT_ROOT / "screenshots" / "capture_20260901_193841_985322.png"
    out_dir = PROJECT_ROOT / "data" / "debug" / "diagnose_1938"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"截图不存在: {img_path}")
        return

    bgr = cv2.imread(str(img_path)) if 'cv2' in globals() else np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
    try:
        import cv2
        bgr = cv2.imread(str(img_path))
    except Exception:
        bgr = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
    h, w = bgr.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    detector = RedDotDetector(config={"data_dir": str(out_dir)})

    # 1) 运行完整检测
    contact_dots = detector._scan_contact_dots(bgr)
    nav_badge = detector._scan_nav_badge(bgr)

    print(f"\n联系人红点数量: {len(contact_dots)}")
    for i, d in enumerate(contact_dots):
        print(f"  #{i+1}: x={d['x']} y={d['y']} w={d['w']} h={d['h']} "
              f"center=({d['center_x']},{d['center_y']}) area={d['area']} "
              f"unread={d.get('unread_count')}")
    print(f"\nNav badge: {nav_badge}")

    # 2) 保存每个候选的裁剪图与统计
    report = {"image_size": [w, h], "contact_dots": [], "nav_badge": None}
    annotated = pil_from_bgr(bgr)
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, d in enumerate(contact_dots):
        crop_path = out_dir / f"contact_{i+1}_{d['center_y']}.png"
        stats = analyze_crop(bgr, d["x"] - 2, d["y"] - 2, d["w"] + 4, d["h"] + 4)
        save_crop(bgr, d["x"] - 2, d["y"] - 2, d["w"] + 4, d["h"] + 4,
                  crop_path, f"#{i+1} U={d.get('unread_count')} {stats.get('mean_rgb')}")
        d_report = dict(d)
        d_report["stats"] = stats
        d_report["crop"] = str(crop_path)
        report["contact_dots"].append(d_report)
        # 画框到标注图
        draw.rectangle([d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]], outline=(0, 255, 0), width=2)
        draw.text((d["x"] + d["w"] + 2, d["y"]), f"#{i+1} U={d.get('unread_count')}", fill=(255, 0, 0), font=font)

    # 3) nav badge 详细分析
    if nav_badge:
        nb = nav_badge
        crop_path = out_dir / "nav_badge_crop.png"
        save_crop(bgr, nb["x"], nb["y"], nb["w"], nb["h"], crop_path,
                  f"nav U={nb.get('unread_count')}")
        ocr_results = ocr_variants(detector, bgr, nb)
        print("\nNav badge OCR 多路结果:")
        for r in ocr_results:
            print(f"  {r['name']}: {r.get('texts')}")
        report["nav_badge"] = {
            **nb,
            "crop": str(crop_path),
            "ocr_variants": ocr_results,
        }
        draw.rectangle([nb["x"], nb["y"], nb["x"] + nb["w"], nb["y"] + nb["h"]], outline=(0, 0, 255), width=2)
        draw.text((nb["x"] + nb["w"] + 2, nb["y"]), "nav", fill=(0, 0, 255), font=font)

    annotated.save(out_dir / "annotated_diagnose.png")

    # 4) 用户质疑区域：#1~#6 头像右上角 30x30 像素
    user_regions = [
        ("#1_服务号", 143, 106),
        ("#2_方舟", 143, 190),
        ("#3_折叠的聊天", 143, 350),
        ("#4_方政", 143, 415),
        ("#5_粉色头像", 143, 678),
        ("#6_Hero", 143, 748),
    ]
    print("\n用户指定区域 30x30 分析（头像右上角）:")
    for label, x, y in user_regions:
        stats = analyze_crop(bgr, x, y, 30, 30)
        crop_path = out_dir / f"user_{label}.png"
        save_crop(bgr, x, y, 30, 30, crop_path, f"{label} {stats.get('mean_rgb')} R={stats.get('red_pixels')}")
        print(f"  {label} @({x},{y}) {stats}")

    # 5) 保存报告
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n报告已保存: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
