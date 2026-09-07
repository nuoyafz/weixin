import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

img_path = r"C:\Users\fangzhou\.workbuddy\clipboard-images\clipboard-2026-09-04T03-31-58-301Z-49d88af4.png"
img = cv2.imread(img_path)

det = RedDotDetector()
ocr = det._get_ocr()
print("ocr ready:", ocr.is_ready() if ocr else None)

def run_ocr(src, scale=10):
    if src is None:
        return None
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        res = ocr.run(big)
    except Exception as e:
        return f"ERR:{e}"
    raw = ""
    try:
        raw = "".join(str(t) for t in (getattr(res, "texts", None) or []) if str(t).strip())
    except Exception:
        pass
    if not raw:
        try:
            raw = "".join(ln.text.strip() for ln in res.iter_items() if getattr(ln, "text", ""))
        except Exception:
            pass
    return raw

# 仅裁剪红色徽章区域
x, y, w, h = 58, 98, 34, 34
crop = img[y:y+h, x:x+w]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

variants = [
    ("gray", gray),
    ("inv", cv2.bitwise_not(gray)),
    ("otsu_bin", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
    ("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
    ("thres140", cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]),
    ("thres160", cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]),
]

for name, src in variants:
    txt = run_ocr(src, 10)
    print(f"{name}: {txt!r}")

# 颜色过滤：只保留红底+白字
b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
red_mask = (r > 120) & (g < 110) & (b < 110)
white_mask = (r > 150) & (g > 150) & (b > 150)
filtered = crop.copy()
filtered[~(red_mask | white_mask)] = [0, 0, 0]
fgray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
for name, src in [("filtered_otsu", cv2.threshold(fgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                  ("filtered_140", cv2.threshold(fgray, 140, 255, cv2.THRESH_BINARY)[1]),
                  ("filtered_120", cv2.threshold(fgray, 120, 255, cv2.THRESH_BINARY)[1])]:
    txt = run_ocr(src, 10)
    print(f"{name}: {txt!r}")

# HSV 红底+白字
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
red1 = (h < 20) & (s > 60) & (v > 80)
red2 = (h > 160) & (s > 60) & (v > 80)
white = (v > 120) & (s < 60)
mask = red1 | red2 | white
masked = np.zeros_like(crop)
masked[mask] = crop[mask]
mgray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
for name, src in [("hsv_otsu", cv2.threshold(mgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                  ("hsv_140", cv2.threshold(mgray, 140, 255, cv2.THRESH_BINARY)[1])]:
    txt = run_ocr(src, 10)
    print(f"{name}: {txt!r}")
