import sys
sys.path.insert(0, ".")

import cv2
import numpy as np

img_path = r"C:\Users\fangzhou\.workbuddy\clipboard-images\clipboard-2026-09-04T03-31-58-301Z-49d88af4.png"
img = cv2.imread(img_path)
print("img shape", img.shape)

x, y, w, h = 45, 85, 55, 55
pad = max(4, int(w * 0.25))
x0 = max(0, x - pad); y0 = max(0, y - pad)
x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
crop = img[y0:y1, x0:x1]
print("crop shape", crop.shape)

def save_debug(src, name, scale=8):
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"tests/badge_debug_{name}.png", big)
    print(f"saved tests/badge_debug_{name}.png size={big.shape}")

# 1. 原始灰度
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
save_debug(gray, "gray", 8)

# 2. 颜色过滤：只保留红底+白字
b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
red_mask = (r > 140) & (g < 110) & (b < 110)
white_mask = (r > 170) & (g > 170) & (b > 170)
filtered = crop.copy()
filtered[~(red_mask | white_mask)] = [0, 0, 0]
fgray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
_, fbin = cv2.threshold(fgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, fbin150 = cv2.threshold(fgray, 150, 255, cv2.THRESH_BINARY)
save_debug(fbin, "filtered_otsu", 10)
save_debug(fbin150, "filtered_150", 10)

# 3. 只保留白色
white_only = np.zeros_like(gray)
white_only[white_mask] = 255
save_debug(white_only, "white_only", 10)

# 4. 反相
inv = cv2.bitwise_not(gray)
save_debug(inv, "inv", 8)

# 5. 均衡化
gray_eq = cv2.equalizeHist(gray)
save_debug(gray_eq, "eq", 8)

# 6. 灰度二值化
_, gray_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save_debug(gray_otsu, "gray_otsu", 10)
