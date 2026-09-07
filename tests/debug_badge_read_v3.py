import sys
sys.path.insert(0, ".")

import cv2
import numpy as np

img_path = r"C:\Users\fangzhou\.workbuddy\clipboard-images\clipboard-2026-09-04T03-31-58-301Z-49d88af4.png"
img = cv2.imread(img_path)

# 仅裁剪红色徽章区域（根据局部图估算）
x, y, w, h = 58, 98, 34, 34
x0 = max(0, x); y0 = max(0, y)
x1 = min(img.shape[1], x + w); y1 = min(img.shape[0], y + h)
crop = img[y0:y1, x0:x1]
print("crop shape", crop.shape)

def save_debug(src, name, scale=12):
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"tests/badge_v3_{name}.png", big)

# 1. 灰度
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
save_debug(gray, "gray")

# 2. HSV 提取红色区域（红底白字徽章）
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
# 红色在 HSV 中跨越 0 和 180
red_mask1 = (h < 15) & (s > 80) & (v > 80)
red_mask2 = (h > 160) & (s > 80) & (v > 80)
red_mask = red_mask1 | red_mask2
# 白字：亮度高、饱和度低
white_mask = (v > 140) & (s < 80)
combined = np.zeros_like(crop)
combined[red_mask] = crop[red_mask]
combined[white_mask] = [255, 255, 255]
c_gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
_, cbin = cv2.threshold(c_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save_debug(cbin, "red_white_otsu")

# 3. 只保留高亮（数字）
_, num_bin = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
save_debug(num_bin, "num_160")

# 4. 反相
inv = cv2.bitwise_not(gray)
save_debug(inv, "inv")

# 5. 自适应阈值
adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
save_debug(adapt, "adapt")

print("saved v3 images")
