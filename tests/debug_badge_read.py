import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

img_path = r"C:\Users\fangzhou\.workbuddy\clipboard-images\clipboard-2026-09-04T03-31-58-301Z-49d88af4.png"
img = cv2.imread(img_path)
print("img shape", img.shape)

det = RedDotDetector()
det._debug_log = lambda msg: print("[dbg]", msg)

# 1. 完整扫描 nav badge（局部图会失败，因为不是完整窗口）
nav = det._scan_nav_badge(img)
print("nav_badge:", nav)

# 2. 直接在局部图上测 _read_badge_number，box 用红色徽章大概位置
box = {"x": 45, "y": 85, "w": 55, "h": 55}
num = det._read_badge_number(img, box)
print("read_badge_number (cropped):", num, "box:", box)

# 3. 也测试 fallback cluster OCR
badges = det._extract_badges_by_cluster_ocr(
    img, 0.0, 1.0, 0.0, 1.0,
    min_area=300, max_area=5000)
print("cluster_badges:", badges)
