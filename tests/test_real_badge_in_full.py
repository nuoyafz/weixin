import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

img_path = r"C:\Users\fangzhou\.workbuddy\clipboard-images\clipboard-2026-09-04T03-31-58-301Z-49d88af4.png"
badge = cv2.imread(img_path)
print("badge shape", badge.shape)

# 拼成 1638x1263 完整窗口，把局部图贴在左上角
full = np.full((1263, 1638, 3), 215, dtype=np.uint8)  # 浅灰背景
h, w = badge.shape[:2]
full[0:h, 0:w] = badge

det = RedDotDetector()
det._debug_log = lambda msg: print("[dbg]", msg)
nav = det._scan_nav_badge(full)
print("scan_nav_badge:", nav)
