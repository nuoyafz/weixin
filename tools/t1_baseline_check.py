# -*- coding: utf-8 -*-
"""T1 数字OCR 改造的前后对比基准：对真实帧跑 _scan_nav_badge，记录识别结果与耗时。"""
import sys
import time
import cv2

sys.path.insert(0, r"C:\VisionLeadAgent-wx\my_agent")

from src.rpa.red_dot_detector import RedDotDetector

img = cv2.imread(r"C:\VisionLeadAgent-wx\my_agent\_preview\last.png")
if img is None:
    print("FATAL: cannot read frame")
    sys.exit(1)
h, w = img.shape[:2]
print(f"frame: {w}x{h}")

det = RedDotDetector()
det._debug = True

t0 = time.time()
nb = det._scan_nav_badge(img)
dt = time.time() - t0
print(f"[baseline] nav_badge={nb}")
print(f"[baseline] elapsed={dt:.2f}s")
