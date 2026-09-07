import sys
sys.path.insert(0, ".")

import cv2
from src.rpa.red_dot_detector import RedDotDetector

img_path = r"C:\VisionLeadAgent-wx\my_agent\tests\synthetic_badge.png"
img = cv2.imread(img_path)
det = RedDotDetector()
det._debug_log = lambda msg: print("[dbg]", msg)

nav = det._scan_nav_badge(img)
print("scan_nav_badge:", nav)
