#!/usr/bin/env python3
"""直接跑 _scan_contact_dots 完整流程，加调试。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector, LIST_X_START_RATIO, LIST_X_END_RATIO, LIST_TOP_SKIP_RATIO

f = 'screenshots/capture_20260901_193818_626989.png'
img = np.array(Image.open(f).convert('RGB'))[:, :, ::-1].copy()
h, w = img.shape[:2]

d = RedDotDetector()
d._avatar_col_smooth = None  # 强制单帧

# 直接重放 _scan_contact_dots 关键步骤
x_start = max(0, int(w * LIST_X_START_RATIO))
x_end = min(w, int(w * LIST_X_END_RATIO))
top_skip = max(0, int(h * LIST_TOP_SKIP_RATIO))
print(f'list_region: x[{x_start},{x_end}] y[{top_skip},{h}]')
print(f'   = x[35,{x_end}] y[{top_skip},1313]')

contact_region = img[top_skip:h, x_start:x_end]
dots = d._find_dots(contact_region)
print(f'\n_find_dots(contact_region) 返回 {len(dots)} 个红块（裁剪子图坐标）：')
for dot in dots:
    cx_abs = dot.center_x + x_start
    cy_abs = dot.center_y + top_skip
    print(f'   ({dot.center_x},{dot.center_y}) -> 绝对({cx_abs},{cy_abs}) {dot.w}x{dot.h} area={dot.area}')
