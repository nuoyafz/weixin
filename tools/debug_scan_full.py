#!/usr/bin/env python3
"""完整重放 _scan_contact_dots 逻辑，看 9 个点去哪了。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
from collections import Counter
from src.rpa.red_dot_detector import RedDotDetector, RedDotResult, LIST_X_START_RATIO, LIST_X_END_RATIO, LIST_TOP_SKIP_RATIO

f = 'screenshots/capture_20260901_193818_626989.png'
img = np.array(Image.open(f).convert('RGB'))[:, :, ::-1].copy()
h, w = img.shape[:2]

d = RedDotDetector()
d._avatar_col_smooth = None
print(f'_avatar_col_smooth={d._avatar_col_smooth} avatar_col_tolerance={d._avatar_col_tolerance}')

x_start = max(0, int(w * LIST_X_START_RATIO))
x_end = min(w, int(w * LIST_X_END_RATIO))
top_skip = max(0, int(h * LIST_TOP_SKIP_RATIO))
nav_end_x = max(60, int(w * 0.12))
print(f'nav_end_x={nav_end_x}  (x_min={x_start} x_max={x_end} top_skip={top_skip})')
print(f'dot_size_min={d._dot_size_min} dot_size_max={d._dot_size_max}')

contact_region = img[top_skip:h, x_start:x_end]
dots = d._find_dots(contact_region)
abs_dots = [RedDotResult(dd.x + x_start, dd.y + top_skip, dd.w, dd.h, dd.area,
                         dd.center_x + x_start, dd.center_y + top_skip) for dd in dots]
print(f'\nabs_dots 起始 {len(abs_dots)} 个：')
for dd in abs_dots:
    print(f'   ({dd.center_x},{dd.center_y}) {dd.w}x{dd.h} area={dd.area}  nav_end_x>={nav_end_x}? {dd.center_x>=nav_end_x}')

# nav 过滤后
abs_dots2 = [dd for dd in abs_dots if dd.center_x >= nav_end_x]
print(f'\nafter nav排除: {len(abs_dots2)} 个')

buckets = Counter((dd.center_x // 12) * 12 for dd in abs_dots2)
print(f'buckets: {dict(buckets)}')
if buckets:
    best_bin, best_cnt = buckets.most_common(1)[0]
    avatar_col = best_bin + 6
    avatar_confident = best_cnt >= 2
    print(f'avatar_col={avatar_col} best_cnt={best_cnt} confident={avatar_confident}')

# 检查 _read_badge_number 触发的问题
print('\n逐个 dot 测试 _read_badge_number：')
for dd in abs_dots2:
    box = {'x': dd.x, 'y': dd.y, 'w': dd.w, 'h': dd.h}
    n = d._read_badge_number(img, box)
    print(f'   ({dd.center_x},{dd.center_y}) {dd.w}x{dd.h} area={dd.area} → unread={n}')
