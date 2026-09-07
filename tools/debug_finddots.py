#!/usr/bin/env python3
"""单独测 _find_dots 与 _is_red_blob，看 (200,175) 这种真实红点为啥被过滤。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
import cv2
from src.rpa.red_dot_detector import RedDotDetector

f = 'screenshots/capture_20260901_193818_626989.png'
img = np.array(Image.open(f).convert('RGB'))[:, :, ::-1].copy()
h, w = img.shape[:2]

d = RedDotDetector()
print(f'r_min={d._r_min} g_max={d._g_max} b_max={d._b_max} rg_diff={d._rg_diff} r_ratio={d._r_ratio}')
print(f'dot_min={d._dot_size_min} dot_max={d._dot_size_max} solidity_min={d._solidity_min}')
print(f'min_pixels={d._min_pixels}')

# 直接调用 _find_dots
dots = d._find_dots(img)
print(f'\n_find_dots 全图返回 {len(dots)} 个红块：')
for dot in dots:
    print(f'  ({dot.center_x},{dot.center_y}) {dot.w}x{dot.h} area={dot.area}')

# 直接看 (200, 175) 位置的颜色，并扫一行 rmask
r = img[:, :, 2].astype(int); g = img[:, :, 1].astype(int); b = img[:, :, 0].astype(int)
rmask = (r >= d._r_min) & (g <= d._g_max) & (b <= d._b_max) & ((r - g) >= d._rg_diff) & (r / (g + b + 1) > d._r_ratio)
print(f'\n总 rmask 像素数: {rmask.sum()}')

# 看 (200, 175) 区域是否进入 rmask
y0, y1 = 165, 195; x0, x1 = 180, 220
print(f'({x0},{y0})-({x1},{y1}) 区域 rmask 像素: {rmask[y0:y1, x0:x1].sum()} / 面积 {(y1-y0)*(x1-x0)}')

# 输出 (200, 175) 邻域内的真实颜色
for cx, cy in [(200, 175), (199, 743), (200, 630), (78, 122), (409, 537), (167, 661), (181, 675), (186, 1102)]:
    bc = img[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
    if bc.size == 0: continue
    R = bc[:,:,2].mean(); G = bc[:,:,1].mean(); B = bc[:,:,0].mean()
    in_mask = rmask[max(0,cy-5):cy+5, max(0,cx-5):cx+5].sum()
    print(f'  ({cx},{cy}) 邻域均值 RGB=({R:.0f},{G:.0f},{B:.0f}) r/(g+b)={R/(G+B+1):.2f} rmask={in_mask}')
