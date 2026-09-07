#!/usr/bin/env python3
"""逐像素扫描 19:38:18 聊天列表页的红点分布。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
import cv2

f = 'screenshots/capture_20260901_193818_626989.png'
img = np.array(Image.open(f).convert('RGB'))[:, :, ::-1].copy()
h, w = img.shape[:2]
print(f'size {w}x{h}')

# 整图红色块扫描
rr = img[:, :, 2].astype(int)
gg = img[:, :, 0].astype(int)
bb = img[:, :, 1].astype(int)
red_mask = (rr >= 170) & (gg <= 165) & (bb <= 165) & ((rr - gg) >= 45) & ((rr - bb) >= 45)
num, labels, stats, cents = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8), connectivity=8)

print(f'\n=== 全图红色块（10<area<2000，排除整片彩色 banner）===')
hits = []
for i in range(1, num):
    x0, y0, bw, bh, area = stats[i][:5]
    if area < 10 or area > 2000:
        continue
    if bw < 3 or bh < 3:
        continue
    cx, cy = int(cents[i][0]), int(cents[i][1])
    hits.append((cx, cy, area, bw, bh))

# 按 y 排序
hits.sort(key=lambda t: t[1])
for cx, cy, area, bw, bh in hits:
    nav = '[NAV]' if cx < 60 else ('[列表]' if cx < 250 else '[其它]')
    print(f'  {nav} ({cx},{cy}) area={area} {bw}x{bh}')

# 检测 nav_badge 上限问题：y=185 是通讯录还是聊天？
# 聊天图标 y=199（动态定位），通讯录图标 y=285
# 185 < 199 → 实际比聊天图标略高，应该是"聊天 tab 上的徽章"
# 但用户截图说在通讯录页（标题区有"通讯录管理"）—— 矛盾！
# 看 19:38:11 通讯录页 + 19:38:18 聊天列表页 的 nav_badge 差异
print('\n=== 通讯录页 (19:38:11) vs 聊天列表页 (19:38:18) 对比 ===')
for label, ff in [('通讯录页', 'capture_20260901_193811_124443.png'),
                  ('聊天列表页', 'capture_20260901_193818_626989.png')]:
    img2 = np.array(Image.open(f'screenshots/{ff}').convert('RGB'))[:, :, ::-1].copy()
    rr2 = img2[:, :, 2].astype(int)
    gg2 = img2[:, :, 0].astype(int)
    bb2 = img2[:, :, 1].astype(int)
    red2 = (rr2 >= 170) & (gg2 <= 165) & (bb2 <= 165) & ((rr2 - gg2) >= 45) & ((rr2 - bb2) >= 45)
    n, _, st, ce = cv2.connectedComponentsWithStats(red2.astype(np.uint8), connectivity=8)
    nav_hits = []
    for i in range(1, n):
        x0, y0, bw, bh, area = st[i][:5]
        if area < 10 or area > 2000 or x0 > 60:  # 仅看导航栏
            continue
        nav_hits.append((int(ce[i][0]), int(ce[i][1]), area))
    nav_hits.sort(key=lambda t: t[1])
    print(f'{label}:')
    for cx, cy, area in nav_hits:
        yh = cy / h
        print(f'   nav 红块 ({cx},{cy}) area={area} y/h={yh:.3f}')
