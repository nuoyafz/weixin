#!/usr/bin/env python3
"""检查每个红块的实际颜色，看 _find_dots 颜色阈值为什么把它们全部漏掉。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
import cv2

f = 'screenshots/capture_20260901_193818_626989.png'
img = np.array(Image.open(f).convert('RGB'))[:, :, ::-1].copy()
h, w = img.shape[:2]

# 重新扫描整图所有红块
rr = img[:, :, 2].astype(int)
gg = img[:, :, 0].astype(int)
bb = img[:, :, 1].astype(int)
red_mask = (rr >= 170) & (gg <= 165) & (bb <= 165) & ((rr - gg) >= 45) & ((rr - bb) >= 45)
num, labels, stats, cents = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8), connectivity=8)

# 注意: rr/gg/bb 与代码里 r/g/b 在 cv2 BGR 中相同
r = rr; g = gg; b = bb

print('=== 检测器是否接受每个红块？(仿 _find_dots 流程)===')
R_MIN = 190; G_MAX = 140; B_MAX = 140; RG_DIFF = 55; R_RATIO = 1.15
MEAN_R_MIN = 170; MEAN_GB_MAX = 165; MEAN_RG_DIFF = 45
DOT_MIN, DOT_MAX = 8, 36
ASPECT_MIN, ASPECT_MAX = 0.45, 2.3
SOLIDITY_MIN = 0.40
MIN_PIXELS = 12

hits = []
for i in range(1, num):
    x0, y0, bw, bh, area = stats[i][:5]
    if not (10 < area < 5000):
        continue
    if bw < 3 or bh < 3:
        continue
    cx, cy = int(cents[i][0]), int(cents[i][1])
    hits.append((cx, cy, area, bw, bh))

hits.sort(key=lambda t: t[1])
for cx, cy, area, bw, bh in hits:
    nav = '[NAV]' if cx < 60 else ('[列表]' if cx < 250 else '[其它]')

    # 仿 _find_dots 颜色阈值
    x0 = max(0, int(cx) - bw // 2); x1 = min(w, int(cx) + bw // 2 + 1)
    y0 = max(0, int(cy) - bh // 2); y1 = min(h, int(cy) + bh // 2 + 1)
    mr = int(r[y0:y1, x0:x1].mean()); mg = int(g[y0:y1, x0:x1].mean()); mb = int(b[y0:y1, x0:x1].mean())
    rratio = mr / max(mg + mb, 1)

    pass_color = (mr >= R_MIN and mg <= G_MAX and mb <= B_MAX and (mr - mg) >= RG_DIFF and rratio > R_RATIO)
    pass_purple = (mr >= 140 and mb >= 140 and mg <= G_MAX)
    pass_green = (mg >= 180 and mr <= 140 and mb <= 140)
    pass_dimension = (bw >= DOT_MIN and bh >= DOT_MIN and bw <= DOT_MAX * 1.2 and bh <= DOT_MAX * 1.2)
    aspect = bw / max(bh, 1)
    pass_aspect = (ASPECT_MIN <= aspect <= ASPECT_MAX)
    solidity = area / max(bw * bh, 1)
    pass_solidity = (solidity >= SOLIDITY_MIN)
    pass_mean = (mr >= MEAN_R_MIN and mg <= MEAN_GB_MAX and mb <= MEAN_GB_MAX and (mr - max(mg, mb)) >= MEAN_RG_DIFF)

    pass_all = pass_color and pass_dimension and pass_aspect and pass_solidity and pass_mean
    reason = []
    if not pass_color: reason.append(f'颜色R={mr}/G={mg}/B={mb}/比={rratio:.2f}')
    if not pass_dimension: reason.append(f'尺寸{bw}x{bh}')
    if not pass_aspect: reason.append(f'比例={aspect:.2f}')
    if not pass_solidity: reason.append(f'实心={solidity:.2f}')
    if not pass_mean: reason.append(f'均值R={mr}G={mg}B={mb}')

    status = '✓' if pass_all else '✗'
    if cx < 60: continue  # 只关心列表区
    print(f'  {status} {nav} ({cx:3},{cy:4}) {bw:2}x{bh:2} a={area:3} RGB({mr:3},{mg:3},{mb:3}) {" ".join(reason)}')
