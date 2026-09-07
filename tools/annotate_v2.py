#!/usr/bin/env python3
"""把标注图裁切到聊天列表区，让用户看清每个红点。"""
import sys, os
sys.path.insert(0, '.')

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

src = 'screenshots/capture_20260901_193841_985322.png'
img = Image.open(src).convert('RGB')
w, h = img.size

# 1) 整图标注
img_bgr = np.array(img)[:, :, ::-1].copy()
d = RedDotDetector()
nav = d._scan_nav_badge(img_bgr)
cds = d._scan_contact_dots(img_bgr)

draw = ImageDraw.Draw(img)
try:
    f_big = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    f_small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
except Exception:
    f_big = ImageFont.load_default(); f_small = ImageFont.load_default()

# 整图标注每个红点
for i, c in enumerate(cds, 1):
    cx, cy = c['center_x'], c['center_y']
    bw, bh = c['w'], c['h']
    r = max(bw, bh) // 2 + 6
    color = (0, 200, 0) if d._dot_clickable(c) else (255, 165, 0)
    draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
    # 数字标签（黑底白字）
    label = f"#{i} U={c['unread_count'] or '?'}"
    bbox = draw.textbbox((0, 0), label, font=f_small)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = cx + r + 4; ty = cy - th // 2
    draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(0, 0, 0))
    draw.text((tx, ty), label, fill=(255, 255, 0), font=f_small)

# nav 徽章
if nav:
    cx, cy = nav['center_x'], nav['center_y']
    draw.rectangle([cx - 20, cy - 20, cx + 20, cy + 20], outline=(255, 0, 255), width=3)
    label = f"NAV U={nav.get('unread_count') or '?'}"
    draw.text((cx + 24, cy - 12), label, fill=(255, 0, 255), font=f_small)

# 顶部统计
sum_text = f"nav_badge={nav or 'None'}  contact_dots={len(cds)}  可点击={sum(1 for c in cds if d._dot_clickable(c))}"
draw.rectangle([5, 5, 5 + 18 * len(sum_text), 38], fill=(255, 255, 255))
draw.text((10, 10), sum_text, fill=(0, 0, 0), font=f_big)

img.save('screenshots/_annotated_full.png')

# 2) 裁切聊天列表区（x: 60-450，y: 100-1300）
list_crop = img.crop((60, 100, 470, 1300))
list_crop.save('screenshots/_annotated_list.png')
print('已保存 screenshots/_annotated_full.png 和 screenshots/_annotated_list.png')
print(f'\n=== 详细扫描结果 ===')
print(f'nav_badge: {nav}')
print(f'contact_dots: {len(cds)} 个')
for i, c in enumerate(cds, 1):
    print(f'  #{i:2d} c=({c["center_x"]:3d},{c["center_y"]:4d}) {c["w"]:2d}x{c["h"]:2d} '
          f'area={c["area"]:4d} unread={c["unread_count"]} '
          f'clickable={d._dot_clickable(c)} click_x={d._click_x_for_dot(c, w) if d._dot_clickable(c) else "-"}')
