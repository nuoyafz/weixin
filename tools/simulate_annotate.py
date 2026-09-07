#!/usr/bin/env python3
"""模拟跑一遍红点扫描：把识别出的未读红点全部标注到截图上并输出。

用法: python tools/simulate_annotate.py <截图路径>
输出: <截图名>_annotated.png（红圈+未读数字+点击位置）
"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector


def main():
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        # 默认：最新一张截图
        shots = sorted(
            [f for f in os.listdir('screenshots') if f.endswith('.png')],
            key=lambda f: os.path.getmtime(os.path.join('screenshots', f)))
        src = os.path.join('screenshots', shots[-1])
    print(f'输入: {src}')

    img_bgr = np.array(Image.open(src).convert('RGB'))[:, :, ::-1].copy()
    h, w = img_bgr.shape[:2]
    print(f'尺寸: {w}x{h}')

    d = RedDotDetector()

    # 1) 导航徽章
    nav = d._scan_nav_badge(img_bgr)
    nav_info = None
    if nav:
        nav_info = (nav['center_x'], nav['center_y'], nav.get('unread_count'), nav['area'])

    # 2) 联系人红点
    cds = d._scan_contact_dots(img_bgr)
    dots_info = []
    for c in cds:
        clickable = d._dot_clickable(c)
        dots_info.append({
            'x': c['center_x'], 'y': c['center_y'],
            'w': c['w'], 'h': c['h'],
            'area': c['area'], 'unread': c['unread_count'],
            'clickable': clickable,
            'click_x': d._click_x_for_dot(c, w) if clickable else None,
        })

    print(f'\n扫描结果:')
    print(f'  nav_badge: {nav_info}')
    print(f'  contact_dots: {len(dots_info)} 个')
    for i, dd in enumerate(dots_info):
        mark = '可点' if dd['clickable'] else '--'
        print(f'    #{i+1} ({dd["x"]},{dd["y"]}) {dd["w"]}x{dd["h"]} area={dd["area"]} '
              f'unread={dd["unread"]} {mark} click_x={dd["click_x"]}')

    # 3) 标注到图上
    img_rgb = Image.open(src).convert('RGB')
    draw = ImageDraw.Draw(img_rgb)
    try:
        font_big = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
        font_small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 红圈 + 数字
    for i, dd in enumerate(dots_info):
        cx, cy = dd['x'], dd['y']
        r = max(dd['w'], dd['h']) // 2 + 4
        color = (0, 200, 0) if dd['clickable'] else (255, 165, 0)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        label = f"#{i+1} U={dd['unread'] or '?'}"
        draw.text((cx + r + 4, cy - 12), label, fill=color, font=font_small)
        if dd['clickable'] and dd['click_x']:
            # 点击位置
            cy2 = dd['y']
            draw.ellipse([dd['click_x'] - 5, cy2 - 5, dd['click_x'] + 5, cy2 + 5],
                         outline=(0, 0, 255), width=3)
            draw.text((dd['click_x'] + 8, cy2 - 10), 'click', fill=(0, 0, 255), font=font_small)

    # nav_badge 标注
    if nav_info:
        nx, ny, ncnt, narea = nav_info
        draw.ellipse([nx - 16, ny - 16, nx + 16, ny + 16], outline=(255, 0, 255), width=3)
        draw.text((nx + 20, ny - 12), f"NAV U={ncnt or '?'}", fill=(255, 0, 255), font=font_small)

    # 顶部说明
    summary = f"contacts={len(dots_info)} clickable={sum(1 for x in dots_info if x['clickable'])}"
    draw.text((10, 10), summary, fill=(0, 0, 255), font=font_big)

    out = src.replace('.png', '_annotated.png')
    img_rgb.save(out)
    print(f'\n已保存标注图: {out}')
    return out


if __name__ == '__main__':
    main()
