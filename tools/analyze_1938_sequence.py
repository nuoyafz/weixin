#!/usr/bin/env python3
"""分析 19:38 序列截图：找列表区为什么是空的。"""
import sys
import os
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

FILES = [
    'capture_20260901_193811_124443.png',
    'capture_20260901_193818_626989.png',
    'capture_20260901_193821_255395.png',
    'capture_20260901_193836_303061.png',
    'capture_20260901_193841_985322.png',
    'capture_20260901_193858_403913.png',
    'capture_20260901_193906_632204.png',
]

d = RedDotDetector()
ocr = d._get_ocr()

for f in FILES:
    p = os.path.join('screenshots', f)
    img = np.array(Image.open(p).convert('RGB'))[:, :, ::-1].copy()
    h, w = img.shape[:2]

    cds = d._scan_contact_dots(img)
    nb = d._scan_nav_badge(img)
    chat_y = d._nav_chat_icon_y(img)

    # 列表区密度
    list_region = img[int(h * 0.16):int(h * 0.9), int(w * 0.08):int(w * 0.20)]
    bg = np.median(list_region[0:20], axis=(0, 1))
    diff = np.abs(list_region.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    list_pct = 100 * (diff > 30).mean()

    nbcy = (nb or {}).get('center_y') or 0
    print(f'{f}')
    print(f'  nav_badge y={nbcy} y/h={nbcy/h:.3f} unread={((nb or {}).get("unread_count"))}')
    print(f'  contact_dots={len(cds)} chat_icon_y={chat_y} 列表区密度={list_pct:.1f}%')
    for c in cds:
        print(f'    dot c=({c["center_x"]},{c["center_y"]}) area={c["area"]} unread={c["unread_count"]}')

    # OCR 右上角标题判断页面
    right_top = img[int(h * 0.06):int(h * 0.16), int(w * 0.08):int(w * 0.35)]
    res = ocr.run(right_top)
    titles = []
    for ln in res.iter_items():
        t = ln.text.strip()
        if t:
            titles.append(t)
    print(f'  列表标题区 OCR: {titles[:6]}')

    # OCR 中部判断当前页（聊天列表特征是多个昵称，通讯录是字母分区）
    mid = img[int(h * 0.18):int(h * 0.50), int(w * 0.08):int(w * 0.22)]
    res2 = ocr.run(mid)
    mid_texts = []
    for ln in res2.iter_items():
        t = ln.text.strip()
        if t:
            mid_texts.append(t)
    print(f'  列表中部 OCR ({len(mid_texts)} 条): {mid_texts[:8]}')
    print()
