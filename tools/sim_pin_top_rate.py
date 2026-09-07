"""模拟「双击置顶 + 点顶行」全链路，在真实列表帧上量正常率。

只验证静态坐标链路（不启动微信）：
  1) _nav_chat_icon_y 是否命中「聊天」图标（打错图标则置顶根本不触发）
  2) 双击后点顶行：先锚定搜索框底边(sb)，再在其下方找第一行会话
     —— 点击点必须严格 > sb，否则会点到搜索框。

判定：
  - pin 触发  = nav 读出未读数字
  - chat_ok   = 聊天图标 y 落在 0.08~0.18H 区间（校验线已排除通讯录 0.225H）
  - top_ok    = 顶行 y > 搜索框底边 sb
  - 该帧成功  = pin 触发 且 chat_ok 且 top_ok

用法:
  ./venv/Scripts/python.exe tools/sim_pin_top_rate.py
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.rpa.red_dot_detector import RedDotDetector

FRAME_DIR = Path("data/wechat")
FRAMES = sorted(glob.glob(str(FRAME_DIR / "auto_open_unread_*.png")))
FRAMES += [str(FRAME_DIR / "latest.png")] if (FRAME_DIR / "latest.png").exists() else []


def nav_bands(img):
    """导航列竖向分段（用于判断聊天图标 y 落在哪个图标上）。"""
    h, w = img.shape[:2]
    nav_w = max(40, int(w * 0.045))
    nav = img[:, 0:nav_w]
    bg = np.median(nav[0:30], axis=(0, 1))
    diff = np.abs(nav.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    row_cnt = (diff > 30).sum(axis=1)
    baseline = int(np.median(row_cnt[:60]))
    threshold = max(20, int(baseline * 1.6))
    segs, in_seg, s = [], False, 0
    for y in range(h):
        if row_cnt[y] > threshold and not in_seg:
            in_seg, s = True, y
        elif row_cnt[y] <= threshold and in_seg:
            in_seg = False
            if y - s >= 14:
                segs.append((s, y))
    if in_seg and h - s >= 14:
        segs.append((s, h))
    return segs


def main():
    det = RedDotDetector()
    print(f"共 {len(FRAMES)} 个列表帧\n")
    hdr = (f"{'frame':42} {'size':>10} {'nav#':>4} {'chatY':>6} {'chat?':>5} "
           f"{'sb':>4} {'topY':>6} {'top?':>5} {'src':>10} {'OK?':>4}")
    print(hdr)
    print("-" * len(hdr))

    n_total = n_nav = n_chat_ok = n_top_ok = n_ok = 0

    for fp in FRAMES:
        name = Path(fp).name
        img = cv2.imread(fp)
        if img is None:
            print(f"{name:42} 读取失败")
            continue
        h, w = img.shape[:2]
        n_total += 1

        # 1) nav 未读数字（决定走不走置顶路径）
        try:
            nb = det._scan_nav_badge(img)
            nav_count = nb.get("unread_count") if nb else None
        except Exception:
            nav_count = None

        # 2) 聊天图标 y
        chat_y = det._nav_chat_icon_y(img)
        bands = nav_bands(img)
        chat_band = None
        if chat_y is not None:
            for a, b in bands:
                if a <= chat_y <= b:
                    chat_band = f"{a}-{b}"
                    break

        # 3) 顶行：新方案 = 搜索框锚点（复刻 _click_top_conversation_row）
        sb = det._detect_search_box_bottom(img, w, h)
        if sb is not None:
            ry = det._first_avatar_below(img, w, h, sb + 2)
            if ry is not None:
                top_y, src = ry, "sb+avatar"
            else:
                top_y, src = sb + int(max(30, h * 0.032)), "sb+offset"
        else:
            fy = det._detect_first_row_center(img, w, h)
            if fy is not None:
                top_y, src = fy, "avatar"
            else:
                top_y, src = int(h * 0.12), "fallback"
        # 安全硬约束
        if sb is not None and top_y <= sb + 4:
            top_y = sb + int(max(30, h * 0.032))
            src += "+safe"

        # 判定
        chat_ok = chat_y is not None and 0.08 * h <= chat_y <= 0.18 * h
        top_ok = (sb is None) or (top_y > sb)
        pin_trigger = isinstance(nav_count, int) and nav_count > 0

        n_nav += 1 if pin_trigger else 0
        n_chat_ok += 1 if chat_ok else 0
        n_top_ok += 1 if top_ok else 0
        ok = pin_trigger and chat_ok and top_ok
        n_ok += 1 if ok else 0

        chat_s = f"{chat_y}" if chat_y is not None else "-"
        if chat_band:
            chat_s += f"[{chat_band}]"
        nav_s = str(nav_count) if nav_count is not None else "-"
        print(f"{name:42} {w}x{h:<5} {nav_s:>4} {chat_s:>6} "
              f"{'Y' if chat_ok else 'n':>5} {str(sb):>4} {top_y:>6} "
              f"{'Y' if top_ok else 'BAD':>5} {src:>10} {'Y' if ok else 'n':>4}")

    print("-" * len(hdr))
    print(f"帧总数: {n_total}")
    print(f"pin 路径触发 (nav 有未读数字): {n_nav}/{n_total}")
    print(f"聊天图标坐标 OK (落在 0.08~0.18H，排除通讯录): {n_chat_ok}/{n_total}")
    print(f"顶行坐标 OK (严格在搜索框底边下方): {n_top_ok}/{n_total}")
    if n_nav:
        print(f"置顶路径整体成功: {n_ok}/{n_nav} = {100.0*n_ok/n_nav:.1f}%")
    else:
        print("置顶路径整体成功: 无触发样本")
    print("\n注: 本模拟只验证「静态坐标链路」。双击是否真被微信识别（窗口激活/两次点击")
    print("     配对）无法离线验证，需真机 `--once` 看日志 `double_click at (...)`。")


if __name__ == "__main__":
    main()
