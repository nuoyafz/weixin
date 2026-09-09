"""nav_anchors.py —— 微信左侧导航栏 landmark 锚点（方案 A：检测优先 + 固定偏移兜底）。

核心认知（修根因，不是治标）：
    微信左侧导航栏（头像 / 聊天 / 通讯录 / 朋友圈）的 landmark 是**固定像素偏移**，
    **不随窗口高度线性缩放**。因此 `h * 0.152` 这类比例只在它被标定的那一个窗口高度
    上成立，换台机器默认窗口大小不同、h 一变就飘（点进头像 / 点进通讯录）。

    标定数据（微信 4.0 @1750×1313，h=1313）：
        头像段中心 ≈ 108px，聊天图标中心 ≈ 200px(=0.152H 仅此时)，
        通讯录中心 ≈ 285px，朋友圈中心 ≈ 367px，首行聊天项 ≈ 113px。
    这些全是像素常数，比例只是它们被反推出来的「假象」。

策略（方案 A，独立模块，只依赖 numpy）：
    1. 几何检测优先：段检测切出左侧导航栏各图标段，聊天图标 = segs[1] 中心、
       通讯录 = segs[2] 中心（直接定位，彻底不碰比例）；
    2. 检测不足时回退「头像段中心 + 固定像素间距(≈92px)」；
    3. 再回退固定常量（基于多窗口实测标定）；
    4. 任一结果都做**绝对像素 clamp**（不是比例 clamp），超界回退下一级。

对外返回 (y, source) 元组，source ∈ {seg1, seg2, avatar+gap, chat+gap, const}，
调用方负责打日志 —— 这样真机能直接看到走了哪条路径、算出什么坐标，便于校准。

注意：本模块不做任何点击，只算坐标。点击派发层（_background_click 等）保持原样。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 标定常量（像素，基于微信 4.0 @1750×1313 多窗口实测；窗口高度变化时基本不变）
# ---------------------------------------------------------------------------
NAV_AVATAR_Y_CENTER = 108      # 头像段中心
NAV_CHAT_Y = 200              # 聊天图标中心（原 0.152H 的标定真值）
NAV_CONTACTS_Y = 285          # 通讯录中心
NAV_FIRST_ROW_Y = 113         # 聊天列表首行 y（搜索框底边≈104 + 间隔）

NAV_GAP_AVATAR_CHAT = 92      # 头像段中心 → 聊天图标中心 固定间距
NAV_GAP_CHAT_CONTACTS = 85    # 聊天图标中心 → 通讯录中心 固定间距

NAV_CLICK_X = 30              # 导航图标点击 x（左侧内一点，原 w*0.019≈26~34 的等效固定值）

# 绝对像素合理区间（不随 h 变；窗口再大，导航图标也不会跑到 260px 以下/以上）
CHAT_Y_MIN, CHAT_Y_MAX = 120, 260
CONTACTS_Y_MIN, CONTACTS_Y_MAX = 210, 360
FIRST_ROW_Y_MIN, FIRST_ROW_Y_MAX = 70, 420


def _seg_center(seg: Tuple[int, int]) -> int:
    return (seg[0] + seg[1]) // 2


def detect_nav_segments(img: np.ndarray) -> List[Tuple[int, int]]:
    """切出左侧导航栏内的图标「段」[(y0, y1), ...]，从上到下依次排列。

    段检测只在左侧窄带（导航图标所在列）做行像素差异统计，与窗口宽度/高度无关
    （采样带宽度随 w 轻微缩放，但有 40px 下限兜底）。segs[0]=头像、[1]=聊天、
    [2]=通讯录、[3]=朋友圈……。检测失败/无段返回空列表。
    """
    try:
        h, w = img.shape[:2]
        nav_w = max(40, int(w * 0.045))
        nav = img[:, 0:nav_w]
        bg = np.median(nav[0:30], axis=(0, 1))
        diff = np.abs(nav.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
        row_cnt = (diff > 30).sum(axis=1)
        baseline = int(np.median(row_cnt[:60]))
        threshold = max(20, int(baseline * 1.6))
        segs: List[Tuple[int, int]] = []
        in_seg = False
        s = 0
        for y in range(h):
            if row_cnt[y] > threshold and not in_seg:
                in_seg = True
                s = y
            elif row_cnt[y] <= threshold and in_seg:
                in_seg = False
                if y - s >= 14:
                    segs.append((s, y))
        if in_seg and h - s >= 14:
            segs.append((s, h))
        return segs
    except Exception:
        return []


def chat_icon_y(img: np.ndarray, h: int) -> Tuple[int, str]:
    """返回 (聊天图标 y 中心, 来源)。来源用于日志，便于真机校准。"""
    segs = detect_nav_segments(img)
    # 主路径：直接取聊天图标段（segs[1]）中心 —— 完全不依赖比例
    if len(segs) >= 2:
        c = _seg_center(segs[1])
        if CHAT_Y_MIN <= c <= CHAT_Y_MAX:
            return c, "seg1"
    # 回退 1：头像段中心 + 固定间距
    if segs:
        c = _seg_center(segs[0]) + NAV_GAP_AVATAR_CHAT
        if CHAT_Y_MIN <= c <= CHAT_Y_MAX:
            return c, "avatar+gap"
    # 回退 2：固定常量（常量本身已在合理区间内）
    return NAV_CHAT_Y, "const"


def contacts_y(img: np.ndarray, h: int) -> Tuple[int, str]:
    """返回 (通讯录 y 中心, 来源)。"""
    segs = detect_nav_segments(img)
    if len(segs) >= 3:
        c = _seg_center(segs[2])
        if CONTACTS_Y_MIN <= c <= CONTACTS_Y_MAX:
            return c, "seg2"
    if len(segs) >= 2:
        c = _seg_center(segs[1]) + NAV_GAP_CHAT_CONTACTS
        if CONTACTS_Y_MIN <= c <= CONTACTS_Y_MAX:
            return c, "chat+gap"
    return NAV_CONTACTS_Y, "const"


def first_row_y(h: int, search_box_bottom: Optional[int] = None) -> int:
    """返回聊天列表首行 y。search_box_bottom 已知时用它 + 固定间隔，否则用常量。

    首行随窗口高度变化范围比导航图标大（聊天列表区随窗口拉伸），但仍远低于比例
    假定；这里用绝对像素夹取 [FIRST_ROW_Y_MIN, max(FIRST_ROW_Y_MAX, 0.25h)]。
    """
    base = (search_box_bottom + 9) if search_box_bottom else NAV_FIRST_ROW_Y
    lo = FIRST_ROW_Y_MIN
    hi = max(FIRST_ROW_Y_MAX, int(0.25 * h))
    return int(max(lo, min(hi, base)))


def avatar_bottom_y(img: np.ndarray) -> Optional[int]:
    """返回头像段底边 y（复用段检测）；检测失败返回 None。"""
    segs = detect_nav_segments(img)
    if segs:
        return segs[0][1]
    return None


def nav_click_x(w: int = 0) -> int:
    """导航图标点击 x。导航栏图标靠左、宽度固定，x 不随窗口宽度缩放，用固定常量。"""
    return NAV_CLICK_X


__all__ = [
    "NAV_AVATAR_Y_CENTER", "NAV_CHAT_Y", "NAV_CONTACTS_Y", "NAV_FIRST_ROW_Y",
    "NAV_GAP_AVATAR_CHAT", "NAV_GAP_CHAT_CONTACTS", "NAV_CLICK_X",
    "CHAT_Y_MIN", "CHAT_Y_MAX", "CONTACTS_Y_MIN", "CONTACTS_Y_MAX",
    "FIRST_ROW_Y_MIN", "FIRST_ROW_Y_MAX",
    "detect_nav_segments", "chat_icon_y", "contacts_y",
    "first_row_y", "avatar_bottom_y", "nav_click_x",
]
