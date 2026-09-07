"""视图闸门（ViewGate）—— 判定截图是「会话视图」还是「会话列表视图」。

这是感知准确率的第一道闸门（P0）。

背景（实测）：
    此前感知层把「会话列表视图」当「会话视图」解析，导致
    ``current_contact='Q搜索'``、8 条"消息"全部是侧边栏的列表项预览
    （联系人名 + 日期 + 消息预览），进而得出"有客户消息、应回复"的错误结论。

根因定位：原项目四级感知瀑布（L1 像素 → L2 布局 → L3 OCR → L4 VLM）中，
「当前在看什么」应由 **L2 布局层** 回答，而旧实现跳过了 L2，直接让 L3 OCR
在全图上裸跑 —— OCR 本身没错，错在让它去读了不该读的区域。

本模块即补上 L2 的第一问：先判视图，再决定要不要解析消息。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# 列表项里常见的时间戳模式（列表视图强特征）
# 增加 HH:MM，因为列表视图右侧常见 "19:22" 这类时间戳，不可当成聊天内容。
_DATE_RE = re.compile(
    r"(\d{1,2}/\d{1,2}|\d{4}/\d{1,2}/\d{1,2}|昨天|前天|星期[一二三四五六日天]|\d{1,2}:\d{2})"
)


@dataclass
class ViewVerdict:
    """视图判定结果。"""
    view: str = "unknown"                 # conversation | list | unknown
    confidence: float = 0.0               # 0.0 ~ 1.0
    evidence: dict = field(default_factory=dict)

    @property
    def is_conversation(self) -> bool:
        return self.view == "conversation"

    @property
    def is_list(self) -> bool:
        return self.view == "list"

    @property
    def is_unknown(self) -> bool:
        return self.view == "unknown"


class ViewGate:
    """判定当前截图属于哪种视图。

    四证据投票：
      A 绿色气泡占比  —— 微信「自己发的消息」气泡为绿色系，列表视图几乎不存在
      B 底部输入区    —— 会话视图底部有输入框（分隔线 + 无文字行）
      C 列表项模式    —— 多行左边界聚集在同一窄带 且 文本含日期
      D 搜索框占位符  —— 列表视图顶部有「搜索」
    """

    # 底部输入区搜索带（占图高比例）
    BOTTOM_BAND_RATIO = 0.18
    # 绿色气泡判定阈值（自己发的气泡 #95EC69 一类）
    GREEN_DOMINANCE = 20
    GREEN_MIN = 110
    GREEN_RATIO_CONV = 0.012           # 超过该占比 -> 认为存在自己的气泡
                                        # （列表视图头像/状态点也含少量绿色，旧阈值 0.0025
                                        #  会把列表误判成会话，已上调到 0.012）
    # 统计气泡色时排除的左侧比例（避开侧边栏头像/图标的绿色干扰）
    SIDEBAR_SKIP_RATIO = 0.30

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._bottom_band = float(
            self.config.get("bottom_band_ratio", self.BOTTOM_BAND_RATIO))
        self._green_ratio_conv = float(
            self.config.get("green_ratio_conv", self.GREEN_RATIO_CONV))

    # ------------------------------------------------------------------
    def judge(self, image: Any,
              ocr_lines: Optional[Sequence[Any]] = None) -> ViewVerdict:
        """判定视图。ocr_lines 为鸭子类型（需 x_min/text 等属性）。"""
        try:
            arr = self._as_bgr(image)
        except Exception:
            return ViewVerdict("unknown", 0.0, {"error": "bad_image"})
        if arr is None or arr.size == 0:
            return ViewVerdict("unknown", 0.0, {"error": "empty_image"})

        h, w = arr.shape[:2]
        lines = list(ocr_lines or [])
        ev: dict[str, Any] = {}
        conv = 0.0
        listv = 0.0

        # 证据 A：绿色气泡（自己发的消息）
        # 权重从 1.0 降到 0.5：绿色只作辅助信号，避免列表视图里偶发绿色像素
        # （头像/状态点）仅凭一个绿色投票就把视图翻成 conversation。
        green = self._green_bubble_ratio(arr)
        ev["green_ratio"] = round(green, 5)
        if green >= self._green_ratio_conv:
            conv += 0.5
        elif green < self._green_ratio_conv * 0.3:
            listv += 0.4

        # 证据 B：底部输入区
        # 注意：实测该判据**不可靠** —— 列表视图与会话视图的底部带在
        # 分隔线强度/深色占比/边缘密度上完全重叠（实测 LIST dark 0.0066~0.0337
        # vs CONV 0.0266），无法干净区分，反而会把列表误判成会话。
        # 故此处仅记录供排查，**不参与投票**。
        has_input = self._bottom_is_input(arr, lines, h)
        ev["has_input_area"] = bool(has_input)

        # 证据 C：列表项模式（需 OCR 行，最可靠的列表特征）
        list_score = self._list_row_pattern(lines, w, h)
        ev["list_row_score"] = round(list_score, 3)
        if list_score >= 0.55:
            listv += 1.0
        elif list_score <= 0.20 and len(lines) >= 3:
            # 仅在有足够 OCR 行时才把"没有列表特征"当作会话的正面证据；
            # 没有 OCR 行时这是"无数据"，不能推断为会话。
            conv += 0.35

        # 证据 E：右侧聊天区是否存在消息内容（分栏模式下的关键判据）
        has_chat = self._has_chat_content(lines, w, h)
        ev["has_chat_content"] = bool(has_chat)
        if has_chat:
            # 右侧有聊天内容 -> 强 conversation 证据，同时抑制左侧列表误判
            conv += 1.0
            listv = max(0.0, listv - 0.6)

        # 证据 D：搜索框占位符
        has_search = self._has_search_placeholder(lines)
        ev["has_search_placeholder"] = bool(has_search)
        if has_search:
            listv += 0.6

        # 证据 F：右侧会话标题栏（分栏模式下的最强会话信号）
        # 微信分栏模式下左侧联系人列表「永远存在」，导致证据 C/D 恒为真、
        # 永远给 list 加分；但只要右侧顶部出现了联系人名（会话标题），就说明
        # 当前已进入会话。该信号独立于「列表项」存在，可强力翻盘。
        has_title = self._has_conversation_title(lines, w, h)
        ev["has_conversation_title"] = bool(has_title)
        if has_title:
            conv += 0.8
            listv = max(0.0, listv - 0.6)

        # 证据 G：右侧聊天区媒体/气泡内容（表情包会话也能识别）
        # 纯文本判据 E 在「对方只发表情包/图片」时失效（OCR 读不到文字），
        # 此时用图像像素判断右侧是否有成簇的消息块（气泡/表情/图片）。
        # 注意：G 仅作为「确认」证据——只有 F 或 E 已触发（right_conv）时才
        # 计入，避免纯列表视图右侧占位图偶发误触发把视图翻成 conversation。
        has_media = self._right_panel_has_media(arr, w, h)
        ev["has_right_media"] = bool(has_media)
        if has_media and (has_title or has_chat):
            conv += 0.7
            listv = max(0.0, listv - 0.5)

        # 分栏模式抑制：右侧已确认存在会话内容时，左侧列表的「恒在特征」
        # （C/D）不应再主导判定，把其 list 权重压到 20%。
        if has_title or has_chat or has_media:
            listv = max(0.0, listv * 0.20)

        total = conv + listv
        ev["score"] = {"conversation": round(conv, 3), "list": round(listv, 3)}
        if total <= 0:
            return ViewVerdict("unknown", 0.0, ev)
        if conv > listv:
            margin = (conv - listv) / total
            return ViewVerdict("conversation", round(min(0.99, 0.5 + margin * 0.5), 3), ev)
        if listv > conv:
            margin = (listv - conv) / total
            return ViewVerdict("list", round(min(0.99, 0.5 + margin * 0.5), 3), ev)
        return ViewVerdict("unknown", 0.4, ev)

    # ------------------------------------------------------------------
    # 证据 A：绿色气泡占比
    # ------------------------------------------------------------------
    def _green_bubble_ratio(self, arr: np.ndarray) -> float:
        try:
            h, w = arr.shape[:2]
            x0 = int(w * self.SIDEBAR_SKIP_RATIO)
            roi = arr[:, max(0, x0):]
            if roi.size == 0:
                return 0.0
            b = roi[:, :, 0].astype(np.int16)
            g = roi[:, :, 1].astype(np.int16)
            r = roi[:, :, 2].astype(np.int16)
            mask = ((g > r + self.GREEN_DOMINANCE)
                    & (g > b + self.GREEN_DOMINANCE)
                    & (g > self.GREEN_MIN))
            return float(mask.mean())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 证据 B：底部是否为输入区（会话视图特征）
    # ------------------------------------------------------------------
    def _bottom_is_input(self, arr: np.ndarray, lines: list, h: int) -> bool:
        """会话视图底部是输入框：有分隔线 且 带内无文字行。

        列表视图底部仍是列表项（有文字行一直延伸到底部）。
        """
        try:
            y0 = int(h * (1.0 - self._bottom_band))
            band = arr[y0:h]
            if band.size == 0 or band.shape[0] < 8:
                return False

            has_separator = self._has_horizontal_separator(band)
            # 底部带内是否有文字行（列表视图会有）
            rows_in_band = 0
            for ln in lines:
                cy = float(getattr(ln, "cy", -1))
                if cy >= y0:
                    rows_in_band += 1
            if has_separator and rows_in_band == 0:
                return True
            # 仅有分隔线、无文字也倾向输入区
            if has_separator and rows_in_band <= 1:
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _has_horizontal_separator(band: np.ndarray) -> bool:
        """底部带内是否存在水平分隔线（行均值突变）。"""
        try:
            gray = band.mean(axis=2).astype(np.float32)
            row_mean = gray.mean(axis=1)
            if row_mean.size < 4:
                return False
            diff = np.abs(np.diff(row_mean))
            peak = float(diff.max())
            base = float(np.median(diff)) + 1e-6
            return bool(peak > max(2.5, base * 3.5))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 证据 C：列表项模式
    # ------------------------------------------------------------------
    def _list_row_pattern(self, lines: list, w: int, h: int) -> float:
        """多行左边界聚集在同一窄带 且 文本含日期 -> 列表视图特征。

        注意：微信桌面版常见「分栏」模式，左侧联系人列表与右侧会话同时存在。
        因此只统计严格落在左侧列表区（x_min 在 0.05w~0.28w）的行，避免把
        右侧聊天区的时间戳误当成列表项。
        """
        if len(lines) < 3:
            return 0.0
        try:
            cand = [ln for ln in lines
                    if w * 0.05 < float(getattr(ln, "x_min", 0.0)) < w * 0.28]
            if len(cand) < 3:
                return 0.0
            xs = np.array([float(getattr(ln, "x_min", 0.0)) for ln in cand])
            spread = float(xs.std())
            if spread < 12:
                align = 1.0
            elif spread < 25:
                align = 0.5
            else:
                align = 0.0
            texts = [str(getattr(ln, "text", "")) for ln in cand]
            hit = sum(1 for t in texts if _DATE_RE.search(t))
            date_rate = hit / max(len(texts), 1)
            return float(min(1.0, align * 0.5 + min(date_rate * 2.0, 1.0) * 0.5))
        except Exception:
            return 0.0

    def _has_chat_content(self, lines: list, w: int, h: int) -> bool:
        """右侧聊天区是否存在消息内容（conversation 的强证据）。

        微信分栏模式下，左侧始终有联系人列表；判断当前是否已进入会话，
        应看右侧聊天区是否有成簇的消息文本。
        """
        if len(lines) < 2 or w <= 0 or h <= 0:
            return False
        try:
            chat_lines = [ln for ln in lines
                          if float(getattr(ln, "x_min", 0.0)) >= w * 0.30
                          and float(getattr(ln, "y_min", 0.0)) >= h * 0.10
                          and float(getattr(ln, "y_max", 0.0)) <= h * 0.88
                          and not _DATE_RE.search(str(getattr(ln, "text", "")))]
            if len(chat_lines) < 2:
                return False
            ys = [float(getattr(ln, "cy",
                                  (getattr(ln, "y_min", 0.0) + getattr(ln, "y_max", 0.0)) / 2.0))
                  for ln in chat_lines]
            return (max(ys) - min(ys)) >= h * 0.20
        except Exception:
            return False

    @staticmethod
    def _has_conversation_title(lines: list, w: int, h: int) -> bool:
        """右侧顶部（标题栏）存在联系人名 → 已进入会话的强信号。

        分栏模式下左侧列表恒在，但会话标题只在进入会话后出现于右侧顶部
        （y < 0.15h 且 x >= 0.30w）。列表项名恒落在 x∈[0.05w,0.28w]，
        不会进入该区域，故不会误触发。
        """
        _BUILTIN = ("搜索", "选择一个会话", "微信", "文件传输", "订阅号",
                    "微信团队", "服务通知", "微信支付", "收藏", "微信表情", "通讯录")
        for ln in lines:
            x = float(getattr(ln, "x_min", -1))
            y = float(getattr(ln, "y_min", -1))
            t = str(getattr(ln, "text", "")).strip()
            if not t or x < 0 or y < 0:
                continue
            if x >= w * 0.30 and y < h * 0.15 and not _DATE_RE.search(t):
                if t in _BUILTIN or t.endswith("搜索") or t.endswith("会话"):
                    continue
                # 非内置、非日期、非纯数字 -> 视为联系人名
                if not re.fullmatch(r"\d+", t):
                    return True
        return False

    def _right_panel_has_media(self, arr: np.ndarray, w: int, h: int) -> bool:
        """右侧聊天区(x>=0.35w, 排除标题/输入区)是否存在成簇的消息内容。

        表情包/图片会话中 OCR 读不到文字，纯文本判据 E 失效；改用像素特征：
        右侧非背景（近白/浅灰）像素占比 + 垂直投影的「内容段」数。
        多条消息/气泡 => 多个垂直段 => 识别为会话内容。
        """
        x0 = int(w * 0.35)
        y0 = int(h * 0.15)
        y1 = int(h * 0.88)
        if x0 >= arr.shape[1] or y1 <= y0:
            return False
        roi = arr[y0:y1, x0:]
        if roi.size == 0:
            return False
        # 微信聊天背景约 (245,245,245)；非背景 = 任一通道与 245 偏差 > 35
        diff = np.abs(roi.astype(np.int16) - 245).max(axis=2)
        nonbg_ratio = float((diff > 35).mean())
        if nonbg_ratio < 0.04:
            return False
        # 垂直投影：统计「有内容的连续段」数量（气泡/表情间隔形成多段）
        row_ratio = (diff > 35).mean(axis=1)
        seg = 0
        in_seg = False
        for v in row_ratio:
            if v > 0.02:
                if not in_seg:
                    seg += 1
                    in_seg = True
            else:
                in_seg = False
        return seg >= 2

    # ------------------------------------------------------------------
    # 证据 D：搜索框占位符
    # ------------------------------------------------------------------
    @staticmethod
    def _has_search_placeholder(lines: list) -> bool:
        for ln in lines:
            t = str(getattr(ln, "text", "")).strip()
            if not t:
                continue
            if t in ("搜索", "Q搜索") or t.endswith("搜索"):
                return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _as_bgr(image: Any) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            return image
        try:
            from PIL import Image  # noqa: WPS433
            if isinstance(image, Image.Image):
                return np.array(image.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            pass
        return np.asarray(image)


__all__ = ["ViewGate", "ViewVerdict"]
