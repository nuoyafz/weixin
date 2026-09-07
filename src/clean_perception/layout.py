"""布局锚定（LayoutEstimator）—— 把微信截图切成「侧边栏 / 聊天区 / 头部 / 消息区 / 输入区」。

这是感知准确率的第二道闸门（P0）。

背景（实测）：旧实现完全没有区域概念，把左侧会话列表区（实测 x∈[60,465]，
窗口宽 1681）里的列表项当成聊天气泡解析，这是"识别准确率过低"的直接原因。

本模块负责回答「聊天区在哪」，并给出硬约束：
    **消息解析只接受 x_min >= chat_left 的 OCR 行**
从而从结构上杜绝"列表项污染消息列表"。

侧边栏右边界采用三级策略（逐级降级，保证任何情况下都有可用结果）：
    1. 分隔线检测  —— 侧边栏与聊天区之间通常有一条纵向分割线
    2. OCR 左边界聚类 —— 列表项共享同一左边界，取聚类边界
    3. 固定比例降级 —— 0.28 * W，并夹取到合理像素区间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class LayoutAnchors:
    """一帧截图的布局锚点（像素坐标）。"""
    sidebar_right: int = 0      # 侧边栏右边界 = 聊天区左边界
    chat_left: int = 0
    chat_right: int = 0
    header_bottom: int = 0
    message_top: int = 0
    message_bottom: int = 0
    input_top: int = 0
    confidence: float = 0.0
    source: str = "fallback"    # separator | ocr_cluster | ratio | fallback

    @property
    def chat_mid(self) -> float:
        """聊天区自身中线 —— 气泡左右归属必须用这个，而非整图宽度中线。"""
        return (self.chat_left + self.chat_right) / 2.0

    @property
    def chat_width(self) -> int:
        return max(1, self.chat_right - self.chat_left)


class LayoutEstimator:
    """估算微信窗口的布局锚点。"""

    # 侧边栏右边界的合理区间（相对窗口宽度的比例）
    SIDEBAR_MIN_RATIO = 0.16
    SIDEBAR_MAX_RATIO = 0.42
    # 最终兜底比例（实测列表视图下侧边栏约占 0.28W）
    SIDEBAR_FALLBACK_RATIO = 0.28
    # 像素硬夹取（兼顾小窗与超大窗）
    SIDEBAR_MIN_PX = 200
    SIDEBAR_MAX_PX = 560
    # 头部/输入区比例（无检测结果时的降级值）
    HEADER_RATIO = 0.10
    INPUT_RATIO = 0.12

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._fallback_ratio = float(
            self.config.get("sidebar_ratio", self.SIDEBAR_FALLBACK_RATIO))
        self._header_ratio = float(
            self.config.get("header_ratio", self.HEADER_RATIO))
        self._input_ratio = float(
            self.config.get("input_ratio", self.INPUT_RATIO))

    # ------------------------------------------------------------------
    def estimate(self, image: Any,
                 ocr_lines: Optional[Sequence[Any]] = None) -> LayoutAnchors:
        """估算布局锚点。任何异常都会降级，绝不抛出。"""
        try:
            arr = image if isinstance(image, np.ndarray) else np.asarray(image)
            h, w = arr.shape[:2]
        except Exception:
            return LayoutAnchors()

        lines = list(ocr_lines or [])

        # ---- 侧边栏右边界：三级策略 ----
        sidebar, source, conf = self._estimate_sidebar(arr, lines, w)

        header_bottom = int(h * self._header_ratio)
        input_top = int(h * (1.0 - self._input_ratio))

        return LayoutAnchors(
            sidebar_right=sidebar,
            chat_left=sidebar,
            chat_right=w,
            header_bottom=header_bottom,
            message_top=header_bottom,
            message_bottom=input_top,
            input_top=input_top,
            confidence=round(conf, 3),
            source=source,
        )

    # ------------------------------------------------------------------
    def _estimate_sidebar(self, arr: np.ndarray, lines: list, w: int):
        """返回 (sidebar_right, source, confidence)。"""
        lo = int(w * self.SIDEBAR_MIN_RATIO)
        hi = int(w * self.SIDEBAR_MAX_RATIO)

        # 1) 分隔线检测优先（已改进：取最右侧合格边缘 + 上限放宽到 0.40W）。
        #    三栏布局下真实分隔线在列表区右边界（实测 1484 宽时 ~465-536），
        #    取第一个会被列表区内部边缘误导（~247），旧上限 420 会排除真实边界。
        sep = self._detect_separator(arr, lo, hi)
        if sep is not None:
            return self._clamp(sep, w), "separator", 0.9

        # 2) OCR 左边界聚类兜底（列表项共享左边界时，右侧边缘即侧边栏边界；
        #    注意：实测聚类常命中摘要列右边缘，值偏小，仅作无分隔线时的降级）
        cluster_edge = self._ocr_cluster_edge(lines, lo, hi, w)
        if cluster_edge is not None:
            return self._clamp(cluster_edge, w), "ocr_cluster", 0.7

        # 3) 比例降级：微信侧栏宽度实际接近固定 260~340px，宽窗口下按比例
        #    会严重偏右，这里把 fallback 限制在合理区间。
        fallback = int(w * self._fallback_ratio)
        fallback = max(280, min(560, fallback))
        return self._clamp(fallback, w), "ratio", 0.45

    @staticmethod
    def _clamp(x: int, w: int) -> int:
        lo_px = min(LayoutEstimator.SIDEBAR_MIN_PX, max(1, int(w * 0.15)))
        hi_px = max(LayoutEstimator.SIDEBAR_MAX_PX, int(w * 0.45))
        return int(max(lo_px, min(hi_px, x)))

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_separator(arr: np.ndarray, lo: int, hi: int) -> Optional[int]:
        """在 [lo, hi] 内寻找侧栏-聊天区分隔线。

        分栏视图下，真实的分隔线是窗口中**最左侧**的显著纵向边缘；
        如果取全局最大边缘，聊天区里的图片/气泡强边界会把分隔线顶歪。
        因此策略改为：从左到右扫描，取第一个超过阈值且落在合理侧栏宽度
        [0.15W, 0.32W] 区间内的边缘。
        """
        try:
            h, w = arr.shape[:2]
            y0, y1 = int(h * 0.20), int(h * 0.85)
            band = arr[y0:y1]
            if band.size == 0:
                return None
            lo = max(0, min(lo, w - 2))
            hi = max(lo + 2, min(hi, w))
            if hi - lo < 4:
                return None
            sub = band[:, lo:hi].astype(np.float32)
            col_mean = sub.mean(axis=0)                       # (W', 3)
            diff = np.abs(np.diff(col_mean, axis=0)).sum(axis=1)
            if diff.size == 0:
                return None
            base = float(np.median(diff)) + 1e-6
            threshold = max(10.0, base * 3.0)
            min_sep = max(int(w * 0.15), 160)
            # 上限放宽到 0.40W：真实聊天区分隔线在列表区右边界（实测 ~0.28W，
            # 1484 宽时约 465），旧上限 min(0.32W,420) 会把它系统性排除。
            max_sep = min(int(w * 0.40), 560)
            # 取最右侧的合格边缘：三栏布局下列表区内部也存在纵向边缘
            # （实测 ~247），取第一个会被它误导，真实分隔线比它更靠右。
            last = None
            for idx, d in enumerate(diff):
                if d >= threshold:
                    sep = lo + idx + 1
                    if min_sep <= sep <= max_sep:
                        last = sep
                    # 过左/过右都继续向后看
            return last
        except Exception:
            return None

    @staticmethod
    def _ocr_cluster_edge(lines: list, lo: int, hi: int, w: int) -> Optional[int]:
        """从 OCR 行推断侧栏右边界。

        列表项的文字（昵称/预览）共享左侧列，而右侧的时间戳是另一列；
        旧实现把 x_max 的 85 分位数当成边界，会把时间戳/长预览一起算进去，
        导致边界严重偏右。这里改为：
          1) 先找 x_min 最密集的那个左列（侧栏内容列）；
          2) 用该列文本的 x_max 中位数 + 边距作为侧栏右边界。
        """
        try:
            cand = [ln for ln in lines
                    if float(getattr(ln, "x_min", 1e9)) < hi]
            if len(cand) < 3:
                return None

            # 按 x_min 排序，找最密集的一段连续行（侧栏左列）
            sorted_by_xmin = sorted(
                cand, key=lambda ln: float(getattr(ln, "x_min", 0.0)))
            best_cluster = []
            cluster_window = 90.0  # 侧栏内文字左边界散布通常 < 90px
            for i, _ in enumerate(sorted_by_xmin):
                x0 = float(getattr(sorted_by_xmin[i], "x_min", 0.0))
                cluster = [sorted_by_xmin[i]]
                for j in range(i + 1, len(sorted_by_xmin)):
                    xj = float(getattr(sorted_by_xmin[j], "x_min", 0.0))
                    if xj - x0 <= cluster_window:
                        cluster.append(sorted_by_xmin[j])
                    else:
                        break
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            if len(best_cluster) < 3:
                return None

            xmaxs = [float(getattr(ln, "x_max", 0.0)) for ln in best_cluster]
            # 中位数对极端长预览更稳健，+30 给头像/边距留空间
            edge = float(np.median(xmaxs)) + 30.0
            min_sep = max(int(w * 0.15), 160)
            max_sep = min(int(w * 0.40), 560)
            if edge < min_sep or edge > max_sep:
                return None
            return int(edge)
        except Exception:
            return None


__all__ = ["LayoutEstimator", "LayoutAnchors"]
