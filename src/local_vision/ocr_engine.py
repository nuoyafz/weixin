"""OCREngine - RapidOCR 封装，对齐原版 app.local_vision.ocr_engine。

使用 RapidOCR (ONNX Runtime) 进行本地 OCR 识别。
提供 OCRResult 结构化输出和便捷查询方法。
"""
from __future__ import annotations

import sys
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple

import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _HAS_PIP_RAPIDOCR = True
except ImportError:
    _HAS_PIP_RAPIDOCR = False
    _RapidOCR = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_INTERNAL_DIR = str(_PROJECT_ROOT / "_internal")


# SVG path 矢量数据噪声：图标/矢量描边被 RapidOCR 误读为文字，
# 典型形如 '18.172Q600.992 19.6 600.992 21.56L600.992 21.896...Z'。
# 这类串会被 layout 误判为聊天消息，污染 latest_message。
# 过滤条件刻意苛刻：整串仅由 SVG 路径合法字符构成 + 至少含一个路径命令字母，
# 因此不会误伤正常中英文，也不会误杀 T1 的纯数字徽章（无命令字母）。
_SVG_PATH_RE = re.compile(r"^[\s0-9.,+\-MLHVCSQTAZmlhvcsqtaz]+$")
_SVG_CMD_RE = re.compile(r"[MLHVCSQTAZmlhvcsqtaz]")


def _looks_like_svg_path(text: str) -> bool:
    """判断一段 OCR 文本是否为 SVG path 矢量串（而非人类可读文本）。"""
    t = (text or "").strip()
    if len(t) < 10:
        return False
    if not _SVG_PATH_RE.match(t):
        return False
    return bool(_SVG_CMD_RE.search(t))


class _OCRItem:
    """OCR 单条结果包装（兼容 voice_to_text 等调用方的 .cx/.cy/.text 访问）。"""

    __slots__ = ("text", "score", "box", "cx", "cy")

    def __init__(self, text: str, score: float, box: np.ndarray):
        self.text = text
        self.score = score
        self.box = box
        if box.size >= 8:
            pts = box.reshape(-1, 2)
            self.cx = float(np.mean(pts[:, 0]))
            self.cy = float(np.mean(pts[:, 1]))
        else:
            self.cx = 0.0
            self.cy = 0.0

    def __repr__(self) -> str:
        return f"_OCRItem(text={self.text!r}, cx={self.cx:.1f}, cy={self.cy:.1f})"


class TextBox(NamedTuple):
    """单个 OCR 检测结果（原始 NamedTuple）。"""
    box: np.ndarray
    text: str
    score: float


class OCRResult:
    """OCR 识别结果，对齐原版 OCRResult。"""

    def __init__(self, boxes: Optional[np.ndarray] = None,
                 texts: Optional[List[str]] = None,
                 scores: Optional[List[float]] = None):
        self.boxes = boxes if boxes is not None else np.array([])
        self.texts = texts if texts is not None else []
        self.scores = scores if scores is not None else []

    def __len__(self) -> int:
        return len(self.texts)

    def __bool__(self) -> bool:
        return len(self.texts) > 0

    @property
    def items(self) -> List[_OCRItem]:
        """兼容旧接口：返回带 .cx/.cy/.text/.score/.box 的条目列表。"""
        result = []
        n = len(self.texts)
        for i in range(n):
            box = self.boxes[i] if i < len(self.boxes) else np.array([])
            score = self.scores[i] if i < len(self.scores) else 0.0
            result.append(_OCRItem(self.texts[i], float(score), box))
        return result

    def iter_items(self) -> List[TextBox]:
        """遍历所有 OCR 条目。"""
        items = []
        n = len(self.texts)
        for i in range(n):
            if i < len(self.boxes) and i < len(self.scores):
                items.append(TextBox(
                    self.boxes[i] if i < len(self.boxes) else np.array([]),
                    self.texts[i],
                    self.scores[i] if i < len(self.scores) else 0.0,
                ))
        return items

    def get_by_text(self, text: str) -> Optional[TextBox]:
        """按文本内容查找 OCR 条目。"""
        for item in self.iter_items():
            if text in item.text:
                return item
        return None

    def get_all_text(self) -> str:
        """获取所有识别文本（空格分隔）。"""
        return " ".join(self.texts)

    def filter_by_text(self, keyword: str) -> List[TextBox]:
        """按关键词过滤 OCR 条目。"""
        return [item for item in self.iter_items() if keyword in item.text]

    def filter_by_region(self, x_min: float, y_min: float,
                         x_max: float, y_max: float) -> List[TextBox]:
        """按区域过滤 OCR 条目。"""
        results = []
        for item in self.iter_items():
            if item.box.size >= 8:
                box = item.box.reshape(-1, 2)
                cx = np.mean(box[:, 0])
                cy = np.mean(box[:, 1])
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    results.append(item)
        return results

    def to_dict_list(self) -> list[dict]:
        """转为 dict 列表。"""
        return [
            {
                "text": item.text,
                "score": item.score,
                "box": item.box.tolist() if item.box.size else [],
            }
            for item in self.iter_items()
        ]


class OCREngine:
    """本地 OCR 引擎，封装 RapidOCR。

    对齐原版 app.local_vision.ocr_engine.OCREngine。
    """

    def __init__(self, text_score: float = 0.5,
                 use_det: bool = True,
                 use_cls: bool = True,
                 use_rec: bool = True,
                 box_thresh: float = 0.5):
        self.text_score = text_score
        self.box_thresh = box_thresh
        self._engine = None
        self._use_det = use_det
        self._use_cls = use_cls
        self._use_rec = use_rec
        self._initialized = False

    def initialize(self) -> bool:
        """初始化 OCR 引擎。"""
        if self._initialized:
            return True

        try:
            if _HAS_PIP_RAPIDOCR:
                params = {
                    "Global.text_score": self.text_score,
                    "Global.use_det": self._use_det,
                    "Global.use_cls": self._use_cls,
                    "Global.use_rec": self._use_rec,
                    "Global.max_side_len": 2000,
                    "Global.min_side_len": 30,
                    "Global.min_height": 30,
                    "Global.width_height_ratio": 8,
                    "Global.box_thresh": self.box_thresh,
                }
                self._engine = _RapidOCR(params=params)
                self._initialized = True
                return True
            else:
                return self._init_from_internal()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def _init_from_internal(self) -> bool:
        """从 _internal 目录加载 OCR 引擎。"""
        try:
            if not Path(_INTERNAL_DIR).exists():
                return False
            if str(_INTERNAL_DIR) not in sys.path:
                sys.path.insert(0, str(_INTERNAL_DIR))
            from rapidocr_onnxruntime import RapidOCR
            params = {
                "Global.text_score": self.text_score,
                "Global.box_thresh": self.box_thresh,
            }
            self._engine = RapidOCR(params=params)
            self._initialized = True
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        """检查引擎是否就绪。"""
        return self._initialized and self._engine is not None

    def run(self, image, use_det: bool = None,
            use_cls: bool = None, use_rec: bool = None,
            **kwargs) -> OCRResult:
        """对图像执行 OCR 识别。

        Args:
            image: PIL Image 或 numpy ndarray
            use_det: 是否使用检测
            use_cls: 是否使用方向分类
            use_rec: 是否使用识别

        Returns:
            OCRResult 包含识别结果
        """
        if not self.is_ready():
            if not self.initialize():
                return OCRResult()

        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                image = np.array(image)

            result, elapse = self._engine(
                image,
                use_det=use_det if use_det is not None else self._use_det,
                use_cls=use_cls if use_cls is not None else self._use_cls,
                use_rec=use_rec if use_rec is not None else self._use_rec,
            )

            if result is None:
                return OCRResult()

            boxes = []
            texts = []
            scores = []
            for item in result:
                # T1: rec-only 模式（use_det=False）下 RapidOCR 返回 2 元组
                # (text, score)，无 box；det+rec 模式返回 3 元组 (box, text, score)。
                if len(item) == 3:
                    box, text, score = item
                else:
                    text, score = item
                    box = np.array([])
                # 过滤 SVG path 矢量噪声行（图标/矢量描边被 OCR 误读为文字，
                # 典型如 '18.172Q600.992 19.6...L...Z'），避免污染聊天消息解析。
                # 纯数字徽章（T1）不含路径命令字母，不会被误杀。
                if _looks_like_svg_path(text):
                    continue
                boxes.append(np.array(box, dtype=np.float32))
                texts.append(text)
                scores.append(float(score))

            return OCRResult(
                boxes=np.array(boxes) if boxes else np.array([]),
                texts=texts,
                scores=scores,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return OCRResult()

    def run_on_region(self, image, region: Tuple[int, int, int, int],
                      **kwargs) -> OCRResult:
        """在指定区域上执行 OCR。

        Args:
            image: PIL Image 或 numpy ndarray
            region: (x, y, w, h) 区域坐标
        """
        try:
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            return self.run(cropped, **kwargs)
        except Exception:
            return OCRResult()

    def get_text_lines(self, image, **kwargs) -> List[dict]:
        """返回文本行列表，供 local_vision 兜底路径（WechatLayoutParser）消费。

        observe_service 的 fallback（_analyze_with_local_ocr）调用本方法，
        再把结果交给 ``WechatLayoutParser.analyze_chat_screen(ocr_lines=...)``。
        字段对齐原版 ``src/ocr/engine.OCRTextEngine.get_text_lines``
        （text/score/box/y_center/x_min/x_max，按 y_center 升序），保证解析器
        可直接消费。直接复用 ``run()``，因此也继承 SVG path 矢量噪声过滤。

        Returns:
            List[dict]: 每行一个 dict；无文本时返回空列表。
        """
        result = self.run(image, **kwargs)
        lines: List[dict] = []
        for item in result.iter_items():
            box = getattr(item, "box", None)
            if getattr(box, "size", 0) < 4:
                # rec-only 模式返回无 box 的纯文本行，布局解析需要坐标，跳过。
                continue
            pts = np.asarray(box, dtype=float).reshape(-1, 2)
            y_center = float(np.mean(pts[:, 1]))
            lines.append({
                "text": str(getattr(item, "text", "")).strip(),
                "score": float(getattr(item, "score", 0.0) or 0.0),
                "box": pts.tolist(),
                "y_center": y_center,
                "x_min": float(np.min(pts[:, 0])),
                "x_max": float(np.max(pts[:, 0])),
            })
        lines.sort(key=lambda l: l["y_center"])
        return lines

    def detect_text_lines(self, image, **kwargs) -> List[TextBox]:
        """检测图像中的文本行。"""
        result = self.run(image, **kwargs)
        return result.iter_items()

    def detect_first(self, image, keyword: str, **kwargs) -> Optional[TextBox]:
        """检测图像中第一个匹配关键词的文本。"""
        result = self.run(image, **kwargs)
        return result.get_by_text(keyword)

    def detect_all_text(self, image, **kwargs) -> str:
        """检测图像中的所有文本（空格分隔）。"""
        result = self.run(image, **kwargs)
        return result.get_all_text()