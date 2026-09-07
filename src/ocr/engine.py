import sys
import os
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


def _check_internal_rapidocr():
    if not Path(_INTERNAL_DIR).exists():
        return False
    dll_path = Path(_INTERNAL_DIR) / "python311.dll"
    if dll_path.exists():
        return False
    return True


class TextBox(NamedTuple):
    box: np.ndarray
    text: str
    score: float


class OCRResult:
    def __init__(self, boxes: Optional[np.ndarray] = None,
                 texts: Optional[List[str]] = None,
                 scores: Optional[List[float]] = None):
        self.boxes = boxes if boxes is not None else np.array([])
        self.texts = texts if texts is not None else []
        self.scores = scores if scores is not None else []

    def __len__(self) -> int:
        return len(self.texts)

    def iter_items(self) -> List[TextBox]:
        items = []
        for i in range(len(self.texts)):
            if i < len(self.boxes) and i < len(self.scores):
                items.append(TextBox(self.boxes[i], self.texts[i], self.scores[i]))
        return items

    def get_by_text(self, text: str) -> Optional[TextBox]:
        for item in self.iter_items():
            if text in item.text:
                return item
        return None

    def get_all_text(self) -> str:
        return " ".join(self.texts)


class OCREngine:
    def __init__(self, text_score: float = 0.5,
                 use_det: bool = True,
                 use_cls: bool = True,
                 use_rec: bool = True):
        self.text_score = text_score
        self._engine = None
        self._use_det = use_det
        self._use_cls = use_cls
        self._use_rec = use_rec
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            if _HAS_PIP_RAPIDOCR:
                params = {
                    "Global.text_score": self.text_score,
                    "Global.use_det": self._use_det,
                    "Global.use_cls": self._use_cls,
                    "Global.use_rec": self._use_rec,
                    "Global.max_side_len": 1280,
                    "Global.min_side_len": 30,
                    "Global.min_height": 30,
                    "Global.width_height_ratio": 8,
                }
                self._engine = _RapidOCR(params=params)
                self._initialized = True
                return True
            else:
                return self._init_from_internal()
        except Exception as e:
            print(f"[OCR] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_from_internal(self) -> bool:
        if not Path(_INTERNAL_DIR).exists():
            print("[OCR] No rapidocr found. Install: pip install rapidocr-onnxruntime")
            return False

        saved_path = sys.path[:]
        try:
            if _INTERNAL_DIR not in sys.path:
                sys.path.insert(0, _INTERNAL_DIR)
            from rapidocr import RapidOCR
            params = {
                "Global.text_score": self.text_score,
                "Global.use_det": self._use_det,
                "Global.use_cls": self._use_cls,
                "Global.use_rec": self._use_rec,
                "Global.max_side_len": 1280,
                "Global.min_side_len": 30,
                "Global.min_height": 30,
                "Global.width_height_ratio": 8,
            }
            self._engine = RapidOCR(params=params)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[OCR] _internal init failed: {e}")
            return False
        finally:
            sys.path[:] = saved_path

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._initialized:
            if not self.initialize():
                return OCRResult()

        try:
            raw_result = self._engine(image)

            if isinstance(raw_result, tuple) and len(raw_result) == 2:
                result_list, scores_list = raw_result
            elif isinstance(raw_result, tuple) and len(raw_result) == 1:
                result_list = raw_result[0]
                scores_list = []
            else:
                result_list = raw_result if raw_result else []
                scores_list = []

            boxes = []
            texts = []
            scores = []

            if result_list:
                for item in result_list:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        box, text, score = item
                        if score >= self.text_score:
                            boxes.append(np.array(box))
                            texts.append(str(text))
                            scores.append(float(score))

            if boxes:
                return OCRResult(np.array(boxes), texts, scores)
            return OCRResult(texts=texts, scores=scores)

        except Exception as e:
            print(f"[OCR] Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return OCRResult()

    def recognize_region(self, image: np.ndarray,
                         region: Tuple[int, int, int, int]) -> OCRResult:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return OCRResult()

        result = self.recognize(roi)

        if len(result.boxes) > 0:
            result.boxes[:, :, 0] += x
            result.boxes[:, :, 1] += y

        return result

    def get_text_lines(self, image: np.ndarray) -> List[dict]:
        result = self.recognize(image)
        lines = []
        for item in result.iter_items():
            if len(item.box) >= 4:
                y_center = np.mean(item.box[:, 1])
                lines.append({
                    "text": item.text,
                    "score": item.score,
                    "box": item.box.tolist(),
                    "y_center": float(y_center),
                    "x_min": float(np.min(item.box[:, 0])),
                    "x_max": float(np.max(item.box[:, 0])),
                })
        lines.sort(key=lambda l: l["y_center"])
        return lines
