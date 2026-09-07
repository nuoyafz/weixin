"""T2 视觉定位（YOLOv8）推理模块 —— 规划骨架，当前未启用。

设计目标：用目标检测模型输出精确 bounding box，替换 red_dot_detector 内基于
规则/比例的坐标定位（_find_dots / _nav_avatar_bottom / _nav_chat_icon_y /
双击置顶的 45px 标题栏 no-fly zone）。详见 UI 设置中心「T2 视觉定位升级」页。

启用方式（由调用方控制，本模块不读配置）：
    1. 训练 yolov8n 并导出 ONNX 到 settings.wechat.yolo_model_path
    2. 调用方在 settings.wechat.localization_mode == "yolo" 时 use 本模块
    3. 模型缺失/推理异常时本模块抛 YoloUnavailable，调用方必须回退 rule_based

识别类别（CLASSES 顺序与训练时的 data.yaml 必须一致）：
    red_dot       未读小红点（无数字）
    unread_badge  带数字未读徽章（导航栏 + 列表行右侧）
    avatar        左侧头像列整段
    nav_chat_icon 聊天导航图标（绿色高亮态）
    title_bar     微信窗口标题栏矩形（用于双击前置顶排除 no-fly zone）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
    _ORT_OK = True
except Exception:  # pragma: no cover - 取决于运行环境是否装了 onnxruntime
    ort = None
    _ORT_OK = False


class YoloUnavailable(Exception):
    """模型不可用（缺失 / 加载失败 / onnxruntime 缺失）。调用方据此回退规则逻辑。"""


# 与训练 data.yaml 的 names 顺序严格一致
CLASSES = ["red_dot", "unread_badge", "avatar", "nav_chat_icon", "title_bar"]

Box = Tuple[int, int, int, int, float]  # (x, y, w, h, score) 原图像素坐标


class YoloLocator:
    def __init__(self, model_path: str, conf: float = 0.5, imgsz: int = 640):
        if not _ORT_OK:
            raise YoloUnavailable("onnxruntime 未安装，无法加载 YOLO 模型")
        if not model_path or not os.path.exists(model_path):
            raise YoloUnavailable(f"YOLO 模型文件不存在: {model_path}")
        try:
            self.sess = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as e:  # noqa: BLE001
            raise YoloUnavailable(f"YOLO 模型加载失败: {e}")
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.input_name = self.sess.get_inputs()[0].name

    # ---------- 预处理：letterbox 到 imgsz×imgsz ----------
    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        left = (self.imgsz - nw) // 2
        top = (self.imgsz - nh) // 2
        import cv2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, scale, left, top

    def _preprocess(self, img: np.ndarray):
        import cv2
        canvas, scale, left, top = self._letterbox(img)
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]  # 1 × 3 × imgsz × imgsz
        return blob, scale, left, top

    # ---------- 后处理：解析输出 + 反 letterbox ----------
    def _postprocess(self, raw, scale: float, left: int, top: int):
        out = raw[0] if isinstance(raw, (list, tuple)) else raw
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        # 归一化为 [N, C]：兼容 ultralytics 两种导出布局（channels-first / last）
        if out.shape[0] < out.shape[1]:
            out = out.T
        n, c = out.shape
        coords = out[:, :4]
        cls_scores = out[:, 4:]  # [N, num_classes]，v8 导出已含 sigmoid
        num_classes = cls_scores.shape[1]
        results: Dict[str, List[Box]] = {name: [] for name in CLASSES}
        for i in range(n):
            probs = cls_scores[i]
            cid = int(np.argmax(probs))
            score = float(probs[cid])
            if score < self.conf or cid >= len(CLASSES) or cid >= num_classes:
                continue
            cx, cy, bw, bh = coords[i]
            # 模型坐标在 letterbox(640) 空间 -> 原图像素
            x = (cx - bw / 2 - left) / scale
            y = (cy - bh / 2 - top) / scale
            w = bw / scale
            h = bh / scale
            x = max(0, int(round(x)))
            y = max(0, int(round(y)))
            w = max(0, int(round(w)))
            h = max(0, int(round(h)))
            results[CLASSES[cid]].append((x, y, w, h, round(score, 3)))
        return results

    # ---------- 对外接口（返回原图像素坐标） ----------
    def locate(self, img: np.ndarray) -> Dict[str, List[Box]]:
        blob, scale, left, top = self._preprocess(img)
        raw = self.sess.run(None, {self.input_name: blob})
        return self._postprocess(raw, scale, left, top)

    def chat_icon_y(self, img: np.ndarray) -> Optional[int]:
        """聊天导航图标中心 y（替代 _nav_chat_icon_y 的中位数投票）。"""
        boxes = self.locate(img).get("nav_chat_icon", [])
        return boxes[0][1] if boxes else None

    def title_bar_rect(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """标题栏矩形 (x, y, w, h)，用于双击前置顶排除 no-fly zone。"""
        boxes = self.locate(img).get("title_bar", [])
        return boxes[0][:4] if boxes else None

    def unread_boxes(self, img: np.ndarray) -> List[Box]:
        """未读红点 + 带数字徽章（替代 _find_unread_row 内的 _find_dots）。"""
        d = self.locate(img)
        return d.get("red_dot", []) + d.get("unread_badge", [])

    def avatar_segments(self, img: np.ndarray) -> List[Box]:
        """头像列段（替代 _nav_avatar_bottom 的动态底边检测）。"""
        return self.locate(img).get("avatar", [])


def get_locator(model_path: Optional[str] = None, conf: float = 0.5) -> Optional[YoloLocator]:
    """工厂：模型可用返回实例，否则返回 None（调用方保持 rule_based）。"""
    if not model_path:
        return None
    try:
        return YoloLocator(model_path, conf=conf)
    except YoloUnavailable:
        return None
