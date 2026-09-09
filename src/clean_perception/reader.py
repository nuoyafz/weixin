"""干净版微信聊天截图感知层（L2 布局 + L3 OCR）。

设计目标：完全从零实现，不依赖任何反编译产物。
只复用项目里已经干净的 OCR 封装 ``local_vision.ocr_engine.OCREngine``
（RapidOCR 本地识别），其余布局/气泡/联系人/草稿推断全部自己写。

原项目架构（感知四级瀑布）：
  L1 像素红点  -> 见 rpa/red_dot_detector.py（已重写）
  L2 布局解析  -> view_gate.py（视图闸门）+ layout.py（区域锚定）+ 本模块气泡归属
  L3 OCR 文字  -> 本模块：复用 OCREngine（含预处理与置信度治理）
  L4 视觉大模型 -> 见 agent/observe_service 的 vision 路径（按需接入）

**准确率关键（v2 升级）**：
  旧实现跳过 L2，直接在全图上裸跑 OCR，导致把「会话列表视图」的侧边栏
  列表项当成聊天消息（实测 current_contact='Q搜索'、8 条消息全为列表预览）。
  本版补齐 L2：
    1. ViewGate 先判视图 —— 列表视图直接返回，绝不产出消息；
    2. LayoutEstimator 定位聊天区 —— 消息解析只接受 x_min >= chat_left 的行；
    3. 气泡左右归属改用三重证据（背景色 > 聊天区中线 > 右对齐），
       取代旧的"整图宽度 52% 中线"错误判据。

本模块把「一张聊天窗口截图」变成结构化 ``ChatAnalysis``：
  - current_contact    : 头部标题里的联系人名（仅取自聊天区）
  - messages           : 按时间排序的消息气泡列表（含 side/sender）
  - last_customer_message : 最新一条对方(左)消息文本
  - draft_text         : 输入框里已输入的草稿
  - is_self_latest     : 最新一条是不是自己发的
  - intent             : 简单意图标签，供决策引擎消费
  - view / perception_confidence : 供下游证据门控判断是否允许自动发送
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .layout import LayoutAnchors, LayoutEstimator
from .view_gate import ViewGate, ViewVerdict


# 微信聊天区常见的独立系统行：时间戳 / 日期分隔 / "撤回了一条消息"等
# 这些不是用户消息，不应参与 last_customer_message / is_self_latest 判定。
_STANDALONE_SYSTEM_RE = re.compile(
    r"^\s*(\d{1,2}:\d{2}|\d{2}:\d{2}|"
    r"昨天|今天|星期一|星期二|星期三|星期四|星期五|星期六|星期日|"
    r"周一|周二|周三|周四|周五|周六|周日|"
    r".{0,12}撤回了一条消息)\s*$"
)


def _is_standalone_system(text: str) -> bool:
    if not text:
        return False
    return bool(_STANDALONE_SYSTEM_RE.match(text.strip()))


# 微信会话里的"未读分隔线"文案：出现在它下方的消息才算未读。
_UNREAD_MARKER_KEYWORDS = (
    "以下为新消息", "以下为未读消息", "你以上是新消息",
    "以下的新消息", "以下新消息",
)


def _is_unread_marker(text: str) -> bool:
    if not text:
        return False
    t = text.replace(" ", "").replace("　", "")
    return any(kw in t for kw in _UNREAD_MARKER_KEYWORDS)


# 窗口控制字符（微信标题栏右上角的 最小化/最大化/关闭 按钮），
# 以及搜索占位符，都不应被当成联系人名。
_WINDOW_CONTROL_TEXTS = {"×", "口", "—", "_", "□", "×", "✕", "▢"}
_SEARCH_PLACEHOLDERS = {"搜索", "Q搜索", "Q搜索", "Q 搜索", "搜索聊天"}


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class ChatMessage:
    """一条被归类的聊天消息气泡。"""
    text: str = ""
    side: str = "left"          # "left"(对方) | "right"(自己) | "center"(系统)
    sender: str = ""
    y_center: float = 0.0
    confidence: float = 0.0
    side_confidence: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    low_conf: bool = False      # 组成行置信度偏低，不参与关键字段判定

    @property
    def is_system(self) -> bool:
        return self.side == "center"


@dataclass
class ChatAnalysis:
    """一次截图的结构化分析结果。"""
    current_contact: str = ""
    messages: List[ChatMessage] = field(default_factory=list)
    last_customer_message: str = ""
    draft_text: str = ""
    is_self_latest: bool = False
    intent: str = ""
    image_size: Tuple[int, int] = (0, 0)
    raw_text_lines: List[str] = field(default_factory=list)
    # 派生元数据，方便下游 send_guard 使用
    sender_is_customer: bool = True
    has_unread_like: bool = False
    # 未读消息数量（会话视图内）
    unread_count: int = 0
    unread_count_source: str = ""   # 'divider' | 'estimate_no_marker' | ''

    # ---- v2 新增：视图 / 布局 / 置信度（供证据门控） ----
    view: str = "unknown"                       # conversation | list | unknown
    view_confidence: float = 0.0
    anchors: Optional[LayoutAnchors] = None
    perception_confidence: float = 0.0
    low_conf_flags: List[str] = field(default_factory=list)

    def to_decision_dict(self) -> dict:
        """转成决策引擎/守卫能直接消费的 dict。"""
        return {
            "current_contact": self.current_contact,
            "last_customer_message": self.last_customer_message,
            "draft_text": self.draft_text,
            "is_self_latest_message": self.is_self_latest,
            "unread_count": self.unread_count,
            "unread_count_source": self.unread_count_source,
            "intent": self.intent,
            "sender_is_customer": self.sender_is_customer,
            "messages": [
                {
                    "text": m.text,
                    "side": m.side,
                    "sender": m.sender,
                    "is_system": m.is_system,
                }
                for m in self.messages
            ],
        }

    def to_analyze_schema(self, image_path: str = "") -> dict:
        """转换为与 local_vision.wechat_layout_parser.analyze_chat_screen
        同构的 dict，供 observe_service._analyze_with_local_ocr →
        decision.normalize 直接消费，**无需改动下游决策/守卫逻辑**。

        干净的感知层只负责「看到了什么」这些事实字段；是否回复、
        回复内容由 decision_engine 统一裁决。
        """
        messages = self.messages
        latest = messages[-1] if messages else None

        if latest is not None:
            side = latest.side
            last_content = latest.text
        elif self.last_customer_message:
            side = "left"
            last_content = self.last_customer_message
        else:
            side = "unknown"
            last_content = ""

        if side == "right":
            sender = "self"
        elif side == "left":
            sender = "customer"
        elif side == "center":
            sender = "system"
        else:
            sender = "unknown"

        # —— 视图闸门：列表视图 / 未知视图 一律不回复 ——
        if self.view == "list":
            return self._schema(
                decision="no_reply", intent="list_view_not_conversation",
                should_reply=False, confidence=self.view_confidence,
                sender=sender, last_content="", image_path=image_path,
            )
        if self.view == "unknown":
            return self._schema(
                decision="no_reply", intent="view_unknown",
                should_reply=False, confidence=self.view_confidence * 0.6,
                sender=sender, last_content="", image_path=image_path,
            )

        has_customer_turn = bool(self.last_customer_message) and not self.is_self_latest

        if has_customer_turn:
            decision = "reply_draft"
            intent = "chat_message"
            should_reply = True
            confidence = max(0.70, min(0.95, 0.70 + self.perception_confidence * 0.25))
        elif messages:
            decision = "no_reply"
            intent = "chat_no_new_message"
            should_reply = False
            confidence = 0.6
        else:
            decision = "no_reply"
            intent = "chat_empty"
            should_reply = False
            confidence = 0.4

        return self._schema(
            decision=decision, intent=intent, should_reply=should_reply,
            confidence=confidence, sender=sender,
            last_content=last_content, image_path=image_path,
        )

    def _schema(self, *, decision: str, intent: str, should_reply: bool,
                confidence: float, sender: str, last_content: str,
                image_path: str) -> dict:
        return {
            "current_contact": self.current_contact,
            "latest_message": {
                "content": last_content,
                "text": last_content,
                "sender": sender,
                "type": "text",
            },
            "intent": intent,
            "decision": decision,
            "is_self_latest_message": self.is_self_latest,
            "should_reply": should_reply,
            "confidence": confidence,
            "messages": [
                {
                    "text": m.text,
                    "side": m.side,
                    "sender": ("customer" if m.side == "left"
                               else "self" if m.side == "right"
                               else "system"),
                    "y_center": m.y_center,
                }
                for m in self.messages
            ],
            "content": last_content,
            "new_messages": (
                [{"content": self.last_customer_message, "side": "left"}]
                if (should_reply and self.last_customer_message) else []
            ),
            "draft_text": self.draft_text,
            "unread_count": self.unread_count,
            "unread_count_source": self.unread_count_source,
            "raw_text_lines": self.raw_text_lines,
            # v2：供证据门控使用
            "view": self.view,
            "view_confidence": self.view_confidence,
            "perception_confidence": self.perception_confidence,
            "low_conf_flags": list(self.low_conf_flags),
            "parser": "clean_perception",
            "image_path": image_path,
        }


# ----------------------------------------------------------------------
# OCR 结果归一化
# ----------------------------------------------------------------------
@dataclass
class _OcrLine:
    text: str
    score: float
    cx: float
    cy: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    low_conf: bool = False


def _normalize_ocr(result: Any) -> List[_OcrLine]:
    """把任意 OCR 输出（OCREngine.OCRResult / RapidOCR 原生）归一化成 _OcrLine。"""
    lines: List[_OcrLine] = []
    if result is None:
        return lines

    # 形态 A：OCREngine.OCRResult
    if hasattr(result, "iter_items"):
        for it in result.iter_items():
            box = getattr(it, "box", None)
            if box is None or getattr(box, "size", 0) < 8:
                continue
            pts = np.asarray(box, dtype=float).reshape(-1, 2)
            xs, ys = pts[:, 0], pts[:, 1]
            lines.append(_OcrLine(
                text=str(getattr(it, "text", "")).strip(),
                score=float(getattr(it, "score", 0.0) or 0.0),
                cx=float(np.mean(xs)), cy=float(np.mean(ys)),
                x_min=float(xs.min()), x_max=float(xs.max()),
                y_min=float(ys.min()), y_max=float(ys.max()),
            ))
        return lines

    # 形态 B：RapidOCR 原生 [(box, text, score), ...]
    if isinstance(result, (list, tuple)):
        for item in result:
            try:
                box, text, score = item
            except Exception:
                continue
            pts = np.asarray(box, dtype=float).reshape(-1, 2)
            xs, ys = pts[:, 0], pts[:, 1]
            lines.append(_OcrLine(
                text=str(text).strip(), score=float(score or 0.0),
                cx=float(np.mean(xs)), cy=float(np.mean(ys)),
                x_min=float(xs.min()), x_max=float(xs.max()),
                y_min=float(ys.min()), y_max=float(ys.max()),
            ))
    return lines


# ----------------------------------------------------------------------
# 核心阅读器
# ----------------------------------------------------------------------
class WechatScreenReader:
    """把一张微信聊天窗口截图解析为结构化 ``ChatAnalysis``。

    全部用几何/统计启发式完成，不依赖任何反编译逻辑。
    """

    # 布局比例（微信桌面版相对稳定的经验值，可用 config 覆盖）
    HEADER_RATIO = 0.10          # 顶部标题栏占整图高度比例
    INPUT_RATIO = 0.12           # 底部输入区占整图高度比例
    # OCR 治理
    MIN_OCR_SCORE = 0.55         # 低于该置信度的行标记为 low_conf
    HARD_DROP_SCORE = 0.25       # 低于该置信度的行直接丢弃
    PREPROCESS = True            # 是否做 CLAHE + 锐化预处理
    UPSCALE = 1.0                # OCR 前放大倍数（>1 可提升小字，但更慢）
    # 证据门控
    AUTO_SEND_MIN_CONFIDENCE = 0.70

    def __init__(self, ocr_engine: Any = None, config: Optional[dict] = None):
        self.config = config or {}
        self._ocr = ocr_engine
        self._header_ratio = float(self.config.get("header_ratio", self.HEADER_RATIO))
        self._input_ratio = float(self.config.get("input_ratio", self.INPUT_RATIO))
        self._min_ocr_score = float(self.config.get("min_ocr_score", self.MIN_OCR_SCORE))
        self._hard_drop_score = float(self.config.get("hard_drop_score", self.HARD_DROP_SCORE))
        self._preprocess = bool(self.config.get("preprocess", self.PREPROCESS))
        self._upscale = float(self.config.get("upscale", self.UPSCALE))

        # L2 组件：视图闸门 + 区域锚定
        self.view_gate = ViewGate(self.config.get("view_gate", {}))
        self.layout = LayoutEstimator(self.config.get("layout", {}))

    # -- OCR 懒加载 -----------------------------------------------------
    def _get_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        # 复用项目里干净的 RapidOCR 封装（local_vision/ocr_engine.py）。
        # 注意：直接按文件路径加载该模块，绕开 local_vision/__init__.py
        # 里那个会触发越级相对导入(..message.parser)的脆弱链。
        try:
            import importlib.util
            import os
            here = os.path.dirname(os.path.abspath(__file__))
            engine_path = os.path.abspath(
                os.path.join(here, "..", "local_vision", "ocr_engine.py"))
            spec = importlib.util.spec_from_file_location(
                "clean_perception._ocrengine", engine_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            OCREngine = getattr(mod, "OCREngine", None)
            if OCREngine is None:
                self._ocr = None
                return None
            self._ocr = OCREngine()
            if not self._ocr.is_ready():
                self._ocr.initialize()
        except Exception:
            self._ocr = None
        return self._ocr

    # -- 主入口 ---------------------------------------------------------
    def analyze(self, image: Any, *, roi_cropped: bool = False) -> ChatAnalysis:
        """分析一张聊天窗口截图，返回结构化结果。

        roi_cropped: 该帧已由 observe_service 做过 ROI 裁剪（裁掉顶部窗口标题栏）。
          为 True 时关闭「顶部 4.5% 标题栏过滤」，避免把本就在帧顶的联系人名
          误判为窗口标题栏而丢弃（详见 _extract_contact 中 top_bar_y 的处理）。
        """
        arr = self._to_array(image)
        h, w = arr.shape[:2]
        analysis = ChatAnalysis(image_size=(w, h))

        ocr = self._get_ocr()
        if ocr is None:
            return analysis

        # 1) OCR（带图像增强预处理）
        enhanced = self._enhance(arr)
        result = ocr.run(enhanced) if hasattr(ocr, "run") else None
        lines = _normalize_ocr(result)
        # 放大后坐标需还原到原图尺度
        if self._upscale and self._upscale > 1.01:
            lines = self._rescale_lines(lines, 1.0 / self._upscale)
        lines = self._quality_filter(lines, analysis)
        analysis.raw_text_lines = [ln.text for ln in lines if ln.text]
        if not lines:
            return analysis

        # 2) 【视图闸门 P0】列表视图不得当会话解析
        verdict = self.view_gate.judge(arr, lines)
        analysis.view = verdict.view
        analysis.view_confidence = verdict.confidence
        if verdict.is_list:
            analysis.low_conf_flags.append("list_view")
            analysis.intent = "list_view"
            return analysis
        if verdict.is_unknown:
            analysis.low_conf_flags.append("view_unknown")

        # 3) 【区域锚定 P0】定位聊天区
        anchors = self.layout.estimate(arr, lines)
        analysis.anchors = anchors

        # 4) 头部联系人（仅取聊天区内、非低置信行）
        analysis.current_contact = self._extract_contact(
            lines, anchors, h, w, roi_cropped=roi_cropped)

        # 5) 输入草稿
        analysis.draft_text = self._extract_draft(lines, anchors, w)

        # 6) 消息气泡：硬约束 —— 只取聊天区内的行
        chat_lines = [
            ln for ln in lines
            if ln.x_min >= anchors.chat_left
            and ln.y_min >= anchors.message_top
            and ln.y_max <= anchors.message_bottom
        ]
        messages = self._group_messages(chat_lines, arr, anchors)
        analysis.messages = messages

        # 6.5) 未读数（会话视图）：检测"以下为新消息"分隔线 + 统计下方客户消息
        self._detect_unread(analysis, lines, chat_lines, messages, anchors, h)

        # 7) 派生结论 + 感知置信度
        self._derive(analysis, messages, w)
        analysis.perception_confidence = self._perception_confidence(
            analysis, verdict, anchors, messages)
        return analysis

    # -- 未读消息计数 ---------------------------------------------------
    def _detect_unread(self, analysis: ChatAnalysis,
                       lines: List[_OcrLine],
                       chat_lines: List[_OcrLine],
                       messages: List[ChatMessage],
                       anchors: LayoutAnchors, h: int) -> None:
        """统计当前会话视图内的未读消息条数。

        - 若截图里存在「以下为新消息」分隔线：其下方的客户(left)消息即未读；
        - 若不存在分隔线（对话框已读 / OCR 未捕获分隔线）：退化为「视图内全部
          客户消息数」估计，并在 source 标注 estimate_no_marker，供下游谨慎使用。
        """
        marker_y: Optional[float] = None
        for ln in lines:
            if ln.x_min >= anchors.chat_left and _is_unread_marker(ln.text):
                marker_y = ln.cy
                break

        customer = [m for m in messages if m.side == "left"
                    and not m.is_system]
        if marker_y is not None:
            unread = [m for m in customer if m.y_center > marker_y]
            analysis.unread_count = len(unread)
            analysis.unread_count_source = "divider"
        else:
            # 无分隔线：把视图内可见客户消息当作未读估计（上限 30 防止极端噪声）
            analysis.unread_count = min(len(customer), 30)
            analysis.unread_count_source = "estimate_no_marker"

    # -- OCR 预处理 -----------------------------------------------------
    def _enhance(self, arr: np.ndarray) -> np.ndarray:
        """OCR 前图像增强：可选放大 + CLAHE 对比度增强 + 轻度锐化。

        对小字、低对比度、缩放模糊的微信截图有显著提升。
        任何异常都返回原图，绝不阻断主流程。
        """
        if not self._preprocess and not (self._upscale and self._upscale > 1.01):
            return arr
        try:
            import cv2
            img = arr
            if self._upscale and self._upscale > 1.01:
                img = cv2.resize(img, None, fx=self._upscale, fy=self._upscale,
                                 interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            merged = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]],
                              dtype=np.float32)
            return cv2.filter2D(merged, -1, kernel)
        except Exception:
            return arr

    @staticmethod
    def _rescale_lines(lines: List[_OcrLine], factor: float) -> List[_OcrLine]:
        """把放大后 OCR 得到的坐标还原回原图尺度。"""
        out: List[_OcrLine] = []
        for ln in lines:
            out.append(_OcrLine(
                text=ln.text, score=ln.score,
                cx=ln.cx * factor, cy=ln.cy * factor,
                x_min=ln.x_min * factor, x_max=ln.x_max * factor,
                y_min=ln.y_min * factor, y_max=ln.y_max * factor,
                low_conf=ln.low_conf,
            ))
        return out

    def _quality_filter(self, lines: List[_OcrLine],
                        analysis: ChatAnalysis) -> List[_OcrLine]:
        """置信度治理：丢弃极低分行；低分行标记 low_conf，不参与关键字段。"""
        kept: List[_OcrLine] = []
        low_count = 0
        for ln in lines:
            if ln.score < self._hard_drop_score:
                continue
            if ln.score < self._min_ocr_score:
                ln.low_conf = True
                low_count += 1
            kept.append(ln)
        if low_count:
            analysis.low_conf_flags.append("low_ocr_score")
        return kept

    # -- 头部联系人 -----------------------------------------------------
    def _extract_contact(self, lines: List[_OcrLine], anchors: LayoutAnchors,
                         h: int = 0, w: int = 0, *,
                         roi_cropped: bool = False) -> str:
        """头部联系人：仅取聊天区内、非低置信、位于头部带的行。

        必须排除：
          - 标题栏右上角的窗口控制按钮（× / 口 / — 等，cy 极小或 x 极靠右）
          - 搜索框占位符（搜索 / Q搜索）
          - 单字符符号（不是联系人名）

        关键修正：头部标题名用「中心 x 落在会话区」判定，而非「x_min >= chat_left」。
        实测分隔线检测的 chat_left 常比会话头部文本实际左界偏右 ~15px（本例
        联系人 x_min=480 而 chat_left=493，被旧逻辑误杀导致 current_contact=''）。
        改用行中心 x 越过 chat_left（允许小幅向左溢出），更稳；并保底用 0.30*w 比例，
        确保左侧栏的「搜索框 / 列表项」永远落选。
        """
        if h <= 0:
            h = (anchors.header_bottom * 10) if anchors else 0
        if w <= 0:
            w = (anchors.chat_right * 2) if anchors else 0
        # 帧已被 ROI 裁掉顶部窗口标题栏时，关闭该过滤（top_bar_y=0），否则会把本就
        # 在帧顶的联系人名误判为标题栏而丢弃。未裁剪时仍按 4.5% 跳过 OS 标题栏。
        top_bar_y = 0 if roi_cropped else h * 0.045
        far_right_x = w * 0.96         # 不与右上角 ×/口 控制按钮重叠
        header_bottom = anchors.header_bottom if anchors else int(h * 0.10)
        chat_left = anchors.chat_left if anchors else int(w * 0.28)
        # 左侧栏列表项最大中心 x 通常 < 0.30*w；会话头部标题中心一定 >= 此阈值
        left_threshold = max(chat_left - 30, int(w * 0.30))
        header_lines = [
            ln for ln in lines
            if ln.cy <= header_bottom          # 用中心而非下沿，避免标题框略超头部带被漏掉
            and ln.y_min >= top_bar_y
            and (ln.x_min + ln.x_max) / 2.0 >= left_threshold   # 中心落在会话区
            and ln.x_max <= far_right_x
            and not ln.low_conf
            and ln.text
            and ln.text not in _WINDOW_CONTROL_TEXTS
            and ln.text not in _SEARCH_PLACEHOLDERS
            and not (len(ln.text.strip()) <= 1 and not ln.text.strip().isascii())
        ]
        if not header_lines:
            return ""
        # 标题通常是头部里字号最大（框最高）且最靠上的那行
        header_lines.sort(key=lambda ln: (-(ln.y_max - ln.y_min), ln.cy))
        return header_lines[0].text

    # -- 输入框草稿 -----------------------------------------------------
    _INPUT_BUTTON_WORDS = {"发送", "语音通话", "视频通话", "表情", "更多",
                            "文件", "截图", "聊天记录", "折叠的聊天"}

    @staticmethod
    def _is_meaningful_draft(text: str) -> bool:
        """输入框里的文本需是「像人打出来的字」：含中文，或长度>=2 且含字母数字。
        单字符拉丁/符号（如图标被误识别成 'U'/'?'）不是用户输入。"""
        t = (text or "").strip()
        if not t:
            return False
        if any("一" <= c <= "鿿" for c in t):
            return True
        return len(t) >= 2 and any(c.isalnum() for c in t)

    def _extract_draft(self, lines: List[_OcrLine], anchors: LayoutAnchors,
                       w: int = 0) -> str:
        """输入框草稿：仅取输入区内、位于输入框正文位置（左侧、非按钮）的文本。

        底部输入栏左侧是 表情/语音 图标、右侧是「发送」按钮，都不应当成草稿。
        这里用 icon_limit 排除左侧图标，用 right_limit 排除右侧「发送」按钮，
        再用 _is_meaningful_draft 过滤掉图标被误识别出的单字符噪声。
        """
        if w <= 0:
            w = anchors.chat_right * 2 if anchors else 0
        right_limit = w * 0.82
        # 输入栏左侧的 表情/语音 等图标（x 约在 chat_left+0~0.30*chat_width 内）
        # 不是用户输入，需排除；真正的输入框在其右侧。
        icon_limit = anchors.chat_left + 0.30 * getattr(anchors, "chat_width", w)
        draft_lines = [
            ln for ln in lines
            if ln.y_min >= anchors.input_top
            and ln.x_min >= icon_limit
            and ln.x_max <= right_limit
            and ln.text
            and ln.text not in self._INPUT_BUTTON_WORDS
            and self._is_meaningful_draft(ln.text)
        ]
        if not draft_lines:
            return ""
        draft_lines.sort(key=lambda ln: ln.cy)
        return " ".join(ln.text for ln in draft_lines).strip()

    # -- 消息分组 -------------------------------------------------------
    def _group_messages(self, lines: List[_OcrLine], image: np.ndarray,
                        anchors: LayoutAnchors) -> List[ChatMessage]:
        """把聊天区内的 OCR 行聚成气泡，并用三重证据判定左右归属。

        时间戳、日期分隔、系统提示等独立行直接跳过，避免把时间戳
        当成最后一条客户消息，也避免与真正的消息气泡错误合并。
        """
        if not lines:
            return []
        ordered = sorted(lines, key=lambda ln: ln.cy)
        # 过滤独立的系统行
        ordered = [ln for ln in ordered if not _is_standalone_system(ln.text)]
        if not ordered:
            return []
        heights = [max(ln.y_max - ln.y_min, 1.0) for ln in ordered]
        median_h = float(np.median(heights))
        gap_thresh = max(median_h * 1.6, 10.0)

        clusters: List[List[_OcrLine]] = []
        cur: List[_OcrLine] = [ordered[0]]
        for prev, nxt in zip(ordered, ordered[1:]):
            if (nxt.cy - prev.cy) <= gap_thresh:
                cur.append(nxt)
            else:
                clusters.append(cur)
                cur = [nxt]
        clusters.append(cur)

        messages: List[ChatMessage] = []
        for cluster in clusters:
            text = " ".join(ln.text for ln in cluster if ln.text).strip()
            if not text:
                continue
            xmin = min(ln.x_min for ln in cluster)
            xmax = max(ln.x_max for ln in cluster)
            ymin = min(ln.y_min for ln in cluster)
            ymax = max(ln.y_max for ln in cluster)
            cx = float(np.mean([ln.cx for ln in cluster]))
            cy = float(np.mean([ln.cy for ln in cluster]))
            conf = float(np.mean([ln.score for ln in cluster]))
            low_conf = all(ln.low_conf for ln in cluster)

            side, side_conf = self._decide_side(
                cx, xmin, xmax, ymin, ymax, image, anchors)

            messages.append(ChatMessage(
                text=text, side=side, y_center=cy, confidence=conf,
                side_confidence=side_conf, x_min=xmin, x_max=xmax,
                y_min=ymin, y_max=ymax, low_conf=low_conf,
            ))

        # 系统消息：跨聊天区宽度 60% 以上且居中
        for m in messages:
            span = m.x_max - m.x_min
            if span > anchors.chat_width * 0.6:
                m.side = "center"
                m.side_confidence = 0.75

        messages.sort(key=lambda m: m.y_center)
        return messages

    # -- 左右归属：三重证据 ---------------------------------------------
    def _decide_side(self, cx: float, xmin: float, xmax: float,
                     ymin: float, ymax: float, image: np.ndarray,
                     anchors: LayoutAnchors) -> Tuple[str, float]:
        """判定气泡归属：背景色 > 聊天区中线 > 右对齐。

        旧实现用「整图宽度 52%」作阈值，在聊天区偏右时会把自己发的消息
        误判为对方，这里改用聊天区自身中线。
        """
        # 证据 2：聊天区中线
        by_mid = "right" if cx > anchors.chat_mid else "left"

        # 证据 3：右对齐（自己发的气泡贴聊天区右缘）
        right_margin = anchors.chat_right - xmax
        left_margin = xmin - anchors.chat_left
        by_align = "right" if right_margin < left_margin else "left"

        # 证据 1（最强）：气泡背景色
        bg = self._sample_bubble_bg(image, xmin, xmax, ymin, ymax)
        if bg is not None:
            b, g, r = bg
            # 绿色系（微信自己发的气泡）-> 自己
            if g > r + 14 and g > b + 14 and g > 110:
                return "right", 0.95
            # 近白/浅灰 -> 对方
            if r > 195 and abs(r - g) < 14 and abs(g - b) < 14:
                return "left", 0.92

        if by_mid == by_align:
            return by_mid, 0.80
        return by_mid, 0.55

    @staticmethod
    def _sample_bubble_bg(image: np.ndarray, xmin: float, xmax: float,
                          ymin: float, ymax: float) -> Optional[Tuple[float, float, float]]:
        """在文字框外围采样气泡背景色（取中位数，文字像素为少数）。"""
        try:
            h, w = image.shape[:2]
            pad = 4
            x0 = max(0, int(xmin) - pad)
            x1 = min(w, int(xmax) + pad)
            y0 = max(0, int(ymin) - pad)
            y1 = min(h, int(ymax) + pad)
            roi = image[y0:y1, x0:x1]
            if roi.size == 0:
                return None
            med = np.median(roi.reshape(-1, 3), axis=0)
            return float(med[0]), float(med[1]), float(med[2])
        except Exception:
            return None

    # -- 派生意图/发送者 -----------------------------------------------
    def _derive(self, analysis: ChatAnalysis, messages: List[ChatMessage], w: int) -> None:
        # 关键字段只用非低置信气泡，避免把模糊识别结果当成客户消息
        reliable = [m for m in messages if not m.low_conf]
        pool = reliable if reliable else messages

        # 把系统提示行（时间戳/撤回等）从"最后一条"判定中剔除
        meaningful = [m for m in pool if not _is_standalone_system(m.text)]
        if not meaningful:
            meaningful = pool

        customer_msgs = [m for m in meaningful if m.side == "left"]
        if customer_msgs:
            analysis.last_customer_message = customer_msgs[-1].text
        if meaningful:
            analysis.is_self_latest = (meaningful[-1].side == "right")
        analysis.sender_is_customer = not analysis.is_self_latest

        if analysis.last_customer_message:
            analysis.intent = "customer_message"
        elif analysis.draft_text:
            analysis.intent = "draft_only"
        else:
            analysis.intent = "idle"

    # -- 感知置信度 -----------------------------------------------------
    def _perception_confidence(self, analysis: ChatAnalysis,
                               verdict: ViewVerdict,
                               anchors: LayoutAnchors,
                               messages: List[ChatMessage]) -> float:
        """综合视图/布局/OCR/归属的感知置信度（0~1），供证据门控使用。"""
        conf = verdict.confidence * 0.35
        conf += float(anchors.confidence) * 0.15
        if analysis.current_contact:
            conf += 0.25
        if messages:
            avg_ocr = float(np.mean([m.confidence for m in messages]))
            avg_side = float(np.mean([m.side_confidence for m in messages]))
            conf += min(1.0, avg_ocr) * 0.15
            conf += min(1.0, avg_side) * 0.10
        if "low_ocr_score" in analysis.low_conf_flags:
            conf -= 0.08
        return round(max(0.0, min(1.0, conf)), 3)

    # -- 工具 -----------------------------------------------------------
    @staticmethod
    def _to_array(image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                return np.array(image.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            pass
        return np.asarray(image)


__all__ = ["WechatScreenReader", "ChatAnalysis", "ChatMessage"]
