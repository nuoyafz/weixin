"""local_vision：微信视觉相关（OCR 结果后处理、布局解析、文本相似度）。

依赖该包前请确保已在 src 包环境下（相对导入依赖 src 为 Python 包）。
"""

from .wechat_ocr_parser import WeChatOCRParser
from .ocr_text_similarity import (
    edit_distance,
    char_overlap_ratio,
    text_similarity,
    is_similar,
)
from .wechat_layout_parser import (
    WechatLayoutParser,
    WechatLayoutResult,
    ChunkMessage,
    LayoutRegion,
    MessageSideLabel,
)

__all__ = [
    "WeChatOCRParser",
    "edit_distance",
    "char_overlap_ratio",
    "text_similarity",
    "is_similar",
    "WechatLayoutParser",
    "WechatLayoutResult",
    "ChunkMessage",
    "LayoutRegion",
    "MessageSideLabel",
]