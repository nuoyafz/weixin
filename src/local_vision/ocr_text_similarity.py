"""文本相似度计算工具：编辑距离 + 字符重叠。

供 OCR 后处理与联系人识别复用，例如判断两个 OCR 行/联系人名是否属于同一条文本。
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

# 关键语义标记（对齐原版 _CRITICAL_SEMANTIC_MARKERS）
# 当这些词出现/消失时，即使其余文本相似也视为不同消息
_CRITICAL_SEMANTIC_MARKERS = (
    "不", "没", "无", "别", "取消", "停止", "退款", "退货",
    "不能", "不要", "不是", "可以",
)


def edit_distance(a: str, b: str, ignore_case: bool = True,
                  ignore_space: bool = True) -> int:
    """计算两个字符串的编辑距离（Levenshtein 距离）。

    可选忽略大小写与空白字符，更适合中英文混排的 OCR 文本。
    """
    if not a and not b:
        return 0
    a = _normalize(a, ignore_case, ignore_space)
    b = _normalize(b, ignore_case, ignore_space)

    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # 滚动数组优化：只保留两行
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,          # 删除 a[i-1]
                cur[j - 1] + 1,       # 插入 b[j-1]
                prev[j - 1] + cost,   # 替换/相同
            )
        prev = cur
    return prev[lb]


def _normalize(s: str, ignore_case: bool, ignore_space: bool) -> str:
    if ignore_space:
        s = "".join(s.split())
    if ignore_case:
        s = s.lower()
    return s


def char_overlap_ratio(a: str, b: str, ignore_case: bool = True) -> float:
    """字符重叠比例：两文本共同字符数（取并集）占较长文本的比例，介于 0~1。"""
    if not a or not b:
        return 0.0
    if ignore_case:
        a, b = a.lower(), b.lower()
    common = len(set(a) & set(b))
    return common / max(len(a), len(b))


def text_similarity(a: str, b: str, ignore_case: bool = True,
                    ignore_space: bool = True) -> float:
    """综合编辑距离与字符重叠的两个文本相似度，返回 0~1。

    1.0 表示完全相同；
    结合 1 - normalized_edit_distance 与 char_overlap_ratio，取加权平均。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    na, nb = _normalize(a, ignore_case, ignore_space), _normalize(b, ignore_case, ignore_space)
    if na == nb:
        return 1.0

    max_len = max(len(na), len(nb))
    edit_sim = 1.0 - (edit_distance(na, nb, ignore_case=False, ignore_space=False) / max_len)
    overlap_sim = char_overlap_ratio(na, nb, ignore_case=False)
    # 编辑距离更能反映字符级接近程度，权重更高
    return 0.6 * edit_sim + 0.4 * overlap_sim


def is_similar(a: str, b: str, threshold: float = 0.8, **kwargs) -> bool:
    """相似度是否达到阈值（用于 OCR 去重 / 联系人判等）。"""
    return text_similarity(a, b, **kwargs) >= threshold


# =====================================================================
# 对齐原版 looks_like_same_ocr_text 及其辅助函数
# =====================================================================

def _compact_text(value: Any) -> str:
    """压缩文本：小写 + 去除非字母数字中文的字符。"""
    text = str(value).casefold() if value else ""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _protected_tokens(value: str) -> tuple:
    """提取受保护的数字/英文标识符。"""
    return tuple(re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?", value or ""))


def _critical_marker_state(value: str) -> tuple:
    """检查关键语义标记是否存在。"""
    return tuple(marker in (value or "") for marker in _CRITICAL_SEMANTIC_MARKERS)


def looks_like_same_ocr_text(left: Any, right: Any) -> bool:
    """保守匹配 OCR 微小变体，但不隐藏真正的语义变化。

    对齐原版 app.local_vision.ocr_text_similarity.looks_like_same_ocr_text。

    保护数字、ASCII 标识符和重要否定/取消词。
    当这些词变化时，即使其余句子几乎相同也视为新消息。
    """
    first = _compact_text(left)
    second = _compact_text(right)

    if not first or not second:
        return False

    if first == second:
        return True

    # 长度差异过大直接判定不同
    shorter = min(len(first), len(second))
    longer = max(len(first), len(second))
    if longer > 0 and shorter / longer < 0.6:
        return False

    # 保护数字/英文标识符
    if _protected_tokens(left) != _protected_tokens(right):
        return False

    # 保护关键语义标记
    if _critical_marker_state(str(left)) != _critical_marker_state(str(right)):
        return False

    # SequenceMatcher 相似度
    matcher = SequenceMatcher(None, first, second)
    if matcher.ratio() < 0.86:
        return False

    # 变化宽度检查：允许少量字符变化
    changed_width = sum(
        max(left_end - left_start, right_end - right_start)
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes()
        if tag != "equal"
    )

    # 根据文本长度动态调整允许的变化宽度
    if longer <= 6:
        allowed_width = 2
    elif longer <= 12:
        allowed_width = 3
    else:
        allowed_width = 4

    return changed_width <= allowed_width