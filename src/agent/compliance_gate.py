"""发送前合规校验门：抽取回复中的价格事实与知识库规范值比对。

目的：即使「开放/异议」类走 LLM 生成，也应防止模型编造与业务资料不符的
价格/收费事实。本门在 wechat_sender 真正打字发送前调用——若回复里出现知识库
规范价格集合中不存在的货币数字（如把 ¥99 说成 ¥199），判定不合规并拦截转人工，
避免「说错价」这类对客服产品最致命的低级错误。

纯本地、不调 API。无知识库参照（FAQ 未加载）时一律放行，避免误杀。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

from ..brain.faq_reply import FaqReplyMatcher

logger = logging.getLogger(__name__)

# 只把「带货币单位/周期」的数字视为价格事实，避免误伤 "1000+ 客户" 这类表述。
_PRICE_RE = re.compile(
    r"(?:[¥￥]\s*|RMB\s*)?(\d+(?:\.\d+)?)\s*(?:元|块钱|块|/(?:月|年|天|季度))",
    re.IGNORECASE,
)


def _extract_prices(text: str) -> set:
    """抽取文本中的价格数字（归一化 token，如 '99/月'、'299'）。"""
    out: set = set()
    for num in _PRICE_RE.findall(text):
        n = str(num).replace(" ", "")
        out.add(n)
        for u in ("", "元", "/月", "/年", "/天"):
            out.add(f"{n}{u}")
    return out


class ComplianceGate:
    def __init__(self, config: Any = None):
        self._config = config
        self._canonical: Optional[set] = None

    def _ensure(self) -> set:
        if self._canonical is not None:
            return self._canonical
        self._canonical = set()
        try:
            matcher = FaqReplyMatcher(self._config)
            for it in matcher._items:
                self._canonical |= _extract_prices(it.get("answer", ""))
        except Exception as e:  # 加载失败不阻断发送，仅失去校验能力
            logger.warning("[compliance] FAQ 规范价加载失败，合规门降级放行: %s", e)
            self._canonical = set()
        return self._canonical

    def check(self, reply: str) -> Tuple[bool, str]:
        """返回 (ok, reason)。ok=False 表示回复含与知识库不符的价格事实。"""
        if not reply:
            return True, ""
        canonical = self._ensure()
        if not canonical:
            return True, ""  # 无可参照规范价，放行
        found = _extract_prices(reply)
        bad = sorted(f for f in found if f not in canonical and f.rstrip("/月年天") not in canonical)
        if bad:
            return False, f"回复含知识库未收录的价格事实: {', '.join(bad)}"
        return True, ""
