"""意图关键词库与对齐函数。

按业务意图组织中文关键词，提供"给定一句话 -> 最佳意图 + 覆盖得分"的对齐算法。
意图可覆盖询价 / 版本 / 功能 / 售后 / 购买意向等常见销售场景，也可被外部覆盖。
纯标准库，无副作用。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 意图 -> 触发关键词（按优先级排列，越靠前越具区分度）。
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "price": ["多少钱", "价格", "报价", "收费", "费用", "怎么收费", "便宜", "贵不贵", "价位", "价格表", "多少钱一个月"],
    "version": ["版本", "有哪些版本", "什么版本", "旗舰版", "标准版", "体验版", "基础版", "专业版", "V1", "V2", "对比资料"],
    "features": ["功能", "有什么用", "功能介绍", "都能做什么", "能做", "支持", "有哪些功能", "怎么用", "使用教程", "操作"],
    "trial": ["试用", "免费试", "体验", "测试版", "免费版", "demo", "试一下"],
    "purchase": ["购买", "怎么买", "下单", "付款", "付费", "买一个", "申请开通", "购买链接", "转账", "扫码支付"],
    "after_sales": ["售后", "退款", "退换", "报错", "坏了", "维修", "发票", "保修", "不工作", "无法使用"],
    "greeting": ["你好", "您好", "在吗", "在不在", "hi", "hello", "早上好", "下午好", "晚上好", "哈喽"],
    "farewell": ["再见", "拜拜", "回聊", "辛苦了", "谢谢", "感谢", "麻烦你", "好的，明白了", "嗯"],
    "lead_capture": ["需要", "感兴趣", "想了解", "怎么合作", "联系方式", "在哪里", "加微信", "联系我", "预约"],
    "contact": ["电话", "手机", "邮箱", "qq", "微信多少", "公司地址", "联系人", "客服"],
}

# 用于命中的"归一化"：去除首尾空白、标点最小化干扰。
_STRIP = str.maketrans("", "", "，。！？!?::，；;、 \t\n\r\u3000")


def _norm(text: str) -> str:
    return (text or "").translate(_STRIP).lower()


def match_intent_keywords(text: str, intents: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """返回 {intent: 命中的关键词列表}。只做子串包含匹配，不做分词。"""
    kw_map = intents if intents is not None else INTENT_KEYWORDS
    norm_text = _norm(text)
    if not norm_text:
        return {}
    hits: Dict[str, List[str]] = {}
    for intent, keywords in kw_map.items():
        matched = [k for k in keywords if k and _norm(k) and _norm(k) in norm_text]
        if matched:
            hits[intent] = matched
    return hits


def align_intent(
    text: str,
    intents: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
) -> Tuple[Optional[str], float]:
    """对齐最佳意图。

    Returns:
        (intent, score)：intent 为 None 表示未命中任何关键词；
        score 大致 = 命中最长关键词长度 / 输入长度，加轻微内置优先级偏移，
        用于在多个意图同时命中时做粗略决策（仅表示覆盖度，非概率）。
    """
    kw_map = intents if intents is not None else INTENT_KEYWORDS
    hits = match_intent_keywords(text, kw_map)
    if not hits:
        return None, 0.0

    norm_len = max(1, len(_norm(text)))
    best_intent: Optional[str] = None
    best_score = 0.0
    order = list(kw_map.keys())

    for intent, matched in hits.items():
        longest = max((len(_norm(k)) for k in matched), default=0)
        coverage = longest / norm_len
        # 轻微优先级奖励：关键词列表越靠前越有区分度。
        priority = 1.0 - (order.index(intent) * 0.01)
        score = coverage * priority
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score < threshold:
        return None, best_score
    return best_intent, best_score


def top_intents(text: str, intents: Optional[Dict[str, List[str]]] = None,
                limit: int = 3) -> List[Tuple[str, float]]:
    """返回按得分降序的多个意图，便于决策层做二次判断。"""
    kw_map = intents if intents is not None else INTENT_KEYWORDS
    hits = match_intent_keywords(text, kw_map)
    if not hits:
        return []
    norm_len = max(1, len(_norm(text)))
    order = list(kw_map.keys())
    ranked = []
    for intent, matched in hits.items():
        longest = max((len(_norm(k)) for k in matched), default=0)
        # 用得分，无需阈值
        coverage = longest / norm_len
        priority = 1.0 - (order.index(intent) * 0.01)
        ranked.append((intent, coverage * priority))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:limit]