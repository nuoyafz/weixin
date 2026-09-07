"""文本意图识别：从消息文本中归纳业务意图（price/version/feature/stability/risk/...）。"""
from __future__ import annotations

import re

_INTENT_ALIASES = {
    "price": ("价格", "多少钱", "报价", "费用", "收费", "价钱", "预算", "套餐", "贵不贵", "贵吗",
              "太贵", "很贵", "这么贵", "那么贵", "便宜", "优惠", "折扣"),
    "version": ("版本", "什么版本", "哪些版本", "哪个版本", "有几个版本", "都有什么版本",
                "有什么版本", "支持版本", "型号"),
    "feature": ("功能", "核心功能", "有什么用", "能做什么", "支持什么", "有哪些功能"),
    "stability": ("稳定", "稳不稳定", "乱回", "回复错误", "回错", "出错"),
    "risk": ("安全", "风险"),
}
_SINGLE_CHAR_ALIASES = {
    "price": ("贵",),
}
_PUNCT_RE = re.compile(r"[\s，,。！？!?；;：:、~～\-_（）()【】\[\]""''']+")

# =====================================================================
# 扩展意图/语义关键词（原版 conversation_policy 范围）
# =====================================================================
_KEYWORD_YES = {"是", "是的", "对", "对的", "嗯", "好", "好的", "可以", "行", "ok", "yes", "yep", "yeah", "要", "需要", "确认", "没问题"}
_KEYWORD_NO = {"不", "不是", "不对", "否", "不要", "不用", "不需要", "no", "nope", "算了", "不用了", "取消", "退订", "TD"}
_KEYWORD_GREETING = {"你好", "您好", "hi", "hello", "hey", "在吗", "在不在", "在么", "在不", "在么"}
_KEYWORD_THANKS = {"谢谢", "感谢", "多谢", "辛苦了", "太感谢了", "谢谢您", "谢谢了", "3q", "thx", "thanks", "thank"}
_KEYWORD_FAREWELL = {"再见", "拜拜", "bye", "88", "886", "晚安", "回头聊", "下次聊", "先这样", "就这样"}
_KEYWORD_URL = {"http://", "https://", "www.", ".com", ".cn", ".net", ".org"}
_KEYWORD_PHONE = {"电话", "手机", "手机号", "联系方式", "打给我", "怎么联系", "联系我", "回电", "回电话"}
_KEYWORD_EMAIL = {"邮箱", "email", "邮件", "发邮件", "发我邮箱", "邮箱地址"}
_KEYWORD_SOCIAL = {"微信", "加我", "加微信", "微信号", "wx", "vx", "二维码", "加好友", "扫码", "加我好友"}
_KEYWORD_WECHAT_GROUP = {"群", "拉我", "拉群", "进群", "加群", "群聊", "拉我进群"}
_KEYWORD_URGENT_HUMAN = {"投诉", "退款", "差评", "举报", "报警", "起诉", "律师", "法院", "12315", "消费者", "维权", "诈骗", "骗人", "骗子", "假货", "骗钱", "被骗", "受骗", "上当"}
_KEYWORD_GROUP_NOISE = {"哈哈", "呵呵", "嗯", "哦", "收到", "了解了", "明白", "知道", "好的", "是的", "对的", "可以的", "没问题", "牛", "厉害", "靠谱", "666", "给力", "不错", "打卡", "签到", "冒泡", "有人吗", "都在吗", "没问题", "可以", "行", "好", "噢", "啊", "没错", "知道了", "了解"}


def compact_text(value: str) -> str:
    if not value:
        return ""
    return _PUNCT_RE.sub("", str(value)).strip()


def detect_text_intents(text: str) -> list:
    """返回命中的意图标签列表，例如 ["price"] 或 ["feature","version"]。"""
    if not text:
        return []
    compact = compact_text(text)
    if not compact:
        return []
    intents = []
    for intent, aliases in _INTENT_ALIASES.items():
        for a in aliases:
            if a in compact and a not in (""):
                intents.append(intent)
                break
        else:
            if len(compact) <= 6:
                for sa in _SINGLE_CHAR_ALIASES.get(intent, ()):
                    if sa in compact:
                        intents.append(intent)
                        break
    return intents


def is_yes(text: str) -> bool:
    """判断是否为肯定回答。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_YES)


def is_no(text: str) -> bool:
    """判断是否为否定回答。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_NO)


def is_greeting(text: str) -> bool:
    """判断是否为问候语。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_GREETING)


def is_thanks(text: str) -> bool:
    """判断是否为感谢语。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_THANKS)


def is_farewell(text: str) -> bool:
    """判断是否为告别语。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_FAREWELL)


def has_url(text: str) -> bool:
    """判断是否包含 URL。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_URL)


def has_phone(text: str) -> bool:
    """判断是否包含电话相关关键词。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_PHONE)


def has_email(text: str) -> bool:
    """判断是否包含邮箱相关关键词。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_EMAIL)


def has_social(text: str) -> bool:
    """判断是否包含社交账号相关关键词。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_SOCIAL)


def has_wechat_group(text: str) -> bool:
    """判断是否包含群聊相关关键词。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_WECHAT_GROUP)


def is_urgent_human(text: str) -> bool:
    """判断是否为紧急需要人工处理的关键词。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_URGENT_HUMAN)


def is_group_noise(text: str) -> bool:
    """判断是否为群聊噪声（闲聊/内部对话）。"""
    if not text:
        return False
    return any(k in text for k in _KEYWORD_GROUP_NOISE)


def contains_at_mention(text: str) -> bool:
    """判断消息是否包含 @ 提及。"""
    if not text:
        return False
    return bool(re.search(r"@\S+", text))


def is_target_at_mentioned(text: str, target_name: str) -> bool:
    """判断消息是否 @ 了指定的目标。"""
    if not text or not target_name:
        return False
    for m in re.finditer(r"@(\S+)", text):
        if m.group(1) == target_name or target_name in m.group(1):
            return True
    return False