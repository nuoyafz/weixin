"""跳过联系人（黑名单）匹配。

真相源 = 软件内置 BUILTIN_CONTACT_BLACKLIST + config 的
wechat.system_contacts / wechat.contact_blacklist / reply_rules.skip_contacts
（用户在 UI「接管范围 → 管理跳过列表」维护，可在 config.yaml 继续追加自定义）。
匹配两层：
  1) 精确 substring
  2) 拼音模糊匹配（兜 OCR/VLM 把"妈咪"读成"妈眯"的同音错读）
"""
from __future__ import annotations

from typing import Any, Iterable

try:
    from pypinyin import lazy_pinyin, Style
    _PYPINYIN_OK = True
except Exception:  # pragma: no cover - 可选依赖
    _PYPINYIN_OK = False


# =====================================================================
# 软件内置黑名单
# 这些联系人（电商 / 外卖 / 出行 / 短视频 / 快递 / 运营商 / 银行 / 支付 /
# 公众号 / 系统号 / 折叠分组等）一律不自动回复。
# 用户仍可在 config.yaml 的 wechat.contact_blacklist 中继续自定义追加。
# =====================================================================
BUILTIN_CONTACT_BLACKLIST = frozenset([
    # 电商 / 外卖 / 出行 / 短视频
    "拼多多", "瑞幸咖啡", "美团", "饿了么", "滴滴", "抖音", "快手",
    "京东", "淘宝", "天猫",
    # 公众号 / 订阅号 / 服务号 / 微信系统号
    "订阅号", "服务号", "公众号", "微信团队", "微信支付", "微信运动",
    "文件传输助手", "腾讯新闻", "腾讯文档",
    # 折叠分组 / 群相关
    "折叠的聊天", "群助手", "群消息",
    # 快递 / 物流
    "京东快递", "顺丰速运", "菜鸟",
    # 运营商 / 银行 / 客服号
    "中国移动", "中国联通", "中国电信", "10086", "10000", "95588",
    # 支付
    "支付宝",
])


def _read_config_lists(config: Any) -> list:
    """从 config（dict 或 Settings 对象）读取所有黑名单来源并合并去重。

    来源：wechat.system_contacts、wechat.contact_blacklist、
    reply_rules.skip_contacts、顶层 skip_contacts。
    """
    out: list = []
    if not config:
        return out

    wechat = config.get("wechat", {}) if isinstance(config, dict) else getattr(config, "wechat", None)
    if wechat is not None:
        if isinstance(wechat, dict):
            out.extend(wechat.get("system_contacts", []) or [])
            out.extend(wechat.get("contact_blacklist", []) or [])
        else:
            out.extend(getattr(wechat, "system_contacts", []) or [])
            out.extend(getattr(wechat, "contact_blacklist", []) or [])

    rr = config.get("reply_rules", {}) if isinstance(config, dict) else getattr(config, "reply_rules", None)
    if rr is not None:
        if isinstance(rr, dict):
            out.extend(rr.get("skip_contacts", []) or [])
        else:
            out.extend(getattr(rr, "skip_contacts", []) or [])

    sc2 = config.get("skip_contacts", []) if isinstance(config, dict) else getattr(config, "skip_contacts", [])
    out.extend(sc2 or [])

    return [str(x).strip() for x in out if x]


def is_blacklisted_contact(contact: Any, config: Any = None) -> bool:
    """判定联系人是否命中黑名单（内置 + 配置），兼容 dict / Settings。

    先查内置全集（精确匹配），再查配置来源（substring + 拼音兜底）。
    """
    name = _normalize(contact)
    if not name:
        return False

    if name in BUILTIN_CONTACT_BLACKLIST:
        return True

    cfg_list = _read_config_lists(config)

    for raw in cfg_list:
        e = _normalize(raw)
        if not e:
            continue
        if e in name or name in e:
            return True

    # 拼音兜底：只对 1~6 字符的短名的整体拼音做相等比较
    if _PYPINYIN_OK and 1 <= len(name) <= 6:
        name_py = _to_pinyin(name)
        if not name_py:
            return False
        for raw in cfg_list:
            e = _normalize(raw)
            if not e or len(e) > 6:
                continue
            if _to_pinyin(e) == name_py:
                return True
    return False


def load_skip_list(config: Any) -> list:
    """从 config 读 wechat.system_contacts，去空、去 None、去重、保序。"""
    if not config:
        return []
    contacts = config.get("wechat", {}).get("system_contacts") if isinstance(config, dict) else getattr(config, "system_contacts", None)
    if not contacts:
        return []
    seen = set()
    out = []
    for x in contacts:
        if not x:
            continue
        name = str(x).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize(text: Any) -> str:
    """归一化：去全角/半角空白，全角字母数字转半角。"""
    if not text:
        return ""
    out = []
    for ch in str(text):
        code = ord(ch)
        # 全角空格
        if code == 12288:
            continue
        # 全角 ASCII 转半角
        if 65281 <= code <= 65374:
            out.append(chr(code - 65248))
            continue
        if ch.isspace():
            continue
        out.append(ch)
    return "".join(out).strip()


def _to_pinyin(text: str) -> str:
    """汉字 → 拼音字符串（小写、无音调、连写）。非汉字原样保留。"""
    if not _PYPINYIN_OK or not text:
        return ""
    try:
        return "".join(lazy_pinyin(text, style=Style.NORMAL, errors="default")).lower()
    except Exception:
        return ""


def is_skipped_contact(contact: Any, config: Any) -> bool:
    """contact 是否命中跳过列表（内置黑名单 + 配置来源）。

    兼容 dict / Settings；精确匹配内置全集，再按 substring + 拼音兜底匹配配置来源。
    """
    return is_blacklisted_contact(contact, config)


def is_skipped_contact_any(candidates: Iterable, config: Any) -> bool:
    """多个候选名字，命中任何一个就 True。"""
    for c in candidates:
        if is_skipped_contact(c, config):
            return True
    return False