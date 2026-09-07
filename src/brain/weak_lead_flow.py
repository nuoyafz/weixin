"""弱线索（weak lead）跟进 + 完整留资漏斗状态机。

包含两部分：
1) WeakLeadManager   —— 从原版反编译移植：处理客户拒绝完整手机号 →
                        改问姓氏 → 再问手机号后4位，最终生成模糊凭证。
2) WeakLeadFollowUp  —— 逆向版原有的静默提醒定时器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Part 1: WeakLeadManager (ported from original decompiled source)
# ---------------------------------------------------------------------------

REFUSE_PHONE_KEYWORDS = (
    "不方便留电话", "不想留手机号", "不留电话", "电话就不留了",
    "直接给我地址", "我自己去", "不用留电话", "不提供手机号",
    "不想给电话", "不想留电话", "不方便给手机号",
)

CHINESE_SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉"
    "岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧"
    "尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席"
    "季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管"
    "卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢滑裴陆荣翁荀"
    "羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬"
    "全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟"
    "薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳"
    "逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温庄晏柴瞿阎"
)

_ASK_FULL_PHONE_KEYWORDS = (
    "完整手机号", "完整电话", "手机号码", "电话号码",
    "留个电话", "留一下电话",
)


class WeakLeadManager:
    """Enforces the weak-lead flow and stores fuzzy lead state locally."""

    def __init__(self, config=None, store_address: str = "", leads_dir: Path = None) -> None:
        self._config = config or {}
        self.store_address = store_address or ""
        base = Path(leads_dir or Path("data") / "leads")
        self.leads_dir = base
        self.leads_dir.mkdir(parents=True, exist_ok=True)

    def apply(self, analysis: dict, reply: str):
        analysis = analysis or {}
        msg = analysis.get("latest_message") or {}
        if msg.get("sender") != "customer":
            return analysis, reply

        if analysis.get("decision") == "no_reply":
            reply = str(reply or "").strip()
            if not reply:
                return analysis, reply

        contact = str(analysis.get("current_contact") or "unknown").strip() \
                  or "unknown"
        content = str(msg.get("content") or "")
        state = self._load_state(contact)

        if analysis.get("intent") == "refuse_full_phone" or \
           self._contains_any(content, REFUSE_PHONE_KEYWORDS):
            state.update({
                "contact": contact,
                "stage": "ask_surname",
                "refused_full_phone": True,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(contact, state)
            analysis.update({
                "intent": "refuse_full_phone",
                "decision": "weak_lead_draft",
                "action": "weak_lead_draft",
                "should_reply": True,
                "lead_stage": "ask_surname",
                "fuzzy_lead": True,
                "fallback_reply_pending": True,
                "reply_draft": "",
                "reason": "客户拒绝完整手机号，交给模型生成弱留资回复。",
            })
            return analysis, ""

        surname = self._extract_surname(content)
        tail4 = self._extract_tail4(content)

        if state.get("stage") == "ask_surname" and surname:
            state.update({
                "stage": "ask_tail4",
                "surname": surname,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(contact, state)
            analysis.update({
                "decision": "weak_lead_draft",
                "action": "weak_lead_draft",
                "should_reply": True,
                "lead_stage": "ask_tail4",
                "lead_name": surname,
                "fuzzy_lead": True,
                "fallback_reply_pending": True,
                "reply_draft": "",
                "reason": "已收姓氏，下一步询问尾号4位。",
            })
            return analysis, ""

        if state.get("stage") in {"ask_tail4", "ask_surname"} and tail4:
            sur = state.get("surname") or surname or "客户"
            voucher = f"{sur}{tail4}"
            state.update({
                "stage": "fuzzy_voucher",
                "surname": sur,
                "tail4": tail4,
                "voucher": voucher,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(contact, state)
            analysis.update({
                "decision": "reply",
                "action": "reply",
                "should_reply": True,
                "lead_stage": "fuzzy_voucher",
                "lead_name": sur,
                "lead_tail4": tail4,
                "fuzzy_lead": True,
                "voucher": voucher,
                "fallback_reply_pending": True,
                "reply_draft": "",
                "reason": "姓氏+尾号齐备，生成模糊线索备注。",
            })
            return analysis, ""

        if state.get("refused_full_phone") and reply and \
           self._asks_full_phone(reply):
            analysis["reply_draft"] = ""
            analysis["decision"] = "weak_lead_draft"
            analysis["action"] = "weak_lead_draft"
            analysis["should_reply"] = True
            analysis["fallback_reply_pending"] = True
            analysis["reason"] = "检测到回复要求完整手机号但客户已拒绝，重写为弱留资话术。"
            return analysis, ""

        return analysis, reply

    def process(self, analysis: dict) -> dict:
        """处理弱线索流（对齐原版 WeakLeadManager.process）。"""
        analysis, _ = self.apply(analysis, "")
        return {"lead_stage": analysis.get("intent", "")}

    def _contact_key(self, contact: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_",
                      contact or "unknown")
        return safe[:80] or "unknown"

    def _state_path(self, contact: str) -> Path:
        return self.leads_dir / f"{self._contact_key(contact)}.json"

    def _load_state(self, contact: str) -> dict:
        p = self._state_path(contact)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, contact: str, state: dict) -> None:
        try:
            self._state_path(contact).write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:
            pass

    def _contains_any(self, text: str, keywords) -> bool:
        return any(k in text for k in keywords)

    def _extract_surname(self, text: str) -> str:
        text = text.strip()
        patterns = [
            r"我姓([\u4e00-\u9fff]{1,2})",
            r"姓([\u4e00-\u9fff]{1,2})",
            r"叫我([\u4e00-\u9fff]{1,3})",
            r"我是([\u4e00-\u9fff]{1,3})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m and m.group(1)[0] in CHINESE_SURNAMES:
                return m.group(1)[0]
        if len(text) <= 3 and text and text[0] in CHINESE_SURNAMES:
            return text[0]
        return ""

    def _extract_tail4(self, text: str) -> str:
        m = re.search(r"(?<!\d)(\d{4})(?!\d)", text or "")
        return m.group(1) if m else ""

    def _asks_full_phone(self, reply: str) -> bool:
        text = reply or ""
        has_ask = any(k in text for k in _ASK_FULL_PHONE_KEYWORDS)
        mentions_partial = ("后 4" in text) or ("后4" in text)
        return has_ask and not mentions_partial


@dataclass
class FollowUpTodo:
    should_follow_up: bool = False
    contact: str = ""
    reply: str = ""
    reason: str = ""
    quiet_hours: float = 0.0

    def to_dict(self) -> dict:
        return {
            "should_follow_up": self.should_follow_up,
            "contact": self.contact,
            "reply": self.reply,
            "reason": self.reason,
            "quiet_hours": self.quiet_hours,
        }


class WeakLeadFollowUp:
    """弱线索跟进判定：按静默时长阈值生成跟进话术。"""

    # 默认静默阈值（小时），配置缺省时使用
    _DEFAULT_QUIET_HOURS = 24
    # 分档跟进话术
    _SCRIPTS = {
        24: (
            "您好~ 之前沟通的场景不知道您考虑得怎么样了？"
            "如果还有不清楚的地方随时问我，我帮您再梳理下。"
        ),
        72: (
            "您好，打扰啦~ 看您这边好像暂时没空，方案我仍然帮您保留着。"
            "要是哪天想起来了，随时喊我，随时都在。"
        ),
        168: (
            "您好，许久没联系啦~ 想知道这个方案是否还符合您的需求？"
            "如果需要，我可以帮您做个简单演示，几分钟就能看完。"
        ),
    }
    _DEFAULT_TIERS = (24, 72, 168)
    _CMT_CONTACTS = ("客户", "顾客", "潜在", "新客", "线索")

    def __init__(self, config: Any = None, quiet_hours: Optional[float] = None):
        self.config = config
        self.quiet_hours = quiet_hours if quiet_hours is not None else self._read_quiet_hours()

    def _read_quiet_hours(self) -> float:
        """从 config.weak_lead.quiet_hours 读取（dict 或对象兼容）。"""
        if not self.config:
            return float(self._DEFAULT_QUIET_HOURS)
        if isinstance(self.config, dict):
            v = (self.config.get("weak_lead") or {}).get("quiet_hours")
        else:
            wl = getattr(self.config, "weak_lead", None) or {}
            v = (wl.get("quiet_hours") if isinstance(wl, dict) else
                 getattr(wl, "quiet_hours", None))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(self._DEFAULT_QUIET_HOURS)

    # ------------------------------------------------------------------ 跟进
    def _tier_script(self, quiet_hours: float) -> str:
        """按静默时长选择最贴合的档位话术。"""
        best = self._SCRIPTS[self._DEFAULT_TIERS[0]]
        for tier in self._DEFAULT_TIERS:
            if quiet_hours >= tier:
                best = self._SCRIPTS[tier]
        return best

    def evaluate(self, contact: str, last_active_at: Any,
                 now: Any = None) -> FollowUpTodo:
        """判定是否需要对某客户跟进。

        last_active_at / now 支持 datetime 或 epoch seconds（时间戳）。
        """
        try:
            last = _coerce_dt(last_active_at)
            cur = _coerce_dt(now) if now is not None else datetime.now()
        except (TypeError, ValueError, OverflowError):
            return FollowUpTodo(should_follow_up=False, contact=contact,
                                reason="invalid_time")
        if last > cur:
            return FollowUpTodo(should_follow_up=False, contact=contact,
                                reason="future_time")
        quiet_hours = (cur - last).total_seconds() / 3600.0
        if quiet_hours < self.quiet_hours:
            return FollowUpTodo(should_follow_up=False, contact=contact,
                                quiet_hours=quiet_hours)
        return FollowUpTodo(
            should_follow_up=True,
            contact=contact,
            reply=self._tier_script(quiet_hours),
            reason="quiet_too_long",
            quiet_hours=quiet_hours,
        )


def _coerce_dt(value: Any) -> datetime:
    """把 datetime 或 epoch seconds 统一转为 datetime。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    raise TypeError(f"unsupported time type: {type(value)}")


def is_generic_contact(contact: str) -> bool:
    """判断是否存在占用「客户/线索」等泛化命名（跟随进名单判断用）。"""
    return bool(contact) and any(k in contact for k in _CMT_CONTACTS)