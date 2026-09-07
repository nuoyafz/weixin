"""售前 SOP 提示词常量模板。

为各销售阶段（intro / pricing / objection / closing）提供可直接用于
AI 回复或直出的话术模板。纯常量与轻逻辑，不依赖模型。

阶段说明：
  intro    开场/破冰       潜在客户刚接触，介绍产品与价值
  pricing  报价/价格       客户询问价格、费用、预算
  objection 异议处理       客户嫌贵、犹豫、质疑
  closing  促成成交       客户条件齐备，推动下单/约定下一步
"""
from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------ 阶段定义
STAGES = ("intro", "pricing", "objection", "closing")
STAGE_LABELS = {
    "intro": "开场介绍",
    "pricing": "报价询价",
    "objection": "异议处理",
    "closing": "促成成交",
}

# ------------------------------------------------------------------ 各阶段话术（直出）
STAGE_SCRIPTS: dict = {
    "intro": (
        "您好~ 我是您的专属顾问。"
        "我们这套系统主打「快、稳、省」，帮您把日常沟通和跟进自动化，"
        "省下大量重复劳动。方便的话可以简单聊聊您的使用场景，我帮您对号入座。"
    ),
    "pricing": (
        "目前这款是 {price}{units}，费用很透明，没有隐藏收费。"
        "您先了解下功能再决定要不要，我可以给您详细算一算哪种方案最合适。"
    ),
    "objection": (
        "理解您的顾虑，价格确实要结合实际收益来看。"
        "它帮您省下的时间精力，通常很快就能回本，而且我们支持先试用再付款，"
        "风险基本为零。您最担心的是哪一点？我针对性说明下。"
    ),
    "closing": (
        "那天我们一起把需求对清楚了，方案也符合您的预期。"
        "要不我帮您把开通手续办好？今天办理的话，马上就给您安排使用。"
    ),
}

# ------------------------------------------------------------------ 各阶段 AI 提示词
STAGE_SYSTEM_PROMPTS: dict = {
    "intro": (
        "你是专业但亲切的售前顾问，正面对一位刚接触产品的潜在客户。\n"
        "目标：建立信任、引出需求。\n"
        "要求：\n"
        "1. 先自然问候，不要一上来就抛价格\n"
        "2. 用一句话说明产品能带来的价值（省时、省事、可靠）\n"
        "3. 用开放式问题引导客户说出使用场景\n"
        "4. 语气真诚，篇幅简短，像真人聊天\n"
    ),
    "pricing": (
        "你是售前顾问，客户正在询问价格或费用。\n"
        "目标：讲清价格结构并推进购买意愿。\n"
        "{quote_context}\n"  # 由调用方注入已知的价格/套餐信息
        "要求：\n"
        "1. 直接、清晰地回答价格，避免绕弯\n"
        "2. 说明价格的「性价比」，把价格与收益绑定\n"
        "3. 若有多档套餐，简要对比后推荐最贴合客户场景的一档\n"
        "4. 结尾给出一个低门槛的下一步（如先试用/加好友详聊）\n"
    ),
    "objection": (
        "你是售前顾问，客户表现出犹豫、嫌贵或质疑。\n"
        "目标：化解异议，不施压。\n"
        "要求：\n"
        "1. 先共情、认可客户的顾虑，不反驳\n"
        "2. 针对具体异议给出事实或体验保障（试用、按需、售后）\n"
        "3. 不硬推，尊重客户节奏\n"
        "4. 结尾留一个开放缓冲，让对话可以继续\n"
    ),
    "closing": (
        "你是售前顾问，客户已表达意向，条件基本齐备。\n"
        "目标：自然地促成成交或约定明确下一步。\n"
        "要求：\n"
        "1. 回顾已对齐的需求，给客户「选择确认」的轻松感\n"
        "2. 给一个明确的、低摩擦的动作（开通、下单、安排演示）\n"
        "3. 若客户仍需考虑，约定一个回访时间，不强迫\n"
    ),
}


def is_valid_stage(stage: str) -> bool:
    """校验销售阶段名是否合法。"""
    return stage in STAGES


def get_stage_script(stage: str) -> str:
    """返回某阶段的直出话术，未知阶段返回空串。"""
    return STAGE_SCRIPTS.get(stage, "")


def build_stage_script(stage: str, **kwargs: Any) -> str:
    """带插值地构建某阶段话术（如 price / units 由调用方传入）。"""
    return STAGE_SCRIPTS.get(stage, "").format(**kwargs)


def get_stage_prompt(stage: str) -> str:
    """返回某阶段的 AI 系统提示词，未知阶段返回空串。"""
    return STAGE_SYSTEM_PROMPTS.get(stage, "")


def build_stage_prompt(stage: str, quote_context: str = "") -> str:
    """构建某阶段 AI 提示词，可注入报价上下文（quote_context）。"""
    prompt = STAGE_SYSTEM_PROMPTS.get(stage, "")
    if not prompt:
        return ""
    return prompt.format(quote_context=quote_context)