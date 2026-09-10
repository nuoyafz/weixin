# -*- coding: utf-8 -*-
"""「无资料 → 澄清追问」规则离线测试。

背景：原 prompt 规定「没有依据必须 no_reply 且 reply_draft 为空」。
打通 enable_thinking 开关后（关思考=不思考），模型对这种硬规则执行得极其死板，
客户正常咨询（如问价）被判定为 no_reply 而静默忽略 —— 真机表现为「最后回复为空」。

本测试锁定修复后的契约：
  1. 无资料时**不得编造**具体事实（价格/地址/承诺…）；
  2. 无资料时**不得沉默**，必须用一句自然的澄清追问承接；
  3. 「非客户发送 / 广告 / 支付通知 / 跳过名单」等硬门禁 no_reply 不得被削弱。

全部为纯逻辑测试，不联网、不依赖真实密钥，可在 CI 稳定运行。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ALIYUN = "https://llm-xxxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

# 修复前存在、必须彻底消失的「无资料即沉默」文案
OLD_SILENCE_RULES = (
    "没有已保存的业务信息、常见问题、关键词回复或知识库依据时，必须 no_reply 且 reply_draft 为空。",
    "没有依据时 no_reply，reply_draft 为空。",
    "不要生成兜底话术。",
    "资料没有写清楚时 no_reply。",
)


def _client(rag_enabled=True, chitchat=True):
    from src.ai.text_model_client import TextModelClient
    return TextModelClient(
        base_url=ALIYUN, api_key="k", model="m",
        config={
            "text_model": {"enable_thinking": False},
            "rag": {"enabled": rag_enabled},
            "reply_fallback": {"no_knowledge_chitchat_enabled": chitchat},
        },
    )


MODES = [
    ("RAG开+闲聊开", True, True),
    ("RAG开+闲聊关", True, False),
    ("RAG关+闲聊开", False, True),
    ("RAG关+闲聊关", False, False),
]


# ---------- 1. 无资料出路：必须存在，且四个分支都要有 ----------

@pytest.mark.parametrize("name,rag,chitchat", MODES)
def test_system_prompt_has_clarify_outlet(name, rag, chitchat):
    from src.ai.text_model_client import TextModelClient
    sp = _client(rag, chitchat)._system_prompt()
    assert TextModelClient.NO_BASIS_OUTLET in sp, f"{name}: system prompt 缺少无资料出路"


@pytest.mark.parametrize("name,rag,chitchat", MODES)
def test_direct_question_rules_have_clarify_outlet(name, rag, chitchat):
    from src.ai.text_model_client import TextModelClient
    rules = _client(rag, chitchat)._direct_question_rules()
    assert TextModelClient.NO_BASIS_OUTLET in rules, f"{name}: 短句规则缺少无资料出路"


@pytest.mark.parametrize("name,rag,chitchat", MODES)
def test_old_silence_rules_removed(name, rag, chitchat):
    from src.ai.text_model_client import TextModelClient
    c = _client(rag, chitchat)
    text = c._system_prompt() + c._direct_question_rules()
    for old in OLD_SILENCE_RULES:
        assert old not in text, f"{name}: 旧「无资料即沉默」规则残留: {old}"


# ---------- 2. 出路文案的语义要求 ----------

def test_outlet_semantics():
    from src.ai.text_model_client import TextModelClient
    t = TextModelClient.NO_BASIS_OUTLET
    assert "澄清" in t and "追问" in t, "出路必须是澄清/追问"
    assert "一个" in t, "必须限制一次最多追问一个问题，避免连环追问"
    assert "沉默" in t or "不能" in t, "必须明确否定沉默"


# ---------- 3. 安全边界不得被削弱 ----------

def test_fabrication_ban_intact():
    """放开「必须沉默」后，绝不放开「禁止编造」。"""
    # 严格分支（RAG 开 + 闲聊关）：无资料时禁止编造具体事实
    strict = _client(rag_enabled=True, chitchat=False)._system_prompt()
    assert "严禁编造价格" in strict, "价格编造红线丢失"
    assert "绝对不能自行给出" in strict, "价格不得自行给出的约束丢失"
    assert "绝不能编造具体事实" in strict, "无资料时禁止编造具体事实的约束丢失"

    # 自由聊分支（RAG 关）：措辞不同，但同样必须禁止编造
    free = _client(rag_enabled=False, chitchat=False)._system_prompt()
    assert "不能编造具体信息" in free, "自由聊分支禁止编造约束丢失"

    for name, sp in (("严格分支", strict), ("自由聊分支", free)):
        assert "价格、金额、地址、承诺" in sp, f"{name}: 禁编造清单未覆盖价格/地址/承诺"


def test_hard_gate_no_reply_rules_intact():
    """这几类 no_reply 是刻意保留的硬门禁，不能被本次改动冲掉。"""
    sp = _client(rag_enabled=True)._system_prompt()
    assert "如果 latest_message.sender 不是 customer，必须 no_reply" in sp, "非客户发送门禁丢失"
    assert "is_ad_or_promotion / is_payment_notice 为 true，必须 no_reply" in sp, "广告/支付门禁丢失"
    assert "一律 no_reply" in sp, "跳过名单门禁丢失"
    assert "攻击辱骂且没有真实问题时，不要强行展开" in sp, "无意义消息不强答的约束丢失"


# ---------- 4. 生成侧再强化一遍（模型对末尾约束最敏感） ----------

def test_user_prompt_reinforces_outlet():
    c = _client(rag_enabled=True)
    up = c._user_prompt(
        {
            "customer_turn_text": "你们这个东西多少钱？",
            "latest_message": {"sender": "customer", "content": "你们这个东西多少钱？"},
        },
        {},
    )
    assert "没有资料依据时不得编造具体事实" in up, "user prompt 未重申无资料处理方式"
    assert "必须用一句自然的澄清追问承接客户" in up, "user prompt 未重申必须澄清追问"
    assert "no_reply 仅限" in up, "user prompt 未限定 no_reply 的适用范围"
