# -*- coding: utf-8 -*-
"""思考模式开关（enable_thinking）离线测试。

背景：UI(config.yaml:39/65) 与前端开关早有 enable_thinking，但请求体从不携带，
导致开关形同虚设——用户以为关了思考，实际 qwen3.8 默认一直思考，
单次回复 12~21s。本测试锁定「开关必须真正进入请求体」。

全部为纯逻辑测试，不联网、不依赖真实密钥，可在 CI 稳定运行。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ALIYUN = "https://llm-xxxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
OPENAI = "https://api.openai.com/v1"


# ---------- 1. 开关解析逻辑 ----------

@pytest.mark.parametrize("name,base_url,cfg,expect", [
    ("配置 false + 阿里云", ALIYUN, {"text_model": {"enable_thinking": False}}, False),
    ("配置 true + 阿里云", ALIYUN, {"text_model": {"enable_thinking": True}}, True),
    ("未配置 + 阿里云 -> 默认关", ALIYUN, {"text_model": {}}, False),
    ("未配置 + 非阿里云 -> 不干预", OPENAI, {"text_model": {}}, None),
    ("配置 false + 非阿里云 -> 显式优先", OPENAI, {"text_model": {"enable_thinking": False}}, False),
])
def test_resolve_enable_thinking(name, base_url, cfg, expect):
    from src.ai.text_model_client import TextModelClient
    c = TextModelClient(base_url=base_url, api_key="k", model="m", config=cfg)
    assert c.enable_thinking is expect, f"{name}: {c.enable_thinking!r} != {expect!r}"


# ---------- 2. 参数真的进入请求体 ----------

class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _capture_payload(client, call):
    """拦截 Session.post，返回实际发出的 payload。"""
    captured = {}

    def _spy(url, **kw):
        captured.update(kw.get("json") or {})
        return _FakeResp()

    orig = client._session.post
    client._session.post = _spy
    try:
        call()
    finally:
        client._session.post = orig
    return captured


def test_payload_carries_enable_thinking_false():
    """非流式：关思考时请求体必须带 enable_thinking=False。"""
    from src.ai.text_model_client import TextModelClient
    c = TextModelClient(base_url=ALIYUN, api_key="k", model="m",
                        config={"text_model": {"enable_thinking": False}})
    p = _capture_payload(c, lambda: c._call_api([{"role": "user", "content": "hi"}]))
    assert p.get("enable_thinking") is False, f"payload 未携带开关: {sorted(p.keys())}"


def test_payload_carries_enable_thinking_true():
    """开启思考时，请求体必须带 True（用户显式选择要生效）。"""
    from src.ai.text_model_client import TextModelClient
    c = TextModelClient(base_url=ALIYUN, api_key="k", model="m",
                        config={"text_model": {"enable_thinking": True}})
    p = _capture_payload(c, lambda: c._call_api([{"role": "user", "content": "hi"}]))
    assert p.get("enable_thinking") is True


def test_payload_omits_param_on_non_aliyun():
    """非阿里云端点不下发该参数，避免给不支持的网关报错。"""
    from src.ai.text_model_client import TextModelClient
    c = TextModelClient(base_url=OPENAI, api_key="k", model="gpt-4o",
                        config={"text_model": {}})
    p = _capture_payload(c, lambda: c._call_api([{"role": "user", "content": "hi"}]))
    assert "enable_thinking" not in p, "非阿里云端点不应下发 enable_thinking"


# ---------- 3. 视觉客户端同样生效 ----------

def test_vision_client_thinking_default_off_on_aliyun():
    pytest.importorskip("src.ai.vision_model_client")
    from src.ai.vision_model_client import VisionModelClient
    v = VisionModelClient(base_url=ALIYUN, api_key="k", model="qwen-vl-plus")
    assert v.enable_thinking is False

    v2 = VisionModelClient(base_url=OPENAI, api_key="k", model="gpt-4o")
    assert v2.enable_thinking is None

    v3 = VisionModelClient(base_url=ALIYUN, api_key="k", model="qvq",
                           enable_thinking=True)
    assert v3.enable_thinking is True


# ---------- 4. 连接测试路径 ----------

def test_ui_test_connection_body_disables_thinking():
    """UI「测试连接」max_tokens=5，若开思考会全部被思考吃掉 -> 正文为空误判失败。"""
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src", "ui", "webview_window.py"), encoding="utf-8").read()
    idx = src.find("def _build_body")
    assert idx > 0, "未找到 _build_body"
    seg = src[idx:idx + 900]
    assert "enable_thinking" in seg, "测试连接路径未关闭思考，max_tokens=5 会被思考吃光"
