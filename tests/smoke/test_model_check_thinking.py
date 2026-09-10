"""验证 ModelChecker 在速度/连通性检查中正确关掉思考模式（修复 speed_check 20s）。

- aliyuncs/dashscope 端点：enable_thinking=None 时自动强制 False
- 显式 enable_thinking=True/False 时按值下发
- 非阿里端点 + None 时不下发该字段（尊重远端默认）
"""
import json
import urllib.request
import urllib.error

import pytest

from src.reply.model_check import ModelChecker


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_CAPTURE = {}


def _fake_urlopen(req, timeout=None):
    body = json.loads(req.data.decode("utf-8"))
    _CAPTURE["body"] = body
    return _FakeResp({"choices": [{"message": {"content": "pong"}}]})


def test_aliyuncs_default_forces_thinking_off(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://maas.aliyuncs.com/compatible-mode/v1", "k", "qwen-turbo")
    assert c.enable_thinking is None
    c._chat("ping", 5)
    assert _CAPTURE["body"].get("enable_thinking") is False


def test_dashscope_default_forces_thinking_off(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://dashscope.aliyuncs.com/compatible", "k", "qwen-flash")
    c._chat("ping", 5)
    assert _CAPTURE["body"].get("enable_thinking") is False


def test_explicit_true_is_respected(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://maas.aliyuncs.com/v1", "k", "qwen-plus",
                     enable_thinking=True)
    c._chat("ping", 5)
    assert _CAPTURE["body"].get("enable_thinking") is True


def test_explicit_false_is_respected(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://maas.aliyuncs.com/v1", "k", "qwen-turbo",
                     enable_thinking=False)
    c._chat("ping", 5)
    assert _CAPTURE["body"].get("enable_thinking") is False


def test_non_aliyun_no_field_when_none(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://api.openai.com/v1", "k", "gpt-4o-mini")
    c._chat("ping", 5)
    assert "enable_thinking" not in _CAPTURE["body"]


def test_speed_check_uses_thinking_off_on_aliyuncs(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    c = ModelChecker("https://maas.aliyuncs.com/v1", "k", "qwen-turbo")
    res = c.speed_check(rounds=2)
    assert res["ok"] is True
    assert res["success_rounds"] == 2
    assert _CAPTURE["body"].get("enable_thinking") is False
