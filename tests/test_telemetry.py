"""匿名使用统计模块单测。pytest 运行：python -m pytest tests/test_telemetry.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.telemetry.usage as u


def test_device_id_stable_and_len():
    a = u.get_device_id()
    b = u.get_device_id()
    assert a == b and len(a) == 32


def test_report_fire_and_forget(monkeypatch):
    captured = {}

    def fake_post(p):
        captured.update(p)

    monkeypatch.setattr(u, "_post", fake_post)
    u.report("startup")
    assert captured.get("event") == "startup"
    assert {"device_id", "version", "ts"} <= set(captured)


def test_cycle_throttle(monkeypatch):
    calls = []
    monkeypatch.setattr(u, "_post", lambda p: calls.append(p))
    # 同一秒内连续调用应被限频，只上报第一次
    u.report_cycle(True)
    u.report_cycle(True)
    u.report_cycle(False)
    assert len(calls) == 1
    assert calls[0]["event"] == "reply" and calls[0]["sent_ok"] is True


def test_collect_url_derivation(monkeypatch):
    import types

    fake = types.ModuleType("src.updater")
    fake._resolve_server = staticmethod(lambda: "https://a.fangzhoui.cn/visreply/update")
    monkeypatch.setitem(sys.modules, "src.updater", fake)
    assert u._resolve_collect_url() == "https://a.fangzhoui.cn/visreply/telemetry/collect"
