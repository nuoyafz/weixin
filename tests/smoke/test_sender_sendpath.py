"""发送路径修复的行为锁定测试（离线、无真机）。

锁住两处真实修复，防止再次退化：

1. 主链路 send_report 必须清洗 reply 里的 {image:name} 标记
   ——修复前该标记会作为脏文本原样发给客户。
2. 素材发送原语 _send_material_once 的分支行为
   ——含拟人鼠标优先、RPA 降级、以及两者皆缺时不崩溃（加固项）。

参考真实缺陷：解析 {image:} 的正确实现原本只存在于无人调用的
send_text 里，而每天都在跑的主链路完全没有调用它。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402

# wechat_sender 顶层依赖 numpy；CI 未安装时整组 skip，
# 避免依赖缺失把 smoke 误报成红（与真实测试失败区分开）。
pytest.importorskip("numpy")

from unittest import mock  # noqa: E402

from src.ai.text_model_client import TextModelClient  # noqa: E402
from src.rpa.wechat_sender import WeChatSender  # noqa: E402


def _const(val):
    def _f(*a, **k):
        return val
    return _f


def build_impl_sender(material_dir=None):
    """构造可跑 send_report 主链路的 sender，外部依赖全部 stub。"""
    s = WeChatSender()
    s._config = {"material_dir": material_dir or ""}
    s._rpa = mock.MagicMock(name="rpa")
    s._send_guard = mock.MagicMock()
    s._send_guard.check.return_value = (True, "")  # 守卫放行，专注发送链路
    s._send_confirm = mock.MagicMock()
    s._send_confirm.inspect_chat_ready.return_value = {"ok": True}
    s._send_confirm.inspect_current_customer_turn.return_value = {"ok": True}
    s._send_confirm.confirm_send.return_value = {"success": True, "reason": "ok"}
    s._compliance_gate = False
    s._human_activity_monitor = None
    s._human_like_mouse = None
    s._send_delay = 0
    s.store = mock.MagicMock()

    class FakeFinder:
        def find(self):
            class Win:
                def to_dict(self):
                    return {"handle": 123,
                            "rect": {"x": 0, "y": 0, "w": 100, "h": 100}}
            return Win()
    s.finder = FakeFinder()

    for name, ret in [
        ("_wait_for_mouse_release_before_window_action", True),
        ("_background_send_enabled", (False, None)),
        ("_prepare_privacy_overlay_interaction", None),
        ("_verify_current_chat_contact", True),
        ("_managed_display_mode", False),
        ("_quote_requested", False),
        ("_clear_input_before_send", True),
        ("_segment_delay", 0.0),
        ("_send_confirm_delay_seconds", 0.0),
        ("_capture_frame", mock.MagicMock(name="frame")),
    ]:
        setattr(s, name, _const(ret))
    s._reply_segments = lambda text: [text]
    s._join_short_reply_lines = lambda segs: segs
    return s


def make_report(contact, msg, reply):
    return {
        "analysis": {
            "contact": contact,
            "current_contact": contact,
            "chat_type": "private",
            "latest_message": {"content": msg, "from_self": False},
            "messages": [{"content": msg, "from_self": False}],
        },
        "reply_draft": reply,
        "current_contact": contact,
        "chat_type": "private",
    }


# ---------------------------------------------------------------
# 修复一：主链路必须清洗 {image:} 脏标记
# ---------------------------------------------------------------
def test_reply_mark_not_leaked_to_customer():
    """标记无论素材是否存在都不能出现在发给客户的文本里。"""
    s = build_impl_sender()
    res = s.send_report(make_report("客户", "你好", "请您看图 {image:缺失图} 谢谢"))

    assert res["ok"] is True, f"发送应成功: {res}"
    pasted = [c.args[1] for c in s._rpa._send_text_via_clipboard.call_args_list]
    assert pasted, "未发生任何粘贴"
    for text in pasted:
        assert "{image:" not in text, f"脏标记泄漏给客户: {text!r}"
    assert "请您看图" in pasted[0], f"期望保留正常文本，实际: {pasted[0]!r}"


def test_image_ref_is_resolved_into_materials(tmp_path):
    """存在的 {image:name} 应转成素材发送，而不是被静默丢弃。"""
    demo = tmp_path / "demo.png"
    demo.write_bytes(b"fake-png")

    s = build_impl_sender(material_dir=str(tmp_path))
    fake_mouse = mock.MagicMock(name="human_like_mouse")
    s._human_like_mouse = fake_mouse

    s.send_report(make_report("客户", "你好", "看图 {image:demo}"))

    pasted = [c.args[1] for c in s._rpa._send_text_via_clipboard.call_args_list]
    for text in pasted:
        assert "{image:" not in text, f"脏标记泄漏: {text!r}"
    assert fake_mouse.paste_image_and_enter.called, \
        "解析出的素材应通过拟人鼠标发送，实际未发送"


# ---------------------------------------------------------------
# 修复二：素材发送原语分支行为
# ---------------------------------------------------------------
def test_material_rejects_unsupported_kind(tmp_path):
    s = WeChatSender()
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    assert s._send_material_once(1, str(p), kind="unknown") is False


def test_material_missing_file_returns_false():
    s = WeChatSender()
    assert s._send_material_once(1, "/nonexistent/x.png", "image") is False


def test_material_prefers_human_like_mouse(tmp_path):
    s = WeChatSender()
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    fake = mock.MagicMock()
    s._human_like_mouse = fake

    assert s._send_material_once(1, str(p), "image") is True
    fake.paste_image_and_enter.assert_called_once_with(1, str(p))


def test_material_file_kind_uses_file_paste(tmp_path):
    """file/video 必须走 paste_file_and_enter，不能误用图片通道。"""
    s = WeChatSender()
    p = tmp_path / "a.pdf"
    p.write_bytes(b"x")
    fake = mock.MagicMock()
    s._human_like_mouse = fake

    assert s._send_material_once(1, str(p), "file") is True
    fake.paste_file_and_enter.assert_called_once_with(1, str(p))
    fake.paste_image_and_enter.assert_not_called()


def test_material_falls_back_to_clipboard(tmp_path):
    """拟人鼠标不可用时应降级为剪贴板 + 回车。"""
    s = WeChatSender()
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    s._human_like_mouse = None
    s._rpa = mock.MagicMock()

    assert s._send_material_once(1, str(p), "image") is True
    s._rpa._send_text_via_clipboard.assert_called_once_with(1, str(p))
    s._rpa.send_key_press.assert_called_once_with(0x0D)


def test_material_without_any_engine_returns_false(tmp_path):
    """两者皆缺时必须优雅返回 False（原实现会抛 AttributeError）。"""
    s = WeChatSender()
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    s._human_like_mouse = None
    s._rpa = None

    assert s._send_material_once(1, str(p), "image") is False


# ---------------------------------------------------------------
# 素材链路接通：UI 上传 -> LLM 引用 -> sender 发图
# ---------------------------------------------------------------
def test_prompt_injects_material_when_present(tmp_path):
    """有素材时，system prompt 必须告知素材名与 {image:} 用法。"""
    (tmp_path / "报价单.png").write_bytes(b"x")
    client = TextModelClient(config={"material_dir": str(tmp_path)})

    hint = client._material_hint()
    assert "报价单" in hint, f"未告知可用素材名: {hint!r}"
    assert "{image:报价单}" in hint, f"未示范素材标记用法: {hint!r}"
    assert client._material_stems() == ("报价单",), \
        f"素材名应去扩展名，实际: {client._material_stems()}"


def test_prompt_stays_silent_without_material(tmp_path):
    """无素材时零注入——保证未使用素材功能的用户行为完全不变。"""
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    client = TextModelClient(config={"material_dir": str(empty)})

    assert client._material_stems() == (), "无素材应返回空元组"
    assert client._material_hint() == "", "无素材时不应注入任何提示"
    assert "图片素材发送能力" not in client._system_prompt(), \
        "无素材时 system prompt 不应出现素材段落"


def test_prompt_refreshes_when_material_added(tmp_path):
    """上传新素材后缓存必须失效，否则新素材永远不生效。"""
    client = TextModelClient(config={"material_dir": str(tmp_path)})
    assert "报价单" not in client._system_prompt(), "初始无素材不应出现该名"

    (tmp_path / "报价单.png").write_bytes(b"x")
    prompt = client._system_prompt()
    assert "报价单" in prompt, "新增素材后 prompt 未刷新（缓存未失效）"


def test_sender_falls_back_to_default_material_dir(tmp_path, monkeypatch):
    """config 未配 material_dir 时必须回退到 UI 上传目录。"""
    (tmp_path / "报价单.png").write_bytes(b"x")
    monkeypatch.setattr(
        WeChatSender, "_default_material_dir",
        classmethod(lambda cls: str(tmp_path)))

    s = WeChatSender()
    s._config = {}  # 模拟「从未配置 material_dir」
    assert s._material_dir() == str(tmp_path), "未回退到默认素材目录"

    resolved = s._image_ref("报价单")
    assert resolved and os.path.basename(resolved) == "报价单.png", \
        f"素材未解析成功，实际: {resolved!r}"


def test_end_to_end_material_reaches_customer(tmp_path, monkeypatch):
    """全链路：LLM 回复含 {image:报价单} -> 客户收到干净文本且图片真发出。

    刻意模拟真实场景：config 里从未配置 material_dir，
    必须靠「回退到 UI 上传目录」才能找到素材——这正是线上实际的形态。
    """
    (tmp_path / "报价单.png").write_bytes(b"x")
    monkeypatch.setattr(
        WeChatSender, "_default_material_dir",
        classmethod(lambda cls: str(tmp_path)))

    s = build_impl_sender()
    s._config = {}  # 模拟真实配置：没有 material_dir 这个键
    fake_mouse = mock.MagicMock(name="human_like_mouse")
    s._human_like_mouse = fake_mouse

    reply = "报价如下 {image:报价单} 请查收，有问题随时联系我。"
    res = s.send_report(make_report("客户A", "报价多少？", reply))

    pasted = [c.args[1] for c in s._rpa._send_text_via_clipboard.call_args_list]
    assert pasted, "未发生任何文本发送"
    for text in pasted:
        assert "{image:" not in text, f"脏标记泄漏给客户: {text!r}"
    assert fake_mouse.paste_image_and_enter.call_count == 1, \
        "报价单图片应被真正发送一次"
    sent_path = fake_mouse.paste_image_and_enter.call_args[0][1]
    assert os.path.basename(sent_path) == "报价单.png", f"发送了错误的素材: {sent_path}"
    assert res["action"] == "send_text_and_materials", \
        f"action 应为 send_text_and_materials，实际: {res['action']}"
