"""全链路模拟测试：观察产出 -> 回复 -> 发送。

不依赖真实微信 / OCR / LLM，用 mock 替换外部依赖，但保留真实的
WeChatRPA 调用链路（activate_window / 剪贴板粘贴 / 回车发送），
验证「观察抓到的客户话术 -> 发送前守卫 -> 真实发送」整条链路在
_rpa 修复后能跑通。

对应真实日志场景：
  [11:06:23] [ocr] contact='方政' msg='再给他发问他咋回事'
  [11:07:11] [reply] LLM生成回复 contact='方政' reply='行，我再发一下问下情况...'
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import mock
from src.rpa.wechat_sender import WeChatSender


class FakeRpa:
    """模拟真实 WeChatRPA，记录调用但不真动鼠标。"""

    def __init__(self):
        self.calls = []

    def activate_window(self, hwnd):
        self.calls.append(("activate_window", hwnd))

    def _send_text_via_clipboard(self, hwnd, text):
        self.calls.append(("paste", hwnd, text))
        return True

    def send_key_press(self, vk, hwnd=None):
        self.calls.append(("key", vk))

    def method_names(self):
        return {n for n, *_ in self.calls}


class FakeFinder:
    def find(self):
        class Win:
            def to_dict(self):
                return {"handle": 123, "rect": {"x": 0, "y": 0, "w": 100, "h": 100}}
        return Win()


def build_sender():
    s = WeChatSender()
    # 注入模拟 RPA（这是被修复的核心）——记录真实调用
    s._rpa = FakeRpa()
    s.finder = FakeFinder()
    s._send_guard = mock.MagicMock()
    s._send_guard.check.return_value = (True, "")
    s._send_guard.mark_sent.return_value = None
    s._send_confirm = mock.MagicMock()
    s._send_confirm.inspect_chat_ready.return_value = {"ok": True}
    s._send_confirm.inspect_current_customer_turn.return_value = {"ok": True}
    s._send_confirm.confirm_send.return_value = {"success": True, "reason": "ok"}
    s._compliance_gate = False  # 跳过合规门，专注发送链路
    s._human_activity_monitor = None
    s._human_like_mouse = None
    s._screen_capture = None
    s.store = mock.MagicMock()
    # patch 掉所有依赖图像/窗口外部状态的私有方法（用工厂固定返回值，
    # 避免闭包共享循环变量导致所有 lambda 都返回最后一项）。
    def _const(val):
        def _f(*a, **k):
            return val
        return _f

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
    ]:
        setattr(s, name, _const(ret))
    s._reply_segments = lambda text: [text]
    s._join_short_reply_lines = lambda segs: segs
    s._capture_frame = lambda hwnd: mock.MagicMock(name="frame")
    return s


def make_report(contact, msg, reply):
    """构造与 ObserveService 产出结构一致的 report（来自真实日志）。"""
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


def test_full_pipeline_send_ok():
    s = build_sender()
    report = make_report(
        "方政", "再给他发问他咋回事", "行，我再发一下问下情况，有回复告诉你。")
    res = s.send_report(report)
    assert res["ok"] is True, f"发送应成功，实际: {res}"
    assert res["reason"] == "", f"reason 应为空，实际: {res['reason']}"
    # 真实 RPA 必须被调用（激活窗口 + 粘贴 + 回车）
    names = s._rpa.method_names()
    assert "activate_window" in names, "窗口未激活"
    assert "paste" in names, "文本未粘贴"
    assert ("key", 0x0D) in s._rpa.calls, "未按下回车发送"
    print("[OK] 全链路（观察->回复->发送）跑通，方政消息已成功发出")
    print("     RPA 调用:", [c[0] for c in s._rpa.calls])


def test_expected_customer_text_resolved():
    """发送前守卫用的 expected 必须来自 latest_message.content。"""
    s = build_sender()
    report = make_report(
        "崔嘉兵", "聊天记录", "收到，你发这条是想确认哪段聊天记录？")
    # 不发真实消息，只验证解析出的期望话术
    etc = s._resolve_expected_customer_text(
        report, report["analysis"])
    assert etc == "聊天记录", f"期望话术解析错误: {etc!r}"
    print("[OK] 期望客户话术正确解析 =", etc)


def test_old_bug_rpa_none_reproduced():
    """回归保护：若 _rpa 缺失（旧 bug），必须透传真实原因而非裸 send_failed。"""
    s = build_sender()
    s._rpa = None  # 旧 bug 复现
    report = make_report("方政", "再给他发问他咋回事", "测试回复")
    res = s.send_report(report)
    assert res["ok"] is False, "旧 bug 下应失败"
    assert "activate_window" in res["reason"] or "NoneType" in res["reason"], \
        f"应透传真实原因，实际: {res['reason']!r}"
    print("[OK] 旧 bug 复现：原因透传 =", res["reason"])


if __name__ == "__main__":
    test_expected_customer_text_resolved()
    test_full_pipeline_send_ok()
    test_old_bug_rpa_none_reproduced()
    print("\n=== 全链路模拟测试全部通过 ===")
