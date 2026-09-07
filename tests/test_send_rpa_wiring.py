"""离线验证：发送管线 RPA 层接线修复（regression: 所有发送被吞成裸 send_failed）。

根因：WeChatRPA 在死代码清理中被误归档，WeChatSender 构造时未传 rpa 且无默认兜底，
导致 self._rpa=None，_send_text_segment_with_confirm 第一步 activate_window 抛
AttributeError 被 except 吞成 ok=False -> ok_count=0 -> send_failed，消息从未发出。

修复：
1) 还原 src/rpa/wechat_rpa.py（WeChatRPA 提供 activate_window/_send_text_via_clipboard/send_key_press）。
2) WeChatSender.__init__ 给 _rpa 补默认兜底（对齐 _send_guard/_send_confirm）。
3) _send_report_impl 失败时透传真实原因（seg_fail/confirm.reason），不再裸 send_failed。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.rpa.wechat_sender import WeChatSender


def make_sender():
    s = WeChatSender()
    # 关闭会拦截/额外调用的组件，聚焦发送段
    s._send_confirm = None
    s._send_guard = None
    s._compliance_gate = False
    # 隔离需要真实窗口/键盘的环节
    s._prepare_privacy_overlay_interaction = lambda hwnd: True
    s._clear_input_before_send = lambda hwnd: True
    s._force_clear_input_before_send = lambda hwnd: True
    s._capture_frame = lambda hwnd: np.zeros((40, 40, 3), dtype=np.uint8)
    s._verify_current_chat_contact = lambda hwnd, name: True

    class F:
        def find(self):
            return {"handle": 1, "hwnd": 1,
                    "rect": {"left": 0, "top": 0, "right": 50, "bottom": 50}}
    s.finder = F()
    return s


def make_report():
    return {
        "reply_draft": "好的，我再发一下问下情况。",
        "current_contact": "方政",
        "analysis": {
            "current_contact": "方政",
            "chat_type": "private",
            "latest_message": {"content": "再给他发问他咋回事"},
        },
    }


def main():
    report = make_report()

    # CASE1: 旧 bug（_rpa=None）→ reason 透传真实 AttributeError，不再是裸 send_failed
    s1 = make_sender()
    s1._rpa = None
    r1 = s1.send_report(report)
    assert r1["ok"] is False, r1
    assert r1["reason"] != "send_failed", r1
    assert "activate_window" in r1["reason"] or "NoneType" in r1["reason"], r1
    print("[CASE1] 旧bug(_rpa=None) reason:", r1["reason"][:90])

    # CASE2: 自动 wiring（默认 WeChatRPA）→ 有 _rpa 且方法齐全
    s2 = make_sender()
    assert s2._rpa is not None, "默认应自动 new WeChatRPA"
    for m in ("activate_window", "_send_text_via_clipboard", "send_key_press"):
        assert hasattr(s2._rpa, m), f"缺方法 {m}"
    print("[CASE2] 默认 wiring OK, _rpa =", type(s2._rpa).__name__)

    # CASE3: 正常 rpa 桩 → 发送成功
    s3 = make_sender()

    class FakeRPA:
        def activate_window(self, hwnd):
            return True

        def _send_text_via_clipboard(self, hwnd, text):
            return True

        def send_key_press(self, vk):
            return True

    s3._rpa = FakeRPA()
    r3 = s3.send_report(report)
    assert r3["ok"] is True, r3
    print("[CASE3] 正常发送 -> ok=True")

    print("ALL PASS")


if __name__ == "__main__":
    main()
