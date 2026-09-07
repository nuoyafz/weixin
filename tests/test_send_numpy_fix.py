#!/usr/bin/env python3
"""验证 wechat_sender.send_report 在截图返回 numpy 数组时不再触发
'The truth value of an array with more than one element is ambiguous'。

之前 `_capture_frame` 返回 ndarray，而 `_send_report_impl` 里
`... if before else {"ok": True}` 对 ndarray 取布尔值会抛上述异常，
导致每条回复都发送失败。现改为 `if before is not None`。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import MagicMock


def main():
    from src.rpa.wechat_sender import WeChatSender

    dummy_img = np.zeros((20, 20, 3), dtype=np.uint8)

    capture = MagicMock()
    cap_result = MagicMock()
    cap_result.success = True
    cap_result.image = dummy_img
    capture.capture_window.return_value = cap_result

    finder = MagicMock()
    found = MagicMock()
    found.to_dict.return_value = {"handle": 123, "rect": {}}
    finder.find.return_value = found

    send_confirm = MagicMock()
    send_confirm.inspect_chat_ready.return_value = {"ok": True}
    send_confirm.inspect_current_customer_turn.return_value = {"ok": True}
    send_confirm.confirm_send.return_value = {"success": True}

    send_guard = MagicMock()
    send_guard.check.return_value = (True, "")
    send_guard.mark_sent.return_value = None

    rpa = MagicMock()
    human_like_mouse = MagicMock()
    human_activity_monitor = MagicMock()
    human_activity_monitor.ignore_for.return_value = None

    sender = WeChatSender(
        config={},
        finder=finder,
        capture=capture,
        screen_capture=capture,
        store=MagicMock(),
        rpa=rpa,
        send_confirm=send_confirm,
        send_guard=send_guard,
        human_like_mouse=human_like_mouse,
        human_activity_monitor=human_activity_monitor,
        window_manager=MagicMock(),
        privacy_overlay=MagicMock(),
    )

    # 把依赖的外部方法打桩，避免真实 UI 交互
    sender._verify_current_chat_contact = lambda *a, **k: True
    sender._clear_input_before_send = lambda *a, **k: True
    sender._reply_segments = lambda text: [text]
    sender._join_short_reply_lines = lambda segs: segs
    sender._managed_display_mode = lambda: False
    sender._quote_requested = lambda *a, **k: False
    sender._segment_delay = lambda: 0
    sender._send_confirm_delay_seconds = lambda: 0
    sender._send_method_name = lambda: "test"
    sender._background_send_enabled = lambda: False
    sender._wait_for_mouse_release_before_window_action = lambda: None
    sender._prepare_privacy_overlay_interaction = lambda *a, **k: None
    sender._merge_reply_segments_for_managed_display = lambda segs: "".join(segs)

    report = {
        "reply_draft": "哈哈，这是测试回复",
        "current_contact": "亚磊",
        "contact_key": "亚磊",
        "chat_type": "private",
        "analysis": {"latest_message": {"text": "x"}},
    }

    result = sender.send_report(report)
    print("send_report result:", result.get("ok"), result.get("reason"))

    assert result.get("ok") is True, f"发送应成功，但得到: {result}"
    print("PASS: numpy 数组布尔值崩溃已修复，send_report 正常返回 ok=True")


if __name__ == "__main__":
    main()
