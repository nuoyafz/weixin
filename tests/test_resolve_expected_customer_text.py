"""离线验证：发送前守卫 expected_customer_text 解析（regression: 文字消息永拦截）。

根因：wechat_sender 只在 expected_customer_text / customer_turn_text 两个字段取
"期望客户话术"，而文字消息下这俩字段全库从未被赋值；真正的客户消息在
analysis['latest_message']['content']（OCR 日志 msg= 即来源于此）。回退链漏掉它，
导致守卫永远拿到空 expected -> expected_customer_turn_empty -> 每条都发送失败。

修复：新增 _resolve_expected_customer_text，按 显式字段 -> customer_turn_text ->
latest_message.content -> messages 最后一条客户消息 顺序回退。

本测试覆盖解析器本身，并做一段"解析器 -> 守卫"的集成验证，证明文字消息不再被拦截。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.rpa.wechat_sender import WeChatSender
from src.rpa.send_confirm import SendConfirm

CUSTOMER_MSG = "再给他发问他咋回事"


def make_sender():
    return WeChatSender()


def make_sc():
    sc = SendConfirm(config={}, screen_capture=None, logger=None)
    sc._ocr_parser = object()  # 强制走 OCR 校验分支
    return sc


def ocr_returns_customer(image):
    # 模拟发送前重新 OCR 当前会话：客户消息未变
    return [{"side": "left", "text": CUSTOMER_MSG, "content": CUSTOMER_MSG,
             "confidence": 0.99}]


def main():
    s = make_sender()

    # ---- 解析器单测 ----
    # CASE1: 日志里的真实结构——只有 latest_message.content，无 customer_turn_text
    report1 = {"analysis": {
        "latest_message": {"content": CUSTOMER_MSG, "side": "left"},
        "messages": [{"side": "left", "text": CUSTOMER_MSG}],
    }}
    assert s._resolve_expected_customer_text(report1, report1["analysis"]) == CUSTOMER_MSG

    # CASE2: 旧字段 customer_turn_text 仍有值时优先用它
    report2 = {"analysis": {"customer_turn_text": "你们这服务多少钱",
                            "latest_message": {"content": "x"}}}
    assert s._resolve_expected_customer_text(report2, report2["analysis"]) == "你们这服务多少钱"

    # CASE3: 全空 -> 返回空串（守卫仍会拦截，行为不变）
    report3 = {"analysis": {}}
    assert s._resolve_expected_customer_text(report3, report3["analysis"]) == ""

    # CASE4: 多消息列表，取最后一条客户消息
    report4 = {"analysis": {"messages": [
        {"side": "left", "text": "在吗"},
        {"side": "right", "text": "在的"},
        {"side": "left", "text": "聊天记录"},
    ]}}
    assert s._resolve_expected_customer_text(report4, report4["analysis"]) == "聊天记录"

    print("[解析器] CASE1-4 PASS")

    # ---- 集成：解析器 -> 守卫，验证文字消息不再被拦截 ----
    resolved = s._resolve_expected_customer_text(report1, report1["analysis"])
    assert resolved == CUSTOMER_MSG, "解析器未取到客户消息"

    sc = make_sc()
    sc._ocr_chat_messages = ocr_returns_customer
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    r = sc.inspect_current_customer_turn(dummy, expected_text=resolved)
    assert r["ok"] is True, r
    assert r["reason"] in ("customer_turn_matched",
                           "customer_turn_latest_message_matched",
                           "customer_turn_contains_expected",
                           "customer_turn_fuzzy_matched"), r
    print(f"[集成] resolved={resolved!r} -> 守卫 ok={r['ok']} reason={r['reason']}  (不再拦截)")

    print("ALL PASS")


if __name__ == "__main__":
    main()
