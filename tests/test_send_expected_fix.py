"""离线验证：发送前守卫 inspect_current_customer_turn 的 expected 为空回归。

回归点：expected_customer_text 全代码库从未被赋值 -> 守卫最早返回
expected_customer_turn_empty -> wechat_sender 包装成
customer_turn_text_mismatch_before_send，导致所有发送被拦截。

修复：wechat_sender 在 expected 为空时回退到 analysis['customer_turn_text']。
本测试直接对守卫函数做单测，覆盖：
  CASE1 (旧 bug 路径) expected=''  -> reason=expected_customer_turn_empty, ok=False
  CASE2 (修复路径)     expected=客户消息 -> 命中 current OCR -> ok=True, customer_turn_matched
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.rpa.send_confirm import SendConfirm

CUSTOMER_MSG = "都不管二高校友了"


def make_sc():
    sc = SendConfirm(config={}, screen_capture=None, logger=None)
    # 注入非空 parser，强制走 OCR 校验分支（否则走 available_skipped 直接放行）
    sc._ocr_parser = object()
    return sc


def ocr_returns_customer(image):
    # 模拟 OCR：底部一条客户气泡（side=left，带高置信，避免被低置信碎片过滤）
    return [{"side": "left", "text": CUSTOMER_MSG, "content": CUSTOMER_MSG,
             "confidence": 0.99}]


def main():
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)

    # CASE1: expected 为空 -> 旧 bug 必须命中 expected_customer_turn_empty
    sc1 = make_sc()
    sc1._ocr_chat_messages = ocr_returns_customer
    r1 = sc1.inspect_current_customer_turn(dummy, expected_text="")
    assert r1["ok"] is False, r1
    assert r1["reason"] == "expected_customer_turn_empty", r1
    print(f"[CASE1] expected='' -> ok={r1['ok']} reason={r1['reason']}  (复现旧 bug)")

    # CASE2: expected=客户消息 -> 应匹配当前 OCR 文本 -> ok=True, customer_turn_matched
    sc2 = make_sc()
    sc2._ocr_chat_messages = ocr_returns_customer
    r2 = sc2.inspect_current_customer_turn(dummy, expected_text=CUSTOMER_MSG)
    assert r2["ok"] is True, r2
    assert r2["reason"] == "customer_turn_matched", r2
    print(f"[CASE2] expected=客户消息 -> ok={r2['ok']} reason={r2['reason']}  (修复生效)")

    print("ALL PASS")


if __name__ == "__main__":
    main()
