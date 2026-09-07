"""发送前客户话轮校验的双重验证测试。

覆盖：
1. 精确匹配 -> ok（matched）
2. 单字/标点 OCR 噪声 -> 模糊相似度放行（fuzzy_matched），解决"方政"类正常回复被误拦
3. 文本完全不同 -> 拦截（mismatch），保留防抢话能力
4. 同帧 baseline 保护 -> 退化为纯 OCR，不失效
5. 视觉双信号 -> 视觉无变化则即使文本不匹配也放行（非噪声导致）
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from src.rpa.send_confirm import SendConfirm


def mk_messages(lines):
    """lines: list of (side, text) -> 消息结构。"""

    def _make(side, text):
        return {"side": side, "text": text, "content": text,
                "y_center": 100.0, "confidence": 0.95}
    return [_make(s, t) for s, t in lines]


def make_sc(messages):
    sc = SendConfirm()
    # 绕过真实 OCR，直接注入消息
    sc._ocr_chat_messages = lambda img: list(messages)
    sc._ocr_parser = object()  # 标记为已注入，避免走 OCR 不可用分支
    return sc


def run(name, expected, customer_msgs, baseline=None, expect_ok=True,
        expect_reason_substr=None):
    msgs = mk_messages(customer_msgs)
    sc = make_sc(msgs)
    # image 用占位 numpy；baseline 若提供须是 ndarray
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = sc.inspect_current_customer_turn(img, expected,
                                           baseline_image=baseline)
    ok = bool(res.get("ok"))
    reason = str(res.get("reason", ""))
    status = "PASS" if ok == expect_ok else "FAIL"
    sub_ok = True
    if expect_reason_substr and expect_reason_substr not in reason:
        sub_ok = False
        status = "FAIL"
    print(f"[{status}] {name}: ok={ok} reason={reason!r}")
    assert ok == expect_ok, f"{name}: ok={ok} expect={expect_ok}"
    if expect_reason_substr:
        assert expect_reason_substr in reason, f"{name}: reason={reason}"
    return res


def main():
    # 1. 精确匹配
    run("精确匹配", "客户确认收到", [("left", "客户确认收到")], expect_ok=True,
        expect_reason_substr="matched")

    # 2. 单字差异（OCR 噪声）：保留字不同 -> 模糊放行
    run("OCR单字噪声", "今天天气真好呀", [("left", "今天天气真好啊")],
        expect_ok=True, expect_reason_substr="fuzzy_matched")

    # 3. 标点差异（normalize 去标点后精确匹配，验证归一化生效）
    run("OCR标点噪声", "那我们约几点", [("left", "那我们约几点。")],
        expect_ok=True, expect_reason_substr="matched")

    # 4. 文本完全不同 -> 拦截（防抢话）
    run("完全不同拦截", "客户确认收到", [("left", "你快递到了吗")], expect_ok=False,
        expect_reason_substr="mismatch")

    # 5. 同帧 baseline 保护：baseline 与 image 同一对象 -> 退回纯 OCR
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    sc = make_sc(mk_messages([("left", "你快递到了吗")]))
    res = sc.inspect_current_customer_turn(img, "客户确认收到", baseline_image=img)
    assert res["ok"] is False, "同帧baseline应退回纯OCR仍拦截"
    assert "mismatch" in res["reason"], res["reason"]
    print(f"[PASS] 同帧baseline保护: ok={res['ok']} reason={res['reason']!r}")

    # 6. 视觉双信号：提供不同对象但内容相同的 baseline -> 视觉无变化 -> 放行
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)  # 与 img 内容相同，但不同对象
    sc = make_sc(mk_messages([("left", "你快递到了吗")]))
    res = sc.inspect_current_customer_turn(img2, "客户确认收到",
                                          baseline_image=img2.copy())
    # 视觉差分=0（unchanged）-> 即便文本不匹配也放行
    assert res["ok"] is True, f"视觉无变化应放行: {res['reason']}"
    assert "visual_unchanged_passed" in res["reason"], res["reason"]
    print(f"[PASS] 视觉双信号放行: ok={res['ok']} reason={res['reason']!r}")

    print("\nALL_SEND_CONFIRM_TURN_TESTS_PASSED")


if __name__ == "__main__":
    main()
