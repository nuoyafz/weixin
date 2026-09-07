"""验证 SendConfirm 在 _ocr_parser=None 时不再阻塞发送。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.rpa.send_confirm import SendConfirm


def main():
    sc = SendConfirm(config={})
    assert sc._ocr_parser is None, "初始 _ocr_parser 应为 None"

    before = np.full((400, 400, 3), 255, dtype=np.uint8)
    after = np.full((400, 400, 3), 255, dtype=np.uint8)

    turn = sc.inspect_current_customer_turn(before, "测试消息")
    print("inspect_current_customer_turn:", turn)
    assert turn.get("ok") is True, "无 OCR 解析器时应跳过校验"
    assert "skipped" in str(turn.get("reason", "")), "reason 应表明已跳过"

    confirm = sc.confirm_send(before, after, expected_reply="回复")
    print("confirm_send:", confirm)
    # 两张空白图无变化，发送不应被确认，但不应因 OCR 不可用而崩溃
    assert "ocr" not in str(confirm.get("reason", "")).lower() or confirm.get("reason") == "sent_reply_text_not_found"

    print("PASS: SendConfirm 在 _ocr_parser=None 时不再阻塞发送流程")


if __name__ == "__main__":
    main()
