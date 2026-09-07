"""回归测试：双击置顶后的 nav 未读验证（_verify_pin_success）。

锁定此前 Bug：进入会话后微信 CEF 的"聊天" tab 徽章刷新有数百毫秒延迟，
旧逻辑只 sleep(0.4) 单次重扫，会截到过渡帧把已清的未读读成"未变" → 误判 fail。
新逻辑间隔 0.5s 重扫 3 次，任一时刻出现"减少/消失"即判 success。

运行：python tools/test_verify_pin.py
"""
import sys
sys.path.insert(0, ".")

from src.rpa.red_dot_detector import RedDotDetector


def _make_detector():
    """构造 RedDotDetector 实例（不依赖 OCR 引擎/窗口）。"""
    return RedDotDetector.__new__(RedDotDetector)


def _test(scan_seq, before_num, expect_status, capture_fails=False):
    """scan_seq: 每次重扫 _scan_nav_badge 返回的 unread_count 序列（None=徽章消失）。
    capture_fails=True 模拟三次截图全部失败（_capture 返回 None）。"""
    det = _make_detector()
    it = iter(scan_seq)

    def fake_capture(hwnd):
        return None if capture_fails else object()  # 非 None 即可

    def fake_scan_nav_badge(img):
        val = next(it, None)
        if val is None:
            return None
        return {"unread_count": val, "kind": "nav_badge"}

    det._capture = fake_capture
    det._scan_nav_badge = fake_scan_nav_badge

    status, after = det._verify_pin_success(0, 1000, 800, None, before_num)
    ok = status == expect_status
    print(f"  {'PASS' if ok else 'FAIL'}  scan_seq={scan_seq} before={before_num} "
          f"-> status={status} after={after} (expect={expect_status})")
    return ok


def main():
    results = []
    print("[1] 过渡帧：前两次读到 1(=before)，第三次徽章消失 -> 应 success")
    results.append(_test([1, 1, None], 1, "success"))

    print("[2] 过渡帧：前两次 1，第三次降到 0 -> 应 success")
    results.append(_test([1, 1, 0], 1, "success"))

    print("[3] 第一次就消失 -> 立即 success（不浪费后两次）")
    results.append(_test([None], 1, "success"))

    print("[4] 始终 1(=before) -> 应 fail（确实未生效）")
    results.append(_test([1, 1, 1], 1, "fail"))

    print("[5] 始终 2(>before=1) -> 应 fail")
    results.append(_test([2, 2, 2], 1, "fail"))

    print("[6] before 读不出(None) -> 应 unknown（不强行判失败）")
    results.append(_test([1, 1, 1], None, "unknown"))

    print("[7] 截图连续失败 -> 应 unknown（不误判 fail）")
    results.append(_test([], 1, "unknown", capture_fails=True))

    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\nVERIFY_PIN_TESTS: {passed}/{total} passed")
    if passed == total:
        print("ALL_VERIFY_PIN_TESTS_PASSED")
        return 0
    print("VERIFY_PIN_TESTS_FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
