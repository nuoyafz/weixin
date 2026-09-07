"""回归测试：_verify_pin_success 的「像素快速路径」（性能优化 1）。

背景：原实现每次验证都要 3 次重扫 + OCR 读未读数字（实测 23~64s）。
新实现传入点击顶行前的基准帧(before_img)时，先用纯像素检测导航栏徽章
是否消失/缩小，毫秒级判成功；像素不可靠时才降级回原 OCR 逻辑。

本文件锁定三条关键契约：
  1. 徽章消失 -> 秒判 success，且**一次 OCR 都不调用**（提速的核心）
  2. 基准帧测不到徽章（融合形态/异常帧）-> 降级 OCR，不误判成功
  3. 徽章仍在（未读未清零）-> 降级 OCR 读数字，保持原 fail 语义
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

H, W = 1190, 1521


def _frame_with_badge():
    """构造导航栏带未读徽章的帧：黑底 + nav 区域一块纯红方块(BGR)。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # nav 区域：x<0.18W，y 落在聊天图标徽章带 [0.15H,0.21H]≈[179,249]
    # （聊天图标中心 0.18H≈214，1638x1264 真机校准值）。注意：不能用纯红
    # (0,0,255)！_find_dots 的红色判定含 (g+b)>0 条件，纯红绿蓝通道为 0 反而
    # 检测不到。这里用微信未读红点实色 #FA5556 -> BGR(85, 85, 250)，保证
    # g+b>0 且通过 _is_red_blob 校验。
    img[196:221, 48:74] = (85, 85, 250)
    return img


def _frame_without_badge():
    """构造导航栏无徽章的帧（纯黑）。"""
    return np.zeros((H, W, 3), dtype=np.uint8)


def _make_detector(after_frames):
    """构造检测器：_capture 依次返回 after_frames，并记录 OCR 调用次数。"""
    det = RedDotDetector()
    it = iter(after_frames)

    def fake_capture(hwnd):
        return next(it, None)

    calls = {"ocr": 0}

    def fake_scan_nav_badge(img):
        calls["ocr"] += 1
        return {"unread_count": 1, "kind": "nav_badge"}

    det._capture = fake_capture
    det._scan_nav_badge = fake_scan_nav_badge
    det._debug_log = lambda msg: None
    return det, calls


def test_nav_badge_metric_basics():
    """_nav_badge_metric：有徽章/无徽章/None 三种输入。"""
    det = RedDotDetector()
    det._debug_log = lambda msg: None

    m = det._nav_badge_metric(_frame_with_badge())
    assert m is not None and m[1] > 0, f"应检测到徽章，实际={m}"

    m2 = det._nav_badge_metric(_frame_without_badge())
    assert m2 is not None and m2[1] == 0, f"不应检测到徽章，实际={m2}"

    assert det._nav_badge_metric(None) is None, "None 帧应返回 None"


def test_pixel_fast_path_badge_gone_skips_ocr():
    """徽章消失 -> success，且零 OCR 调用（这是提速的核心断言）。"""
    det, calls = _make_detector([_frame_without_badge()])
    status, after = det._verify_pin_success(
        0, W, H, None, 1, before_img=_frame_with_badge())
    assert status == "success", f"徽章消失应判成功，实际={status}"
    assert calls["ocr"] == 0, (
        f"像素快速路径不应调用 OCR，实际调用了 {calls['ocr']} 次")


def test_pixel_fallback_when_before_frame_has_no_badge():
    """基准帧测不到徽章（融合形态/异常帧）-> 重采仍无徽章 -> 降级 OCR，不误判成功。

    新语义（用户要求：识别失败立即重采一次，第二次仍失败才降级 OCR）：
    这里让重采帧同样无徽章，模拟「过渡帧确实落空」的真实失败场景。
    """
    det, calls = _make_detector([_frame_without_badge()] * 4)
    status, after = det._verify_pin_success(
        0, W, H, None, 1, before_img=_frame_without_badge())
    # 降级后 OCR 桩恒返回 1 == before=1 -> 未读未减少 -> fail
    assert status == "fail", f"未读未减少应判 fail，实际={status}"
    assert calls["ocr"] > 0, "基准帧无徽章时应降级 OCR，不能静默判成功"


def test_pixel_retry_catches_transient_miss():
    """重采命中：基准帧无徽章，但重采帧检出徽章 -> 走像素快速路径，且后续消失即判成功。

    对应真机首轮「双击置顶动画未稳定，基准帧截到无徽章帧」的场景：
    用户要求识别失败立即重采，重采命中即启用像素路径（免去 19s OCR）。
    """
    # 帧序列：重采#1=有徽章(作为 before 度量)，像素循环 3 帧=无徽章(徽章已消失)
    det, calls = _make_detector(
        [_frame_with_badge(), _frame_without_badge(),
         _frame_without_badge(), _frame_without_badge()])
    status, after = det._verify_pin_success(
        0, W, H, None, 1, before_img=_frame_without_badge())
    assert status == "success", f"重采命中后应走像素路径判成功，实际={status}"
    assert calls["ocr"] == 0, (
        f"重采命中后像素路径不应调用 OCR，实际调用了 {calls['ocr']} 次")


def test_pixel_fallback_when_badge_still_present():
    """徽章仍在 -> 像素无法区分数字变小，降级 OCR 保持原 fail 语义。"""
    # 需 6 帧：像素快速路径 3 次重扫 + 降级后 OCR 路径 3 次重扫。
    # 帧数不足会让 OCR 阶段拿到 None 而判 unknown，掩盖真实的降级行为。
    det, calls = _make_detector([_frame_with_badge()] * 6)
    status, after = det._verify_pin_success(
        0, W, H, None, 1, before_img=_frame_with_badge())
    assert status == "fail", f"徽章仍在且数字未变应判 fail，实际={status}"
    assert calls["ocr"] > 0, "徽章仍在时应降级 OCR 读数字确认"


if __name__ == "__main__":
    test_nav_badge_metric_basics()
    print("PASS nav_badge_metric_basics")
    test_pixel_fast_path_badge_gone_skips_ocr()
    print("PASS pixel_fast_path_badge_gone_skips_ocr")
    test_pixel_fallback_when_before_frame_has_no_badge()
    print("PASS pixel_fallback_when_before_frame_has_no_badge")
    test_pixel_retry_catches_transient_miss()
    print("PASS pixel_retry_catches_transient_miss")
    test_pixel_fallback_when_badge_still_present()
    print("PASS pixel_fallback_when_badge_still_present")
    print("\nALL_PIXEL_VERIFY_TESTS_PASSED")
