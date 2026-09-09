"""测试：导航栏徽章扫描的 X 边界放宽 + 绿色图标过滤。

背景（真机 1638x1272，日志 20:22:50）：
  - _nav_badge_metric / _scan_nav_badge 主路径用 NAV_X_END_RATIO(0.075) 扫描，
    把「聊天」图标右上角徽章的右半裁掉，连通块退化成细长弧形，
    被宽高比校验(ASPECT_MIN=0.55)干掉 -> 返回 (0,0) -> 像素快速路径失效
    -> 降级 19s OCR 读数字。
  - 浅色主题下「聊天」图标本身是绿色，会被 green_mask 抓到误当徽章。

修复：
  1. X 边界 0.075 -> 0.12（与 _extract_badges_by_cluster_ocr 对齐）
  2. max_size 40 -> 80（容纳高 DPI 下 60~80px 徽章）
  3. RedDotResult 新增 kind 字段；nav 扫描只留 red/purple，排除绿色图标

本文件锁定契约：
  - _blob_kind 正确分类 red/purple/green/None
  - _find_dots 返回结果携带 kind
  - 徽章 x 落在 (0.075W, 0.12W) 区间时仍能被 nav 扫描检出（旧版会漏）
  - 绿色「聊天」图标被过滤，不误判为徽章
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.rpa.red_dot_detector import RedDotDetector

H, W = 1272, 1638


def _frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_blob_kind_classifies():
    det = RedDotDetector()
    assert det._blob_kind(250, 85, 85) == "red", "红色均值应为 red"
    assert det._blob_kind(0, 227, 111) == "green", "绿色均值应为 green"
    assert det._blob_kind(200, 80, 200) == "purple", "紫色均值应为 purple"
    assert det._blob_kind(150, 150, 150) is None, "灰色均值应为 None"


def test_find_dots_returns_kind():
    det = RedDotDetector()
    img = _frame()
    img[200:226, 150:176] = (85, 85, 250)   # 红实色徽章 26x26
    dots = det._find_dots(img, max_size=80)
    assert dots, "应检测到红徽章"
    assert all(d.kind == "red" for d in dots), f"红徽章 kind 应为 red，实际={[d.kind for d in dots]}"


def test_nav_metric_detects_badge_beyond_0075():
    """徽章 x 在 (0.075W, 0.12W) 区间时，新逻辑应检测到（旧 0.075 会漏检）。"""
    det = RedDotDetector()
    det._debug_log = lambda msg: None
    img = _frame()
    # 徽章 x=[150,180]（>0.075W=122，<0.12W=196），y 落在聊天图标徽章带
    # [0.15H,0.21H]≈[191,267] 内（聊天图标中心 0.18H≈229，真机校准值）
    img[205:230, 150:180] = (85, 85, 250)
    m = det._nav_badge_metric(img)
    assert m is not None and m[1] > 0, f"0.12 边界应检测到徽章，实际={m}"


def test_nav_metric_filters_green_icon():
    """绿色「聊天」图标不应被当成 nav 徽章。"""
    det = RedDotDetector()
    det._debug_log = lambda msg: None
    img = _frame()
    # 绿色聊天图标放聊天徽章带内（模拟浅色主题聊天 tab 图标，BGR=绿）
    img[205:230, 150:180] = (111, 227, 0)
    m = det._nav_badge_metric(img)
    assert m is None or m[1] == 0, f"绿色图标应被过滤，实际={m}"


def test_scan_nav_badge_detects_badge_beyond_0075():
    """_scan_nav_badge 主路径也应检出 x>0.075W 的徽章，不再只靠 cluster-OCR。"""
    det = RedDotDetector()
    det._debug_log = lambda msg: None
    det._read_badge_number = lambda img, box: 2   # 桩：读数字返回 2
    img = _frame()
    # 徽章须落在 2134 版扫描窗内：y ∈ [0.115H≈146, 0.17H≈216) 且 x < 0.12W≈196
    # 真实形态：红底白字（白块居中）——红包围白校验（2026-09-08）要求
    # 数字笔画被红色包围，实心红块会被判为假徽章弃读。
    img[155:181, 150:180] = (85, 85, 250)
    img[163:173, 160:170] = (255, 255, 255)
    res = det._scan_nav_badge(img)
    assert res is not None, "_scan_nav_badge 应命中徽章"
    assert res.get("kind") == "nav_badge", res
    assert res.get("unread_count") == 2, res


def test_nav_metric_falls_back_to_color_segmentation():
    """抗锯齿把徽章碎成细长条（_find_dots 因 aspect 过滤失败）时，颜色分割兜底仍能检出。"""
    det = RedDotDetector()
    det._debug_log = lambda msg: None
    img = _frame()
    # 5x20 细长红色块，宽高比 0.25 < ASPECT_MIN=0.55，_find_dots 会过滤
    img[205:225, 160:165] = (55, 55, 255)
    m = det._nav_badge_metric(img)
    assert m is not None and m[1] > 0, f"颜色分割兜底应检出细长红碎片，实际={m}"


if __name__ == "__main__":
    test_blob_kind_classifies()
    print("PASS blob_kind_classifies")
    test_find_dots_returns_kind()
    print("PASS find_dots_returns_kind")
    test_nav_metric_detects_badge_beyond_0075()
    print("PASS nav_metric_detects_badge_beyond_0075")
    test_nav_metric_filters_green_icon()
    print("PASS nav_metric_filters_green_icon")
    test_scan_nav_badge_detects_badge_beyond_0075()
    print("PASS scan_nav_badge_detects_badge_beyond_0075")
    test_nav_metric_falls_back_to_color_segmentation()
    print("PASS nav_metric_falls_back_to_color_segmentation")
    print("\nALL_NAV_BADGE_WIDE_TESTS_PASSED")
