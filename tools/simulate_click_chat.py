"""端到端模拟：点击未读联系人 → 本地 OCR 读取聊天记录 → 返回结构化信息。

用法:
  python tools/simulate_click_chat.py <点击前列表图> <点击后会话图>

阶段1（模拟点击）：对列表图跑与生产同款的 red_dot 检测，给出应点击的窗口坐标；
阶段2（本地 OCR）：对会话图跑 clean_reader（本地 RapidOCR，与生产同一路径），
  输出联系人 / 聊天记录 / 最新消息 / 意图 / 是否应回复。
全程不联网，纯本地验证感知链路。
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402


def stage1_detect_click(list_img: str) -> None:
    """阶段1：红点检测 + 给出点击坐标。"""
    from src.rpa.red_dot_detector import RedDotDetector

    print("=" * 62)
    print("阶段1｜模拟点击未读联系人")
    print("=" * 62)
    print("输入画面 : %s" % list_img)
    img = cv2.imread(list_img)
    if img is None:
        print("  [失败] 无法读取图片")
        return
    h, w = img.shape[:2]
    print("窗口尺寸 : %dx%d" % (w, h))

    det = RedDotDetector()
    nav = det._scan_nav_badge(img)
    dots = det._scan_contact_dots(img)
    print("导航徽章 : %s" % ("有（聊天图标有未读总数）" if nav else "无"))
    if not dots:
        print("联系人红点 : 0 个 —— 未发现可点击的未读联系人")
        return
    print("联系人红点 : %d 个" % len(dots))
    for i, d in enumerate(dots):
        print("  #%d 中心=(%d, %d) 尺寸=%dx%d" % (
            i, d.get("center_x", 0), d.get("center_y", 0),
            d.get("w", 0), d.get("h", 0)))
    first = dots[0]
    print("→ 模拟点击 : 窗口坐标 (%d, %d)（真实鼠标点击该未读联系人）"
          % (first.get("center_x", 0), first.get("center_y", 0)))


def stage2_read_chat(conv_img: str) -> None:
    """阶段2：本地 OCR 读取聊天记录。"""
    from src.clean_perception.reader import WechatScreenReader

    print()
    print("=" * 62)
    print("阶段2｜本地 OCR 读取聊天记录")
    print("=" * 62)
    print("输入画面 : %s" % conv_img)
    img = cv2.imread(conv_img)
    if img is None:
        print("  [失败] 无法读取图片")
        return

    a = WechatScreenReader().analyze(img)
    print("视图判定 : %s (conf=%.3f)" % (a.view, a.view_confidence))
    print("联系人   : %r" % a.current_contact)
    print("意图     : %s" % a.intent)
    print("自己最新 : %s" % a.is_self_latest)
    should = getattr(a, "should_reply", None)
    if should is None:
        # ChatAnalysis 不含 should_reply（由决策引擎派生），用意图推断
        should = a.intent in ("customer_message",) and not a.is_self_latest
    print("是否回复 : %s" % should)
    print("-" * 62)
    print("聊天记录（按时间顺序，本地 OCR 提取）:")
    if not a.messages:
        print("  （无消息 —— 视图不是会话或 OCR 未提取到内容）")
        return
    for m in a.messages:
        who = {"left": "对方", "right": "自己", "center": "系统"}.get(m.side, m.side)
        print("  [%s] %s" % (who, m.text))
    print("-" * 62)
    latest = a.messages[-1] if a.messages else None
    if latest:
        print("最新一条 : [%s] %s" % (
            {"left": "对方", "right": "自己"}.get(latest.side, latest.side),
            latest.text))
        print("→ 应回复  : %s" % ("是（等待 LLM 生成回复）" if should else "否"))


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: python tools/simulate_click_chat.py <列表图> <会话图>")
        return 1
    stage1_detect_click(sys.argv[1])
    stage2_read_chat(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
