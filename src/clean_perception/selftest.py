"""clean_perception 无屏自测。

生成一张「合成微信聊天截图」（PIL 绘制头部/气泡/输入框 + 真实字体渲染中文），
跑 WechatScreenReader，验证能从截图中正确抽出：
  - 头部联系人名
  - 最新一条客户(左)消息
  - 输入框草稿
不依赖真实微信、不依赖屏幕。
"""
from __future__ import annotations

import os
import sys

# 让脚本能直接以 `python selftest.py` 运行（把 my_agent/src 加入路径）
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from clean_perception.reader import WechatScreenReader  # noqa: E402


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_bubble(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
                 font: ImageFont.FreeTypeFont, own: bool) -> int:
    """画一个气泡（圆角矩形 + 文本），返回气泡底部 y。"""
    pad_x, pad_y = 14, 10
    # 估算文本宽度/高度
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    color = (220, 245, 200) if own else (235, 235, 235)  # 自己绿、对方灰
    draw.rounded_rectangle([x, y, x + bw, y + bh], radius=12, fill=color)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=(20, 20, 20))
    return y + bh + 24  # 间距


def build_synthetic_chat(width: int = 900, height: int = 720) -> Image.Image:
    """合成一张**带侧边栏**的微信会话截图（贴近真实布局）。

    真实微信桌面版永远是「左侧会话列表 + 右侧聊天区」的结构，
    这是此前感知层把列表项当消息的根因。因此合成图必须包含侧边栏，
    才能真实检验「区域锚定 + 只解析聊天区」这条硬约束是否生效。
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    header_font = _font(30)
    bubble_font = _font(26)
    draft_font = _font(26)
    list_font = _font(18)

    sidebar_w = int(width * 0.33)     # 侧边栏宽（真实约 0.28~0.33）

    # ---- 左侧：会话列表（侧边栏）----
    draw.rectangle([0, 0, sidebar_w, height], fill=(247, 247, 247))
    draw.text((18, 18), "搜索", font=list_font, fill=(150, 150, 150))
    row_y = 70
    for name, preview in [(" file_helper", "01/04 文件传输助手 嗯嗯"),
                          ("李四", "2025/12/26 好的收到"),
                          ("王五", "昨天 明天见")]:
        draw.rectangle([14, row_y, 54, row_y + 40], fill=(200, 205, 210))
        draw.text((64, row_y + 2), name, font=list_font, fill=(30, 30, 30))
        draw.text((64, row_y + 22), preview, font=list_font, fill=(140, 140, 140))
        row_y += 64
    # 分隔线
    draw.line([(sidebar_w, 0), (sidebar_w, height)], fill=(214, 214, 214), width=2)

    # ---- 右侧：聊天区 ----
    cx0 = sidebar_w
    draw.rectangle([cx0, 0, width, 70], fill=(245, 245, 245))
    draw.line([(cx0, 70), (width, 70)], fill=(210, 210, 210), width=1)
    draw.text((cx0 + (width - cx0) / 2 - 30, 22), "张三", font=header_font, fill=(0, 0, 0))

    # 气泡：客户在聊天区左侧、自己在聊天区右侧
    y = 100
    y = _draw_bubble(draw, cx0 + 20, y, "在吗？", bubble_font, own=False)
    y = _draw_bubble(draw, cx0 + 20, y, "这款手机多少钱", bubble_font, own=False)
    y = _draw_bubble(draw, width - 420, y, "您好，这边为您报价", bubble_font, own=True)
    y = _draw_bubble(draw, cx0 + 20, y, "有优惠吗", bubble_font, own=False)

    # 输入区（仅聊天区底部）
    draw.rectangle([cx0, height - 90, width, height], fill=(248, 248, 248))
    draw.line([(cx0, height - 90), (width, height - 90)], fill=(210, 210, 210), width=1)
    draw.text((cx0 + 30, height - 62), "您好，这款是2999元，现在下单立减200。",
              font=draft_font, fill=(40, 40, 40))
    return img


def main() -> int:
    print("== 生成合成微信聊天截图 ==")
    img = build_synthetic_chat()

    print("== 运行 WechatScreenReader ==")
    reader = WechatScreenReader()
    analysis = reader.analyze(img)

    print(f"image_size        = {analysis.image_size}")
    print(f"current_contact   = {analysis.current_contact!r}")
    print(f"draft_text        = {analysis.draft_text!r}")
    print(f"last_customer_msg = {analysis.last_customer_message!r}")
    print(f"is_self_latest    = {analysis.is_self_latest}")
    print(f"intent            = {analysis.intent}")
    print(f"messages({len(analysis.messages)}):")
    for m in analysis.messages:
        print(f"   [{m.side:6}] {m.text!r}")

    # 断言（宽松：OCR 可能个别字误差，只校验关键子串）
    ok = True
    checks = []

    def check(name: str, cond: bool):
        checks.append((name, cond))
        nonlocal ok
        ok = ok and cond

    check("contact_extracted", bool(analysis.current_contact))
    check("contact_has_张三", "张三" in analysis.current_contact)
    check("customer_msg_found", bool(analysis.last_customer_message))
    check("customer_msg_is_优惠", "优惠" in analysis.last_customer_message
          or "有优惠吗" in analysis.last_customer_message)
    check("draft_found", bool(analysis.draft_text))
    check("draft_has_2999", "2999" in analysis.draft_text)
    check("self_not_latest", analysis.is_self_latest is False)
    check("intent_customer", analysis.intent == "customer_message")

    # ---- v2 新增断言：视图闸门 + 区域锚定 + 置信度 ----
    check("view_is_conversation", analysis.view == "conversation")
    check("perception_confidence_ok", analysis.perception_confidence >= 0.60)

    # 区域锚定：侧边栏右边界应落在合成图真实分隔线(33%)附近
    if analysis.anchors is not None:
        expect = int(analysis.image_size[0] * 0.33)
        check("anchor_near_separator", abs(analysis.anchors.sidebar_right - expect) <= 60)

    # 硬约束：任何消息都不得来自侧边栏（此前 8 条伪消息的根因）
    if analysis.anchors is not None:
        leaked = [m for m in analysis.messages
                  if m.x_max <= analysis.anchors.chat_left]
        check("no_sidebar_leak", len(leaked) == 0)
    else:
        check("no_sidebar_leak", False)

    # 左右归属：应同时存在客户(左)与自己(右)两种气泡
    sides = {m.side for m in analysis.messages}
    check("has_customer_side", "left" in sides)
    check("has_self_side", "right" in sides)

    print("\n== 校验 ==")
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    print("\nRESULT:", "ALL_PASS" if ok else "HAS_FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
