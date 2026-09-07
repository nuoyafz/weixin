"""联系人识别 + 列表视图判别 回归测试。

锁定两类已修复的 bug：
  A. _extract_contact 在会话视图下返回 ''：
     根因是头部标题名 x_min 比分隔线检测的 chat_left 略小（实测 480 < 493），
     旧逻辑 `x_min >= chat_left` 把发言人名字误杀。改用「中心 x 落在会话区」。
  B. view_gate 把列表视图误判成 conversation：
     列表视图头像/状态点含少量绿色，旧 green_ratio_conv=0.0025 误触发 conversation 投票。

用法:
  ./venv/Scripts/python.exe tools/test_contact_and_list_view.py
"""
import sys
import os

sys.path.insert(0, ".")

from src.clean_perception.reader import WechatScreenReader, _OcrLine
from src.clean_perception.layout import LayoutAnchors

FIXTURE_DIR = os.path.join("data", "wechat", "fixtures")
CONV_IMG = os.path.join(FIXTURE_DIR, "conv_filehelper.png")   # 会话：文件传输助手
LIST_IMG = os.path.join(FIXTURE_DIR, "list_view.png")         # 列表视图


def _mk(text, x_min, y_min, x_max=None, y_max=None, cy=None, score=1.0,
        low_conf=False):
    x_max = x_max if x_max is not None else x_min + 130
    h = (y_max - y_min) if y_max is not None else 32
    y_max = y_max if y_max is not None else y_min + h
    cy = cy if cy is not None else (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    return _OcrLine(text=text, score=score, cx=cx, cy=cy,
                    x_min=float(x_min), x_max=float(x_max),
                    y_min=float(y_min), y_max=float(y_max), low_conf=low_conf)


def test_extract_contact_handles_left_overflow():
    """复现真实 bug：头部标题 x_min(480) < chat_left(493)，旧逻辑误杀 -> 应正确抽出。"""
    # 窗口 1405x1021，分隔线把 chat_left 估成 493
    anchors = LayoutAnchors(chat_left=493, header_bottom=102, chat_right=1405)
    lines = [
        # 右上角控制按钮（cy 极小，应被 y_min>=46 排除）
        _mk("口", 1287, 18, x_max=1305, y_max=33, cy=24),
        # 左侧栏搜索框（中心 x 远小于会话区，应被排除）
        _mk("Q 搜索", 129, 71, x_max=203, y_max=99, cy=85),
        # 会话头部联系人：x_min=480 < chat_left=493，但中心 547 在会话区 -> 必须抽出
        _mk("文件传输助手", 480, 66, x_max=614, y_max=98, cy=82),
        # 左侧栏某会话列表项（cy 超过头部带，不在头部）
        _mk("家庭群", 187, 239, x_max=255, y_max=267, cy=253),
    ]
    reader = WechatScreenReader()
    got = reader._extract_contact(lines, anchors, h=1021, w=1405)
    assert got == "文件传输助手", f"期望抽出'文件传输助手'，实际={got!r}"
    print(f"  抽出联系人={got!r}（chat_left={anchors.chat_left}，标题 x_min=480 已正确放行）")


def test_extract_contact_excludes_left_sidebar_search():
    """左侧栏搜索框不应被当成联系人。"""
    anchors = LayoutAnchors(chat_left=493, header_bottom=102, chat_right=1405)
    lines = [
        _mk("Q 搜索", 129, 71, x_max=203, y_max=99, cy=85),
        _mk("文件传输助手", 480, 66, x_max=614, y_max=98, cy=82),
    ]
    reader = WechatScreenReader()
    got = reader._extract_contact(lines, anchors, h=1021, w=1405)
    assert got == "文件传输助手"
    print(f"  左侧栏'Q搜索'被排除，联系人={got!r}")


def test_conversation_fixture_extracts_contact():
    """集成：真实会话截图应抽到'文件传输助手'，且感知置信度足够高。"""
    if not os.path.exists(CONV_IMG):
        print(f"  [SKIP] 缺少夹具 {CONV_IMG}")
        return
    import cv2
    img = cv2.imread(CONV_IMG)
    reader = WechatScreenReader()
    a = reader.analyze(img)
    assert a.view == "conversation", f"视图应为 conversation，实际={a.view}"
    assert a.current_contact == "文件传输助手", f"联系人应为'文件传输助手'，实际={a.current_contact!r}"
    assert a.perception_confidence >= 0.7, f"感知置信度应>=0.7，实际={a.perception_confidence:.2f}"
    assert len(a.messages) >= 1, "应解析到至少 1 条消息"
    print(f"  会话视图: contact={a.current_contact!r} perc={a.perception_confidence:.2f} msgs={len(a.messages)}")


def test_list_fixture_is_list():
    """集成：真实列表截图应判为 list，且无消息（左侧栏列表项不能污染聊天区）。"""
    if not os.path.exists(LIST_IMG):
        print(f"  [SKIP] 缺少夹具 {LIST_IMG}")
        return
    import cv2
    img = cv2.imread(LIST_IMG)
    reader = WechatScreenReader()
    a = reader.analyze(img)
    assert a.view == "list", f"视图应为 list，实际={a.view}（列表误判会话会让列表项污染消息）"
    assert len(a.messages) == 0, f"列表视图不应解析出消息，实际={len(a.messages)}"
    print(f"  列表视图: view={a.view} msgs={len(a.messages)}（列表项未污染聊天区）")


def main():
    print("用例1: 头部标题 x_min 略小于 chat_left 时仍能抽出联系人（修复前返回 ''）")
    test_extract_contact_handles_left_overflow()
    print("PASSED\n")

    print("用例2: 左侧栏搜索框不被误当联系人")
    test_extract_contact_excludes_left_sidebar_search()
    print("PASSED\n")

    print("用例3: 真实会话截图 -> 抽到'文件传输助手'，感知置信度>=0.7")
    test_conversation_fixture_extracts_contact()
    print("PASSED\n")

    print("用例4: 真实列表截图 -> 判为 list 且无消息（列表项不污染聊天区）")
    test_list_fixture_is_list()
    print("PASSED\n")

    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
