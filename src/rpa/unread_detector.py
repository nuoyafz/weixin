"""未读消息检测器 —— 对齐原版 app.rpa.unread_detector。

原版 UnreadDetector 使用 pywinauto UIA 扫描联系人列表：
  1. Badge 控件检测（未读数 "1", "3", "99+"）
  2. 红点指示器检测（联系人名旁小圆点）
  3. UIA 父子关联提取联系人名称
  4. 屏幕坐标验证
  5. 自动滚动（未读不在可见区域时）
  6. 点击联系人名中心（而非红点偏移）
  7. 自然人鼠标移动

同时保留像素级红点检测作为回退通道。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import re
import time
import math
import random

import numpy as np

from .red_dot_detector import RedDotResult, RedDotDetector
from ..brain.skip_contacts import is_blacklisted_contact, BUILTIN_CONTACT_BLACKLIST


# =====================================================================
# 数据结构
# =====================================================================

class ContactDot(NamedTuple):
    """一条有未读的位置（侧边栏某联系人处）。"""
    x: int
    y: int
    w: int
    h: int
    center_x: int
    center_y: int
    dot_count: int

    def as_clickable(self) -> tuple:
        """返回可直接用于点击的 (x, y) —— 头像位置。"""
        avatar_x = max(28, int(self.center_x * 0.18))
        return (avatar_x, self.center_y)

    def as_avatar_click(self, window_width: int) -> tuple:
        """按窗口宽度比例计算头像精确坐标。"""
        if window_width <= 0:
            return self.as_clickable()
        avatar_x = 100
        avatar_x = max(70, min(avatar_x, 140))
        return (avatar_x, self.center_y)


@dataclass
class UnreadSidebar:
    contacts: List[ContactDot] = field(default_factory=list)
    error: str = ""


@dataclass
class UnreadResult:
    """find_and_click_unread 的返回结果（对齐原版）。"""
    found: bool = False
    contact: str = ""
    clicked: bool = False
    reason: str = ""
    unread_count: int = 0
    unread_items: List[dict] = field(default_factory=list)


# =====================================================================
# 系统联系人黑名单（对齐原版）
# =====================================================================

SYSTEM_CONTACTS = {
    "微信支付", "微信团队", "微信运动", "QQ邮箱提醒", "微信游戏",
    "微信安全", "微信广告", "微信公众平台", "微信读书", "腾讯新闻",
    "微信红包", "微信电话", "QQ音乐", "腾讯视频", "订阅号",
    "服务号", "公众号", "文件传输助手", "企业微信",
}

# 额外黑名单（可配置）
DEFAULT_CONTACT_BLACKLIST = set()
DEFAULT_SKIP_CONTACTS = set()


# =====================================================================
# UnreadDetector —— 对齐原版
# =====================================================================

class UnreadDetector:
    """基于 UIA（pywinauto）+ 像素红点检测的未读消息定位器。

    对齐原版 app.rpa.unread_detector.UnreadDetector：
      - 优先使用 UIA 扫描 Badge 控件和红点指示器
      - 像素级红点检测作为回退通道
      - 自动滚动 + 自然人鼠标点击
    """

    # 像素检测常量
    NAV_END = 55
    TOP_SKIP = 75
    AVATAR_RIGHT_X_MIN = 85
    AVATAR_RIGHT_X_MAX = 120
    CLUSTER_GAP = 8

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 rpa_click: bool = True,
                 red_dot: Optional[RedDotDetector] = None,
                 cluster_gap: Optional[int] = None):
        self.config = config or {}
        self.red_dot = red_dot or RedDotDetector()
        self.rpa_click = rpa_click
        if cluster_gap is not None:
            self.CLUSTER_GAP = cluster_gap

        # ---- 黑名单（对齐原版） ----
        self.system_contacts: set = set(SYSTEM_CONTACTS)
        self.contact_blacklist: set = set(DEFAULT_CONTACT_BLACKLIST)
        self.test_mode: bool = False
        self.use_file_transfer: bool = False
        self.skip_contacts: set = set(DEFAULT_SKIP_CONTACTS)
        self.blacklists: list = []

        # 合并软件内置黑名单（电商/外卖/出行/短视频/快递/运营商/银行/支付/公众号等），
        # 保证即使未加载 config，这些联系人也不会被点击/分析。
        self.system_contacts.update(BUILTIN_CONTACT_BLACKLIST)
        self.contact_blacklist.update(BUILTIN_CONTACT_BLACKLIST)

        # 加载配置黑名单
        if self.config.get("system_contacts"):
            self.system_contacts.update(self.config["system_contacts"])
        if self.config.get("contact_blacklist"):
            self.contact_blacklist.update(self.config["contact_blacklist"])
        if self.config.get("skip_contacts"):
            self.skip_contacts.update(self.config["skip_contacts"])

    # =================================================================
    # 主入口：find_and_click_unread —— 对齐原版
    # =================================================================

    def find_and_click_unread(self, window_handle: int) -> Dict[str, Any]:
        """在微信侧边栏找到第一个未读联系人的未读消息并点击。

        对齐原版 UnreadDetector.find_and_click_unread(window_handle)

        Returns:
            {"found": bool, "contact": str, "clicked": bool, "reason": str}
        """
        result = {"found": False, "contact": "", "clicked": False, "reason": ""}

        try:
            from pywinauto import Application
            from pywinauto.findwindows import find_window
        except ImportError:
            result["reason"] = "pywinauto not available"
            return result

        # 查找微信窗口
        wechat_window = self._find_wechat_window(window_handle)
        if not wechat_window:
            result["reason"] = "WeChat window not found"
            return result

        unread_items = []
        max_scroll = 5

        for scroll_pass in range(max_scroll):
            items = self._scan_for_unread(wechat_window)
            if items:
                unread_items = items
                break
            if scroll_pass < max_scroll - 1:
                self._scroll_contact_list(wechat_window, direction="down", amount=3)
                time.sleep(0.5)

        if not unread_items:
            result["reason"] = "no unread messages found"
            return result

        # 过滤黑名单
        valid_items = []
        for item in unread_items:
            contact = item.get("contact", "")
            if self._is_skipped_contact(contact):
                continue
            valid_items.append(item)

        if not valid_items:
            result["reason"] = "unread messages only from blacklisted contacts"
            return result

        # 点击第一个有效未读联系人
        target = valid_items[0]
        contact = target.get("contact", "")
        count = target.get("count", 0)

        result["found"] = True
        result["contact"] = contact
        result["unread_count"] = count

        try:
            clicked = self._click_contact(target, wechat_window)
            result["clicked"] = clicked
            if clicked:
                result["reason"] = f"found unread from {contact} ({count})"
            else:
                result["reason"] = f"found unread from {contact} but click failed"
        except Exception as e:
            result["reason"] = f"error: {e}"

        return result

    # =================================================================
    # 查找指定联系人 —— 对齐原版 find_specific_contact
    # =================================================================

    def find_specific_contact(self, contact_name: str,
                               window_handle: Optional[int] = None) -> Dict[str, Any]:
        """查找并点击指定联系人（用于测试，如"文件传输助手"）。

        对齐原版 UnreadDetector.find_specific_contact(contact_name)
        """
        result = {"found": False, "contact": contact_name, "clicked": False, "reason": ""}

        try:
            from pywinauto import Application
        except ImportError:
            result["reason"] = "pywinauto not available"
            return result

        if window_handle:
            wechat_window = self._find_wechat_window(window_handle)
        else:
            wechat_window = self._find_wechat_window(None)

        if not wechat_window:
            result["reason"] = "WeChat window not found"
            return result

        try:
            desktop = wechat_window
            # 搜索联系人列表
            all_elements = desktop.descendants()
            for elem in all_elements:
                try:
                    if elem.element_info.control_type == "Text":
                        text = elem.element_info.name or ""
                        if contact_name in text:
                            rect = elem.rectangle()
                            cx = rect.left + (rect.right - rect.left) // 2
                            cy = rect.top + (rect.bottom - rect.top) // 2
                            self._human_click(cx, cy)
                            result["found"] = True
                            result["clicked"] = True
                            return result
                except Exception:
                    continue
            result["reason"] = f"contact {contact_name} not found in contact list"
        except Exception as e:
            result["reason"] = str(e)

        return result

    # =================================================================
    # 查找微信窗口（对齐原版 _find_wechat_window）
    # =================================================================

    def _find_wechat_window(self, window_handle: Optional[int] = None):
        """定位微信主窗口。"""
        try:
            from pywinauto import Application
            from pywinauto.findwindows import find_window

            if window_handle:
                try:
                    return Application(backend="uia").connect(handle=window_handle).window(handle=window_handle)
                except Exception:
                    pass

            # 尝试通过标题查找
            try:
                windows = find_window(title_re=".*微信.*", backend="uia")
                if windows:
                    return Application(backend="uia").connect(handle=windows).window(handle=windows)
            except Exception:
                pass

            # 尝试通过进程名
            try:
                app = Application(backend="uia").connect(title="微信")
                return app.window(title="微信")
            except Exception:
                pass

            return None
        except ImportError:
            return None
        except Exception:
            return None

    # =================================================================
    # 扫描未读（对齐原版 _scan_for_unread）
    # =================================================================

    def _scan_for_unread(self, wechat_window) -> List[Dict[str, Any]]:
        """扫描微信窗口控件，获取未读消息指示器列表。

        返回 list of dicts with keys: contact, count, element, rect

        检测方式：
          1. Badge 控件（带数字 1, 3, 99+ 等）
          2. 红点指示器（无数字的纯红点）
        """
        results = []

        try:
            all_elements = wechat_window.descendants()
        except Exception:
            return results

        # 窗口尺寸（用于边界判断）
        try:
            win_rect = wechat_window.rectangle()
            list_right_bound = win_rect.left + int(win_rect.width() * 0.35)
        except Exception:
            list_right_bound = 400

        prev_contact = ""

        for elem in all_elements:
            try:
                info = elem.element_info
                ctrl_type = info.control_type

                # 方式1：Badge 控件（带数字）
                if ctrl_type in ("Text", "ListItem", "Button", "Image", "Pane"):
                    name = info.name or ""
                    if not name:
                        continue

                    # 匹配未读数 "1", "3", "99+"
                    if re.fullmatch(r"\d{1,3}\+?", name.strip()):
                        rect = elem.rectangle()
                        cx = rect.left + (rect.right - rect.left) // 2
                        # 必须在左侧列表区域
                        if cx > list_right_bound:
                            continue

                        contact = self._find_contact_from_parent(elem)
                        if not contact:
                            contact = prev_contact or ""

                        results.append({
                            "contact": contact,
                            "count": int(name.strip().rstrip("+")),
                            "element": elem,
                            "rect": self._rect_to_dict(rect),
                        })
                        if contact:
                            prev_contact = contact

                    # 方式2：时间格式的行（用于定位联系人位置）
                    elif re.fullmatch(r"\d{1,2}:\d{2}", name.strip()):
                        pass  # 仅用于辅助定位，不直接作为未读

            except Exception:
                continue

        return results

    # =================================================================
    # 从 Badge 元素的父控件查找联系人名（对齐原版 _find_contact_from_parent）
    # =================================================================

    def _find_contact_from_parent(self, badge_element) -> str:
        """通过检查 Badge 元素的父行来查找联系人名称。

        在微信 UIA 树中，联系人列表项通常包含：
          {头像} {名称} {最后消息} {时间} {Badge}
        如果 Badge 有父控件，搜索其兄弟/子控件查找 Text 控件。
        """
        try:
            parent = badge_element.parent()
            if parent:
                children = parent.children()
                candidates = []
                for child in children:
                    try:
                        if child.element_info.control_type == "Text":
                            txt = child.element_info.name or ""
                            txt = txt.strip()
                            if txt and len(txt) < 60 and not re.match(r"\d{1,2}:\d{2}", txt):
                                candidates.append(txt)
                    except Exception:
                        continue

                if candidates:
                    return candidates[0]

                # 尝试查找父级的 Text 子控件
                sib_txt = self._find_contact_text_control(parent)
                if sib_txt:
                    return sib_txt
        except Exception:
            pass

        return ""

    # =================================================================
    # 点击联系人（对齐原版 _click_contact）
    # =================================================================

    def _click_contact(self, target: dict, wechat_window) -> bool:
        """可靠地点击联系人名称区域。

        策略（按优先级）：
          1. 找到联系人名称 Text 控件并点击其中心
          2. 回退：点击 Badge 元素 rect 区域（带偏移）
          3. 回退：点击存储的 rect
        """
        contact = target.get("contact", "")
        element = target.get("element")
        rect = target.get("rect", {})

        # 策略1：查找联系人名称 Text 控件
        target_ctrl = self._find_contact_text_control(wechat_window)
        if target_ctrl:
            try:
                t_rect = target_ctrl.rectangle()
                cx = t_rect.left + (t_rect.right - t_rect.left) // 2
                cy = t_rect.top + (t_rect.bottom - t_rect.top) // 2
                if self._is_on_screen(cx, cy):
                    self._human_click(cx, cy)
                    return True
            except Exception:
                try:
                    # 尝试通过 element_info 获取坐标
                    info = target_ctrl.element_info
                    t_rect = info.rectangle
                    cx = t_rect.left + (t_rect.right - t_rect.left) // 2
                    cy = t_rect.top + (t_rect.bottom - t_rect.top) // 2
                    if self._is_on_screen(cx, cy):
                        self._human_click(cx, cy)
                        return True
                except Exception:
                    pass

        # 策略2：点击 Badge 元素区域（带偏移到联系人名）
        if element:
            try:
                badge_rect = element.rectangle()
                cx = badge_rect.left - 30  # 偏移到左侧联系人名
                cy = badge_rect.top + (badge_rect.bottom - badge_rect.top) // 2
                if self._is_on_screen(cx, cy):
                    self._human_click(cx, cy)
                    return True
            except Exception:
                pass

        # 策略3：使用存储的 rect
        if rect:
            try:
                cx = rect.get("left", 0) + 50
                cy = rect.get("top", 0) + (rect.get("height", 40) // 2)
                if self._is_on_screen(cx, cy):
                    self._human_click(cx, cy)
                    return True
            except Exception:
                pass

        return False

    # =================================================================
    # 查找联系人名称 Text 控件（对齐原版 _find_contact_text_control）
    # =================================================================

    def _find_contact_text_control(self, container) -> Optional[Any]:
        """搜索实际包含联系人名称的 Text 控件。"""
        try:
            descendants = container.descendants()
            for elem in descendants:
                try:
                    if elem.element_info.control_type == "Text":
                        txt = elem.element_info.name or ""
                        txt = txt.strip()
                        if txt and len(txt) < 40 and not re.match(r"\d{1,2}:\d{2}", txt):
                            return elem
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # =================================================================
    # 屏幕坐标验证（对齐原版 _is_on_screen）
    # =================================================================

    def _is_on_screen(self, x: int, y: int, margin: int = 5) -> bool:
        """检查坐标是否在可见屏幕区域内。"""
        import ctypes
        try:
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            return margin <= x <= screen_w - margin and margin <= y <= screen_h - margin
        except Exception:
            return 0 <= x <= 3840 and 0 <= y <= 2160

    # =================================================================
    # 滚动联系人列表（对齐原版 _scroll_contact_list）
    # =================================================================

    def _scroll_contact_list(self, wechat_window,
                              direction: str = "down",
                              amount: int = 3) -> None:
        """后台滚动联系人列表（PostMessage WM_MOUSEWHEEL，不移动真实鼠标）。

        对齐原版 _scroll_contact_list 的意图，但用后台消息注入，
        无需移动真实光标、不抢焦点。
        """
        try:
            import win32gui
            import win32con
            import win32api
            win_rect = wechat_window.rectangle()
            list_x = win_rect.left + 50
            list_y = win_rect.top + win_rect.height() // 2
            hwnd = getattr(wechat_window, "handle", None)
            if not hwnd:
                hwnd = win32gui.WindowFromPoint((int(list_x), int(list_y)))
            if not hwnd:
                return
            client = win32gui.ScreenToClient(hwnd, (int(list_x), int(list_y)))
            wheel = -amount if direction == "down" else amount
            wparam = win32api.MAKELONG(0, int(wheel) * 120)
            lparam = win32api.MAKELONG(int(client[0]), int(client[1]))
            win32gui.PostMessage(hwnd, win32con.WM_MOUSEWHEEL, wparam, lparam)
        except Exception:
            pass

    # =================================================================
    # 自然人鼠标点击（对齐原版 _human_click）
    # =================================================================

    def _human_click(self, target_x: int, target_y: int) -> None:
        """点击（后台 PostMessage，不移动真实鼠标）。"""
        try:
            from .background_clicker import background_click_screen
            background_click_screen(target_x, target_y)
        except Exception:
            pass

    # =================================================================
    # 工具方法
    # =================================================================

    def _rect_to_dict(self, rect) -> dict:
        """将 pywinauto rect 转为 dict。"""
        try:
            return {
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
                "width": rect.right - rect.left,
                "height": rect.bottom - rect.top,
            }
        except Exception:
            return {}

    def _is_skipped_contact(self, contact: str) -> bool:
        """判断联系人是否在黑名单中（内置 + 配置，兼容 substring）。"""
        if not contact:
            return True
        if contact in self.system_contacts:
            return True
        if contact in self.contact_blacklist:
            return True
        if contact in self.skip_contacts:
            return True
        for bl in self.blacklists:
            if contact in bl:
                return True
        # 统一判定（内置全集 + 配置的 substring / 拼音兜底），
        # 兜住「京东」命中「京东快递」这类包含关系。
        return is_blacklisted_contact(contact, self.config)

    # =================================================================
    # 像素级红点检测（回退通道）
    # =================================================================

    def detect(self, bgr: np.ndarray) -> List[ContactDot]:
        """在截图 BGR 帧的侧边栏区域定位未读联系人坐标列表。

        对齐原版 v3.10 过滤链：
          1. 红色像素判定：r>=200, g<=130, b<=130, r-g>=60, r/(g+b)>1.15
          2. 白芯校验：徽章内必须有白色数字像素
          3. 几何范围：徽章中心 cx ∈ [AVATAR_RIGHT_X_MIN, AVATAR_RIGHT_X_MAX]
          4. 垂直位置：cy > TOP_SKIP
        """
        if bgr is None or bgr.ndim != 3:
            return []
        h, w = bgr.shape[:2]

        nav_end = self.NAV_END
        top_skip = self.TOP_SKIP
        avatar_x_max = min(self.AVATAR_RIGHT_X_MAX,
                           max(nav_end + 120, int(w * 0.35)))

        raw = self.red_dot.detect_region(
            bgr, region=(nav_end, top_skip, w - nav_end, max(1, h - top_skip)))

        sidebar_dots = [
            d for d in raw
            if (self.AVATAR_RIGHT_X_MIN <= d.center_x <= avatar_x_max and
                d.center_y > top_skip)
        ]
        if not sidebar_dots:
            return []

        return self._cluster_by_y(sidebar_dots)

    def _cluster_by_y(self, dots: List[RedDotResult]) -> List[ContactDot]:
        """按中心 y 从小到大排序并聚合近邻红点，得到联系人级定位。"""
        ordered = sorted(dots, key=lambda d: d.center_y)

        clusters: List[List[RedDotResult]] = []
        for d in ordered:
            if clusters and d.center_y - clusters[-1][-1].center_y <= self.CLUSTER_GAP:
                clusters[-1].append(d)
            else:
                clusters.append([d])

        contacts: List[ContactDot] = []
        for group in clusters:
            xs = [d.center_x for d in group]
            ys = [d.center_y for d in group]
            min_x = min(d.x for d in group)
            min_y = min(d.y for d in group)
            max_x = max(d.x + d.w for d in group)
            max_y = max(d.y + d.h for d in group)
            contacts.append(ContactDot(
                x=min_x, y=min_y,
                w=max_x - min_x, h=max_y - min_y,
                center_x=int(sum(xs) / len(xs)),
                center_y=int(sum(ys) / len(ys)),
                dot_count=len(group),
            ))
        contacts.sort(key=lambda c: c.center_y)
        return contacts

    def detect_hwnd(self, hwnd: int,
                    screen_capture=None) -> UnreadSidebar:
        """截取窗口后进行像素级未读检测。"""
        try:
            if screen_capture is None:
                from ..capture.screen_capture import ScreenCapture
                screen_capture = ScreenCapture()
            result = screen_capture.capture_window(hwnd)
            if not result.success or result.image is None:
                return UnreadSidebar(error=result.error or "capture failed")
            return UnreadSidebar(contacts=self.detect(result.image))
        except Exception as e:
            return UnreadSidebar(error=str(e))