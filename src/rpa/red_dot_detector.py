"""未读红点感知与点击 —— 干净重写版。

设计思路（参考原项目 VisionLeadAgent 的「感知 → 决策 → 执行」架构）：

  1. 窗口几何单一真相源：所有坐标计算一律基于实时 GetWindowRect，
     绝不信任调用方可能缓存的 park 屏外矩形（left=32767 这类），
     这是此前「点不到/识别不到」这一系列 bug 的根因。
  2. 截图复用 ScreenCapture：本模块不再自己实现 PrintWindow/ImageGrab
     与 park/unpark，而是委托注入的 ScreenCapture（它已正确处理
     「unpark → 截图 → repark」后台托管流程），避免两套重复且易错的逻辑。
  3. 四级感知瀑布的 L1 像素层：扫描左导航栏徽章(nav_badge) 与
     联系人列表红点(contact_dots)。导航徽章只是「聊天图标有未读总数」，
     不等于某个具体会话；真正的未读入口是联系人列表红点，优先级更高。
  4. 执行层：后台 PostMessage 点击（不移动真实鼠标、不抢焦点），
     点联系人直接进入会话，点导航徽章仅展开聊天列表（需二次扫描验证）。

对外保持与原 observe_service 兼容的接口：
  RedDotDetector.find_and_click_unread(window_handle, window_rect=None,
                                       visible_rect=None)
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image


# =====================================================================
# 检测常量（来自对真实微信窗口的实测标定，非反编译猜测）
# =====================================================================

# 红色像素阈值：R 足够高、G/B 足够低、R-G 差值足够大、R/(G+B) 足够大
# 微信未读徽章实测色约 (B,G,R)=(82,82,250)，灰度 UI 文字约 (213,211,208)。
# 该阈值既能命中真实红点，又能把"略带暖色的灰白文字"排除在外。
RED = {"r_min": 190, "g_max": 165, "b_max": 165, "rg_diff": 55, "r_ratio": 0.90}
# 绿色像素阈值：G 足够高、R/B 足够低；实测微信支付/群聊未读徽章在微信浅色主题下为绿色
GREEN = {"g_min": 180, "r_max": 140, "b_max": 140, "gb_diff": 50, "g_ratio": 1.15}
# 红点内部白色数字的最小亮度（近白即算，10 即可）
LIGHT_BRIGHTNESS_MIN = 10
# 红点内部须包含的近白像素下限（0 表示不强制，仅作可选二次校验）
MIN_LIGHT_PIXELS_INSIDE = 0
# 连通分量最小像素数（太小的是噪声）
MIN_PIXELS = 18
# 单个红点（含数字）的合理尺寸区间（微信实测 12×13 ~ 20×20，"99+" 长条徽章 ≤ 22×15）
# 上限 24：避免把彩色群头像/服务图标（实测 35×31、41×40、42×30）误识别为徽章。
DOT_SIZE_MIN = 8
DOT_SIZE_MAX = 24
# 宽高比（圆形/圆角方形徽章接近 1；"99+" 长条徽章 ≤ 1.6）
ASPECT_MIN = 0.55
ASPECT_MAX = 1.7
# 实心度下限：红点应是较实心的圆，灰度文字里夹杂的零星红像素实心度很低
SOLIDITY_MIN = 0.42
# 平均颜色二次校验容差（连通块平均色也须明显偏红，否则是灰度文字/彩色头像误染）
MEAN_R_MIN = 185
MEAN_GB_MAX = 145
MEAN_RG_DIFF = 55
# 真实红点的 G/B 分量接近（均为中低亮度）；彩色头像边缘常出现 G/B 明显分离（如
# 橙色/粉色/洋红色），用该约束把它们剔掉。
MEAN_GB_DIFF_MAX = 30
# 头像列锚定：联系人红点出现在头像右上角；按出现次数最多的 center_x 推断头像列，
# 偏离该列超过该阈值（px）的红块视为其它 UI 元素（如导航图标/搜索框 X）而剔除。
AVATAR_COL_TOLERANCE = 18
# 点击时相对检测点向右偏移，落在「昵称/消息」文本区（而非头像/红点本身），
# 确保单击打开的是该联系人的会话（写死 x=100 在宽窗口会点到头像左侧空白）。
CLICK_OFFSET_X = 48
# 联系人列表扫描区域（相对整窗，比例化以适配不同窗口尺寸）
LIST_X_START_RATIO = 0.02  # 覆盖列表项头像列（含左侧导航栏边缘），后续用众数中心列剔除导航栏
LIST_X_END_RATIO = 0.24
LIST_TOP_SKIP_RATIO = 0.10   # 跳过顶部导航栏/搜索框区域，避免把聊天图标总 badge 当联系人
# 左导航栏扫描区域（聊天/通讯录等图标列，比例化）
NAV_X_END_RATIO = 0.075
NAV_Y_END_RATIO = 0.17
# 唯一被当作 nav_badge（聊天未读指示）扫描的 y 上限：覆盖「聊天」图标及其 badge。
# 实测微信 4.0 1750x1313：用户头像 y=78~138、聊天图标 y=186~213、badge 在图标
# 右上角 y≈180；通讯录 y≈270~300。上限 0.18H 可包含聊天 badge 且远离通讯录。
NAV_BUDDY_Y_MAX_RATIO = 0.18
# 导航栏扫描时跳过顶部用户头像区域，避免把用户头像（如红色衣服/头像框）
# 误判为「聊天未读 badge」。
NAV_AVATAR_SKIP_RATIO = 0.115
# 纯红点未读：面积达到该值（px²）且读不出数字时，视为未读=1（微信"纯红点"形态，
# 表示有未读但未显示数字）。低于该值的是残片/噪声，不降级、不点击。
# 注意：红点面积随窗口缩放（1750 宽下 13px 红点在小窗口 946 宽会缩到 ~7px、
# area≈34-44），所以实际阈值按窗口像素面积等比缩放（见 _min_unread_dot_area）。
MIN_UNREAD_DOT_AREA = 60
# 截图校验：导航栏非黑像素下限（GPU/CEF 渲染不完整的图非黑像素常落在 60~100）
SCREENSHOT_MIN_NONBLACK_NAV = 60
# 点击后等待（让界面响应）
CLICK_SETTLE_DELAY = 0.35


# =====================================================================
# 数据结构
# =====================================================================

class RedDotResult(NamedTuple):
    """单个红点区域的几何信息。"""
    x: int
    y: int
    w: int
    h: int
    area: int
    center_x: int
    center_y: int
    kind: str = "red"  # 颜色类别：red / purple / green（默认 red，兼容旧调用）


# 别名兼容（给 rpa/__init__.py 导出使用）

# 置顶后未读行定位的系统账号黑名单（与 unread_detector.SYSTEM_CONTACTS 同名单）
_SYSTEM_ROW_NAMES = {
    "微信支付", "微信团队", "微信运动", "QQ邮箱提醒", "微信游戏",
    "微信安全", "微信广告", "微信公众平台", "微信读书", "腾讯新闻",
    "微信红包", "微信电话", "QQ音乐", "腾讯视频", "订阅号",
    "服务号", "公众号", "文件传输助手", "企业微信",
}
RedDot = RedDotResult


class RedDotDetection(NamedTuple):
    """红点检测结果集合。"""
    dots: list
    total: int
    source: str  # "pixel" | "uia"


class RedDotDetector:
    """微信未读红点检测器：感知未读 → 后台点击进入会话。

    构造时可注入：
      - screen_capture: ScreenCapture 实例（负责截图 + park/unpark）
      - window_manager: WeChatWindowManager 实例（负责实时矩形/还原）
      - logger: 回调 logger(tag, msg)，日志同时进 UI
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 screen_capture=None,
                 window_manager=None,
                 logger=None):
        self._config = config or {}
        self._screen_capture = screen_capture
        self._window_manager = window_manager
        self._logger = logger

        # 检测参数（可被 config 覆盖，便于调参）
        self._r_min = int(self._cfg("r_min", RED["r_min"]))
        self._g_max = int(self._cfg("g_max", RED["g_max"]))
        self._b_max = int(self._cfg("b_max", RED["b_max"]))
        self._rg_diff = int(self._cfg("rg_diff", RED["rg_diff"]))
        self._r_ratio = float(self._cfg("r_ratio", RED["r_ratio"]))
        self._light_brightness_min = int(self._cfg("light_brightness_min", LIGHT_BRIGHTNESS_MIN))
        self._min_pixels = int(self._cfg("min_pixels", MIN_PIXELS))
        self._dot_size_min = int(self._cfg("dot_size_min", DOT_SIZE_MIN))
        self._dot_size_max = int(self._cfg("dot_size_max", DOT_SIZE_MAX))
        self._aspect_min = float(self._cfg("aspect_min", ASPECT_MIN))
        self._aspect_max = float(self._cfg("aspect_max", ASPECT_MAX))
        self._min_light_pixels_inside = int(self._cfg("min_light_pixels_inside", MIN_LIGHT_PIXELS_INSIDE))
        # 新增：平均颜色二次校验 + 实心度 + 结构锚定
        self._solidity_min = float(self._cfg("solidity_min", SOLIDITY_MIN))
        self._mean_r_min = int(self._cfg("mean_r_min", MEAN_R_MIN))
        self._mean_gb_max = int(self._cfg("mean_gb_max", MEAN_GB_MAX))
        self._mean_rg_diff = int(self._cfg("mean_rg_diff", MEAN_RG_DIFF))
        self._mean_gb_diff_max = int(self._cfg("mean_gb_diff_max", MEAN_GB_DIFF_MAX))
        self._avatar_col_tolerance = int(self._cfg("avatar_col_tolerance", AVATAR_COL_TOLERANCE))
        self._click_offset_x = int(self._cfg("click_offset_x", CLICK_OFFSET_X))
        # 识别模式：double_click_pin=双击置顶 / red_dot=红点直点（其他）
        self._recognition_mode = str(
            self._cfg("recognition_mode", "double_click_pin") or "double_click_pin")

        # 红点徽章数字 OCR（懒加载，仅当需要读取未读数量时初始化）
        self._ocr_engine = None

        self._data_dir = self._config.get("data_dir", "data/debug")
        self._auto_archive = bool(self._config.get("auto_archive", True))
        self._current_hwnd: Optional[int] = None
        self._last_screenshot: Optional[np.ndarray] = None
        self._nav_badge: Optional[dict] = None
        self._contact_dots: List[dict] = []
        self._consecutive_empty: int = 0
        # 头像列时间平滑锚：单点未读时本帧聚类锚定不可用（需 ≥2 点），
        # 用历史多帧的红点 center_x 中位数做先验，过滤偏离列的红块并
        # 让点击坐标稳定落在昵称区。
        self._avatar_col_smooth: Optional[int] = None
        # 联系人红点防死循环守卫（镜像 nav 的 MAX_CONSECUTIVE_NAV_CLICKS）
        self._last_contact_y: int = -1
        self._consecutive_contact_clicks: int = 0
        self.MAX_CONSECUTIVE_CONTACT_CLICKS: int = 3
        # 短期失败标记：点击后未进入会话/发送失败的 y，下一轮跳过，
        # 避免永远卡在第一个红点（如亚磊失败后永远轮不到方舟）。
        self._failed_contact_ys: set = set()
        self._failed_contact_y_frames: Dict[int, int] = {}
        self._click_round: int = 0
        self.FAILED_Y_TTL_ROUNDS: int = 8
        # 昵称 OCR 懒加载（点前黑名单判定用）与本次黑名单命中记录
        self._nick_ocr = None
        self._blacklisted_hits: List[dict] = []

    # =================================================================
    # 主入口
    # =================================================================

    # -----------------------------------------------------------------
    # 点前黑名单拦截（OCR 昵称行 → is_blacklisted_contact）
    # -----------------------------------------------------------------
    def _get_nick_ocr(self):
        """懒加载昵称 OCR 引擎（与徽章数字 OCR 分开，按需初始化）。"""
        if self._nick_ocr is None:
            try:
                from src.local_vision.ocr_engine import OCREngine
                ocr = OCREngine()
                ocr.initialize()
                self._nick_ocr = ocr
            except Exception as e:
                self._debug_log(f"昵称 OCR 初始化失败: {e}")
                self._nick_ocr = False  # 哨兵，避免反复初始化
        return self._nick_ocr if self._nick_ocr is not False else None

    def _is_contact_blacklisted(self, name: str) -> bool:
        """联系人名是否命中黑名单（内置 32 项 + 用户自定义）。"""
        if not name:
            return False
        try:
            from src.brain.skip_contacts import is_blacklisted_contact
            return bool(is_blacklisted_contact(name, self._config))
        except Exception:
            return False

    def _resolve_contact_name(self, img: np.ndarray, dot: dict, win_w: int) -> str:
        """点前：OCR 该行昵称区，返回整行文本（供黑名单判定 / 日志用）。

        微信列表行布局：[头像][昵称+预览][时间][未读红点]。红点在行最右，
        昵称在其左侧，因此裁剪从头像左侧一路覆盖到列表列宽(~36%)，无论昵称
        在左在右都能扫到。借鉴已验证的 _run_unread_v2.row_is_blacklisted。
        """
        ocr = self._get_nick_ocr()
        if ocr is None:
            return ""
        h, w = img.shape[:2]
        cy = int(dot.get("center_y", 0))
        x0 = max(0, int(win_w * 0.01))
        x1 = min(w, int(win_w * 0.36))
        y0 = max(0, cy - 30)
        y1 = min(h, cy + 30)
        if x1 <= x0 or y1 <= y0:
            return ""
        strip = img[y0:y1, x0:x1]
        try:
            res = ocr.run(strip)
            texts = getattr(res, "texts", None)
            if texts is None and hasattr(res, "to_dict_list"):
                texts = [d.get("text", "") for d in res.to_dict_list()]
            row_text = " ".join(str(t) for t in (texts or []))
        except Exception:
            row_text = ""
        return row_text

    def mark_contact_click_failed(self, center_y: int) -> None:
        """外部通知：某个 y 的红点点击后未成功进入会话/发送失败。

        该 y 会在接下来 FAILED_Y_TTL_ROUNDS 轮内被跳过，让后续红点有机会被处理。
        """
        y = int(center_y)
        self._failed_contact_ys.add(y)
        self._failed_contact_y_frames[y] = self._click_round

    def _prune_failed_contact_ys(self) -> None:
        """清理过期的失败标记。"""
        cutoff = self._click_round - self.FAILED_Y_TTL_ROUNDS
        stale = [y for y, frame in self._failed_contact_y_frames.items()
                 if frame < cutoff]
        for y in stale:
            self._failed_contact_ys.discard(y)
            self._failed_contact_y_frames.pop(y, None)

    def _pick_and_click_contact_dot(self, window_handle: int, dots: List[dict],
                                    img: np.ndarray, win_w: int, wm) -> Dict[str, Any]:
        """从候选红点中挑第一个「非黑名单、近期未失败」的点击。

        遍历排序后的候选（按 y 从上到下），对每个：
          1) 防死循环守卫（同一 y 连续点击超限则跳过该候选）；
          2) 近期失败标记跳过（点击后没打开会话/发送失败的红点）；
          3) OCR 该行昵称 → 命中黑名单则跳过（不点击、不分析、不回复）；
          4) 否则点击该候选并立即返回。
        全部被拦截/跳过则返回 found=False（kind=all_blacklisted）。
        """
        self._click_round += 1
        self._prune_failed_contact_ys()
        ranked = sorted(dots, key=lambda d: d["center_y"])
        for target in ranked:
            ty = int(target["center_y"])
            if ty in self._failed_contact_ys:
                self._debug_log(f"点击跳过: y={ty} 在近期失败记录中")
                continue
            if ty == self._last_contact_y:
                self._consecutive_contact_clicks += 1
            else:
                self._consecutive_contact_clicks = 0
                self._last_contact_y = ty
            if self._consecutive_contact_clicks > self.MAX_CONSECUTIVE_CONTACT_CLICKS:
                self._debug_log(
                    f"防死循环守卫: 同一行 y={ty} 已连点 "
                    f">{self.MAX_CONSECUTIVE_CONTACT_CLICKS} 次，跳过")
                self._consecutive_contact_clicks = 0
                self._last_contact_y = -1
                continue

            name = self._resolve_contact_name(img, target, win_w)
            if name and self._is_contact_blacklisted(name):
                self._debug_log(f"黑名单命中 {name!r} (y={ty}) -> 跳过，不点击")
                self._blacklisted_hits.append({
                    "contact": name,
                    "unread_count": target.get("unread_count"),
                })
                continue

            click_x = self._click_x_for_dot(target, win_w)
            method = self._do_click(
                window_handle, click_x, ty, double=False, wm=wm)
            self._debug_log(
                f"点击红点行 行位置y={ty} 未读数="
                f"{target.get('unread_count')} -> {'成功' if method is not None else '失败'}")
            return {
                "found": True, "clicked": method is not None,
                "kind": "contact_dot", "contact": name or target.get("label", ""),
                "entered_conversation": True, "click_method": "sendmessage_inject",
                "reason": "clicked contact dot (blacklist filtered)",
                "unread_count": target.get("unread_count"),
                "click_x": click_x, "click_y": ty,
            }
        # 如果全被近期失败标记跳过，清空标记允许再试（避免永久卡死）
        if self._failed_contact_ys and all(
                int(d["center_y"]) in self._failed_contact_ys for d in ranked):
            self._debug_log("可点红点行全部近期失败过，清空失败记录允许重试")
            self._failed_contact_ys.clear()
            self._failed_contact_y_frames.clear()
            return {
                "found": False, "clicked": False, "kind": "all_recently_failed",
                "contact": "", "entered_conversation": False,
                "reason": "所有可点击红点近期均失败，已清空标记准备重试",
                "unread_count": None,
            }
        return {
            "found": False, "clicked": False, "kind": "all_blacklisted",
            "contact": "", "entered_conversation": False,
            "reason": "所有可点击红点均被黑名单拦截或死循环跳过",
            "unread_count": None,
        }

    def find_and_click_unread(self, window_handle: int,
                              window_rect: Optional[Dict[str, Any]] = None,
                              visible_rect: Optional[Dict[str, Any]] = None,
                              wm=None) -> Dict[str, Any]:
        """在微信侧边栏找到第一个未读红点并点击进入会话。

        Returns:
            {"found", "contact", "clicked", "reason", "nav_badge",
             "contact_dots", "kind", "entered_conversation", "click_method"}
        """
        result: Dict[str, Any] = {
            "found": False, "contact": "", "clicked": False,
            "reason": "", "nav_badge": {}, "contact_dots": [],
            "kind": "none", "entered_conversation": False, "click_method": "",
            "unread_count": None,
        }
        self._current_hwnd = int(window_handle)
        self._nav_badge = None
        self._contact_dots = []

        # —— 几何单一真相源：永远用实时矩形 ——
        # 调用方传入的 rect 可能是缓存的 park 屏外坐标（left=32767），
        # 只用它作为「实时获取失败」时的兜底，且屏外坐标直接丢弃。
        live = self._live_rect(window_handle)
        rect = live
        if rect is None:
            if window_rect and not self._is_offscreen(window_rect):
                rect = window_rect
        if rect is None:
            result["reason"] = "window rect not found"
            return result
        self._window_rect = rect

        # —— 截图（委托 ScreenCapture，内部处理 park/unpark）——
        img = self._capture(window_handle)
        if img is None:
            result["reason"] = "screenshot failed"
            return result
        self._last_screenshot = img

        h, w = img.shape[:2]
        self._debug_log(f"截图完成 尺寸={w}x{h}")

        # —— L1 像素扫描：导航徽章 + 联系人红点 ——
        nav_badge = self._scan_nav_badge(img)
        contact_dots = self._scan_contact_dots(img)
        self._nav_badge = nav_badge
        self._contact_dots = contact_dots
        result["nav_badge"] = nav_badge or {}
        result["contact_dots"] = contact_dots
        self._debug_log(
            f"扫描结果 导航栏未读徽章={'有' if nav_badge is not None else '无'} "
            f"列表红点数={len(contact_dots)}")

        if self._auto_archive:
            self._archive_debug(img)

        # —— 决策与执行 ——
        # 优先级（新增置顶策略）：
        #   1) nav 有未读数字 → 双击聊天图标置顶未读 → 直接点顶行（不扫红点/不黑名单）
        #   2) 列表红点 → 直接点（保持原行为）
        #   3) nav 仅徽章无数字（头像/噪声）→ 不双击，避免误点，走原恢复逻辑
        #   4) 全无 → 主页恢复
        # 只把"读出未读数字"或"面积足够大的完整红块"当作可点击的真实未读徽章；
        # 无数字的小红块很可能是头像/服务图标/消息预览里的彩色噪声，误点风险高。
        clickable_dots = [d for d in contact_dots if self._dot_clickable(d)]
        nav_has_count = bool(
            nav_badge and nav_badge.get("unread_count") is not None)
        # 单一真相源：进入 pin 分支前就锁定「双击前」的未读总数。
        # 否则下面 early-return（426行）与 _click_top_conversation_row 调用（454行）
        # 都会用到未定义的 nav_num，触发 NameError，导致守卫失效 / 整轮崩溃。
        nav_num = nav_badge.get("unread_count") if nav_badge else None

        if nav_has_count and self._recognition_mode == "double_click_pin":
            # —— 不再在双击前用首扫 contact_dots 判空短路 ——
            # 首扫 T1 漏检会误判「列表无红点」→ 误跳过真实未读（复现全员不点）。
            # 双击后置顶列表是否有红点，改由 _click_top_conversation_row 内部的
            # pin_not_effective 信号判定（T1无未读行+顶行非客户+nav>0 → 不点、交重试），
            # 更准且避免空转误杀。
            # —— 强制确保在聊天列表（修真机 bug）——
            # 历史事故：真机上微信偶尔停留在通讯录页就开始这轮，nav 仍能读出徽章，
            # 但双击聊天图标只能让通讯录→聊天列表切换一次，后续坐标全错位；甚至
            # 出现「双击后页面跳错、点搜索框」连锁问题。先单击聊天图标回聊天列表，
            # 校验 contact_dots 分布合理再走 pin。
            try:
                if not self._ensure_chat_list(window_handle, w, h, wm):
                    self._debug_log("[PIN] ensure_chat_list 校验失败，放弃本轮置顶")
                    result.update(
                        found=False, clicked=False, kind="nav_badge",
                        entered_conversation=False,
                        reason="ensure_chat_list failed before pin",
                        unread_count=nav_num)
                    return result
            except Exception as exc:
                self._debug_log(f"[PIN] ensure_chat_list 异常: {exc}")
            # ensure_chat_list 可能改了 rect，重新拿一遍
            try:
                live = self._live_rect(window_handle)
                if live:
                    w, h = live["width"], live["height"]
            except Exception:
                pass
            # 复用第一次扫描已锁定的 nav 未读数（nav_num 在进入本分支前已赋值）。
            # 原此处会重新截图 + 再 OCR 一次 nav 徽章（约 15~20s），但「单击聊天
            # 图标回聊天列表」不改变未读总数，数字不会变，属纯冗余——删除以消除
            # 启动后到双击置顶之间约 20s 的空转。若真机发现切页导致数字变化，
            # 由 _verify_pin_success 在进会话后复验兜底（after<before 判成功）。
            self._debug_log(
                f"[PIN] 识别模式=双击置顶 nav未读数={nav_num}，双击聊天图标置顶未读")
            # 双击置顶 + 点顶行 + 验证：验证失败(未读未减少)时重试双击置顶，
            # 而非直接按成功处理（避免过渡帧误判导致误回复发错）。
            MAX_PIN_RETRY = 3  # 含首次共最多 3 次双击置顶尝试
            top = None
            status = "fail"
            after = nav_num
            for _pin_try in range(MAX_PIN_RETRY):
                if _pin_try == 0:
                    self._debug_log(
                        f"[PIN] 识别模式=双击置顶 nav未读数={nav_num}，双击聊天图标置顶未读")
                else:
                    # 重试前先回聊天列表，避免上次进入的会话视图干扰本次双击置顶
                    try:
                        self._ensure_chat_list(window_handle, w, h, wm=wm)
                        time.sleep(0.4)
                    except Exception:
                        pass
                    self._debug_log(
                        f"nav-pin: 验证失败，重试双击置顶 (第 {_pin_try + 1}/{MAX_PIN_RETRY} 次)")

                self._pin_unread_to_top(window_handle, w, h, wm)

                img2 = self._capture(window_handle)
                top = self._click_top_conversation_row(
                    window_handle, w, h, wm, nav_num=nav_num, img=img2)
                if not top.get("clicked"):
                    if top.get("pin_not_effective"):
                        # 双击置顶未生效（T1无未读行+顶行非客户+nav>0）：
                        # 进入重试，下一轮开头会先回聊天列表再双击，不盲点、不兜底。
                        self._debug_log(
                            "nav-pin: 双击置顶未生效（前置软信号），进入重试")
                        continue
                    # 顶行点击失败：保守回退，不重复点击
                    break
                self._consecutive_empty = 0

                # 顶行昵称 OCR 仅作日志回显，不再用于黑名单拦截
                top_y = top.get("click_y") or int(h * 0.105)
                name = ""
                if img2 is not None:
                    try:
                        name = self._resolve_contact_name(
                            img2, {"center_y": top_y}, w)
                    except Exception:
                        name = ""
                top["contact"] = name or top.get("contact", "")

                # —— 双击置顶成功校验：点击后未读数量是否减少 ——
                # 进入未读会话后微信会清零该会话未读，nav 总数随之下降；
                # 若总数未变化，说明双击未生效 / 点错行 → 重试双击置顶。
                # 传 img2（点击顶行前的基准帧）启用像素快速路径：徽章消失可秒判
                # 成功，免去 3 次 OCR 读数字（原约 20~60s）；徽章仍在时自动降级 OCR。
                status, after = self._verify_pin_success(
                    window_handle, w, h, wm, nav_num, before_img=img2)
                verify_ok = status == "success"
                self._debug_log(
                    f"nav-pin: 验证 双击前未读={nav_num} "
                    f"点击后未读={after} -> "
                    f"{'成功' if verify_ok else ('失败(双击可能未生效)' if status == 'fail' else '无法判定')}")

                # 顶行是否带未读红点（仅供双击生效排查）
                row_had_dot = False
                if img2 is not None:
                    try:
                        cds = self._scan_contact_dots(img2)
                        row_had_dot = any(
                            abs(c["center_y"] - top_y) < int(h * 0.04)
                            for c in cds)
                    except Exception:
                        pass

                entered = bool(top.get("entered_conversation"))
                # 打开的若是系统会话（文件传输助手等），视为「置顶未生效」→ 降级兜底
                opened_system = bool(
                    status == "fail" and entered
                    and self._opened_is_system_contact(window_handle, w, h))

                top["pin_echo"] = {
                    "name": top["contact"],
                    "unread": nav_num,
                    "method": "top_row",
                    "row_had_dot": row_had_dot,
                    "verify": status,
                    "before": nav_num,
                    "after": after,
                    "pin_try": _pin_try + 1,
                }
                top["pin_verify"] = status

                if status != "fail":
                    # success 或 unknown（无法判定）：沿用原行为返回本次点击，
                    # 不重试、不兜底（unknown 表示 before 读不出/截图全失败，强行重试无意义）。
                    return top

                # 验证失败分支（status == "fail"）
                if not entered or opened_system:
                    # 未进入会话，或进入了系统会话（无真实客户未读）
                    # → 退出重试，走红点扫描兜底。
                    self._debug_log(
                        "nav-pin: 验证失败，"
                        + ("打开了系统会话，转红点扫描兜底)"
                           if opened_system else "未进入会话，转红点扫描兜底)"))
                    break

                # status == "fail" and entered and not opened_system：
                # 用户要求——继续双击置顶重试，而不是按成功处理（疑似过渡帧误判）。
                # 下一轮循环开头会先回聊天列表再双击，避免会话视图挡住双击。
                self._debug_log(
                    f"nav-pin: 验证失败(双击前={nav_num} 点击后={after}) "
                    f"但已进入会话，继续双击置顶重试 (第 {_pin_try + 1}/{MAX_PIN_RETRY} 次)")

            # ---- 重试结束后的兜底决策 ----
            # 走到这里的情况：① 顶行点击失败；② 验证失败且未进入会话/系统会话；
            # ③ 重试 MAX_PIN_RETRY 次仍 fail+entered（不再按成功处理）。
            # 对 ②③（曾进入会话）统一降级红点扫描兜底；对 ① 保守返回未点击。
            if top is not None and top.get("clicked"):
                # 曾进入会话但重试耗尽仍未验证成功 → 降级红点扫描兜底
                self._debug_log(
                    f"nav-pin: 重试 {MAX_PIN_RETRY} 次仍未验证成功，降级红点扫描兜底")
                try:
                    self._ensure_chat_list(window_handle, w, h, wm=wm)
                    time.sleep(0.3)
                except Exception:
                    pass
                img_fb = self._capture(window_handle)
                if img_fb is not None:
                    cds2 = self._scan_contact_dots(img_fb)
                    clickable2 = [d for d in cds2 if self._dot_clickable(d)]
                    if clickable2:
                        res = self._pick_and_click_contact_dot(
                            window_handle,
                            sorted(clickable2, key=lambda d: d["center_y"]),
                            img_fb, w, wm)
                        if res.get("clicked"):
                            res["pin_echo"] = None
                            res["pin_verify"] = "fallback_red_dot"
                            self._debug_log(
                                f"nav-pin: 降级成功 点红点会话="
                                f"{res.get('contact')!r}")
                            return res
                result.update(
                    found=True, clicked=False, kind="nav_badge",
                    entered_conversation=False,
                    reason="nav pin retry exhausted, fallback empty",
                    unread_count=nav_num, pin_verify="fail")
                return result
            # 顶行点击失败（从未进入会话）：保守返回，不重复点击
            self._consecutive_empty += 1
            result.update(
                found=True, clicked=False, kind="nav_badge",
                entered_conversation=False,
                reason="nav pin: top click failed",
                unread_count=nav_num)
            self._debug_log(
                f"nav pin: top click failed unread={nav_num}")
            return result

        # —— 关键修复（启动即点默认会话窗口 / 末尾还有未读却 idle）——
        # 只要列表里扫到红点（已通过颜色+结构锚定双重过滤），就优先点最顶红点，
        # 不再退回「无未读→处理当前打开的会话」。否则启动后 WeChat 若正停在某个
        # 会话里，助手会先处理那个「默认会话窗口」而非按未读列表逐个处理。
        # 严格阈值（_dot_clickable）未过但确实扫到红点时，退而用全量 contact_dots，
        # 由黑名单守卫兜底，避免漏点真实未读（如方舟）。
        if contact_dots:
            self._consecutive_empty = 0
            candidates = clickable_dots if clickable_dots else contact_dots
            res = self._pick_and_click_contact_dot(
                window_handle,
                sorted(candidates, key=lambda d: d["center_y"]),
                img, self._window_rect.get("width", 0), wm)
            if res.get("clicked"):
                return res
            # 全黑名单/点击失败 → 继续下方 nav_badge / 主页恢复兜底

        if nav_badge:
            # —— nav_badge 安全策略（修复：此前反复误点通讯录/头像 tab）——
            # 导航栏徽章只说明「某个 tab 有未读」（通讯录新好友/收藏等），
            # 直接点击会切走当前页面（如切到通讯录），不是聊天会话流程，
            # 而且会导致「nav click not verified」每轮重试。
            # 正确动作：双击「聊天」图标回聊天列表 → 重扫联系人红点 → 点联系人。
            # 无数字的 nav_badge（头像/图标/噪声）绝不点击。
            nav_num = nav_badge.get("unread_count")
            if self._consecutive_empty >= 2:
                # 防抖：连续多轮只有 nav_badge 时暂停回列表动作，安静等待，
                # 避免双击聊天图标造成列表反复滚动。
                self._consecutive_empty += 1
                result.update(found=False, clicked=False, kind="nav_badge",
                              entered_conversation=False,
                              reason="nav badge only (recovery paused)",
                              unread_count=nav_num)
                self._debug_log(
                    f"nav badge only, recovery paused unread={nav_num}")
                return result

            self._ensure_chat_list(window_handle, w, h, wm=wm)
            img2 = self._capture(window_handle)
            if img2 is not None:
                cd2 = self._scan_contact_dots(img2)
                clickable2 = [d for d in cd2 if self._dot_clickable(d)]
                if clickable2:
                    self._contact_dots = cd2
                    self._consecutive_empty = 0
                    res = self._pick_and_click_contact_dot(
                        window_handle, clickable2, img2, w, wm)
                    res["reason"] = "nav recovery -> " + str(
                        res.get("reason", "contact dot"))
                    self._debug_log(
                        f"nav recovery: clicked contact dot "
                        f"unread={res.get('unread_count')} "
                        f"reason={res.get('reason')}")
                    return res
                nb2 = self._scan_nav_badge(img2)
                if nb2 is not None:
                    self._nav_badge = nb2
                    result["nav_badge"] = nb2
                    nav_num = nb2.get("unread_count")
            self._consecutive_empty += 1
            result.update(found=True, clicked=False, kind="nav_badge",
                          entered_conversation=False,
                          reason="nav badge: back to chat list, no clickable contact",
                          unread_count=nav_num)
            self._debug_log(
                f"nav badge: back to chat list, no clickable contact "
                f"unread={nav_num}")
            return result

        # —— 完全扫不到未读：主页恢复（双击聊天图标回聊天列表）后重试 ——
        result["reason"] = "no unread badge or dot found"
        if self._consecutive_empty < 2:
            self._ensure_chat_list(window_handle, w, h, wm=wm)
            img2 = self._capture(window_handle)
            if img2 is not None:
                nb2 = self._scan_nav_badge(img2)
                cd2 = self._scan_contact_dots(img2)
                clickable_cd2 = [d for d in cd2 if self._dot_clickable(d)]
                if clickable_cd2:
                    self._contact_dots = cd2
                    result["contact_dots"] = cd2
                    self._consecutive_empty = 0
                    ranked = sorted(clickable_cd2, key=lambda d: d["center_y"])
                    target = ranked[0]
                    # 点前黑名单拦截（主页恢复分支同样不点黑名单）
                    _rname = self._resolve_contact_name(img2, target, w)
                    if _rname and self._is_contact_blacklisted(_rname):
                        self._debug_log(f"首页恢复: 黑名单命中 {_rname!r} -> 跳过")
                        result.update(
                            found=False, clicked=False,
                            kind="contact_dot_blacklisted",
                            contact=_rname,
                            reason="黑名单·点前 OCR 命中，已跳过",
                            unread_count=target.get("unread_count"))
                        return result
                    self._last_contact_y = int(target["center_y"])
                    self._consecutive_contact_clicks = 0
                    self._do_click(
                        window_handle,
                        self._click_x_for_dot(target, self._window_rect.get("width", 0)),
                        int(target["center_y"]), double=False, wm=wm)
                    result.update(
                        found=True, clicked=True, kind="contact_dot",
                        contact=target.get("label", ""),
                        entered_conversation=True, click_method="foreground_real_mouse",
                        reason="home recovery -> contact dot",
                        unread_count=target.get("unread_count"))
                    self._debug_log(
                        f"首页恢复: 重扫后找到红点行 未读数={target.get('unread_count')}")
                    return result
                if nb2:
                    # 与主分支一致：导航徽章不点击 tab 本身（避免切页/误点），
                    # 只记录状态并计入防抖计数。
                    self._nav_badge = nb2
                    result["nav_badge"] = nb2
                    result.update(
                        found=True, clicked=False, kind="nav_badge",
                        entered_conversation=False,
                        reason="nav badge after home recovery (not clicked)",
                        unread_count=nb2.get("unread_count"))
                    self._consecutive_empty += 1
                    return result
                self._debug_log("首页恢复: 重扫仍无红点")
            self._consecutive_empty += 1
        else:
            self._consecutive_empty += 1
            result["reason"] = "no unread badge or dot found (recovery skipped)"
        return result

    def _do_click(self, hwnd: int, window_x: int, window_y: int,
                  double: bool = False, wm=None) -> Optional[bool]:
        """统一点击入口：优先 SendMessageW 注入（屏内/屏外通用），真实鼠标仅兜底。

        坐标约定：
            window_x/window_y 来自 PrintWindow 整窗截图（PW_RENDERFULLCONTENT），
            是「物理设备像素」且原点含标题栏。window_manager 的 click_inject /
            click_offscreen 内部会减去边框偏移换算到客户区，无需调用方做任何
            DPI 缩放 —— 原样传入即可。

        路由策略：
            - 屏外 / 最小化态：走 click_offscreen（先解除最小化到屏外再注入，
              全程不回桌面）；
            - GUI 屏内态：走 click_inject（SendMessageW 注入，不移动窗口、不藏屏外）；
            两者都是 SendMessageW 注入，规避了真实鼠标的 DPI 缩放与窗口跳动坑。
            真实鼠标（foreground_click）仅作为注入不可用时的最后兜底。
        """
        cx, cy = int(window_x), int(window_y)
        if wm is not None:
            # 优先 SendMessageW 注入（屏内/屏外通用），真实鼠标仅最后兜底。
            # 关键：注入前 click_offscreen / click_inject 内部会先真实激活窗口
            # （AllowSetForegroundWindow + ALT 技巧 + SetForegroundWindow）再扣边框
            # 偏移换算到客户区。这两步补齐后 CEF 可稳定响应合成点击 —— 此前
            # 「注入无效」的结论，根因是缺了激活与坐标换算（现已实现），并非
            # 注入本身不可行。真实鼠标会把屏外窗口临时拉回主屏 (0,0)，用户全程
            # 可见，只在注入确实不可用时才降级。
            offscreen = False
            minimized = False
            try:
                offscreen = bool(wm.is_window_offscreen(hwnd))
                minimized = bool(wm.is_minimized(hwnd))
            except Exception:
                pass

            if offscreen or minimized:
                # 屏外 / 最小化 → click_offscreen：解除最小化到屏外再注入，全程不回桌面
                try:
                    if wm.click_offscreen(hwnd, cx, cy, double=double):
                        return True
                except Exception as e:
                    self._debug_log(f"屏外注入点击失败，转下一级: {e}")
            else:
                # 屏内 → click_inject：原地注入，不移动窗口、不藏屏外
                if hasattr(wm, "click_inject"):
                    try:
                        if wm.click_inject(hwnd, cx, cy, double=double):
                            return True
                    except Exception as e:
                        self._debug_log(f"原地注入点击失败，转下一级: {e}")

            # 兜底：真实鼠标（注入失败才走到；会把屏外窗口临时拉回主屏，用户可见）
            if hasattr(wm, "foreground_click"):
                try:
                    return wm.foreground_click(
                        cx, cy, hwnd=hwnd, double=double, repark=False)
                except Exception as e:
                    self._debug_log(f"真实鼠标点击失败，转后台点击: {e}")
        # 最终兜底：后台点击（真实屏幕坐标命中 CEF 子控件）
        return self._background_click(hwnd, cx, cy, double=double)

    # =================================================================
    # 窗口几何（单一真相源）
    # =================================================================

    def _live_rect(self, hwnd: int) -> Optional[Dict[str, int]]:
        """实时窗口矩形 —— 永远从系统取，不缓存。"""
        try:
            import win32gui
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return {
                "left": int(left), "top": int(top),
                "right": int(right), "bottom": int(bottom),
                "width": int(right - left), "height": int(bottom - top),
            }
        except Exception:
            return None

    def _border_offset(self, hwnd: int) -> Tuple[int, int]:
        """客户区相对窗口的边框偏移（用于截图坐标 → 客户区点击坐标换算）。

        截图基于整窗(GetWindowRect)，含边框/标题栏；
        PostMessage 点击用客户区坐标，需减去该偏移。
        基于实时矩形计算，与窗口是否被 park 到屏外无关（差值恒定）。
        """
        try:
            import win32gui
            wr = win32gui.GetWindowRect(hwnd)
            origin = win32gui.ClientToScreen(hwnd, (0, 0))
            border_l = int(origin[0]) - int(wr[0])
            border_t = int(origin[1]) - int(wr[1])
            return border_l, border_t
        except Exception:
            return 0, 0

    @staticmethod
    def _is_offscreen(r: Dict[str, Any]) -> bool:
        try:
            left = r.get("left", 0)
            top = r.get("top", 0)
            return left > 20000 or left < -2000 or top < -2000
        except Exception:
            return False

    # =================================================================
    # 截图（委托 ScreenCapture；无则自带兜底）
    # =================================================================

    def _capture(self, hwnd: int) -> Optional[np.ndarray]:
        """截图：优先用注入的 ScreenCapture（已处理 park/unpark + 校验）。"""
        if self._screen_capture is not None:
            try:
                cap = self._screen_capture.capture_window(hwnd=hwnd)
                if cap is not None and getattr(cap, "success", False) and cap.image is not None:
                    return cap.image
            except Exception:
                pass
        return self._capture_fallback(hwnd)

    def _capture_fallback(self, hwnd: int) -> Optional[np.ndarray]:
        """自带兜底截图（未注入 ScreenCapture 时）：PrintWindow → ImageGrab。

        优先 PrintWindow（走窗口自身绘制表面，屏外也能抓到真实画面，
        且不需要把窗口移动到桌面中央）；仅当 PrintWindow 失败且窗口在桌面
        可见时，回退到 ImageGrab 从屏幕抓取。
        """
        try:
            import win32gui
            import win32ui
            import ctypes
            from PIL import Image as PILImage

            # 仅最小化才需恢复：在屏外恢复渲染态，不闪现桌面
            if self._window_manager is not None:
                try:
                    if self._window_manager.is_minimized(hwnd):
                        self._window_manager.restore_offscreen(hwnd)
                except Exception:
                    pass

            # 1) PrintWindow（屏外安全，无需移动窗口）
            try:
                if not ctypes.windll.user32.IsIconic(hwnd):
                    wr = win32gui.GetWindowRect(hwnd)
                    w = wr[2] - wr[0]
                    h = wr[3] - wr[1]
                    if w > 0 and h > 0:
                        hwnd_dc = win32gui.GetWindowDC(hwnd)
                        save_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                        bitmap_dc = save_dc.CreateCompatibleDC()
                        bitmap = win32ui.CreateBitmap()
                        bitmap.CreateCompatibleBitmap(save_dc, w, h)
                        bitmap_dc.SelectObject(bitmap)
                        ok = ctypes.windll.user32.PrintWindow(hwnd, bitmap_dc.GetSafeHdc(), 2)
                        if not ok:
                            ok = ctypes.windll.user32.PrintWindow(hwnd, bitmap_dc.GetSafeHdc(), 1)
                        if ok:
                            bmpinfo = bitmap.GetInfo()
                            bmpstr = bitmap.GetBitmapBits(True)
                            img = PILImage.frombuffer(
                                "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                                bmpstr, "raw", "BGRX", 0, 1)
                            result = np.array(img.convert("RGB"))[:, :, ::-1].copy()
                            try:
                                bitmap_dc.DeleteDC()
                            except Exception:
                                pass
                            try:
                                save_dc.DeleteDC()
                            except Exception:
                                pass
                            win32gui.ReleaseDC(hwnd, hwnd_dc)
                            try:
                                win32gui.DeleteObject(bitmap.GetHandle())
                            except Exception:
                                pass
                            return result
                        try:
                            bitmap_dc.DeleteDC()
                        except Exception:
                            pass
                        try:
                            save_dc.DeleteDC()
                        except Exception:
                            pass
                        win32gui.ReleaseDC(hwnd, hwnd_dc)
                        try:
                            win32gui.DeleteObject(bitmap.GetHandle())
                        except Exception:
                            pass
            except Exception:
                pass

            # 2) ImageGrab 回退（仅对桌面可见窗口有效）
            from PIL import ImageGrab
            wr = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = wr[0], wr[1], wr[2], wr[3]
            if right <= left or bottom <= top:
                return None
            img = ImageGrab.grab(bbox=(left, top, right, bottom))
            if img is None:
                return None
            return np.array(img.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            return None

    # =================================================================
    # 头像/图标右上角数字徽章扫描（OCR fallback）
    # =================================================================

    def _extract_badges_by_cluster_ocr(
            self, img: np.ndarray,
            y0_ratio: float, y1_ratio: float,
            x0_ratio: float, x1_ratio: float,
            min_area: int = 300, max_area: int = 5000) -> List[dict]:
        """颜色分割出头像/图标大簇，对其右上角做 OCR 读取未读数字。

        用于微信主题下未读徽章与头像/图标融合、传统红色圆点检测失效的场景。
        """
        try:
            import re
            import cv2
        except Exception:
            return []
        h, w = img.shape[:2]
        x0 = max(0, int(w * x0_ratio))
        x1 = min(w, int(w * x1_ratio))
        y0 = max(0, int(h * y0_ratio))
        y1 = min(h, int(h * y1_ratio))
        if x1 <= x0 or y1 <= y0:
            return []
        region = img[y0:y1, x0:x1]
        rh, rw = region.shape[:2]

        bg = region[0:min(30, rh), 0:min(30, rw)].reshape(-1, 3).mean(0)
        diff = np.abs(region.astype(np.float32) - bg).max(axis=2)
        b = region[:, :, 0].astype(np.int16)
        g = region[:, :, 1].astype(np.int16)
        r = region[:, :, 2].astype(np.int16)
        colorful = (np.abs(r - g) > 25) | (np.abs(g - b) > 25) | (np.abs(b - r) > 25)
        mask = (diff > 30) & colorful
        if int(mask.sum()) < min_area:
            return []

        num, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        ocr = self._get_ocr()
        if ocr is None:
            return []

        badges: List[dict] = []
        for i in range(1, int(num)):
            sx, sy, bw, bh, area = [int(v) for v in stats[i][:5]]
            if area < min_area or area > max_area:
                continue
            if min(bw, bh) < 12:
                continue
            found_digit: Optional[int] = None
            for xr in (0.45, 0.50, 0.55):
                for yr in (0.0, 0.05):
                    cx1 = int(sx + bw * xr)
                    cx2 = int(sx + bw * 0.95)
                    cy1 = int(sy + bh * yr)
                    cy2 = int(sy + bh * 0.45)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    crop = region[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    big = cv2.resize(crop, None, fx=5, fy=5,
                                     interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    for proc in (big, binary, cv2.bitwise_not(binary)):
                        res = ocr.run(proc)
                        try:
                            texts = [ln.text.strip() for ln in res.iter_items() if ln.text]
                        except Exception:
                            texts = []
                        # T1: 数字提取放宽 —— isdigit() 全等匹配会把 "2]"、"2."
                        # 这类带噪声的识别结果丢掉；用正则提取数字 token。
                        for t in texts:
                            m = re.search(r"\d{1,3}", t)
                            if m:
                                n = int(m.group())
                                if 1 <= n <= 99:
                                    found_digit = n
                                    self._debug_log(
                                        f"[T1] 导航栏聚类OCR 命中徽章 文字={t!r} 未读数={n}")
                                    break
                        if found_digit is not None:
                            break
                    if found_digit is not None:
                        break
                if found_digit is not None:
                    break
            if found_digit is None:
                continue
            badges.append({
                "x": sx + x0, "y": sy + y0, "w": bw, "h": bh,
                "center_x": int(cents[i][0]) + x0,
                "center_y": int(cents[i][1]) + y0,
                "area": area, "unread_count": found_digit,
            })
        return badges

    # =================================================================
    # 导航栏徽章扫描
    # =================================================================

    def _scan_nav_badge(self, img: np.ndarray) -> Optional[dict]:
        """扫描左导航栏区域的未读红点/数字徽章（比例化坐标）。"""
        if img is None:
            return None
        h, w = img.shape[:2]
        # X 边界放宽到 0.12：0.075 会把聊天图标右上角徽章的右半裁掉，
        # 连通块退化成细长弧形被宽高比校验干掉，导致 nav 徽章漏检。
        nav_end_x = max(40, int(w * 0.12))
        nav_end_y = max(60, int(h * NAV_Y_END_RATIO))
        if nav_end_x <= 0 or nav_end_y <= 0:
            return None

        nav_region = img[:nav_end_y, :nav_end_x]
        # 关键：只有「聊天」tab 上的红点（y<0.12H）才算 nav_badge，
        # 否则会把「通讯录/朋友圈/我的」上的通知红点误当成「聊天未读」，
        # 进而触发切页（实际切到通讯录而非回聊天列表）。通讯录 5 红点
        # 等好友请求不应让助手点通讯录 tab。
        nav_y_max = int(h * NAV_BUDDY_Y_MAX_RATIO)
        nav_y_min = int(h * NAV_AVATAR_SKIP_RATIO)
        # T1: 头像区过滤升级 —— 固定比例 0.115H 在矮窗口上罩不住头像
        # （实测 984 高帧：过滤线 113，头像红衣 center_y=122 漏网被当徽章）。
        # 动态检测头像段底边，取两者较大值。
        avatar_bottom = self._nav_avatar_bottom(img)
        if avatar_bottom is not None:
            nav_y_min = max(nav_y_min, avatar_bottom + 6)
        # nav_badge 走宽松 max_size：高 DPI/大窗口下徽章可达 60~80px，放宽到 80
        dots = self._find_dots(nav_region, max_size=80)
        if dots:
            # 跳过顶部用户头像区域（头像常是彩色/红色，易被误认为 badge），
            # 并排除绿色「聊天」图标（浅色主题下被 green_mask 抓到，非徽章）。
            dots = [d for d in dots
                    if d.center_y >= nav_y_min and d.kind in ("red", "purple")]
            if dots:
                best = max(dots, key=lambda d: d.area)
                if best.center_y < nav_y_max:
                    box = {"x": best.x, "y": best.y, "w": best.w, "h": best.h}
                    return {
                        "x": best.x, "y": best.y,
                        "w": best.w, "h": best.h,
                        "center_x": best.center_x, "center_y": best.center_y,
                        "area": best.area, "kind": "nav_badge",
                        "unread_count": self._read_badge_number(img, box),
                    }
            # 传统红点检测只在头像区命中（已被过滤）或未命中时，
            # 继续走 cluster-OCR fallback，避免绿色/融合 badge 漏检。

        # Fallback：主题下徽章与图标融合，用颜色分割 + 右上角 OCR 读数字。
        # 导航栏图标列可能比 NAV_X_END_RATIO 稍宽，这里放宽到 0.12。
        # y 范围严格限制在「聊天」tab 区间（<0.12H）以避开通讯录及以下。
        badges = self._extract_badges_by_cluster_ocr(
            img, 0.0, NAV_BUDDY_Y_MAX_RATIO, 0.0, 0.12,
            min_area=300, max_area=5000)
        if badges:
            # T1: cluster-OCR fallback 同样要滤掉头像区假徽章
            #（实测 unread_debug 帧 (29,66) 头像被当徽章读出假数字）。
            valid = [b for b in badges if b["center_y"] >= nav_y_min]
            if not valid:
                return None
            best = min(valid, key=lambda b: b["center_y"])
            if best["center_y"] >= nav_y_max:
                return None
            best["kind"] = "nav_badge"
            return best
        return None

    # =================================================================
    # 红点徽章数字识别（未读数量权威来源）
    # =================================================================

    def _get_ocr(self):
        """懒加载 OCR 引擎（按路径加载，绕开 local_vision/__init__ 脆弱链）。"""
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            import importlib.util
            import os
            here = os.path.dirname(os.path.abspath(__file__))
            engine_path = os.path.abspath(
                os.path.join(here, "..", "local_vision", "ocr_engine.py"))
            spec = importlib.util.spec_from_file_location(
                "rpa._ocrengine", engine_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            OCREngine = getattr(mod, "OCREngine", None)
            if OCREngine is None:
                self._ocr_engine = None
                return None
            self._ocr_engine = OCREngine()
            if not self._ocr_engine.is_ready():
                self._ocr_engine.initialize()
        except Exception:
            self._ocr_engine = None
        return self._ocr_engine

    def _read_badge_number(self, img: np.ndarray, box: dict) -> Optional[int]:
        """读取红点徽章内的未读数字（多策略 OCR）。

        红点徽章是「红底白字/白底红字」两种形态都可能出现，且 13px 级小徽章
        里的数字只有 3~4px 高，单一阈值 + 固定 5 倍放大经常读不出来（实测
        1750 宽帧里 (223,925) 的 13x11 徽章就曾返回 None）。这里改用多路
        预处理（原灰度 / 阈值反相 / Otsu / 直方图均衡）x 多倍率（6x/8x），
        只要有一路读出 1~99 的数字即返回；大于 99 一律按 99（微信上限）。

        T1 优化（2026-09-04）：
          1. 每路预处理先走 rec-only 快速通道（use_det=False）——徽章 crop
             本身就是单数字区域，跳过检测模型直接识别。小图上 det 经常框
             不准导致读不出，是「nav未读数=None」的主因之一。
          2. 新增 CLAHE 对比度增强变体（低对比度徽章）。
          3. 命中日志：[t1] badge OCR hit variant=xxx mode=xxx n=xx。
        """
        try:
            import re
            import cv2
            x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
            pad = max(6, int(w * 0.5))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr = self._get_ocr()
            if ocr is None:
                return None

            # 多路预处理候选（提速：14:33 真机日志统计，6 变体中只有
            # gray/inv140 实际命中过，otsu/eq_otsu_inv/clahe_otsu_inv 从未
            # 命中但每次探测都白跑 ~1s。保留命中过的 gray/inv140 + 兜底
            # otsu_inv 共 3 个，单次探测耗时约减半）。
            variants = [
                ("gray", gray, 6),
                ("inv140", cv2.threshold(
                    gray, 140, 255, cv2.THRESH_BINARY_INV)[1], 6),
                ("otsu_inv", cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 8),
            ]

            def _digits_from(raw: str):
                """从 OCR 原始文本提取 1~99 的未读数（>99 按 99）。"""
                for tok in re.findall(r"\d+", raw):
                    n = int(tok)
                    if 1 <= n <= 99:
                        return n
                    if n > 99:
                        return 99
                return None

            # 变体名 → 中文说明（仅用于日志可读性）
            _variant_cn = {
                "gray": "灰度", "inv140": "反色140", "otsu_inv": "大津反色",
                "otsu": "大津", "eq_otsu_inv": "均衡反色", "clahe_otsu_inv": "自适应反色",
            }
            REC_ONLY_MIN_SCORE = 0.70
            for _name, src, scale in variants:
                big = cv2.resize(src, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
                # —— T1: rec-only 快速通道（跳过 det，小徽章首选）——
                # 防护：rec-only 无检测框约束，在衣服纹理等噪声上更容易
                # 读出假数字，必须卡置信度门槛（det+rec 路径保持原语义）。
                try:
                    res = ocr.run(big, use_det=False, use_cls=False)
                    raw = ""
                    top_score = 0.0
                    try:
                        pairs = [(str(t).strip(), float(s)) for t, s in
                                 zip(getattr(res, "texts", None) or [],
                                     getattr(res, "scores", None) or [])
                                 if str(t).strip()]
                    except Exception:
                        pairs = []
                    if not pairs:
                        try:
                            pairs = [(ln.text.strip(), float(ln.score))
                                     for ln in res.iter_items()
                                     if getattr(ln, "text", "") and ln.text.strip()]
                        except Exception:
                            pairs = []
                    raw = "".join(t for t, _s in pairs)
                    if pairs:
                        top_score = max(s for _t, s in pairs)
                    n = _digits_from(raw)
                    if n is not None and top_score >= REC_ONLY_MIN_SCORE:
                        self._debug_log(
                            f"[T1] 徽章OCR命中 变体={_variant_cn.get(_name, _name)} "
                            f"模式=纯识别 未读数={n} 置信度={top_score:.2f}")
                        return n
                except Exception:
                    pass
                # —— 原路径：det + rec ——
                try:
                    res = ocr.run(big)
                except Exception:
                    continue
                raw = ""
                try:
                    raw = "".join(str(t) for t in (getattr(res, "texts", None) or [])
                                  if str(t).strip())
                except Exception:
                    raw = ""
                if not raw:
                    try:
                        raw = "".join(ln.text.strip() for ln in res.iter_items()
                                      if getattr(ln, "text", "") and ln.text.strip())
                    except Exception:
                        raw = ""
                n = _digits_from(raw)
                if n is not None:
                    self._debug_log(
                        f"[T1] 徽章OCR命中 变体={_variant_cn.get(_name, _name)} "
                        f"模式=检测识别 未读数={n}")
                    return n
            self._debug_log("[T1] 徽章OCR 所有变体均未读出数字")
            return None
        except Exception:
            return None

    def _nav_avatar_bottom(self, img: np.ndarray) -> Optional[int]:
        """动态检测左导航头像段的底边 y（不依赖固定比例）。

        教训：NAV_AVATAR_SKIP_RATIO=0.115 在 984 高的窗口上过滤线 y=113，
        罩不住头像红衣区（center_y=122），头像被误检为 nav_badge ——
        又一次「固定比例跨分辨率失效」。头像段是导航栏顶部第一个大色块，
        用行像素统计直接找它的底边，跨窗口尺寸稳健。
        复用 _nav_chat_icon_y 的段检测思路；失败返回 None（调用方回退比例线）。
        """
        try:
            h, w = img.shape[:2]
            nav_w = max(40, int(w * 0.045))
            nav = img[:, 0:nav_w]
            bg = np.median(nav[0:30], axis=(0, 1))
            diff = np.abs(nav.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
            row_cnt = (diff > 30).sum(axis=1)
            baseline = int(np.median(row_cnt[:60]))
            threshold = max(20, int(baseline * 1.6))
            in_seg = False
            s = 0
            for y in range(h):
                if row_cnt[y] > threshold and not in_seg:
                    in_seg = True
                    s = y
                elif row_cnt[y] <= threshold and in_seg:
                    if y - s >= 14:
                        return y  # 头像段底边
                    in_seg = False
            if in_seg and h - s >= 14:
                return h
        except Exception:
            pass
        return None

    def _nav_chat_icon_y(self, img: np.ndarray) -> Optional[int]:
        """动态定位「聊天」图标 y 中心（不依赖窗口尺寸/微信版本）。

        微信 4.0 左导航栏从上到下：头像、聊天、通讯录、朋友圈……头像段
        是最稳定、色块最大的段（cnt 显著高于其它图标）。聊天图标紧邻其下，
        间距实测为 0.070H（1750 宽窗口 92px、1072 高 76px、712 高 50px）。
        所以：检测头像段中心 + 0.070H 偏移 = 聊天图标 y。
        """
        try:
            h, w = img.shape[:2]
            nav_w = max(40, int(w * 0.045))
            nav = img[:, 0:nav_w]
            bg = np.median(nav[0:30], axis=(0, 1))
            diff = np.abs(nav.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
            row_cnt = (diff > 30).sum(axis=1)
            # 导航栏左边缘有恒定非背景列（实测 1750 宽基线=12），阈值须高于基线
            baseline = int(np.median(row_cnt[:60]))
            threshold = max(20, int(baseline * 1.6))
            segs: list[tuple[int, int]] = []
            in_seg = False
            s = 0
            for y in range(h):
                if row_cnt[y] > threshold and not in_seg:
                    in_seg = True
                    s = y
                elif row_cnt[y] <= threshold and in_seg:
                    in_seg = False
                    if y - s >= 14:
                        segs.append((s, y))
            if in_seg and h - s >= 14:
                segs.append((s, h))
            if not segs:
                return None
            # 头像 = 第 1 段（顶部第一个大色块）
            avatar_center = (segs[0][0] + segs[0][1]) // 2
            chat_y = avatar_center + int(h * 0.070)
            # 合理性校验：聊天图标只可能落在 0.08H~0.18H 区间
            # （头像 0.06H、聊天≈0.152H、通讯录 0.22H）。上限收紧到 0.18H，
            # 与通讯录 0.22H 留出安全间距，避免误点到通讯录/朋友圈。
            if not (0.08 * h <= chat_y <= 0.18 * h):
                return None
            return chat_y
        except Exception:
            pass
        return None

    def _ensure_chat_list(self, hwnd: int, w: int, h: int, wm=None) -> bool:
        """主页恢复：单击左导航「聊天」图标，把微信拉回聊天列表。

        当微信被切到其它页面且未读徽章与列表红点都扫不到时，主动点击
        聊天图标回聊天列表，避免助手因进不去聊天列表而永久失能。

        坐标用窗口比例估算（跨窗口尺寸稳健），并在返回前对截图做轻量
        验证（看 contact_dots 中心 y 是否合理分布）：如果仍不在聊天列表
        （红点全在底部或异常），返回 False 让调用方重试。

        —— 历史教训：坐标必须实测校准。微信 4.0（1750x1313 实测定标）：
          头像 y=78~138、聊天图标 y=186~213（中心≈200，0.152H）、
          通讯录 y=270~300（中心≈285）、朋友圈 y=351~384。
          早期用 h*0.085=111 会点中头像；h*0.067=87 也点中头像。
          只有 h*0.152=200 才命中聊天图标。
        """
        x = int(min(34, max(26, w * 0.019)))
        # 优先动态定位聊天图标（跨尺寸/版本稳健）；失败回退比例估算。
        y = None
        try:
            img0 = self._capture(hwnd)
            if img0 is not None:
                y = self._nav_chat_icon_y(img0)
        except Exception:
            y = None
        if y is None:
            y = int(max(100, h * 0.152))
        self._debug_log(f"确保聊天列表 点击=({x},{y}) 窗口尺寸={w}x{h}")
        self._do_click(hwnd, x, y, double=False, wm=wm)
        time.sleep(0.6)
        # 轻量验证：截图后看 contact_dots 中心 y 是否落在聊天列表合理范围
        try:
            img = self._capture(hwnd)
            if img is not None:
                cds = self._scan_contact_dots(img)
                if cds:
                    ys = [c["center_y"] for c in cds]
                    # 聊天列表中红点可分布在 0.10~0.70H 区间。
                    # 下界必须是 0.10H（搜索框底边略下方），不能再高：
                    # 未读置顶后红点就落在列表第 1~2 行（实测 1405x1027 下
                    # 搜索框底边=104、第一行 y=113 ≈ 0.11H）。曾用 0.18H 作下界，
                    # 会把置顶未读整片误判为"不在聊天列表" → 每轮放弃置顶 →
                    # 助手永不点击未读、卡死在"读取当前已打开会话"的循环。
                    # 上界 0.70H：再往下多为列表底部/截断项，不足以证明在聊天列表。
                    hi, hh = img.shape[1], img.shape[0]
                    in_chat = any(0.10 * hh <= y < 0.70 * hh for y in ys)
                    if not in_chat:
                        self._debug_log(
                            f"ensure_chat_list 校验未通过: 红点 y="
                            f"{[int(v) for v in ys]} 不在 "
                            f"[{int(0.10 * hh)},{int(0.70 * hh)}) 内")
                    return in_chat
        except Exception:
            pass
        # 扫不到红点（无法判定）时放行并记录告警，不阻塞本轮。
        # 曾改为悲观 return False，结果只要红点检测不到就永久放弃置顶、
        # 助手彻底失能。权衡：点错最多本轮白跑，放弃则是 100% 不工作，
        # 因此无法判定时选择放行。
        self._debug_log("ensure_chat_list 校验：未扫到红点，无法判定 -> 放行本轮")
        return True

    # =================================================================
    # 置顶策略：双击聊天图标 + 坐标点顶行
    # =================================================================

    def _nav_green_icon_y(self, img: np.ndarray) -> Optional[int]:
        """定位左导航「绿色激活态图标」y 中心（聊天 tab 激活态为品牌绿）。

        绿色主导判据（G 显著高于 R/B），在导航列 x<0.045W 内取最大连续段
        中心。返回 None = 未检出（深色主题/检测失败），调用方降级其它信号。
        """
        try:
            h, w = img.shape[:2]
            nav_w = max(40, int(w * 0.045))
            nav = img[:, 0:nav_w].astype(np.float32)
            b, g, r = nav[:, :, 0], nav[:, :, 1], nav[:, :, 2]
            green_mask = (g > 110) & (g > r * 1.30) & (g > b * 1.30)
            row_cnt = green_mask.sum(axis=1)
            ys = np.where(row_cnt >= 6)[0]
            if ys.size == 0:
                return None
            segs: list[tuple[int, int]] = []
            s = p = int(ys[0])
            for y in ys[1:]:
                y = int(y)
                if y - p <= 4:
                    p = y
                else:
                    segs.append((s, p))
                    s = p = y
            segs.append((s, p))
            a, b2 = max(segs, key=lambda sg: sg[1] - sg[0])
            return (a + b2) // 2
        except Exception:
            return None

    def _locate_pin_y(self, hwnd: int, w: int, h: int) -> Optional[int]:
        """三信号融合定位聊天图标 y（中位数投票），供双击置顶使用。

        信号：①绿色激活图标 ②头像段+0.070H 推算（_nav_chat_icon_y）
        ③固定比例 0.152H。任一单信号失准（横幅/搜索框混入 seg[0]、
        绿检测抖动）都会被另外两票拉回。
        """
        img = None
        try:
            img = self._capture(hwnd)
        except Exception:
            img = None
        votes: list[tuple[str, int]] = []
        if img is not None:
            gy = self._nav_green_icon_y(img)
            if gy is not None:
                votes.append(("绿点", int(gy)))
            dy = self._nav_chat_icon_y(img)
            if dy is not None:
                votes.append(("头像段", int(dy)))
        votes.append(("比例", int(max(100, h * 0.152))))
        ys = sorted(v[1] for v in votes)
        y = ys[len(ys) // 2]
        self._debug_log(f"pin 定位融合: {votes} -> y={y} (size={w}x{h})")
        return y

    def _pin_unread_to_top(self, hwnd: int, w: int, h: int, wm) -> bool:
        """双击左导航「聊天」图标，让微信把未读会话置顶。

        双击带未读数量的聊天图标后，微信会把未读会话排到列表最前，
        这样即使红点颜色检测漏了（如方舟），点最顶行也能命中未读。

        三层防线防「双击落标题栏 → 窗口最大化」真机事故：
        1) 事前：三信号融合定位（中位数投票）+ 标题栏禁区（y<45px 拒绝）
           + 点击前窗口原点复核（被挪动则重定位，持续漂移则放弃本轮）；
        2) 事后自愈：双击前记录 IsZoomed，双击后若"原本没最大化却被最大化"
           = 双击落到了标题栏，立即 SW_RESTORE 还原 + 日志；
        3) 坐标换算走 _do_click 既有路由（foreground_click/注入），不变。

        坐标用 _locate_pin_y 融合定位（跨尺寸/版本稳健），不再单信任何一个
        动态检测。
        """
        x = int(min(34, max(26, w * 0.019)))
        try:
            import win32gui as _wg
            was_zoomed = bool(_wg.IsZoomed(hwnd))
        except Exception:
            was_zoomed = False

        y = self._locate_pin_y(hwnd, w, h)
        if y is None:
            return False
        # 标题栏禁区：双击窗口顶部 45px 内 = Windows 最大化/还原切换，绝不执行
        if y < 45:
            self._debug_log(
                f"[pin] 定位 y={y} 落入标题栏禁区(<45px)，放弃本轮双击")
            return False

        # 窗口原点复核：截图定位与点击之间窗口若被挪动（重绘恢复/最小化
        # 还原），按旧坐标双击会落到挪动后窗口的标题栏。重定位一次，
        # 仍漂移则放弃本轮——宁可不置顶，不乱双击。
        for attempt in range(2):
            r0 = self._live_rect(hwnd)
            time.sleep(0.05)
            r1 = self._live_rect(hwnd)
            if (r0 and r1
                    and r0["left"] == r1["left"]
                    and r0["top"] == r1["top"]):
                break
            self._debug_log(f"[pin] 窗口位置漂移(第{attempt + 1}次复核)，重新定位")
            y = self._locate_pin_y(hwnd, w, h)
            if y is None or y < 45:
                self._debug_log("[pin] 重定位失败/落入禁区，放弃本轮双击")
                return False
        else:
            self._debug_log("[pin] 窗口持续漂移，放弃本轮双击")
            return False

        self._debug_log(
            f"双击置顶执行 点击=({x},{y}) 窗口尺寸={w}x{h}")
        method = self._do_click(hwnd, x, y, double=True, wm=wm)
        time.sleep(0.6)
        # 事后自愈：原本没最大化、双击后被最大化 = 双击落到了标题栏
        try:
            import win32gui as _wg
            import win32con as _wc
            if (not was_zoomed) and _wg.IsZoomed(hwnd):
                _wg.ShowWindow(hwnd, _wc.SW_RESTORE)
                self._debug_log(
                    "[pin] 检测到双击后窗口被最大化(双击落标题栏)，已自动还原")
        except Exception:
            pass
        return method is not None

    def _detect_first_row_center(self, img: "np.ndarray", w: int, h: int):
        """动态检测列表第一行会话中心 y（分辨率无关）。

        用头像列(x≈4%~10%W)的饱和度/亮度带检测所有会话行，取第一个带中心
        作为第一行中心。不依赖任何写死比例，窗口缩放/DPI 变化都能自适应。
        搜索框会被 c>0.055h 过滤掉（它通常在 0.04~0.06H）。
        找不到时返回 None，由调用方走兜底比例。
        """
        try:
            import cv2
            ax0, ax1 = int(w * 0.04), int(w * 0.10)
            ay0, ay1 = int(h * 0.04), int(h * 0.97)
            aroi = img[ay0:ay1, ax0:ax1]
            ag = cv2.cvtColor(aroi, cv2.COLOR_BGR2GRAY)
            ahsv = cv2.cvtColor(aroi, cv2.COLOR_BGR2HSV)
            amask = (ag < 210) | (ahsv[:, :, 1] > 50)
            row_count = amask.sum(axis=1).astype(int)
            thr = max(2, int(aroi.shape[1] * 0.08))
            bands, in_b, s = [], False, 0
            for i, c in enumerate(row_count):
                if c > thr and not in_b:
                    in_b, s = True, i
                elif c <= thr and in_b:
                    in_b = False
                    bands.append((ay0 + s, ay0 + i, ay0 + (s + i) // 2))
            if in_b:
                bands.append(
                    (ay0 + s, ay0 + len(row_count), ay0 + (s + len(row_count)) // 2))
            # 合并同一行碎片（间距 < 0.03H）
            merged = []
            for b in sorted(bands, key=lambda b: b[2]):
                if merged and b[2] - merged[-1][2] < h * 0.03:
                    merged[-1] = (merged[-1][0], b[1], (merged[-1][2] + b[2]) // 2)
                else:
                    merged.append(list(b))
            # 排除搜索框（<0.055H）及异常带
            centers = [c for _, _, c in merged if c > h * 0.055]
            return centers[0] if centers else None
        except Exception:
            return None

    def _detect_search_box_bottom(self, img: "np.ndarray", w: int, h: int):
        """检测左侧会话列表顶部「搜索框」的底边 y。

        搜索框是 WeChat 列表区顶部一个稳定的灰底圆角矩形，跨版本/分辨率位置固定、
        体量大、对比强，比「薄头像行」好检测得多。找到它的底边后，点击点只要落在
        底边下方就绝不会误中搜索框——这正是此前点中搜索框的根因（旧方案按头像列
        饱和度检测，在矮窗口里搜索框底边会越过 0.055H 过滤线，或放大镜图标被误判
        成头像带，被当成第一行返回）。

        —— 关键坑（真机实测 1637x1213 像素剖面）——
        搜索框是「白底」，而它上下都是「灰底」，与我最初的假设正好相反：
            顶部工具条  y 15~70   灰底 (232,230,230)  V≈232
            搜索框      y 75~120  白底 (250,250,250)  V≈250  ← 内含"搜索"占位符
            列表背景    y 122+    灰底 (232,230,230)  V≈232
            首行头像    y 165~205 微信绿 (96,194,1)

        旧实现按「灰底带」找，命中的是上方**工具条**，返回的 sb=75 其实是搜索框的
        **顶边**！于是所有 sb+offset 的点击坐标（如 row_y=108）全部落进搜索框内部
        (75~120) —— 这正是反复误点搜索框的终极根因。
        改为：找顶部最靠上的「近白底带」= 搜索框，返回其底边。

        做法：在列表区中部取水平扫描带（x≈12%~38%W），逐行统计「近白像素占比」
        （低饱和 + V≥245，区别于灰底背景 V≈232），取最靠上的高占比连续带即搜索框。
        行区也可能有白色（如行内文字间隙），但搜索框更靠上，取 topmost 即可。
        找不到返回 None，由调用方退回头像列检测。
        """
        try:
            import cv2
            x0, x1 = int(w * 0.12), int(w * 0.38)
            y0, y1 = int(h * 0.005), int(h * 0.25)
            if y1 <= y0 or x1 <= x0:
                return None
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(int)
            s = hsv[:, :, 1].astype(int)
            # 近白底：低饱和 + 高亮度（灰底背景 V≈232 会被排除）
            white = (s < 40) & (v >= 245)
            frac = white.mean(axis=1)  # 每行近白像素占比
            thr = 0.6
            mask = frac > thr
            # 取最靠上的高占比连续带 = 搜索框
            in_run, start = False, 0
            runs = []
            for i, m in enumerate(mask):
                if m and not in_run:
                    in_run, start = True, i
                elif not m and in_run:
                    in_run = False
                    runs.append((start, i))
            if in_run:
                runs.append((start, len(mask)))
            if not runs:
                return None
            top = runs[0]
            # 搜索框高度需合理（过滤极短噪点带）
            if top[1] - top[0] < max(8, int((y1 - y0) * 0.02)):
                return None
            return y0 + top[1]
        except Exception:
            return None

    def _first_avatar_below(self, img: "np.ndarray", w: int, h: int, from_y: int):
        """从 from_y（应在搜索框底边下方）向下，在头像列找第一个会话头像带中心 y。

        头像列 x≈4%~10%W，头像带 = 「灰度偏暗 或 彩色饱和」的竖向连续像素段。
        从搜索框底边开始扫，第一个命中的带即是列表第一行会话，返回其带中心 y；
        找不到（如空列表）返回 None。
        """
        try:
            import cv2
            ax0, ax1 = int(w * 0.04), int(w * 0.10)
            ay0 = max(0, int(from_y))
            ay1 = int(h * 0.97)
            if ay1 <= ay0 or ax1 <= ax0:
                return None
            aroi = img[ay0:ay1, ax0:ax1]
            if aroi.size == 0:
                return None
            ag = cv2.cvtColor(aroi, cv2.COLOR_BGR2GRAY)
            ahsv = cv2.cvtColor(aroi, cv2.COLOR_BGR2HSV)
            amask = (ag < 210) | (ahsv[:, :, 1] > 50)
            col_n = aroi.shape[1]
            row_count = amask.sum(axis=1).astype(int)
            thr = max(2, int(col_n * 0.08))
            in_b, s = False, 0
            bands_found = []
            for i, c in enumerate(row_count):
                if c > thr and not in_b:
                    in_b, s = True, i
                elif c <= thr and in_b:
                    in_b = False
                    bands_found.append((s, i))
            if in_b:
                bands_found.append((s, len(row_count)))
            if not bands_found:
                return None
            s0, e0 = bands_found[0]
            return ay0 + (s0 + e0) // 2
        except Exception:
            return None

    def _ocr_first_contact_row(self, img: "np.ndarray", w: int, h: int, sb):
        """双击置顶后，OCR 识别列表第一个联系人，返回 (name, click_x, click_y)。

        思路（替换纯坐标推测）：在「搜索框底边下方」裁剪列表名字区做 OCR，取最靠上
        的文本框 —— 它必然落在第一行会话内，直接点它的中心即可。点击坐标来自真实
        识别到的文本，而不是按窗口比例推算，从根上杜绝误点搜索框。

        裁剪区（两条关键约束）：
          y: sb+5 → 0.97H —— 搜索框整块排除在外，它的"搜索"占位符不可能进入
             识别结果，也就不可能被点到；
          x: nav_w+50 → nav_w+270 —— 列表列内、头像右侧的名字区。列表列是固定
             像素宽（不随窗口按比例变），故用绝对偏移而非比例；上限收在列内，
             避开右侧聊天区文本污染（否则可能识别到聊天消息而点错）。实测昵称
             可延伸到 x≈340，裁太窄会把名字切碎（曾只识别出 '文件传'）。

        Returns: (name, click_x, click_y)，OCR 不可用/没识别到则返回 None。
        """
        ocr = self._get_nick_ocr()
        if ocr is None or img is None:
            return None
        nav_w = max(40, int(w * 0.045))
        x0 = max(0, nav_w + 50)
        # 右界放宽到 nav_w+360：实测昵称可延伸到 x≈340，nav_w+270 会把名字切碎
        # （曾只识别出 '文件传'），进而绕过下面的跳过集 → 空点文件传输助手。
        x1 = min(int(img.shape[1]), nav_w + 360)
        y0 = max(0, int(sb) + 5) if sb is not None else int(h * 0.09)
        y1 = min(int(img.shape[0]), int(h * 0.97))
        if x1 <= x0 or y1 <= y0:
            return None
        try:
            res = ocr.run(img[y0:y1, x0:x1])
        except Exception as exc:
            self._debug_log(f"OCR 第一行联系人失败: {exc}")
            return None
        # 收集 (text, score, box)，box 为 4 点多边形
        items = []
        try:
            if hasattr(res, "to_dict_list"):
                items = res.to_dict_list() or []
            elif hasattr(res, "iter_items"):
                for it in res.iter_items():
                    box = getattr(it, "box", None)
                    items.append({
                        "text": getattr(it, "text", ""),
                        "score": float(getattr(it, "score", 0.0)),
                        "box": box.tolist() if box is not None
                        and getattr(box, "size", 0) else [],
                    })
        except Exception:
            items = []
        cands = []
        for d in items:
            try:
                if float(d.get("score", 0.0)) < 0.5:
                    continue
                box = d.get("box") or []
                if len(box) < 3:
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                text = str(d.get("text", "")).strip()
                if not text:
                    continue
                cands.append((sum(ys) / len(ys), sum(xs) / len(xs), text))
            except Exception:
                continue
        if not cands:
            return None
        import re as _re
        cands.sort()  # 按 y 升序 → 最靠上的即第一行联系人
        for cy, cx, name in cands:
            # 垃圾名过滤：OCR 偶发把 UI artifact（emoji/sticker 的 alt 文本、
            # SVG/XML 片段）或「未读徽章数字」误读成联系人名，必须跳过回退坐标法，
            # 否则会点到错误的行（实测 name='<?xmlversion=' / name='1'）。
            if self._is_junk_ocr_name(name):
                self._debug_log(
                    f"顶行:OCR 命中垃圾名={name!r}，跳过取下一行")
                continue
            # 跳过内置/固定置顶项（如"文件传输助手"），取下一个真实未读会话行。
            # 微信里文件传输助手永远置顶且无未读，双击置顶后它常在第一行，
            # 点了也只是空操作（未读不减），必须跳过取真正未读的那一行。
            norm = _re.sub(r'[\s\W_]+', '', name.lower())
            # 抗截断：OCR 把"文件传输助手"切成"文件传"时，"文件传输" in name 会失配，
            # 故同时用 startswith/前缀命中，避免内置置顶项被点成空操作。
            if (name.startswith("文件传")
                    or "文件传输" in name
                    or "filetransfer" in norm
                    or norm.startswith("file")):
                self._debug_log(
                    f"顶行:OCR 命中内置项={name!r}，跳过取下一行")
                continue
            return name, int(x0 + cx), int(y0 + cy)
        self._debug_log(
            "顶行: OCR 识别到的均为内置项，回退坐标法")
        return None

    @staticmethod
    def _is_junk_ocr_name(name: str) -> bool:
        """OCR 顶行名字是否是无意义的垃圾（artifact/未读数字）。"""
        import re
        if not name:
            return True
        s = name.strip()
        if not s:
            return True
        # 纯数字 = 未读徽章数字，不是联系人名
        if re.fullmatch(r"\d+", s):
            return True
        low = s.lower()
        # XML/HTML/SVG artifact（emoji、sticker 的渲染残留）
        if any(k in low for k in ("<?", "xml", "version", "http", "</", "/>")):
            return True
        if "<" in s or ">" in s:
            return True
        # 代码/标点字符污染
        if any(c in s for c in "{}[]()|/\\#*=+@~^$%&"):
            return True
        return False

    def _opened_is_system_contact(self, hwnd: int, w: int, h: int) -> bool:
        """进入会话后，OCR 聊天标题栏判断是否打开了「文件传输助手/微信团队」等系统会话。

        仅用于置顶失败兜底判定：若双击置顶未生效、实际打开的是内置系统会话
        （无真实客户未读），应当降级红点扫描而非把系统会话当成功。OCR 不可用时
        返回 False（不拦截，保持原 success_by_entered 行为，避免误伤真客户）。
        """
        try:
            img = self._capture(hwnd)
            if img is None:
                return False
            ocr = self._get_nick_ocr()
            if ocr is None:
                return False
            # 微信聊天标题位于窗口顶部居中区域（工具条与聊天视图之间）
            x0 = int(w * 0.28)
            x1 = min(img.shape[1], int(w * 0.72))
            y0 = 0
            y1 = max(1, int(h * 0.06))
            if y1 <= y0 or x1 <= x0:
                return False
            res = ocr.run(img[y0:y1, x0:x1])
            texts = []
            if hasattr(res, "to_dict_list"):
                texts = [str(d.get("text", "")) for d in (res.to_dict_list() or [])]
            elif hasattr(res, "iter_items"):
                for it in res.iter_items():
                    texts.append(str(getattr(it, "text", "")))
            blob = "".join(texts)
            system_markers = ["文件传输助手", "微信团队", "订阅号", "微信支付",
                              "微信运动", "服务通知", "收藏", "文件传输",
                              "微信表情", "拍一拍"]
            return any(m in blob for m in system_markers)
        except Exception:
            return False

    def _ocr_row_name_at(self, img: "np.ndarray", w: int, h: int, row_y: int):
        """OCR 会话列表第 row_y 行的名字区，返回行名文本（失败返回 None）。

        裁剪区与 _ocr_first_contact_row 同款：头像右侧名字列（nav_w+50 ~ nav_w+360），
        行高取 ±0.035H（实测会话行高约 0.055~0.07H）。
        """
        try:
            ocr = self._get_nick_ocr()
            if ocr is None or img is None:
                return None
            ih, iw = img.shape[:2]
            nav_w = max(40, int(w * 0.045))
            x0 = max(0, nav_w + 50)
            x1 = min(iw, nav_w + 360)
            half = max(18, int(h * 0.035))
            y0 = max(0, int(row_y) - half)
            y1 = min(ih, int(row_y) + half)
            if x1 <= x0 or y1 <= y0:
                return None
            res = ocr.run(img[y0:y1, x0:x1])
            texts = []
            if hasattr(res, "to_dict_list"):
                texts = [str(d.get("text", "")).strip()
                         for d in (res.to_dict_list() or [])]
            elif hasattr(res, "iter_items"):
                texts = [str(getattr(it, "text", "")).strip()
                         for it in res.iter_items()]
            texts = [t for t in texts if t]
            if not texts:
                return None
            # 取置信度下最靠前的完整文本片段拼接（行名可能被切成两段）
            return texts[0]
        except Exception:
            return None

    def _is_builtin_row_name(self, name: str) -> bool:
        """会话行名是否为内置/系统账号（文件传输助手、公众号、订阅号等）。

        与 _ocr_first_contact_row 的内置项判断同一套语义，另加系统账号
        黑名单（与 unread_detector.SYSTEM_CONTACTS 同名单）。
        """
        if not name:
            return False
        import re as _re
        norm = _re.sub(r'[\s\W_]+', '', name.lower())
        if (name.startswith("文件传")
                or "文件传输" in name
                or "filetransfer" in norm
                or norm.startswith("file")):
            return True
        for sys_name in _SYSTEM_ROW_NAMES:
            s_norm = _re.sub(r'[\s\W_]+', '', sys_name.lower())
            if s_norm and (s_norm in norm or norm in s_norm):
                return True
        return False

    def _is_non_customer_row(self, name: str) -> bool:
        """顶行名是否「非真实客户」：系统号 / 草稿 / [图片]等预览占位。

        命中即不应盲点——这是「双击置顶未生效」的软信号之一：若列表里 T1
        没扫到未读红点行、但 nav 仍有未读，且顶行是这类非客户项，说明双击没
        把未读会话顶上来（如 22:05 日志误点 [草稿]）。用于前置判断触发重试。

        注意：只用「方括号占位 + 草稿 + 系统号」判定，绝不用裸词（转账/红包/
        文件等）——真实联系人的备注/群名可能含这些字，裸词会误伤导致漏点真客户。
        """
        if not name:
            return True
        if self._is_builtin_row_name(name):
            return True
        s = name.strip()
        # 微信系统占位行：[草稿]、[图片]、[视频]、[语音]、[表情]、[位置]… 带方括号
        if s.startswith("[") or s.startswith("【"):
            return True
        # 草稿箱（无方括号，名字即「草稿」）
        if "草稿" in s:
            return True
        return False

    def _find_unread_row(self, img: "np.ndarray", w: int, h: int, sb):
        """置顶后在会话列表找「带未读徽章的行」，返回 (数字, click_x, click_y)。

        背景：双击置顶后盲点第一行会踩坑 —— 第一行可能是『[图片]』等非联系人
        预览或置顶项，点了空跑一轮（实测 13:49 日志：点了 [图片] 进了搜索空页）。
        这里直接扫列表区的未读红点徽章：
          1. _find_dots（纯像素，毫秒级）找列表区红点候选；
          2. 每个候选用 _read_badge_number（T1 rec-only 增强通道）读数字：
             读出数字 = 确认未读行；读不出按纯红点未读=1 处理（微信单条
             未读就是无数字红点）；
          3. 点最上面的未读行文字区（y=徽章所在行，x=行文字区，避开头像）。
        没有红点返回 None，调用方回落原「OCR 第一行」逻辑。
        """
        if img is None:
            return None
        try:
            import cv2  # noqa: F401
            ih, iw = img.shape[:2]
            x_start = max(0, int(w * LIST_X_START_RATIO))
            x_end = min(iw, int(w * 0.32))
            y_top = (int(sb) + 4) if sb is not None else max(0, int(h * 0.09))
            y_bot = int(h * 0.90)
            if x_end <= x_start or y_bot <= y_top:
                return None
            region = img[y_top:y_bot, x_start:x_end]
            if region.size == 0:
                return None
            dots = self._find_dots(region)
            if not dots:
                return None
            # 排除太靠左的（导航栏列 x < 0.08W 的图标徽章不会进这个裁剪区，
            # 因为 x_start ≥ 0.10W；这里再按相对位置兜一道）
            dots = [d for d in dots if (x_start + d.center_x) >= int(w * 0.12)]
            if not dots:
                return None
            for d in sorted(dots, key=lambda dd: dd.center_y):
                box = {"x": d.x, "y": d.y, "w": d.w, "h": d.h}
                n = self._read_badge_number(region, box)
                if n is None:
                    # 数字读不出（常见于单条未读的纯红点）→ 面积/形状合理的
                    # 红点按未读=1 处理；噪声块（细长/过小）跳过
                    if d.area < 60 or d.w < 8 or d.h < 8:
                        continue
                    if max(d.w, d.h) > 3 * min(d.w, d.h):
                        continue
                    n = 1
                cy = y_top + d.center_y
                cx = int(w * 0.20)
                # 行名黑名单校验：红点行可能是「公众号/文件传输助手」等系统账号
                # （实测 unread_debug 帧的未读红点就在公众号行），点了空跑且
                # 可能误入系统会话。OCR 该行名字区，命中内置项 → 跳过取下一行。
                row_name = self._ocr_row_name_at(img, w, h, cy)
                if row_name is not None and self._is_builtin_row_name(row_name):
                    self._debug_log(
                        f"未读行: 红点行命中内置项={row_name!r}，跳过")
                    continue
                self._debug_log(
                    f"未读行: 列表未读徽章 数字={n} 名字={row_name!r} "
                    f"红点位置=({x_start + d.center_x},{cy}) -> 点行 ({cx},{cy})")
                return n, cx, int(cy)
            return None
        except Exception as exc:
            self._debug_log(f"未读行: 扫描失败 {exc}")
            return None

    def _click_top_conversation_row(self, hwnd: int, w: int, h: int, wm,
                                    nav_num=None, img=None) -> Dict[str, Any]:
        """置顶兜底：红点没扫到时直接按坐标点列表最顶会话行。

        双击置顶后未读会话必在第一行。点行内文字区（避开头像，防误开资料卡），
        进入会话后由 observe_service 的意图/黑名单判定决定是否回复，
        不会把群或无关人当客户发消息。
        """
        # 新思路（换掉旧的头像列检测）：先定位顶部稳定的灰色搜索框，找到其底边，
        # 再在底边下方检测第一个会话行/头像并点击——保证落在搜索框下方，不再误中。
        sb = None
        if img is None:
            try:
                img = self._capture(hwnd)
            except Exception:
                img = None
        row_y = None
        if img is not None:
            sb = self._detect_search_box_bottom(img, w, h)
            if sb is not None:
                # 搜索框底边下方第一个头像带 = 第一行会话
                ry = self._first_avatar_below(img, w, h, sb + 2)
                if ry is not None:
                    row_y = ry
                else:
                    # 没扫到头像（极少见/空列表）→ 退到搜索框下一行
                    row_y = sb + int(max(45, h * 0.04))
                self._debug_log(
                    f"顶行:搜索框底边={sb} -> 第一行 y={row_y} "
                    f"(size={w}x{h})")
            else:
                # 搜索框没检测到（主题/版本差异）→ 退回头像列检测
                row_y = self._detect_first_row_center(img, w, h)
                if row_y is not None:
                    self._debug_log(
                        f"顶行:搜索框未检出，退回头像列检测 y={row_y}")
        if row_y is None:
            row_y = int(h * 0.12)  # 兜底：全部检测失败
            self._debug_log(
                f"顶行:全部检测失败，使用兜底 y={row_y} size={w}x{h}")
        # 安全硬约束：最终点击点必须严格落在搜索框底边下方，
        # 杜绝任何路径下误点搜索框（此前点搜索框的根因）。
        if sb is not None and row_y is not None and row_y <= sb + 4:
            row_y = sb + int(max(45, h * 0.04))
            self._debug_log(
                f"顶行:安全约束下压到搜索框下方 y={row_y}")
        y = int(row_y)
        # 文字区 x：头像列约 0.06~0.08W，点右侧文字区避免误触头像
        x = int(w * 0.20)
        click_src = "coordinate"
        ocr_name = ""
        # —— 优先：未读行定位（T1 数字识别确认）——
        # 双击置顶后第一行未必是未读（可能是 [图片] 预览/置顶项），盲点会空跑
        # 一轮。先扫列表未读徽章行，点「带红点数字的那一行」。
        if img is not None:
            try:
                unread_hit = self._find_unread_row(img, w, h, sb)
            except Exception:
                unread_hit = None
            if unread_hit:
                n, ux, uy = unread_hit
                # 安全校验：同样必须严格在搜索框底边下方
                if sb is None or uy > sb + 4:
                    x, y = int(ux), int(uy)
                    ocr_name = f"未读行(n={n})"
                    click_src = "unread_row"
                    self._debug_log(
                        f"顶行:未读行优先 name={ocr_name!r} "
                        f"click=({x},{y}) sb={sb}")
                else:
                    self._debug_log(
                        f"顶行:未读行 ({ux},{uy}) 未越过搜索框底边 "
                        f"sb={sb}，回落第一行逻辑")
        # —— 次优：OCR 识别第一行联系人，点它的文本框中心 ——
        # 坐标来自真实识别到的文本（天然落在第一行会话内），裁剪区从搜索框底边下方
        # 开始、"搜索"占位符进不了识别结果，从根上杜绝误点搜索框。OCR 不可用时
        # 自动回落到上面的坐标法（sb + 头像带），行为不变。
        # 同时承担「双击置顶前置软信号」：T1 未找到未读行时先看顶行是不是非客户
        # 项（草稿/系统号/预览），是则判定双击未生效→放弃盲点、交外层重试。
        if click_src != "unread_row" and img is not None:
            try:
                tgt = self._ocr_first_contact_row(img, w, h, sb)
            except Exception:
                tgt = None
            # —— 前置软信号：双击置顶未生效保护 ——
            # 条件：T1 没找到未读行（列表无红点）+ nav 仍有未读 + 顶行是非客户项。
            # 三者同时满足 → 双击没把未读顶上来，放弃盲点、交外层重试，
            # 避免误点 [草稿]/系统号 空跑（实测 22:05 日志误点[草稿]）。
            if tgt and nav_num:
                _top_name = tgt[0] if isinstance(tgt, (list, tuple)) else ""
                if _top_name and self._is_non_customer_row(_top_name):
                    self._debug_log(
                        f"顶行:T1未找到未读行 且 顶行={_top_name!r} 非客户项 "
                        f"nav={nav_num} → 判定双击置顶未生效，放弃盲点交外层重试")
                    return {
                        "found": True, "clicked": False, "kind": "nav_badge",
                        "contact": _top_name,
                        "entered_conversation": False,
                        "click_method": "top_row_pin_not_effective",
                        "reason": "pin not effective: no unread row + top is non-customer",
                        "unread_count": nav_num,
                        "pin_not_effective": True,
                    }
            if tgt:
                name, ox, oy = tgt
                # 安全校验：OCR 结果也必须严格在搜索框底边下方（双保险）
                if sb is None or oy > sb + 4:
                    x, y = int(ox), int(oy)
                    ocr_name = name
                    click_src = "ocr"
                    self._debug_log(
                        f"顶行:OCR 命中第一行 name={name!r} "
                        f"click=({x},{y}) sb={sb}")
                else:
                    self._debug_log(
                        f"顶行:OCR 结果 ({ox},{oy}) 未越过搜索框底边 "
                        f"sb={sb}，回退坐标法")
        _src_cn = {"coordinate": "坐标法", "ocr": "OCR识别", "unread_row": "未读行"}
        self._debug_log(
            f"点击顶行 坐标=({x},{y}) 窗口尺寸={w}x{h} "
            f"来源={_src_cn.get(click_src, click_src)}")
        method = self._do_click(hwnd, x, y, double=False, wm=wm)
        if method is None:
            return {
                "found": True, "clicked": False, "kind": "nav_badge",
                "contact": ocr_name, "entered_conversation": False,
                "click_method": f"top_row_{click_src}",
                "reason": "top row click failed (no input method)",
                "unread_count": nav_num,
            }
        self._last_contact_y = y
        self._consecutive_contact_clicks = 0
        return {
            "found": True, "clicked": True, "kind": "contact_dot",
            "contact": ocr_name, "entered_conversation": True,
            "click_method": f"top_row_{click_src}",
            "reason": "clicked top conversation row after pin (no red dot)",
            "unread_count": nav_num,
            "click_x": x, "click_y": y,
        }

    def _nav_badge_metric(self, img: Any):
        """像素级统计导航栏未读徽章规模：返回 (红点个数, 总面积)。

        与 _scan_nav_badge 使用同一套「跳过顶部头像区」过滤，但**完全不调用 OCR**，
        纯 numpy/cv2 连通分量，毫秒级（_find_dots 注释：比逐像素循环快约 100x）。

        返回 (0, 0) 表示导航栏没有可检测的徽章；返回 None 表示画面不可用/无法判定。
        用于双击置顶后的帧间对比：徽章消失 => 未读已清零，可直接判成功，免去 OCR。
        """
        if img is None:
            return None
        try:
            h, w = img.shape[:2]
            # 关键修复：X 边界放宽到 0.12，与 _extract_badges_by_cluster_ocr 对齐。
            # 旧值 NAV_X_END_RATIO(0.075) 会把聊天图标右上角徽章的右半裁掉，
            # 连通块退化成细长弧形，被宽高比校验(ASPECT_MIN)干掉 -> 返回 (0,0)。
            nav_end_x = max(40, int(w * 0.12))
            nav_end_y = max(60, int(h * NAV_Y_END_RATIO))
            if nav_end_x <= 0 or nav_end_y <= 0:
                return None
            nav_y_min = int(h * NAV_AVATAR_SKIP_RATIO)
            # 先裁剪出真正的 nav 扫描区，再检测（避免把顶部头像区/其他列误当 nav）
            nav_region = img[nav_y_min:nav_end_y, :nav_end_x]
            if nav_region.size == 0:
                return None
            # max_size 放宽到 80：高 DPI/大窗口下徽章直径可达 60~80px，旧 40 不够。
            dots = self._find_dots(nav_region, max_size=80)
            # 排除绿色「聊天」图标（浅色主题下被 green_mask 抓到，非徽章）
            dots = [d for d in dots if d.kind in ("red", "purple")]
            if dots:
                return (len(dots), sum(int(d.area) for d in dots))

            # 兜底：严格红点检测失效时（如抗锯齿把徽章碎成细长条），
            # 用颜色分割直接统计红/紫色像素面积。OCR 路径已证明这些像素
            # 就是 nav_badge，这里只须回答"有没有"，不读数字。
            import cv2
            nr = nav_region[:, :, 2].astype(np.int16)
            ng = nav_region[:, :, 1].astype(np.int16)
            nb = nav_region[:, :, 0].astype(np.int16)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = nr.astype(np.float64) / ((ng + nb).astype(np.float64))
                red_mask = (
                    (nr >= self._r_min) & (ng <= self._g_max) & (nb <= self._b_max)
                    & ((nr - ng) >= self._rg_diff) & ((ng + nb) > 0)
                    & (ratio > self._r_ratio)
                )
                purple_mask = (
                    (nr >= 140) & (nb >= 140) & (ng <= self._g_max)
                    & (np.abs(nr - nb) <= 80)
                    & ((nr + nb) / 2.0 > ng + 30)
                )
            mask = red_mask | purple_mask
            num, labels, stats, cents = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            total_area = 0
            for i in range(1, int(num)):
                area = int(stats[i, cv2.CC_STAT_AREA])
                # 只统计合理大小的红/紫块：太大是图标/头像，太小是噪声
                if 30 <= area <= 500:
                    total_area += area
            if total_area > 0:
                return (1, total_area)
            return (0, 0)
        except Exception:
            return None

    def _verify_pin_success(self, hwnd: int, w: int, h: int, wm,
                            before_num: Any,
                            before_img: Any = None) -> Tuple[str, Any]:
        """双击置顶+点顶行后，重新扫描 nav 未读数量，判断是否真打开了未读会话。

        判定（before=双击前置顶前的 nav 未读总数，after=点击并进入会话后的总数）：
          - "success" : 多次重扫中任一时刻 after < before（未读减少），或徽章消失(None)
          - "unknown" : before 读不出，或每次截图都失败（无法判定），不强行判失败
          - "fail"    : 多次重扫后 after 仍 >= before（未读数未变化 → 双击可能未生效）

        多次重扫原因：进入会话后微信 CEF 的"聊天" tab 徽章刷新有 ~数百毫秒延迟，
        单次 sleep(0.4) 会截到过渡帧把已清的未读读成"未变"，导致误判 fail（实为成功）。
        这里间隔 0.5s 重扫 3 次，任一时刻出现"减少/消失"即判成功，消除过渡帧误判。
        """
        # —— 快速路径：像素级判断徽章是否消失（未读清零），纯 numpy，毫秒级 ——
        # 传入点击顶行前的基准帧(before_img)时，先用像素检测导航栏徽章是否消失。
        # 徽章消失是最常见的成功形态（处理完最后一个未读），可秒判成功，
        # 避免原来动辄 20~60s 的 OCR 读数字。像素同样重扫 3 次以覆盖过渡帧延迟。
        # 若 3 次重扫徽章都还在，说明未读未清零（可能只是数字变小、也可能点错行），
        # 像素无法区分数字 2 变 1，降级回原 OCR 读数字逻辑，保留全部原语义。
        if before_img is not None:
            # 仅在「基准帧里确实检测到徽章」时才启用像素快速路径。
            # 基准帧都测不到徽章（融合形态徽章 / 画面异常 / 纯色帧），
            # 说明像素对比不可靠，直接降级 OCR，避免误判成"徽章消失=成功"。
            before_metric = self._nav_badge_metric(before_img)
            # 提速(14:33 日志)：基准帧未检出徽章时，重采上限 2次×0.35s → 4次×0.5s。
            # 双击置顶的列表滚动动画常需 1~2s 才稳，之前 2 次重采全落空时像素
            # 快速路径作废、退化 OCR 三连（~10s/轮）；4 次×0.5s 覆盖 2s 动画窗口，
            # 大概率把验证拉回毫秒级像素秒判。
            if before_metric is None or before_metric[1] <= 0:
                for _ in range(4):
                    time.sleep(0.5)
                    fb = self._capture(hwnd)
                    if fb is None:
                        continue
                    m = self._nav_badge_metric(fb)
                    if m is not None and m[1] > 0:
                        before_metric = m
                        before_img = fb
                        self._debug_log(
                            "nav-pin: 重采帧检测到徽章，启用像素快速路径")
                        break
            if before_metric is not None and before_metric[1] > 0:
                pixel_scanned = False
                for _ in range(3):
                    time.sleep(0.5)
                    img3 = self._capture(hwnd)
                    if img3 is None:
                        continue
                    pixel_scanned = True
                    metric = self._nav_badge_metric(img3)
                    if metric is None:
                        continue
                    if metric[1] == 0:
                        # 徽章消失 = 未读全部清零
                        self._debug_log(
                            "nav-pin: 像素检测 徽章已消失 -> 未读清零，成功（免 OCR）")
                        return "success", 0
                    if metric[1] < before_metric[1] * 0.6:
                        # 徽章规模明显缩小（多个红点变少）也视为未读减少。
                        # 阈值取 0.6 是保守值：正常渲染波动达不到 40% 的幅度，
                        # 而真实未读减少必然带来可观测的规模变化。
                        self._debug_log(
                            f"nav-pin: 像素检测 徽章面积 {before_metric[1]}"
                            f"->{metric[1]} 明显缩小 -> 成功（免 OCR）")
                        return "success", before_num
                if pixel_scanned:
                    self._debug_log(
                        "nav-pin: 像素检测 徽章仍在 -> 降级 OCR 读数字确认")
            else:
                self._debug_log(
                    "nav-pin: 基准帧未检测到徽章，像素对比不可用 -> 降级 OCR")

        # —— 原路径：OCR 读数字对比（保留全部原语义，作为兜底）——
        best = None
        scanned = False
        for _ in range(3):
            time.sleep(0.5)
            img3 = self._capture(hwnd)
            if img3 is None:
                continue
            scanned = True
            nb2 = self._scan_nav_badge(img3)
            after = nb2.get("unread_count") if nb2 else None
            # 徽章消失 = 未读已全部清零 → 直接成功，无需继续
            if after is None:
                return "success", 0
            if before_num is not None and after < before_num:
                return "success", after
            # 记录"最乐观"读数（取最小），用于最终兜底判断
            if best is None or (after is not None and after < best):
                best = after
        # 三次截图全部失败：无法判定，返回 unknown（不触发降级重击），
        # 而非误判 fail（会触发多余的回列表重截点击）。
        if not scanned:
            return "unknown", None
        if before_num is None:
            return "unknown", best
        if best is not None and best < before_num:
            return "success", best
        return "fail", best

    # =================================================================
    # 联系人红点扫描
    # =================================================================

    def _min_unread_dot_area(self, window_w: int, window_h: int) -> int:
        """纯红点降级的面积阈值（随窗口尺寸等比缩放）。

        微信 4.0 头像红点在 1750 宽窗口约 13px/116px²；窗口缩小时红点同比例
        缩小（946 宽下约 7px/44px²）。固定 60px² 会把小窗口的真实红点误判为
        残片。按 (窗口面积 / 1750x1313) 线性缩放，下限 30 兜底、上限 60 封顶。
        """
        try:
            scale = int((int(window_w) * int(window_h)) / (1750 * 1313) * MIN_UNREAD_DOT_AREA)
        except Exception:
            scale = MIN_UNREAD_DOT_AREA
        return max(30, min(scale, MIN_UNREAD_DOT_AREA))

    def _dot_clickable(self, d: dict) -> bool:
        """判定一个已通过红块校验的点是否值得点击。

        - 读出未读数字（unread_count 非 None）→ 必点；
        - 无数字但面积达标（≥MIN_UNREAD_DOT_AREA，随窗口缩放）
          → 也可点：点击后 observe_service 会进会话 OCR，无客户消息会退出，
            误点成本低，但能覆盖「纯红点未读」这种微信常见形态。
        """
        if d.get("unread_count") is not None:
            return True
        return int(d.get("area") or 0) >= self._min_unread_dot_area(
            d.get("_win_w") or 0, d.get("_win_h") or 0)

    def _scan_contact_dots(self, img: np.ndarray) -> List[dict]:
        """扫描联系人列表区域的红点（结构锚定头像列，比例化坐标）。"""
        if img is None:
            return []
        h, w = img.shape[:2]
        x_start = max(0, int(w * LIST_X_START_RATIO))
        x_end = min(w, int(w * LIST_X_END_RATIO))
        top_skip = max(0, int(h * LIST_TOP_SKIP_RATIO))
        if x_end <= x_start or h <= top_skip:
            return []

        # 导航栏排除阈值（先定义，fallback 也要用）。
        # 聊天图标本身约 0.15~0.20 高、0.04~0.08 宽，且 badge 在图标右上角会偏右，
        # nav 限制设为 0.08W（实测微信导航图标 = x∈[10,60]）。阈值太宽会把 x=200
        # 附近的真实头像红点误排除（曾发生：方舟/华医未读全部漏掉）。
        nav_end_x = max(60, int(w * 0.08))
        nav_end_y = max(90, int(h * 0.22))

        contact_region = img[top_skip:h, x_start:x_end]
        dots = self._find_dots(contact_region)
        if not dots:
            # Fallback：主题下徽章与头像/图标融合，传统红色圆点检测失效，
            # 用颜色分割 + 右上角 OCR 读数字。
            badges = self._extract_badges_by_cluster_ocr(
                img, LIST_TOP_SKIP_RATIO, 0.90, 0.04, 0.32,
                min_area=300, max_area=5000)
            if badges:
                results: List[dict] = []
                for b in sorted(badges, key=lambda d: d["center_y"]):
                    # 排除导航栏整列（微信左导航贯穿全高，仅排除上部会让
                    # 通讯录/发现/我的 图标徽章漏进联系人列表导致误点击）
                    if b["center_x"] < nav_end_x:
                        continue
                    # 读不出数字且面积足够 → 按纯红点未读=1 处理
                    if b.get("unread_count") is None and int(b.get("area") or 0) >= self._min_unread_dot_area(w, h):
                        b["unread_count"] = 1
                    b["_win_w"] = w
                    b["_win_h"] = h
                    b.update(kind="contact_dot", label=f"contact_{b['center_y']}")
                    results.append(b)
                self._update_avatar_anchor(results)
                return sorted(results, key=lambda d: d["center_y"])
            return []

        # 裁剪图相对坐标 -> 绝对坐标
        abs_dots = [
            RedDotResult(d.x + x_start, d.y + top_skip, d.w, d.h, d.area,
                         d.center_x + x_start, d.center_y + top_skip)
            for d in dots
        ]

        # 排除导航栏整列（微信左导航栏贯穿整个窗口高度：聊天/通讯录/发现/我的
        # 图标纵向排布，通讯录等图标上的徽章 y 可远大于 0.22h）。仅排除上部
        # 会让这些徽章漏进联系人列表，导致把导航当未读点击（曾误点通讯录）。
        # 列表项头像从 x≈0.14W 开始，与 nav_end_x（0.12W）不重叠，用 x 阈值安全。
        abs_dots = [
            d for d in abs_dots
            if d.center_x >= nav_end_x
        ]

        # —— 结构锚定：推断头像列 x（center_x 出现次数最多的位置）——
        # 真实未读红点都落在头像右上角、纵向等距排列，center_x 高度一致；
        # 偏离该列的红色块（搜索框 X / 其它 UI 红元素）应剔除。
        from collections import Counter
        buckets = Counter((d.center_x // 12) * 12 for d in abs_dots)
        avatar_col = None
        avatar_confident = False
        if buckets:
            best_bin, best_cnt = buckets.most_common(1)[0]
            avatar_col = best_bin + 6
            # 仅当存在「成列」的红点（≥2 个聚集在同一 x）时才算有信心锚定
            avatar_confident = best_cnt >= 2

        # 面积最大的红点（真实未读徽章通常最大）豁免头像列锚定剔除，
        # 避免唯一的大红点（如列表第一项头像上的未读徽章）被误剔除。
        max_area = max((d.area for d in abs_dots), default=0)
        anchor_exempt = {id(d) for d in abs_dots if d.area >= max(max_area * 0.6, 60)}

        # 本帧锚 = 成列锚 或 历史平滑锚（单点帧也能量化「正常头像列位置」）
        frame_anchor = avatar_col if avatar_confident else self._avatar_col_smooth

        results: List[dict] = []
        for d in abs_dots:
            # 锚定过滤：偏离头像列的红块剔除（面积最大的红点豁免）。
            # 成列锚容差小（±18px）；历史平滑锚跨帧，容差放宽到 ±36px，
            # 避免把少数真实红点当噪声误杀。
            if id(d) not in anchor_exempt and frame_anchor is not None:
                tol = (self._avatar_col_tolerance if avatar_confident
                       else self._avatar_col_tolerance * 2)
                if abs(d.center_x - frame_anchor) > tol:
                    continue
            if d.w < self._dot_size_min or d.h < self._dot_size_min:
                continue
            # 同 _is_red_blob：放行扁长"99+"徽章，但拒绝 60+px 头像/服务图标。
            if d.w > self._dot_size_max * 1.2 or d.h > self._dot_size_max * 1.2:
                continue
            box = {"x": d.x, "y": d.y, "w": d.w, "h": d.h}
            unread = self._read_badge_number(img, box)
            if unread is None and d.area >= self._min_unread_dot_area(w, h):
                # 纯红点未读（微信无数字形态）：面积足够即视为 ≥1 条未读
                unread = 1
            results.append({
                "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                "center_x": d.center_x, "center_y": d.center_y,
                "area": d.area, "kind": "contact_dot",
                "label": f"contact_{d.center_y}",
                "unread_count": unread,
                "_anchor_x": frame_anchor or avatar_col or d.center_x,
                "_win_w": w, "_win_h": h,
            })
        self._update_avatar_anchor(results)
        return sorted(results, key=lambda d: d["center_y"])

    def _update_avatar_anchor(self, results: List[dict]) -> None:
        """用本帧红点的 center_x 中位数更新跨帧头像列平滑锚（EMA）。"""
        xs = [int(r.get("center_x") or 0) for r in results if r.get("center_x")]
        if not xs:
            return
        xs.sort()
        median_x = xs[len(xs) // 2]
        if self._avatar_col_smooth is None:
            self._avatar_col_smooth = median_x
        else:
            self._avatar_col_smooth = int(self._avatar_col_smooth * 0.7 + median_x * 0.3)

    def _click_x_for_dot(self, dot: dict, window_w: int) -> int:
        """点击 x：落在会话行内的昵称/消息文本区，确保打开的是该联系人的会话。

        微信 4.0 未读红点钉在头像右上角，红点 center_x ≈ 头像中心；昵称区在
        头像右侧（头像宽约 36~40px）。旧实现 cx-30 会落到头像左侧的导航栏/
        空白边缘，宽窗口下经常点不中行 —— 这里统一改点「头像右侧 +42px」，
        保证落在昵称/消息文本区。

        对 cluster-OCR 识别出的头像/图标融合 badge，其 center_x 是头像中心而非
        红点，同样向右偏移。
        """
        cx = int(dot.get("center_x", 0))
        anchor = int(dot.get("_anchor_x") or 0) or cx
        # cluster OCR badge（area 较大）落在头像/图标上，点击其右侧文本区
        if dot.get("area", 0) > 1000:
            return min(int(window_w * 0.30), max(120, cx + 60))
        base = max(anchor, cx)
        click_x = base + 42
        # 限制在列表区 [0.10W, 0.38W] 内，避免点进右侧聊天面板
        lo = max(90, int(window_w * 0.10))
        hi = max(lo + 40, int(window_w * 0.38))
        return min(max(click_x, lo), hi)

    # =================================================================
    # 核心红点检测（像素聚类）
    # =================================================================

    def _find_dots(self, img: np.ndarray, max_size: int = None) -> List[RedDotResult]:
        """检测图中所有红色连通块（向量化 + 连通分量，比逐像素循环快约 100x）。

        返回的每块均经过：尺寸 / 宽高比 / 实心度 / 平均颜色 四重校验，
        确保是「未读红点」而非灰度文字里夹杂的零星红色像素。
        """
        if img is None or img.size == 0:
            return []
        try:
            import cv2
        except Exception:
            return self._find_dots_fallback(img)

        # BGR 顺序：img[...,2]=R, [...,1]=G, [...,0]=B
        r = img[:, :, 2].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 0].astype(np.int16)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_mask = r.astype(np.float64) / ((g + b).astype(np.float64))
            red_mask = (
                (r >= self._r_min) & (g <= self._g_max) & (b <= self._b_max)
                & ((r - g) >= self._rg_diff) & ((g + b) > 0)
                & (ratio_mask > self._r_ratio)
            )
            # 部分主题/模式下微信未读徽章呈紫色/洋红色（B 与 R 都高、G 低）
            purple_mask = (
                (r >= 140) & (b >= 140) & (g <= self._g_max)
                & (np.abs(r - b) <= 80)
                & ((r + b) / 2.0 > g + 30)
            )
            # 微信浅色主题下未读徽章也可能是绿色（如群聊、导航栏"聊天"图标）
            green_mask = (
                (g >= GREEN["g_min"]) & (r <= GREEN["r_max"]) & (b <= GREEN["b_max"])
                & ((g - r) >= GREEN["gb_diff"]) & ((g - b) >= GREEN["gb_diff"])
                & ((r + b) > 0)
                & (g.astype(np.float64) / ((r + b).astype(np.float64)) > GREEN["g_ratio"])
            )
        mask = red_mask | purple_mask | green_mask
        if int(mask.sum()) < self._min_pixels:
            return []

        num, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        results: List[RedDotResult] = []
        h, w = img.shape[:2]
        for i in range(1, int(num)):
            x, y, bw, bh, area = [int(v) for v in stats[i][:5]]
            if area < self._min_pixels:
                continue
            cx = int(cents[i][0]); cy = int(cents[i][1])
            # 该连通块平均颜色（二次校验，排除灰度文字误染的零星红像素）
            comp = (labels == i)
            mr = int(r[comp].mean()); mg = int(g[comp].mean()); mb = int(b[comp].mean())
            if not self._is_red_blob(mr, mg, mb, area, bw, bh, max_size=max_size):
                continue
            results.append(RedDotResult(
                x=x, y=y, w=bw, h=bh, area=area, center_x=cx, center_y=cy,
                kind=self._blob_kind(mr, mg, mb) or "red"))
        return results

    def _blob_kind(self, mr: int, mg: int, mb: int) -> Optional[str]:
        """按连通块平均颜色归类：red / purple / green，无法归类返回 None。

        与 _is_red_blob 的颜色判定保持一致，但额外返回类别，
        供 _nav_badge_metric / _scan_nav_badge 过滤掉绿色「聊天」图标
        （浅色主题下聊天 tab 图标是绿色，会被 green_mask 抓到误当徽章）。
        """
        is_red = (mr >= self._mean_r_min
                  and mg <= self._mean_gb_max and mb <= self._mean_gb_max
                  and (mr - max(mg, mb)) >= self._mean_rg_diff
                  and abs(int(mg) - int(mb)) <= self._mean_gb_diff_max)
        if is_red:
            return "red"
        is_purple = (mr >= 130 and mb >= 130 and mg <= self._mean_gb_max
                     and abs(mr - mb) <= 90)
        if is_purple:
            return "purple"
        is_green = (mg >= 160 and mr <= self._mean_gb_max and mb <= self._mean_gb_max
                    and (mg - max(mr, mb)) >= 40)
        if is_green:
            return "green"
        return None

    def _is_red_blob(self, mr: int, mg: int, mb: int, area: int,
                     bw: int, bh: int, max_size: int = None) -> bool:
        """对红色连通块做 尺寸/比例/实心度/平均颜色 校验。"""
        if bw < self._dot_size_min or bh < self._dot_size_min:
            return False
        # 真实徽章直径不超过 DOT_SIZE_MAX；放宽到 1.2 倍只容纳"99+"等扁长徽章，
        # 排除 60+px 的大头像/服务图标（如实测微信支付的 63×63 绿图标）被误当红点。
        # max_size 由调用方传入：contact_dots 走严格 24（拒绝 35x31 彩色头像），
        # nav_badge 走宽松 36（保留聊天 tab 上 39x38 的大数字徽章）。
        cap = max_size if max_size is not None else self._dot_size_max
        if bw > cap * 1.2 or bh > cap * 1.2:
            return False
        ratio = bw / max(bh, 1)
        if ratio < self._aspect_min or ratio > self._aspect_max:
            return False
        solidity = area / max(bw * bh, 1)
        if solidity < self._solidity_min:
            return False
        if self._blob_kind(mr, mg, mb) is None:
            return False
        return True

    def _find_dots_fallback(self, img: np.ndarray) -> List[RedDotResult]:
        """无 cv2 时的慢速回退（保持老逻辑可用）。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        red_mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if self._is_red(img[y, x]):
                    red_mask[y, x] = True
        if int(red_mask.sum()) < self._min_pixels:
            return []
        try:
            import cv2
            num, labels, stats, cents = cv2.connectedComponentsWithStats(
                red_mask.astype(np.uint8), connectivity=8)
        except Exception:
            return []
        results = []
        for i in range(1, int(num)):
            x, y, bw, bh, area = [int(v) for v in stats[i][:5]]
            if area < self._min_pixels:
                continue
            if not self._is_red_blob(255, 120, 120, area, bw, bh):
                continue
            results.append(RedDotResult(
                x=x, y=y, w=bw, h=bh, area=area,
                center_x=int(cents[i][0]), center_y=int(cents[i][1])))
        return results

    def _is_red(self, pixel: np.ndarray) -> bool:
        """像素级红色/紫色/绿色徽章判定。"""
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        is_red = (r >= self._r_min and g <= self._g_max and b <= self._b_max
                  and r - g >= self._rg_diff)
        if is_red:
            gb = g + b
            if gb == 0:
                return False
            if r / float(gb) <= self._r_ratio:
                return False
            return True
        # 紫色/洋红徽章（部分主题下）
        if r >= 140 and b >= 140 and g <= self._g_max:
            if abs(r - b) <= 80 and (r + b) / 2.0 > g + 30:
                return True
        # 绿色徽章（浅色主题下群聊/导航栏等）
        if g >= GREEN["g_min"] and r <= GREEN["r_max"] and b <= GREEN["b_max"]:
            if g - r >= GREEN["gb_diff"] and g - b >= GREEN["gb_diff"]:
                rb = r + b
                if rb == 0:
                    return False
                if g / float(rb) > GREEN["g_ratio"]:
                    return True
        return False

    def _is_light(self, pixel: np.ndarray) -> bool:
        return (int(pixel[0]) >= self._light_brightness_min and
                int(pixel[1]) >= self._light_brightness_min and
                int(pixel[2]) >= self._light_brightness_min)

    def _count_light_inside(self, img: np.ndarray,
                            x: int, y: int, w: int, h: int) -> int:
        count = 0
        for dy in range(h):
            for dx in range(w):
                px = x + dx
                py = y + dy
                if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                    if self._is_light(img[py, px]):
                        count += 1
        return count

    def _cluster(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """网格聚类。"""
        if not mask.any():
            return []
        h, w = mask.shape
        grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        cell_size = self._cluster_radius * 2

        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            cell = (int(x // cell_size), int(y // cell_size))
            grid.setdefault(cell, []).append((int(x), int(y)))

        clusters: List[List[Tuple[int, int]]] = []
        for points in grid.values():
            placed = False
            for cluster in clusters:
                ref = cluster[-1]
                ref_cell = (int(ref[0] // cell_size), int(ref[1] // cell_size))
                if abs(int(points[0][0] // cell_size) - ref_cell[0]) <= 1 and \
                   abs(int(points[0][1] // cell_size) - ref_cell[1]) <= 1:
                    cluster.extend(points)
                    placed = True
                    break
            if not placed:
                clusters.append(points)

        return [c for c in clusters if len(c) >= self._min_pixels]

    # =================================================================
    # 后台点击（唯一执行路径，不移动真实鼠标）
    # =================================================================

    def _background_click(self, hwnd: int, window_x: int, window_y: int,
                          double: bool = False) -> Optional[bool]:
        """通过后台点击（不移动真实鼠标、不抢焦点）打开会话。

        入参为「窗口坐标系」坐标（与截图像素同源，调用方**无需**再扣边框）。
        内部统一换算：
          - 客户区坐标 = window - border（发给顶层 HWND 用）
          - 屏幕坐标   = 窗口原点 + window（WindowFromPoint 命中真实子控件用）
        顺序：① 屏幕坐标命中真实子控件(CEF 渲染窗) → ② 回退顶层 SendMessage → ③ PostMessage。
        """
        try:
            import win32gui as _g
            wr = _g.GetWindowRect(hwnd)
            origin = _g.ClientToScreen(hwnd, (0, 0))
            border_l = origin[0] - wr[0]
            border_t = origin[1] - wr[1]
            client_x = int(window_x) - border_l
            client_y = int(window_y) - border_t
            screen_x = wr[0] + int(window_x)
            screen_y = wr[1] + int(window_y)
            from .background_clicker import background_click_screen, background_click
            ok = background_click_screen(screen_x, screen_y, double=double)
            if not ok:
                ok = background_click(hwnd, client_x, client_y, double=double)
            self._debug_log(
                f"后台点击 {'成功' if ok else '失败'} 句柄={hwnd:#x} "
                f"窗口坐标=({window_x},{window_y}) 客户区=({client_x},{client_y}) "
                f"屏幕=({screen_x},{screen_y}) 双击={double}")
            time.sleep(CLICK_SETTLE_DELAY)
            return ok
        except Exception as e:
            self._debug_log(f"后台点击失败: {e}")
            return None

    # =================================================================
    # 兼容接口（供 unread_detector 等使用）
    # =================================================================

    def detect(self, bgr: np.ndarray) -> List[RedDotResult]:
        return self._find_dots(bgr)

    def detect_region(self, bgr: Optional[np.ndarray],
                       region: Optional[tuple] = None) -> List[RedDotResult]:
        if bgr is None:
            return []
        if region:
            x, y, w, h = region
            x = max(0, int(x)); y = max(0, int(y))
            crop = bgr[y:y + int(h), x:x + int(w)]
            if crop.size == 0:
                return []
            dots = self._find_dots(crop)
            return [RedDotResult(d.x + x, d.y + y, d.w, d.h, d.area,
                                 d.center_x + x, d.center_y + y) for d in dots]
        return self._find_dots(bgr)

    def analyze(self, bgr: Optional[np.ndarray],
                region: Optional[tuple] = None) -> dict:
        try:
            dots = self.detect_region(bgr, region)
            return {"image_size": bgr.shape[:2] if bgr is not None else (0, 0),
                    "dots": dots, "error": ""}
        except Exception as e:
            return {"image_size": (0, 0), "dots": [], "error": str(e)}

    # =================================================================
    # 调试归档 / 日志
    # =================================================================

    def _archive_debug(self, img: Optional[np.ndarray] = None) -> None:
        try:
            debug_dir = Path(self._data_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if img is not None:
                try:
                    import cv2
                    cv2.imwrite(str(debug_dir / f"red_dot_{ts}.png"), img)
                except Exception:
                    Image.fromarray(img[:, :, ::-1]).save(
                        str(debug_dir / f"red_dot_{ts}.png"))
            if self._nav_badge:
                (debug_dir / f"nav_badge_{ts}.txt").write_text(
                    str(self._nav_badge), encoding="utf-8")
            if self._contact_dots:
                (debug_dir / f"contact_dots_{ts}.txt").write_text(
                    str(self._contact_dots), encoding="utf-8")
        except Exception:
            pass

    def _debug_log(self, msg: str) -> None:
        try:
            import sys, time
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] [red_dot] {msg}\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            if self._logger is not None:
                self._logger("red_dot", msg)
        except Exception:
            pass

    def _cfg(self, key: str, default=None):
        """读配置：顶层优先，wechat 子字典兜底。"""
        if isinstance(self._config, dict):
            if key in self._config:
                return self._config[key]
            wechat_cfg = self._config.get("wechat", {})
            if isinstance(wechat_cfg, dict) and key in wechat_cfg:
                return wechat_cfg[key]
        return default

    @property
    def last_screenshot(self) -> Optional[np.ndarray]:
        return self._last_screenshot

    @property
    def nav_badge(self) -> Optional[dict]:
        return self._nav_badge

    @property
    def contact_dots(self) -> List[dict]:
        return self._contact_dots


# =====================================================================
# ImageBasedDetector —— 模板匹配红点检测器（独立工具，可选使用）
# =====================================================================

class ImageBasedDetector:
    """基于模板匹配的红点检测器，用于像素检测不可靠时的补充手段。"""

    def __init__(self, template_dir: Optional[str] = None,
                 threshold: float = 0.8):
        self._template_dir = Path(template_dir) if template_dir else Path("data/red_dot_templates")
        self._threshold = threshold
        self._templates: List[np.ndarray] = []
        self._load_templates()

    def _load_templates(self) -> None:
        self._templates = []
        if not self._template_dir.exists():
            return
        try:
            import cv2
            for f in self._template_dir.glob("*.png"):
                img = cv2.imread(str(f))
                if img is not None:
                    self._templates.append(img)
        except Exception:
            pass

    def detect(self, bgr: np.ndarray) -> List[dict]:
        if bgr is None or not self._templates:
            return []
        try:
            import cv2
            results = []
            for tmpl in self._templates:
                th, tw = tmpl.shape[:2]
                if th > bgr.shape[0] or tw > bgr.shape[1]:
                    continue
                res = cv2.matchTemplate(bgr, tmpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self._threshold)
                for pt in zip(*loc[::-1]):
                    results.append({
                        "x": int(pt[0]), "y": int(pt[1]),
                        "w": tw, "h": th,
                        "center_x": int(pt[0] + tw / 2),
                        "center_y": int(pt[1] + th / 2),
                        "confidence": float(res[pt[1], pt[0]]),
                    })
            return results
        except Exception:
            return []


__all__ = [
    "RedDotResult", "RedDot", "RedDotDetection",
    "RedDotDetector", "ImageBasedDetector",
]
