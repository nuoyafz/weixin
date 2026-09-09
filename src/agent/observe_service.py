"""ObserveService v3.10 — 微信自动处理主循环。

对齐原版 app.agent.observe_service v3.10：
  - 完整观察-分析-决策-回复主循环
  - 六级瀑布过滤 (skip→FAQ→keyword→AI)
  - 自动发送保护 (置信度检查 + 证据门控)
  - 发送失败重试队列
  - 发送前客户消息重识别
  - 无未读时补看当前聊天
  - 好友申请自动通过
  - 语音转文字
  - 托管屏检测与恢复
"""
from __future__ import annotations

import copy
import datetime
import difflib
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..ai.text_model_client import TextModelClient
from ..ai.vision_model_client import VisionModelClient
from ..ai.model_call_trace import record_model_call, update_model_call_outcome
from ..agent.conversation_policy import ConversationPolicy
from ..agent.message_pipeline import (  # noqa: F401  消息处理流水线（编排+可观测）
    MessagePipeline, ProcessingRecord,
    STAGE_ENTERING, STAGE_CAPTURED, STAGE_OCR,
    STAGE_ANALYZING, STAGE_SENDING,
    STATUS_SENT, STATUS_SKIPPED, STATUS_MANUAL, STATUS_FAILED,
)
from ..brain.decision_engine import DecisionEngine
from ..brain.evidence_gate import EvidenceGate
from ..brain.fallback_reply import FallbackReply
from ..brain.faq_reply import FaqReplyMatcher
from ..brain.keyword_reply import KeywordReplyMatcher
from ..brain.natural_reply import NaturalReplyPolisher
from ..brain.presales_sop import PresalesSOP
from ..brain.quote_reply_policy import QuoteReplyPolicy
from ..brain.reply_draft import ReplyDraft
from ..brain.weak_lead_flow import WeakLeadManager
from ..config import DATA_DIR, load_config
from ..desktop.display_scale import display_scale_report
from ..desktop.vdd_health import vdd_health_report
from ..desktop.window_finder import WeChatWindowFinder
from ..desktop.privacy_overlay import PrivacyOverlay
from ..desktop.wechat_theme import validate_wechat_theme
from ..desktop.wechat_window_manager import WeChatWindowManager
from ..local_vision.ocr_text_similarity import looks_like_same_ocr_text
from ..rag.indexer import RagIndexer
from ..rpa.human_activity_monitor import HumanActivityMonitor
from ..rpa.friend_request_acceptor import FriendRequestAcceptor
from ..rpa.red_dot_detector import RedDotDetector
from ..rpa.unread_detector import UnreadDetector
from ..rpa.voice_to_text import VoiceToTextConverter
from ..rpa.wechat_sender import WechatSender
from ..storage.file_store import FileStore
from ..storage.leads_repo import LeadsRepo
from ..storage.messages_repo import CyclesRepo, MessagesRepo
from ..storage.db import save_chat_session
from ..capture.screen_capture import ScreenCapture

# 进程级协调锁：防止 UI 和 Worker 两个 ObserveService 实例同时操作
_CYCLE_COORDINATION_LOCK = threading.Lock()


class ObserveService:
    """微信自动处理主循环服务 v3.10。

    核心流程：
      1. detect_wechat() → 找到微信窗口
      2. click_unread() → 点击未读消息
      3. capture_once() → 截图
      4. analyze_once() → 视觉模型分析 + 决策引擎
      5. send_current_reply() → 发送回复
      6. persist() → 持久化周期记录
    """

    # 内部常量：不记录日志的意图类型
    _INTERNAL_NO_REPLY_LOG_INTENTS = frozenset({
        "ocr_message_unstable", "system_notice", "ad_or_promotion",
        "payment_notice", "self_latest_message", "group_not_allowed",
        "group_not_mentioned", "group_reply_disabled",
        "group_auto_reply_disabled", "group_allowed_list_empty",
        "contact_blacklist", "wechat_voice_input_screen",
        "voice_auto_reply_disabled", "image_sticker_no_reply",
        "test_contact_not_allowed",
    })

    # 内部常量：硬性不回复意图（来自原版 _is_hard_no_reply）
    _HARD_NO_REPLY_INTENTS = frozenset({
        "image_sticker_no_reply", "payment_notice", "group_allowed_list_empty",
        "system_notice", "ad_or_promotion", "contact_blacklist",
        "group_reply_disabled", "test_contact_not_allowed", "voice_auto_reply_disabled",
        "group_not_allowed", "self_latest_message", "wechat_voice_input_screen",
        "group_auto_reply_disabled", "group_not_mentioned",
    })

    _MIN_RECOGNITION_CONFIDENCE = 0.65
    _STARTUP_DISPLAY_RECOVERY_ATTEMPTS = 6
    _STARTUP_DISPLAY_RECOVERY_DELAY_SECONDS = 0.75

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self._config_reload_lock = threading.RLock()
        self._pending_config_reload: Optional[dict[str, Any]] = None

        cfg = load_config() if config is None else config
        # 兼容 Settings 对象（dataclass）和 dict 两种配置格式
        if hasattr(cfg, "__dataclass_fields__"):
            from dataclasses import asdict
            cfg = asdict(cfg)
        self.config = cfg

        # 窗口查找
        self.finder = WeChatWindowFinder()

        # 文件存储/日志
        self.store = FileStore()
        self.store.append_log("logger", "ObserveService initialized")

        # 窗口管理器
        self.window_manager = WeChatWindowManager(
            self.store, self.config.get("wechat", {}).get("window_title_keywords", ["微信"])
        )

        # 隐私遮罩
        self.privacy_overlay = PrivacyOverlay()

        # 屏幕捕获
        self.capture = ScreenCapture(
            self.store,
            self.config.get("wechat", {}).get("store_address", ""),
            self.window_manager,
        )
        # 实时预览：每次截图后把画面推给 UI（覆盖红点扫描/点击重扫/分析/手动刷新）
        self.capture.on_captured = self._on_captured
        self._last_preview_at = 0.0

        # 视觉模型（完整读取 vision_model 配置；base_url+model 未配置时自动走本地 OCR）
        vm_cfg = self.config.get("vision_model", {}) or {}
        self.vision = VisionModelClient(
            base_url=vm_cfg.get("base_url", ""),
            api_key=vm_cfg.get("api_key", ""),
            model=vm_cfg.get("model", ""),
        )

        # 文本模型（修复：必须传入完整 config，否则内部 available()/ModelRouter
        # 读不到 text_model 段，LLM 永远不会被调用 -> 本地模式永不回复）
        tm_cfg = self.config.get("text_model", {}) or {}
        self.text_model = TextModelClient(
            base_url=tm_cfg.get("base_url", ""),
            api_key=tm_cfg.get("api_key", ""),
            model=tm_cfg.get("model", ""),
            config=self.config,
        )
        # ⑥ 冷启动预热：发一条最小 dummy 请求，提前完成建连/服务端冷启动
        self._prewarm_text_model()

        # 本地 OCR 分析（vision 未配置时的主读取路径，惰性初始化）
        self._local_ocr = None
        self._local_layout = None

        # 决策引擎
        self.decision = DecisionEngine(self.config)

        # 证据门控
        self.evidence_gate = EvidenceGate(self.config)

        # 兜底回复
        self.reply_fallback = FallbackReply(self.config)

        # 会话策略
        self.conversation_policy = ConversationPolicy(self.config)

        # 关键词回复
        self.keyword_reply = KeywordReplyMatcher(self.config)

        # FAQ 回复
        self.faq_reply = FaqReplyMatcher(self.config)

        # 引用回复策略
        self.quote_reply = QuoteReplyPolicy(self.config)

        # 回复草稿
        self.draft = ReplyDraft()

        # 弱线索管理
        self.weak_lead = WeakLeadManager(self.config)

        # 自然语言润色
        self.reply_polisher = NaturalReplyPolisher(self.config)

        # 线索仓库
        self.leads_repo = LeadsRepo()

        # 消息仓库
        self.messages_repo = MessagesRepo()
        self.cycles_repo = CyclesRepo()

        # 售前 SOP
        self.sop = PresalesSOP(self.config)

        # RPA 组件
        self.unread_detector = UnreadDetector()
        self.red_dot_detector = RedDotDetector(
            self.config,
            screen_capture=self.capture,
            window_manager=self.window_manager,
            logger=self.store.append_log,
        )
        self.voice_converter = VoiceToTextConverter()
        self.friend_acceptor = FriendRequestAcceptor()

        # 知识库索引
        try:
            self.indexer = RagIndexer()
            self.indexer.incremental()
        except Exception:
            self.indexer = None

        # 人类活动监控
        self.human_monitor = HumanActivityMonitor(
            self.window_manager, self.privacy_overlay
        )

        # 微信发送器
        self.sender = WechatSender(
            config=self.config,
            finder=self.finder,
            capture=self.capture,
            store=self.store,
            human_monitor=self.human_monitor,
            window_manager=self.window_manager,
            privacy_overlay=self.privacy_overlay,
        )

        # 状态
        self.last_window: Optional[dict[str, Any]] = None
        self.last_image: Optional[Any] = None
        self.last_report: Optional[dict[str, Any]] = None
        self.current_contact: Optional[str] = None

        # 未读点击状态
        self._unread_clicked: bool = False
        self._last_unread_result: Optional[dict[str, Any]] = None

        # 当前聊天状态
        self._last_current_chat_signature: Optional[str] = None
        self._last_current_chat_visual_image: Optional[str] = None
        self._force_current_chat_next_cycle: bool = False
        self._force_current_chat_reason: str = ""
        self._force_current_chat_target_contact: str = ""

        # 发送前重识别
        self._send_guard_reidentify_signatures: dict[str, float] = {}
        self._send_guard_reidentify_contacts: dict[str, list[float]] = {}
        self._send_guard_reidentify_ttl_seconds: float = 30.0

        # 黏滞导航抑制
        self._suppress_stuck_nav_until: float = 0.0
        self._suppress_stuck_nav_reason: str = ""
        self._stuck_nav_current_chat_next_allowed_at: float = 0.0

        # 回复签名追踪
        self._recent_reply_signatures: dict[str, float] = {}
        self._recent_reply_signature_ttl_seconds: float = 180.0
        self._recent_no_reply_signatures: dict[str, float] = {}
        self._recent_no_reply_signature_ttl_seconds: float = 180.0

        # 发送重试
        self._pending_send_report: Optional[dict[str, Any]] = None
        self._pending_send_next_retry_at: float = 0.0

        # 好友申请
        self._last_friend_request_check_at: float = 0.0
        self._last_voice_to_text_result: Optional[dict[str, Any]] = None

        # 停止标志
        self._stop_requested: bool = False

        # 后台循环线程
        self._loop_thread: Optional[threading.Thread] = None

        # 事件回调
        self._event_callbacks: list[Callable[[str, dict], None]] = []

        # 黏滞导航计数器
        self._consecutive_nav_double_clicks: int = 0

        # 无回复熔断状态（防"同一条消息反复烧 LLM"）
        self._no_reply_streak: dict = {}          # contact -> 连续无回复次数
        self._no_reply_skip_until: dict = {}      # contact -> 冷却截止时间戳
        self._last_no_reply_contact: str = ""    # 最近一次无回复的联系人
        self.MAX_CONSECUTIVE_NAV_CLICKS: int = 3
        # 窗口隐身：标记本轮是否由我们拉起微信窗口。
        # 只要本轮点了窗口，周期末就把微信推回屏外，恢复挂机静默态。
        self._window_shown_by_us: bool = False
        # 周期开始前的窗口原状态（最小化/桌面/屏外），周期末按原状态恢复。
        self._window_state_before_cycle: Optional[dict[str, Any]] = None

    # ================================================================
    # 生命周期
    # ================================================================

    def request_stop(self) -> None:
        """请求停止自动化循环。"""
        self._stop_requested = True
        self._append_runtime_log("assistant_stop_requested")

    def request_config_reload(self, config: Optional[dict[str, Any]] = None) -> None:
        """请求线程安全配置重载。"""
        if isinstance(config, dict):
            cfg = copy.deepcopy(config)
        else:
            cfg = load_config()
        with self._config_reload_lock:
            self._pending_config_reload = cfg
        self._append_runtime_log("runtime_config_reload_requested")

    def apply_pending_config_reload(self) -> bool:
        """应用待重载的配置。"""
        with self._config_reload_lock:
            if self._pending_config_reload is None:
                return False
            cfg = self._pending_config_reload
            self._pending_config_reload = None
        self.reload_config(cfg)
        return True

    def reload_config(self, config: Optional[dict[str, Any]] = None) -> None:
        """重新加载配置。"""
        if isinstance(config, dict):
            cfg = copy.deepcopy(config)
        else:
            cfg = load_config()
            # load_config() 可能返回 Settings 数据类（非 dict），与 __init__ 的
            # asdict 兼容分支对齐，避免把 Settings 对象塞进 self.config 导致
            # 后续 config.get(...) AttributeError。
            if not isinstance(cfg, dict):
                try:
                    from dataclasses import asdict as _asdict
                    cfg = _asdict(cfg)
                except Exception:  # noqa: BLE001
                    cfg = {}
        with self._config_reload_lock:
            self.config = cfg
        self._rebind_runtime_config(cfg)
        self._append_runtime_log("runtime_config_reloaded")

    def _rebind_runtime_config(self, config: dict[str, Any]) -> None:
        """根据新配置重新绑定运行时组件。"""
        logger = None
        if hasattr(self, "store") and self.store is not None:
            logger = self.store.append_log

        wechat_cfg = config.get("wechat", {})
        business_cfg = config.get("business", {})

        try:
            self.finder = WeChatWindowFinder()
        except Exception:
            pass

        try:
            self.window_manager = WeChatWindowManager(
                logger, wechat_cfg.get("window_title_keywords", ["微信"])
            )
        except Exception:
            pass

        try:
            self.privacy_overlay = PrivacyOverlay()
        except Exception:
            pass

        try:
            self.capture = ScreenCapture(
                logger,
                business_cfg.get("store_address", ""),
                self.window_manager,
            )
        except Exception:
            pass
        # 重建后必须重绑预览回调，否则截图不再推送到 UI「微信真实画面」
        if getattr(self, "capture", None) is not None:
            self.capture.on_captured = self._on_captured

        try:
            # 与 __init__ 同步：完整读取 vision_model/text_model 配置
            vm_cfg = config.get("vision_model", {}) or {}
            self.vision = VisionModelClient(
                base_url=vm_cfg.get("base_url", ""),
                api_key=vm_cfg.get("api_key", ""),
                model=vm_cfg.get("model", ""),
            )
        except Exception:
            pass

        try:
            tm_cfg = config.get("text_model", {}) or {}
            self.text_model = TextModelClient(
                base_url=tm_cfg.get("base_url", ""),
                api_key=tm_cfg.get("api_key", ""),
                model=tm_cfg.get("model", ""),
                config=config,
            )
        except Exception:
            pass

        try:
            self.decision = DecisionEngine(config)
        except Exception:
            pass

        try:
            self.evidence_gate = EvidenceGate(config)
        except Exception:
            pass

        try:
            self.reply_fallback = FallbackReply(config)
        except Exception:
            pass

        try:
            self.conversation_policy = ConversationPolicy(config)
        except Exception:
            pass

        try:
            self.keyword_reply = KeywordReplyMatcher(config)
        except Exception:
            pass

        try:
            self.faq_reply = FaqReplyMatcher(config)
        except Exception:
            pass

        try:
            self.quote_reply = QuoteReplyPolicy(config)
        except Exception:
            pass

        try:
            self.draft = ReplyDraft()
        except Exception:
            pass

        try:
            self.weak_lead = WeakLeadManager(config)
        except Exception:
            pass

        try:
            self.reply_polisher = NaturalReplyPolisher(config)
        except Exception:
            pass

        try:
            self.sop = PresalesSOP(config)
        except Exception:
            pass

        try:
            self.unread_detector = UnreadDetector()
        except Exception:
            pass

        try:
            self.red_dot_detector = RedDotDetector(
                self.config,
                screen_capture=getattr(self, "capture", None),
                window_manager=getattr(self, "window_manager", None),
                logger=logger,
            )
        except Exception:
            pass

        try:
            self.voice_converter = VoiceToTextConverter()
        except Exception:
            pass

        try:
            self.friend_acceptor = FriendRequestAcceptor()
        except Exception:
            pass

        try:
            self.indexer = RagIndexer()
        except Exception as e:
            self._append_runtime_log(
                f"runtime_config_rag_rebind_failed error={e}"
            )

        try:
            self.sender = WechatSender(
                config=config,
                finder=self.finder,
                capture=self.capture,
                store=self.store,
                human_monitor=self.human_monitor,
                window_manager=self.window_manager,
                privacy_overlay=self.privacy_overlay,
            )
        except Exception:
            if hasattr(self, "sender") and self.sender is not None:
                pass

        try:
            self.human_monitor = HumanActivityMonitor(
                self.window_manager, self.privacy_overlay
            )
        except Exception:
            pass

    def _append_runtime_log(self, message: str) -> None:
        """追加运行时日志。"""
        try:
            if hasattr(self, "store") and self.store is not None:
                self.store.append_log("runtime", message)
        except Exception:
            pass

    # ================================================================
    # 检测与捕获
    # ================================================================

    def detect_wechat(self) -> dict[str, Any]:
        """检测微信窗口并返回窗口信息。"""
        info = self.finder.find()
        if info is None:
            return {"found": False, "error": "wechat_window_not_found", "handle": 0}

        info_dict = info.to_dict()
        handle = info_dict.get("hwnd", 0)
        if not handle:
            return {"found": False, "error": "wechat_window_not_found", "handle": 0}

        # 必须在 prepare 改变窗口状态之前记录用户真实的「原状态」。
        # prepare_window 发现窗口最小化会 ShowWindow(SW_RESTORE) 把它拉到桌面，
        # 若记录时机晚于 prepare，记录的「原状态」就是拉起后的桌面态，
        # 周期末 _restore_window_state_after_cycle 便会把窗口「恢复到桌面」，
        # 与用户「最小化挂机」的预期冲突。
        self.last_window = info_dict
        self._record_window_state_before_cycle()

        self.window_manager.prepare(int(handle))

        if self.privacy_overlay.enabled:
            self.privacy_overlay.show(self.last_window)

        self.finder.focus_target(int(handle))

        self.store.append_log(
            f"detect found={info_dict.get('title', '')}"
            f" title={info_dict.get('title', '')}"
        )

        return info_dict

    def capture_once(self, subdir: str = "wechat",
                     prefix: str = "auto_open_unread",
                     force_refresh: bool = False) -> dict[str, Any]:
        """捕获一次微信窗口截图。

        Args:
            force_refresh: 截图前先强制窗口重绘，规避微信 CEF 窗口的
                PrintWindow 帧滞后（刚点击/切换界面后必须置 True）。
        """
        if self.last_window is None:
            self.detect_wechat()

        if not self.last_window:
            return {"ok": False, "error": "wechat_window_not_found"}

        rect = self.last_window.get("rect", {})
        handle = int(self.last_window.get("hwnd", 0))
        if not handle:
            return {"ok": False, "error": "wechat_window_not_found"}

        result = self.capture.capture_window(
            rect, hwnd=handle, subdir=subdir, prefix=prefix,
            force_refresh=force_refresh,
        )
        if isinstance(result, str):
            self.last_image = result
            return {"ok": True, "image": self.last_image}
        if result.success and result.image is not None:
            import cv2, os
            from datetime import datetime
            from pathlib import Path
            from ..config import DATA_DIR
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base = Path(DATA_DIR) / subdir
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"{prefix}_{ts}.png"
            cv2.imwrite(str(path), result.image)
            latest = base / "latest.png"
            cv2.imwrite(str(latest), result.image)
            self.last_image = str(path)
            return {"ok": True, "image": self.last_image}
        else:
            return {"ok": False, "error": str(getattr(result, "error", "unknown"))}

    def _on_captured(self, image) -> None:
        """截图回调：把微信实时画面推送到 UI 预览（覆盖每次截图动作）。

        轻量节流（150ms）避免同一轮内的多次截图刷屏；base64 JPEG 经
        'preview' 事件发给前端，前端 window.updatePreview 即时渲染。
        """
        try:
            now = time.time()
            if now - self._last_preview_at < 0.15:
                return
            self._last_preview_at = now
            import cv2
            if image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                bgr = image
            ok_, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if not ok_:
                return
            import base64 as _b64
            data_url = "data:image/jpg;base64," + _b64.b64encode(buf.tobytes()).decode("ascii")
            h, w = bgr.shape[:2]
            self._emit_event("preview", {
                "image": data_url, "width": w, "height": h,
                "ts": int(now * 1000),
            })
        except Exception:
            pass

    # ================================================================
    # 消息处理流水线（编排 + 可观测中枢）
    # ================================================================

    @property
    def pipeline(self) -> MessagePipeline:
        """一条消息的生命周期编排器（懒加载）。

        负责：状态机推进、阶段事件广播、终态计数聚合、消息指纹去重。
        不做识别/决策/发送 —— 那些仍由各自模块负责。
        """
        pipe = getattr(self, "_pipeline", None)
        if pipe is None:
            pipe = MessagePipeline(
                emit=self._emit_event,
                log=lambda m: self.store.append_log(m),
            )
            self._pipeline = pipe
        return pipe

    def shutdown(self) -> None:
        """关闭服务，清理资源。"""
        if self.privacy_overlay:
            self.privacy_overlay.close()

        self._recall_wechat_on_stop()

        self._append_runtime_log("assistant_shutdown_keep_wechat_position")

    # ------------------------------------------------------------------
    # 可见日志（控制台 stderr，与 [red_dot] 同款，便于实时观察 OCR/回复）
    # ------------------------------------------------------------------
    def _trace(self, msg: str) -> None:
        """同时写 stderr（控制台可见）与 store（UI 面板），避免关键步骤"看不见"。"""
        try:
            import sys
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] [trace] {msg}\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            self.store.append_log(msg)
        except Exception:
            pass

    def _frame_fingerprint(self, image_path: Any) -> str:
        """截图像素指纹（用于判断"是否拿到了新的一帧"）。"""
        try:
            import hashlib
            with open(str(image_path), "rb") as fh:
                return hashlib.md5(fh.read()).hexdigest()
        except Exception:
            return ""

    def _wait_for_conversation_view(self, max_attempts: int = 6,
                                    interval: float = 0.5) -> bool:
        """点击未读后，轮询等待微信窗口**真正进入会话视图**，再交给 OCR 解析。

        背景（实测根因）：
            微信是 CEF/GPU 自绘窗口，PrintWindow 存在**帧滞后** —— 点击未读后
            立刻截图，拿到的往往是点击**之前**的旧帧（列表视图）。实测同一轮内
            隔 4 秒的重截两张图 md5 完全相同，但红点确实在逐条消失（说明点击
            生效、会话已打开），即截图滞后于真实界面。

        对策（直接针对根因，而非靠"猜这是不是会话"）：
            1. 每次重截都带 force_refresh（RedrawWindow + 等待合成器 flush）；
            2. 不只看"画面变了"，而是用视图闸门确认**确实进入了会话**（view 不再是
               list/unknown），避免把"仍是列表"误当新帧直接拿去 OCR；
            3. 整段轮询**完全不跑 OCR**（每次仅 ~200ms），避免每轮多耗 3 秒。

        Returns:
            True  = 画面已切换为会话视图（可以进入 OCR）
            False = 连续多次仍停留在列表/未知视图（交由证据门控转人工）
        """
        first_fp = self._frame_fingerprint(self.last_image)
        if not first_fp:
            return False
        for attempt in range(max(1, max_attempts)):
            try:
                interval_used = 0.25 if attempt == 0 else interval
                time.sleep(interval_used)
                self.detect_wechat()
                cap = self.capture_once(force_refresh=True, prefix="wait_conv")
                if not (isinstance(cap, dict) and cap.get("ok")):
                    continue
                # 视图闸门：确认真正进入会话（不再是列表/未知）
                if not self._last_frame_looks_like_list():
                    self._trace(
                        f"[step] 画面已切换为会话视图(第{attempt + 1}次重截)，"
                        f"交给 OCR 解析")
                    return True
                self.store.append_log(
                    f"[analyze] 第{attempt + 1}次重截仍是聊天列表"
                    f"（PrintWindow 帧滞后），继续等待")
            except Exception as e:  # noqa: BLE001
                self.store.append_log(f"[analyze] 等待画面刷新异常={e}")
        self._trace(f"[warn] 连续 {max_attempts} 次重截画面仍停留在列表视图，"
                    f"可能点击未真正生效")
        return False

    def _last_frame_looks_like_list(self) -> bool:
        """轻量判断上次截图是否仍是「聊天列表」而非「已打开的会话」。

        信号：clean_reader 解析后 current_contact 为空或仅单字符（列表头部
        容易被误读成单个字母，如 'X'），且消息数偏多 —— 这是没真正进入
        会话的典型特征。用于在点击未读后复核截图时机是否过早。
        """
        try:
            if not self.last_image or not hasattr(self.last_image, "__fspath__") \
                    and not isinstance(self.last_image, str):
                return False
            import cv2
            import numpy as _np
            frame = cv2.imdecode(
                _np.fromfile(str(self.last_image), dtype=_np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return False
            reader = self._ensure_clean_reader()
            if reader is None:
                return False
            a = reader.analyze(frame)
            # v2：直接采用视图闸门的判定（比旧的"联系人长度/右侧气泡"启发式可靠得多）
            if a.view == "list":
                return True
            if a.view == "unknown":
                return True
            # 视图闸门已判定为会话且置信度足够 -> 直接视为已进入会话。
            # 旧的 right==0 启发式在"自己消息因 chat_mid 偏右被误判为 left"
            # 时会误判成列表，造成 wait_conv 轮询白等 + 错误告警。
            if a.view == "conversation" and a.view_confidence >= 0.5:
                return False
            contact = (a.current_contact or "").strip()
            # 真实会话头部是联系人名（>=2 字）；列表误读常为单字符/空
            if len(contact) <= 1:
                return True
            # 列表视图里消息多为「时间戳+联系人」的整行预览，气泡极少在右侧
            right = sum(1 for m in a.messages if m.side == "right")
            if len(a.messages) >= 4 and right == 0:
                return True
            return False
        except Exception:
            return False

    # ================================================================
    # 托管屏管理
    # ================================================================

    def managed_display_status(self) -> dict[str, Any]:
        """读取托管屏实时状态。"""
        raw_displays = self.window_manager.display_manager.list_displays()
        displays = [d.to_dict() if hasattr(d, "to_dict") else d
                    for d in raw_displays]

        selected = self._select_display_from(displays)
        scale = display_scale_report()

        info = self.finder.find()
        info_dict = info.to_dict() if info is not None else {"found": False, "handle": 0, "title": "", "rect": {}}
        rect = info_dict.get("rect", {})
        current_display = ""
        on_managed = False

        if rect and self.window_manager.display_manager:
            current_display = self.window_manager.display_manager.display_for_rect(rect)
            on_managed = self.window_manager.display_manager.is_on_managed_display(rect)

        runtime_check = True
        ok = bool(on_managed)
        reason = ""
        too_small = self._first_too_small_fixed_display()

        if not ok and too_small:
            reason = "managed_display_too_small_for_fixed_window"
        elif not ok:
            reason = "managed_display_unavailable_no_non_primary_display"

        return {
            "ok": ok,
            "displays": displays,
            "display_scale": scale,
            "selected_display": selected,
            "wechat": info_dict,
            "current_display": current_display,
            "on_managed_display": on_managed,
            "runtime_check": runtime_check,
            "reason": reason,
            "message": "",
        }

    def recover_managed_display_status(self) -> dict[str, Any]:
        """读取实时显示状态并恢复延迟的 VDD 枚举。"""
        status = self.managed_display_status()
        vdd = getattr(vdd_health_report, "__call__", vdd_health_report)()
        if hasattr(vdd, "__call__"):
            vdd = {}

        display_cleanup = {"changed": False, "detached_devices": []}
        if hasattr(self.window_manager, "display_manager"):
            dm = self.window_manager.display_manager
            if hasattr(dm, "prune_extra_vdd_displays"):
                display_cleanup = dm.prune_extra_vdd_displays()

        selected = status.get("selected_display", "")
        displays = status.get("displays", [])

        has_non_primary_display = any(
            isinstance(d, dict) and not bool(d.get("primary")) for d in displays
        )

        display_recovery = {"attempted": False, "ok": False, "attempts": 0, "reason": "not_needed"}
        too_small = self._first_too_small_fixed_display()

        if too_small:
            display_recovery = self._attempt_display_recovery(
                "managed_display_too_small_for_fixed_window"
            )
        elif not has_non_primary_display:
            display_recovery = self._attempt_display_recovery(
                "managed_display_unavailable_no_non_primary_display"
            )

        self.store.append_log(
            f"managed_display_recheck ok={status.get('ok')}"
            f" reason={status.get('reason')}"
            f" display_count={len(displays)}"
            f" selected_device={selected}"
            f" primary_scale={status.get('display_scale', {}).get('primary_scale_percent', '')}"
            f" vdd_ok={vdd.get('ok', '')}"
            f" recovery_attempted={display_recovery.get('attempted')}"
            f" recovery_ok={display_recovery.get('ok')}"
            f" recovery_attempts={display_recovery.get('attempts')}"
            f" cleanup_changed={display_cleanup.get('changed')}"
            f" cleanup_detached={display_cleanup.get('detached_devices')}"
        )

        return {
            "vdd": vdd,
            "display_recovery": display_recovery,
            "display_cleanup": display_cleanup,
        }

    def _attempt_display_recovery(self, reason: str) -> dict[str, Any]:
        """尝试恢复显示状态。"""
        result = {"attempted": False, "ok": False, "attempts": 0, "reason": reason}
        for attempt in range(self._STARTUP_DISPLAY_RECOVERY_ATTEMPTS):
            time.sleep(self._STARTUP_DISPLAY_RECOVERY_DELAY_SECONDS)
            switch_result = self._request_extend_display()
            result["attempted"] = True
            result["attempts"] = attempt + 1
            if switch_result.get("ok"):
                result["ok"] = True
                result["reason"] = "managed_display_resolution_recovered"
                result["resolution"] = switch_result
                break
        return result

    def startup_environment_check(self) -> dict[str, Any]:
        """运行一次性环境检查。"""
        recall = self.recall_wechat_to_primary()
        status = self.recover_managed_display_status()

        vdd = status.get("vdd", {})
        selected = status.get("selected_display", "")
        display_recovery = status.get("display_recovery", {})

        runtime = status.get("runtime_check", True)
        ok = bool(runtime)
        reason = status.get("reason", "")
        if not ok:
            reason = "managed_display_unavailable"

        result = {
            "ok": ok,
            "reason": reason,
            "vdd": vdd,
            "display_recovery": display_recovery,
            "wechat_recall": recall,
            "startup_check": True,
        }

        self.store.append_log(
            f"startup_environment_check ok={ok}"
            f" reason={reason}"
            f" vdd_ok={vdd.get('ok', '')}"
            f" vdd_reason={vdd.get('reason', '')}"
            f" display_recovery_attempted={display_recovery.get('attempted')}"
            f" display_recovery_ok={display_recovery.get('ok')}"
            f" display_recovery_attempts={display_recovery.get('attempts')}"
            f" wechat_recalled={recall}"
        )

        return result

    def _request_extend_display(self) -> dict[str, Any]:
        """请求 Windows 暴露 VDD 为扩展屏。"""
        if os.name != "nt":
            return {"ok": False, "reason": "display_extend_unsupported_platform", "return_code": -1}

        system_root = Path(os.environ.get("SystemRoot", "C:\\Windows"))
        display_switch = system_root / "System32" / "DisplaySwitch.exe"
        if not display_switch.exists():
            return {"ok": False, "reason": "display_extend_unsupported_platform", "return_code": -1}

        try:
            completed = subprocess.run(
                [str(display_switch), "/extend"],
                check=False, timeout=12,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            if completed.returncode == 0:
                return {"ok": True, "reason": "display_extend_requested", "return_code": 0}
            return {"ok": False, "reason": "display_extend_request_failed",
                    "return_code": completed.returncode, "error": str(completed.stderr)}
        except Exception as e:
            return {"ok": False, "reason": "display_extend_request_failed",
                    "return_code": -1, "error": str(e)}

    def move_wechat_to_managed_display(self, quick: bool = False) -> dict[str, Any]:
        """将微信窗口移动到托管屏。"""
        human_monitor = getattr(self, "human_monitor", None)
        if human_monitor and hasattr(human_monitor, "wait_for_mouse_release"):
            released = human_monitor.wait_for_mouse_release(timeout_seconds=1.2)
            if not released.get("ok"):
                return {
                    "ok": False,
                    "reason": "user_dragging_mouse",
                    "message": "检测到鼠标仍在按住或拖动，已暂缓移动微信窗口。松开鼠标后会在下一次检测时继续。",
                    "mouse_guard": released,
                }

        recovered_status = self.recover_managed_display_status()
        selected_display = recovered_status.get("selected_display", "")
        display_recovery = recovered_status.get("display_recovery", {})

        if not display_recovery.get("ok", False) and recovered_status.get("reason", ""):
            return {
                "ok": False,
                "reason": recovered_status.get("reason", "managed_display_unavailable_no_non_primary_display"),
                "display_status": recovered_status,
                "display_recovery": display_recovery,
                "runtime_check": recovered_status.get("runtime_check", True),
            }

        info = self.finder.find()
        info_dict = info.to_dict() if info is not None else {"found": False, "handle": 0, "title": ""}
        if not info_dict.get("found"):
            return {"ok": False, "reason": "wechat_window_not_found", "window": info_dict, "runtime_check": True}

        prepared = self.window_manager.prepare(int(info_dict.get("handle", 0)))
        self.last_window = info_dict

        handle = int(info_dict.get("handle", 0))
        preview_path = self.capture.capture_window(
            hwnd=handle, subdir="managed_preview", prefix="startup_preflight"
        )
        if not preview_path:
            return {
                "ok": False,
                "reason": "managed_preview_capture_failed",
                "message": "后台托管屏截图失败，未启动自动回复。请检查微信窗口、虚拟屏/第二屏和窗口遮挡状态后重新检测。",
                "window": info_dict,
                "runtime_check": True,
            }

        theme_check = validate_wechat_theme(preview_path)
        if not theme_check.get("ok", True):
            return {
                "ok": False,
                "reason": "wechat_theme_not_light",
                "message": "后台托管要求微信使用浅色模式。请在微信 设置 → 通用 → 外观 改为浅色，然后重新一键检测/启动助手。",
                "window": info_dict,
                "startup_preview_path": str(preview_path),
                "runtime_check": True,
                "wechat_theme_check": theme_check,
            }

        mode = "full" if not quick else "quick"
        return {
            "ok": True,
            "reason": "managed_display_ready",
            "window": info_dict,
            "startup_preview_path": str(preview_path),
            "runtime_check": True,
            "wechat_theme_check": theme_check,
            "check_mode": mode,
        }

    def recall_wechat_to_primary(self) -> bool:
        """召回微信到主显示器。"""
        try:
            info = self.finder.find()
            info_dict = info.to_dict() if info is not None else {"found": False, "handle": 0, "title": ""}
            handle = int(info_dict.get("handle", 0))
            if not handle:
                return False
            self.window_manager.find_window(handle)
            self.last_window = info_dict
            return self.window_manager.move_to_primary()
        except Exception:
            return False

    def capture_managed_preview(self) -> dict[str, Any]:
        """捕获托管屏预览。"""
        status = self.managed_display_status()
        wechat = status.get("wechat", {})
        if not wechat.get("found"):
            return {"ok": False, "reason": "wechat_window_not_found", "status": status}

        rect = wechat.get("rect", {})
        handle = int(wechat.get("handle", 0))
        if not handle:
            return {"ok": False, "reason": "wechat_window_not_found", "status": status}

        try:
            path = self.capture.capture_window(
                hwnd=handle, subdir="managed_preview", prefix="preview"
            )
            if not path:
                return {"ok": False, "reason": "managed_preview_capture_failed", "status": status}
            return {"ok": True, "image_path": str(path), "status": status}
        except RuntimeError:
            return {"ok": False, "reason": "managed_preview_capture_failed", "status": status}

    def _managed_display_mode(self) -> bool:
        """判断是否处于托管屏模式。"""
        wechat = self.config.get("wechat", {})
        if not isinstance(wechat, dict):
            return False
        strategy = str(wechat.get("background_mode_strategy", "")).lower().replace("-", "_")
        return strategy in {"virtual_display", "second_display", "managed_display"}

    def _recall_wechat_on_stop(self) -> None:
        """停止时召回微信到主显示器。"""
        pass

    def _first_too_small_fixed_display(self) -> bool:
        """检查是否有太小而无法容纳固定窗口的托管屏。"""
        return False

    def _select_display_from(self, displays: list[dict[str, Any]]) -> str:
        """从显示器列表中选择托管屏。"""
        for d in displays:
            if isinstance(d, dict) and not d.get("primary", False):
                return d.get("device", "")
        return ""

    # ================================================================
    # 未读检测与点击
    # ================================================================

    def click_unread(self) -> dict[str, Any]:
        """寻找并点击未读消息 —— v3.10：死锁防护。

        死锁场景：双击 tab → 微信滚到联系人 → 但下一轮截图扫不到联系人
        → tab 还有红点 → 又双击 → 死循环
        防护：连续 3 次"双击 tab 但下次还扫不到联系人" → 强制 idle 一轮，
              清零计数，给微信和环境恢复时间。
        """
        handle = 0
        if self.last_window:
            handle = int(self.last_window.get("hwnd", 0) or self.last_window.get("handle", 0))

        if not handle:
            result: dict[str, Any] = {
                "found": False, "contact": "", "clicked": False,
                "reason": "no_window_handle", "dot_count": 0,
                "entered_conversation": False, "kind": "none",
                "nav_badge": {}, "contact_dots": [],
            }
        else:
            # 检测器 + ScreenCapture 已统一负责：实时窗口矩形（几何单一真相源）
            # 与截图前后的 unpark/repark（后台托管不抢焦点）。此处只取句柄并调用，
            # 不再重复 park/unpark 与取矩形逻辑。
            try:
                result = self.red_dot_detector.find_and_click_unread(
                    window_handle=handle, wm=self.window_manager)
            except Exception as e:
                self.store.append_log(f"click_unread error: {e}")
                result = {
                    "found": False, "contact": "", "clicked": False,
                    "reason": f"exception: {e}", "kind": "none",
                    "entered_conversation": False, "nav_badge": {}, "contact_dots": [],
                }

        self._unread_clicked = bool(result.get("clicked"))
        if result.get("clicked"):
            self._window_shown_by_us = True
        self._last_unread_result = dict(result)

        kind = str(result.get("kind", ""))
        if kind == "nav_badge":
            self._consecutive_nav_double_clicks += 1
            if self._consecutive_nav_double_clicks > self.MAX_CONSECUTIVE_NAV_CLICKS:
                self.store.append_log(
                    f"unread_deadlock_break 连续双击 tab 超 "
                    f"{self.MAX_CONSECUTIVE_NAV_CLICKS} 次，强制 idle"
                )
                self._consecutive_nav_double_clicks = 0
                return {"found": False, "contact": "", "clicked": False,
                        "reason": "防死锁：连续双击 tab 3+ 次仍扫不到联系人",
                        "dot_count": 0}
        elif kind:
            self._consecutive_nav_double_clicks = 0

        self.store.append_log(
            f"unread_detect found={result.get('found')}"
            f" clicked={result.get('clicked')}"
            f" reason={result.get('reason')}"
            f" kind={kind}"
            f" method={result.get('click_method', '')}"
            f" nav_streak={self._consecutive_nav_double_clicks}"
        )

        return result

    def click_file_transfer(self) -> dict[str, Any]:
        """测试用：点击文件传输助手。"""
        contact = self.config.get("test_mode", {}).get(
            "file_transfer_contact", "文件传输助手"
        )
        result = self.unread_detector.find_specific_contact(contact)
        self.store.append_log(f"click_file_transfer result={result}")
        return result

    # ================================================================
    # 好友申请
    # ================================================================

    def accept_friend_requests(self, force: bool = False,
                               max_requests: int = 3) -> dict[str, Any]:
        """通过待验证好友申请。

        默认只在通讯录入口有红点时才进入"新的朋友"，避免无申请时频繁打扰用户。
        """
        limit = int(self.config.get("friend_requests", {}).get("max_per_cycle", 3))
        if max_requests:
            limit = max_requests

        handle = 0
        if self.last_window:
            handle = int(self.last_window.get("handle", 0))

        # accept_all 返回 dict {ok, accepted_count, reason, screenshots}，
        # 之前直接把整个 dict 当数量 -> 'dict' > 'int' 崩溃 cycle_error
        fr_result = self.friend_acceptor.accept_all(max_requests=limit)
        if isinstance(fr_result, dict):
            accepted = int(fr_result.get("accepted_count", 0) or 0)
        else:
            accepted = int(fr_result or 0)

        self.store.append_log(
            f"friend_request_accept_result accepted={accepted}"
        )

        return {"ok": True, "accepted_count": accepted}

    # ================================================================
    # 本地 OCR 分析（vision 未配置时的主读取路径）
    # ================================================================

    def _vision_ready(self) -> bool:
        """视觉模型是否已完整配置 base_url+model。"""
        return bool(getattr(self.vision, "base_url", "") and getattr(self.vision, "model", ""))

    def _ocr_mode(self) -> str:
        """读取 OCR 模式：local / ai / hybrid（默认 hybrid）。"""
        cfg = self.config or {}
        mode = ""
        if isinstance(cfg, dict):
            mode = str(
                (cfg.get("vision_model") or {}).get("ocr_mode", "")
                or cfg.get("ocr_mode", "")
                or "hybrid"
            ).strip().lower()
        valid = ("local", "ai", "hybrid")
        return mode if mode in valid else "hybrid"

    def _ensure_local_ocr(self) -> bool:
        """惰性初始化本地 OCR 引擎 + 布局解析器。"""
        if self._local_ocr is None:
            try:
                from ..ocr.engine import OCREngine
                from ..local_vision.wechat_layout_parser import WechatLayoutParser
                engine = OCREngine()
                if not engine.initialize():
                    self.store.append_log("local_ocr_init_failed reason=initialize_false")
                    return False
                self._local_ocr = engine
                self._local_layout = WechatLayoutParser()
                self.store.append_log("local_ocr_ready engine=rapidocr")
            except Exception as e:
                self.store.append_log(f"local_ocr_init_failed err={e}")
                return False
        return True

    def _ensure_clean_reader(self) -> Any:
        """惰性初始化干净版感知层 WechatScreenReader（主路径）。

        实例级缓存：避免每个周期都重新加载 RapidOCR onnx 模型。
        """
        reader: Any = getattr(self, "_clean_reader", None)
        if reader is None:
            try:
                from ..clean_perception.reader import WechatScreenReader
                reader = WechatScreenReader()
                self._clean_reader = reader
            except Exception as e:
                self.store.append_log(f"clean_reader_init_failed err={e}")
                self._clean_reader = False  # 哨兵，避免反复尝试
        return reader if reader is not False else None

    # ================================================================
    # ⑤ OCR ROI 裁剪 + 调试图
    # ================================================================

    def _ocr_roi_crop(self, frame, top_ratio=None, bottom_px=None):
        """⑤ OCR ROI 裁剪：按配置裁掉微信窗口标题栏(顶)与输入框(底)，
        缩小喂给 RapidOCR 的图像，减少噪声(×/□按钮、输入框占位符)并提速。
        坐标基于全窗比例，跨分辨率自适配；裁剪失败回退原帧。"""
        cfg = (self.config.get("ocr_roi") or {}) if isinstance(self.config, dict) else {}
        if not cfg.get("enabled", True):
            return frame
        try:
            h, w = frame.shape[:2]
            if top_ratio is None:
                top_ratio = float(cfg.get("top_ratio", 0.045))
            if bottom_px is None:
                bottom_px = int(cfg.get("bottom_px", 60))
            y0 = max(0, int(h * top_ratio))
            y1 = max(y0 + 1, h - max(0, bottom_px))
            return frame[y0:y1, :]
        except Exception:
            return frame

    def _save_roi_debug(self, frame, roi) -> None:
        """⑤ 存调试图：original.png(整窗) / roi.png(裁剪后) / overlay.png(整窗+红框)。
        供核对裁剪是否裁错（尤其顶部是否误伤联系人名）。"""
        cfg = (self.config.get("ocr_roi") or {}) if isinstance(self.config, dict) else {}
        # 默认关闭：调试图每轮 3 张(~1.3MB)，真机长时间运行会堆积。
        # 需要核对裁剪效果时，在 config.yaml 的 ocr_roi 段显式设 save_debug: true。
        if not cfg.get("save_debug", False):
            return
        try:
            import cv2
            from pathlib import Path
            h, w = frame.shape[:2]
            top_ratio = float(cfg.get("top_ratio", 0.045))
            bottom_px = int(cfg.get("bottom_px", 60))
            y0 = max(0, int(h * top_ratio))
            y1 = max(y0 + 1, h - max(0, bottom_px))
            base = Path(__file__).resolve().parents[2] / "logs" / "roi_debug"
            base.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(base / "original.png"), frame)
            cv2.imwrite(str(base / "roi.png"), roi)
            ov = frame.copy()
            cv2.rectangle(ov, (0, y0), (w - 1, y1), (0, 0, 255), 3)
            cv2.imwrite(str(base / "overlay.png"), ov)
        except Exception:
            pass

    # ================================================================
    # ⑥ 冷启动预热
    # ================================================================

    def _prewarm_text_model(self) -> None:
        """⑥ 冷启动预热：发一条最小 dummy 请求，提前完成 DNS+TLS 握手与服务端
        首次推理冷启动，使首个真实回复不再承担建连/预热延迟。失败不影响主流程，
        且不打印任何 api_key。"""
        tm = getattr(self, "text_model", None)
        if tm is None:
            return
        tm_cfg = (self.config.get("text_model") or {}) if isinstance(self.config, dict) else {}
        if not (tm_cfg.get("base_url") and tm_cfg.get("model")):
            return
        store = getattr(self, "store", None)
        if store is not None:
            store.append_log("[prewarm] 文本模型冷启动预热开始")
        try:
            _old_to = tm.timeout_seconds
            _old_max = tm.max_tokens
            tm.timeout_seconds = 8          # 预热失败最多阻塞 8s，不影响启动
            tm.max_tokens = 1               # 极小回复，控制预热成本
            _t0 = time.time()
            res = tm.complete("你是预热用的静默助手，只回复一个字。", "ping")
            _dt = int((time.time() - _t0) * 1000)
            ok = bool(getattr(res, "success", False))
            if store is not None:
                store.append_log(f"[prewarm] 文本模型预热完成 latency={_dt}ms ok={ok}")
        except Exception as e:
            if store is not None:
                store.append_log(f"[prewarm] 文本模型预热异常(已忽略) err={e}")
        finally:
            try:
                tm.timeout_seconds = _old_to
                tm.max_tokens = _old_max
            except Exception:
                pass


    def _analyze_with_local_ocr(self, image_path: str) -> dict:
        """本地 OCR 分析：截图 → RapidOCR → 布局解析 → 原版 schema。

        主路径：干净版 clean_perception.WechatScreenReader（从零自写，
        不依赖反编译泥潭）。若它未能解析出有效联系人，则 fallback 到
        原版 local_vision.wechat_layout_parser 兜底，保证行为不退化。
        """
        frame = None
        try:
            import cv2
            import numpy as _np
            frame = cv2.imdecode(
                _np.fromfile(str(image_path), dtype=_np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.store.append_log(f"local_ocr_read_failed err={e}")
            return {}
        if frame is None:
            self.store.append_log("local_ocr_read_failed reason=image_none")
            return {}

        # ⑤ OCR ROI 裁剪：按配置裁掉标题栏(顶)与输入框(底)，缩小喂给 RapidOCR 的
        # 图像，减少噪声并提速；同步存 original/roi/overlay 三张调试图到 logs/roi_debug/。
        roi_frame = self._ocr_roi_crop(frame)
        self._save_roi_debug(frame, roi_frame)

        # 顶部被裁掉时，通知 reader 关闭「顶部 4.5% 标题栏过滤」，
        # 避免把帧顶的联系人名误判为窗口标题栏而丢弃（reader.analyze 的 roi_cropped 标志）。
        _roi_cfg = (self.config.get("ocr_roi") or {}) if isinstance(self.config, dict) else {}
        _top_cropped = bool(_roi_cfg.get("enabled", True)) and float(_roi_cfg.get("top_ratio", 0.0)) > 0

        # ---- 主路径：干净版 reader ----
        try:
            reader = self._ensure_clean_reader()
            if reader is not None:
                analysis = reader.analyze(roi_frame, roi_cropped=_top_cropped)
                if analysis is not None:
                    is_conversation = getattr(analysis, "view", "") == "conversation"
                    has_messages = bool(getattr(analysis, "messages", None))
                    # 放宽：只要视图是会话且识别到消息，即使联系人名因 OCR
                    # 模糊而为空，也返回结果，后续可由视觉模型兜底补联系人名。
                    if analysis.current_contact or (is_conversation and has_messages):
                        result = analysis.to_analyze_schema(image_path)
                        lm = result.get("latest_message", {}) or {}
                        self.store.append_log(
                            f"clean_reader contact={result.get('current_contact', '')!r} "
                            f"intent={result.get('intent')} "
                            f"msg={str(lm.get('content', ''))[:48]!r}")
                        return result
                self.store.append_log(
                    "clean_reader returned no_contact -> fallback to local_vision")
        except Exception as e:
            self.store.append_log(f"clean_reader_failed err={e}")

        # ---- fallback：原版 local_vision ----
        if not self._ensure_local_ocr() or self._local_layout is None:
            return {}
        try:
            lines = self._local_ocr.get_text_lines(roi_frame)
            result = self._local_layout.analyze_chat_screen(roi_frame, ocr_lines=lines)
            lm = result.get("latest_message", {}) or {}
            self.store.append_log(
                f"local_ocr_analyze contact={result.get('current_contact', '')!r} "
                f"intent={result.get('intent')} "
                f"msg={str(lm.get('content', ''))[:48]!r}"
            )
            return result
        except Exception as e:
            self.store.append_log(f"local_ocr_analyze_failed err={e}")
            return {}

    # ================================================================
    # L4 视觉模型兜底升级（能低不高）
    # ================================================================

    def _should_upgrade_to_vision(self, analysis: dict, local: dict) -> bool:
        """判断是否需要升级到视觉模型（L4 兜底）。

        原项目四级感知瀑布「能低不高」：低层能确定就不升级，
        只有本地 OCR 结果不足以采信时才调用更贵更慢的视觉模型。

        升级条件（满足任一）：
          - 视图判定为 unknown（不知道在看什么）
          - 视图判定为 list（没真正进入会话，值得再确认一次）
          - 感知置信度低于 fallback_min_confidence
          - 结果自相矛盾：判定要回复，却拿不到联系人或消息内容
        """
        if not self._vision_ready():
            return False
        cfg = self.config.get("vision_model", {}) or {}
        if not bool(cfg.get("fallback_enabled", True)):
            return False

        min_conf = float(cfg.get("fallback_min_confidence", 0.70) or 0.70)
        view = str(analysis.get("view") or (local or {}).get("view") or "")
        conf = float(analysis.get("perception_confidence")
                     or (local or {}).get("perception_confidence") or 0.0)

        if view == "unknown":
            return True
        if view == "list":
            return bool(cfg.get("fallback_on_list_view", True))
        if conf < min_conf:
            return True
        if analysis.get("should_reply"):
            if not str(analysis.get("current_contact") or "").strip():
                return True
            latest = analysis.get("latest_message") or {}
            if not str(latest.get("content") or "").strip():
                return True
        return False

    def _upgrade_with_vision(self, image_path: str, local: dict) -> Optional[dict]:
        """调用视觉模型（L4）重新分析，用其语义结论覆盖本地结果。

        重要：视觉模型**不返回像素坐标**，因此必须保留本地 OCR 的
        messages / anchors 等坐标字段，只覆盖语义字段，
        避免丢失「侧边栏排除」与「气泡左右归属」的成果。
        """
        try:
            self._trace("[vision] 本地OCR结果不足信，升级到视觉模型(L4)兜底")
            vision_result = self.vision.analyze_wechat_screen(image_path, self.last_window)
            # ReplyResult -> dict（对齐原版：视觉模型返回 JSON 分析结果）
            if hasattr(vision_result, "content") and not isinstance(vision_result, dict):
                raw = vision_result.content
                try:
                    import json
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    vision_result = parsed if isinstance(parsed, dict) else {"raw_response": raw}
                except Exception:
                    vision_result = {"raw_response": str(raw)}
            if not isinstance(vision_result, dict) or not vision_result:
                return None

            merged = self.decision.normalize(vision_result)
            # 保留本地 OCR 的坐标/布局成果（视觉模型不提供这些）
            for key in ("messages", "anchors", "raw_text_lines", "low_conf_flags"):
                if key in (local or {}):
                    merged.setdefault(key, local[key])
            # 视觉模型确认拿到了「联系人 + 消息内容」：说明确实进入了会话，
            # 覆盖本地低置信度/列表误判，否则兜底结果永远被证据门控拦下、永不发送。
            v_contact = str(merged.get("current_contact") or "").strip()
            v_content = str((merged.get("latest_message") or {}).get("content") or "").strip()
            if v_contact and v_content:
                merged["view"] = "conversation"
                merged["view_confidence"] = 0.90
                merged["perception_confidence"] = 0.82
            merged["analysis_source"] = "vision_fallback"
            self.store.append_log(
                f"[analyze] 视觉模型兜底完成 "
                f"contact={merged.get('current_contact', '')!r} "
                f"intent={merged.get('intent')!r} "
                f"should_reply={merged.get('should_reply')!r}")
            self._trace(
                f"[vision] 兜底完成 contact={merged.get('current_contact', '')!r} "
                f"intent={merged.get('intent')!r} "
                f"should_reply={merged.get('should_reply')!r}")
            return merged
        except Exception as e:  # noqa: BLE001
            self.store.append_log(f"[analyze] 视觉模型兜底失败 err={e}")
            self._trace(f"[warn] 视觉模型兜底失败 err={e}")
            return None

    # ================================================================
    # 分析（视觉模型 + 决策引擎）
    # ================================================================

    def analyze_once(self, force_new_capture: bool = False,
                     stop_before_reply_for_voice: bool = False) -> dict[str, Any]:
        """分析单帧截图——视觉模型 + 决策引擎完整管道。

        对齐原版 ObserveService.analyze_once()。
        核心流程：
          1. 截图（或复用已有）
          2. 视觉模型分析
          3. 修复未知发送者
          4. 关键词/FAQ 快速通道
          5. 文本模型精炼（可选）
          6. 自动发送保护
          7. 自然语言润色
        """
        started = time.perf_counter()

        if self.last_window is None:
            self.detect_wechat()

        if self.last_image is None or force_new_capture:
            self.capture_once()

        if not self.last_image:
            self.store.append_log(
                "analyze_skip reason=capture_failed last_image=None")
            return {"intent": "", "decision": "no_reply",
                    "action": "no_reply", "reason": "capture_failed"}

        image_path = str(self.last_image) if hasattr(self.last_image, "__fspath__") else str(self.last_image)

        # 保存分析时刻截图作发送前客户话轮校验的视觉基准（enter vs send 视觉差分）
        try:
            import cv2
            if isinstance(self.last_image, (str, bytes)):
                self._last_analyzed_image = cv2.imread(str(self.last_image))
            elif hasattr(self.last_image, "shape"):
                self._last_analyzed_image = self.last_image
            else:
                self._last_analyzed_image = None
        except Exception:
            self._last_analyzed_image = None

        ocr_mode = self._ocr_mode()
        self.store.append_log(
            f"[analyze] analyze_once vision_ready={self._vision_ready()} "
            f"ocr_mode={ocr_mode}")

        local: Optional[dict] = None
        analysis: dict[str, Any] = {}

        if ocr_mode == "ai":
            # —— AI OCR 模式：跳过本地 OCR，直接调用视觉模型 ——
            self._trace("[analyze] OCR 模式=ai，跳过本地 OCR")
            if self._vision_ready():
                vision_result = self.vision.analyze_wechat_screen(image_path, self.last_window)
                if hasattr(vision_result, 'content') and not isinstance(vision_result, dict):
                    raw = vision_result.content
                    try:
                        import json
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        vision_result = parsed if isinstance(parsed, dict) else {"raw_response": raw}
                    except Exception:
                        vision_result = {"raw_response": str(raw)}
                analysis = self.decision.normalize(vision_result if isinstance(vision_result, dict) else {})
                analysis["analysis_source"] = "vision_direct"
                # 视觉模型有联系人+内容时强制视为会话视图
                v_contact = str(analysis.get("current_contact") or "").strip()
                v_content = str((analysis.get("latest_message") or {}).get("content") or "").strip()
                if v_contact and v_content:
                    analysis["view"] = "conversation"
                    analysis["view_confidence"] = 0.90
                    analysis["perception_confidence"] = max(
                        float(analysis.get("perception_confidence") or 0.0), 0.82)
            else:
                analysis = self.decision.normalize({})
                analysis["reason"] = "AI OCR 模式需要配置视觉模型"
                analysis["analysis_source"] = "vision_direct"
        else:
            # —— L3 本地 OCR 主力路径（RapidOCR + 布局解析）——
            # 设计：原项目四级感知瀑布「能低不高」。本地 OCR 快(~3s)、免费、
            # 且有精确像素坐标（支撑侧边栏排除与气泡左右归属），因此**始终先跑**；
            # 只有结果不足信时才升级到更慢更贵的 L4 视觉模型兜底。
            local = self._analyze_with_local_ocr(image_path)
            analysis = self.decision.normalize(local)
            if not local:
                analysis["reason"] = "本地OCR未产出有效分析结果"
            # normalize 可能丢弃感知层新增字段，补回来供证据门控/升级判断使用
            for _k in ("view", "view_confidence", "perception_confidence",
                       "low_conf_flags", "anchors", "messages", "raw_text_lines"):
                if _k in (local or {}):
                    analysis.setdefault(_k, local[_k])
            analysis["analysis_source"] = "local_ocr"

            # —— L4 视觉模型兜底升级：仅当本地结果不足信，且非 local 模式 ——
            if ocr_mode != "local" and self._should_upgrade_to_vision(analysis, local):
                upgraded = self._upgrade_with_vision(image_path, local)
                if upgraded is not None:
                    analysis = upgraded

            # local 模式下视觉模型未配置或结果为空时，保留本地结果
            if not local and ocr_mode != "local" and self._vision_ready():
                vision_result = self.vision.analyze_wechat_screen(image_path, self.last_window)
                if hasattr(vision_result, 'content') and not isinstance(vision_result, dict):
                    raw = vision_result.content
                    try:
                        import json
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        vision_result = parsed if isinstance(parsed, dict) else {"raw_response": raw}
                    except Exception:
                        vision_result = {"raw_response": str(raw)}
                analysis = self.decision.normalize(vision_result if isinstance(vision_result, dict) else {})
                analysis["analysis_source"] = "vision_direct"

        lm = (analysis.get("latest_message") or {}) or {}
        self._trace(
            f"[ocr] contact={analysis.get('current_contact')!r} "
            f"intent={analysis.get('intent')!r} "
            f"should_reply={analysis.get('should_reply')!r} "
            f"conf={analysis.get('perception_confidence')!r} "
            f"msg={str(lm.get('content', ''))[:40]!r}")

        # 红点徽章数字是未读消息数的权威来源（优先于会话视图内数消息/分隔线）。
        # 该数字在 click_unread 扫描红点时就已读取，存于 _last_unread_result。
        _unread = (self._last_unread_result or {}).get("unread_count")
        if _unread is not None:
            analysis["unread_count"] = int(_unread)
            analysis["unread_count_source"] = "red_dot_badge"

        # 修复：未知发送者
        analysis = self._repair_unknown_latest_sender_from_ocr_guard(analysis)
        analysis = self._repair_customer_text_blocked_as_system(analysis)

        # 恢复语音输入屏幕
        if str(analysis.get("intent", "")) == "wechat_voice_input_screen":
            analysis = self._recover_wechat_voice_input_screen(analysis)

        # 尝试文本重分析
        if self._should_retry_unread_text(analysis):
            analysis = self._retry_unread_text_analysis(analysis)

        # 语音检测
        if self._analysis_latest_is_voice(analysis) and stop_before_reply_for_voice:
            analysis["voice_preflight"] = True

        # 构建报告
        report = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "window": self.last_window,
            "image_path": image_path,
            "analysis": analysis,
            "reply_draft": "",
            "text_meta": None,
            "group_mention_diagnostics": analysis.get("group_mention_diagnostics", {}),
            "decision_trail": [],
            "debug_timing": {},
        }

        self.last_report = report

        # 决策管线
        decision_trail = []
        contact = analysis.get("current_contact", "unknown")
        self.current_contact = contact

        reply = ""
        source_label = ""

        # 别处模型的兜底
        keyword_fast_path = self._apply_keyword_reply(analysis)
        if keyword_fast_path.get("keyword_reply_matched"):
            reply = keyword_fast_path.get("reply", "")
            source_label = "关键词回复"
            decision_trail.append({"step": "keyword_reply", "decision": "keyword_reply_matched"})

        faq_fast_path = self._apply_faq_reply(analysis)
        if faq_fast_path.get("faq_reply_matched"):
            if reply:
                reply = reply + "\n" + faq_fast_path.get("reply", "")
            else:
                reply = faq_fast_path.get("reply", "")
            if not source_label:
                source_label = "常见问题"
            decision_trail.append({"step": "faq_reply", "decision": "faq_reply_matched"})

        # 单次模型管道
        if self._is_single_call_vision_analysis(analysis):
            decision_trail.append({"step": "vision", "decision": "single_call_vision_pipeline",
                                   "called": True})
            model_reply = analysis.get("reply_draft", "")
            if model_reply:
                reply = model_reply
                source_label = "本地识别后的AI回复"

        # 如果还没回复，用文本模型
        if not reply and analysis.get("should_reply", True):
            # 弱线索流
            weak_result = self.weak_lead.process(analysis)
            if weak_result.get("lead_stage"):
                decision_trail.append({"step": "weak_lead", "decision": weak_result.get("lead_stage")})

            # SOP 管线
            sop_result = self.sop.process(analysis)
            if sop_result.get("sop_stage"):
                decision_trail.append({"step": "sop", "decision": sop_result.get("sop_stage")})

            # 会话策略
            policy_result = self.conversation_policy.apply(analysis)
            if policy_result.get("blocked"):
                decision_trail.append({"step": "conversation_policy", "decision": "会话规则"})

            # 证据门控
            if self._requires_authoritative_business_basis(analysis, reply):
                gate_result = self.evidence_gate.check(analysis, reply)
                if not gate_result.get("ok"):
                    decision_trail.append({"step": "evidence_gate", "decision": "handoff",
                                           "reason": gate_result.get("reason", "")})
                    report["reply_draft"] = ""
                    report["reply_source"] = "handoff"
                    report["auto_send_blocked"] = True
                    report["auto_send_block_reason"] = (
                        "当前回复没有在正式业务资料或 FAQ 中找到明确依据，"
                        "已停止自动发送并等待人工确认。"
                    )
                    self._analysis_timing(analysis, started, False)
                    return report

            # === 核心修复：本地 OCR 模式下必须有地方真正调用 LLM 生成回复 ===
            # 上述 weak_lead / sop / policy / evidence_gate 只做策略裁决，
            # 不会产出「客户可见的回复文本」。真正生成回复的是
            # text_model.refine_reply()（RAG 检索 + LLM 生成 + JSON 解析），
            # 它在整个项目里此前从未被调用，导致本地模式下 reply 永远为空、
            # 主循环走到 no_reply 永不发送。这里补上这一步。
            if not reply:
                try:
                    _refined, _model_reply, _meta = self.text_model.refine_reply(
                        analysis, self.last_window, analysis.get("decision"))
                    if isinstance(_refined, dict):
                        analysis = _refined
                    if _model_reply:
                        reply = _model_reply
                        if not source_label:
                            source_label = "AI模型回复"
                        decision_trail.append({
                            "step": "text_model",
                            "decision": "text_model_reply",
                            "called": bool((_meta or {}).get("called")),
                        })
                        self.store.append_log(
                            f"text_model_reply_ok contact={analysis.get('current_contact','')!r} "
                            f"reply={reply[:48]!r}")
                        self._trace(
                            f"[reply] LLM生成回复 contact={analysis.get('current_contact','')!r} "
                            f"reply={reply[:60]!r}")
                    else:
                        decision_trail.append({
                            "step": "text_model",
                            "decision": "text_model_no_reply",
                            "called": bool((_meta or {}).get("called")),
                        })
                        _llm_err = str((analysis or {}).get("text_model_error") or "").strip()
                        if _llm_err:
                            self.store.append_log(f"text_model_reply_empty err={_llm_err}")
                            self._trace(f"[warn] LLM未返回可发送回复：{_llm_err}")
                        else:
                            self.store.append_log("text_model_reply_empty (LLM 未返回可发送回复)")
                            self._trace("[warn] LLM未返回可发送回复（上下文可能为空）")
                except Exception as e:
                    self.store.append_log(f"text_model_refine_failed err={e}")

            # 修复#1：停止后 LLM 迟到结果直接丢弃，不再写聊天历史/发送/恢复窗口。
            # 用户在 analyze_once 执行期间（阻塞于 refine_reply 的同步 LLM 调用）点停止时，
            # 这是迟到结果的第一拦截点，必须立即收尾，避免「已停止还动微信/写历史」。
            if self._stop_requested:
                _sc = analysis.get("current_contact", "") if isinstance(analysis, dict) else ""
                self._append_runtime_log(
                    f"text_model_returned_after_stop contact={_sc!r} skipped")
                report["error"] = "assistant_stop_requested_before_send"
                return report

            # 对齐原版：FallbackReply 策略链
            # 1. relax — 尝试放松 handoff/no_reply 决策，允许非黑名单客户继续
            analysis = self.reply_fallback.relax(analysis)
            if analysis.get("fallback_policy_relaxed"):
                decision_trail.append({"step": "fallback_relax", "decision": "fallback_relaxed"})

            # 2. apply_final — 所有证据路径失败后，标记为模型兜底
            analysis, final_reply = self.reply_fallback.apply_final(analysis, reply)
            if analysis.get("fallback_reply_pending"):
                decision_trail.append({"step": "fallback_apply_final", "decision": "fallback_reply_pending"})
                if not reply and not final_reply:
                    source_label = "AI闲聊兜底"

            # 3. sanitize_outgoing — 阻止内部解释/策略模板到达微信
            analysis, reply = self.reply_fallback.sanitize_outgoing(analysis, reply)
            if analysis.get("fallback_replaced_internal_reply"):
                decision_trail.append({"step": "fallback_sanitize", "decision": "sanitized_internal_reply"})

        # 自动发送保护
        if reply or report.get("reply_draft"):
            protection = self._apply_auto_send_protection(analysis, reply or report.get("reply_draft", ""))
            if not protection.get("ok", True):
                report["auto_send_blocked"] = True
                report["auto_send_block_reason"] = protection.get("reason", "自动发送保护")
                decision_trail.append({"step": "auto_send_protection",
                                       "decision": "auto_send_blocked"})

        # 自然语言润色
        if reply and not report.get("auto_send_blocked"):
            polished = self.reply_polisher.polish(reply, analysis)
            if polished:
                reply = polished

        report["reply_draft"] = reply
        # 同步到 report["analysis"]：send_report 会把 report["analysis"] 透传给
        # SendGuard.check，而 SendGuard 只读 analysis['reply_draft']。本地 OCR 路径下
        # analysis 不会被 _resolve_vision_refine_reply 写入，若不同步守卫会误判
        # "reply_draft is empty" 而跳过发送。
        _ra = report.get("analysis")
        if isinstance(_ra, dict):
            _ra["reply_draft"] = reply
        if isinstance(analysis, dict):
            analysis["reply_draft"] = reply
        report["reply_source"] = source_label
        report["decision_trail"] = decision_trail

        # 引用回复（原版 apply 返回 {'quote_reply': decision}）
        quote_result = self.quote_reply.apply(analysis, reply)
        decision = quote_result.get("quote_reply") if isinstance(quote_result, dict) else None
        if decision:
            report["quote_reply"] = decision

        # 更新模型调用结果
        self._update_vision_model_outcome(analysis, reply)

        self._analysis_timing(analysis, started, keyword_fast_path.get("keyword_reply_matched", False))

        # 聊天历史记录：每轮识别到的整段会话都存一份快照（方哥 2026-09-06 要求）。
        # 旧表 save_message / record_cycle 只定义从未被调用，运行时不落任何聊天记录，
        # 故在此统一入库，供「历史记录」视图按微信会话窗口的样式回看。
        try:
            _isd = isinstance(analysis, dict)
            _msgs = []
            if _isd:
                _msgs = (analysis.get("messages")
                         or analysis.get("visible_conversation_messages")
                         or analysis.get("customer_turn_messages") or [])
            _contact = str(analysis.get("current_contact") or "").strip() if _isd else ""
            _ckey = str(analysis.get("contact_key") or "").strip() if _isd else ""
            _intent = str(analysis.get("intent") or "") if _isd else ""
            _conf = float(analysis.get("confidence") or 0.0) if _isd else 0.0
            _sid = save_chat_session(
                contact=_contact,
                messages=_msgs,
                contact_key=_ckey,
                intent=_intent,
                reply_text=reply or "",
                confidence=_conf,
                source=source_label or "ocr",
            )
            if _sid:
                self._append_runtime_log(
                    "chat_history_saved session=%s msgs=%s contact=%s"
                    % (_sid, len(_msgs), _contact))
        except Exception as e:  # noqa: BLE001
            self._append_runtime_log("chat_history_save error=" + str(e))

        return report

    # ================================================================
    # 发送回复
    # ================================================================

    def send_current_reply(self) -> dict[str, Any]:
        """发送当前回复。"""
        if self.last_report is None:
            self.last_report = self.analyze_once()

        report = self.last_report
        send_result = self.sender.send_report(report)

        self.store.save_report(report, send_result)

        return send_result

    # ================================================================
    # 主循环
    # ================================================================

    def run_one_cycle(self) -> dict[str, Any]:
        """运行一个不可分割的微信事务。

        UI 和 Worker 可能各自持有 ObserveService 实例。
        没有进程级锁时，两个循环可能点击不同联系人，
        而第一个回复还在准备或发送中。
        """
        if not _CYCLE_COORDINATION_LOCK.acquire(blocking=False):
            self.store.append_log("cycle_skip reason=another_cycle_is_running")
            return {
                "ok": False,
                "flow": [],
                "report": None,
                "send_result": None,
                "unread_result": None,
                "paused": True,
                "error": "同一时间只处理一个联系人",
                "cycle_id": "",
                "lead": None,
                "contact": "",
                "idle": True,
                "busy": "等待上一轮处理完成",
            }

        try:
            # 每轮重置当前消息记录，避免沿用上一轮的旧记录
            self._current_record = None
            return self._run_one_cycle_impl()
        finally:
            # 无论走哪条分支退出，都广播一次计数，保证左侧面板计数始终最新
            try:
                self.pipeline.emit_counts()
            except Exception:
                pass
            # 挂机隐身：若本轮是我们拉起了微信窗口、且周期开始时它本就在屏外，
            # 这里把它推回屏外，恢复后台静默态（用户手动打开的微信不动）。
            self._restore_window_state_after_cycle()
            _CYCLE_COORDINATION_LOCK.release()

    def _record_window_state_before_cycle(self) -> None:
        """在截图/点击前记录微信窗口原状态，供周期末按原状态恢复。"""
        wm = self.window_manager
        if wm is None:
            self._window_state_before_cycle = None
            return
        try:
            hw = int((self.last_window or {}).get("hwnd", 0)
                     or (self.last_window or {}).get("handle", 0) or 0)
            if not hw:
                self._window_state_before_cycle = None
                return
            rect = wm.get_rect(hw)
            self._window_state_before_cycle = {
                "hwnd": hw,
                "minimized": wm.is_minimized(hw),
                "on_screen": wm._is_on_screen(rect) if rect is not None else False,
                "rect": rect._asdict() if rect is not None else None,
            }
        except Exception:
            self._window_state_before_cycle = None

    def _restore_window_state_after_cycle(self) -> None:
        """周期末把微信恢复到本轮开始前的状态。

        原来一律 push 回屏外，会导致：
          - 用户把微信最小化挂机 → bot 运行后窗口变成屏外（或闪现到桌面后留在屏外）
          - 用户把微信放在桌面用 → bot 运行完可能被推到屏外
        现在按原状态恢复：
          - 原状态最小化 → 最小化
          - 原状态在桌面可见 → 恢复回原矩形
          - 原状态在屏外（bot 托管）→ 推回屏外
        """
        wm = self.window_manager
        if wm is None:
            self._window_shown_by_us = False
            self._window_state_before_cycle = None
            return
        try:
            hw = int((self.last_window or {}).get("hwnd", 0)
                     or (self.last_window or {}).get("handle", 0) or 0)
            if not hw:
                return
            before = getattr(self, "_window_state_before_cycle", None)
            if before is None:
                # 未记录到原状态时的兜底：仅当本轮确实由我们拉起过窗口才推回屏外
                if getattr(self, "_window_shown_by_us", False):
                    wm.move_offscreen(hw)
                    self.store.append_log("wechat reparked offscreen after cycle (fallback)")
                return

            was_minimized = bool(before.get("minimized", False))
            was_on_screen = bool(before.get("on_screen", False))
            before_rect = before.get("rect")

            if was_minimized:
                # 用户本来就最小化挂机：保持「屏外不可见」态（不闪桌面）。
                # 周期中窗口已被 prepare 转成屏外可见态，这里若它仍在屏外则不动，
                # 仅当意外回到屏内时才重新移出，避免周期末任何桌面闪现。
                if not wm.is_window_offscreen(hw):
                    wm.move_offscreen(hw)
                self.store.append_log("wechat kept offscreen after cycle (was minimized)")
            elif was_on_screen and before_rect:
                # 用户本就在桌面使用：还回原位置
                wm.move(before_rect["x"], before_rect["y"],
                        before_rect["width"], before_rect["height"], hw)
                self.store.append_log("wechat restored to on-screen rect after cycle")
            else:
                # bot 托管在屏外：推回屏外
                wm.move_offscreen(hw)
                self.store.append_log("wechat reparked offscreen after cycle")
        except Exception as e:
            self.store.append_log(f"restore_window_state_after_cycle failed: {e}")
        finally:
            self._window_shown_by_us = False
            self._window_state_before_cycle = None

    def _run_one_cycle_impl(self) -> dict[str, Any]:
        """实际执行一个完整的观察-分析-决策-回复周期。"""
        cycle_started = time.perf_counter()
        debug_timing: dict[str, float] = {}
        flow: list[dict[str, Any]] = []

        def step(name: str, status: str, detail: str = "") -> None:
            flow.append({"name": name, "status": status, "detail": detail})

        result: dict[str, Any] = {
            "ok": True,
            "flow": flow,
            "report": None,
            "send_result": None,
            "unread_result": None,
            "paused": False,
            "error": "",
            "cycle_id": "",
            "lead": None,
            "contact": "",
            "idle": True,
        }

        try:
            # 1. 检测微信
            t0 = time.perf_counter()
            window = self.detect_wechat()
            debug_timing["detect_ms"] = int((time.perf_counter() - t0) * 1000)

            if not window.get("found"):
                step("detect", "failed", "微信窗口未找到")
                result["ok"] = False
                result["error"] = "wechat_window_not_found"
                return result

            # 在截图/点击等可能改变窗口状态的操作之前，先记录本轮开始前的状态，
            # 周期末按原状态恢复（最小化/桌面/屏外）。
            self._record_window_state_before_cycle()

            step("detect", "ok", "检测到微信活动")

            # 检查停止请求
            if self._stop_requested:
                step("stop", "paused", "助手已暂停")
                result["paused"] = True
                result["error"] = "assistant_stop_requested"
                return result

            # 2. 好友申请
            t1 = time.perf_counter()
            friend_result = self._maybe_accept_friend_requests()
            debug_timing["friend_ms"] = int((time.perf_counter() - t1) * 1000)
            if friend_result:
                accepted = friend_result.get("accepted_count", 0)
                if accepted > 0:
                    step("friend_request", "ok", f"已通过 {accepted} 个")

            # 3. 检查待发送重试
            pending_report = self._pending_send_report
            wait_reason = self._pending_send_wait_reason(pending_report or {})

            if pending_report and not wait_reason and not self._stop_requested:
                step("send_retry", "running", "后台补发")
                send_result = self.sender.send_report(pending_report)
                self._pending_send_report = None
                self._pending_send_next_retry_at = 0.0

                if send_result.get("ok"):
                    self._mark_reply_turn(self._reply_turn_signature(
                        str(self._send_guard_contact(pending_report)),
                        str(pending_report.get("reply_draft", ""))
                    ))
                    self._reset_send_guard_reidentify(pending_report)
                    step("send_retry", "done", "RPA操作已完成")
                else:
                    self._handle_failed_send(
                        pending_report, send_result,
                        self._reply_turn_signature(
                            str(self._send_guard_contact(pending_report)),
                            str(pending_report.get("reply_draft", ""))
                        ),
                        result,
                    )

                debug_timing["send_ms"] = int((time.perf_counter() - t1) * 1000)
                self._persist(result)
                return result

            if wait_reason:
                step("send_retry", "deferred", wait_reason)

            # 4. 检查是否强制当前聊天
            force_current_chat = self._force_current_chat_next_cycle
            if force_current_chat:
                step("force_current_chat", "ok", self._force_current_chat_reason)
                self._force_current_chat_next_cycle = False

            # 5. 点击未读
            t2 = time.perf_counter()
            unread = self.click_unread()
            debug_timing["unread_ms"] = int((time.perf_counter() - t2) * 1000)
            # 双击置顶回显：把置顶路径识别到的「进入会话名/是否带未读」透传到本轮结果
            result["pin_echo"] = unread.get("pin_echo")

            self.store.append_log(
                f"[analyze] 点击结果 clicked={unread.get('clicked')} "
                f"entered={unread.get('entered_conversation')} "
                f"kind={unread.get('kind')}")

            if unread.get("clicked"):
                # —— 埋点：识别到未读（recognition 事件，驱动左侧面板）——
                dot_count = len(unread.get("contact_dots") or [])
                if dot_count:
                    self._emit_event("recognition", {"count": dot_count, "contacts": []})

                # 导航徽章仅展开聊天列表、尚未进入具体会话：
                # 不分析列表本身，等下一轮点击联系人（对齐原版
                # “只展开了消息列表，尚未进入具体会话”）。
                if not unread.get("entered_conversation", True):
                    self.store.append_log(
                        f"[analyze] 仅展开聊天列表（nav_badge），本轮不OCR "
                        f"kind={unread.get('kind')} clicked={unread.get('clicked')}")
                    step("click_unread", "ok", "已展开聊天列表，下轮点击会话")
                    result["idle"] = True
                    self._persist(result)
                    self.pipeline.emit_counts()
                    return result
                step("click_unread", "ok", "点击未读消息")
                result["idle"] = False
                self.store.append_log(
                    f"[analyze] 点击未读成功 kind={unread.get('kind')} "
                    f"contact={unread.get('contact')!r} -> 准备截图+OCR识别会话")
                # —— 埋点：开启一条消息的处理记录 ——
                _rec = self.pipeline.begin(
                    contact=str(unread.get("contact") or ""),
                    seed=f"unread:{unread.get('kind')}:{dot_count}",
                )
                # 去重命中已结束的记录：不复活，本轮直接跳过（避免重复处理/重复发送）
                if _rec is not None and _rec.status in (
                        STATUS_SENT, STATUS_SKIPPED, STATUS_MANUAL, STATUS_FAILED):
                    self._trace(f"[pipeline] 去重命中已处理记录 {_rec.id}，跳过本轮")
                    result["idle"] = True
                    self._persist(result)
                    return result
                self._current_record = _rec
                self.pipeline.advance(self._current_record, STAGE_ENTERING)
                open_delay = float(self.config.get("wechat", {}).get(
                    "open_unread_delay_seconds", 0.5))
                time.sleep(max(0.1, open_delay))
                # 点击后重新检测窗口（对齐原版）
                self.detect_wechat()
                report = None
            elif not force_current_chat:
                # —— 无回复熔断：上一轮起该联系人已连续多轮"无回复可发" ——
                # 典型故障：未读点击链路失效后，助手每轮都去读"当前已打开的同一个会话"，
                # OCR 判定 should_reply=True，于是每轮都对同一条消息再烧一次 LLM，
                # 而 LLM 因知识库为空返回空 -> 无回复 -> 未读不清 -> 死循环。
                # 这里用上一轮的联系人做代理判定（死循环场景联系人恒定），
                # 命中冷却期就直接跳过，不再浪费一次慢调用。
                if self._is_no_reply_quarantined():
                    self._trace(
                        f"[skip] 无回复熔断生效，本轮跳过当前聊天 "
                        f"contact={self._last_no_reply_contact!r}")
                    self.store.append_log(
                        "current_chat_skip_no_reply_quarantine contact="
                        + str(self._last_no_reply_contact or ""))
                    step("click_unread", "idle", "无回复熔断，跳过重复会话")
                    result["idle"] = True
                    self._persist(result)
                    return result
                # 对齐原版：无未读时尝试检查当前打开的聊天
                report = self._try_current_chat_when_no_unread(unread)
                if report:
                    result["idle"] = False
                    step("click_unread", "ok", "当前聊天补处理")
                else:
                    self.store.append_log(
                        f"[analyze] 无未读/未点击，本轮空闲 "
                        f"clicked={unread.get('clicked')} "
                        f"entered={unread.get('entered_conversation')} "
                        f"kind={unread.get('kind')}")
                    step("click_unread", "idle", "无未读，本轮等待")
                    result["idle"] = True
                    self._persist(result)
                    return result
            else:
                report = None

            # 6. 截图/分析 —— 对齐原版：若 report 已有值则跳过
            if report is not None:
                # 必须解包嵌套 analysis 键，与 else 支路保持一致。
                # _try_current_chat_when_no_unread() 返回的是完整 report，
                # current_contact/should_reply/intent 都在 report["analysis"] 里；
                # 直接 analysis = report 会让下方全部从 report 顶层取空
                # （表现为 contact=''、should_reply=None，证据门控与日志全失真）。
                analysis = (report.get("analysis", report)
                            if isinstance(report, dict) else {})
            else:
                t3 = time.perf_counter()
                # 点击会话后微信 CEF 视图刚切换，PrintWindow 有帧滞后，
                # 必须 force_refresh 强制重绘，否则拿到旧帧/半渲染白壳 → 截图失败
                cap = self.capture_once(force_refresh=True)
                debug_timing["capture_ms"] = int((time.perf_counter() - t3) * 1000)
                step("capture", "ok", "微信截图已上传")
                if isinstance(cap, dict) and not cap.get("ok"):
                    self.store.append_log(
                        f"[analyze] 截图失败 reason={cap.get('error')} -> 无法OCR")
                    self._trace(f"[warn] 截图失败 reason={cap.get('error')} -> 无法OCR")
                    # —— 埋点：截图失败 ——
                    self.pipeline.finish(
                        self._current_record, STATUS_FAILED,
                        reason=f"截图失败: {cap.get('error')}")
                    step("capture", "failed", f"截图失败: {cap.get('error')}")
                    result["idle"] = False
                    self._persist(result)
                    return result
                else:
                    # —— 埋点：截图完成 ——
                    self.pipeline.advance(self._current_record, STAGE_CAPTURED)
                    self.store.append_log(
                        f"[analyze] 截图完成 path={getattr(self, 'last_image', '')!r}")
                    # 复核：微信 CEF 窗口的 PrintWindow 存在帧滞后，点击后首帧
                    # 常是切换前的列表。轮询等待真正进入会话视图再定稿。
                    if self._last_frame_looks_like_list():
                        self._trace("[step] 首帧仍是聊天列表，等待微信切换会话视图")
                        t3b = time.perf_counter()
                        self.detect_wechat()
                        entered = self._wait_for_conversation_view(
                            max_attempts=3, interval=0.35)
                        debug_timing["capture_ms"] += int((time.perf_counter() - t3b) * 1000)
                        self._trace(f"[step] 截图定稿 path={getattr(self, 'last_image', '')!r}")

                # 7. 语音转文字
                t4 = time.perf_counter()
                voice_result = self._maybe_convert_voice_to_text(self.last_image)
                debug_timing["voice_ms"] = int((time.perf_counter() - t4) * 1000)
                if voice_result and voice_result.get("converted"):
                    step("voice_to_text", "ok", "已转文字")

                # 8. 分析
                t5 = time.perf_counter()
                # 停止检查：分析（含可能升级的视觉模型调用，最长 ~18s）前若已收到
                # 停止指令，立即收尾退出，避免"已停止还在运行"的观感。
                if self._stop_requested:
                    self._trace("[stop] 分析前收到停止指令，收尾退出")
                    rec = getattr(self, "_current_record", None)
                    if rec is not None:
                        self.pipeline.finish(rec, STATUS_FAILED,
                                             reason="assistant_stop_requested")
                    step("analyze", "paused", "已收到停止指令，终止分析")
                    result["idle"] = False
                    self._persist(result)
                    return result
                # analyze_once 返回的是完整 report（含嵌套 analysis 键），
                # 不是 analysis 本身。必须解包，否则下方 view/confidence/
                # current_contact/latest_message 全部从 report 顶层取空，
                # 导致证据门控失效、contact 字段丢失。
                report = self.analyze_once()
                analysis = report.get("analysis", report) if isinstance(report, dict) else {}
                debug_timing["analyze_ms"] = int((time.perf_counter() - t5) * 1000)
                step("analyze", "ok", "视觉分析完成")
                self.store.append_log(
                    f"[analyze] OCR完成 current_contact={analysis.get('current_contact')!r} "
                    f"intent={analysis.get('intent')!r} "
                    f"view={analysis.get('view')!r} "
                    f"conf={analysis.get('perception_confidence')!r} "
                    f"reply_draft={'有' if analysis.get('reply_draft') else '无'}")

                # —— 埋点：OCR 提取完成 ——
                rec = getattr(self, "_current_record", None)
                if rec is not None:
                    latest = analysis.get("latest_message") or {}
                    self.pipeline.advance(
                        rec, STAGE_OCR,
                        contact=str(analysis.get("current_contact") or rec.contact),
                        customer_text=str(analysis.get("content")
                                          or latest.get("content", "")),
                        confidence=float(analysis.get("perception_confidence", 0.0) or 0.0),
                    )

                # —— 证据门控：视图非会话 / 置信度不足 -> 转人工，绝不猜测 ——
                if rec is not None:
                    gate_reason = ""
                    view = str(analysis.get("view") or "")
                    if view == "list":
                        gate_reason = "截图仍是会话列表，未真正进入会话"
                    elif view == "unknown":
                        gate_reason = "无法确定当前视图"
                    elif analysis.get("should_reply") and \
                            float(analysis.get("perception_confidence", 0.0) or 0.0) < 0.70:
                        gate_reason = (f"感知置信度不足"
                                       f"({analysis.get('perception_confidence')})")
                    if gate_reason:
                        self._trace(f"[gate] 转人工 reason={gate_reason}")
                        self.pipeline.finish(rec, STATUS_MANUAL, reason=gate_reason)
                        step("evidence_gate", "manual", gate_reason)
                        result["contact"] = analysis.get("current_contact", "")
                        result["idle"] = False
                        self._persist(result)
                        return result

                # —— 定时任务：自动回复时段门控（P1）。时段外不回复，静默跳过 ——
                _sched = (self.config or {}).get("schedule") or {}
                if _sched.get("enabled"):
                    try:
                        from datetime import datetime as _dt
                        _now = _dt.now().strftime("%H:%M")
                        _s = str(_sched.get("start") or "09:00")
                        _e = str(_sched.get("end") or "22:00")
                        if _s <= _e:
                            _in = _s <= _now <= _e
                        else:  # 跨零点（如 22:00-08:00）
                            _in = _now >= _s or _now <= _e
                        if not _in:
                            _reason = f"非自动回复时段（{_s}-{_e}，当前 {_now}），已跳过"
                            self._trace(f"[schedule] {_reason}")
                            if rec is not None:
                                self.pipeline.finish(rec, STATUS_MANUAL, reason=_reason)
                            step("schedule_gate", "skipped", _reason)
                            result["contact"] = analysis.get("current_contact", "")
                            result["idle"] = False
                            self._persist(result)
                            return result
                    except Exception as _se:  # noqa: BLE001
                        self._trace(f"[schedule] gate error(忽略): {_se}")

            # 9. 发送前检查
            if report.get("auto_send_blocked"):
                step("send_guard", "blocked", report.get("auto_send_block_reason", ""))
                result["contact"] = analysis.get("current_contact", "")
                # —— 埋点：发送守卫拦截 -> 转人工 ——
                self.pipeline.finish(
                    getattr(self, "_current_record", None), STATUS_MANUAL,
                    reason=str(report.get("auto_send_block_reason") or "发送守卫拦截"))
                self._persist(result)
                return result

            if self._stop_requested:
                step("send", "paused", "已收到停止指令，未发送")
                result["paused"] = True
                result["error"] = "assistant_stop_requested_before_send"
                # —— 埋点：停止 -> 记录收尾 ——
                self.pipeline.finish(
                    getattr(self, "_current_record", None), STATUS_FAILED,
                    reason="assistant_stop_requested")
                return result

            reply_draft = report.get("reply_draft", "") or analysis.get("reply_draft", "")
            if not reply_draft:
                step("no_reply", "idle", "没有配置可用回复内容，已停止自动回复。")
                self._trace(f"[warn] 无回复内容可发送 contact={analysis.get('current_contact','')!r} "
                            f"should_reply={analysis.get('should_reply')!r}")
                # —— 埋点：无需回复 -> 跳过（附具体原因）——
                skip_reason = str(analysis.get("intent") or "无可用回复内容")
                # 明确根因：文本模型不可用（多为 api_key 未配置）时不显示为通用
                # "无可用回复内容"，否则用户会误以为识别/置顶逻辑出问题。
                tm = getattr(self, "text_model", None)
                if tm is not None and not tm.available():
                    skip_reason = "文本模型不可用（API Key 未配置/缺失），无法生成回复"
                if analysis.get("is_self_latest_message"):
                    skip_reason = "最新一条是自己发的，等待对方回复"
                self.pipeline.finish(
                    getattr(self, "_current_record", None), STATUS_SKIPPED,
                    reason=skip_reason)
                self._mark_no_reply_turn(analysis)
                # 无回复熔断：连续多轮无回复则把该联系人放进冷却期，
                # 避免对同一条消息反复调用慢 LLM（知识库为空时会无限循环）。
                self._note_no_reply(str(analysis.get("current_contact") or ""))
                self._persist(result)
                return result

            # 已产出可发送回复 -> 清零该联系人的"连续无回复"计数并解除冷却
            self._clear_no_reply(str(analysis.get("current_contact") or ""))
            # —— 埋点：进入分析/生成回复阶段 ——
            self.pipeline.advance(
                getattr(self, "_current_record", None), STAGE_ANALYZING,
                reply_text=reply_draft,
                contact=str(analysis.get("current_contact") or ""))

            # 10. 发送
            t6 = time.perf_counter()
            self._trace(f"[send] 准备发送 contact={analysis.get('current_contact','')!r} "
                        f"reply={reply_draft[:60]!r}")
            # —— 埋点：执行 RPA 发送 ——
            self.pipeline.advance(
                getattr(self, "_current_record", None), STAGE_SENDING,
                reply_text=reply_draft,
                contact=str(analysis.get("current_contact") or ""))
            # 注入分析时刻基准帧，供发送前客户话轮校验做视觉差分双信号
            report["_send_baseline_image"] = getattr(self, "_last_analyzed_image", None)
            send_result = self.sender.send_report(report)
            debug_timing["send_ms"] = int((time.perf_counter() - t6) * 1000)

            reply_signature = self._reply_turn_signature(
                str(analysis.get("current_contact", "")),
                reply_draft,
            )

            if send_result.get("ok"):
                self._mark_reply_turn(reply_signature)
                self._reset_send_guard_reidentify(analysis)
                step("send", "done", "RPA操作已完成")
                self._trace(f"[send] 发送成功 contact={analysis.get('current_contact','')!r}")
                # —— 埋点：发送成功 ——
                self.pipeline.finish(
                    getattr(self, "_current_record", None), STATUS_SENT,
                    contact=str(analysis.get("current_contact") or ""),
                    reply_text=reply_draft)
                result["send_result"] = send_result
            else:
                self._handle_failed_send(analysis, send_result, reply_signature, result)
                # 发送失败时，把本轮点击的红点 y 标记为近期失败，
                # 让下一轮优先处理其他未读（如方舟）。
                if isinstance(self._last_unread_result, dict):
                    last_y = self._last_unread_result.get("click_y")
                    if last_y is not None:
                        try:
                            self.red_dot_detector.mark_contact_click_failed(int(last_y))
                        except Exception:
                            pass
                step("send", "failed", "发送失败")
                self._trace(f"[warn] 发送失败 contact={analysis.get('current_contact','')!r} "
                            f"reason={send_result.get('reason','')} "
                            f"original_reason={send_result.get('original_reason','')}")
                # —— 埋点：发送失败 ——
                self.pipeline.finish(
                    getattr(self, "_current_record", None), STATUS_FAILED,
                    reason=str(send_result.get("reason") or "发送失败"))

            self._suppress_stuck_nav_after_send(send_result)

            result["report"] = report
            result["contact"] = analysis.get("current_contact", "")
            result["send_result"] = send_result

            self._persist(result)
            return result

        except Exception as e:
            result["ok"] = False
            result["error"] = f"cycle_error={e}"
            self.store.append_log(f"cycle_error={e}")
            step("exception", "异常", str(e))
            self._emit_event("error", {"message": f"cycle_error={e}"})
            # —— 埋点：异常 -> 记录收尾为失败，避免永久"处理中" ——
            self.pipeline.finish(
                getattr(self, "_current_record", None), STATUS_FAILED,
                reason=f"cycle_error={e}")
            return result

    def _handle_failed_send(self, report: dict[str, Any],
                            send_result: dict[str, Any],
                            reply_signature: str,
                            result: dict[str, Any]) -> None:
        """处理发送失败。"""
        new_customer = self._send_result_new_customer_blocked(send_result)
        after_enter_blocked = self._send_result_after_enter_new_customer_blocked(send_result)

        if new_customer or after_enter_blocked:
            self._mark_send_unconfirmed(send_result)
            self._register_send_guard_reidentify(report, send_result)

            contact = self._send_guard_contact(report)
            guard_signature = self._send_guard_reidentify_signature(report, send_result)
            contact_attempts = len(self._send_guard_reidentify_contacts.get(
                contact, []))

            if contact_attempts >= 1:
                self.store.append_log(
                    f"send_reidentify_loop_blocked contact={contact}"
                    f" block_reason={send_result.get('reason', '')}"
                    f" contact_attempts={contact_attempts}"
                    f" guard_signature={guard_signature}"
                )
                result["send_result"] = send_result
                result["contact"] = contact
                self._force_current_chat_next_cycle = True
                self._force_current_chat_reason = "发送前发现客户补充消息，锁定当前聊天重新识别"
                self._force_current_chat_target_contact = contact
                self._unmark_reply_turn(reply_signature)
                return

            self._queue_send_retry(report, send_result)

        elif self._send_result_chat_input_not_ready(send_result):
            self._queue_send_retry(report, send_result)
        elif self._send_result_may_have_reached_wechat(send_result):
            self._mark_reply_turn(reply_signature)
        else:
            self._queue_send_retry(report, send_result)

        result["send_result"] = send_result

    def _queue_send_retry(self, report: dict[str, Any],
                          send_result: dict[str, Any],
                          allow_no_retry: bool = False) -> None:
        """将发送失败加入重试队列。"""
        if not isinstance(send_result, dict):
            return

        if allow_no_retry and send_result.get("ok") == "no_retry":
            return

        if self._send_result_chat_input_not_ready(send_result):
            self.store.append_log(
                f"send_retry_suppressed_chat_input_not_ready"
                f" contact={report.get('current_contact', '')}"
            )
            return

        if self._send_result_has_delivery_evidence(send_result):
            return

        if self._send_result_may_have_reached_wechat(send_result):
            if not self._send_retry_uncertain_confirmation_enabled():
                return

        retry_state = send_result.setdefault("_retry_state", {})
        attempts = self._int_value(retry_state.get("attempts", 0))
        limit = self._send_retry_limit()

        if attempts >= limit:
            self.store.append_log(
                f"send_retry_exhausted attempts={attempts}"
                f" limit={limit}"
                f" contact={report.get('current_contact', '')}"
            )
            return

        now = time.time()
        retry_state["attempts"] = attempts + 1
        retry_state["last_reason"] = str(send_result.get("reason", ""))
        retry_state["queued_at"] = datetime.datetime.now().isoformat(timespec="seconds")

        next_retry_at = now + self._send_retry_delay_seconds(attempts)
        retry_state["next_retry_at"] = next_retry_at

        self._pending_send_next_retry_at = next_retry_at
        self._pending_send_report = report

        self.store.append_log(
            f"send_retry_queued attempt={attempts + 1}/{limit}"
            f" contact={report.get('current_contact', '')}"
        )

    # ================================================================
    # 发送结果判断
    # ================================================================

    @staticmethod
    def _send_result_chat_input_not_ready(send_result: dict[str, Any]) -> bool:
        if not isinstance(send_result, dict):
            return False
        reason = str(send_result.get("reason", "")).lower()
        return "chat_input_not_ready" in reason

    @staticmethod
    def _send_result_new_customer_blocked(send_result: dict[str, Any]) -> bool:
        if not isinstance(send_result, dict):
            return False
        reason = str(send_result.get("reason", "")).lower()
        # 仅对"确认有新客户消息/模型不确定"做 no_retry 强拦截与重识别；
        # customer_turn 类文本不匹配已由 send_confirm 用相似度/视觉双信号细判，
        # 不再直接放弃（误判时走重试队列即可）。
        return any(kw in reason for kw in [
            "new_customer_message",
            "send_blocked_model_uncertain_before_enter",
        ])

    @staticmethod
    def _send_result_after_enter_new_customer_blocked(send_result: dict[str, Any]) -> bool:
        if not isinstance(send_result, dict):
            return False
        reason = str(send_result.get("reason", "")).lower()
        return any(kw in reason for kw in [
            "send_blocked_new_customer_message_before_reply_confirm",
            "new_customer_message_arrived_before_reply_confirm",
        ])

    @staticmethod
    def _send_result_may_have_reached_wechat(send_result: dict[str, Any]) -> bool:
        if not isinstance(send_result, dict):
            return False
        return bool(send_result.get("may_have_reached_wechat", False))

    @staticmethod
    def _send_result_has_delivery_evidence(send_result: dict[str, Any]) -> bool:
        if not isinstance(send_result, dict):
            return False
        return bool(send_result.get("has_delivery_evidence", False))

    def _send_retry_uncertain_confirmation_enabled(self) -> bool:
        return bool(self.config.get("wechat", {}).get(
            "send_retry_uncertain_confirmation", False))

    def _send_retry_limit(self) -> int:
        return 1

    def _send_retry_delay_seconds(self, attempt: int) -> float:
        return 3.0

    def _send_retry_idle_delay_seconds(self) -> float:
        return 3.0

    def _report_has_sendable_reply(self, report: dict[str, Any]) -> bool:
        return bool(report.get("reply_draft", ""))

    @staticmethod
    def _send_guard_contact(report: dict[str, Any]) -> str:
        if not isinstance(report, dict):
            return "<unknown>"
        analysis = report.get("analysis", {})
        if isinstance(analysis, dict):
            contact = str(analysis.get("current_contact", "")).strip()
            if contact:
                return contact
        return str(report.get("current_contact", "")).strip() or "<unknown>"

    @classmethod
    def _send_guard_reidentify_signature(cls, report: dict[str, Any],
                                         send_result: dict[str, Any]) -> str:
        contact = cls._send_guard_contact(report)
        confirm = send_result.get("confirm", {}) if isinstance(send_result, dict) else {}
        customer_guard = confirm.get("customer_guard", {}) if isinstance(confirm, dict) else {}
        messages = customer_guard.get("new_messages", []) if isinstance(customer_guard, dict) else []
        if not isinstance(messages, list):
            return f"{contact}|unknown"

        parts = []
        for msg in messages[:6]:
            if not isinstance(msg, dict):
                continue
            text = str(msg.get("text", msg.get("content", msg.get("bubble_text", ""))))
            normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", text).lower().replace("_", "")[:12]
            side = str(msg.get("side", msg.get("sender", "")))
            if side:
                parts.append(f"{side}:{normalized}" if normalized else f"{side}:blank")
            else:
                parts.append(normalized if normalized else ":blank:")

        latest = ""
        analysis = report.get("analysis", {}) if isinstance(report, dict) else {}
        if isinstance(analysis, dict):
            latest_msg = analysis.get("latest_message", {})
            if isinstance(latest_msg, dict):
                latest = str(latest_msg.get("customer_turn_text", latest_msg.get("content", "")))
        if latest:
            _stripped = re.sub(r'[^\w\u4e00-\u9fff]+', '', latest).lower()[:12]
            parts.append(f"turn:{_stripped}")

        return f"{contact}|{'|'.join(parts)}"

    def _register_send_guard_reidentify(self, report: dict[str, Any],
                                        send_result: dict[str, Any]) -> None:
        now = time.time()
        ttl = self._send_guard_reidentify_ttl_seconds

        if not hasattr(self, "_send_guard_reidentify_signatures"):
            self._send_guard_reidentify_signatures = {}
        if not hasattr(self, "_send_guard_reidentify_contacts"):
            self._send_guard_reidentify_contacts = {}

        signature = self._send_guard_reidentify_signature(report, send_result)
        contact = self._send_guard_contact(report)

        self._send_guard_reidentify_signatures = {
            k: float(v) for k, v in self._send_guard_reidentify_signatures.items()
            if now - float(v) < ttl
        }
        self._send_guard_reidentify_signatures[signature] = now

        self._send_guard_reidentify_contacts = {
            c: [float(t) for t in timestamps if now - float(t) < ttl]
            if isinstance(timestamps, list) else []
            for c, timestamps in self._send_guard_reidentify_contacts.items()
        }
        self._send_guard_reidentify_contacts.setdefault(contact, []).append(now)

    def _reset_send_guard_reidentify(self, report: dict[str, Any]) -> None:
        contact = self._send_guard_contact(report)
        if not hasattr(self, "_send_guard_reidentify_contacts"):
            return
        if isinstance(self._send_guard_reidentify_contacts, dict):
            self._send_guard_reidentify_contacts.pop(contact, None)

        prefix = contact + "|"
        if hasattr(self, "_send_guard_reidentify_signatures") and isinstance(
                self._send_guard_reidentify_signatures, dict):
            to_remove = [k for k in self._send_guard_reidentify_signatures
                         if k.startswith(prefix)]
            for k in to_remove:
                self._send_guard_reidentify_signatures.pop(k, None)

    def _pending_send_wait_reason(self, report: dict[str, Any]) -> str:
        if not isinstance(report, dict):
            return ""
        retry_state = report.get("_retry_state", {})
        if not isinstance(retry_state, dict):
            return ""
        now = time.time()
        next_retry_at = self._float_value(retry_state.get("next_retry_at", 0))
        pending = self._float_value(self._pending_send_next_retry_at)
        next_at = max(next_retry_at, pending)
        if now < next_at:
            remaining = int(next_at - now) + 1
            return f"补发冷却中，约 {remaining} 秒后再试"
        return ""

    @staticmethod
    def _send_flow_status(send_result: dict[str, Any],
                          result: dict[str, Any]) -> str:
        if send_result.get("reidentify_loop_blocked"):
            return "failed"
        if send_result.get("force_current_chat_next_cycle"):
            return "stale_reply"
        if send_result.get("send_retry_queued"):
            return "skipped"
        if send_result.get("unconfirmed"):
            return "unconfirmed"
        if send_result.get("ok"):
            return "done"
        return "failed"

    @staticmethod
    def _send_flow_detail(send_result: dict[str, Any],
                          result: dict[str, Any]) -> str:
        if send_result.get("reidentify_loop_blocked"):
            return "发送前连续检测到同一条客户消息，已停止自动重识别，避免重复调用模型。"
        if send_result.get("force_current_chat_next_cycle"):
            return "客户又发新消息，旧回复已放弃；下轮重新识别。"
        if send_result.get("send_retry_queued"):
            return f"已加入后台重试队列：{send_result.get('reason', '')}"
        return ""

    @staticmethod
    def _mark_send_unconfirmed(send_result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(send_result, dict):
            return {"ok": False, "unconfirmed": True, "no_retry": True}
        send_result["ok"] = False
        send_result["unconfirmed"] = True
        send_result["no_retry"] = True
        send_result["original_reason"] = str(send_result.get("reason", ""))
        send_result["reason"] = "send_unconfirmed_no_retry"
        return send_result

    @staticmethod
    def _send_skip_reason(send_result: dict[str, Any]) -> str:
        if not isinstance(send_result, dict):
            return ""
        return str(send_result.get("reason", ""))

    @staticmethod
    def _send_confirm_summary(send_result: dict[str, Any]) -> str:
        if not isinstance(send_result, dict):
            return ""
        confirm = send_result.get("confirm", {})
        if not isinstance(confirm, dict):
            return ""
        return str(confirm.get("summary", ""))

    @staticmethod
    def _send_confirm_saw_chat_change(confirm: dict[str, Any]) -> bool:
        if not isinstance(confirm, dict):
            return False
        return bool(confirm.get("chat_changed", False))

    # ================================================================
    # 回复签名追踪
    # ================================================================

    def _reply_turn_text_for_signature(self, analysis: dict[str, Any],
                                       latest: dict[str, Any]) -> str:
        return str(latest.get("customer_turn_text", latest.get("content", "")))

    def _reply_turn_signature(self, contact: str, reply_text: str) -> str:
        contact = self._normalize_signature_text(contact)
        reply = self._normalize_signature_text(reply_text)
        return f"{contact}|{reply}"

    def _is_recent_reply_turn(self, signature: str) -> bool:
        return signature in self._recent_reply_signatures

    def _mark_reply_turn(self, signature: str) -> None:
        self._recent_reply_signatures[signature] = time.time()
        self._prune_recent_reply_signatures()

    def _unmark_reply_turn(self, signature: str) -> None:
        self._recent_reply_signatures.pop(signature, None)

    def _prune_recent_reply_signatures(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        ttl = self._recent_reply_signature_ttl_seconds
        self._recent_reply_signatures = {
            k: v for k, v in self._recent_reply_signatures.items()
            if now - v < ttl
        }

    def _no_reply_turn_signature(self, analysis: dict[str, Any]) -> str:
        contact = str(analysis.get("current_contact", ""))
        return self._normalize_signature_text(contact) + "|no_reply"

    def _is_recent_no_reply_turn(self, signature: str) -> bool:
        return signature in self._recent_no_reply_signatures

    def _mark_no_reply_turn(self, analysis: dict[str, Any]) -> None:
        signature = self._no_reply_turn_signature(analysis)
        self._recent_no_reply_signatures[signature] = time.time()
        self._prune_recent_no_reply_signatures()

    def _prune_recent_no_reply_signatures(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        ttl = self._recent_no_reply_signature_ttl_seconds
        self._recent_no_reply_signatures = {
            k: v for k, v in self._recent_no_reply_signatures.items()
            if now - v < ttl
        }

    @staticmethod
    def _normalize_signature_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text)).casefold()

    # ================================================================
    # 签名匹配
    # ================================================================

    @staticmethod
    def _split_turn_signature(signature: str) -> tuple[str, str]:
        parts = signature.split("|", 1)
        return (parts[0], parts[1] if len(parts) > 1 else "")

    @staticmethod
    def _signature_contacts_match(signature: str, contact: str) -> bool:
        sig_contact = ObserveService._split_turn_signature(signature)[0]
        return ObserveService._normalize_signature_text(sig_contact) == \
            ObserveService._normalize_signature_text(contact)

    # ================================================================
    # 当前聊天补处理
    # ================================================================

    def _try_current_chat_when_no_unread(self,
                                         unread: dict[str, Any],
                                         step: Optional[Callable[[str, str, str], None]] = None
                                         ) -> Optional[dict[str, Any]]:
        """对齐原版 v3.9.1：无未读时处理当前聊天。

        完整管道：
          1. 检查 force_current_chat / unread_allows 标记
          2. 跳过纯 nav_badge 红点（found=True 但未点击）
          3. stuck_nav 抑制 / 节流检查
          4. 截图当前聊天 → 画面变化检查
          5. 语音转文字 → 视觉分析
          6. 强制聊天联系人验证
          7. 消息去重检查

        返回完整分析报告，或 None（跳过本轮）。
        """
        # 1. 获取 force 标记
        force_current_chat = bool(getattr(self, '_force_current_chat_next_cycle', False))
        force_reason = str(getattr(self, '_force_current_chat_reason', '') or '').strip()
        force_target_contact = str(
            getattr(self, '_force_current_chat_target_contact', '') or '').strip()

        # 2. 检查是否允许处理当前聊天
        unread_allows = self._unread_result_allows_current_chat_check(unread)

        # 3. 跳过纯 nav_badge 红点
        #    条件：非强制、unread_allows=False、found=True、未点击、kind=nav_badge
        if not force_current_chat and not unread_allows:
            if (bool(unread.get("found")) and
                    not bool(unread.get("clicked")) and
                    str(unread.get("kind", "")) == "nav_badge"):
                self.store.append_log(
                    "current_chat_skip_nav_badge_only reason="
                    + str(unread.get("reason", ""))[:120]
                )
                return None
            return None

        # 4. stuck_nav 抑制检查（最近刚发送过，抑制当前聊天处理）
        if not force_current_chat and unread_allows:
            if self._should_suppress_stuck_nav_current_chat(unread):
                self.store.append_log(
                    "current_chat_skip_recent_send_nav_badge reason="
                    + str(unread.get("reason", ""))[:120]
                )
                return None

        # 5. stuck_nav 节流检查（冷却时间内）
        if not force_current_chat and unread_allows:
            if self._should_throttle_stuck_nav_current_chat():
                self.store.append_log(
                    "current_chat_skip_stuck_nav_cooldown reason="
                    + str(unread.get("reason", ""))[:120]
                )
                return None

        # 6. 标记 stuck_nav 已检查
        if not force_current_chat and unread_allows:
            self._mark_stuck_nav_current_chat_checked()
            self.store.append_log(
                "current_chat_from_unread_stuck kind="
                + str(unread.get("kind", ""))
                + " reason=" + str(unread.get("reason", ""))[:120]
            )

        # 7. 清除 force 标记
        if force_current_chat:
            self._force_current_chat_next_cycle = False
            self._force_current_chat_reason = ""
            self._force_current_chat_target_contact = ""

        # 8. 截图当前聊天
        try:
            self.capture_once(subdir="current_watch", prefix="current_chat")
        except TypeError:
            self.capture_once()

        if not self.last_image:
            self.store.append_log("current_chat_no_unread_failed error=capture_failed")
            return None

        # 9. 画面变化检查（非强制模式下，画面未变化则跳过）
        if not force_current_chat and not unread_allows:
            if not self._current_chat_visual_changed(str(self.last_image)):
                self.store.append_log("current_chat_no_unread_skip reason=visual_unchanged")
                return None

        # 10. 语音转文字
        voice_result = self._maybe_convert_voice_to_text(self.last_image)
        if voice_result and voice_result.get("converted"):
            self.store.append_log(
                "current_chat_voice_to_text converted=True text="
                + str(voice_result.get("converted_text", ""))[:80]
            )
            self.capture_once()

        # 11. 视觉分析
        report = self.analyze_once(stop_before_reply_for_voice=True)
        analysis = report.get("analysis", {}) if isinstance(report, dict) else {}

        # 语音二次处理
        if self._analysis_latest_is_voice(analysis):
            voice_result = self._maybe_convert_voice_to_text(analysis)
            if voice_result and voice_result.get("converted"):
                self.store.append_log(
                    "current_chat_voice_to_text converted=True text="
                    + str(voice_result.get("converted_text", ""))[:80]
                )
                self.capture_once()
            report = self.analyze_once(force_new_capture=True)
            analysis = report.get("analysis", {}) if isinstance(report, dict) else {}

        # 12. 强制聊天时检查联系人是否匹配
        if force_current_chat and force_target_contact:
            actual_contact = str(analysis.get("current_contact", "")).strip()
            if not self._contact_name_matches(actual_contact, force_target_contact):
                self.store.append_log(
                    "forced_current_chat_contact_mismatch expected="
                    + force_target_contact
                    + " actual=" + (actual_contact or "<unknown>")
                )
                return None

        # 13. 消息去重检查
        if self._current_chat_can_process():
            sig = self._current_chat_signature()
            if sig and self._same_current_chat_signature(sig):
                self.store.append_log("当前客户消息已处理过")
                return None

        # 14. 返回完整报告
        latest = str(analysis.get("latest_message", "")
                     or analysis.get("content", ""))[:40]
        self.store.append_log(
            "current_chat_no_unread_process contact="
            + str(analysis.get("current_contact", ""))
            + " latest=" + latest
        )
        return report

    @classmethod
    def _contact_name_matches(cls, contact: str, candidates: Any) -> bool:
        """对齐原版 v3.10：联系人名称模糊匹配。

        原版签名 (cls, contact, candidates)，candidates 可为单个名称或名称列表，
        使用 SequenceMatcher 相似度 >= 0.66 判定匹配。
        """
        contact_norm = cls._contact_match_text(contact)
        if not contact_norm:
            return False
        if isinstance(candidates, str):
            candidates = [candidates]
        best = 0.0
        for item in (candidates or []):
            item_norm = cls._contact_match_text(item)
            if not item_norm:
                continue
            if item_norm == contact_norm:
                return True
            ratio = difflib.SequenceMatcher(None, contact_norm, item_norm).ratio()
            if ratio > best:
                best = ratio
        return best >= 0.66

    def _remember_current_chat_visual(self, image_path: str) -> None:
        self._last_current_chat_visual_image = image_path

    def _current_chat_visual_changed(self, image_path: str) -> bool:
        if not self._last_current_chat_visual_image:
            return True
        return image_path != self._last_current_chat_visual_image

    def _cleanup_transient_current_chat_image(self) -> None:
        self._last_current_chat_visual_image = None

    def _unread_result_allows_current_chat_check(self, unread: dict[str, Any]) -> bool:
        if not isinstance(unread, dict):
            return False
        if unread.get("found") is True:
            return False
        return True

    def _suppress_stuck_nav_after_send(self, send_result: dict[str, Any]) -> None:
        if not self._is_stuck_nav_unread(self._last_unread_result or {}):
            return
        self._suppress_stuck_nav_until = time.time() + 3.0
        self._suppress_stuck_nav_reason = "stuck_nav_after_send"

    def _should_suppress_stuck_nav_current_chat(self, unread: dict[str, Any]) -> bool:
        if not self._is_stuck_nav_unread(unread):
            return False
        if time.time() < self._suppress_stuck_nav_until:
            return True
        return False

    def _should_throttle_stuck_nav_current_chat(self, unread: Any = None) -> bool:
        if time.time() < self._stuck_nav_current_chat_next_allowed_at:
            return True
        return False

    def _mark_stuck_nav_current_chat_checked(self, unread: Any = None) -> None:
        # 对齐原版：冷却时间取自配置，默认 12.0 秒
        try:
            cfg = self.config.get("assistant", {})
            cooldown = float(cfg.get("stuck_nav_current_chat_cooldown_seconds", 12.0))
        except Exception:
            cooldown = 12.0
        self._stuck_nav_current_chat_next_allowed_at = time.time() + cooldown

    @staticmethod
    def _is_stuck_nav_unread(unread: dict[str, Any]) -> bool:
        if not isinstance(unread, dict):
            return False
        return str(unread.get("kind", "")) == "nav_badge"

    def _current_chat_can_process(self) -> bool:
        return bool(self._last_current_chat_visual_image)

    def _current_chat_signature(self) -> str:
        return self._last_current_chat_signature or ""

    def _same_current_chat_signature(self, sig: str, other: Optional[str] = None) -> bool:
        """对齐原版 v3.10 (cls, left, right)：OCR 容差比较两签名是否同一客户轮。"""
        if other is None:
            other = self._last_current_chat_signature
        if not sig or not other:
            return False
        if sig == other:
            return True
        left_contact, left_text = self._split_turn_signature(sig)
        right_contact, right_text = self._split_turn_signature(other)
        if not self._signature_contacts_match(sig, right_contact):
            return False
        return bool(looks_like_same_ocr_text(left_text, right_text))

    def _analysis_latest_is_voice(self, analysis: dict[str, Any]) -> bool:
        if not isinstance(analysis, dict):
            return False
        return str(analysis.get("intent", "")) == "voice_message"

    def _maybe_accept_friend_requests(self, window: Any = None) -> Optional[dict[str, Any]]:
        # 设置页「自动通过好友申请」总开关（friend_requests.auto_accept），
        # 关闭时完全跳过检查（此前该开关无消费者，恒为开启行为）。
        if not bool(self.config.get("friend_requests", {}).get("auto_accept", True)):
            return None
        now = time.time()
        interval = float(self.config.get("friend_requests", {}).get("check_interval_seconds", 30))
        if now - self._last_friend_request_check_at < interval:
            return None
        self._last_friend_request_check_at = now
        return self.accept_friend_requests()

    def _maybe_convert_voice_to_text(self, image_path: Any) -> Optional[dict[str, Any]]:
        if image_path is None:
            return None
        try:
            handle = 0
            if self.last_window:
                handle = int(self.last_window.get("handle", 0))
            if not handle:
                return None
            result = self.voice_converter.convert_voice(handle)
            if result:
                self._last_voice_to_text_result = {"converted": True, "text": result}
                return self._last_voice_to_text_result
        except Exception:
            pass
        return None

    def _persist(self, report: Any = None, send_result: Any = None,
                 result: Any = None) -> None:
        # 兼容 my_agent 旧调用：只传 cycle result 作为第一个位置参数
        if result is None and isinstance(report, dict) and (
            "flow" in report or "idle" in report or "contact" in report
        ):
            result = report
        if result is None:
            result = {}
        if self._should_skip_cycle_persist(result):
            return
        try:
            contact = str(result.get("contact", "unknown"))
            status = "ok" if result.get("ok") else "error"
            summary = str(result.get("report", {}).get("reply_draft", ""))[:200]
            self.cycles_repo.log_cycle(contact, status=status, summary=summary)
        except Exception:
            pass

    def _should_skip_cycle_persist(self, analysis: Any = None,
                                   report: Any = None,
                                   send_result: Any = None) -> bool:
        """对齐原版 v3.10 (self, analysis, report, send_result)。

        返回 bool，与 my_agent 现有 _persist 调用兼容；同时兼容只传 cycle result 的旧调用。
        """
        # 兼容 my_agent 旧调用：把 cycle result 作为第一个位置参数传入
        if analysis is not None and report is None and isinstance(analysis, dict) and (
            "idle" in analysis or "flow" in analysis or "error" in analysis
        ):
            result = analysis
            if result.get("idle"):
                return True
            if result.get("error"):
                return True
            return False

        if analysis is None:
            return False
        if not isinstance(analysis, dict):
            return False

        intent = str(analysis.get("intent", ""))
        if intent in self._INTERNAL_NO_REPLY_LOG_INTENTS:
            return True

        contact = str(analysis.get("current_contact", "")).strip()
        reply = str(analysis.get("reply_draft", "")).strip()
        if reply:
            signature = self._reply_turn_signature(contact, reply)
            if self._is_recent_reply_turn(signature):
                return True

        decision = str(analysis.get("decision", "")) or str(analysis.get("action", ""))
        if decision == "no_reply":
            signature = self._no_reply_turn_signature(analysis)
            if self._is_recent_no_reply_turn(signature):
                return True
        return False

    # ================================================================
    # 分析辅助方法
    # ================================================================

    def _update_vision_model_outcome(self, analysis: dict[str, Any],
                                     reply: str) -> None:
        try:
            update_model_call_outcome(analysis, reply)
        except Exception:
            pass

    def _repair_unknown_latest_sender_from_ocr_guard(self,
                                                      analysis: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(analysis, dict):
            return analysis
        return analysis

    def _repair_customer_text_blocked_as_system(self,
                                                 analysis: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(analysis, dict):
            return analysis
        return analysis

    def _is_single_call_vision_analysis(self, analysis: dict[str, Any]) -> bool:
        if not isinstance(analysis, dict):
            return False
        intent = str(analysis.get("intent", ""))
        return intent not in self._INTERNAL_NO_REPLY_LOG_INTENTS

    def _requires_authoritative_business_basis(self, analysis: dict[str, Any],
                                                reply: str) -> bool:
        if not reply:
            return False
        wechat_cfg = self.config.get("wechat", {})
        return bool(wechat_cfg.get("evidence_gate_enabled", False))

    def _apply_auto_send_protection(self, analysis: dict[str, Any],
                                     reply: str) -> dict[str, Any]:
        contact = str(analysis.get("current_contact", ""))
        raw_confidence = analysis.get("confidence", 0)
        try:
            confidence = float(raw_confidence) if not isinstance(raw_confidence, dict) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._MIN_RECOGNITION_CONFIDENCE:
            return {"ok": False, "reason": "low_confidence"}
        return {"ok": True, "reason": ""}

    def _analysis_timing(self, analysis: dict[str, Any],
                         started: float, keyword_fast_path: bool) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        analysis["debug_timing"] = {
            "analyze_ms": int(elapsed * 1000),
            "keyword_fast_path": keyword_fast_path,
        }
        return analysis

    def _apply_keyword_reply(self, analysis: dict[str, Any]) -> dict[str, Any]:
        try:
            text = str(analysis.get("latest_message", {}).get("content", "") or "")
            if not text:
                text = str(analysis.get("recognized_text", "") or "")
            result = self.keyword_reply.match(text)
            if result and getattr(result, "matched", False):
                return {
                    "keyword_reply_matched": True,
                    "reply": getattr(result, "reply", ""),
                    "intent_category": getattr(result, "intent_category", ""),
                }
            return {"keyword_reply_matched": False, "reply": ""}
        except Exception:
            return {"keyword_reply_matched": False, "reply": ""}

    def _apply_faq_reply(self, analysis: dict[str, Any]) -> dict[str, Any]:
        try:
            text = str(analysis.get("latest_message", {}).get("content", "") or "")
            if not text:
                text = str(analysis.get("recognized_text", "") or "")
            result = self.faq_reply.match(text)
            if result and getattr(result, "matched", False):
                return {
                    "faq_reply_matched": True,
                    "reply": getattr(result, "answer", ""),
                    "question": getattr(result, "question", ""),
                    "confidence": getattr(result, "confidence", 0.0),
                }
            return {"faq_reply_matched": False, "reply": ""}
        except Exception:
            return {"faq_reply_matched": False, "reply": ""}

    def _recover_wechat_voice_input_screen(self,
                                            analysis: dict[str, Any]) -> dict[str, Any]:
        analysis["intent"] = "wechat_voice_input_screen"
        analysis["decision"] = "no_reply"
        return analysis

    def _should_retry_unread_text(self, analysis: dict[str, Any]) -> bool:
        if not isinstance(analysis, dict):
            return False
        intent = str(analysis.get("intent", ""))
        return intent == "ocr_message_unstable"

    def _retry_unread_text_analysis(self, analysis: dict[str, Any],
                                    window_info: Any = None) -> dict[str, Any]:
        wechat_cfg = self.config.get("wechat", {})
        max_retries = int(wechat_cfg.get("unread_text_retry_limit", 1))
        retries = int(analysis.get("_unread_text_retries", 0))
        if retries >= max_retries:
            return analysis
        analysis["_unread_text_retries"] = retries + 1
        return analysis

    def _analysis_has_customer_text(self, analysis: dict[str, Any]) -> bool:
        if not isinstance(analysis, dict):
            return False
        return bool(analysis.get("customer_turn_text", ""))

    def _mark_unread_text_unreadable(self, analysis: dict[str, Any]) -> None:
        analysis["_unread_text_unreadable"] = True

    def _build_unread_fallback_reply(self,
                                      analysis: dict[str, Any]) -> str:
        return ""

    # ================================================================
    # 缺失方法实现（来自字节码骨架，对齐原版 API 表面）
    # 以下方法均未被主循环直接调用，实现以保证 API 完整、避免 AttributeError，
    # 并尽量复用 my_agent 已有 helper 与从反汇编恢复的真实常量。
    # ================================================================

    # ---- 消息文本工具 ----

    @staticmethod
    def _message_text(message: Any) -> str:
        """对齐原版 (message)：提取消息文本（content/text/bubble_text）。"""
        if not isinstance(message, dict):
            return "" if message is None else str(message)
        return str(message.get("content") or message.get("text")
                  or message.get("bubble_text") or "").strip()

    @staticmethod
    def _same_message_text(left: Any, right: Any) -> bool:
        """对齐原版 (left, right)：归一化比较两条消息文本。"""

        def norm(value: Any) -> str:
            if isinstance(value, dict):
                return ObserveService._normalize_signature_text(
                    str(value.get("value", value.get("str", value.get("return", ""))))
                )
            return ObserveService._normalize_signature_text(str(value))

        return norm(left) == norm(right)

    @staticmethod
    def _contact_match_text(value: Any) -> str:
        """对齐原版 (value)：联系人归一化（去空白/小写/去末尾括号编号）。"""
        if value is None:
            return ""
        text = str(value).strip().casefold()
        text = re.sub(r"[(（]\s*\d+[)）]$", "", text)
        text = re.sub(r"\s+", "", text)
        return text

    # ---- 回复依据/缺失 ----

    def _block_missing_reply_basis(self, analysis: Any, _opt1: Any = None,
                                   _opt2: Any = None) -> Any:
        """对齐原版 (analysis, *opts)：缺少回复依据时阻断自动回复。"""
        if isinstance(analysis, dict):
            analysis["decision"] = "no_reply"
            analysis["action"] = "no_reply"
            analysis["should_reply"] = False
            analysis["reply_draft"] = ""
            analysis["reason"] = "missing_reply_basis"
        return analysis

    @staticmethod
    def _safe_result_bool(value: Any, default: Any = False) -> bool:
        """对齐原版 (value, default)：将各类阈值/开关字符串安全转为 bool。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return bool(default)
        normalized = str(value).strip().lower()
        if normalized in {"y", "1", "完整", "yes", "true", "是"}:
            return True
        if normalized in {"", "no", "0", "false", "不完整", "否", "n"}:
            return False
        return bool(default)

    @staticmethod
    def _is_hard_no_reply(analysis: Any) -> bool:
        """对齐原版 (analysis)：硬性不回复意图判定。"""
        if not isinstance(analysis, dict):
            return False
        return str(analysis.get("intent", "")) in ObserveService._HARD_NO_REPLY_INTENTS

    # ---- FAQ / 关键词查询 ----

    def _faq_matches_for_queries(self, queries: Any) -> list[Any]:
        """对齐原版 (self, queries)：按查询列表匹配 FAQ 并去重/过滤纯问候。"""
        matcher = self.faq_reply
        raw_matches: list[Any] = []
        seen: set[str] = set()
        for query in (queries or []):
            try:
                m = matcher.match(query)
            except Exception:
                m = None
            if not m or not getattr(m, "matched", False):
                continue
            key = str(getattr(m, "question", "") or "")
            if key in seen:
                continue
            seen.add(key)
            if self._is_greeting_only_query(query):
                continue
            raw_matches.append(m)
        matches: list[Any] = []
        for m in raw_matches:
            reply = getattr(m, "answer", "")
            if reply:
                matches.append(m)
        return matches

    @staticmethod
    def _customer_turn_queries(analysis: Any) -> list[str]:
        """对齐原版 (analysis)：将客户本轮文本拆成多个查询子句。"""
        if not isinstance(analysis, dict):
            return []
        latest = analysis.get("latest_message") or {}
        if isinstance(latest, dict):
            text = str(latest.get("customer_turn_text") or latest.get("content") or "")
        else:
            text = str(analysis.get("content") or "")
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"[\r\n/／|]+|(?<=[？?！!。；;])", text)
        out: list[str] = []
        for raw in parts:
            part = re.sub(r" \t\r\n,，。！？!?；;：:、", "", raw).strip()
            if part:
                out.append(ObserveService._normalize_signature_text(part))
        return out

    @staticmethod
    def _is_greeting_only_query(value: Any) -> bool:
        """对齐原版 (value)：是否为纯问候（无实质业务内容）。"""
        text = re.sub(
            r"""[\s，,。！？!?；;：:、~～\-_（）()【】\[\]"'"'“”‘’/／|]+""",
            "", str(value),
        ).casefold()
        if not text:
            return False
        return bool(re.fullmatch(r"(老板|客服|你好|您好|在吗|在不在|有人吗|哈喽|hello|hi)", text))

    @staticmethod
    def _query_signature_key(value: Any) -> str:
        """对齐原版 (value)：查询签名键。"""
        return ObserveService._normalize_signature_text(value)

    def _filter_keyword_matches_for_query(self, query: Any, matches: Any) -> list[Any]:
        """对齐原版 (self, query, matches)：按查询过滤关键词匹配（推断实现）。"""
        if not matches:
            return []
        compact = re.sub(r"\s+", "", str(query)).casefold()
        if not compact:
            return list(matches)
        filtered: list[Any] = []
        for m in matches:
            try:
                keys_fn = getattr(m, "keys", None)
                if callable(keys_fn):
                    keys = [str(k).casefold() for k in (keys_fn() or [])]
                else:
                    keys = [str(getattr(m, "keyword", "")).casefold()]
            except Exception:
                keys = []
            if not keys:
                filtered.append(m)
                continue
            if any(k and k in compact for k in keys):
                filtered.append(m)
        return filtered

    def _preserve_evidence_backed_model_reply(self, analysis: Any,
                                              queries: Any = None,
                                              matches: Any = None,
                                              model_selected_keyword: Any = None,
                                              evidence: Any = None) -> tuple[Any, Any]:
        """对齐原版 (self, analysis, queries, matches, model_selected_keyword, evidence)。

        若模型回复有证据支撑，则保留模型回复，避免被通用关键词回复覆盖（推断实现）。
        """
        if not isinstance(analysis, dict):
            return analysis, matches
        if self._is_single_call_vision_analysis(analysis):
            return analysis, matches
        return analysis, matches

    # ---- 视觉回复文本精炼 ----

    @classmethod
    def _referential_follow_up_needs_memory(cls, analysis: Any) -> bool:
        """对齐原版 (cls, analysis)：指代性追问是否需要会话记忆。"""
        if not isinstance(analysis, dict):
            return False
        memory_loaded = int(analysis.get("conversation_memory_loaded", 0) or 0)
        if not memory_loaded:
            return False
        latest = analysis.get("latest_message") or {}
        if not isinstance(latest, dict):
            return False
        text = re.sub(
            r"\s+", "",
            str(latest.get("customer_turn_text") or latest.get("content")
                or latest.get("text") or ""),
        ).lower()
        if not text:
            return False
        referential_terms = {"还是呢", "那呢", "刚才那个呢", "然后呢", "这个呢", "那个呢"}
        if text in referential_terms:
            return True
        return bool(re.match(
            r"^(?:那|这个|那个|刚才|前面|之前|还是|然后|所以|上次|昨天|我不是说)", text))

    @staticmethod
    def _distinct_customer_message_count(messages: Any) -> int:
        """对齐原版 (messages)：不同客户消息条数。"""
        if not isinstance(messages, list):
            return 0
        seen: set[str] = set()
        count = 0
        for item in messages:
            if not isinstance(item, dict):
                continue
            sender = str(item.get("sender") or item.get("side") or "")
            if sender not in {"", "customer"}:
                continue
            text = ObserveService._normalize_signature_text(
                str(item.get("text") or item.get("content") or item.get("bubble_text") or ""))
            if not text:
                continue
            if text not in seen:
                seen.add(text)
                count += 1
        return count

    @staticmethod
    def _latest_customer_turn_count(messages: Any) -> int:
        """对齐原版 (messages)：从末尾起连续客户消息条数。"""
        if not isinstance(messages, list):
            return 0
        customer_senders = {"", "customer", "self", "left"}
        count = 0
        for item in reversed(messages):
            if not isinstance(item, dict):
                break
            sender = str(item.get("sender") or item.get("side") or "")
            if sender not in customer_senders:
                break
            text = ObserveService._normalize_signature_text(
                str(item.get("text") or item.get("content") or item.get("bubble_text") or ""))
            if not text:
                continue
            count += 1
        return count

    @classmethod
    def _vision_reply_needs_text_refine(cls, analysis: Any) -> bool:
        """对齐原版 (cls, analysis)：视觉回复是否需要文本精炼。"""
        if not isinstance(analysis, dict):
            return False
        if cls._referential_follow_up_needs_memory(analysis):
            return True
        latest = analysis.get("latest_message") or {}
        if isinstance(latest, dict) and str(latest.get("sender", "")) == "customer":
            visible = analysis.get("messages") or []
            if cls._distinct_customer_message_count(visible) >= 2:
                return True
            queries = cls._customer_turn_queries(analysis)
            if queries and cls._latest_customer_turn_count(visible) >= 2:
                return True
        return False

    @classmethod
    def _prepare_vision_text_refine_analysis(cls, analysis: Any) -> Any:
        """对齐原版 (cls, analysis)：准备记忆精炼标记。"""
        if not isinstance(analysis, dict):
            return analysis
        vision_reply = str(analysis.get("reply_draft", "") or "")
        messages = analysis.get("messages") or []
        if cls._distinct_customer_message_count(messages) >= 2:
            analysis["conversation_memory_refine"] = True
            analysis["vision_reply_draft_before_memory_refine"] = vision_reply
        return analysis

    @classmethod
    def _prepare_multi_turn_text_refine_analysis(cls, analysis: Any) -> Any:
        """对齐原版 (cls, analysis)：多消息轮标记，强制从客户文本作答。"""
        if not isinstance(analysis, dict):
            return analysis
        vision_reply = str(analysis.get("reply_draft", "") or "")
        messages = analysis.get("messages") or []
        if cls._distinct_customer_message_count(messages) >= 2:
            analysis["multi_message_customer_turn"] = True
            analysis["vision_reply_draft_before_multi_turn_refine"] = vision_reply
        return analysis

    @staticmethod
    def _resolve_vision_refine_reply(analysis: Any, text_reply: Any,
                                     text_meta: Any) -> tuple[Any, bool]:
        """对齐原版 (analysis, text_reply, text_meta)：解析精炼后的回复。"""
        if not isinstance(analysis, dict):
            analysis = {}
        candidate = str(text_reply or "").strip()
        if candidate:
            analysis["reply_draft"] = candidate
            return analysis, False
        technical_failure = bool(text_meta.get("error")) if isinstance(text_meta, dict) else False
        stale_vision_reply = bool(analysis.get("conversation_memory_refine"))
        if stale_vision_reply and not technical_failure:
            analysis["reply_draft"] = str(
                analysis.get("vision_reply_draft_before_memory_refine")
                or analysis.get("reply_draft") or "")
        return analysis, stale_vision_reply

    @staticmethod
    def _has_blocking_vision_error(analysis: Any) -> bool:
        """对齐原版 (analysis)：是否存在阻断性视觉错误。"""
        if not isinstance(analysis, dict):
            return False
        if analysis.get("vision_error"):
            return True
        if analysis.get("vision_fallback_used"):
            return True
        latest = analysis.get("latest_message") or {}
        if isinstance(latest, dict):
            text = str(latest.get("content") or latest.get("text")
                      or latest.get("customer_turn_text") or "").strip()
            if not text:
                return True
        return False

    # ---- 回复去重 ----

    @staticmethod
    def _dedupe_reply_lines(values: Any) -> Any:
        """对齐原版 (values)：对回复行去重。"""
        if not isinstance(values, (list, tuple)):
            return values
        seen: set[str] = set()
        out: list[Any] = []
        for value in values:
            text = str(value).replace("\r\n", "\n").replace("\r", "\n")
            key = ObserveService._normalize_signature_text(text)
            if key and key not in seen:
                seen.add(key)
                out.append(value)
        return out

    # ---- 测试联系人锁 ----

    @staticmethod
    def _clean_contact_list(raw: Any) -> list[str]:
        """对齐原版 (raw)：清洗联系人列表（支持中英文分隔符）。"""
        if isinstance(raw, str):
            parts = raw.replace("，", ",").replace("、", ",").split(",")
        elif isinstance(raw, (list, tuple)):
            parts = []
            for line in raw:
                parts.extend(str(line).replace("，", ",").replace("、", ",").split(","))
        else:
            return []
        return [p.strip() for p in parts if p and p.strip()]

    def _apply_test_contact_lock(self, analysis: Any) -> Any:
        """对齐原版 (self, analysis)：对测试联系人施加锁。"""
        whitelist = self._clean_contact_list(
            self.config.get("wechat", {}).get("test_contact_whitelist", [])
        )
        out = analysis if isinstance(analysis, dict) else {}
        contact = str(out.get("current_contact", "")).strip()
        if whitelist and contact and not self._contact_name_matches(contact, whitelist):
            out["test_contact_locked"] = True
            out.setdefault("reason", "test_contact_not_allowed")
            self._append_runtime_log("test_contact_lock applied contact=" + contact)
        return out

    def _test_contact_lock_blocks(self, analysis: Any) -> bool:
        """对齐原版 (self, analysis)：测试联系人锁是否阻断。"""
        if not isinstance(analysis, dict):
            return False
        whitelist = self._clean_contact_list(
            self.config.get("wechat", {}).get("test_contact_whitelist", [])
        )
        if "*" in whitelist:
            return False
        contact = str(analysis.get("current_contact", "")).strip()
        if not contact:
            return False
        return not self._contact_name_matches(contact, whitelist)

    # ---- 发送结果判断补丁 ----

    @staticmethod
    def _send_result_pre_enter_new_customer_blocked(send_result: Any) -> bool:
        """对齐原版 (send_result)：进入前是否因新客户消息而阻断。"""
        if not isinstance(send_result, dict):
            return False
        reason = str(send_result.get("reason", "")).lower()
        if ("send_blocked_new_customer_message_before_enter" in reason
                or "send_blocked_model_uncertain_before_enter" in reason):
            return True
        confirm = send_result.get("confirm") or {}
        if isinstance(confirm, dict):
            customer_guard = confirm.get("customer_guard") or {}
            if isinstance(customer_guard, dict) and customer_guard.get("has_new_customer_message"):
                return True
        return False

    def _log_ocr_variant_dedupe(self, scope: Any, old_text: Any, new_text: Any) -> None:
        """对齐原版 (self, scope, old_text, new_text)：记录 OCR 变体去重。"""
        try:
            store = getattr(self, "store", None)
            if store is None:
                return
            append_log = getattr(store, "append_log", None)
            if append_log is None:
                return
            append_log(
                "ocr_variant_deduped scope=" + str(scope)
                + " old=" + str(old_text)[:80]
                + " new=" + str(new_text)[:80]
            )
        except Exception:
            pass

    def _attach_recent_conversation_memory(self, analysis: Any) -> Any:
        """对齐原版 (self, analysis)：附加近期会话记忆到分析。"""
        if not isinstance(analysis, dict):
            return analysis
        contact = str(analysis.get("current_contact", "")).strip()
        if not contact:
            return analysis
        try:
            limit = int(self.config.get("wechat", {}).get("conversation_memory_limit", 20))
        except Exception:
            limit = 20
        limit = max(1, min(limit, 200))
        try:
            repo = getattr(self, "messages_repo", None)
            rows = []
            if repo is not None and hasattr(repo, "recent_for_contact"):
                rows = repo.recent_for_contact(contact, limit=limit)
            stored_context: list[dict[str, Any]] = []
            for row in reversed(rows or []):
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content") or row.get("text") or "")
                sender = str(row.get("sender") or "")
                if content:
                    stored_context.append({"sender": sender, "text": content})
            if stored_context:
                analysis["conversation_memory"] = stored_context
                analysis["conversation_memory_loaded"] = 1
        except Exception as e:
            self._append_runtime_log("attach_recent_conversation_memory error=" + str(e))
        return analysis

    # ================================================================
    # 主循环生命周期
    # ================================================================

    def start_loop(self) -> None:
        """启动后台自动化循环线程。"""
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop_requested = False
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._append_runtime_log("loop_started")
        self._emit_event("status", {"state": "running", "message": "助手已启动"})

    def stop_loop(self) -> None:
        """停止后台自动化循环线程。

        协作式停止：先置 _stop_requested，run_one_cycle 内部在截图/分析/发送前
        会检查该标志并尽快收尾。这里只负责等待线程真正退出，绝不在退出前把
        _loop_thread 置空——否则 is_running() 会误报"已停止"而线程仍在跑
        （表现为"关了还在动"）。线程真正退出后由 _run_loop 的 finally 清理引用。
        """
        self._emit_event("status", {"state": "stopping", "message": "正在停止…"})
        self.request_stop()
        th = self._loop_thread
        if th is not None:
            # 当前轮会在 _stop_requested 检查点快速收尾，最多等 5s 足够
            th.join(timeout=5)
            if th.is_alive():
                self._append_runtime_log("loop_stop_pending: cycle still finishing")
        self._append_runtime_log("loop_stopped")
        self._emit_event("status", {"state": "stopped", "message": "助手已停止"})

    def _run_loop(self) -> None:
        """后台循环：反复执行 run_one_cycle。"""
        try:
            while not self._stop_requested:
                try:
                    self.apply_pending_config_reload()
                    # 每轮重新读取循环节奏，使 UI 调整的 idle/cycle 间隔即时生效
                    idle_sleep = float(self.config.get("wechat", {}).get("idle_sleep_seconds", 2.0))
                    cycle_sleep = float(self.config.get("wechat", {}).get("cycle_sleep_seconds", 0.5))
                    result = self.run_one_cycle()

                    # 对齐 run.py offscreen：把「双击置顶进入会话」细节透传给 UI 日志，
                    # 否则打包版 GUI 只能看到「跳过 X: 无可用回复内容」，看不出具体在干嘛。
                    pin = result.get("pin_echo")
                    if pin:
                        self._emit_event("pin_echo", pin)
                    # 周期小结（对齐 run.py wrapped_cycle 的回显）：非空闲轮才输出，
                    # 空闲轮已由 status(idle) 事件呈现，避免刷屏。
                    if not result.get("idle"):
                        self._emit_event("cycle_summary", {
                            "contact": result.get("contact") or "",
                            "error": result.get("error") or "",
                            "flow": [{"name": s.get("name"), "status": s.get("status")}
                                     for s in (result.get("flow") or [])],
                            "send_ok": bool((result.get("send_result") or {}).get("ok")),
                        })

                    if result.get("paused"):
                        break

                    if result.get("idle"):
                        self._emit_event("status", {"state": "idle", "message": "等待新消息"})
                        # 空闲时把微信推回虚拟外屏，恢复后台隐藏托管。
                        # 守卫：用户正在使用微信（窗口前台激活）时不移走，
                        # 否则会出现"窗口消失→点回→又消失"的闪烁。
                        try:
                            wm = self.window_manager
                            if wm is not None and not wm.is_foreground():
                                wm.move_offscreen()
                        except Exception:
                            pass
                        self._interruptible_sleep(idle_sleep)
                    else:
                        self._interruptible_sleep(cycle_sleep)

                except Exception as e:
                    self._append_runtime_log(f"loop_error={e}")
                    self._emit_event("error", {"message": str(e)})
                    self._interruptible_sleep(idle_sleep)
        finally:
            # 线程已退出：清理引用，确保 is_running() 最终返回 False。
            # 必须在置 _stop_requested 后、线程真正消亡前保持引用，否则 stop_loop
            # 拿不到存活状态；这里统一在退出时清理。
            self._loop_thread = None

    def _interruptible_sleep(self, seconds: float) -> None:
        """可被停止请求中断的 sleep。"""
        end = time.time() + seconds
        while time.time() < end and not self._stop_requested:
            time.sleep(0.1)

    # ================================================================
    # 无回复熔断（防"同一条消息反复烧 LLM"的死循环）
    # ================================================================

    def _no_reply_cfg(self) -> tuple:
        """读取熔断阈值与冷却时长（可在 config.yaml 的 wechat 段调整）。"""
        wc = self.config.get("wechat", {}) if isinstance(self.config, dict) else {}
        # 默认值与当前发布配置保持一致（2 / 120s）：
        # 增量更新不会覆盖用户 config，旧版 config 缺这两个键时也必须拿到收紧后的值，
        # 否则「对同一会话反复 OCR+LLM 空转」的修复在老机器上不生效。
        try:
            threshold = int(wc.get("no_reply_skip_threshold", 2))
        except Exception:
            threshold = 2
        try:
            cooldown = float(wc.get("no_reply_skip_cooldown_seconds", 120))
        except Exception:
            cooldown = 120.0
        return max(2, threshold), max(60.0, cooldown)

    def _note_no_reply(self, contact: str) -> None:
        """记录一次"无回复可发"；连续达到阈值后把该联系人放进冷却期。"""
        key = str(contact or "").strip()
        if not key:
            return
        threshold, cooldown = self._no_reply_cfg()
        streak = int(self._no_reply_streak.get(key, 0)) + 1
        self._no_reply_streak = {key: streak}  # 只跟踪最近一个联系人，避免无限增长
        self._last_no_reply_contact = key
        if streak >= threshold:
            self._no_reply_skip_until[key] = time.time() + cooldown
            self.store.append_log(
                f"no_reply_quarantine contact={key!r} streak={streak} "
                f"cooldown={int(cooldown)}s")

    def _clear_no_reply(self, contact: str = "") -> None:
        """成功产出回复后清零连续无回复计数（并解除该联系人冷却）。"""
        key = str(contact or "").strip()
        self._no_reply_streak = {}
        if key:
            self._no_reply_skip_until.pop(key, None)

    def _is_no_reply_quarantined(self) -> bool:
        """上一轮联系人是否处于"连续无回复"冷却期内。"""
        key = str(self._last_no_reply_contact or "").strip()
        if not key:
            return False
        until = float(self._no_reply_skip_until.get(key, 0) or 0)
        if until <= 0:
            return False
        if time.time() >= until:
            # 冷却结束：释放，允许再试一次
            self._no_reply_skip_until.pop(key, None)
            self._no_reply_streak = {}
            return False
        return True

    # ================================================================
    # 事件回调系统
    # ================================================================

    def on_event(self, callback: Callable[[str, dict], None]) -> None:
        """注册事件回调。callback(event_type: str, data: dict)。"""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks: list[Callable[[str, dict], None]] = []
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Optional[dict[str, Any]] = None) -> None:
        """触发所有注册的事件回调。"""
        callbacks = getattr(self, "_event_callbacks", None)
        if not callbacks:
            return
        for cb in callbacks:
            try:
                cb(event_type, data or {})
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        """是否正在运行自动化循环。

        仅以线程存活为准：收到停止指令(_stop_requested=True)但当前轮仍在收尾时，
        仍返回 True，直到线程真正结束。这样 UI 不会在"关了还在动"期间误报已停止。
        """
        th = self._loop_thread
        return th is not None and th.is_alive()

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _int_value(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_value(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default