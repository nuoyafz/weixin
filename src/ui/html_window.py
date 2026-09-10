"""QWebEngine UI 骨架，加载原版 HTML 界面并通过 QWebChannel 桥接后端。

结构参照原版 app.ui.html_main_window：
  - HtmlBridge   : 暴露给 JS 的 window.wepulseBridge（QObject）
  - FramelessWebView : 承载 HTML 的无边框 Web 视图
  - HtmlMainWindow : 无边框主窗口，负责加载 HTML、启动助手、事件驱动前端刷新
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Qt, Signal, Slot, QTimer, QThread
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow

import sys
import time as _time


def _term(msg: str) -> None:
    """把一条日志输出到 run.py 终端（stdout），便于用户直接看到前端反馈。"""
    try:
        sys.stdout.write("[%s] %s\n" % (_time.strftime("%H:%M:%S"), msg))
        sys.stdout.flush()
    except Exception:
        pass

from ..agent.observe_service import ObserveService


class _ConsoleForwardPage(QWebEnginePage):
    """自定义 Page：override javaScriptConsoleMessage 把前端 console 转发到终端。"""

    def __init__(self, parent, sink):
        super().__init__(parent)
        self._sink = sink

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        try:
            self._sink(level, message, lineNumber, sourceID)
        except Exception:
            pass
from ..config import Settings, load_settings


RESOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "resources"
HTML_FILE = RESOURCE_DIR / "html" / "main.html"


# ---------------------------------------------------------------------------
# HtmlBridge
# ---------------------------------------------------------------------------
class HtmlBridge(QObject):
    """通过 QWebChannel 暴露为前端 JS 的 window.wepulseBridge。

    桥接层支持两种调用模式：
    1) 原版模式：bridge.method(json_payload, callback)  ← QWebChannel 原生
    2) 兼容模式：bridge.method(id, payload)             ← 回调 ID 机制（解决部分 PySide6 版本不支持 object 回调的问题）
    """

    def __init__(self, window: "HtmlMainWindow"):
        super().__init__(window)
        self.window = window

    # ---- 内部工具 ----
    def _call_callback(self, callback, payload: str) -> None:
        """统一回调：支持 QWebChannel object 回调，也支持 ID 回调。"""
        if callback is None:
            return
        try:
            # 如果是可调用对象（QWebChannel 原生模式）
            if callable(callback):
                callback(payload)
                return
        except Exception:
            pass
        try:
            # 如果是字符串 ID（兼容模式）
            if isinstance(callback, str) and callback:
                self._emit_js_callback(callback, payload)
        except Exception:
            pass

    def _emit_js_callback(self, callback_id: str, payload: str) -> None:
        """通过 runJavaScript 执行 JS 回调。"""
        js_code = f"""
        try {{
            if (window.__bridgeCallbacks && window.__bridgeCallbacks[{callback_id}]) {{
                var cb = window.__bridgeCallbacks[{callback_id}];
                cb({payload});
                delete window.__bridgeCallbacks[{callback_id}];
            }}
        }} catch(e) {{ console.error('callback error', e); }}
        """
        try:
            self.window.view.page().runJavaScript(js_code)
        except Exception:
            pass

    # ---- 模拟问答 / 客服资料 ----
    @Slot(str, result=str)
    def simulateLearning(self, payload_json: str) -> str:
        """模拟问答识别。

        前端调用方式：wepulseBridge.simulateLearning(JSON.stringify(payload), cb)
        QWebChannel 会把末位的 JS 函数剥离，原生只收到 1 个参数 payload_json，
        返回值经 QWebChannel 回传给 cb。因此必须用 @Slot(str, result=str) 声明
        返回值，否则 cb 拿到的是空值，前端会一直显示「生成失败」。
        """
        try:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            question = payload.get("question") or ""
            history = payload.get("history") or []
            if not isinstance(history, list):
                history = []
            try:
                # run_sim_reply 返回结构化结果：ok/reply/source/model/
                # model_called/rag_has_context/message。旧版只返回一段字符串，
                # 且硬编码「感谢咨询…」兜底，导致任何问题都是同一个答案。
                res = self.window.run_sim_reply(question, history)
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "reply": "", "message": f"模拟回复生成失败：{e}",
                       "question": question, "source": "", "model": "",
                       "model_called": False, "rag_has_context": False, "error": str(e)}
            # 终端留痕，方便真机排查（模型名 / 是否真的调了模型 / 错误原因）
            _term("[ui] simulateLearning q=%r ok=%s model=%s called=%s rag=%s err=%s" % (
                res.get("question", "")[:40], res.get("ok"), res.get("model"),
                res.get("model_called"), res.get("rag_has_context"),
                (res.get("error") or "")[:160]))
            result = {
                "ok": bool(res.get("ok")),
                "question": res.get("question", question),
                "reply": res.get("reply", ""),
                "source": res.get("source") or ("模拟" if res.get("ok") else ""),
                "model": res.get("model", ""),
                "model_called": bool(res.get("model_called")),
                "rag_has_context": bool(res.get("rag_has_context")),
                "message": res.get("message", ""),
                "review": history,
            }
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"模拟问答异常：{e}", "reply": ""}
        return json.dumps(result, ensure_ascii=False)

    @Slot(str, result=str)
    def adoptLearning(self, payload_json: str) -> str:
        """保存一条模拟问答到回复记忆。

        前端调用：wepulseBridge.adoptLearning(JSON.stringify(payload), cb)
        QWebChannel 剥离末位函数回调 → 原生收到 1 个参数 payload_json，
        返回值回传给 cb，故用 @Slot(str, result=str)。
        """
        try:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            question = str(payload.get("question") or "").strip()
            answer = str(payload.get("answer") or "").strip()

            # 旧实现只返回一句硬编码成功文案、什么都不写，属于「假保存」：
            # 用户以为存进去了，下一轮提问模型依旧答不上来。这里真正落两处：
            #   1) config.yaml 的 test_scenarios —— 由 _approved_learning_samples()
            #      读进 prompt 的 approved_learning_samples，下一轮立即生效；
            #   2) data/knowledge/常见问题.txt —— 供 RAG 检索相似问题。
            if not question or not answer:
                result = {"ok": False, "message": "问题和回答都不能为空，无法保存。",
                          "question": question, "answer": answer}
                return json.dumps(result, ensure_ascii=False)

            wrote_cfg = self.window._append_test_scenario(question, answer)
            kb_path = self.window._append_knowledge_qa(question, answer)
            self.window._invalidate_knowledge_cache()

            if not wrote_cfg and not kb_path:
                result = {"ok": False, "message": "保存失败：未能写入配置与知识库。",
                          "question": question, "answer": answer}
                return json.dumps(result, ensure_ascii=False)

            result = {
                "ok": True,
                "message": "已保存这条问答，下一轮提问即可按此回答。"
                           + ("" if wrote_cfg else "（提示：回复记忆写入失败，仅存进知识库）"),
                "question": question,
                "answer": answer,
                "saved_to_config": bool(wrote_cfg),
                "saved_to_knowledge": bool(kb_path),
            }
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"保存失败：{e}"}
        return json.dumps(result, ensure_ascii=False)

    @Slot(str, str)
    def previewBusinessIdentity(self, callback_id: str = "", payload: str = "") -> None:
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        text = str(data.get("business_identity_text") or "")
        if not text.strip():
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请先填写业务资料"}, ensure_ascii=False))
            return
        result = json.dumps({
            "ok": True,
            "opening": "我会按已填写的业务资料接待客户。",
            "preview": {"key_points": [
                ln.strip() for ln in text.splitlines() if ln.strip()][:6]},
            "missing": [],
            "handoff": [],
            "status_label": "资料整理",
            "score_text": "保存后即时生效",
        }, ensure_ascii=False)
        self._call_callback(callback, result)

    @Slot(str, str)
    def completeBusinessIdentity(self, callback_id: str = "", payload: str = "") -> None:
        """保存业务资料原文到 config.yaml 的 business.identity_text 并热重载。

        （此前此槽返回硬编码成功、不落盘，「AI 润写」为假保存；现在至少把
        原文持久化，真正可被 prompt 资料读取。）
        """
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        text = str(data.get("business_identity_text") or "").strip()
        if not text:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请先填写业务资料"}, ensure_ascii=False))
            return
        ok = self.window._patch_sections_in_yaml(
            {"business": {"identity_text": text}})
        if ok and self.window.assistant is not None:
            try:
                cfg = self.window._load_yaml()
                if cfg:
                    self.window.assistant.reload_config(cfg)
            except Exception:  # noqa: BLE001
                pass
        self._call_callback(callback, json.dumps({
            "ok": bool(ok),
            "message": "业务资料已保存到 config.yaml" if ok else "写入 config.yaml 失败",
            "opening": "已保存业务资料，自动回复将按此资料执行。",
            "identity_text": text,
        }, ensure_ascii=False))

    # ---- 启动 / 停止 / 检测 / 窗口 ----
    @Slot()
    def startAssistant(self) -> None:
        self.window.start_assistant()

    @Slot()
    def stopAssistant(self) -> None:
        self.window.stop_assistant()

    @Slot(str, str)
    def toggleAssistant(self, callback_id: str = "", payload: str = "") -> None:
        """启动/暂停助手。注入脚本以 (callback_id, payload) 两个字符串调用，
        避免 QWebChannel 把字符串当 PyObjectWrapper 函数包装导致崩溃。"""
        callback = callback_id or None
        if self.window.assistant and self.window.assistant.is_running:
            self.window.stop_assistant()
            is_running = False
        else:
            self.window.start_assistant()
            is_running = self.window.assistant.is_running if self.window.assistant else False
        self._call_callback(callback, json.dumps({"running": is_running}, ensure_ascii=False))

    @Slot(str, str)
    def checkEnvironment(self, callback_id: str = "", payload: str = "") -> None:
        callback = callback_id or None
        report = self.window.check_environment()
        self._call_callback(callback, report)

    @Slot(str, str)
    def refreshPreview(self, callback_id: str = "", payload: str = "") -> None:
        """截取微信窗口真实画面，回调返回 base64 图片；失败返回 ok:False。
        （对应前端「微信真实画面」的「刷新画面」按钮）"""
        result = {"ok": False, "message": "微信窗口未就绪", "image": ""}
        try:
            from ..capture.screen_capture import ScreenCapture
            capture = getattr(self.window.assistant, "capture", None)
            if capture is None:
                backend = getattr(getattr(self.window.settings, "wechat", None), "capture_backend", "printwindow")
                capture = ScreenCapture(backend=backend)
            info = capture.find_wechat_window()
            if not info or not info.hwnd:
                result = {"ok": False, "message": "未找到微信窗口", "image": ""}
            else:
                parked_hint = ""
                wm = None
                try:
                    from ..desktop.wechat_window_manager import WeChatWindowManager
                    wm = getattr(self.window.assistant, "window_manager", None)
                    if wm is None:
                        wm = WeChatWindowManager()
                        wm.set_wechat_hwnd(info.hwnd)
                    rect = wm.get_rect(info.hwnd)
                    offscreen = rect is not None and not wm._is_on_screen(rect)
                    if wm.is_minimized(info.hwnd):
                        # 最小化：capture_window 内部会用 restore_offscreen 在屏外恢复渲染，
                        # 全程不进入物理桌面，用户看不到任何窗口移动/闪烁。
                        parked_hint = " (微信此前最小化，已在屏外恢复渲染后捕获)"
                    elif offscreen:
                        # 屏外后台：PrintWindow 直接从窗口表面捕获，无需把窗口拉回桌面
                        parked_hint = " (微信在虚拟外屏后台，已从其窗口表面直接捕获)"
                except Exception as e:  # noqa: BLE001
                    parked_hint = f" (窗口状态处理异常: {e})"
                from ..capture.screen_capture import CaptureResult
                # capture_window 已内置屏外/最小化无感处理，绝不把窗口挪到桌面中央
                cr = capture.capture_window(hwnd=info.hwnd, prefix="preview")
                if cr.success and cr.image is not None:
                    import base64 as _b64
                    import cv2
                    ok_, buf = cv2.imencode(".jpg", cr.image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok_:
                        import os
                        try:
                            bgr = cv2.cvtColor(cr.image, cv2.COLOR_BGRA2BGR) if (cr.image.shape[2] == 4) else cr.image
                            os.makedirs("_preview", exist_ok=True)
                            cv2.imwrite(os.path.join("_preview", "last.png"), bgr)
                        except Exception:
                            pass
                        _term(f"[preview] 微信截图成功 {cr.width}x{cr.height} base64={len(buf.tobytes())}B → _preview/last.png{parked_hint}")
                        result = {"ok": True, "message": "success",
                                  "image": "data:image/jpg;base64," + _b64.b64encode(buf.tobytes()).decode("ascii"),
                                  "width": cr.width, "height": cr.height}
                    else:
                        result = {"ok": False, "message": "图片编码失败", "image": ""}
                else:
                    result = {"ok": False, "message": cr.error or "截图失败", "image": ""}
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"截图异常: {e}", "image": ""}
        self._call_callback(callback_id or None, json.dumps(result, ensure_ascii=False))

    @Slot()
    def minimizeWindow(self) -> None:
        self.window.showMinimized()

    @Slot()
    def toggleMaximize(self) -> None:
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()

    @Slot(str, str)
    def showWechat(self, callback_id: str = "", payload: str = "") -> None:
        """UI「显示微信」按钮：把后台(虚拟外屏/最小化)的微信恢复到桌面可见区。

        优先恢复到 park 前记录的原始位置；记录丢失时回退主屏居中，
        确保窗口一定落回可见屏幕（多屏环境也不会掉到屏外）。
        """
        result = {"ok": False, "message": "未找到微信窗口"}
        try:
            from ..capture.screen_capture import ScreenCapture
            from ..desktop.wechat_window_manager import WeChatWindowManager

            capture = getattr(self.window.assistant, "capture", None)
            if capture is None:
                backend = getattr(getattr(self.window.settings, "wechat", None),
                                  "capture_backend", "printwindow")
                capture = ScreenCapture(backend=backend)
            info = capture.find_wechat_window()
            if info and info.hwnd:
                wm = getattr(self.window.assistant, "window_manager", None)
                if wm is None:
                    wm = WeChatWindowManager()
                wm.set_wechat_hwnd(info.hwnd)
                pos = wm.show_window(info.hwnd)
                if pos is not None:
                    result = {"ok": True,
                              "message": "已将微信恢复到桌面",
                              "position": pos}
                else:
                    result = {"ok": False, "message": "恢复窗口失败"}
            _term(f"[ui] showWechat -> {result}")
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"恢复失败: {e}"}
        self._call_callback(callback_id or None, json.dumps(result, ensure_ascii=False))

    @Slot(str, str)
    def hideWechat(self, callback_id: str = "", payload: str = "") -> None:
        """UI「后台运行」按钮：把微信移回虚拟外屏(屏外)后台，桌面不再显示。"""
        result = {"ok": False, "message": "未找到微信窗口"}
        try:
            from ..capture.screen_capture import ScreenCapture
            from ..desktop.wechat_window_manager import WeChatWindowManager

            capture = getattr(self.window.assistant, "capture", None)
            if capture is None:
                backend = getattr(getattr(self.window.settings, "wechat", None),
                                  "capture_backend", "printwindow")
                capture = ScreenCapture(backend=backend)
            info = capture.find_wechat_window()
            if info and info.hwnd:
                wm = getattr(self.window.assistant, "window_manager", None)
                if wm is None:
                    wm = WeChatWindowManager()
                wm.set_wechat_hwnd(info.hwnd)
                ok = wm.park(info.hwnd)
                result = {"ok": bool(ok),
                          "message": "已将微信移至后台(虚拟外屏)" if ok
                          else "隐藏失败"}
            _term(f"[ui] hideWechat -> {result}")
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"隐藏失败: {e}"}
        self._call_callback(callback_id or None, json.dumps(result, ensure_ascii=False))

    @Slot()
    def closeWindow(self) -> None:
        self.window.close()

    @Slot(int, int)
    def moveRelative(self, dx: int, dy: int) -> None:
        """无边框窗口拖动：按相对位移移动窗口（供前端 titlebar 拖拽调用）。"""
        try:
            p = self.window.pos()
            self.window.move(p.x() + dx, p.y() + dy)
        except Exception:
            pass

    # ---- 设置读写 ----
    @Slot(str, str)
    def loadSettings(self, callback_id: str = "", payload: str = "") -> None:
        """返回当前模型配置，供前端预填。优先 text_model，空则取 vision_model。"""
        callback = callback_id or None
        cfg = self.window._load_yaml()
        cfg = cfg if isinstance(cfg, dict) else {}
        tm = cfg.get("text_model") or {}
        vm = cfg.get("vision_model") or {}
        wechat = cfg.get("wechat") or {}
        fr = cfg.get("friend_requests") or {}
        voice = cfg.get("voice_messages") or {}
        quote = cfg.get("quote_reply") or {}
        fb = cfg.get("reply_fallback") or {}
        biz = cfg.get("business") or {}
        ui_cfg = cfg.get("ui") or {}
        ocr_mode = str(
            vm.get("ocr_mode", "")
            or wechat.get("ocr_mode", "")
            or cfg.get("ocr_mode", "")
            or "hybrid"
        ).strip().lower()
        if ocr_mode not in ("local", "ai", "hybrid"):
            ocr_mode = "hybrid"
        # 全量设置回显：设置页开关/下拉框状态一律以 config.yaml 为准，
        # 不再使用 HTML 硬编码状态（此前 UI 显示与实际配置脱节）。
        result = json.dumps({
            "app_name": self.window.settings.app_name,
            "app_version": self.window.settings.app_version,
            "base_url": tm.get("base_url") or vm.get("base_url", ""),
            "api_key": tm.get("api_key") or vm.get("api_key", ""),
            "model": tm.get("model") or vm.get("model", ""),
            "ocr_mode": ocr_mode,
            "enable_thinking": bool(tm.get("enable_thinking", vm.get("enable_thinking", False))),
            # 自动回复规则
            "enable_rpa_send": bool(wechat.get("enable_rpa_send", True)),
            "min_confidence_on": float(cfg.get("min_confidence_to_reply", 0.6) or 0) > 0,
            "fallback_chitchat": bool(fb.get("enabled", fb.get("no_knowledge_chitchat_enabled", True))),
            "merge_mode": "merge" if wechat.get("merge_reply_segments", True) else "split",
            "foreground_mode": str(wechat.get("foreground_keyboard_mode", "efficiency")),
            "quote_reply": bool(quote.get("enabled", True)),
            "handoff_trouble": bool(wechat.get("handoff_trouble", True)),
            "recognition_mode": str(wechat.get("recognition_mode", "double_click_pin")),
            # 性能与识别优化
            "vision_fallback_enabled": bool(vm.get("fallback_enabled", True)),
            "fallback_min_confidence": float(vm.get("fallback_min_confidence", 0.70) or 0.70),
            "vision_timeout_seconds": int(vm.get("timeout_seconds", 18) or 18),
            "idle_sleep_seconds": float(wechat.get("idle_sleep_seconds", 2.0) or 2.0),
            "cycle_sleep_seconds": float(wechat.get("cycle_sleep_seconds", 0.5) or 0.5),
            # 群聊
            "group_reply": bool(wechat.get("group_reply", True)),
            "require_mention": bool(wechat.get("require_mention", True)),
            "allowed_groups": list(wechat.get("allowed_groups") or []),
            "mention_names": list(wechat.get("mention_names") or []),
            # 好友申请
            "friend_auto_accept": bool(fr.get("auto_accept", True)),
            "friend_accept_mode": str(fr.get("accept_mode", "accept_all")),
            "friend_keywords": list(fr.get("keyword_rules") or []),
            "friend_check_interval": fr.get("check_interval_seconds", 30),
            "friend_max_per_cycle": fr.get("max_per_cycle", 3),
            "friend_send_welcome": bool(fr.get("send_welcome", True)),
            "friend_welcome_text": str(fr.get("welcome_text", "")),
            # 语音 / 业务资料 / 杂项
            "voice_auto_convert": bool(voice.get("auto_convert_to_text", True)),
            "business_identity_text": str(biz.get("identity_text", "")),
            "ui_auto_start": bool(ui_cfg.get("auto_start", False)),
            "ui_auto_update_check": bool(ui_cfg.get("auto_update_check", True)),
            "ui_screenshot_retention": str(ui_cfg.get("screenshot_retention_days", 7)),
        }, ensure_ascii=False)
        self._call_callback(callback, result)

    @Slot(str, str)
    def saveSettings(self, callback_id: str = "", payload_json: str = "") -> None:
        callback, payload_json = self._normalize_slot_args(callback_id, payload_json)
        try:
            data = json.loads(payload_json or "{}")
        except Exception:
            data = {}
        self.window.save_settings_from_ui(data)
        self._call_callback(callback, json.dumps(
            {"ok": True, "message": "设置已保存"}, ensure_ascii=False))

    # ---- 参数兼容工具 ----
    @staticmethod
    def _normalize_slot_args(callback_id: str, payload: str):
        """兼容旧前端 (payload, fn) 的调用方式。

        QWebChannel 直接传 JS 函数时函数会被剥离，JSON 落到第一个参数、
        第二个参数为空。检测到这种错位时把 JSON 还原为 payload、回调置空，
        避免把空 payload 当成有效输入（曾导致黑名单被整体清空）。
        """
        cb = (callback_id or "").strip()
        if cb.startswith(("{", "[", '"')) and not (payload or "").strip():
            return None, callback_id
        return (callback_id or None), payload or ""

    # ---- 不回复联系人（黑名单）管理 ----
    @Slot(str, str)
    def get_skip_contacts(self, callback_id: str = "", payload: str = "") -> None:
        """返回当前黑名单联系人列表（内置默认 + config.yaml 配置）。

        前端统一走注入的 callBridge(method, payload, cb)。
        """
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            from ..brain.skip_contacts import BUILTIN_CONTACT_BLACKLIST
            cfg = self.window._load_yaml()
            wechat = (cfg.get("wechat") or {}) if isinstance(cfg, dict) else {}
            configured = list(wechat.get("contact_blacklist") or [])
            contacts = [str(x).strip() for x in configured if str(x).strip()]
            # 合并内置默认（去重保序）
            seen = set(contacts)
            for name in BUILTIN_CONTACT_BLACKLIST:
                if name not in seen:
                    contacts.append(name)
                    seen.add(name)
            result = {"ok": True, "contacts": contacts}
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"读取黑名单失败：{e}", "contacts": []}
        self._call_callback(callback, json.dumps(result, ensure_ascii=False))

    @Slot(str, str)
    def save_skip_contacts(self, callback_id: str = "", payload: str = "") -> None:
        """保存黑名单联系人列表到 config.yaml 的 wechat.contact_blacklist 并热重载。

        前端统一走注入的 callBridge(method, payload, cb)。
        """
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
            # 数据破坏防护：payload 必须显式带 contacts 字段（哪怕是空数组）。
            # 缺字段说明调用方参数错位/损坏，拒绝写入以防误清空名单。
            if not isinstance(data, dict) or "contacts" not in data:
                self._call_callback(callback, json.dumps(
                    {"ok": False,
                     "message": "payload 缺少 contacts 字段，已拒绝写入（防止误清空）"},
                    ensure_ascii=False))
                return
            raw = data.get("contacts") or []
            items = []
            seen = set()
            for x in raw:
                name = str(x).strip()
                if name and name not in seen:
                    items.append(name)
                    seen.add(name)
            ok = self.window._set_wechat_contact_blacklist(items)
            if not ok:
                self._call_callback(callback, json.dumps(
                    {"ok": False, "message": "写入 config.yaml 失败"}, ensure_ascii=False))
                return
            # 热重载，使新黑名单即时生效
            try:
                cfg = self.window._load_yaml()
                if cfg and self.window.assistant is not None:
                    self.window.assistant.reload_config(cfg)
            except Exception:  # noqa: BLE001
                pass
            self._call_callback(callback, json.dumps(
                {"ok": True, "message": f"已保存 {len(items)} 条不回复联系人"}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"保存黑名单失败：{e}"}, ensure_ascii=False))

    @Slot(str, str)
    def test_model(self, callback_id: str = "", payload: str = "") -> None:
        """立即检查：用用户填写的 API 地址/模型做一次真实调用探测。

        探测策略：
        - 如果模型名看起来像多模态模型（含 vl/vision/omni/qvq/image 等），
          会同时发送一张 1x1 测试图，确认图片理解能力；否则发送纯文本 ping。
        - 遇到 404 model_not_found 时，会再调用 /models 列表给出"该端点可用模型"提示，
          避免把整段 JSON 错误塞到 UI 上导致布局被撑变形。
        """
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        base_url = (data.get("base_url") or "").strip()
        api_key = (data.get("api_key") or "").strip()
        model = (data.get("model") or "").strip()

        # 未填全时回退到 config.yaml 当前值（优先 text_model，其次 vision_model）
        if not base_url or not model:
            cfg = self.window._load_yaml()
            tm = (cfg.get("text_model") or {}) if isinstance(cfg, dict) else {}
            vm = (cfg.get("vision_model") or {}) if isinstance(cfg, dict) else {}
            base_url = base_url or (tm.get("base_url") or vm.get("base_url") or "").strip()
            api_key = api_key or (tm.get("api_key") or vm.get("api_key") or "").strip()
            model = model or (tm.get("model") or vm.get("model") or "").strip()

        if not base_url or not model:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请填写 API 地址和模型名称"}, ensure_ascii=False))
            return

        import time as _t
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        # 16x16 白底 PNG（base64），用于轻量测试视觉输入（某些 VL 模型限制宽高>10）
        _TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHUlEQVR4nGP8////fwYKABMlmkcNGDVg1IDBZAAAa8oEHMu6Z0kAAAAASUVORK5CYII="

        def _looks_vision(m: str) -> bool:
            return any(k in m.lower() for k in (
                "vl", "vision", "omni", "qvq", "image", "gpt-4o", "claude-3",
                "glm-4v", "kimi-v", "internvl", "llava", "minicpm-v"))

        def _build_body(m: str) -> dict:
            text_msg = {"role": "user", "content": "ping"}
            if _looks_vision(m):
                text_msg["content"] = [
                    {"type": "text", "text": "ping"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{_TINY_PNG}", "detail": "low"}}
                ]
            body = {"model": m, "messages": [text_msg], "max_tokens": 5,
                    "temperature": 0, "stream": False}
            # 测试连接同样要关思考：qwen3 系列默认思考会吃掉 max_tokens=5 的全部预算，
            # 导致正文为空、连接测试误判失败（2026-09-10）
            if "aliyuncs.com" in base_url.lower() or "dashscope" in base_url.lower():
                body["enable_thinking"] = False
            return body

        def _list_models() -> list:
            try:
                list_url = base_url.rstrip("/") + "/models"
                headers = {}
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key
                req = Request(list_url, headers=headers, method="GET")
                with urlopen(req, timeout=15) as resp:
                    j = json.loads(resp.read().decode("utf-8", errors="replace"))
                return [m.get("id", "") for m in j.get("data", [])]
            except Exception:
                return []

        start = _t.time()
        try:
            endpoint = base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
            body = _build_body(model)
            req = Request(endpoint, data=json.dumps(body).encode("utf-8"),
                          headers=headers, method="POST")
            with urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    j = json.loads(raw)
                    ok = "choices" in j
                    msg = "连接成功，模型可调用" + ("（已验证图文输入）" if _looks_vision(model) else "")
                    if not ok:
                        msg = "返回异常: " + str(j.get("error", ""))[:120]
                except Exception:
                    ok = True
                    msg = "连接成功（响应非标准 JSON）"
            latency = int((_t.time() - start) * 1000)
            # 检测通过后，顺带跑一轮回复速度检查（默认 3 轮短请求），回传延迟统计
            speed_payload = None
            if ok:
                try:
                    from ..reply.model_check import ModelChecker
                    _sp = ModelChecker(base_url, api_key, model,
                                       timeout_seconds=20).speed_check(rounds=3)
                    if _sp.get("ok"):
                        speed_payload = {
                            "avg_ms": _sp.get("avg_ms", 0),
                            "min_ms": _sp.get("min_ms", 0),
                            "max_ms": _sp.get("max_ms", 0),
                            "rounds": _sp.get("rounds", 0),
                            "success_rounds": _sp.get("success_rounds", 0),
                            "latencies": _sp.get("latencies", []),
                        }
                except Exception as _e:  # noqa: BLE001
                    _term(f"[ui] model speed_check failed: {_e}")
            self._call_callback(callback, json.dumps(
                {"ok": ok, "message": msg, "latency_ms": latency,
                 "model": model, "speed": speed_payload},
                ensure_ascii=False))
        except HTTPError as e:
            err = ""
            try:
                err = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            detail = err
            try:
                ej = json.loads(err)
                detail = ej.get("error", {}).get("message", err) or err
            except Exception:
                pass
            # 404 model_not_found 时给出可用视觉模型提示，避免 JSON 撑爆 UI
            if e.code == 404 and ("model_not_found" in err.lower() or "model not exist" in err.lower() or "not exist" in err.lower()):
                models = _list_models()
                vision_candidates = [m for m in models if _looks_vision(m)]
                hint = ""
                if vision_candidates:
                    top = ", ".join(vision_candidates[:6])
                    hint = f"。该端点可用视觉模型：{top} 等"
                detail = f"模型名不存在{hint}，请从列表复制准确名称后重试"
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"HTTP {e.code}: {detail}"}, ensure_ascii=False))
        except URLError as e:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"网络错误: {e.reason}"}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"检测失败: {e}"}, ensure_ascii=False))

    # =================================================================
    # 知识库 / 诊断 / 素材：此前前端只有 toast 提示、后端没有实现，
    # 属于「点了没反应」的死功能。这里补齐全部真实实现。
    # =================================================================
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _knowledge_root(self) -> Path:
        cfg = self.window._load_yaml() or {}
        root = str((cfg.get("knowledge") or {}).get("root") or "data/knowledge").strip()
        p = Path(root)
        return p if p.is_absolute() else (self._project_root() / p)

    def _fresh_knowledge_base(self):
        """构造并加载一个全新的知识库实例（不复用旧缓存）。"""
        try:
            from ..rag.knowledge_base import KnowledgeBase, extra_roots_from_config, skip_unreviewed_from_config
            cfg = self._load_yaml() if hasattr(self, "_load_yaml") else None
            kb = KnowledgeBase(
                root_path=str(self._knowledge_root()),
                extra_roots=extra_roots_from_config(cfg),
                skip_unreviewed=skip_unreviewed_from_config(cfg),
            )
            kb.load_documents()
            return kb
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] knowledge base load failed: {e}")
            return None

    @Slot(str, str)
    def ragStatus(self, callback_id: str = "", payload: str = "") -> None:
        """资料读取结果：文件数/段落数/健康度/失败文件。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            root = self._knowledge_root()
            kb = self._fresh_knowledge_base()
            docs = []
            chunks = 0
            if kb is not None:
                stats = kb.get_stats() or {}
                chunks = int(stats.get("total_chunks") or 0)
                per_source: dict[str, int] = {}
                for c in getattr(kb, "_chunks", []) or []:
                    per_source[getattr(c, "source", "?")] = (
                        per_source.get(getattr(c, "source", "?"), 0) + 1)
                for src, n in sorted(per_source.items()):
                    docs.append({
                        "name": Path(src).name if src else "?",
                        "path": str(src or ""),
                        "chunks": n,
                        "state": "ok",
                    })
            # 磁盘上真实存在的候选文件（含未成功入库的）
            on_disk = []
            for ext in ("*.txt", "*.md", "*.json", "*.csv"):
                on_disk.extend(root.rglob(ext))
            indexed_names = {d["name"] for d in docs}
            failed = [{"name": f.name, "path": str(f), "chunks": 0, "state": "failed"}
                      for f in on_disk
                      if f.name.lower() not in {"readme.md", "readme.txt", "readme"}
                      and f.name not in indexed_names]

            health = "充足" if chunks >= 8 else ("偏少" if chunks > 0 else "不足")
            result = {
                "ok": True,
                "docs": docs,
                "failed": failed,
                "chunk_count": chunks,
                "file_count": len(docs),
                "image_count": self._count_image_materials(),
                "health": health,
                "text": (f"已读取 {len(docs)} 个文件 / {chunks} 段资料。"
                         + ("资料不足，助手会转人工确认。" if health == "不足"
                            else "" if health == "充足" else "建议继续补充资料。")),
            }
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "message": f"读取资料状态失败：{e}",
                      "docs": [], "failed": [], "chunk_count": 0}
        self._call_callback(callback, json.dumps(result, ensure_ascii=False))

    @Slot(str, str)
    def rebuildKnowledge(self, callback_id: str = "", payload: str = "") -> None:
        """重新整理资料：重建索引并使新内容立即生效。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            kb = self._fresh_knowledge_base()
            if kb is None:
                self._call_callback(callback, json.dumps(
                    {"ok": False, "message": "知识库加载失败"}, ensure_ascii=False))
                return
            # 同步到正在运行的助手，下一轮回复即可用上新资料
            if self.window.assistant is not None:
                self.window.assistant.text_model.knowledge_base = kb
                self.window.assistant.text_model._kb_failed = False  # noqa: SLF001
            stats = kb.get_stats() or {}
            chunks = int(stats.get("total_chunks") or 0)
            self._call_callback(callback, json.dumps({
                "ok": True,
                "message": f"已重新整理，当前共 {chunks} 段资料。",
                "chunk_count": chunks,
            }, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"整理失败：{e}"}, ensure_ascii=False))

    @Slot(str, str)
    def clearUploadedKnowledge(self, callback_id: str = "", payload: str = "") -> None:
        """清空上传资料（先整体备份到 data/knowledge_backup）。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        if not data.get("confirm"):
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请先确认清空操作"}, ensure_ascii=False))
            return
        try:
            root = self._knowledge_root()
            if not root.exists():
                self._call_callback(callback, json.dumps(
                    {"ok": False, "message": "资料目录不存在"}, ensure_ascii=False))
                return
            backup = self._project_root() / "data" / (
                "knowledge_backup_" + time.strftime("%Y%m%d_%H%M%S"))
            shutil.copytree(root, backup, dirs_exist_ok=True)
            removed = 0
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    f.unlink()
                    removed += 1
            self.window._invalidate_knowledge_cache()
            self._call_callback(callback, json.dumps({
                "ok": True,
                "message": f"已清空 {removed} 个文件，备份在 {backup.name}",
                "backup": str(backup),
            }, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"清空失败：{e}"}, ensure_ascii=False))

    @Slot(str, str)
    def importKnowledgeFiles(self, callback_id: str = "", payload: str = "") -> None:
        """弹出系统文件选择窗口，把资料复制进知识库并重建索引。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            from PySide6.QtWidgets import QFileDialog
            paths, _ = QFileDialog.getOpenFileNames(
                None, "选择要加入知识库的资料", "",
                "资料文件 (*.txt *.md *.json *.csv *.pdf);;所有文件 (*.*)")
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"无法打开文件选择窗口：{e}"}, ensure_ascii=False))
            return
        if not paths:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "未选择文件"}, ensure_ascii=False))
            return
        try:
            root = self._knowledge_root()
            uploads = root / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            copied = []
            for p in paths:
                src = Path(p)
                if not src.is_file():
                    continue
                dst = uploads / src.name
                if dst.exists():
                    dst = uploads / f"{src.stem}_{int(time.time())}{src.suffix}"
                shutil.copy2(src, dst)
                copied.append(dst.name)
            self.window._invalidate_knowledge_cache()
            self._call_callback(callback, json.dumps({
                "ok": bool(copied),
                "message": (f"已导入 {len(copied)} 个文件" if copied else "导入失败"),
                "files": copied,
            }, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"导入失败：{e}"}, ensure_ascii=False))

    # ---- 图片素材 ----
    def _image_material_dir(self) -> Path:
        d = self._project_root() / "data" / "materials" / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _count_image_materials(self) -> int:
        try:
            return len([f for f in self._image_material_dir().iterdir()
                        if f.is_file() and f.suffix.lower()
                        in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")])
        except Exception:
            return 0

    @Slot(str, str)
    def listImageMaterials(self, callback_id: str = "", payload: str = "") -> None:
        """列出图片素材（data:URL 直接可渲染）。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            import base64
            items = []
            for f in sorted(self._image_material_dir().iterdir()):
                if not f.is_file() or f.suffix.lower() not in (
                        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                    continue
                try:
                    b64 = base64.b64encode(f.read_bytes()).decode("ascii")
                    items.append({
                        "name": f.name,
                        "url": f"data:image/{f.suffix.lower().lstrip('.')};base64,{b64}",
                    })
                except Exception:
                    continue
            self._call_callback(callback, json.dumps(
                {"ok": True, "items": items, "count": len(items)}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"读取素材失败：{e}", "items": []},
                ensure_ascii=False))

    @Slot(str, str)
    def addImageMaterial(self, callback_id: str = "", payload: str = "") -> None:
        """选图加入素材库。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            from PySide6.QtWidgets import QFileDialog
            paths, _ = QFileDialog.getOpenFileNames(
                None, "选择图片素材", "",
                "图片 (*.png *.jpg *.jpeg *.gif *.webp *.bmp);;所有文件 (*.*)")
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"无法打开文件选择窗口：{e}"}, ensure_ascii=False))
            return
        if not paths:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "未选择图片"}, ensure_ascii=False))
            return
        try:
            d = self._image_material_dir()
            added = []
            for p in paths:
                src = Path(p)
                dst = d / src.name
                if dst.exists():
                    dst = d / f"{src.stem}_{int(time.time())}{src.suffix}"
                shutil.copy2(src, dst)
                added.append(dst.name)
            self._call_callback(callback, json.dumps({
                "ok": True, "message": f"已添加 {len(added)} 张图片",
                "count": self._count_image_materials(),
            }, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"添加失败：{e}"}, ensure_ascii=False))

    @Slot(str, str)
    def removeImageMaterial(self, callback_id: str = "", payload: str = "") -> None:
        """删除一张图片素材。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        name = str(data.get("name") or "").strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "文件名不合法"}, ensure_ascii=False))
            return
        try:
            target = self._image_material_dir() / name
            if not target.is_file():
                self._call_callback(callback, json.dumps(
                    {"ok": False, "message": "素材不存在"}, ensure_ascii=False))
                return
            target.unlink()
            self._call_callback(callback, json.dumps({
                "ok": True, "message": "已删除",
                "count": self._count_image_materials()}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"删除失败：{e}"}, ensure_ascii=False))

    # ---- 常见问题 / 业务信息编辑器：此前点「保存」只关弹窗、不落盘 ----
    @staticmethod
    def _faq_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        d = root / "data" / "knowledge"
        d.mkdir(parents=True, exist_ok=True)
        return d / "常见问题.txt"

    @Slot(str, str)
    def loadFaq(self, callback_id: str = "", payload: str = "") -> None:
        """读取常见问题原文，供编辑器回填。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            p = self._faq_path()
            text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            self._call_callback(callback, json.dumps(
                {"ok": True, "text": text}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"读取失败：{e}", "text": ""}, ensure_ascii=False))

    @Slot(str, str)
    def saveFaq(self, callback_id: str = "", payload: str = "") -> None:
        """把常见问题写进知识库文件并重建索引。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        text = str(data.get("text") or "").strip()
        if not text:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "内容为空，未保存"}, ensure_ascii=False))
            return
        try:
            self._faq_path().write_text(text + "\n", encoding="utf-8")
            self.window._invalidate_knowledge_cache()
            self._call_callback(callback, json.dumps(
                {"ok": True, "message": "已保存常见问题，并已重新整理资料。"},
                ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"保存失败：{e}"}, ensure_ascii=False))

    # ---- 系统设置：数据目录 / 诊断 / 清理 / 更新记录 ----
    @Slot(str, str)
    def openDataDir(self, callback_id: str = "", payload: str = "") -> None:
        """在资源管理器里打开数据目录（日志/截图/配置所在）。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        target = str(data.get("target") or "data").strip()
        root = self._project_root()
        mapping = {"data": root / "data", "logs": root / "logs",
                   "screenshots": root / "screenshots", "config": root}
        path = mapping.get(target, root / "data")
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # noqa: S606 - 用户主动打开自己的数据目录
            ok, msg = True, f"已打开 {path.name}"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"打开失败：{e}"
        self._call_callback(callback, json.dumps(
            {"ok": ok, "message": msg, "path": str(path)}, ensure_ascii=False))

    @Slot(str, str)
    def exportDiagnostics(self, callback_id: str = "", payload: str = "") -> None:
        """导出诊断包：配置（脱敏）+ 最近日志 + 最近截图 + 环境信息。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        try:
            import platform
            root = self._project_root()
            out_dir = root / "data" / "diagnostics"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            zip_path = out_dir / f"diagnostics_{stamp}.zip"
            tmp = out_dir / f"_tmp_{stamp}"
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                # 1) 配置脱敏
                cfg_text = (root / "config.yaml").read_text(encoding="utf-8", errors="replace")
                safe = re.sub(r'(api_key\s*:\s*)(\S+)',
                              lambda m: m.group(1) + (m.group(2)[:3] + "***(已脱敏)"),
                              cfg_text)
                (tmp / "config_redacted.yaml").write_text(safe, encoding="utf-8")
                # 2) 最近日志（最多 5 个，每个只取最后 2000 行）
                logs_dir = root / "logs"
                if logs_dir.exists():
                    for f in sorted(logs_dir.glob("*.log"))[-5:]:
                        try:
                            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                            (tmp / f"logs_{f.name}").write_text(
                                "\n".join(lines[-2000:]), encoding="utf-8")
                        except Exception:
                            continue
                # 3) 最近 5 张截图
                shot_dir = root / "screenshots"
                if shot_dir.exists():
                    for f in sorted(shot_dir.glob("*.png"))[-5:]:
                        try:
                            shutil.copy2(f, tmp / f.name)
                        except Exception:
                            continue
                # 4) 环境信息
                (tmp / "environment.txt").write_text(
                    "\n".join([f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                               f"python: {platform.python_version()}",
                               f"system: {platform.platform()}",
                               f"cwd: {os.getcwd()}"]), encoding="utf-8")
                shutil.make_archive(str(zip_path.with_suffix("")), "zip", tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            self._call_callback(callback, json.dumps({
                "ok": True, "message": f"诊断包已导出：{zip_path.name}",
                "path": str(zip_path)}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"导出失败：{e}"}, ensure_ascii=False))

    @Slot(str, str)
    def cleanOldLogs(self, callback_id: str = "", payload: str = "") -> None:
        """按保留天数清理旧日志和旧截图。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        if not data.get("confirm"):
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请先确认清理操作"}, ensure_ascii=False))
            return
        try:
            days = int(data.get("days") or 7)
        except Exception:
            days = 7
        days = max(1, min(days, 365))
        cutoff = time.time() - days * 86400
        root = self._project_root()
        removed = 0
        freed = 0
        for sub in ("logs", "screenshots", "data/debug"):
            d = root / sub
            if not d.exists():
                continue
            for f in d.rglob("*"):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        freed += f.stat().st_size
                        f.unlink()
                        removed += 1
                except Exception:
                    continue
        self._call_callback(callback, json.dumps({
            "ok": True,
            "message": f"已清理 {removed} 个文件，释放 {freed // 1024 // 1024} MB",
            "removed": removed, "freed_bytes": freed,
        }, ensure_ascii=False))

    @Slot(str, str)
    def getChangelog(self, callback_id: str = "", payload: str = "") -> None:
        """返回更新记录。优先读 CHANGELOG.md，缺失时给出默认说明。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        root = self._project_root()
        text = ""
        for name in ("CHANGELOG.md", "docs/CHANGELOG.md", "../docs/CHANGELOG.md"):
            p = root / name
            try:
                if p.is_file():
                    text = p.read_text(encoding="utf-8", errors="replace")
                    break
            except Exception:
                continue
        if not text.strip():
            text = ("# 更新记录\n\n暂未维护 CHANGELOG.md。\n\n"
                    "最近一次改动：\n"
                    "- 修复「模拟问答」不调用大模型、任何问题都返回同一答案的问题。\n"
                    "- 修复「保存这条问答」假保存（只提示成功、不落盘）。\n"
                    "- 补齐知识库、图片素材、诊断工具等此前点了没反应的按钮。\n")
        self._call_callback(callback, json.dumps(
            {"ok": True, "text": text[:20000]}, ensure_ascii=False))

    # ---- AI 润写业务资料 / 固定测试 ----
    @Slot(str, str)
    def polishBusinessIdentity(self, callback_id: str = "", payload: str = "") -> None:
        """调用大模型把大白话业务介绍整理成结构化客服资料。"""
        callback, payload = self._normalize_slot_args(callback_id, payload)
        try:
            data = json.loads(payload or "{}")
        except Exception:
            data = {}
        raw = str(data.get("business_identity_text") or "").strip()
        if not raw:
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": "请先填写业务介绍"}, ensure_ascii=False))
            return
        tm = self.window._sim_text_model()
        if tm is None or not tm.available():
            self._call_callback(callback, json.dumps(
                {"ok": False,
                 "message": "未配置可用的文本模型，无法润写。请先到系统设置填写模型配置。"},
                ensure_ascii=False))
            return
        system = ("你是资深客服资料整理助手。把用户用大白话写的业务介绍，整理成结构化、"
                  "可直接给客服 AI 使用的中文资料。只输出整理后的纯文本，不要 Markdown "
                  "代码块，不要解释，不要编造用户没提到的信息（价格、地址、承诺一律保留原话，"
                  "缺失就写「未填写」）。")
        user = ("请把下面的业务介绍整理成以下结构，每段独立成段：\n"
                "一、卖什么\n二、价格与套餐\n三、常见问题（Q&A，至少 3 问 3 答）\n"
                "四、交付与售后\n五、联系方式与工作时间\n六、不能随便承诺的内容\n\n"
                "原始业务介绍：\n" + raw[:6000])
        try:
            result = tm.complete(system, user)
        except Exception as e:  # noqa: BLE001
            self._call_callback(callback, json.dumps(
                {"ok": False, "message": f"润写失败：{e}"}, ensure_ascii=False))
            return
        if not getattr(result, "success", False):
            self._call_callback(callback, json.dumps(
                {"ok": False,
                 "message": f"润写失败：{getattr(result, 'error', '未知错误')}"},
                ensure_ascii=False))
            return
        polished = str(getattr(result, "content", "") or "").strip()
        self._call_callback(callback, json.dumps({
            "ok": bool(polished),
            "message": "已整理完成，内容已覆盖到上方输入框。" if polished else "模型未返回内容",
            "identity_text": polished,
        }, ensure_ascii=False))

    @Slot(str, str)
    def runFixedTests(self, callback_id: str = "", payload: str = "") -> None:
        """固定测试：用一组固定题目跑真实「RAG + 模型」链路，逐条给结果。"""
        callback, _payload = self._normalize_slot_args(callback_id, payload)
        cases = [
            {"name": "普通闲聊", "question": "你好"},
            {"name": "价格咨询", "question": "你们的价格是多少"},
            {"name": "售后问题", "question": "买了之后能退款吗"},
            {"name": "范围问题", "question": "你们支持上门吗"},
            {"name": "无关消息", "question": "呲呲呲"},
        ]
        tm = self.window._sim_text_model()
        if tm is None or not tm.available():
            self._call_callback(callback, json.dumps(
                {"ok": False,
                 "message": "未配置可用的文本模型，无法运行测试。",
                 "results": []}, ensure_ascii=False))
            return
        results = []
        for case in cases:
            try:
                res = self.window.run_sim_reply(case["question"], [])
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "reply": "", "message": str(e),
                       "rag_has_context": False, "model_called": False}
            results.append({
                "name": case["name"],
                "question": case["question"],
                "reply": res.get("reply", ""),
                "ok": bool(res.get("ok")),
                "rag_has_context": bool(res.get("rag_has_context")),
                "model_called": bool(res.get("model_called")),
                "note": "" if res.get("ok") else (res.get("message") or ""),
            })
        passed = sum(1 for r in results if r["ok"])
        self._call_callback(callback, json.dumps({
            "ok": True,
            "message": f"{passed}/{len(results)} 条得到回复",
            "passed": passed, "total": len(results), "results": results,
        }, ensure_ascii=False))

    @Slot(str)
    def log(self, message: str) -> None:
        # 防御回环：以 [APP:desktop] 开头的消息说明它是 pushLog 的展示日志
        # 被 JS 端又回传了回来。若继续走 push_log 会被再次推给
        # window.pushLog → JS 再回传 → 死循环（日志前缀逐层膨胀刷屏）。
        # 这里只打终端、不再推回前端，彻底切断回路。
        if (message or "").lstrip().startswith("[APP:desktop]"):
            _term("[信息] %s" % message)
            return
        self.window.push_log(message, "info")


# ---------------------------------------------------------------------------
# FramelessWebView
# ---------------------------------------------------------------------------
class FramelessWebView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)


# ---------------------------------------------------------------------------
# HtmlMainWindow
# ---------------------------------------------------------------------------
class HtmlMainWindow(QMainWindow):
    _event_signal = Signal(str, dict)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.assistant: ObserveService = None

        self._msg_count = 0
        self._reply_count = 0
        self._error_count = 0

        self.setWindowTitle(f"{self.settings.app_name} {self.settings.app_version}")
        self.resize(1240, 820)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setStyleSheet("QMainWindow { background: #f5f5f7; }")

        self._event_signal.connect(self._process_event)

        self._setup_web()
        self._setup_assistant()

    # ------------------------------------------------------------------ UI
    def _setup_web(self):
        self.view = FramelessWebView(self)
        # 前端 console.* 转发：javaScriptConsoleMessage 是 QWebEnginePage 的方法
        # 而非信号，不能 connect（此前 connect 报 'builtin_function_or_method'
        # object has no attribute 'connect'）。正确做法是自定义 Page 子类 override。
        self._console_page = _ConsoleForwardPage(self.view, self._on_js_console)
        self.view.setPage(self._console_page)
        self.setCentralWidget(self.view)

        self.bridge = HtmlBridge(self)
        channel = QWebChannel(self.view.page())
        channel.registerObject("wepulseBridge", self.bridge)
        self.view.page().setWebChannel(channel)

        self.view.loadFinished.connect(self._inject_bridge_script)
        self.view.setUrl(QUrl.fromLocalFile(str(HTML_FILE)))

    def _on_js_console(self, level, message: str, lineNumber: int, sourceID: str) -> None:
        """捕获前端 console.log / console.error 并打印到终端。"""
        message = (message or "").strip()
        if not message:
            return
        # 降噪：跳过 Python 侧已记录的桥接回显（[APP:bridge]）与超大 base64 预览，
        # 避免同一事件在终端被重复刷屏（[JS]/[信息]/[desktop] 三份）。
        if "[APP:bridge]" in message or "base64," in message:
            return
        _term("[JS] " + message)

    def _inject_bridge_script(self, ok: bool) -> None:
        """页面加载完成后注入脚本：
        1) 初始化 wepulseBridge（QWebChannel）
        2) 把关键按钮的点击事件代理到后端 bridge（保留原有 UI 反馈）
        """
        _term(f"[ui] 页面加载完成 ok={ok}")
        if not ok:
            return
        js = r"""
          (function(){
            // ---- 全局日志与错误上报（经 javaScriptConsoleMessage 转发到 run.py 终端）----
            window.__logs = window.__logs || [];
            window.__flushLogs = function(){
              if (!window.wepulseBridge || !window.wepulseBridge.log || !window.__logs.length) return;
              var copy = window.__logs.slice(); window.__logs.length = 0;
              for (var i=0;i<copy.length;i++){ try{ window.wepulseBridge.log(copy[i]); }catch(_){} }
            };
            window.onerror = function(msg, src, line, col, err){
              try { window.__log && window.__log('jserror', (msg||'') + ' (line ' + (line||'') + ')'); } catch(_) {}
              return false;
            };
            window.__log = function(tag, msg){
              var line='[APP:' + tag + '] ' + msg;
              if (window.__logs) window.__logs.push(line);
              try { console.log(line); } catch(_) {}
              try { if (window.wepulseBridge && window.wepulseBridge.log) window.wepulseBridge.log(line); else window.__flushLogs && window.__flushLogs(); } catch(_) {}
            };

            // ---- Bridge 回调 ID 机制（解决 WebChannel 不能直接传 JS 函数的问题）----
            window.__bridgeCallbacks = window.__bridgeCallbacks || {};
            window.__bridgeCallbackCounter = window.__bridgeCallbackCounter || 0;

            function callBridge(methodName, payload, callback){
              window.__log('bridge', '调用 ' + methodName + (payload ? ' 参数=' + String(payload).slice(0,60) : ''));
              if (!window.wepulseBridge || !window.wepulseBridge[methodName]) {
                window.__log('bridge', '警告 方法未就绪: ' + methodName);
                if (callback) callback({ok: false, message: 'bridge not ready'});
                return;
              }
              var id = String(++window.__bridgeCallbackCounter);
              var cb = function(r){
                // Python 可能以 JS 对象或 JSON 字符串两种方式回传，统一转成 JSON 字符串
                var s = (typeof r === 'string') ? r : (r ? JSON.stringify(r) : '');
                // 凭据脱敏：loadSettings 回传体含 api_key，若原样打日志会把密钥
                // 明文写进终端和页面日志区。这里只脱敏"日志副本"，
                // 交给 callback 的仍必须是原始数据 s（否则表单会回填成掩码）。
                var logged = s;
                try {
                  logged = String(s).replace(
                    /"(api_key|apiKey|token|secret|password)"\s*:\s*"([^"]*)"/g,
                    function(m, k, v){
                      return '"' + k + '":"' + (v ? v.slice(0, 3) + '***(len=' + v.length + ')' : '') + '"';
                    });
                } catch(_) {}
                // 预览图 base64 精简：refreshPreview 返回几十 KB 的 image 数据，
                // 原样打日志会把终端刷爆。这里只保留图片类型与大小，不打印正文。
                try {
                  logged = String(logged).replace(
                    /("image"\s*:\s*)"data:image\/([a-zA-Z0-9+.\-]+);base64,([^"]*)"/g,
                    function(m, key, mime, data){
                      return key + '"<' + mime + ' ' + Math.round(data.length / 1024) + 'KB>"';
                    });
                } catch(_) {}
                window.__log('bridge', methodName + ' 返回 -> ' + String(logged).slice(0,160));
                if (callback) callback(s);
              };
              window.__bridgeCallbacks[id] = cb;
              // 调用 Bridge：第一个参数是 callback_id，第二个参数是 payload
              window.wepulseBridge[methodName](id, payload || '');
            }
            // 暴露为全局：main.html 内嵌脚本统一走 callBridge（回调可拿到结果），
            // 避免 QWebChannel 直接传 JS 函数被剥离导致 payload 丢失/回调拿不到数据。
            window.callBridge = callBridge;

            // ---- 工具：加载 qwebchannel.js 并建立 bridge ----
            function ensureBridge(fn){
              if (window.wepulseBridge) { window.__flushLogs && window.__flushLogs(); fn(); return; }
              var s = document.createElement('script');
              s.src = 'qrc:///qtwebchannel/qwebchannel.js';
              s.onload = function(){
                try {
                  new QWebChannel(qt.webChannelTransport, function(channel){
                    window.wepulseBridge = channel.objects.wepulseBridge;
                    window.__flushLogs && window.__flushLogs();
                    fn();
                  });
                } catch(e) { console.error('webchannel init failed', e); }
              };
              s.onerror = function(){ console.error('qwebchannel.js load failed'); };
              (document.head||document.documentElement).appendChild(s);
            }

            // ---- 重写关键按钮：用 onclick 属性（一次性覆盖原有 addEventListener 处理器）----
            var startBtn = document.getElementById('startButton');
            if (startBtn){
              startBtn.onclick = function(){
                var self = this;
                // 先切换图标（本地反馈）
                window._assistantRunning = !window._assistantRunning;
                self.innerHTML = window._assistantRunning
                  ? '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M9 5v14"></path><path d="M15 5v14"></path></svg></span>暂停助手'
                  : '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7Z"></path></svg></span>启动助手';
                // 再调用后端
                ensureBridge(function(){
                  callBridge('toggleAssistant', '', function(result){
                    try {
                      var data = JSON.parse(result || '{}');
                      window._assistantRunning = !!data.running;
                      self.innerHTML = window._assistantRunning
                        ? '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M9 5v14"></path><path d="M15 5v14"></path></svg></span>暂停助手'
                        : '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7Z"></path></svg></span>启动助手';
                    } catch(e){}
                  });
                });
              };
            }

            var checkBtn = document.getElementById('checkButton');
            if (checkBtn){
              checkBtn.onclick = function(){
                var self = this;
                ensureBridge(function(){
                  callBridge('checkEnvironment', '', function(result){
                    var data = {};
                    try { data = JSON.parse(result || '{}'); } catch(e){}
                    var ok = !!data.ok;
                    var icons = {
                      check: '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"></path></svg></span>',
                      alert: '<span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M12 8v5"></path><path d="M12 17h.01"></path><path d="M10.3 3.9 2.7 17.1A2 2 0 0 0 4.4 20h15.2a2 2 0 0 0 1.7-2.9L13.7 3.9a2 2 0 0 0-3.4 0Z"></path></svg></span>'
                    };
                    self.classList.toggle('fail', !ok);
                    self.innerHTML = (ok ? icons.check : icons.alert) + (ok ? '状态：正常' : '状态：失败');
                  });
                });
              };
            }

            // ---- 窗口控制：显示微信 / 后台运行 / 刷新画面 ----
            // 这三个按钮此前只有 HTML 标签、没有任何 JS 绑定（点了完全没反应），
            // 而 Python 端 showWechat / hideWechat / refreshPreview 早已实现，
            // 属于「后端已实现、前端未接线」的死 UI，这里补上绑定。
            function __bindWindowBtn(id, method, okText){
              var btn = document.getElementById(id);
              if (!btn) return;
              btn.onclick = function(){
                ensureBridge(function(){
                  callBridge(method, '', function(result){
                    var data = {};
                    try { data = JSON.parse(result || '{}'); } catch(e){}
                    // 刷新画面：成功后把截图直接渲染到预览区，不弹 toast
                    if (method === 'refreshPreview') {
                      if (data.ok && data.image) {
                        try { window.updatePreview && window.updatePreview(data.image); } catch(_) {}
                        return;
                      }
                      try { if (window.showToast) window.showToast(data.message || '刷新失败'); } catch(_) {}
                      return;
                    }
                    var msg = data.message || (data.ok ? okText : '操作失败');
                    try { if (window.showToast) window.showToast(msg); } catch(_) {}
                  });
                });
              };
            }
            __bindWindowBtn('showWechat', 'showWechat', '已将微信恢复到桌面');
            __bindWindowBtn('hideWechat', 'hideWechat', '已将微信移至后台');
            __bindWindowBtn('refreshPreview', 'refreshPreview', '画面已刷新');

            // ---- 模型配置：预填（保存/检测由 main.html 内嵌脚本统一处理，避免双绑定） ----
            function __collectModelInputs(){
              var b = document.getElementById('textBaseUrl');
              var k = document.getElementById('textApiKey');
              var m = document.getElementById('customTextModel');
              return {
                base_url: b ? b.value.trim() : '',
                api_key: k ? k.value.trim() : '',
                model: m ? m.value.trim() : ''
              };
            }

            // 启动后预填已有模型配置
            (function(){
              ensureBridge(function(){
                callBridge('loadSettings', '', function(result){
                  var d = {};
                  try { d = JSON.parse(result || '{}'); } catch(e){}
                  var b = document.getElementById('textBaseUrl');
                  var k = document.getElementById('textApiKey');
                  var m = document.getElementById('customTextModel');
                  var om = document.getElementById('ocrMode');
                  if (b && d.base_url) b.value = d.base_url;
                  if (k && d.api_key) k.value = d.api_key;
                  if (m && d.model) m.value = d.model;
                  if (om && d.ocr_mode) om.value = d.ocr_mode;
                });
              });
            })();

            // ---- 实时预览：后端每次截图(动作)都会推送 updatePreview ----
            window.updatePreview = function(src){
              try {
                var wrap = document.getElementById('wechatPreview');
                if (!wrap || !src) return;
                // 复用刷新画面的渲染结构：.live-image-shell + img.live-image
                wrap.className = 'live-preview preview-wide';
                wrap.innerHTML = '<div class="live-image-shell" style="background:#0b0c10">' +
                  '<img class="live-image" src="' + src + '" alt="微信真实画面" ' +
                  'style="width:100%;height:100%;object-fit:contain;object-position:center;display:block"/>' +
                  '</div>';
                var pill = document.getElementById('previewStatus');
                if (pill) { pill.textContent = '实时同步'; pill.className = 'pill green'; }
                var dbg = document.getElementById('debugTiming');
                if (dbg) dbg.textContent = '实时更新：' + new Date().toLocaleTimeString();
              } catch(e){}
            };

            // ---- 微信真实画面：刷新画面 -> 后端截图并渲染 ----
            var refreshBtn = document.getElementById('refreshPreview');
            var statusPill = document.getElementById('previewStatus');
            var debugTiming = document.getElementById('debugTiming');
            if (refreshBtn){
              refreshBtn.onclick = function(){
                var self = this;
                self.disabled = true;
                var old = self.textContent;
                self.textContent = '刷新中…';
                if (statusPill) { statusPill.textContent = '截图处理中'; statusPill.className = 'pill blue'; }
                ensureBridge(function(){
                  callBridge('refreshPreview', '', function(result){
                    var data = {};
                    try { data = JSON.parse(result || '{}'); } catch(e){}
                    try {
                      if (data.ok && data.image) {
                        var wrap = document.getElementById('wechatPreview');
                        if (wrap) {
                          // 按原版 CSS 结构渲染：.live-image-shell + img.live-image + --preview-src（模糊底）
                          var wide = data.width && data.height && data.width >= data.height;
                          wrap.className = 'live-preview ' + (wide ? 'preview-wide' : 'preview-tall');
                          wrap.innerHTML = '<div class="live-image-shell" style="background:#0b0c10">' +
                            '<img class="live-image" src="' + data.image + '" alt="微信真实画面" ' +
                            'style="width:100%;height:100%;object-fit:contain;object-position:center;display:block"/>' +
                            '</div>';
                        }
                        if (debugTiming) debugTiming.textContent = '调试耗时：' + new Date().toLocaleTimeString() +
                          ' · ' + (data.width||'') + '×' + (data.height||'');
                        if (statusPill) { statusPill.textContent = '截图成功'; statusPill.className = 'pill green'; }
                      } else {
                        if (statusPill) { statusPill.textContent = (data.message||'截图失败'); statusPill.className = 'pill red'; }
                        if (debugTiming) debugTiming.textContent = '调试耗时：' + new Date().toLocaleTimeString() +
                          ' · ' + (data.message||'');
                        if (window.showToast) window.showToast('刷新画面：' + (data.message||'失败'));
                      }
                    } catch(e){ statusPill && (statusPill.textContent = '渲染异常'); }
                    self.disabled = false;
                    self.textContent = old;
                  });
                });
              };
            }

            // ---- 显示微信 / 后台运行：手动恢复或隐藏微信窗口 ----
            var showBtn = document.getElementById('showWechat');
            if (showBtn) {
              showBtn.onclick = function () {
                var self = this;
                self.disabled = true;
                var old = self.textContent;
                self.textContent = '恢复中…';
                ensureBridge(function () {
                  callBridge('showWechat', '', function (result) {
                    var d = {};
                    try { d = JSON.parse(result || '{}'); } catch (e) {}
                    if (window.showToast) window.showToast('显示微信：' + (d.message || (d.ok ? '已恢复' : '失败')));
                    self.disabled = false;
                    self.textContent = old;
                  });
                });
              };
            }

            var hideBtn = document.getElementById('hideWechat');
            if (hideBtn) {
              hideBtn.onclick = function () {
                var self = this;
                self.disabled = true;
                var old = self.textContent;
                self.textContent = '隐藏中…';
                ensureBridge(function () {
                  callBridge('hideWechat', '', function (result) {
                    var d = {};
                    try { d = JSON.parse(result || '{}'); } catch (e) {}
                    if (window.showToast) window.showToast('后台运行：' + (d.message || (d.ok ? '已隐藏' : '失败')));
                    self.disabled = false;
                    self.textContent = old;
                  });
                });
              };
            }

            // ---- 无边框窗口：铺满去白框 + 标题栏按钮 + 拖动 ----
            (function(){
              // 1) 铺满窗口，去掉手机壳两侧露出的白底
              var s = document.createElement('style');
              s.textContent = 'html,body{height:100%;margin:0;padding:0;overflow:hidden;background:var(--page)}' +
                'main.stage{padding:0 !important;min-height:100%;place-items:stretch center}' +
                '.phone.product-shell{width:100% !important;height:100vh !important;max-width:none;border-radius:0}';
              document.head.appendChild(s);

              // 2) 标题栏窗口按钮 -> 后端无参方法
              function doMinimize(){ window.__log('win','minimize clicked'); try{ window.wepulseBridge && window.wepulseBridge.minimizeWindow(); }catch(e){ console.error(e); } }
              function doMaximize(){ window.__log('win','maximize clicked'); try{ window.wepulseBridge && window.wepulseBridge.toggleMaximize(); }catch(e){ console.error(e); } }
              function doClose(){ window.__log('win','close clicked'); try{ window.wepulseBridge && window.wepulseBridge.closeWindow(); }catch(e){ console.error(e); } }
              var mb = document.getElementById('windowMinimize');
              var xb = document.getElementById('windowMaximize');
              var cbx = document.getElementById('windowClose');
              if (mb) mb.onclick = doMinimize;
              if (xb) xb.onclick = doMaximize;
              if (cbx) cbx.onclick = doClose;

              // 3) titlebar 拖拽移动无边框窗口（避开按钮）
              var tb = document.querySelector('.topbar, header');
              if (tb) {
                var dragging = false, lastX = 0, lastY = 0;
                tb.addEventListener('mousedown', function(e){
                  if (e.target.closest && e.target.closest('button, .win-btn, .window-controls, .icon-btn, a')) return;
                  dragging = true; lastX = e.screenX || 0; lastY = e.screenY || 0;
                  try { e.preventDefault(); } catch(_) {}
                });
                document.addEventListener('mousemove', function(e){
                  if (!dragging) return;
                  var x = e.screenX || 0, y = e.screenY || 0, dx = x - lastX, dy = y - lastY;
                  lastX = x; lastY = y;
                  if ((dx || dy) && window.wepulseBridge) { try { window.wepulseBridge.moveRelative(dx, dy); } catch(_) {} }
                });
                document.addEventListener('mouseup', function(){ dragging = false; });
              }
            })();

            // ---- 暴露给后端事件回写的函数 ----
            window.setProcessStatus = function(text){
              var el = document.getElementById('processStatus');
              if (el) el.textContent = text || '';
            };
            window.setKnowledgeCount = function(n){
              var el = document.getElementById('knowledgeChunkCount');
              if (el) el.textContent = String(n || '0');
            };
            // 补齐 5 个状态计数器（旧版只写 2 个，导致待办/人工/跳过恒为 0）
            window.setCounts = function(total, replies, errors, pending, manual, skipped){
              var map = {
                totalMessageCountLeft: total,
                autoReplyCountLeft: replies,
                pendingMessageCountLeft: pending,
                manualReplyCountLeft: manual,
                skippedMessageCountLeft: skipped
              };
              // 兼容对象入参 setCounts({total,pending,sent,manual,skipped})
              if (total && typeof total === 'object') {
                map = {
                  totalMessageCountLeft: total.total,
                  autoReplyCountLeft: total.sent,
                  pendingMessageCountLeft: total.pending,
                  manualReplyCountLeft: total.manual,
                  skippedMessageCountLeft: total.skipped
                };
              }
              Object.keys(map).forEach(function(id){
                var el = document.getElementById(id);
                if (el) el.textContent = String(map[id] || 0);
              });
            };

            // 状态 -> 徽章文案/样式
            function _replyStateMeta(status){
              switch (status) {
                case 'sent':    return { label: '已回复', cls: 'queue-state' };
                case 'manual':  return { label: '待人工', cls: 'queue-state manual' };
                case 'skipped': return { label: '已跳过', cls: 'queue-state failed' };
                case 'failed':  return { label: '失败',   cls: 'queue-state failed' };
                default:        return { label: '处理中', cls: 'queue-state waiting' };
              }
            }
            // 阶段 -> 进度高亮
            var _STAGE_STEPS = { detected:0, entering:0, captured:1, ocr:1, analyzing:2, sending:3 };

            window.appendReplyRow = function(item){
              var list = document.getElementById('replyList');
              if (!list || !item) return;
              var empty = list.querySelector('.queue-item.empty');
              if (empty) empty.remove();

              var tpl = document.getElementById('replyRowTemplate');
              var art = null;

              // 按 id 原地更新：同一条消息随阶段推进而变形，不重复堆叠行
              if (item.id) {
                art = list.querySelector('.queue-item[data-id="' + item.id + '"]');
              }
              if (!art) {
                if (tpl && tpl.content) {
                  art = tpl.content.firstElementChild.cloneNode(true);
                } else {
                  art = document.createElement('article');
                  art.className = 'queue-item';
                  art.innerHTML = '<div class="queue-main"><strong data-role="contact"></strong>' +
                    '<div class="queue-insights"><div class="insight-row recognize">' +
                    '<div class="insight-copy"><b>识别到</b><span data-role="text"></span></div></div>' +
                    '<div class="insight-row suggest"><div class="insight-copy">' +
                    '<b>建议回复</b><span data-role="reply"></span></div></div></div></div>' +
                    '<span class="queue-state waiting" data-role="state">待处理</span>';
                }
                if (item.id) art.setAttribute('data-id', item.id);
                if (list.firstChild) list.insertBefore(art, list.firstChild);
                else list.appendChild(art);
              }

              // 联系人（旧版丢弃了 contact 字段，这里补上）
              var contactEl = art.querySelector('[data-role="contact"]');
              if (contactEl) {
                contactEl.textContent = item.contact || item.who || '客户';
              }
              var textEl = art.querySelector('[data-role="text"]');
              if (textEl) textEl.textContent = item.text || item.reason || '';
              var replyEl = art.querySelector('[data-role="reply"]');
              if (replyEl) {
                replyEl.textContent = item.reply || (item.status === 'manual' ? '待人工处理' : '');
              }
              var stateEl = art.querySelector('[data-role="state"]');
              if (stateEl) {
                var meta = _replyStateMeta(item.status || 'running');
                stateEl.textContent = item.stage_label || meta.label;
                stateEl.className = meta.cls;
              }
              // 进度：按阶段点亮
              var prog = art.querySelector('[data-role="progress"]');
              if (prog && item.stage) {
                var step = _STAGE_STEPS[item.stage] || 0;
                var spans = prog.querySelectorAll('span');
                for (var i = 0; i < spans.length; i++) {
                  spans[i].className = (i <= step) ? 'done' : '';
                }
              }
              // 列表过长时裁剪，避免无限增长
              while (list.children.length > 60) list.removeChild(list.lastChild);
            };
            window.pushLog = function(entry){
              if (!entry) return;
              var text = (entry && entry.text) || '';
              var level = (entry && entry.level) || 'info';
              console.log('[desktop][' + level + '] ' + text);
              // 桌面端推送的日志统一写到主页面真实日志容器（modelLogList，
              // 旧版的 #logContainer 早已不存在，原代码静默落空）。
              // 注意：这里【禁止】再调用 window.__log() 回传桥接。
              // __log 会走 wepulseBridge.log → Python HtmlBridge.log →
              // push_log → window.pushLog()，与本函数构成无限回环，且每转一圈
              // 前缀就多叠一层 "[APP:desktop] [info] "，表现为终端/日志区刷屏。
              // pushLog 是「Python → UI」的终点，只负责展示。
              var el = document.getElementById('modelLogList');
              if (el) {
                var ph = el.querySelector('.model-log-empty');
                if (ph) ph.remove();
                var line = document.createElement('div');
                line.className = 'log-line ' + level;
                var _t = document.createElement('span');
                _t.className = 'log-time';
                _t.textContent = new Date().toLocaleTimeString();
                var _m = document.createElement('span');
                _m.className = 'log-msg';
                _m.textContent = text;
                line.appendChild(_t);
                line.appendChild(document.createTextNode(' '));
                line.appendChild(_m);
                el.appendChild(line);
                while (el.children.length > 200) el.removeChild(el.firstChild);
              }
            };

            console.log('[bridge] 界面桥接已注入');
            // ---- 兼容层：原版 html 由打包器注入的状态函数，静态文件里没有 ----
            if (typeof window.setProcessStatus !== 'function') {
              window.setProcessStatus = function(text){
                var el = document.getElementById('processStatus');
                if (el) el.textContent = text;
                console.log('[status]', text);
              };
            }
            if (typeof window.setStartButton !== 'function') {
              window.setStartButton = function(running){
                var b = document.getElementById('startButton');
                if (!b) return;
                b.dataset.running = running ? '1' : '';
                b.classList.toggle('running', !!running);
                var span = b.querySelector('.btn-icon svg path');
                try { b.blur(); } catch(_){}
                console.log('[start-button] running=' + !!running);
              };
            }
          })();
        """
        try:
            self.view.page().runJavaScript(js)
        except Exception:
            pass
        # 启动后自动刷新一次「微信真实画面」（顺带自诊断，出错也会打到终端日志）
        QTimer.singleShot(1500, lambda: self._run_js(
            "try{document.getElementById('refreshPreview').click();}catch(e){}"))

    # ------------------------------------------------------------ assistant
    def _setup_assistant(self):
        # 传 raw dict（config.yaml 原文），避免无参构造内部 load_config()→asdict
        # 把 business/friend_requests/skip_contacts/offscreen/evidence_gate 等
        # Settings 数据类没有的段削掉（此前这些配置 GUI 启动即失效）。
        raw_cfg = self._load_yaml()
        if raw_cfg:
            self.assistant = ObserveService(config=raw_cfg)
        else:
            self.assistant = ObserveService()
        self.assistant.on_event(self._handle_event)

    def _handle_event(self, event_type: str, data: dict):
        self._event_signal.emit(event_type, data)

    def _process_event(self, event_type: str, data: dict):
        if event_type == "status":
            self._apply_state(data)
        elif event_type == "message_received":
            # 计数与渲染统一交给 msg_counts / msg_pipeline，避免重复建行与计数打架
            self._msg_count += 1
        elif event_type == "reply_generated":
            self._run_js("window.setLearningStatus('回复已生成', 'ok');")
        elif event_type == "reply_sent":
            # 渲染由 msg_pipeline 统一负责；计数由 msg_counts 负责
            self._reply_count += 1
        elif event_type == "recognition":
            count = data.get("count", 0)
            self._run_js("window.setProcessStatus('识别中：%s 个未读');" % count)
            self.push_log(f"识别到 {count} 个未读会话", "info")
        elif event_type == "message_skipped":
            contact = data.get("contact", "未知")
            reason = data.get("reason", "")
            self.push_log(f"跳过 {contact}: {reason}", "info")
        elif event_type == "error":
            # 左侧面板计数由 msg_counts 统一驱动，这里只记日志
            self._error_count += 1
            self.push_log(data.get("message", "错误"), "error")
        elif event_type == "warning":
            self.push_log(data.get("message", "警告"), "warning")
        elif event_type == "preview":
            img = data.get("image", "")
            if img:
                self._run_js("window.updatePreview(%s);" % json.dumps(img))
        elif event_type == "msg_pipeline":
            # 消息处理流水线：一条消息随阶段推进而原地更新
            self._run_js("window.appendReplyRow(%s);" % json.dumps({
                "id": data.get("id", ""),
                "contact": data.get("contact", ""),
                "text": data.get("text", ""),
                "reply": data.get("reply", ""),
                "reason": data.get("reason", ""),
                "stage": data.get("stage", ""),
                "status": data.get("status", ""),
                "stage_label": data.get("stage_label", ""),
            }, ensure_ascii=False))
        elif event_type == "msg_counts":
            # 5 个状态计数（全部/待办/回复/人工/跳过）
            self._run_js("window.setCounts(%s);" % json.dumps({
                "total": data.get("total", 0),
                "pending": data.get("pending", 0),
                "sent": data.get("sent", 0),
                "manual": data.get("manual", 0),
                "skipped": data.get("skipped", 0),
            }))

    def _apply_state(self, data: dict):
        state = data.get("state", "")
        if state == "running":
            self._run_js("window.setProcessStatus('运行中');")
            self._run_js("window.setStartButton(true);")
        elif state == "stopped":
            self._run_js("window.setProcessStatus('已停止');")
            self._run_js("window.setStartButton(false);")
        elif state == "idle":
            # 运行循环内的空闲（等待新消息）：助手仍在运行，绝不能显示"已停止"，
            # 否则用户误以为停了会反复手动点击（此前状态抖动的直接根因）。
            self._run_js("window.setProcessStatus('运行中 · 等待新消息');")
            self._run_js("window.setStartButton(true);")
        elif state in ("paused",):
            self._run_js("window.setProcessStatus('已暂停');")
        elif state == "initializing":
            self._run_js("window.setProcessStatus('初始化中');")
        message = data.get("message")
        # 启动/停止/idle 等关键状态已由 start_assistant/stop_assistant/run_one_cycle
        # 单独输出，避免状态事件再重复打印一次；只把非高频状态消息落日志。
        if message and state not in ("running", "stopped", "idle"):
            self.push_log(message, "info")
        progress = data.get("progress")
        if progress is not None:
            self._run_js("window.setProcessStatus('初始化 " + str(progress) + "%');")

    def _set_counts(self):
        self._run_js("window.setCounts(%d, %d, %d);" % (
            self._msg_count, self._reply_count, self._error_count))

    # ------------------------------------------------------------- actions
    def start_assistant(self):
        self.push_log("正在初始化助手...", "info")
        try:
            self.assistant.start_loop()
            self.push_log("助手已启动", "success")
        except Exception as e:
            self._run_js("window.setProcessStatus('初始化失败');")
            self._run_js("window.setStartButton(false);")
            self.push_log(f"启动失败: {e}", "error")

    def stop_assistant(self):
        self.assistant.stop_loop()
        self.push_log("助手已停止", "info")

    def _sim_text_model(self):
        """取得文本模型客户端；assistant 未初始化时按 config.yaml 现造一个。

        模拟问答页面与「系统设置」里的模型配置是同一份 config.yaml，
        所以这里永远以磁盘上的最新配置为准，避免改了模型却不生效。
        """
        cfg = self._load_yaml() or {}
        try:
            from ..ai.text_model_client import TextModelClient

            tm_cfg = cfg.get("text_model") or {}
            tm = TextModelClient(
                base_url=tm_cfg.get("base_url", ""),
                api_key=tm_cfg.get("api_key", ""),
                model=tm_cfg.get("model", ""),
                config=cfg,
            )
            # 同步给 assistant，保证「模拟问答」与正式运行共用同一个客户端
            if self.assistant is not None:
                self.assistant.text_model = tm
            return tm
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] sim text model init failed: {e}")
            return getattr(self.assistant, "text_model", None)

    def run_sim_reply(self, question: str, history: "list | None" = None) -> dict:
        """模拟问答：走与正式运行完全一致的「RAG 检索 + 大模型生成」链路。

        旧实现从未调用大模型：只是把 RAG 命中结果拼成「感谢咨询。+ 前 120
        字」，RAG 关闭或检索为空时一律返回同一句硬编码兜底 —— 所以不管输入
        什么问题都得到一个答案。这里改为复用 text_model.refine_reply()，
        并把 403 额度耗尽 / 404 模型不存在 / 未配置模型等真实原因透传到 UI，
        不再用假答案掩盖故障。
        """
        rows = [h for h in (history or []) if isinstance(h, dict)]
        q = str(question or "").strip()
        out = {
            "ok": False, "question": q, "reply": "", "source": "",
            "model": "", "model_called": False, "rag_has_context": False,
            "error": "", "message": "",
        }
        if not q:
            out["message"] = "请输入客户问题。"
            return out

        tm = self._sim_text_model()
        if tm is None:
            out["message"] = "未能初始化文本模型客户端，请检查 config.yaml 的 text_model 配置。"
            return out

        cfg = tm.config if isinstance(getattr(tm, "config", None), dict) else {}
        tm_cfg = cfg.get("text_model") or {}
        out["model"] = str(tm_cfg.get("model") or getattr(tm, "model", "") or "")

        # 构造与正式运行同构的 analysis。两个硬条件必须满足，否则
        # refine_reply 会在入口直接短路、根本不调模型：
        #   1) decision 落在 SENDABLE_DECISIONS（reply / reply_draft / weak_lead_draft）
        #   2) latest_message.sender == 'customer'
        conv_ctx = []
        for row in rows[-10:]:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            conv_ctx.append({
                "role": "assistant" if row.get("role") == "assistant" else "customer",
                "content": text,
            })
        analysis = {
            "current_contact": "模拟客户",
            "decision": "reply",
            "action": "reply",
            "intent": "brand_question",
            "latest_message": {"sender": "customer", "content": q},
            "customer_turn_text": q,
            "customer_turn_messages": [{"sender": "customer", "content": q}],
            "conversation_context": conv_ctx,
            "visible_conversation_text": q,
            "visible_conversation_messages": [{"sender": "customer", "content": q}],
            "confidence": 1.0,
        }
        window_info = {"title": "模拟问答", "visible_text": ""}

        try:
            refined, reply, meta = tm.refine_reply(analysis, window_info, "reply")
        except Exception as e:  # noqa: BLE001
            out["message"] = f"模型调用异常：{e}"
            return out

        if isinstance(refined, dict):
            out["rag_has_context"] = bool((refined.get("rag") or {}).get("has_context"))
        meta = meta or {}
        out["model_called"] = bool(meta.get("called"))
        reply = str(reply or "").strip()

        if reply:
            out["ok"] = True
            out["reply"] = reply
            out["source"] = f"模型 · {out['model']}" if out["model"] else "模型"
            out["message"] = "已生成回复"
            return out

        # 没拿到回复：把真实原因报出来，不再编造兜底话术
        err = ""
        if isinstance(refined, dict):
            err = str(refined.get("text_model_error") or "").strip()
        if not err:
            err = str((meta or {}).get("error") or "").strip()
        reason = str((meta or {}).get("reason") or "").strip()
        if not tm.available():
            out["message"] = ("未配置可用的文本模型（base_url / api_key / model 缺一不可），"
                              "请到「系统设置 → 模型配置」填写并保存后再试。")
        elif err:
            out["message"] = f"模型未返回回复：{err[:300]}"
        else:
            out["message"] = ("模型判定当前问题不需要回复（no_reply）。"
                              "若是正常业务问题，请到知识库补充资料后重试。")
        out["error"] = err[:300]
        return out

    # ---- 学习样本：模拟问答「保存这条问答」真正落盘 ----
    def _append_test_scenario(self, question: str, answer: str) -> bool:
        """往 config.yaml 的 test_scenarios 列表追加一条已采纳样本。

        text_model_client._approved_learning_samples() 直接读这一段并写进
        prompt 的 approved_learning_samples，所以保存后下一轮立即生效。
        采用行级写入而不是 yaml.dump，以免整份注释被冲掉。
        """
        try:
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                lines = f.readlines()

            entry = [
                f"  - name: {self._yaml_scalar('已采纳问答')}\n",
                f"    messages:\n",
                f"      - {self._yaml_scalar(question)}\n",
                f"    approved_reply: {self._yaml_scalar(answer)}\n",
            ]

            # 找到 test_scenarios: 段（顶层、顶格）
            start = -1
            for i, line in enumerate(lines):
                if line.startswith("test_scenarios:"):
                    start = i
                    break

            if start < 0:
                # 段不存在：在文件末尾新建
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append("\n# 模拟问答中保存的已采纳问答（自动生成，可手工编辑）\n")
                lines.append("test_scenarios:\n")
                lines.extend(entry)
            else:
                # 段已存在：先找段尾（第一个顶格且非注释的非空行），
                # 再回退跳过空行与段后注释，插到最后一个真实列表项之后。
                end = len(lines)
                for j in range(start + 1, len(lines)):
                    s = lines[j].strip()
                    if s and not lines[j][0].isspace() and not s.startswith("#"):
                        end = j
                        break
                insert_at = end
                for j in range(end - 1, start, -1):
                    s = lines[j].strip()
                    if not s:
                        continue
                    if not lines[j][0].isspace() and s.startswith("#"):
                        continue  # 段后的分隔注释，不插在它后面
                    insert_at = j + 1
                    break
                lines[insert_at:insert_at] = entry

            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] append test_scenario failed: {e}")
            return False

    def _append_knowledge_qa(self, question: str, answer: str) -> str:
        """把问答追加进知识库文本（供 RAG 检索），返回写入的文件路径。"""
        root = Path(__file__).resolve().parents[2]
        kb_dir = root / "data" / "knowledge"
        kb_dir.mkdir(parents=True, exist_ok=True)
        target = kb_dir / "常见问题.txt"
        header = "# 常见问题（模拟问答保存，自动生成）\n"
        block = f"\n问：{question}\n答：{answer}\n"
        try:
            if target.exists():
                text = target.read_text(encoding="utf-8", errors="replace")
            else:
                text = header
            if "（模拟问答保存，自动生成）" not in text:
                text = header + text
            target.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] append knowledge qa failed: {e}")
            return ""
        return str(target)

    def _invalidate_knowledge_cache(self) -> None:
        """知识库文件变化后清掉懒加载缓存，下一轮查询重新读取磁盘。"""
        for tm in (getattr(self.assistant, "text_model", None),):
            if tm is None:
                continue
            try:
                tm.knowledge_base = None
                tm._kb_failed = False  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass
        # 同步刷新助手配置，使 test_scenarios 立即生效
        try:
            cfg = self._load_yaml()
            if cfg and self.assistant is not None:
                self.assistant.reload_config(cfg)
        except Exception:  # noqa: BLE001
            pass

    def check_environment(self) -> str:
        import importlib
        checks = []
        gate = [("PySide6", "PySide6.QtWidgets"),
                ("QtWebEngine", "PySide6.QtWebEngineWidgets"),
                ("OpenCV", "cv2"),
                ("PyYAML", "yaml"),
                ("RapidOCR", "rapidocr_onnxruntime"),
                ("pywin32", "win32api")]
        for label, mod in gate:
            try:
                importlib.import_module(mod)
                checks.append({"label": label, "ok": True})
            except Exception:
                checks.append({"label": label, "ok": False})

        # 额外：能定位到微信窗口视为环境正常
        wechat_ok = True
        wechat_info = ""
        try:
            if self.assistant:
                info = self.assistant.finder.find()
                if info and getattr(info, "hwnd", 0):
                    wechat_info = f"已找到微信窗口 {getattr(info, 'title', '')}"
                else:
                    wechat_ok = False
                    wechat_info = "未发现微信窗口"
            checks.append({"label": "微信窗口", "ok": wechat_ok, "extra": wechat_info})
        except Exception as e:
            checks.append({"label": "微信窗口", "ok": False, "extra": f"检测异常: {e}"})

        return json.dumps({"ok": all(c["ok"] for c in checks), "items": checks},
                          ensure_ascii=False)

    def save_settings_from_ui(self, data: dict):
        """把 UI 设置写入 config.yaml 并热重载。

        支持两类输入：
        - 结构化 settings（新）：data["settings"] = {键: 值}，见 _UI_SETTINGS_MAP；
        - 旧模型四件套（兼容）：base_url/api_key/model/ocr_mode。
        两类同时存在时都写。空 payload 直接返回（不再假成功）。
        """
        s = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        # 模型字段可能来自两处：旧前端放顶层（base_url/api_key/model），
        # 新前端（collectAllSettings）放进 settings 对象。两者都兼容。
        base_url = (data.get("base_url") or s.get("base_url") or "").strip()
        api_key = (data.get("api_key") or s.get("api_key") or "").strip()
        model = (data.get("model") or s.get("model") or "").strip()
        ocr_mode = str(data.get("ocr_mode") or s.get("ocr_mode") or "").strip().lower()
        if ocr_mode not in ("local", "ai", "hybrid"):
            ocr_mode = ""

        if not s and not (base_url or api_key or model):
            self.push_log("未提供任何可保存的设置，未修改 config.yaml。", "info")
            return {"ok": False, "message": "未提供任何可保存的设置"}

        # ---- 1) 结构化设置：按映射写回各配置段 ----
        ok_all = True
        if s:
            sections: dict[str, dict] = {}
            top_level: dict = {}

            def _set(section, key, value):
                sections.setdefault(section, {})[key] = value

            def _opt(key, fn):
                if key in s:
                    fn(s[key])

            _opt("enable_rpa_send", lambda v: _set("wechat", "enable_rpa_send", bool(v)))
            _opt("foreground_mode", lambda v: _set(
                "wechat", "foreground_keyboard_mode", str(v or "efficiency")))
            _opt("merge_mode", lambda v: _set(
                "wechat", "merge_reply_segments", str(v or "merge") == "merge"))
            _opt("handoff_trouble", lambda v: _set("wechat", "handoff_trouble", bool(v)))
            _opt("recognition_mode", lambda v: _set("wechat", "recognition_mode", str(v or "double_click_pin")))
            # 思考模式开关 → 同时写 text_model 与 vision_model（两个客户端都读）
            _opt("enable_thinking", lambda v: (
                _set("text_model", "enable_thinking", bool(v)),
                _set("vision_model", "enable_thinking", bool(v)),
            ))
            # 性能与识别优化
            _opt("vision_fallback_enabled", lambda v: _set("vision_model", "fallback_enabled", bool(v)))
            _opt("fallback_min_confidence", lambda v: _set("vision_model", "fallback_min_confidence", float(v if v is not None else 0.70)))
            _opt("vision_timeout_seconds", lambda v: _set("vision_model", "timeout_seconds", int(float(v if v is not None else 18))))
            _opt("idle_sleep_seconds", lambda v: _set("wechat", "idle_sleep_seconds", float(v if v is not None else 2.0)))
            _opt("cycle_sleep_seconds", lambda v: _set("wechat", "cycle_sleep_seconds", float(v if v is not None else 0.5)))
            _opt("group_reply", lambda v: _set("wechat", "group_reply", bool(v)))
            _opt("require_mention", lambda v: _set("wechat", "require_mention", bool(v)))
            _opt("allowed_groups", lambda v: _set(
                "wechat", "allowed_groups", self._to_str_list(v)))
            _opt("mention_names", lambda v: _set(
                "wechat", "mention_names", self._to_str_list(v)))
            _opt("friend_auto_accept", lambda v: _set(
                "friend_requests", "auto_accept", bool(v)))
            _opt("friend_accept_mode", lambda v: _set(
                "friend_requests", "accept_mode", str(v or "accept_all")))
            _opt("friend_keywords", lambda v: _set(
                "friend_requests", "keyword_rules", self._to_str_list(v)))
            _opt("friend_check_interval", lambda v: _set(
                "friend_requests", "check_interval_seconds", int(float(v or 30))))
            _opt("friend_max_per_cycle", lambda v: _set(
                "friend_requests", "max_per_cycle", int(float(v or 3))))
            _opt("friend_send_welcome", lambda v: _set(
                "friend_requests", "send_welcome", bool(v)))
            _opt("friend_welcome_text", lambda v: _set(
                "friend_requests", "welcome_text", str(v or "")))
            _opt("voice_auto_convert", lambda v: _set(
                "voice_messages", "auto_convert_to_text", bool(v)))
            _opt("quote_reply", lambda v: _set("quote_reply", "enabled", bool(v)))
            _opt("fallback_chitchat", lambda v: _set("reply_fallback", "enabled", bool(v)))
            _opt("business_identity_text", lambda v: _set(
                "business", "identity_text", str(v or "")))
            _opt("ui_auto_start", lambda v: _set("ui", "auto_start", bool(v)))
            _opt("ui_auto_update_check", lambda v: _set("ui", "auto_update_check", bool(v)))
            _opt("ui_screenshot_retention", lambda v: _set(
                "ui", "screenshot_retention_days", int(float(v or 7))))
            # 低置信度审核开关 → 顶层 min_confidence_to_reply（SendGuard 读取点）
            if "min_confidence_on" in s:
                top_level["min_confidence_to_reply"] = 0.6 if s["min_confidence_on"] else 0.0

            for section, fields in sections.items():
                if not self._patch_sections_in_yaml({section: fields}):
                    ok_all = False
            for key, value in top_level.items():
                if not self._set_top_level_in_yaml(key, value):
                    ok_all = False

        # ---- 2) 旧模型四件套（有值才写，避免空值清掉配置） ----
        if base_url or api_key or model:
            ok1 = self._patch_text_model_in_yaml(base_url, api_key, model)
            ok2 = self._patch_vision_model_in_yaml(
                base_url, api_key, model, ocr_mode or "hybrid")
            if not (ok1 and ok2):
                ok_all = False

        if not ok_all:
            self.push_log("设置保存失败（写入 config.yaml 出错）。", "error")
            return {"ok": False, "message": "写入 config.yaml 失败"}

        self.push_log("设置已保存到 config.yaml。", "success")

        # 热重载客户端，使新配置即时生效（传 dict，避免 load_config 返回 Settings 的坑）
        if self.assistant is not None:
            try:
                cfg = self._load_yaml()
                if cfg:
                    self.assistant.reload_config(cfg)
                    self.push_log("配置已即时生效。", "success")
            except Exception as e:  # noqa: BLE001
                self.push_log(f"配置已保存，但热重载失败，重启后生效: {e}", "warning")
        return {"ok": True, "message": "设置已保存"}

    @staticmethod
    def _to_str_list(value) -> list:
        """把文本框（换行分隔）或列表统一转为去重保序的字符串列表。"""
        if isinstance(value, str):
            items = value.splitlines()
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        out, seen = [], set()
        for x in items:
            name = str(x or "").strip()
            if name and name not in seen:
                out.append(name)
                seen.add(name)
        return out

    def _load_yaml(self) -> dict:
        """读取 config.yaml 返回 dict（安全修复#2：返回已展开的配置）。

        与 webview_window._load_yaml 对齐：`.env` 的 `${VAR}` 展开成真值、
        空 api_key 用环境变量兜底，避免把字面量占位符当密钥用。
        """
        try:
            import yaml
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                if not isinstance(raw, dict):
                    return {}
                try:
                    from ..config.settings import expand_env_config
                    return expand_env_config(raw)
                except Exception:  # noqa: BLE001
                    return raw
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] load config.yaml failed: {e}")
        return {}

    @staticmethod
    def _api_key_for_config(api_key: str) -> str:
        """把前端传来的 key 转成 config.yaml 该写的值（占位符，非明文）。"""
        try:
            from ..config.secret_store import to_env_placeholder
        except Exception:  # noqa: BLE001
            try:
                from src.config.secret_store import to_env_placeholder
            except Exception:  # noqa: BLE001
                return api_key
        try:
            return to_env_placeholder(api_key) or api_key
        except Exception:  # noqa: BLE001
            return api_key

    def _patch_text_model_in_yaml(self, base_url: str, api_key: str, model: str) -> bool:
        """就地更新 config.yaml 的 text_model 段（base_url/api_key/model），
        保留其余内容与注释。只写非空字段，避免空值清掉已有配置。"""
        fields = {}
        if base_url:
            fields["base_url"] = base_url
        if api_key:
            fields["api_key"] = self._api_key_for_config(api_key)
        if model:
            fields["model"] = model
        if not fields:
            return True
        return self._patch_block_in_yaml("text_model:", fields)

    def _patch_vision_model_in_yaml(self, base_url: str, api_key: str, model: str,
                                     ocr_mode: str = "hybrid") -> bool:
        """就地更新 config.yaml 的 vision_model 段（base_url/api_key/model/ocr_mode），
        并自动把 provider/api_style 切换到 openai_compatible 以启用 AI OCR 兜底。
        只写非空字段，避免空值清掉已有配置。"""
        fields = {}
        if base_url:
            fields["base_url"] = base_url
        if api_key:
            fields["api_key"] = self._api_key_for_config(api_key)
        if model:
            fields["model"] = model
            # 有模型名即视为走 openai 兼容（不能因 base_url 空而降级 local，
            # 否则用户只改模型名时会把 provider 降坏）
            fields["provider"] = "openai_compatible"
            fields["api_style"] = "openai_compatible"
        if ocr_mode in ("local", "ai", "hybrid"):
            fields["ocr_mode"] = ocr_mode
        if not fields:
            return True
        return self._patch_block_in_yaml("vision_model:", fields)

    def _patch_block_in_yaml(self, block_key: str, fields: dict) -> bool:
        """就地更新 config.yaml 中指定段的字段，保留其余内容与注释。

        字段若已存在则替换，不存在则在段末尾追加。
        """
        try:
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                lines = f.readlines()

            in_block = False
            indent = "  "
            block_start = -1
            block_end = -1
            updated = {key: False for key in fields}

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(block_key):
                    in_block = True
                    block_start = i
                    indent = line[:len(line) - len(line.lstrip())] + "  "
                    continue
                if in_block:
                    if line and not line[0].isspace():
                        block_end = i
                        in_block = False
                        continue
                    for key, value in fields.items():
                        if stripped.startswith(f"{key}:"):
                            lines[i] = f'{indent}{key}: {self._yaml_scalar(value)}\n'
                            updated[key] = True
                            break

            if block_start < 0:
                return False

            # 在段末尾追加尚未存在的字段
            insert_pos = block_end if block_end >= 0 else len(lines)
            pending = []
            for key, value in fields.items():
                if updated.get(key):
                    continue
                pending.append(f'{indent}{key}: {self._yaml_scalar(value)}\n')
            if pending:
                lines[insert_pos:insert_pos] = pending

            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] patch config.yaml failed: {e}")
            return False

    @staticmethod
    def _yaml_scalar(value) -> str:
        """把 Python 值序列化为单行 YAML 标量/流式序列（保留注释的行级写入用）。"""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            try:
                import yaml as _yaml
                out = _yaml.safe_dump(
                    list(value), default_flow_style=True, allow_unicode=True,
                    width=10 ** 6).strip()
                # safe_dump 可能输出 "...]" 换行等形式，压成单行
                return " ".join(out.split())
            except Exception:
                return "[" + ", ".join(str(v) for v in value) + "]"
        if isinstance(value, str):
            # 含特殊字符的字符串加引号
            if value == "" or any(c in value for c in ":#{}[],&*?|>%@`\"'!\n"):
                return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
            return value
        return str(value)

    def _patch_sections_in_yaml(self, sections: dict) -> bool:
        """批量更新 config.yaml 多个顶层段的字段：{段名: {键: 值}}。

        段不存在时自动追加到文件末尾。返回是否全部成功。
        """
        ok = True
        for section, fields in sections.items():
            if not fields:
                continue
            if not self._patch_block_in_yaml(f"{section}:", fields):
                # 段不存在 → 追加新段
                try:
                    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
                    with open(cfg_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{section}:\n")
                        for key, value in fields.items():
                            f.write(f"  {key}: {self._yaml_scalar(value)}\n")
                except Exception as e:  # noqa: BLE001
                    _term(f"[ui] append section {section} failed: {e}")
                    ok = False
        return ok

    def _set_top_level_in_yaml(self, key: str, value) -> bool:
        """更新/追加 config.yaml 顶层标量键（如 min_confidence_to_reply）。"""
        try:
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                lines = f.readlines()
            prefix = f"{key}:"
            for i, line in enumerate(lines):
                if line.strip().startswith(prefix) and not line[0].isspace():
                    lines[i] = f"{key}: {self._yaml_scalar(value)}\n"
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    return True
            lines.append(f"{key}: {self._yaml_scalar(value)}\n")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] set top-level {key} failed: {e}")
            return False

    # -------------------------------------------------------------- helpers
    def _set_wechat_contact_blacklist(self, items: "list[str]") -> bool:
        """把名单写入 config.yaml 的 wechat.contact_blacklist（列表形式），保留其余内容与注释。

        若 wechat 段或 contact_blacklist 键不存在则新建。
        """
        try:
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                lines = f.readlines()

            # 定位 wechat: 段及其结束行
            wechat_start = -1
            wechat_indent = "  "
            wechat_end = len(lines)
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("wechat:"):
                    wechat_start = i
                    wechat_indent = line[:len(line) - len(line.lstrip())] + "  "
                    continue
                if wechat_start >= 0 and line and not line[0].isspace():
                    wechat_end = i
                    break

            if wechat_start < 0:
                # 没有 wechat 段，整体追加
                lines.append("\nwechat:\n")
                wechat_indent = "  "
                wechat_start = len(lines) - 1
                wechat_end = len(lines)

            # 在 wechat 段内定位 contact_blacklist:
            cb_start = -1
            cb_end = -1
            for i in range(wechat_start + 1, wechat_end):
                stripped = lines[i].strip()
                if stripped.startswith("contact_blacklist:"):
                    cb_start = i
                    # 找到该列表的结束（下一个非缩进更深、或不以 - 开头的行）
                    for j in range(i + 1, wechat_end):
                        s2 = lines[j]
                        if not s2.strip():
                            continue
                        if s2[0].isspace() and (s2.strip().startswith("-") or s2.strip().startswith("#")):
                            continue
                        cb_end = j
                        break
                    if cb_end < 0:
                        cb_end = wechat_end
                    break

            new_block = [f"{wechat_indent}contact_blacklist:\n"]
            for name in items:
                new_block.append(f'{wechat_indent}  - "{name}"\n')

            if cb_start >= 0:
                lines[cb_start:cb_end] = new_block
            else:
                # 插入到 wechat 段开头下方
                lines[wechat_start + 1:wechat_start + 1] = new_block

            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception as e:  # noqa: BLE001
            _term(f"[ui] set contact_blacklist failed: {e}")
            return False

    def _run_js(self, js: str):
        if self.view.page() is not None:
            try:
                self.view.page().runJavaScript(js)
            except Exception:
                pass

    def push_log(self, msg: str, level: str = "info"):
        _term("[%s] %s" % (level, msg))
        self._run_js("window.pushLog(%s);" % json.dumps({"text": msg, "level": level},
                                                        ensure_ascii=False))

    def closeEvent(self, event):
        if self.assistant and self.assistant.is_running:
            self.assistant.stop_loop()
        event.accept()