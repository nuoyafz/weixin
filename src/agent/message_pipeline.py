"""消息处理流水线（MessagePipeline）—— 一条消息的生命周期编排与可观测中枢。

背景（实测缺陷）：
    UI 左侧「消息处理」面板元素齐全（状态药丸、5 个过滤芯片、队列容器），
    ``html_window._process_event`` 里也写好了 5 类事件的处理分支，但
    ``observe_service`` 从不发送这些事件 —— 面板"接了线但没通电"，永远停在
    "等待客户消息"，且 5 个计数器中 3 个恒为 0。

定位：
    本模块是**编排与可观测中枢**，不是新的业务实现。它不负责识别、不负责
    决策、不负责发送；只负责：
      1. 为每条消息建立 ProcessingRecord 并推进状态机；
      2. 在每个阶段跃迁时广播事件（驱动 UI 实时呈现）；
      3. 聚合终态计数（全部/待办/回复/人工/跳过）；
      4. 用消息指纹去重，避免同一条消息被反复处理。

状态机：
    detected → entering → captured → ocr → analyzing → sending ─► sent
         └──────────────┴──────────┴──────────┴─ 证据不足 ─► manual
                                                 主动放弃   ─► skipped
                                                 异常       ─► failed
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


# ---- 阶段（处理中） ----
STAGE_DETECTED = "detected"
STAGE_ENTERING = "entering"
STAGE_CAPTURED = "captured"
STAGE_OCR = "ocr"
STAGE_ANALYZING = "analyzing"
STAGE_SENDING = "sending"

# ---- 终态 ----
STATUS_RUNNING = "running"
STATUS_SENT = "sent"
STATUS_SKIPPED = "skipped"
STATUS_MANUAL = "manual"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = (STATUS_SENT, STATUS_SKIPPED, STATUS_MANUAL, STATUS_FAILED)

# 阶段 -> 人类可读（UI 展示）
STAGE_LABELS = {
    STAGE_DETECTED: "识别到未读",
    STAGE_ENTERING: "点击进入会话",
    STAGE_CAPTURED: "截图完成",
    STAGE_OCR: "提取聊天记录",
    STAGE_ANALYZING: "分析内容/生成回复",
    STAGE_SENDING: "执行 RPA 发送",
}

STATUS_LABELS = {
    STATUS_RUNNING: "处理中",
    STATUS_SENT: "已回复",
    STATUS_SKIPPED: "已跳过",
    STATUS_MANUAL: "待人工",
    STATUS_FAILED: "失败",
}


@dataclass
class ProcessingRecord:
    """一条消息的处理记录。"""
    id: str = ""
    contact: str = ""
    stage: str = STAGE_DETECTED
    status: str = STATUS_RUNNING
    customer_text: str = ""
    reply_text: str = ""
    reason: str = ""
    confidence: float = 0.0
    started_at: float = 0.0
    updated_at: float = 0.0

    @property
    def stage_label(self) -> str:
        if self.status != STATUS_RUNNING:
            return STATUS_LABELS.get(self.status, self.status)
        return STAGE_LABELS.get(self.stage, self.stage)

    def to_event(self) -> dict:
        return {
            "id": self.id,
            "contact": self.contact,
            "stage": self.stage,
            "status": self.status,
            "stage_label": self.stage_label,
            "text": self.customer_text,
            "reply": self.reply_text,
            "reason": self.reason,
            "confidence": self.confidence,
            "ts": self.updated_at,
        }


class MessagePipeline:
    """持有处理中/已结束的记录，负责事件广播与计数聚合。"""

    def __init__(self, emit: Optional[Callable[[str, dict], None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 dedup_window: int = 300):
        self._emit = emit
        self._log = log
        self._dedup_window = int(dedup_window or 300)   # 秒
        self._lock = threading.RLock()
        self._records: Dict[str, ProcessingRecord] = {}
        self._order: List[str] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def begin(self, contact: str = "", seed: str = "",
              **fields: Any) -> ProcessingRecord:
        """开启一条消息的处理（阶段 detected）。"""
        now = time.time()
        rid = self._fingerprint(contact, seed, now)
        with self._lock:
            existing = self._records.get(rid)
            if existing is not None and existing.status in TERMINAL_STATUSES:
                # 去重窗口内的同一条消息：不重复处理
                if now - existing.updated_at < self._dedup_window:
                    self._log_msg(f"pipeline dedup hit id={rid} status={existing.status}")
                    return existing
            rec = ProcessingRecord(
                id=rid, contact=contact or "",
                stage=STAGE_DETECTED, status=STATUS_RUNNING,
                started_at=now, updated_at=now, **fields,
            )
            self._records[rid] = rec
            if rid not in self._order:
                self._order.append(rid)
        self._broadcast(rec)
        return rec

    def advance(self, rec: Optional[ProcessingRecord], stage: str,
                **fields: Any) -> None:
        """推进到下一个阶段（记录仍在进行中）。"""
        if rec is None:
            return
        with self._lock:
            rec.stage = stage
            rec.status = STATUS_RUNNING
            for k, v in fields.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = time.time()
        self._broadcast(rec)
        # 兼容派生事件
        if stage == STAGE_OCR and rec.customer_text:
            self._emit_event("message_received", {
                "id": rec.id, "contact": rec.contact,
                "content": rec.customer_text, "confidence": rec.confidence,
            })
        if rec.reply_text and stage == STAGE_ANALYZING:
            self._emit_event("reply_generated", {
                "id": rec.id, "contact": rec.contact,
                "content": rec.reply_text,
            })

    def finish(self, rec: Optional[ProcessingRecord], status: str,
               reason: str = "", **fields: Any) -> None:
        """结束一条消息的处理（终态）。"""
        if rec is None:
            return
        with self._lock:
            rec.status = status
            rec.reason = reason
            for k, v in fields.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = time.time()
        self._broadcast(rec)
        # 兼容派生事件
        if status == STATUS_SENT:
            self._emit_event("reply_sent", {
                "id": rec.id, "contact": rec.contact,
                "content": rec.reply_text,
            })
        elif status in (STATUS_SKIPPED, STATUS_MANUAL, STATUS_FAILED):
            self._emit_event("message_skipped", {
                "id": rec.id, "contact": rec.contact,
                "content": rec.customer_text, "reason": reason or status,
            })
        self.emit_counts()

    # ------------------------------------------------------------------
    # 查询 / 计数
    # ------------------------------------------------------------------
    def counts(self) -> dict:
        with self._lock:
            total = len(self._records)
            pending = sent = manual = skipped = 0
            for rec in self._records.values():
                if rec.status == STATUS_RUNNING:
                    pending += 1
                elif rec.status == STATUS_SENT:
                    sent += 1
                elif rec.status == STATUS_MANUAL:
                    manual += 1
                else:                      # skipped / failed
                    skipped += 1
            return {"total": total, "pending": pending, "sent": sent,
                    "manual": manual, "skipped": skipped}

    def recent(self, limit: int = 50) -> List[dict]:
        with self._lock:
            ids = self._order[-max(1, limit):]
            return [self._records[i].to_event() for i in ids if i in self._records]

    def get(self, record_id: str) -> Optional[ProcessingRecord]:
        with self._lock:
            return self._records.get(record_id)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._order.clear()
        self.emit_counts()

    # ------------------------------------------------------------------
    def emit_counts(self) -> None:
        self._emit_event("msg_counts", self.counts())

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _broadcast(self, rec: ProcessingRecord) -> None:
        self._emit_event("msg_pipeline", rec.to_event())

    def _emit_event(self, event_type: str, data: dict) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event_type, data)
        except Exception:
            pass

    def _log_msg(self, msg: str) -> None:
        if self._log is None:
            return
        try:
            self._log(f"[pipeline] {msg}")
        except Exception:
            pass

    @staticmethod
    def _fingerprint(contact: str, seed: str, now: float) -> str:
        """消息指纹：联系人 + 内容 + 时间桶，用于去重。"""
        bucket = int(now // 60)
        norm = " ".join((seed or "").split())[:120]
        raw = f"{contact}|{norm}|{bucket}"
        return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()[:12]


__all__ = [
    "MessagePipeline", "ProcessingRecord",
    "STAGE_DETECTED", "STAGE_ENTERING", "STAGE_CAPTURED",
    "STAGE_OCR", "STAGE_ANALYZING", "STAGE_SENDING",
    "STATUS_RUNNING", "STATUS_SENT", "STATUS_SKIPPED",
    "STATUS_MANUAL", "STATUS_FAILED",
]
