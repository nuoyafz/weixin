"""已回复消息去重（2026-09-10 修复 #2）。

问题背景
--------
点击未读链路失效后，助手会每轮重读「当前已打开的同一个会话」，OCR 判定
should_reply=True，于是对**同一条消息反复调用 LLM**（单次 20~30s），造成
无意义的慢调用与 API 消耗（observe_service.py:2174 注释原话）。

修复思路
--------
以「联系人 + 最新消息内容」做指纹，记录**已成功发送**的回复。下一轮若检测到
同一指纹，直接跳过 LLM 生成与发送。注意：

- 仅当**发送成功**才入去重集；被 evidence_gate 拦截 / handoff / 发送失败的不入集，
  下一轮仍可正常重试，不会误伤。
- 指纹按消息内容（非回复文本）计算，因此「同一条客户消息」无论走关键词/FAQ/
  视觉/文本哪条回复路径都能命中。
- 带 TTL（默认 6h）与条目上限，避免无限增长；持久化到磁盘，重启后不重烧。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

_DEFAULT_TTL = 21600.0  # 6 小时
_DEFAULT_MAX = 1000


class RepliedDedup:
    def __init__(
        self,
        path: str,
        ttl: float = _DEFAULT_TTL,
        max_entries: int = _DEFAULT_MAX,
    ) -> None:
        self._path = path
        self._ttl = float(ttl)
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._data: dict[str, float] = {}
        self.load()

    # ------------------------------------------------------------------
    @staticmethod
    def fingerprint(contact: str, message: str) -> str:
        h = hashlib.sha1()
        # 联系人大小写不敏感（微信昵称 OCR 可能大小写抖动）
        h.update((contact or "").strip().lower().encode("utf-8"))
        h.update(b"\x00")
        # 消息保留大小写，仅去首尾空白（避免同一句因空白差异被当成新消息）
        h.update((message or "").strip().encode("utf-8"))
        return h.hexdigest()

    def should_skip(self, contact: str, message: str) -> bool:
        if not contact or not message:
            return False
        fp = self.fingerprint(contact, message)
        with self._lock:
            ts = self._data.get(fp)
            if ts is None:
                return False
            if time.time() - ts > self._ttl:
                self._data.pop(fp, None)
                return False
            return True

    def mark_replied(self, contact: str, message: str) -> None:
        if not contact or not message:
            return
        fp = self.fingerprint(contact, message)
        with self._lock:
            self._data[fp] = time.time()
            if len(self._data) > self._max:
                expired = [
                    k for k, v in self._data.items()
                    if time.time() - v > self._ttl
                ]
                for k in expired:
                    self._data.pop(k, None)
                if len(self._data) > self._max:
                    oldest = sorted(
                        self._data.items(), key=lambda kv: kv[1]
                    )[: len(self._data) - self._max]
                    for k, _ in oldest:
                        self._data.pop(k, None)

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            with open(self._path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            data = obj.get("entries", {}) if isinstance(obj, dict) else {}
            now = time.time()
            self._data = {
                k: float(v)
                for k, v in data.items()
                if now - float(v) <= self._ttl
            }
        except Exception:
            self._data = {}

    def save(self) -> None:
        try:
            d = os.path.dirname(self._path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": self._data}, f, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception:
            pass
