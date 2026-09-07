"""ModelCallTrace - trace and log AI model call outcomes.

Aligned with original app.ai.model_call_trace - records model call
results for debugging, analytics, and cost tracking.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _next_sequence_ns() -> int:
    """Generate a nanosecond-resolution sequence number."""
    return time.time_ns()


def sanitize_endpoint(url: str) -> str:
    """Sanitize a URL endpoint for logging."""
    return str(url or "").split("?")[0][:200]


def sanitize_text(value: str) -> str:
    """Sanitize text for logging."""
    return str(value or "")[:2000]


def messages_to_visible_text(messages: list[dict[str, Any]]) -> str:
    """Convert messages to a visible text representation."""
    parts: list[str] = []
    for msg in (messages or []):
        role = str(msg.get("role", "") or "")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t == "text":
                        texts.append(str(item.get("text", "")))
                    elif t in frozenset({"input_image", "image", "image_url"}):
                        texts.append("[图片 1 张，日志不保存图片编码]")
            content = " ".join(texts)
        content = str(content or "")
        parts.append(f"[{role}] {content[:300]}")
    return "\n".join(parts)


def friendly_model_error(error: str, status_code: int = 0) -> str:
    """Make model error messages more user-friendly."""
    msg = str(error or "").strip()
    if not msg:
        return "未知错误"
    if status_code:
        return f"HTTP {status_code}: {msg[:200]}"
    return msg[:200]


def record_model_call(
    purpose: str = "",
    provider: str = "",
    model: str = "",
    endpoint: str = "",
    request_text: str = "",
    status: str = "",
    started_at: float = 0.0,
    response_text: str = "",
    status_code: int = 0,
    usage: dict[str, Any] | None = None,
    trace_dir: str = "",
    summary_text: str = "",
    outcome_summary: str = "",
    error_text: str = "",
) -> str:
    """Record a model call to the trace log.

    Returns the call_id (trace_id) for later status updates.
    """
    if not trace_dir:
        trace_dir = str(
            Path(__file__).resolve().parent.parent.parent / "data" / "traces"
        )

    try:
        os.makedirs(trace_dir, exist_ok=True)
    except Exception:
        pass

    call_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_next_sequence_ns():020d}"
    call_id = call_id[:64]

    record = {
        "id": call_id,
        "purpose": sanitize_text(purpose),
        "provider": sanitize_text(provider),
        "model": sanitize_text(model),
        "endpoint": sanitize_endpoint(endpoint),
        "request_text": sanitize_text(request_text),
        "status": sanitize_text(status),
        "started_at": started_at,
        "response_text": sanitize_text(response_text),
        "status_code": status_code,
        "usage": usage or {},
        "outcome_status": "",
        "outcome_label": "",
        "outcome_summary": sanitize_text(outcome_summary or summary_text),
        "error_text": sanitize_text(error_text),
    }

    tmp_path = os.path.join(trace_dir, f"{call_id}.tmp")
    final_path = os.path.join(trace_dir, f"{call_id}.json")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return call_id


def update_model_call_outcome(
    call_id: str,
    outcome: str,
    model: str = "",
    tokens: int = 0,
    cost: float = 0.0,
    duration_ms: float = 0.0,
    error: str = "",
    trace_dir: str = "",
) -> None:
    """Attach the business-level result to an already persisted API call.

    A HTTP 200 response only means the platform answered. The returned content
    can still be empty or unusable, so the UI needs this second-stage outcome.
    """
    if not trace_dir:
        trace_dir = str(
            Path(__file__).resolve().parent.parent.parent / "data" / "traces"
        )

    if not call_id or not os.path.isdir(trace_dir):
        return

    json_path = os.path.join(trace_dir, f"{call_id}.json")
    if not os.path.isfile(json_path):
        for fname in os.listdir(trace_dir):
            if fname.startswith(call_id[:8]) and fname.endswith(".json"):
                json_path = os.path.join(trace_dir, fname)
                break
        if not os.path.isfile(json_path):
            return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return

    record["outcome_status"] = sanitize_text(outcome)
    record["outcome_label"] = sanitize_text(outcome)
    record["outcome_summary"] = sanitize_text(error or "")

    tmp_path = json_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, json_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class ModelCallTrace:
    """Trace model call outcomes for debugging and analytics."""

    def __init__(self, trace_dir: str = ""):
        self._trace_dir = trace_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "traces")
        self._calls: list[dict[str, Any]] = []

    def record(self, call_id: str, outcome: str, **kwargs) -> None:
        """Record a model call."""
        start = time.time()
        update_model_call_outcome(
            call_id=call_id,
            outcome=outcome,
            trace_dir=self._trace_dir,
            **kwargs,
        )
        duration = (time.time() - start) * 1000
        self._calls.append({
            "call_id": call_id,
            "outcome": outcome,
            "duration_ms": duration,
            **kwargs,
        })

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics."""
        total = len(self._calls)
        if total == 0:
            return {"total": 0}
        successes = sum(1 for c in self._calls if c.get("outcome") == "success")
        total_tokens = sum(c.get("tokens", 0) for c in self._calls)
        total_cost = sum(c.get("cost", 0.0) for c in self._calls)
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }

    def clear(self) -> None:
        self._calls.clear()