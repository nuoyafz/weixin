"""OpenAICompatibleClient - generic OpenAI-compatible API client.

Aligned with original app.ai.openai_compatible_client - provides a
unified client for any OpenAI-compatible API endpoint.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional


class ModelCallError(RuntimeError):
    """Error raised when a model call fails."""

    def __init__(self, message: str, trace_id: str = ""):
        super().__init__(str(message))
        self.trace_id = trace_id


def _quote_can_close_json_string(text: str, quote_index: int) -> bool:
    """判断当前英文引号是否是 JSON 字符串的真正结束符。"""
    cursor = quote_index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    following = text[cursor]
    if following in frozenset({'}', ':', ']'}):
        return True
    if following == ',':
        return True
    return False


def _repair_unescaped_json_string_quotes(text: str) -> str:
    """保守修复模型在 JSON 字符串内部写入的未转义英文引号。

    只处理能够通过上下文明确判断为"字符串正文"的引号；修复后仍会交给
    标准 json.loads 校验，无法完整解析时继续按失败处理，不会猜字段。
    """
    output: list[str] = []
    in_string = False
    escaped = False
    repaired = False

    for i, char in enumerate(text):
        if escaped:
            output.append(char)
            escaped = False
            continue

        if char == '\\':
            output.append(char)
            escaped = True
            continue

        if char == '"':
            if not in_string:
                in_string = True
                output.append(char)
            elif _quote_can_close_json_string(text, i):
                in_string = False
                output.append(char)
            else:
                output.append('\\"')
                repaired = True
        else:
            output.append(char)

    return ''.join(output)


def _load_json_candidate(candidate: str) -> dict[str, Any] | None:
    """Try to load JSON, with repair fallback."""
    try:
        return json.loads(candidate)
    except Exception:
        repaired = _repair_unescaped_json_string_quotes(candidate)
        if repaired == candidate:
            return None
        try:
            return json.loads(repaired)
        except Exception:
            return None


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a string, with repair."""
    text = content.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    parsed = _load_json_candidate(text)
    if parsed is not None:
        return parsed

    start = text.find('{')
    if start < 0:
        return None

    depth = 0
    in_str = False
    escaped = False
    outer_candidate_closed = False

    for i, ch in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"' and not outer_candidate_closed:
            if not in_str:
                in_str = True
            elif _quote_can_close_json_string(text, i):
                in_str = False
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                parsed = _load_json_candidate(candidate)
                if parsed is not None:
                    return parsed
                outer_candidate_closed = True

    if not outer_candidate_closed:
        return None

    return None


class OpenAICompatibleClient:
    """Generic client for OpenAI-compatible API endpoints."""

    def __init__(self, config=None, logger=None):
        self._config = config or {}
        self._logger = logger
        self._api_key = self._config.get("api_key", "")
        self._base_url = self._config.get("base_url", "https://api.openai.com/v1")
        self._model = self._config.get("model", "gpt-4o-mini")
        self._temperature = self._config.get("temperature", 0.7)
        self._max_tokens = self._config.get("max_tokens", 4096)

    def chat(self, messages: list[dict[str, Any]],
             model: str = None, base_url: str = None, api_key: str = None,
             use_response_format_json: bool = False, **kwargs) -> dict[str, Any]:
        """Send a chat completion request.

        关键修复：调用方（model_router）会把「任务级配置」(text_model/vision_model
        段) 传入，里面含正确的 base_url / api_key / model。此前 chat 只用
        self._base_url（构造时取自完整 config，顶层没有 base_url -> 回退成
        api.openai.com），从不使用调用方传入的 base_url，导致请求发往错误端点、
        鉴权失败、content 永远为空 -> 上层拿到空回复。
        现在优先使用调用方显式传入的 base_url / api_key / model。
        """
        base_url = base_url or self._base_url
        api_key = api_key or self._api_key
        model = model or self._model
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        result = {
            "success": False,
            "content": "",
            "model": model,
            "usage": {},
        }

        try:
            import requests
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_response_format_json:
                payload["response_format"] = {"type": "json_object"}
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                result["success"] = True
                result["content"] = choice.get("message", {}).get("content", "")
                result["usage"] = data.get("usage", {})
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def chat_with_image(self, messages: list[dict[str, Any]],
                        image_base64: str = "",
                        model: str = None) -> dict[str, Any]:
        """Send a chat request with an image."""
        if image_base64:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    msg["content"] = [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"}},
                    ]
        return self.chat(messages, model=model)