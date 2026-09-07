"""视觉补全客户端：支持消息中携带 image_url 的多模态模型调用。

构造与 text_model_client.TextModelClient 保持一致（复用 urllib 风格），
在 user 消息中按 OpenAI 多模态格式注入图片：
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "...", "detail": "high"}},
    ]}
统一返回 ReplyResult。
"""
import base64
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from ..reply.engine import ReplyResult


class VisionModelClient:
    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "", temperature: float = 0.1,
                 max_tokens: int = 1800, timeout_seconds: int = 110,
                 image_detail: str = "high"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.image_detail = image_detail

    # ---------- 对外接口 ----------

    def complete_with_images(self, prompt: str, image_urls: List[str],
                             system_prompt: str = "",
                             history: Optional[List[Dict]] = None,
                             stream: bool = False) -> ReplyResult:
        """以文本 + 若干图片发起视觉补全，返回 ReplyResult。"""
        messages = self.build_messages(prompt, image_urls, system_prompt, history)
        return self.complete_messages(messages, stream=stream)

    def analyze_wechat_screen(self, image_path: str, window_info=None,
                              prompt: str = "", system_prompt: str = ""):
        """分析微信屏幕截图（兼容原项目 VisionModelClient.analyze_wechat_screen）。

        image_path 为截图文件路径；window_info 为可选窗口元信息，拼接到 prompt。
        """
        path = Path(image_path)
        if not path.exists():
            return ReplyResult(success=False, error=f"Image not found: {image_path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        image_url = f"data:image/png;base64,{b64}"
        if not prompt:
            prompt = (
                "请分析这张微信界面截图，识别其中的聊天内容、联系人、"
                "未读消息、按钮等关键信息，并以 JSON 形式输出。"
            )
        if window_info is not None:
            try:
                prompt = prompt + f"\n窗口信息：{window_info}"
            except Exception:
                pass
        return self.complete_with_images(prompt, [image_url], system_prompt=system_prompt)

    def complete_messages(self, messages: List[Dict],
                          stream: bool = False) -> ReplyResult:
        start_time = time.time()
        try:
            if stream:
                raw_content = self._stream_collect(messages)
            else:
                raw_content = self._call_api(messages)
            latency_ms = int((time.time() - start_time) * 1000)

            if not raw_content:
                return ReplyResult(success=False, error="Empty response",
                                   latency_ms=latency_ms)

            return ReplyResult(
                success=True,
                content=self._clean(raw_content),
                raw_response=raw_content,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return ReplyResult(success=False, error=str(e), latency_ms=latency_ms)

    def stream_chunks(self, messages: List[Dict]) -> Iterator[str]:
        """流式发起视觉补全，逐 token 产出文本片段。"""
        for chunk in self._call_stream(messages):
            yield chunk

    # ---------- 消息构造 ----------

    def build_messages(self, prompt: str, image_urls: List[str],
                       system_prompt: str = "",
                       history: Optional[List[Dict]] = None) -> List[Dict]:
        content: List[Dict] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for url in image_urls:
            image_part: Dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": url, "detail": self.image_detail},
            }
            content.append(image_part)

        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": content})
        return messages

    # ---------- 底层 HTTP ----------

    def _request(self, payload: Dict[str, Any]) -> "urlopen":
        if not self.base_url:
            raise ValueError("API base_url is not configured")
        if not self.model:
            raise ValueError("API model is not configured")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(endpoint, data=data, headers=headers, method="POST")
        return urlopen(req, timeout=self.timeout_seconds)

    def _call_api(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            with self._request(payload) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
            return self._extract_content(resp_data)
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {error_body}") from e
        except URLError as e:
            raise Exception(f"Network error: {e.reason}") from e

    def _call_stream(self, messages: List[Dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        response = None
        try:
            response = self._request(payload)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {error_body}") from e
        except URLError as e:
            raise Exception(f"Network error: {e.reason}") from e
        finally:
            if response is not None:
                response.close()

    def _stream_collect(self, messages: List[Dict]) -> str:
        return "".join(self._call_stream(messages))

    def _extract_content(self, resp_data: Dict[str, Any]) -> str:
        if "choices" in resp_data and len(resp_data["choices"]) > 0:
            choice = resp_data["choices"][0]
            message = choice.get("message") or {}
            return message.get("content", "")
        if "error" in resp_data:
            raise Exception(f"API Error: {resp_data['error']}")
        return ""

    def _clean(self, text: str) -> str:
        return text.strip()


def client_from_config(cfg) -> VisionModelClient:
    """由 ModelConfig 构造 VisionModelClient。"""
    return VisionModelClient(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout_seconds=cfg.timeout_seconds,
    )