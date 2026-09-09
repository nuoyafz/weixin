"""模型路由：按 provider 将调用路由到正确的客户端。

支持 OpenAI 兼容、Gemini 和本地（local）provider。
包含视觉调试保存、JSON 响应格式重试、Gemini 视觉调用等完整功能。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .openai_compatible_client import OpenAICompatibleClient, ModelCallError, extract_json_object
from .model_call_trace import record_model_call, update_model_call_outcome

OPENAI_COMPATIBLE_PROVIDERS = frozenset({
    'relay',
    'openai',
    'builtin',
    'openai_compatible',
    'local_openai_compatible',
    'relay_openai_compatible',
})


class ModelRouter:

    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger
        self.openai_client = OpenAICompatibleClient(config, logger=logger)

    def call_vision_json(
        self,
        cfg: Dict[str, Any] = None,
        prompt: str = '',
        image_path: str = '',
        detail_image_paths: List[str] | None = None,
        image_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        provider = str(cfg.get('provider', 'openai_compatible') or 'openai_compatible').strip().lower()
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            return self._call_openai_compatible_vision(cfg, prompt, image_path, detail_image_paths, image_labels)
        if provider == 'gemini':
            return self._call_gemini_vision(cfg, prompt, image_path, detail_image_paths, image_labels)
        if provider == 'local':
            raise ModelCallError(
                'vision: provider=local is reserved for explicit local fallback '
                'and does not call a remote vision model.'
            )
        raise ModelCallError(f"vision: unsupported provider '{provider}'.")

    def call_text_json(
        self,
        cfg: Dict[str, Any] = None,
        system_prompt: str = '',
        user_prompt: str = '',
    ) -> Dict[str, Any]:
        provider = str(cfg.get('provider', 'openai_compatible') or 'openai_compatible').strip().lower()
        trace_purpose = str(cfg.get('_trace_purpose') or 'text_reply')
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            try:
                result = self._chat_openai(messages, cfg, 0.2, 1200, trace_purpose)
                return self._retry_with_response_format_if_needed(
                    cfg, result, trace_purpose,
                    lambda retry_cfg: self._chat_openai(
                        messages, retry_cfg, 0.2, 1200, trace_purpose),
                )
            except ModelCallError as exc:
                if not self._should_retry_without_response_format(exc, cfg):
                    raise
                retry_cfg = dict(cfg)
                retry_cfg['use_response_format_json'] = False
                if self.logger:
                    self.logger(f'text_reply retry_without_response_format error={exc}')
                return self._chat_openai(messages, retry_cfg, 0.2, 1200, trace_purpose)
        if provider == 'gemini':
            return self._call_gemini_text(cfg, system_prompt + '\n\n' + user_prompt, purpose=trace_purpose)
        if provider == 'local':
            raise ModelCallError(
                'text: provider=local is reserved and does not call a remote text model.'
            )
        raise ModelCallError(f"text: unsupported provider '{provider}'.")

    # ------------------------------------------------------------------
    # OpenAI 兼容调用（统一参数拼装）
    # ------------------------------------------------------------------
    def _chat_openai(self, messages: List[Dict[str, Any]], cfg: Dict[str, Any],
                     default_temperature: float, default_max_tokens: int,
                     purpose: str) -> Dict[str, Any]:
        """统一封装 openai_client.chat 的参数拼装。

        修复#9：原先「正常调用 / JSON 重试 / 无 response_format 重试」三处
        重复拼同样 8 个参数，改一处要改三处。收敛到单点后便于统一维护，
        顺便把超时真正传递下去（修复#3——原先从未传 timeout，落到 client
        默认，也就是硬编码的 120s）。
        """
        return self.openai_client.chat(
            messages,
            model=cfg.get('model'),
            base_url=cfg.get('base_url'),
            api_key=cfg.get('api_key'),
            use_response_format_json=bool(cfg.get('use_response_format_json', False)),
            temperature=float(cfg.get('temperature', default_temperature) or default_temperature),
            max_tokens=int(cfg.get('max_tokens', default_max_tokens) or default_max_tokens),
            timeout=int(cfg.get('timeout_seconds', 45) or 45),
            force_json=True,
            purpose=purpose,
        )

    # ------------------------------------------------------------------
    # OpenAI 兼容视觉调用
    # ------------------------------------------------------------------
    def _call_openai_compatible_vision(
        self,
        cfg: Dict[str, Any] = None,
        prompt: str = '',
        image_path: str = '',
        detail_image_paths: List[str] | None = None,
        image_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        trace_purpose = str(cfg.get('_trace_purpose') or 'vision_screen')
        all_image_paths = [str(image_path)] + [
            str(path).strip() for path in (detail_image_paths or []) if str(path).strip()
        ]
        image_payloads = [self._image_b64(path) for path in all_image_paths]
        detail = str(cfg.get('image_detail', 'high') or 'high')
        content = [{'type': 'text', 'text': prompt}]
        image_content_format = str(cfg.get('image_content_format', 'openai_image_url') or 'openai_image_url')

        for index, image_b64 in enumerate(image_payloads):
            label = self._vision_image_label(index, image_labels)
            content.append({'type': 'text', 'text': label})
            image_url = f'data:image/png;base64,{image_b64}'
            if image_content_format == 'openai_image_url_string':
                content.append({'type': 'image_url', 'image_url': image_url})
            else:
                content.append({'type': 'image_url', 'image_url': {'url': image_url, 'detail': detail}})

        call_cfg = dict(cfg)
        if bool(call_cfg.get('prefer_response_format_json_first', True)):
            call_cfg['use_response_format_json'] = True
        user_msg = [{'role': 'user', 'content': content}]

        try:
            result = self._chat_openai(user_msg, call_cfg, 0.1, 1800, trace_purpose)
            result = self._retry_with_response_format_if_needed(
                call_cfg, result, trace_purpose,
                lambda retry_cfg: self._chat_openai(
                    user_msg, retry_cfg, 0.1, 1800, trace_purpose),
            )
        except ModelCallError as exc:
            if not self._should_retry_without_response_format(exc, call_cfg):
                raise
            retry_cfg = dict(call_cfg)
            retry_cfg['use_response_format_json'] = False
            if self.logger:
                self.logger(f'vision_screen retry_without_response_format error={exc}')
            result = self._chat_openai(user_msg, retry_cfg, 0.1, 1800, trace_purpose)

        self._save_vision_debug(cfg, prompt, image_path, all_image_paths, image_labels, image_payloads, result)
        return result

    def _save_vision_debug(
        self, cfg: Dict[str, Any], prompt: str, image_path: str,
        all_image_paths: List[str], image_labels: List[str] | None,
        image_payloads: List[str], result: Dict[str, Any],
    ) -> None:
        if cfg.get('_skip_vision_debug'):
            return
        try:
            from datetime import datetime
            from pathlib import Path as _Path
            from ..config import DATA_DIR
            debug_dir = _Path(DATA_DIR) / 'logs' / 'vision_debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            debug_file = debug_dir / f'vision_{ts}.json'
            debug_payload = {
                'timestamp': ts,
                'provider': cfg.get('provider'),
                'model': cfg.get('model'),
                'base_url': cfg.get('base_url'),
                'image_path': str(image_path),
                'detail_image_paths': all_image_paths[1:],
                'image_labels': [
                    self._vision_image_label(i, image_labels)
                    for i in range(len(all_image_paths))
                ],
                'image_b64_size': len(image_payloads[0]) if image_payloads else 0,
                'image_b64_sizes': [len(p) for p in image_payloads],
                'prompt': prompt,
                'response_content': result.get('content', '') if isinstance(result, dict) else str(result),
                'usage': (result.get('meta') or {}).get('usage') if isinstance(result, dict) else {},
                'max_tokens': (result.get('meta') or {}).get('max_tokens') if isinstance(result, dict) else None,
                'response_format_json_requested': (
                    (result.get('meta') or {}).get('response_format_json_requested')
                    if isinstance(result, dict) else None
                ),
            }
            debug_file.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            if self.logger:
                self.logger(f'vision_debug_saved path={debug_file}')
            # 修复#6：视觉调试文件含 prompt 与客户会话内容，原先只写不清，
            # 会无限堆积且留存隐私。这里按保留天数清理过期文件。
            self._prune_vision_debug(debug_dir)
        except Exception as exc:
            if self.logger:
                self.logger(f'vision_debug_save_failed error={exc}')

    @staticmethod
    def _prune_vision_debug(debug_dir: '_Path | Path', keep_days: int = 7) -> None:
        """删除超过保留天数的视觉调试文件（默认 7 天）。"""
        try:
            import time as _time
            cutoff = _time.time() - max(1, int(keep_days)) * 86400
            for f in Path(debug_dir).glob('vision_*.json'):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except Exception:
                    continue
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gemini 调用
    # ------------------------------------------------------------------
    def _call_gemini_vision(
        self,
        cfg: Dict[str, Any] = None,
        prompt: str = '',
        image_path: str = '',
        detail_image_paths: List[str] | None = None,
        image_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        import requests
        api_key = str(cfg.get('api_key', '') or '').strip()
        model = str(cfg.get('model', 'gemini-2.0-flash') or 'gemini-2.0-flash').strip()
        # 超时默认值收敛(原110)：与主回复链路口径一致，避免单次调用拖太久。
        timeout = int(cfg.get('timeout_seconds', 45) or 45)
        trace_purpose = str(cfg.get('_trace_purpose') or 'vision_screen')

        if not api_key:
            raise ModelCallError('gemini vision: api_key is empty.')

        all_image_paths = [str(image_path)] + [
            str(p).strip() for p in (detail_image_paths or []) if str(p).strip()
        ]
        parts = [{'text': prompt}]
        for index, path in enumerate(all_image_paths):
            label = self._vision_image_label(index, image_labels)
            parts.append({'text': label})
            b64 = self._image_b64(path)
            parts.append({'inline_data': {'mime_type': 'image/png', 'data': b64}})

        body = {
            'contents': [{'parts': parts}],
            'generationConfig': {
                'temperature': float(cfg.get('temperature', 0.1) or 0.1),
                'maxOutputTokens': int(cfg.get('max_tokens', 1800) or 1800),
            },
        }
        # 安全修复#2：api_key 不再拼进 URL query（会在网关/代理/服务端访问日志留痕），
        # 改用官方推荐的 x-goog-api-key 请求头传递。
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key,
        }
        url = (
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
        )

        started_at = time.perf_counter()
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                record_model_call(
                    purpose=trace_purpose, provider='gemini', model=model,
                    endpoint=url, request_text=prompt, status='failed',
                    started_at=started_at, error_text=resp.text,
                    status_code=resp.status_code,
                )
                raise ModelCallError(f'gemini vision: HTTP {resp.status_code}: {resp.text[:200]}')
            data = resp.json()
            candidates = data.get('candidates', [])
            if not candidates:
                raise ModelCallError('gemini vision: no candidates in response')
            text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            trace_id = record_model_call(
                purpose=trace_purpose, provider='gemini', model=model,
                endpoint=url, request_text=prompt, status='success',
                started_at=started_at,
                response_text=text,
                meta={'usage': data.get('usageMetadata', {})},
            )
            return {
                'content': text,
                'meta': {'usage': data.get('usageMetadata', {}), 'model': model},
                'trace_id': trace_id.get('id', '') if isinstance(trace_id, dict) else '',
            }
        except requests.RequestException as e:
            record_model_call(
                purpose=trace_purpose, provider='gemini', model=model,
                endpoint=url, request_text=prompt, status='failed',
                started_at=started_at, error_text=str(e),
            )
            raise ModelCallError(f'gemini vision: {e}')

    def _call_gemini_text(
        self,
        cfg: Dict[str, Any] = None,
        prompt: str = '',
        purpose: str = 'text_reply',
    ) -> Dict[str, Any]:
        import requests
        api_key = str(cfg.get('api_key', '') or '').strip()
        model = str(cfg.get('model', 'gemini-2.0-flash') or 'gemini-2.0-flash').strip()
        # 超时默认值收敛(原110)：与主回复链路口径一致，避免单次调用拖太久。
        timeout = int(cfg.get('timeout_seconds', 45) or 45)

        if not api_key:
            raise ModelCallError('gemini text: api_key is empty.')

        body = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': float(cfg.get('temperature', 0.2) or 0.2),
                'maxOutputTokens': int(cfg.get('max_tokens', 1200) or 1200),
            },
        }
        # 安全修复#2：api_key 改由 x-goog-api-key 请求头传递，不进 URL query。
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key,
        }
        url = (
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
        )

        started_at = time.perf_counter()
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                record_model_call(
                    purpose=purpose, provider='gemini', model=model,
                    endpoint=url, request_text=prompt, status='failed',
                    started_at=started_at, error_text=resp.text,
                    status_code=resp.status_code,
                )
                raise ModelCallError(f'gemini text: HTTP {resp.status_code}: {resp.text[:200]}')
            data = resp.json()
            candidates = data.get('candidates', [])
            if not candidates:
                raise ModelCallError('gemini text: no candidates in response')
            text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            trace_id = record_model_call(
                purpose=purpose, provider='gemini', model=model,
                endpoint=url, request_text=prompt, status='success',
                started_at=started_at, response_text=text,
                meta={'usage': data.get('usageMetadata', {})},
            )
            return {
                'content': text,
                'meta': {'usage': data.get('usageMetadata', {}), 'model': model},
                'trace_id': trace_id.get('id', '') if isinstance(trace_id, dict) else '',
            }
        except requests.RequestException as e:
            record_model_call(
                purpose=purpose, provider='gemini', model=model,
                endpoint=url, request_text=prompt, status='failed',
                started_at=started_at, error_text=str(e),
            )
            raise ModelCallError(f'gemini text: {e}')

    # ------------------------------------------------------------------
    # 图片编码
    # ------------------------------------------------------------------
    @staticmethod
    def _image_b64(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode('ascii')

    @staticmethod
    def _vision_image_label(index: int, image_labels: List[str] | None) -> str:
        if image_labels and index < len(image_labels) and str(image_labels[index]).strip():
            return str(image_labels[index]).strip()
        return f'[Image {index + 1}]'

    # ------------------------------------------------------------------
    # 重试逻辑
    # ------------------------------------------------------------------
    def _retry_with_response_format_if_needed(
        self,
        cfg: Dict[str, Any],
        result: Dict[str, Any],
        purpose: str,
        retry_fn,
    ) -> Dict[str, Any]:
        content = str(result.get('content', '') or '')
        if not content or not content.strip():
            return result
        parsed = extract_json_object(content)
        if parsed is not None:
            return result
        if not bool(cfg.get('use_response_format_json', False)):
            return result
        retry_cfg = dict(cfg)
        retry_cfg['use_response_format_json'] = False
        if self.logger:
            self.logger(f'{purpose} retry_without_response_format: json parse failed on first attempt')
        # 修复#5：ModelCallError 本身是 Exception 子类，原写法等价于 except Exception，
        # 冗余且误导。重试失败时保守返回首次结果。
        try:
            return retry_fn(retry_cfg)
        except Exception:
            return result

    @staticmethod
    def _should_retry_without_response_format(exc: Exception, cfg: Dict[str, Any]) -> bool:
        if not bool(cfg.get('use_response_format_json', False)):
            return False
        msg = str(exc).lower()
        trigger_phrases = [
            'response_format', 'json_object', 'json mode',
            'does not support', 'not supported', 'unsupported',
            'invalid_request_error', '400',
        ]
        return any(phrase in msg for phrase in trigger_phrases)