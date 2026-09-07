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
                result = self.openai_client.chat(
                    messages,
                    model=cfg.get('model'),
                    base_url=cfg.get('base_url'),
                    api_key=cfg.get('api_key'),
                    use_response_format_json=bool(cfg.get('use_response_format_json', False)),
                    temperature=float(cfg.get('temperature', 0.2) or 0.2),
                    max_tokens=int(cfg.get('max_tokens', 1200) or 1200),
                    force_json=True,
                    purpose=trace_purpose,
                )
                return self._retry_with_response_format_if_needed(
                    cfg, result, trace_purpose,
                    lambda retry_cfg: self.openai_client.chat(
                        messages,
                        model=retry_cfg.get('model'),
                        base_url=retry_cfg.get('base_url'),
                        api_key=retry_cfg.get('api_key'),
                        use_response_format_json=bool(retry_cfg.get('use_response_format_json', False)),
                        temperature=float(retry_cfg.get('temperature', 0.2) or 0.2),
                        max_tokens=int(retry_cfg.get('max_tokens', 1200) or 1200),
                        force_json=True, purpose=trace_purpose,
                    ),
                )
            except ModelCallError as exc:
                if not self._should_retry_without_response_format(exc, cfg):
                    raise
                retry_cfg = dict(cfg)
                retry_cfg['use_response_format_json'] = False
                if self.logger:
                    self.logger(f'text_reply retry_without_response_format error={exc}')
                return self.openai_client.chat(
                    messages,
                    model=retry_cfg.get('model'),
                    base_url=retry_cfg.get('base_url'),
                    api_key=retry_cfg.get('api_key'),
                    use_response_format_json=False,
                    temperature=float(retry_cfg.get('temperature', 0.2) or 0.2),
                    max_tokens=int(retry_cfg.get('max_tokens', 1200) or 1200),
                    force_json=True, purpose=trace_purpose,
                )
        if provider == 'gemini':
            return self._call_gemini_text(cfg, system_prompt + '\n\n' + user_prompt, purpose=trace_purpose)
        if provider == 'local':
            raise ModelCallError(
                'text: provider=local is reserved and does not call a remote text model.'
            )
        raise ModelCallError(f"text: unsupported provider '{provider}'.")

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

        try:
            result = self.openai_client.chat(
                [{'role': 'user', 'content': content}],
                model=call_cfg.get('model'),
                base_url=call_cfg.get('base_url'),
                api_key=call_cfg.get('api_key'),
                use_response_format_json=bool(call_cfg.get('use_response_format_json', False)),
                temperature=float(call_cfg.get('temperature', 0.1) or 0.1),
                max_tokens=int(call_cfg.get('max_tokens', 1800) or 1800),
                force_json=True,
                purpose=trace_purpose,
            )
            result = self._retry_with_response_format_if_needed(
                call_cfg, result, trace_purpose,
                lambda retry_cfg: self.openai_client.chat(
                    [{'role': 'user', 'content': content}],
                    model=retry_cfg.get('model'),
                    base_url=retry_cfg.get('base_url'),
                    api_key=retry_cfg.get('api_key'),
                    use_response_format_json=bool(retry_cfg.get('use_response_format_json', False)),
                    temperature=float(retry_cfg.get('temperature', 0.1) or 0.1),
                    max_tokens=int(retry_cfg.get('max_tokens', 1800) or 1800),
                    force_json=True, purpose=trace_purpose,
                ),
            )
        except ModelCallError as exc:
            if not self._should_retry_without_response_format(exc, call_cfg):
                raise
            retry_cfg = dict(call_cfg)
            retry_cfg['use_response_format_json'] = False
            if self.logger:
                self.logger(f'vision_screen retry_without_response_format error={exc}')
            result = self.openai_client.chat(
                [{'role': 'user', 'content': content}],
                model=retry_cfg.get('model'),
                base_url=retry_cfg.get('base_url'),
                api_key=retry_cfg.get('api_key'),
                use_response_format_json=False,
                temperature=float(retry_cfg.get('temperature', 0.1) or 0.1),
                max_tokens=int(retry_cfg.get('max_tokens', 1800) or 1800),
                force_json=True, purpose=trace_purpose,
            )

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
        except Exception as exc:
            if self.logger:
                self.logger(f'vision_debug_save_failed error={exc}')

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
        timeout = int(cfg.get('timeout_seconds', 110) or 110)
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
        headers = {'Content-Type': 'application/json'}
        url = (
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            f'?key={api_key}'
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
        timeout = int(cfg.get('timeout_seconds', 110) or 110)

        if not api_key:
            raise ModelCallError('gemini text: api_key is empty.')

        body = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': float(cfg.get('temperature', 0.2) or 0.2),
                'maxOutputTokens': int(cfg.get('max_tokens', 1200) or 1200),
            },
        }
        headers = {'Content-Type': 'application/json'}
        url = (
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            f'?key={api_key}'
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
        try:
            return retry_fn(retry_cfg)
        except (ModelCallError, Exception):
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