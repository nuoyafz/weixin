"""ProviderRegistry - registry of AI model providers.

Aligned with original app.ai.provider_registry - manages multiple
AI providers (OpenAI, DeepSeek, local, etc.) with unified interface.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

OPENAI_COMPATIBLE_PROVIDER_IDS = frozenset({
    "openai", "builtin", "zhipu", "custom", "openrouter",
    "siliconflow", "deepseek", "ollama", "qwen", "doubao", "kimi",
})

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "builtin": {
        "label": "内置模型服务",
        "provider": "builtin",
        "base_url": "",
        "default_model": "builtin-text",
        "vision_model": "builtin-vision",
        "supports_vision": True,
        "api_key_required": False,
        "api_style": "openai_chat",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容",
        "provider": "openai_compatible",
        "base_url": "",
        "default_model": "",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "openai": {
        "label": "OpenAI",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5-mini",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "kimi": {
        "label": "Kimi",
        "provider": "kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.5",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "doubao": {
        "label": "豆包/火山方舟",
        "provider": "doubao",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "deepseek": {
        "label": "DeepSeek",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "qwen": {
        "label": "通义千问/百炼",
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3.6-flash",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "zhipu": {
        "label": "智谱 GLM",
        "provider": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4.7-flash",
        "vision_model": "glm-4.6v-flash",
        "supports_vision": True,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "siliconflow": {
        "label": "SiliconFlow",
        "provider": "siliconflow",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "openrouter": {
        "label": "OpenRouter",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "openai_chat",
    },
    "gemini": {
        "label": "Gemini",
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-3.5-flash",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": True,
        "api_style": "gemini_generate_content",
    },
    "ollama": {
        "label": "Ollama / 本地 OpenAI 兼容",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen3:8b",
        "vision_model": "",
        "supports_vision": False,
        "api_key_required": False,
        "api_style": "openai_chat",
    },
}


def provider_preset(choice: str) -> dict[str, Any]:
    """Return a deep copy of the provider preset for the given choice."""
    key = str(choice or "").strip().lower()
    if key not in PROVIDER_PRESETS:
        key = "custom"
    return deepcopy(PROVIDER_PRESETS[key])


def provider_choice_from_config(provider: str, base_url: str) -> str:
    """Detect provider choice from configured provider name and base URL."""
    provider_value = str(provider or "").strip().lower()
    url = str(base_url or "").strip().rstrip("/")

    if not provider_value and not url:
        return "builtin"

    if "/vla-license/model/" in url:
        return "builtin"

    if provider_value in ("gemini",) or "generativelanguage.googleapis.com" in url:
        return "gemini"
    if "api.openai.com" in url:
        return "openai"
    if "moonshot.cn" in url:
        return "kimi"
    if "volces.com" in url:
        return "doubao"
    if "deepseek.com" in url:
        return "deepseek"
    if "dashscope.aliyuncs.com" in url:
        return "qwen"
    if "bigmodel.cn" in url:
        return "zhipu"
    if "siliconflow.cn" in url:
        return "siliconflow"
    if "openrouter.ai" in url:
        return "openrouter"
    if "127.0.0.1:11434" in url or "localhost:11434" in url:
        return "ollama"

    if provider_value and provider_value in PROVIDER_PRESETS:
        return provider_value

    return "custom"


def apply_provider_choice(
    choice: str, current_base_url: str
) -> dict[str, Any]:
    """Apply provider choice and return the resolved provider settings dict."""
    base_url = str(choice or "").strip().lower()
    selected = base_url

    if base_url == "custom":
        detected = provider_choice_from_config("", current_base_url)
        preset = provider_preset(
            detected if detected not in ("custom", "builtin") else "custom"
        )
    elif base_url == "builtin":
        preset = provider_preset("builtin")
    else:
        preset = provider_preset(base_url)
        selected = preset.get("provider", base_url)

    return {
        "choice": selected,
        "label": preset.get("label", ""),
        "provider": preset.get("provider", "openai_compatible"),
        "base_url": preset.get("base_url", ""),
        "default_model": preset.get("default_model", ""),
        "vision_model": preset.get("vision_model", ""),
        "supports_vision": bool(preset.get("supports_vision", False)),
        "api_key_required": bool(preset.get("api_key_required", True)),
        "api_style": preset.get("api_style", "openai_chat"),
    }


class ProviderRegistry:
    """Registry of AI model providers."""

    def __init__(self, config=None):
        self._config = config or {}
        self._providers = {}

    def register(self, name: str, config: dict[str, Any]) -> None:
        """Register a provider."""
        self._providers[name] = config

    def unregister(self, name: str) -> None:
        """Unregister a provider."""
        self._providers.pop(name, None)

    def get(self, name: str) -> dict[str, Any] | None:
        """Get provider config."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())

    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self._config.get("default_provider", "openai")

    def get_provider_config(self, name: str = None) -> dict[str, Any]:
        """Get config for a specific provider."""
        if name is None:
            name = self.get_default_provider()
        return self._providers.get(name, {
            "api_key": self._config.get("api_key", ""),
            "base_url": self._config.get("base_url", "https://api.openai.com/v1"),
            "model": self._config.get("model", "gpt-4o-mini"),
        })

    def load_from_config(self) -> None:
        """Load providers from config."""
        providers = self._config.get("providers", {})
        for name, cfg in providers.items():
            self.register(name, cfg)