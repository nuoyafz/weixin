"""模型配置管理：读取 config.yaml 的 ai 段，加载各任务类型的模型配置。

支持的任务类型：
    - text   通用文本补全（不含图片消息）
    - vision 视觉补全（消息可含 image_url）
    - chat   对话补全（带历史上下文的文本补全）

config.yaml 的 ai 段示例：
    ai:
      text:
        provider: openai_compatible
        base_url: "https://api.deepseek.com/v1"
        api_key: "sk-xxx"
        model: "deepseek-chat"
        temperature: 0.2
        max_tokens: 1200
        backup:
          provider: openai_compatible
          base_url: ""
          api_key: ""
          model: ""
          temperature: 0.2
          max_tokens: 1200
      vision:
        provider: openai_compatible
        base_url: ""
        api_key: ""
        model: "gpt-4o-mini"
        temperature: 0.1
        max_tokens: 1800
        backup: {...}
      chat:
        ...

未配置 ai 段时，会回退读取旧的顶层 text_model / vision_model 段，保证兼容。
"""
import yaml
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .provider_registry import apply_provider_choice


@dataclass
class ModelConfig:
    """单个模型的连接配置。backup 为自动降级时使用的备用模型。"""
    provider: str = "openai_compatible"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    api_key_required: bool = False
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_seconds: int = 110
    backup: Optional["ModelConfig"] = None

    def is_configured(self) -> bool:
        """是否至少配置了 base_url 与 model，可发起调用。"""
        return bool(self.base_url and self.model)


class ModelSettings:
    """按任务类型管理模型配置，并提供降级链解析。"""

    TASKS = ("text", "vision", "chat")
    _DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._DEFAULT_CONFIG_PATH
        self._configs: Dict[str, ModelConfig] = {}
        self.load()

    def load(self) -> None:
        raw: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        ai = raw.get("ai")
        if isinstance(ai, dict):
            for task in self.TASKS:
                self._configs[task] = self._parse_config(ai.get(task))
        else:
            self._load_legacy(raw)

    def _load_legacy(self, raw: Dict[str, Any]) -> None:
        """兼容旧的顶层 text_model / vision_model 段。"""
        text_cfg = self._parse_config(raw.get("text_model"))
        vision_cfg = self._parse_config(raw.get("vision_model"))
        self._configs["text"] = text_cfg
        self._configs["chat"] = text_cfg
        self._configs["vision"] = vision_cfg

    def _parse_config(self, data: Any) -> ModelConfig:
        if not isinstance(data, dict):
            return ModelConfig()
        cfg = ModelConfig(
            provider=str(data.get("provider", "openai_compatible")),
            model=str(data.get("model", "") or ""),
            api_key=str(data.get("api_key", "") or ""),
            base_url=str(data.get("base_url", "") or ""),
            api_key_required=bool(data.get("api_key_required", False)),
            temperature=float(data.get("temperature", 0.2)),
            max_tokens=int(data.get("max_tokens", 1200)),
            timeout_seconds=int(data.get("timeout_seconds", 110)),
        )
        backup = data.get("backup")
        if isinstance(backup, dict):
            cfg.backup = self._parse_config(backup)
        return cfg

    def get(self, task: str) -> ModelConfig:
        """返回指定任务的主模型配置。task 未配置时返回空配置（不抛错）。"""
        return self._configs.get(task, ModelConfig())

    def get_chain(self, task: str) -> List[ModelConfig]:
        """返回指定任务的降级链（主模型 -> backup -> backup...）。

        chat 任务未单独配置时，回退到 text 任务的配置。
        """
        cfg = self._configs.get(task)
        if cfg is None and task == "chat":
            cfg = self._configs.get("text")

        chain: List[ModelConfig] = []
        seen_base = set()
        cursor = cfg
        while cursor is not None:
            key = (cursor.base_url, cursor.model)
            if not cursor.is_configured() or key in seen_base:
                break
            seen_base.add(key)
            chain.append(cursor)
            cursor = cursor.backup
        return chain

    def tasks(self) -> List[str]:
        return list(self.TASKS)

    def reload(self) -> None:
        self.load()


def apply_model_settings_payload(config=None, data=None, *,
                                 empty_key_keeps_existing: bool = True) -> Dict[str, Any]:
    """Apply UI model form fields to a config copy.

    `empty_key_keeps_existing=True` 匹配保存设置行为（空字段保留原值）；
    `False` 用于"检查模型"场景，使未保存表单按输入精确测试。

    兼容原项目 app.ai.model_settings.apply_model_settings_payload。
    """
    if not isinstance(config, dict):
        config = {}
    cfg = deepcopy(config)
    data = data if isinstance(data, dict) else {}

    model_service = cfg.get("model_service")
    if not isinstance(model_service, dict):
        model_service = {}
        cfg["model_service"] = model_service
    text_model = cfg.setdefault("text_model", {})
    vision_model = cfg.setdefault("vision_model", {})

    requested_provider = str(
        data.get("model_provider") or data.get("provider") or ""
    ).strip().lower()
    if not requested_provider:
        requested_provider = "openai_compatible"

    provider_settings = apply_provider_choice(
        requested_provider,
        str(text_model.get("base_url", "") or ""),
    )

    def _pick(form_keys, saved, default=""):
        for k in form_keys:
            v = str(data.get(k) or "").strip()
            if v:
                return v
        if not empty_key_keeps_existing:
            return default
        return str(saved or default)

    base_url = _pick(
        ("custom_text_base_url", "text_base_url"),
        text_model.get("base_url"), provider_settings.get("base_url", ""),
    )
    api_key = _pick(
        ("text_api_key", "custom_text_api_key"),
        text_model.get("api_key"),
    )
    text_model_name = _pick(
        ("custom_text_model", "text_model"),
        text_model.get("model"),
    )
    vision_model_name = _pick(
        ("custom_vision_model", "multimodal_model", "vision_model"),
        vision_model.get("model"),
    )

    # 回写 text_model
    text_model["provider"] = provider_settings.get("provider", "openai_compatible")
    if base_url:
        text_model["base_url"] = base_url
    text_model["api_key_required"] = bool(
        provider_settings.get("api_key_required", True)
    )
    text_model["api_style"] = provider_settings.get("api_style", "openai_chat")
    if text_model_name:
        text_model["model"] = text_model_name
    text_model["use_response_format_json"] = False
    if api_key:
        text_model["api_key"] = api_key
    elif not empty_key_keeps_existing:
        text_model["api_key"] = ""

    # 回写 vision_model
    ai_vision_enabled = (
        bool(data.get("ai_vision_enabled", True))
        and bool(provider_settings.get("supports_vision", False))
    )
    if vision_model_name:
        vision_model["provider"] = text_model["provider"]
        vision_model["model"] = vision_model_name
        vision_model["base_url"] = base_url
        vision_model["api_key_required"] = text_model["api_key_required"]
        vision_model["api_style"] = text_model["api_style"]
        vision_model["use_response_format_json"] = True
        if api_key:
            vision_model["api_key"] = api_key
        elif not empty_key_keeps_existing:
            vision_model["api_key"] = ""
    else:
        vision_model["provider"] = "local"
        vision_model["base_url"] = ""
        vision_model["api_key"] = ""
        vision_model["model"] = ""
        vision_model["api_key_required"] = False
        vision_model["api_style"] = "local_ocr"
        vision_model["use_response_format_json"] = False

    # model_service.custom 同步
    model_service["mode"] = "custom"
    custom_settings = model_service.get("custom")
    if not isinstance(custom_settings, dict):
        custom_settings = {}
        model_service["custom"] = custom_settings
    if base_url:
        custom_settings["base_url"] = base_url
    if text_model_name:
        custom_settings["text_model"] = text_model_name
    if vision_model_name:
        custom_settings["vision_model"] = vision_model_name
    custom_settings["ai_vision_enabled"] = ai_vision_enabled
    if api_key:
        custom_settings["api_key"] = api_key
    elif not empty_key_keeps_existing:
        custom_settings["api_key"] = ""

    return cfg