"""AI 模型层扩展包：模型配置、文本补全、视觉补全与路由降级。

复用 reply.engine 的 ReplyResult 作为统一返回结构。
"""
from .model_settings import ModelSettings, ModelConfig
from .text_model_client import TextModelClient
from .vision_model_client import VisionModelClient
from .model_router import ModelRouter

__all__ = [
    "ModelSettings",
    "ModelConfig",
    "TextModelClient",
    "VisionModelClient",
    "ModelRouter",
]