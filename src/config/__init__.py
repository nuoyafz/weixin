import os
from pathlib import Path
from typing import Optional

from .settings import Settings, load_settings

DATA_DIR = Path(os.environ.get("VLA_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def ensure_dirs():
    """确保数据目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Settings:
    """加载配置，对齐原版 load_config。"""
    return load_settings(config_path)


__all__ = ["Settings", "load_settings", "load_config", "DATA_DIR", "ensure_dirs"]