import sys
import os
from pathlib import Path

# 在 import 业务模块（含 OCR 引擎）之前根据 config.yaml 的 ocr.dml 设置
# VISREPLY_OCR_DML，使 DirectML(GPU) 加速可被配置项一键开启。
try:
    import yaml
    _cfg_p = Path(__file__).resolve().parent / "config.yaml"
    if _cfg_p.exists():
        with open(_cfg_p, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        if _cfg.get("ocr", {}).get("dml"):
            os.environ["VISREPLY_OCR_DML"] = "1"
except Exception:
    pass

from src.ui.webview_window import launch


def resource_path(*parts: str) -> Path:
    if getattr(sys, 'frozen', False):
        base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent)).resolve()
    else:
        base = Path(__file__).resolve().parent
    return base.joinpath(*parts)


def main():
    from src.config import Settings, load_settings

    config_path = resource_path('config.yaml')
    settings = load_settings(str(config_path)) if config_path.exists() else Settings()

    launch(settings)


if __name__ == '__main__':
    main()
