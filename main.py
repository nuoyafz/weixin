import sys
from pathlib import Path

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
