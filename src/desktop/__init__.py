"""desktop：桌面 / 窗口系统相关能力（微信窗口查找、控制与运行环境检测）。

依赖该包前请确保已在 src 包环境下（相对导入依赖 src 为 Python 包）。
"""

from .window_finder import WindowFinder
from .wechat_window_manager import (
    WeChatWindowManager,
    SW_HIDE, SW_SHOWNORMAL, SW_SHOWMINIMIZED, SW_SHOWMAXIMIZED,
    SW_SHOWNOACTIVATE, SW_SHOW, SW_MINIMIZE, SW_RESTORE,
)
from .wechat_environment import (
    WeChatEnvironment,
    WeChatWindowState,
    VirtualScreen,
)

__all__ = [
    "WindowFinder",
    "WeChatWindowManager",
    "SW_HIDE", "SW_SHOWNORMAL", "SW_SHOWMINIMIZED", "SW_SHOWMAXIMIZED",
    "SW_SHOWNOACTIVATE", "SW_SHOW", "SW_MINIMIZE", "SW_RESTORE",
    "WeChatEnvironment",
    "WeChatWindowState",
    "VirtualScreen",
]