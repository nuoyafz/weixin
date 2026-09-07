"""FriendlyErrors - user-friendly error messages for UI display.

Aligned with original app.ui.friendly_errors - maps technical errors
to human-readable Chinese messages for the WeChat automation UI.
"""
from __future__ import annotations

from typing import Any


ERROR_MAP = {
    "WeChat not found": "未找到微信窗口，请确保微信已启动",
    "window not found": "窗口未找到，请检查微信是否正在运行",
    "capture failed": "截图失败，请检查显示设置",
    "ocr failed": "OCR识别失败，请检查屏幕分辨率",
    "model timeout": "AI模型响应超时，请检查网络连接",
    "api key invalid": "API密钥无效，请检查配置",
    "rate limit": "API请求频率过高，请稍后重试",
    "network error": "网络连接失败，请检查网络",
    "clipboard error": "剪贴板操作失败",
    "permission denied": "权限不足，请以管理员身份运行",
    "send failed": "发送失败，请检查微信窗口状态",
    "detection failed": "未读检测失败，请检查微信窗口",
    "no unread": "未检测到未读消息",
    "already sent": "消息已发送过，跳过重复",
    "blacklisted": "联系人已加入黑名单",
    "skip contact": "联系人已跳过",
    "unknown": "发生未知错误，请查看日志",
}


def friendly_error(error: str) -> str:
    """Convert a technical error to a user-friendly message."""
    if not error:
        return ERROR_MAP.get("unknown", "未知错误")

    error_lower = error.lower()
    for key, message in ERROR_MAP.items():
        if key.lower() in error_lower:
            return message

    return f"错误: {error}"


def friendly_success(action: str) -> str:
    """Generate a friendly success message."""
    SUCCESS_MAP = {
        "send": "消息发送成功",
        "detect": "未读检测完成",
        "capture": "截图成功",
        "ocr": "OCR识别完成",
        "reply": "回复发送成功",
        "accept": "好友请求已通过",
    }
    return SUCCESS_MAP.get(action, f"{action} 完成")


def friendly_status(status: str) -> str:
    """Generate a friendly status message."""
    STATUS_MAP = {
        "idle": "空闲",
        "running": "运行中",
        "detecting": "检测未读中...",
        "sending": "发送消息中...",
        "paused": "已暂停",
        "stopped": "已停止",
        "error": "错误",
    }
    return STATUS_MAP.get(status, status)