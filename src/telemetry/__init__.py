"""匿名使用统计模块（隐蔽）。对外暴露 usage.report_* 系列。"""
from .usage import report_startup, report_cycle  # noqa: F401
