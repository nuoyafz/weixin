# UI 层包：pywebview 迁移后不再在包导入期拉起 html_window（它会 import PySide6）。
# 新入口：src/ui/webview_window.py 的 launch(settings) / create_app(settings)。
# html_window.py 保留作回退参考，但不作为默认导出。
