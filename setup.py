"""
环境初始化脚本
将原 VisionLeadAgent 的 _internal 目录链接到新项目
"""
import os
import sys
import shutil
from pathlib import Path


def setup_environment():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir
    parent_dir = project_dir.parent

    source_internal = parent_dir / "_internal"
    target_internal = project_dir / "_internal"

    if not source_internal.exists():
        print(f"[错误] 未找到原 VisionLeadAgent 的 _internal 目录")
        print(f"  路径: {source_internal}")
        print(f"  请确保本脚本与 VisionLeadAgent.exe 位于同一父目录下")
        return False

    if target_internal.exists():
        choice = input(f"[提示] _internal 目录已存在，是否重新链接？(y/n): ").strip().lower()
        if choice != "y":
            return True
        if target_internal.is_symlink() or target_internal.is_dir():
            if target_internal.is_symlink():
                target_internal.unlink()
            else:
                shutil.rmtree(target_internal)

    try:
        if os.name == "nt":
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                os.symlink(source_internal, target_internal, target_is_directory=True)
                print(f"[成功] 已创建符号链接: {target_internal} -> {source_internal}")
            else:
                print("[提示] 需要管理员权限创建符号链接，使用复制方式...")
                shutil.copytree(source_internal, target_internal)
                print(f"[成功] 已复制 _internal 到: {target_internal}")
        else:
            os.symlink(source_internal, target_internal, target_is_directory=True)
            print(f"[成功] 已创建符号链接: {target_internal} -> {source_internal}")

        return True

    except Exception as e:
        print(f"[错误] 链接失败: {e}")
        print("[替代方案] 请手动复制 _internal 目录到本项目")
        return False


def check_dependencies():
    print("\n[检查] 验证 Python 依赖...")
    missing = []

    try:
        import PySide6
        print("  ✓ PySide6")
    except ImportError:
        missing.append("PySide6")
        print("  ✗ PySide6")

    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("  ✗ numpy")

    try:
        import cv2
        print("  ✓ opencv-python")
    except ImportError:
        missing.append("opencv-python")
        print("  ✗ opencv-python")

    try:
        import yaml
        print("  ✓ PyYAML")
    except ImportError:
        missing.append("PyYAML")
        print("  ✗ PyYAML")

    try:
        import win32api
        print("  ✓ pywin32")
    except ImportError:
        missing.append("pywin32")
        print("  ✗ pywin32")

    try:
        import rapidocr
        print("  ✓ rapidocr (来自 _internal)")
    except ImportError:
        print("  ⚠ rapidocr (将在运行时从 _internal 加载)")

    if missing:
        print(f"\n[警告] 缺少 {len(missing)} 个依赖:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\n[成功] 所有依赖已就绪！")

    return len(missing) == 0


if __name__ == "__main__":
    print("=" * 50)
    print("  My WeChat Agent 环境初始化")
    print("=" * 50)
    print()

    success = setup_environment()
    if not success:
        print("\n[提示] 您可以手动将 _internal 目录复制到项目根目录")
        print(f"  源目录: {Path(__file__).resolve().parent.parent / '_internal'}")
        print(f"  目标目录: {Path(__file__).resolve().parent / '_internal'}")
    else:
        print()

    deps_ok = check_dependencies()

    if success and deps_ok:
        print("\n" + "=" * 50)
        print("  初始化完成！运行 start.bat 启动助手")
        print("=" * 50)
    else:
        print("\n[部分完成] 请解决上述问题后重新运行")
