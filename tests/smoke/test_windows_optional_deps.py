"""守卫：pywin32 必须是**可选**依赖，非 Windows 环境照样能 import 本包。

回归背景（CI 红）：
    tests/smoke/test_sender_sendpath.py 里 `from src.rpa.wechat_sender import ...`
    → 触发 src/rpa/__init__.py 顶层导入 human_like_mouse
    → human_like_mouse 顶层 `import win32con` 在 Linux runner 上 ModuleNotFoundError
    → 整个模块无法收集，smoke job 直接红（exit 2）。

产品只在 Windows 跑，但 CI / 静态检查在 Linux —— 所以顶层导入必须先
降级为可选（try/except + 明确报错），而不是把整包拖死。

本测试用**子进程**屏蔽 win32* 模块，等价于 Linux runner 的环境，
因此即使在 Windows 本机开发也能提前发现同类问题。
"""

import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]

_BLOCKED = ("win32con", "win32gui", "win32process", "win32api", "win32clipboard")

_PROBE = """
import sys

_BLOCK = {blocked!r}


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in _BLOCK:
            raise ImportError('blocked (simulated non-Windows): ' + fullname)
        return None


sys.meta_path.insert(0, _Blocker())

import src.rpa                      # noqa: F401  触发包顶层导入链
import src.rpa.wechat_sender        # noqa: F401
import src.rpa.red_dot_detector     # noqa: F401
import src.rpa.unread_detector      # noqa: F401
print('IMPORT_OK')
""".format(blocked=_BLOCKED)


def test_package_imports_without_pywin32():
    """屏蔽 pywin32 后，核心模块仍必须可导入（模拟 Linux CI）。"""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=180)
    assert "IMPORT_OK" in proc.stdout, (
        "非 Windows 环境下导入失败：\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


def test_human_like_mouse_flags_missing_pywin32():
    """缺 pywin32 时 HumanLikeMouse 导入得到可用标志，实例化给出明确报错。"""
    code = _PROBE.replace(
        "print('IMPORT_OK')",
        "import src.rpa.human_like_mouse as _h\n"
        "assert _h.WIN32_AVAILABLE is False, 'WIN32_AVAILABLE 应为 False'\n"
        "try:\n"
        "    _h.HumanLikeMouse()\n"
        "except RuntimeError:\n"
        "    print('IMPORT_OK')\n"
        "else:\n"
        "    raise AssertionError('缺 pywin32 时应抛 RuntimeError')\n")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=180)
    assert "IMPORT_OK" in proc.stdout, (
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


def test_human_like_mouse_uses_src_imports():
    """源码守卫：human_like_mouse 不得再出现裸 `import win32*` 顶层导入。"""
    src = (_ROOT / "src" / "rpa" / "human_like_mouse.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped or line[:1].isspace():
            continue  # 只检查顶层（0 缩进）
        assert not stripped.startswith(("import win32", "from win32")), (
            f"human_like_mouse.py 顶层出现裸 win32 导入: {line!r}")
    assert "WIN32_AVAILABLE" in src
