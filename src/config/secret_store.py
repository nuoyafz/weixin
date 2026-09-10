"""密钥外置存储（安全修复#2）。

背景：settings.py 已支持「config.yaml 只写 `${VAR}` 占位、真值放 .env」，
但 UI 保存设置时走的是 `save_settings_from_ui` → `_patch_*_model_in_yaml`，
它把前端传回的**展开后明文 key** 直接写进了 config.yaml —— 占位符机制
形同虚设，明文密钥随配置落盘（并可能被打进安装包）。

本模块提供唯一入口 `to_env_placeholder()`：
    ""            -> ""（不动）
    "${VISREPLY_API_KEY}" -> 原样返回（已是占位）
    "sk-xxxx"     -> 写入 .env（整行替换或追加）+ 进程环境变量，
                     返回 "${VISREPLY_API_KEY}"

只依赖标准库，便于 CI 直接测试（不 import 重量级 UI 模块）。
"""

import os
import re
from pathlib import Path
from typing import Optional

#: config.yaml 与 .env 默认使用的环境变量名
DEFAULT_ENV_NAME = "VISREPLY_API_KEY"

#: 整值占位引用，如 ${VISREPLY_API_KEY}
_PLACEHOLDER = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")

#: 项目根目录（my_agent/）：src/config/secret_store.py -> parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_placeholder(value: object) -> bool:
    """判断值是否是 `${VAR}` 形式的占位引用（含内嵌也算，保守用整值判定）。"""
    return bool(_PLACEHOLDER.match(str(value or "").strip()))


def contains_secret(value: object) -> bool:
    """粗判字符串里是否含 API key 明文（用于自检/埋点）。"""
    return bool(re.search(r"sk-[A-Za-z0-9_\-]{8,}", str(value or "")))


def env_file_path(project_root: Optional[Path] = None) -> Path:
    """项目根的 .env 路径。"""
    root = Path(project_root) if project_root else _PROJECT_ROOT
    return root / ".env"


def upsert_env_secret(value: str, env_name: str = DEFAULT_ENV_NAME,
                      project_root: Optional[Path] = None) -> bool:
    """把密钥写入 .env：存在 `NAME=` 行则整行替换，否则追加到末尾。

    保留文件里其它行与注释；不删除已有键。写入失败返回 False。
    """
    value = str(value or "").strip()
    if not value:
        return False
    p = env_file_path(project_root)
    try:
        lines = []
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
        prefix = f"{env_name}="
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(prefix):
                lines[i] = f"{prefix}{value}"
                replaced = True
                break
        if not replaced:
            if lines and lines[-1].strip():
                lines.append("")
            if not any(l.strip().startswith("#") for l in lines):
                lines.insert(0, "# 本地密钥文件 —— 不要提交到仓库，也不要打包进安装包")
            lines.append(f"{prefix}{value}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def to_env_placeholder(value: str, env_name: str = DEFAULT_ENV_NAME,
                       project_root: Optional[Path] = None,
                       set_environ: bool = True) -> str:
    """把「可能明文」的 key 转成 config.yaml 里该写的占位符。

    返回规则：
      · 空值 / 纯空白        -> ""（调用方应跳过写入，不要清掉已有配置）
      · 已是 ${VAR} 占位      -> 原样返回
      · 其它（视为明文密钥）  -> 写 .env + 进程环境变量，返回 "${env_name}"

    注意：`set_environ=True` 会让本次改动立即在本进程生效（否则重启后才生效，
    且系统环境变量同名时会把用户刚填的新 key 顶掉）。本地单用户程序取即时性。
    """
    value = str(value or "").strip()
    if not value:
        return ""
    if is_placeholder(value):
        return value
    upsert_env_secret(value, env_name=env_name, project_root=project_root)
    if set_environ:
        os.environ[env_name] = value
    return "${" + env_name + "}"
