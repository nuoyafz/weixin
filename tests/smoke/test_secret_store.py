"""密钥外置（安全修复#2）冒烟测试。

覆盖点：
1. 空值不动（不得把已有配置清空）
2. 已是 ${VAR} 占位 → 原样返回
3. 明文 key → 写进 .env，返回占位符；config.yaml 方向永远拿不到明文
4. upsert 保留原文件其它行与注释，整行替换同名键
5. 明文探测 contains_secret 可用于自检/CI 红线
"""

import pytest

from src.config.secret_store import (
    DEFAULT_ENV_NAME,
    contains_secret,
    is_placeholder,
    to_env_placeholder,
    upsert_env_secret,
)

#: 测试用假密钥。刻意分段拼接 —— 若在源码里写成完整字符串，
#: 会被仓库红线自检 `git grep -E "sk-[A-Za-z0-9]{8,}"` 误报成真实泄漏。
_FAKE_KEY = "sk-" + "abcdef1234567890"


def test_empty_value_is_noop(tmp_path):
    """空/空白 → 返回空串（调用方据此跳过写入，避免清掉已有配置）。"""
    assert to_env_placeholder("", project_root=tmp_path) == ""
    assert to_env_placeholder("   ", project_root=tmp_path) == ""
    assert not (tmp_path / ".env").exists()


def test_placeholder_passthrough(tmp_path):
    """已是 ${VAR} → 原样返回，不写 .env。"""
    assert is_placeholder("${VISREPLY_API_KEY}")
    out = to_env_placeholder("${VISREPLY_API_KEY}", project_root=tmp_path)
    assert out == "${VISREPLY_API_KEY}"
    assert not (tmp_path / ".env").exists()


def test_plaintext_is_externalized(tmp_path, monkeypatch):
    """明文 key → 返回占位符，明文只落 .env，绝不返回明文。"""
    monkeypatch.delenv(DEFAULT_ENV_NAME, raising=False)
    secret = _FAKE_KEY
    out = to_env_placeholder(secret, project_root=tmp_path)

    assert out == "${" + DEFAULT_ENV_NAME + "}"
    assert secret not in out
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert f"{DEFAULT_ENV_NAME}={secret}" in env_text


def test_upsert_replaces_and_preserves(tmp_path):
    """已有同名键 → 整行替换；其它行与注释保留。"""
    p = tmp_path / ".env"
    p.write_text(
        "# 我的注释\nOTHER=1\n" + DEFAULT_ENV_NAME + "=sk-old\n", encoding="utf-8")
    assert upsert_env_secret("sk-new", project_root=tmp_path)
    text = p.read_text(encoding="utf-8")
    assert "# 我的注释" in text
    assert "OTHER=1" in text
    assert f"{DEFAULT_ENV_NAME}=sk-new" in text
    assert "sk-old" not in text


def test_upsert_appends_when_missing(tmp_path):
    """没有同名键 → 追加，不删除已有内容。"""
    p = tmp_path / ".env"
    p.write_text("KEEP=1\n", encoding="utf-8")
    assert upsert_env_secret("sk-fresh", project_root=tmp_path)
    text = p.read_text(encoding="utf-8")
    assert "KEEP=1" in text
    assert f"{DEFAULT_ENV_NAME}=sk-fresh" in text


@pytest.mark.parametrize("value,expected", [
    (_FAKE_KEY, True),
    ("${VISREPLY_API_KEY}", False),
    ("", False),
    ("hello world", False),
])
def test_contains_secret(value, expected):
    assert contains_secret(value) is expected


@pytest.mark.parametrize("rel", ["src/ui/webview_window.py", "src/ui/html_window.py"])
def test_ui_source_never_writes_plaintext_api_key(rel):
    """源码守卫：UI 补丁方法必须走 _api_key_for_config，不得直写明文。

    反向验证依赖：修复前这两个文件里存在 `fields["api_key"] = api_key`
    （前端回传的展开真值直接落 config.yaml）——本断言在修复前会失败。
    """
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / rel
    text = p.read_text(encoding="utf-8")
    assert "_api_key_for_config" in text, f"{rel} 未走密钥外置转换"
    assert 'fields["api_key"] = api_key' not in text, f"{rel} 仍在直写明文 api_key"
    assert "fields[\"api_key\"] = self._api_key_for_config(api_key)" in text
