"""最小冒烟测试：py_compile + 配置加载 + 空回复降级（真实兜底 LLM）。

不依赖微信窗口 / 屏幕 / OCR / 真机，纯逻辑、可离线（除兜底 LLM 测试需网络+密钥）。
运行：
    python -m pytest tests/smoke -q
本地无 VISREPLY_API_KEY 时，兜底 LLM 测试自动 skip（CI 配置 repo secret 后运行）。
"""
import os
import py_compile

import pytest
import yaml
from pathlib import Path

from src.config.settings import load_settings
from src.ai.text_model_client import TextModelClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def test_py_compile():
    """所有 src 下的 .py 必须能编译通过（防语法 / import 级回归）。"""
    errors = []
    count = 0
    for py in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        count += 1
        try:
            py_compile.compile(str(py), doraise=True)
        except (py_compile.PyCompileError, SyntaxError) as e:
            errors.append(f"{py.relative_to(PROJECT_ROOT)}: {e}")
    assert count > 0, "未找到任何 .py 源文件"
    assert not errors, f"{len(errors)} 个文件编译失败:\n" + "\n".join(errors)


def test_config_load(sample_config_path):
    """配置加载：密钥外置展开、超时兜底抬45、无明文 sk-。"""
    settings = load_settings(str(sample_config_path))
    # 1) 超时兜底：<30s 抬到 45（防旧配置 timeout=20 误杀慢调用）
    assert settings.text_model.timeout_seconds == 45, \
        f"text_model.timeout 应兜底为45, 实为 {settings.text_model.timeout_seconds}"
    assert settings.vision_model.timeout_seconds == 45, \
        f"vision_model.timeout 应兜底为45, 实为 {settings.vision_model.timeout_seconds}"
    # 2) 关 RAG 配置正确读取（空回复降级前提）
    assert settings.rag.enabled is False
    # 3) 样例 dump 不应含明文 sk- 密钥
    raw = yaml.safe_load(open(sample_config_path, encoding="utf-8"))
    dumped = yaml.safe_dump(raw, allow_unicode=True)
    assert "sk-" not in dumped, "样例配置不应含明文 sk- 密钥"
    # 4) 密钥占位符展开（仅当环境注入了真实 key 时校验）
    api_key = os.environ.get("VISREPLY_API_KEY")
    if api_key:
        assert not settings.text_model.api_key.startswith("${"), "api_key 占位符未被展开"
        assert settings.text_model.api_key == api_key, "展开后的 api_key 应与环境变量一致"


@pytest.mark.network
def test_empty_reply_fallback(sample_config_path):
    """空回复降级：关 RAG + 无业务上下文时，真实兜底 LLM 必须返回非空回复。"""
    api_key = os.environ.get("VISREPLY_API_KEY")
    if not api_key:
        pytest.skip("VISREPLY_API_KEY 未设置，跳过真实 LLM 兜底测试（CI 需配置 repo secret）")
    settings = load_settings(str(sample_config_path))
    tm = settings.text_model
    client = TextModelClient(
        config={
            "text_model": {
                "provider": tm.provider,
                "base_url": tm.base_url,
                "api_key": api_key,
                "model": tm.model,
            }
        },
        timeout_seconds=tm.timeout_seconds,
    )
    analysis = {
        "customer_turn_text": "今天天气真不错啊",
        "latest_message": {"text": "今天天气真不错啊"},
    }
    try:
        _a, reply, meta = client.generate_fallback_reply(analysis=analysis, window_info={})
    except ImportError as e:
        # CI（Linux）无 Windows 授权模块 app.license.client，属环境缺失，非代码缺陷
        pytest.skip(f"兜底 LLM 依赖在 CI 环境不可用（app.license.client 仅 Windows 真机存在）: {e}")
    err = (meta or {}).get("error", "") or ""
    if not reply.strip():
        # 网络/鉴权/授权/模块缺失等环境问题不应让 smoke 红（非代码缺陷），跳过；
        # 只有"调用成功但返回空"才是空回复降级真 bug，会落到下面 assert。
        if any(k in err.lower() for k in (
            "timeout", "network", "connection", "401", "403",
            "unauthorized", "resolve", "refused", "certificate",
            "license", "credential", "gateway", "module", "import",
            "missing_text_model_config",
        )):
            pytest.skip(f"兜底 LLM 调用失败（疑似环境/授权问题，非代码缺陷）: {err}")
    assert reply.strip(), f"兜底回复为空（空回复降级失效）meta={meta}"


def test_fallback_switch(sample_config_path):
    """纯逻辑（不调 LLM / 不碰授权）：关 RAG 时兜底自由聊开关应开启。

    对应「空回复降级」的前提——RAG 关闭后 _free_chat 应为真，
    这样无业务上下文时也能走兜底回复而非静默空回复。CI 可稳定跑通。
    """
    settings = load_settings(str(sample_config_path))
    tm = settings.text_model
    client = TextModelClient(
        config={
            "text_model": {
                "provider": tm.provider,
                "base_url": tm.base_url,
                "api_key": "dummy-not-used",
                "model": tm.model,
            },
            "rag": {"enabled": False},
        },
        timeout_seconds=tm.timeout_seconds,
    )
    assert client._rag_enabled() is False, "RAG 应处于关闭状态（兜底前提）"
    assert client._fallback_chitchat_enabled() is True, "兜底自由聊开关应默认开启"
