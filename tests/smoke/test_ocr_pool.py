"""OCR 引擎单例池的守卫测试（对应 2026-09-10 提速体检）。

背景：同一份 RapidOCR 模型此前在进程里被重复实例化最多 6 处 ——
  clean_perception/reader、red_dot_detector(徽章)、red_dot_detector(昵称)、
  friend_request_acceptor、observe_service(回落)、voice_to_text、
  wechat_ocr_parser(每次调用重建)。
实测每多一份约 +32MB RSS、+0.16s 初始化，且各自独立 onnxruntime 线程池，
OCR 推理时互相抢核。本文件守住：
  1. 语义：同一类引擎多次取用必须是**同一个对象**；两类引擎不互相串；
  2. 缓存：连续取用只构建一次；失败进入冷却、冷却后允许重试；
  3. 源码守卫：各调用点不得回退为自建引擎（防止后续改动把优化又拆掉）。
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.ocr.ocr_pool as pool  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_pool():
    """每个用例前后都清空单例，避免用例间相互污染。"""
    pool.reset_ocr_pool()
    yield
    pool.reset_ocr_pool()


# --------------------------- 1. 语义：同一对象 ----------------------------


def test_vision_engine_is_singleton():
    """同一类引擎多次取用必须是同一个对象（这是本次优化的核心契约）。"""
    a = pool.get_vision_ocr()
    b = pool.get_vision_ocr()
    c = pool.get_vision_ocr()
    assert a is b is c


def test_text_engine_is_singleton():
    a = pool.get_text_ocr()
    b = pool.get_text_ocr()
    assert a is b


def test_two_kinds_are_distinct():
    """视觉引擎与文本引擎是两个不同的引擎类，不能混为一谈。"""
    v = pool.get_vision_ocr()
    t = pool.get_text_ocr()
    assert v is not t


# --------------------------- 2. 缓存与失败冷却 ----------------------------


def test_build_called_once(monkeypatch):
    """连续取用只构建一次（第 2、3 次必须命中缓存）。"""
    calls = []
    real = pool._build_engine

    def counting(kind):
        calls.append(kind)
        return real(kind)

    monkeypatch.setattr(pool, "_build_engine", counting)
    pool.get_vision_ocr()
    pool.get_vision_ocr()
    pool.get_vision_ocr()
    assert calls == [pool.VISION]


def test_failure_enters_cooldown(monkeypatch):
    """构建失败后进入冷却：冷却期内不再反复重试（避免每轮白试一次）。"""
    monkeypatch.setattr(pool, "_build_engine", lambda kind: None)
    assert pool.get_vision_ocr() is None

    calls = []
    monkeypatch.setattr(pool, "_build_engine",
                        lambda kind: calls.append(kind) or object())
    assert pool.get_vision_ocr() is None     # 冷却中
    assert calls == []                        # 没有再次构建


def test_failure_recovers_after_reset(monkeypatch):
    """工具/reset 后必须能重新构建（不能因一次瞬时失败永久失去 OCR）。"""
    monkeypatch.setattr(pool, "_build_engine", lambda kind: None)
    assert pool.get_vision_ocr() is None

    pool.reset_ocr_pool()
    sentinel = object()
    calls = []
    monkeypatch.setattr(pool, "_build_engine",
                        lambda kind: calls.append(kind) or sentinel)
    assert pool.get_vision_ocr() is sentinel
    assert calls == [pool.VISION]


def test_reset_is_idempotent():
    pool.reset_ocr_pool()
    pool.reset_ocr_pool()
    a = pool.get_vision_ocr()
    pool.reset_ocr_pool()
    b = pool.get_vision_ocr()
    assert a is not b  # reset 后是新对象（且不抛异常）


# --------------------------- 3. 源码守卫 ----------------------------


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel,getter", [
    ("src/clean_perception/reader.py", "get_vision_ocr"),
    ("src/rpa/red_dot_detector.py", "get_vision_ocr"),
    ("src/rpa/friend_request_acceptor.py", "get_vision_ocr"),
    ("src/rpa/voice_to_text.py", "get_vision_ocr"),
    ("src/agent/observe_service.py", "get_text_ocr"),
    ("src/local_vision/wechat_ocr_parser.py", "get_text_ocr"),
])
def test_call_site_uses_pool(rel, getter):
    """每个 OCR 调用点都必须经单例池取引擎。"""
    src = _read(rel)
    assert f"from ..ocr.ocr_pool import {getter}" in src, (
        f"{rel} 未走 ocr_pool.{getter}()，OCR 引擎会被重复加载")


def _engine_call_lines(rel: str) -> list:
    """用 AST 找 `OCREngine(...)` 调用行号。

    用 AST 而不是正则/grep：注释与 docstring 里会解释性提到 ``OCREngine()``，
    正则会把说明文字也当成违规。
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name == "OCREngine":
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("rel", [
    "src/clean_perception/reader.py",
    "src/rpa/red_dot_detector.py",
    "src/rpa/friend_request_acceptor.py",
    "src/rpa/voice_to_text.py",
    "src/local_vision/wechat_ocr_parser.py",
    "src/agent/observe_service.py",
])
def test_no_self_built_engine(rel):
    """这些文件不得再出现自建 OCREngine(...) —— 每处一份就是多一份模型。"""
    lines = _engine_call_lines(rel)
    assert lines == [], (
        f"{rel} 第 {lines} 行仍在自建 OCREngine(...)，会多占一份 RapidOCR 模型")


def test_pool_uses_fixed_alias_for_vision_module():
    """视觉引擎模块必须用**固定别名**注册，否则同一文件仍会被 exec 两次。"""
    src = _read("src/ocr/ocr_pool.py")
    assert "_VISION_MODULE_ALIAS" in src
    assert "sys.modules[alias] = mod" in src
    # 只允许一处按路径加载（复用同一实现），不得每个调用点各写一份
    assert src.count("spec_from_file_location") == 1


# --------------------------- 4. 生产配置守卫 ----------------------------


def test_roi_debug_disabled_in_repo_config():
    """生产配置不得开 roi 调试图：每轮同步写 3 张 PNG（~1.3MB）拖慢并伤盘。

    config.yaml 未纳入版本控制（含密钥占位符约定），CI 上不存在 → skip。
    """
    cfg = PROJECT_ROOT / "config.yaml"
    if not cfg.exists():
        pytest.skip("config.yaml 不在仓库中（已 gitignore）")
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"(?m)^\s*save_debug:\s*(\S+)", text)
    assert m, "config.yaml 缺少 ocr_roi.save_debug 配置"
    assert m.group(1).lower() in ("false", "no", "off"), (
        "ocr_roi.save_debug 必须为 false（每轮写 3 张调试图是纯浪费的同步 I/O）")
