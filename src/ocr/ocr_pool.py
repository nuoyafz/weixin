"""进程级 OCR 引擎单例池。

—— 为什么需要它（2026-09-10 体检结论）——
同一份 RapidOCR 模型此前在进程里被重复实例化最多 6 次：

  · src/clean_perception/reader.py      importlib("clean_perception._ocrengine")
  · src/rpa/red_dot_detector.py         importlib("rpa._ocrengine")
  · src/rpa/friend_request_acceptor.py  直接 from ..local_vision.ocr_engine
  · src/agent/observe_service.py        from ..ocr.engine（另一个引擎类）
  · 以及 observe_service 里 new 出来的多个 RedDotDetector / UnreadDetector
    （每个 detector 各自惰性建一份引擎）

因为 importlib 用了**不同的模块名**，连 OCREngine 类本身都被加载了两遍，
sys.modules 去重完全失效，无法共享实例。每次 ``RapidOCR(params=...)``
都要重建 det+rec+cls 三个 onnxruntime session（各约 0.5~2s、数百 MB），
且各实例互不共享预热结果 —— 造成：
  · 冷启动慢好几秒；
  · 内存成倍占用；
  · onnxruntime 线程池重复，OCR 推理时 CPU 相互抢占；
  · 主 OCR 已预热，红点徽章那份仍是冷的，第一次识别必卡。

—— 本模块做什么 ——
把「同一份引擎类」收敛为**进程级单例**，构造参数与原先逐字一致，行为零变化：

  get_vision_ocr()  -> local_vision/ocr_engine.py 的 OCREngine
                       （``run(rgb, use_det=..., use_cls=..., use_rec=...)`` 接口，
                        用于 clean_perception reader 与 red_dot_detector）
  get_text_ocr()    -> src/ocr/engine.py 的 OCREngine
                       （``get_text_lines`` / ``recognize`` 接口，
                        用于 observe_service 的 local_vision 回落路径）

线程安全：OCR 调用本身由单条感知循环线程串行执行（observe_service._loop_thread），
因此不做调用级加锁；仅对**初始化**加锁，避免首轮并发初始化时重复建 session。
初始化失败会带冷却期重试（默认 30s），既不会每次调用都白试，也不会因一次
瞬时失败（模型文件正在拷贝等）永久失去 OCR 能力。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from typing import Any, Optional

__all__ = [
    "get_vision_ocr",
    "get_text_ocr",
    "reset_ocr_pool",
    "init_lock",
    "VISION",
    "TEXT",
]

VISION = "vision"
TEXT = "text"

# 初始化失败后的重试冷却（秒）：避免"一次瞬时失败 = 永久无 OCR"。
_RETRY_AFTER_SECONDS = 30.0

_LOCK = threading.RLock()
_ENGINES: dict[str, Any] = {}
_FAILED_AT: dict[str, float] = {}

# 视觉引擎模块别名：固定名字 + 注册进 sys.modules，保证全进程只 exec 一次。
_VISION_MODULE_ALIAS = "visreply._shared_vision_ocr_engine"

# 顶层便捷锁（供调用方在需要时包住"初始化 + 首次调用"）
init_lock = _LOCK


def _load_module_once(alias: str, engine_relpath: tuple) -> Optional[Any]:
    """按文件路径加载模块，并以固定别名注册进 sys.modules（只 exec 一次）。

    用文件路径加载是为了绕开 ``local_vision/__init__.py`` 里那条会触发越级
    相对导入的脆弱链（与改动前 reader/red_dot 的做法一致）；用**固定别名**
    注册则是本模块的关键修正 —— 原先两个调用点各用各的别名，导致同一文件
    被 exec 两次、类对象不共享。
    """
    mod = sys.modules.get(alias)
    if mod is not None:
        return mod
    here = os.path.dirname(os.path.abspath(__file__))          # src/ocr
    engine_path = os.path.abspath(os.path.join(here, *engine_relpath))
    if not os.path.exists(engine_path):
        return None
    spec = importlib.util.spec_from_file_location(alias, engine_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        sys.modules.pop(alias, None)
        return None
    return mod


def _engine_class(kind: str) -> Optional[type]:
    """取得对应引擎类（不实例化）。"""
    if kind == VISION:
        # 相对路径以本文件所在目录(src/ocr)为基准，故先回到 src 再进 local_vision
        mod = _load_module_once(
            _VISION_MODULE_ALIAS, ("..", "local_vision", "ocr_engine.py"))
        return getattr(mod, "OCREngine", None) if mod is not None else None
    # TEXT：走正常包内导入（src/ocr/__init__.py 只 re-export engine，链路干净）
    try:
        from .engine import OCREngine
        return OCREngine
    except Exception:  # noqa: BLE001
        return None


def _build_engine(kind: str) -> Optional[Any]:
    """实例化并（按各类原有语义）初始化引擎；不可用返回 None。

    两类引擎的初始化语义刻意保持与改动前一致：
      · TEXT （src/ocr/engine.py）：原 observe_service 显式调 ``initialize()``
        并检查返回值，失败即视为不可用 → 这里同样以返回值为准。
      · VISION（local_vision/ocr_engine.py）：原 reader / red_dot 只在
        ``is_ready()`` 为假时调 ``initialize()``，且**不因返回假而丢弃引擎**
        （``run()`` 内部还会再试一次 initialize）→ 这里保持一致。
    """
    cls = _engine_class(kind)
    if cls is None:
        return None
    engine = cls()
    inited = getattr(engine, "initialize", None)
    ready = getattr(engine, "is_ready", None)
    if kind == TEXT:
        if callable(inited) and not inited():
            return None
        return engine
    if callable(ready) and not ready() and callable(inited):
        inited()
    return engine


def _get_engine(kind: str) -> Optional[Any]:
    """取单例引擎；失败返回 None 并进入冷却，冷却到期后允许重试。"""
    eng = _ENGINES.get(kind)
    if eng is not None:
        return eng

    failed_at = _FAILED_AT.get(kind)
    if failed_at is not None and (time.time() - failed_at) < _RETRY_AFTER_SECONDS:
        return None

    with _LOCK:
        eng = _ENGINES.get(kind)
        if eng is not None:
            return eng
        failed_at = _FAILED_AT.get(kind)
        if failed_at is not None and (time.time() - failed_at) < _RETRY_AFTER_SECONDS:
            return None
        try:
            engine = _build_engine(kind)
            if engine is None:
                _FAILED_AT[kind] = time.time()
                return None
            _ENGINES[kind] = engine
            _FAILED_AT.pop(kind, None)
            return engine
        except Exception:  # noqa: BLE001
            _FAILED_AT[kind] = time.time()
            return None


def get_vision_ocr() -> Optional[Any]:
    """共享的视觉 OCR 引擎（local_vision/ocr_engine.py OCREngine）。

    供 clean_perception.reader 与 rpa.red_dot_detector 复用同一实例，
    避免"主 OCR 一套、红点徽章一套"两份模型与两份 onnxruntime 线程池。
    """
    return _get_engine(VISION)


def get_text_ocr() -> Optional[Any]:
    """共享的文本 OCR 引擎（src/ocr/engine.py OCREngine）。

    供 observe_service._ensure_local_ocr 的回落路径使用。
    """
    return _get_engine(TEXT)


def reset_ocr_pool() -> None:
    """清空单例（仅测试/热重载使用；正常运行时不要调用）。"""
    with _LOCK:
        _ENGINES.clear()
        _FAILED_AT.clear()
