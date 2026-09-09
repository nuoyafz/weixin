"""本地运行日志：全量落盘到 logs/，每次运行一个文件，最多保留最近 N 个。

需求（方哥）：
- 新增一个本地 log，和应用控制台 / UI 面板日志同源
  （统一 hook 两个汇聚点：_term 与 FileStore.append_log）。
- 只保存最近 5 次运行的日志文件，超出自动删除最旧的。
- 安全：写盘前对 sk- 类密钥脱敏，绝不把 API Key 落进文件（红线）。

目录策略：
- 优先放到「启动 exe / run.py 所在目录」的 logs/ 下（打包后就是
  WeChatAIAssistant/logs/，双击 exe 即可在同目录找到；源码跑就是 my_agent/logs/）。
- 若该目录不可写再回退到项目根 logs/。
"""
from __future__ import annotations

import atexit
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_KEEP = 5
_SK_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")
# ④ 提速/稳定性：所有线程共享同一个文件句柄 + 互斥锁，避免每句 open/close 的
# 磁盘 syscall 抖动，也杜绝多线程并发写导致的行交错/截断。
_lock = threading.Lock()


def _sanitize(msg: str) -> str:
    """把可能出现的 API Key / 密钥打码，避免落盘泄露。"""
    if not msg:
        return msg
    return _SK_RE.sub("sk-***", msg)


class LocalLogger:
    """进程级单例本地日志器。"""

    _path = None          # 当前运行的日志文件路径
    _started = False      # 防止同一次运行重复建文件
    _fh = None            # ④ 常驻文件句柄（单句柄共享，行缓冲）

    # ------------------------------------------------------------------
    @classmethod
    def start_run(cls, keep: int = _KEEP):
        """开始一次运行：建日志文件 + rollover 只保留最近 keep 个。"""
        if cls._started:
            return cls._path
        cls._started = True
        try:
            base = Path(sys.argv[0]).resolve().parent
            logs_dir = base / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            probe = logs_dir / ".probe"
            try:
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except Exception:
                # 回退到项目根 logs（适用于 argv[0] 所在目录不可写的情况）
                logs_dir = Path(__file__).resolve().parents[2] / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cls._path = logs_dir / f"run_{ts}.log"
            try:
                # ④ 一次性打开常驻句柄（行缓冲），后续所有写入复用，零 open/close 抖动
                cls._fh = open(cls._path, "a", encoding="utf-8", buffering=1)
            except Exception:
                cls._fh = None
            cls._rollover(logs_dir, keep)
            cls._write("===== 运行开始 %s =====" % time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:  # 落盘失败绝不能影响主流程
            cls._path = None
            try:
                sys.stdout.write("[local_logger] start_run failed: %s\n" % e)
            except Exception:
                pass
        return cls._path

    @classmethod
    def _rollover(cls, logs_dir: Path, keep: int) -> None:
        """删除最旧的日志文件，使总数不超过 keep（新建文件前调用）。"""
        try:
            files = sorted(
                logs_dir.glob("run_*.log"),
                key=lambda p: p.stat().st_mtime,
            )
            # 当前 _path 尚未写入，所以现有文件保留 keep-1 个即可，
            # 新建后总文件数 = keep。
            while len(files) >= keep:
                oldest = files.pop(0)
                try:
                    oldest.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def _write(cls, line: str) -> None:
        if not cls._path:
            return
        try:
            with _lock:
                if cls._fh is not None:
                    cls._fh.write(line + "\n")
                    cls._fh.flush()
                else:
                    # 句柄未建立时的兜底（极端情况），仍加锁防止并发交错
                    with open(cls._path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
        except Exception:
            pass

    @classmethod
    def log(cls, level: str, msg: str) -> None:
        """追加一条日志（自动脱敏）。"""
        if cls._path is None:
            return
        safe = _sanitize(str(msg))
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cls._write("[%s] [%s] %s" % (ts, level, safe))

    @classmethod
    def get_current_log_path(cls):
        return cls._path

    @classmethod
    def close_run(cls) -> None:
        """进程退出前刷新并关闭日志句柄（④ 单句柄共享）。"""
        try:
            if cls._fh is not None:
                cls._fh.flush()
                cls._fh.close()
        except Exception:
            pass
        cls._fh = None


# 进程退出时确保日志落盘（避免最后几行丢失）
atexit.register(LocalLogger.close_run)
