"""VisReply 匿名使用统计（隐蔽、零侵入、不影响主流程）。

设计要点：
- 仅上报匿名事件：启动(startup) / 自动回复成功(reply, sent_ok=1)。
- 客户端【不】采集、也【不】上传用户 IP；IP 由服务器在接收端从连接层获取。
- device_id 为本地随机 UUID，仅用于去重「活跃设备数」，不涉任何个人信息。
- 全链路 fire-and-forget(daemon 线程) + try/except 吞错，零日志 / 零弹窗 / 零阻塞。
- 限频：reply 每秒最多 1 次；单进程每日上限 50000，防异常刷爆服务器。
"""
import os
import sys
import json
import time
import uuid
import threading
from pathlib import Path

__all__ = ["report_startup", "report_cycle", "get_device_id"]

_START_TS = time.time()
_last_cycle_ts = 0.0
_daily_count = 0
_DAILY_CAP = 50000
_device_id = None
_device_lock = threading.Lock()


def _resolve_collect_url() -> str:
    """从现有 OTA 域名自动推导上报地址，复用 app.update_server，无需新增配置。"""
    try:
        from src.updater import _resolve_server
        base = _resolve_server().rstrip("/")
    except Exception:
        base = "https://a.fangzhoui.cn/visreply/update"
    if base.endswith("/update"):
        base = base[: -len("/update")]
    return base.rstrip("/") + "/telemetry/collect"


def _local_version() -> str:
    try:
        from src.updater import get_local_version
        return get_local_version()
    except Exception:
        return "unknown"


def _data_dir() -> Path:
    try:
        from src.config import DATA_DIR
        return Path(DATA_DIR)
    except Exception:
        base = Path(sys.executable).resolve().parent
        d = base / "data"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            return Path(os.devnull).parent
        return d


def get_device_id() -> str:
    """返回稳定的本地匿名设备标识（随机 UUID，持久化到 data/.device_id）。"""
    global _device_id
    if _device_id:
        return _device_id
    with _device_lock:
        if _device_id:
            return _device_id
        try:
            p = _data_dir() / ".device_id"
            if p.exists() and str(p) != os.devnull:
                _device_id = p.read_text(encoding="utf-8").strip() or None
            if not _device_id:
                _device_id = uuid.uuid4().hex
                if str(p) != os.devnull:
                    p.write_text(_device_id, encoding="utf-8")
        except Exception:
            _device_id = uuid.uuid4().hex
    return _device_id


def _post(payload: dict) -> None:
    try:
        import urllib.request
        url = _resolve_collect_url()
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": "VisReply-Telemetry/1.0",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            r.read(1)
    except Exception:
        pass


def report(event: str, **extra) -> None:
    try:
        global _daily_count
        if _DAILY_CAP and _daily_count >= _DAILY_CAP:
            return
        payload = {
            "event": event,
            "version": _local_version(),
            "ts": int(time.time()),
            "uptime_min": int((time.time() - _START_TS) / 60),
            "device_id": get_device_id(),
        }
        payload.update({k: v for k, v in extra.items() if k != "event"})
        _daily_count += 1
        threading.Thread(target=_post, args=(payload,), daemon=True).start()
    except Exception:
        pass


def report_startup() -> None:
    """静默上报一次启动事件（GUI / 屏外两种模式入口各调一次）。"""
    report("startup")


def report_cycle(sent_ok: bool = False) -> None:
    """静默上报一次自动回复轮次（sent_ok=True 表示本轮成功发送了回复）。"""
    global _last_cycle_ts
    try:
        now = time.time()
        if now - _last_cycle_ts < 1.0:
            return
        _last_cycle_ts = now
        report("reply", sent_ok=bool(sent_ok))
    except Exception:
        pass
