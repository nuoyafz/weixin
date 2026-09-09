"""模型检测与回复速度检查。

独立于 UI / ReplyEngine，仅依赖标准库 urllib，提供：
- ModelChecker.detect()      检测模型是否可调用（连通性 / 鉴权 / 模型存在）。
- ModelChecker.speed_check() 连续发送若干轮短请求，统计平均 / 最小 / 最大延迟。

既可在后端代码里直接调用，也可由 tools/check_model_speed.py 命令行运行。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class ModelChecker:
    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 timeout_seconds: int = 20):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout_seconds = timeout_seconds

    # ---- 内部工具 ----
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _chat(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """单次调用，返回 {ok, latency_ms, content, error, error_type, status}。"""
        if not self.base_url or not self.model:
            return {"ok": False, "latency_ms": 0, "content": "",
                    "error": "base_url 或 model 未配置", "error_type": "config",
                    "status": 0}
        endpoint = self.base_url + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        req = urllib.request.Request(
            endpoint, data=json.dumps(body).encode("utf-8"),
            headers=self._headers(), method="POST")
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            latency = int((time.time() - start) * 1000)
            try:
                j = json.loads(raw)
            except Exception:
                return {"ok": True, "latency_ms": latency, "content": "",
                        "error": "", "error_type": "", "status": 200, "raw": raw}
            if "choices" in j and j["choices"]:
                content = j["choices"][0].get("message", {}).get("content", "")
                return {"ok": True, "latency_ms": latency, "content": content,
                        "error": "", "error_type": "", "status": 200, "raw": raw}
            err = j.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            return {"ok": False, "latency_ms": latency, "content": "",
                    "error": msg or str(j)[:200], "error_type": "api_error",
                    "status": 200, "raw": raw}
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return {"ok": False, "latency_ms": int((time.time() - start) * 1000),
                    "content": "", "error": f"HTTP {e.code}: {err_body[:300]}",
                    "error_type": "http", "status": e.code, "raw": err_body}
        except urllib.error.URLError as e:
            return {"ok": False, "latency_ms": int((time.time() - start) * 1000),
                    "content": "", "error": f"网络错误: {e.reason}",
                    "error_type": "network", "status": 0, "raw": ""}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "latency_ms": int((time.time() - start) * 1000),
                    "content": "", "error": f"检测失败: {e}",
                    "error_type": "unknown", "status": 0, "raw": ""}

    def _list_models(self) -> List[str]:
        try:
            url = self.base_url + "/models"
            req = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                j = json.loads(resp.read().decode("utf-8", errors="replace"))
            return [m.get("id", "") for m in j.get("data", [])]
        except Exception:
            return []

    # ---- 对外 API ----
    def detect(self) -> Dict[str, Any]:
        """检测模型可调用性，返回结构化结果。

        {ok, message, latency_ms, model, error_type, available_models?}
        """
        if not self.base_url or not self.model:
            return {"ok": False, "message": "请填写 API 地址和模型名称",
                    "latency_ms": 0, "model": self.model,
                    "error_type": "config"}
        r = self._chat("ping", max_tokens=5)
        if r["ok"]:
            return {"ok": True, "message": "连接成功，模型可调用",
                    "latency_ms": r["latency_ms"], "model": self.model,
                    "error_type": ""}
        msg = r["error"]
        # 404 model_not_found 时给出清晰提示；若端点支持 /models 列表则附带可选模型
        if r["error_type"] == "http" and r["status"] == 404:
            low = (r["error"] or "").lower()
            if "model" in low or "not" in low or "exist" in low:
                models = self._list_models()
                hint = ""
                if models:
                    top = ", ".join(models[:6])
                    hint = f"，该端点可用模型：{top}"
                msg = f"模型名不存在{hint}，请从端点复制准确模型名后重试"
        return {"ok": False, "message": msg, "latency_ms": r["latency_ms"],
                "model": self.model, "error_type": r["error_type"]}

    def speed_check(self, rounds: int = 3, prompt: str = "你好",
                    max_tokens: int = 32) -> Dict[str, Any]:
        """连续发送 rounds 轮短请求，统计延迟。

        返回 {ok, rounds, latencies, avg_ms, min_ms, max_ms,
              success_rounds, message}
        """
        if rounds < 1:
            rounds = 1
        latencies: List[int] = []
        success = 0
        last_error = ""
        for _ in range(rounds):
            r = self._chat(prompt, max_tokens=max_tokens)
            if r["ok"]:
                success += 1
                latencies.append(r["latency_ms"])
            else:
                last_error = r["error"]
        if not latencies:
            return {"ok": False, "rounds": rounds, "latencies": [],
                    "avg_ms": 0, "min_ms": 0, "max_ms": 0,
                    "success_rounds": 0, "message": f"全部失败: {last_error}"}
        avg = sum(latencies) / len(latencies)
        return {
            "ok": True,
            "rounds": rounds,
            "latencies": latencies,
            "avg_ms": int(round(avg)),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "success_rounds": success,
            "message": (f"成功 {success}/{rounds} 轮，平均 "
                        f"{int(round(avg))}ms，最快 {min(latencies)}ms，"
                        f"最慢 {max(latencies)}ms"),
        }


def check_from_config(config_path: str = "config.yaml",
                      rounds: int = 3) -> Optional[Dict[str, Any]]:
    """从 config.yaml 读取 text_model 配置并跑一次完整检测 + 速度检查。

    返回 {model, base_url, detect, speed}；配置缺失返回 None。
    """
    try:
        import yaml  # PyYAML，项目已依赖
    except Exception:
        yaml = None
    if yaml is None:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # 修复#1 配套：api_key 可能写成 ${VAR} 占位（密钥已外置到 .env/环境变量），
        # 必须展开成真实值再用来请求，否则连通性检查会拿占位串去鉴权必然失败。
        try:
            from ..config.settings import expand_env_config
            cfg = expand_env_config(cfg)
        except Exception:
            pass
    except Exception:
        return None
    tm = (cfg.get("text_model") or {}) if isinstance(cfg, dict) else {}
    base_url = (tm.get("base_url") or "").strip()
    api_key = (tm.get("api_key") or "").strip()
    model = (tm.get("model") or "").strip()
    if not base_url or not model:
        return None
    checker = ModelChecker(base_url, api_key, model)
    return {
        "model": model,
        "base_url": base_url,
        "detect": checker.detect(),
        "speed": checker.speed_check(rounds=rounds),
    }
