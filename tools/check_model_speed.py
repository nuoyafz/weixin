#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型检测 + 回复速度检查（命令行）。

用法:
    python tools/check_model_speed.py            # 用 config.yaml 的 text_model，跑 3 轮
    python tools/check_model_speed.py 5          # 跑 5 轮
    python tools/check_model_speed.py --model xxx --base-url https://... --api-key sk-xxx

依赖: 仅标准库 + PyYAML（项目已依赖，用于读取 config.yaml）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse_args(argv):
    """极简参数解析，支持位置 rounds 与 --model/--base-url/--api-key。"""
    rounds = 3
    model = base_url = api_key = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            model = argv[i + 1]; i += 2; continue
        if a == "--base-url":
            base_url = argv[i + 1]; i += 2; continue
        if a == "--api-key":
            api_key = argv[i + 1]; i += 2; continue
        if a.lstrip("-").isdigit():
            rounds = int(a.lstrip("-"))
        i += 1
    return rounds, model, base_url, api_key


def main():
    rounds, model, base_url, api_key = _parse_args(sys.argv[1:])

    from src.reply.model_check import ModelChecker

    # 优先命令行参数，否则回退到 config.yaml 的 text_model
    if not (base_url and model):
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                tm = cfg.get("text_model") or {}
                base_url = base_url or (tm.get("base_url") or "").strip()
                api_key = api_key or (tm.get("api_key") or "").strip()
                model = model or (tm.get("model") or "").strip()
            except Exception as e:
                print(f"[warn] 读取 config.yaml 失败: {e}")
        else:
            print("[warn] 未找到 config.yaml，请使用 --model/--base-url/--api-key 指定。")

    if not base_url or not model:
        print("错误：缺少 base_url / model。请用命令行参数或配置 config.yaml。")
        sys.exit(2)

    print(f"模型: {model}")
    print(f"地址: {base_url}")
    print(f"速度检查轮数: {rounds}")
    print("-" * 48)

    checker = ModelChecker(base_url, api_key, model)
    det = checker.detect()
    mark = "✓" if det["ok"] else "✗"
    print(f"[{mark}] 检测: {det['message']}  (首响 {det['latency_ms']}ms)")

    if not det["ok"]:
        sys.exit(1)

    print(f"[…] 正在跑 {rounds} 轮短请求…")
    sp = checker.speed_check(rounds=rounds)
    if not sp["ok"]:
        print(f"[✗] 速度检查失败: {sp['message']}")
        sys.exit(1)

    print(f"[✓] 速度: {sp['message']}")
    print(f"    逐轮延迟: {sp['latencies']} ms")
    avg = sp["avg_ms"]
    if avg < 800:
        grade = "快（适合实时客服）"
    elif avg < 2500:
        grade = "中等"
    else:
        grade = "偏慢（建议换更快模型或启用 FAQ 分流）"
    print(f"    评级: {grade}")


if __name__ == "__main__":
    main()
