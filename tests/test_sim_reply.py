"""模拟问答链路离线验证。

复用 my_agent/src/ui/html_window.py 中 run_sim_reply 的调用方式，
直接驱动 TextModelClient.refine_reply()，验证：
  1) 模型是否真的被调用（而不是走硬编码兜底）；
  2) 不同问题是否得到不同回复；
  3) 失败时是否拿到可读的真实原因（403 额度 / 404 模型名 / 未配置）。

用法：
    python tools/test_sim_reply.py
    python tools/test_sim_reply.py "你们价格多少" "能上门吗"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_analysis(question: str, history: list[dict] | None = None) -> dict:
    """与 html_window.HtmlMainWindow.run_sim_reply 内构造保持一致。"""
    rows = [h for h in (history or []) if isinstance(h, dict)]
    conv_ctx = []
    for row in rows[-10:]:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        conv_ctx.append({
            "role": "assistant" if row.get("role") == "assistant" else "customer",
            "content": text,
        })
    return {
        "current_contact": "模拟客户",
        "decision": "reply",
        "action": "reply",
        "intent": "brand_question",
        "latest_message": {"sender": "customer", "content": question},
        "customer_turn_text": question,
        "customer_turn_messages": [{"sender": "customer", "content": question}],
        "conversation_context": conv_ctx,
        "visible_conversation_text": question,
        "visible_conversation_messages": [{"sender": "customer", "content": question}],
        "confidence": 1.0,
    }


def main() -> int:
    import yaml
    from src.ai.text_model_client import TextModelClient

    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        print(f"[!] 找不到 {cfg_path}")
        return 2
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    tm_cfg = cfg.get("text_model") or {}
    print("=" * 66)
    print("模拟问答链路验证")
    print("=" * 66)
    print(f"base_url : {tm_cfg.get('base_url', '')}")
    print(f"model    : {tm_cfg.get('model', '')}")
    print(f"api_key  : {'已配置' if tm_cfg.get('api_key') else '（空）'}")

    tm = TextModelClient(
        base_url=tm_cfg.get("base_url", ""),
        api_key=tm_cfg.get("api_key", ""),
        model=tm_cfg.get("model", ""),
        config=cfg,
    )
    print(f"available: {tm.available()}")
    print()

    questions = sys.argv[1:] or ["你好", "你们价格是多少", "能上门服务吗"]
    window_info = {"title": "模拟问答", "visible_text": ""}

    replies: list[str] = []
    for q in questions:
        print("-" * 66)
        print(f"问题：{q}")
        analysis = build_analysis(q)
        try:
            refined, reply, meta = tm.refine_reply(analysis, window_info, "reply")
        except Exception as e:
            print(f"  异常：{e}")
            continue
        meta = meta or {}
        rag = (refined or {}).get("rag") or {}
        print(f"  模型被调用 : {bool(meta.get('called'))}")
        print(f"  RAG 命中   : {bool(rag.get('has_context'))}"
              f"（{len(rag.get('chunks') or [])} 段）")
        reply = str(reply or "").strip()
        if reply:
            print(f"  回复       : {reply}")
            replies.append(reply)
        else:
            err = str((refined or {}).get("text_model_error") or "").strip()
            print(f"  无回复     : {err or meta.get('error') or meta.get('reason') or '未知'}")

    print()
    print("=" * 66)
    if not replies:
        print("结论：模型未产出任何回复 —— 上面已打印真实原因，请按原因处理。")
        return 1
    uniq = set(replies)
    if len(uniq) == len(replies) and len(replies) > 1:
        print(f"结论：正常 —— {len(replies)} 个问题得到 {len(uniq)} 个不同回复。")
    else:
        print(f"结论：可疑 —— {len(replies)} 个问题只有 {len(uniq)} 种回复。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
