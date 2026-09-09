import json, time, urllib.request, re, yaml, sys
sys.path.insert(0, ".")
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
tm = cfg["text_model"]
base, key, model = tm["base_url"], tm["api_key"], tm["model"]

def chat(system_p, user_p, max_tokens=200):
    payload = json.dumps({"model": model, "messages": [
        {"role": "system", "content": system_p},
        {"role": "user", "content": user_p}], "max_tokens": max_tokens, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(base + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def llm_call(system_p, user_p):
    try:
        content = chat(system_p, user_p, max_tokens=300)
        try: return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

from src.rag.knowledge_base import KnowledgeBase
questions = [("命中型", "你们营业时间是几点到几点"), ("口语型", "咋个整你们这玩意儿咋用啊")]

print("=== 对照组A：启用 RAG 增强（rewrite+rerank）===")
kb_on = KnowledgeBase(root_path="data/knowledge", enable_rewrite=True, enable_rerank=True,
                      rewrite_trigger=0.18, rerank_trigger=0.6, llm_call=llm_call)
kb_on.load_documents(); kb_on.query("warmup")
for tag, q in questions:
    t0=time.time(); items=kb_on.query(q); t=time.time()-t0
    print(f"  [{tag}] t_query={t:.2f}s 命中={len(items)} best={max([s for _,s in items],default=0):.3f}")

print("=== 对照组B：禁用 RAG 增强（纯词法，不调 LLM）===")
kb_off = KnowledgeBase(root_path="data/knowledge", enable_rewrite=False, enable_rerank=False, llm_call=None)
kb_off.load_documents()
for tag, q in questions:
    t0=time.time(); items=kb_off.query(q); t=time.time()-t0
    print(f"  [{tag}] t_query={t*1000:.1f}ms 命中={len(items)} best={max([s for _,s in items],default=0):.3f}")
