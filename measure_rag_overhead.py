import json, time, urllib.request, re, yaml, sys
sys.path.insert(0, ".")
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
tm = cfg["text_model"]
base, key, model = tm["base_url"], tm["api_key"], tm["model"]

def chat(system_p, user_p, max_tokens=200):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
        return d["choices"][0]["message"]["content"]

def llm_call(system_p, user_p):
    # 真实回调：发给同一 endpoint，解析 JSON 给 RAG 用
    try:
        content = chat(system_p, user_p, max_tokens=300)
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
        return {}
    except Exception:
        return {}

from src.rag.knowledge_base import KnowledgeBase
kb = KnowledgeBase(
    root_path="data/knowledge",
    enable_rewrite=True, enable_rerank=True,
    rewrite_trigger=0.18, rerank_trigger=0.6,
    llm_call=llm_call,
)
print("loading knowledge base ...")
t0 = time.time(); kb.load_documents(); print(f"  index built in {time.time()-t0:.2f}s")
# warmup（不计时，避免首次检索建缓存干扰）
kb.query("warmup question")

questions = [
    ("命中型", "你们营业时间是几点到几点"),
    ("口语型", "咋个整你们这玩意儿咋用啊"),
]
for tag, q in questions:
    t0 = time.time(); items = kb.query(q); t_query = time.time() - t0
    best = max([s for _, s in items], default=0)
    t0 = time.time(); chat("你是一个客服助手", q, max_tokens=200); t_reply = time.time() - t0
    print(f"\n[{tag}] {q}")
    print(f"  RAG检索+改写/精排 t_query={t_query:.2f}s  命中={len(items)}条 best={best:.3f}")
    print(f"  主回复          t_reply={t_reply:.2f}s")
    print(f"  → 读RAG再回复={t_query+t_reply:.2f}s   直接回复={t_reply:.2f}s   差值={t_query:.2f}s")
