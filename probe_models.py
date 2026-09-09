import json
import time
import urllib.request
import yaml

# 探测当前 maas endpoint 支持哪些 model 名（不打印 api_key）
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
tm = cfg["text_model"]
base = tm["base_url"]
key = tm["api_key"]

candidates = [
    "qwen-flash", "qwen-turbo", "qwen-plus", "qwen-max",
    "qwen3.8-27b", "qwen3-27b", "qwen-flash-latest",
]

for m in candidates:
    payload = json.dumps({
        "model": m,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read())
            content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"OK   {m:22s} latency={int((time.time()-t)*1000)}ms resp={content!r}")
    except Exception as e:
        print(f"FAIL {m:22s} {str(e)[:130]}")
