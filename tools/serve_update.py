#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地自测：把 update_dist/ 当作静态目录起 HTTP 服务。

生产请用 Nginx / 对象存储 / 宝塔站点托管，本脚本仅用于本机联调。
用法:
  python tools/serve_update.py --dir update_dist --port 8000
然后浏览器访问 http://127.0.0.1:8000/version.json 验证。
"""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="update_dist", help="要托管的目录（含 version.json）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    if not d.is_dir():
        raise SystemExit(f"目录不存在: {d}")

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *x, **k):
            super().__init__(*x, directory=str(d), **k)

        def log_message(self, *x):  # 静默日志
            pass

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"serving {d}  ->  http://{a.host}:{a.port}/")
    print(f"version.json: http://{a.host}:{a.port}/version.json")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
