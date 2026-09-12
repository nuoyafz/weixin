#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VisReply 使用统计服务端：接收上报 + 漂亮仪表盘。零第三方依赖，兼容 Python 2.7 / 3.x。
运行: python telemetry_server.py --port 8000 --db telemetry.db
部署: 宝塔「Python 项目」命令行启动，或用反向代理把 visreply/telemetry 指到本服务。
"""
from __future__ import print_function

import os
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime, timedelta
try:
    # Python 3
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
except ImportError:  # Python 2
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn

# Python 2/3 兼容：json.loads 在 py2 需要 unicode 字符串
def _parse_json(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw or "{}")


def _to_bytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8")


DB_PATH = "telemetry.db"
PORT = 8000


def init_db(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, ip TEXT, "
        "event TEXT, version TEXT, device_id TEXT, sent_ok INTEGER DEFAULT 0)")
    conn.commit()


def get_client_ip(handler):
    fwd = handler.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return handler.client_address[0]


def insert_event(conn, ip, payload):
    conn.execute(
        "INSERT INTO events(ts,ip,event,version,device_id,sent_ok) VALUES(?,?,?,?,?,?)",
        (int(payload.get("ts", 0)), str(payload.get("ip", ip))[:64],
         str(payload.get("event", ""))[:32], str(payload.get("version", ""))[:32],
         str(payload.get("device_id", ""))[:64], 1 if payload.get("sent_ok") else 0))
    conn.commit()


def aggregate(conn):
    cur = conn.cursor()
    out = {}
    cur.execute("SELECT COUNT(*) FROM events WHERE event='startup'")
    out["startups"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM events WHERE event='reply' AND sent_ok=1")
    out["replies"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT device_id) FROM events")
    out["devices"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT ip) FROM events")
    out["ips"] = cur.fetchone()[0]
    cur.execute(
        "SELECT date(ts,'unixepoch') d, SUM(event='startup'), SUM(event='reply' AND sent_ok=1) "
        "FROM events WHERE ts > ? GROUP BY d ORDER BY d",
        (int(datetime.now().timestamp()) - 30 * 86400,))
    out["daily"] = [{"d": r[0], "startups": r[1] or 0, "replies": r[2] or 0}
                    for r in cur.fetchall()]
    cur.execute("SELECT ip, COUNT(*) c, MIN(ts) f, MAX(ts) l, "
                "SUM(event='reply' AND sent_ok=1) r FROM events "
                "GROUP BY ip ORDER BY c DESC LIMIT 20")
    out["by_ip"] = [{"ip": r[0], "count": r[1], "first": r[2], "last": r[3],
                     "replies": r[4] or 0} for r in cur.fetchall()]
    cur.execute("SELECT version, COUNT(*) FROM events GROUP BY version")
    out["versions"] = [{"v": r[0], "c": r[1]} for r in cur.fetchall()]
    return out


DASHBOARD = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisReply 使用统计</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,"PingFang SC",sans-serif;
background:#0f1221;color:#e6e8f0;padding:28px}
h1{font-size:20px;font-weight:600;margin-bottom:18px;letter-spacing:.5px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}
.card{background:linear-gradient(145deg,#1b1f3a,#161a30);border:1px solid #2a3050;
border-radius:14px;padding:20px}
.card .n{font-size:30px;font-weight:700;background:linear-gradient(90deg,#5eead4,#818cf8);
-webkit-background-clip:text;background-clip:text;color:transparent}
.card .l{font-size:13px;color:#9aa3c0;margin-top:6px}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
.panel{background:#161a30;border:1px solid #2a3050;border-radius:14px;padding:16px}
.panel h2{font-size:14px;color:#c3c9e6;margin-bottom:10px;font-weight:600}
#trend{height:300px}#ver{height:300px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 6px;border-bottom:1px solid #232845;color:#c8cee8}
th{color:#8b93b8;font-weight:500}
.ip{color:#7dd3fc}
.refresh{position:fixed;right:24px;bottom:22px;background:#6366f1;color:#fff;border:none;
border-radius:10px;padding:10px 16px;font-size:13px;cursor:pointer;box-shadow:0 6px 18px rgba(99,102,241,.4)}
</style></head><body>
<h1>VisReply · 使用情况仪表盘</h1>
<div class="cards">
  <div class="card"><div class="n" id="c_start">-</div><div class="l">总启动次数</div></div>
  <div class="card"><div class="n" id="c_rep">-</div><div class="l">自动回复成功次数</div></div>
  <div class="card"><div class="n" id="c_dev">-</div><div class="l">活跃设备数</div></div>
  <div class="card"><div class="n" id="c_ip">-</div><div class="l">独立 IP 数</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>近 30 天趋势（启动 / 回复成功）</h2><div id="trend"></div></div>
  <div class="panel"><h2>版本分布</h2><div id="ver"></div></div>
</div>
<div class="panel" style="margin-top:16px"><h2>使用人 IP 排行（Top 20）</h2>
  <table><thead><tr><th>IP</th><th>总次数</th><th>回复成功</th><th>首次</th><th>末次</th></tr></thead>
  <tbody id="iptb"></tbody></table></div>
<button class="refresh" onclick="load()">刷新</button>
<script>
function fmt(t){return t?new Date(t*1000).toLocaleString('zh-CN'):''}
function load(){
 fetch('api/data').then(r=>r.json()).then(d=>{
  c_start.textContent=d.startups;c_rep.textContent=d.replies;
  c_dev.textContent=d.devices;c_ip.textContent=d.ips;
  const tr=echarts.init(document.getElementById('trend'));
  tr.setOption({tooltip:{trigger:'axis'},legend:{textStyle:{color:'#c8cee8'},data:['启动','回复成功']},
   xAxis:{type:'category',data:d.daily.map(x=>x.d.slice(5)),axisLabel:{color:'#8b93b8'}},
   yAxis:{type:'value',axisLabel:{color:'#8b93b8'},splitLine:{lineStyle:{color:'#232845'}}},
   series:[{name:'启动',type:'line',smooth:true,data:d.daily.map(x=>x.startups),areaStyle:{opacity:.15},itemStyle:{color:'#5eead4'}},
           {name:'回复成功',type:'bar',data:d.daily.map(x=>x.replies),itemStyle:{color:'#818cf8'}}]});
  const ve=echarts.init(document.getElementById('ver'));
  ve.setOption({tooltip:{},series:[{type:'pie',radius:['40%','70%'],
   data:d.versions.map(v=>({name:v.v||'unknown',value:v.c})),
   label:{color:'#c8cee8'},itemStyle:{borderColor:'#161a30',borderWidth:2}}]});
  iptb.innerHTML=d.by_ip.map(r=>'<tr><td class="ip">'+r.ip+'</td><td>'+r.count+'</td>'
   +'<td>'+r.replies+'</td><td>'+fmt(r.first)+'</td><td>'+fmt(r.last)+'</td></tr>').join('');
 });
}
load();setInterval(load,30000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        body = _to_bytes(body)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if "collect" not in self.path.split("?")[0]:
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b"{}"
            payload = _parse_json(raw)
            conn = sqlite3.connect(DB_PATH)
            insert_event(conn, get_client_ip(self), payload)
            conn.close()
            self._send(200, "application/json", '{"ok":true}')
        except Exception as e:
            self._send(500, "application/json", '{"ok":false,"err":"%s"}' % str(e))

    def do_GET(self):
        _s = self.path.split("?")[0].rstrip("/")
        # 兼容反代前缀：/、/visreply/telemetry、/telemetry、/dashboard 均返回看板
        if _s == "" or _s.endswith("telemetry") or _s.endswith("dashboard") or _s.endswith("index.html"):
            self._send(200, "text/html; charset=utf-8", DASHBOARD)
        elif _s.endswith("api/data"):
            try:
                conn = sqlite3.connect(DB_PATH)
                init_db(conn)
                data = aggregate(conn)
                conn.close()
                self._send(200, "application/json", json.dumps(data))
            except Exception as e:
                self._send(500, "application/json", '{"ok":false,"err":"%s"}' % str(e))
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global DB_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--db", default=DB_PATH)
    a = ap.parse_args()
    DB_PATH = a.db
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    conn.close()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print("VisReply telemetry running on Python %s @ %s" % (sys.version.split()[0], sys.executable))
    print("VisReply telemetry on http://0.0.0.0:%d  (POST /collect  GET /  GET /api/data)" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
