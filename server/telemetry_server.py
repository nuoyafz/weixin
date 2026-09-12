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
<title>VisReply · 使用情况仪表盘</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<meta name="theme-color" content="#05060f">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:dark}
body{font-family:'Inter','PingFang SC',system-ui,sans-serif;color:#e8ecff;min-height:100vh;
  padding:32px clamp(16px,4vw,48px);position:relative;overflow-x:hidden;
  background:
    radial-gradient(1100px 600px at 82% -12%,rgba(129,140,248,.20),transparent 60%),
    radial-gradient(900px 520px at -8% 112%,rgba(94,234,212,.14),transparent 55%),
    #05060f;}
body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(120,140,255,.07) 1px,transparent 1px),
    linear-gradient(90deg,rgba(120,140,255,.07) 1px,transparent 1px);
  background-size:46px 46px;
  -webkit-mask-image:radial-gradient(circle at 50% 28%,#000 26%,transparent 82%);
  mask-image:radial-gradient(circle at 50% 28%,#000 26%,transparent 82%);}
.brand{display:flex;align-items:center;gap:15px;margin-bottom:8px}
.logo{width:44px;height:44px;border-radius:13px;flex:none;position:relative;
  background:linear-gradient(135deg,#5eead4,#818cf8);box-shadow:0 0 26px rgba(129,140,248,.55)}
.logo::after{content:"";position:absolute;inset:9px;border-radius:7px;
  background:radial-gradient(circle at 50% 50%,#fff 1.5px,transparent 2.5px);opacity:.85}
.brandTitle{font-family:'Orbitron','Inter',sans-serif;font-size:clamp(18px,2.4vw,26px);font-weight:700;
  letter-spacing:2px;background:linear-gradient(90deg,#e0e7ff,#a5b4fc,#5eead4);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.brand .sub{font-size:11px;color:#7e86bf;letter-spacing:4px;text-transform:uppercase;margin-top:3px}
.status{font-size:12px;color:#7e86bf;letter-spacing:1px;margin:0 0 24px;font-family:'JetBrains Mono',monospace}
.status b{color:#5eead4;font-weight:500}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:22px}
@media(max-width:920px){.cards{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){.cards{grid-template-columns:1fr}}
.card{position:relative;padding:22px 22px 20px;border-radius:18px;background:rgba(20,24,48,.55);
  border:1px solid rgba(120,140,255,.18);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  overflow:hidden;animation:rise .6s both}
.card::after{content:"";position:absolute;inset:0;border-radius:18px;padding:1px;pointer-events:none;
  background:linear-gradient(135deg,rgba(94,234,212,.55),rgba(129,140,248,.22),transparent 70%);
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude}
.card .n{font-family:'Orbitron','Inter',sans-serif;font-size:clamp(26px,3vw,40px);font-weight:700;
  font-variant-numeric:tabular-nums;color:#eafff8;text-shadow:0 0 18px rgba(94,234,212,.45);line-height:1.1}
.card .l{margin-top:9px;font-size:13px;color:#9aa3d4;letter-spacing:.5px}
.card .tag{position:absolute;top:14px;right:16px;font-size:10px;letter-spacing:2px;color:#6b73a8;
  font-family:'Orbitron',sans-serif}
.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:18px}
@media(max-width:920px){.grid{grid-template-columns:1fr}}
.panel{background:rgba(20,24,48,.55);border:1px solid rgba(120,140,255,.18);border-radius:18px;
  padding:18px 18px 10px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  position:relative;animation:rise .7s both}
.panel h2,.ipwrap h2{font-size:12px;color:#aab2e6;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;
  margin-bottom:10px;display:flex;align-items:center;gap:9px}
.panel h2::before{content:"";width:8px;height:8px;border-radius:50%;background:#5eead4;box-shadow:0 0 10px #5eead4}
#trend{height:320px}#ver{height:320px}
.ipwrap{margin-top:18px;background:rgba(20,24,48,.55);border:1px solid rgba(120,140,255,.18);
  border-radius:18px;padding:18px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);animation:rise .8s both}
.ipwrap h2::before{content:"";width:8px;height:8px;border-radius:50%;background:#818cf8;box-shadow:0 0 10px #818cf8}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:#8b93c4;font-weight:500;text-align:left;padding:10px 8px;border-bottom:1px solid rgba(120,140,255,.2);
  font-size:11px;letter-spacing:1px;text-transform:uppercase}
td{padding:11px 8px;border-bottom:1px solid rgba(120,140,255,.08);color:#d8defc}
tbody tr{transition:background .2s,box-shadow .2s}
tbody tr:hover{background:rgba(129,140,248,.1);box-shadow:inset 0 0 0 1px rgba(129,140,248,.25)}
.ip{font-family:'JetBrains Mono',monospace;color:#38bdf8;text-shadow:0 0 8px rgba(56,189,248,.5)}
.refresh{position:fixed;right:26px;bottom:24px;z-index:5;border:none;border-radius:14px;cursor:pointer;
  background:linear-gradient(135deg,#818cf8,#5eead4);color:#06121a;padding:12px 18px;font-weight:700;
  font-size:13px;letter-spacing:1px;box-shadow:0 8px 26px rgba(129,140,248,.5);
  transition:transform .2s ease,box-shadow .2s ease;animation:pulse 2.6s infinite}
.refresh:hover{transform:translateY(-2px);box-shadow:0 12px 34px rgba(94,234,212,.6)}
.refresh:focus-visible{outline:2px solid #fff;outline-offset:3px}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{box-shadow:0 8px 26px rgba(129,140,248,.4)}50%{box-shadow:0 8px 30px rgba(94,234,212,.7)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<h1 style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)">VisReply 使用情况仪表盘</h1>
<div class="brand">
  <div class="logo" aria-hidden="true"></div>
  <div><div class="brandTitle">VISREPLY</div><div class="sub">Telemetry · 使用情况</div></div>
</div>
<div class="status" id="status" aria-live="polite">连接中…</div>
<div class="cards">
  <div class="card"><span class="tag">START</span><div class="n" id="c_start">0</div><div class="l">总启动次数</div></div>
  <div class="card"><span class="tag">REPLY</span><div class="n" id="c_rep">0</div><div class="l">自动回复成功次数</div></div>
  <div class="card"><span class="tag">DEVICE</span><div class="n" id="c_dev">0</div><div class="l">活跃设备数</div></div>
  <div class="card"><span class="tag">IP</span><div class="n" id="c_ip">0</div><div class="l">独立 IP 数</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>近 30 天趋势 · 启动 / 回复成功</h2><div id="trend"></div></div>
  <div class="panel"><h2>版本分布</h2><div id="ver"></div></div>
</div>
<div class="ipwrap">
  <h2>使用人 IP 排行 · Top 20</h2>
  <table><thead><tr><th>IP</th><th>总次数</th><th>回复成功</th><th>首次</th><th>末次</th></tr></thead>
  <tbody id="iptb"></tbody></table>
</div>
<button class="refresh" onclick="load()">↻ 刷新</button>
<script>
const dtf=new Intl.DateTimeFormat('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
const tmf=new Intl.DateTimeFormat('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
function fmt(t){return t?dtf.format(new Date(t*1000)):''}
function animNum(el,to){
  const from=parseInt(el.dataset.v||'0',10);
  if(from===to){el.textContent=to;return;}
  const t0=performance.now(),dur=750;
  (function step(t){const p=Math.min(1,(t-t0)/dur);
    el.textContent=Math.round(from+(to-from)*(1-Math.pow(1-p,3)));
    if(p<1)requestAnimationFrame(step);else el.dataset.v=to;})(t0);
}
function load(){
 fetch('api/data').then(r=>r.json()).then(d=>{
  animNum(c_start,d.startups);animNum(c_rep,d.replies);
  animNum(c_dev,d.devices);animNum(c_ip,d.ips);
  const tr=echarts.init(document.getElementById('trend'));
  tr.setOption({grid:{left:44,right:18,top:42,bottom:28},
   tooltip:{trigger:'axis',backgroundColor:'rgba(10,14,32,.92)',borderColor:'rgba(129,140,248,.4)',textStyle:{color:'#e8ecff'}},
   legend:{top:0,textStyle:{color:'#aab2e6'},icon:'roundRect',itemWidth:14,itemHeight:8},
   xAxis:{type:'category',boundaryGap:false,data:d.daily.map(x=>x.d.slice(5)),
     axisLine:{lineStyle:{color:'rgba(120,140,255,.25)'}},axisLabel:{color:'#8b93c4',fontSize:11}},
   yAxis:{type:'value',axisLabel:{color:'#8b93c4'},splitLine:{lineStyle:{color:'rgba(120,140,255,.1)'}}},
   series:[
    {name:'启动',type:'line',smooth:true,symbol:'circle',symbolSize:6,showSymbol:false,
     data:d.daily.map(x=>x.startups),lineStyle:{width:2.5,color:'#5eead4'},itemStyle:{color:'#5eead4'},
     areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,
       [{offset:0,color:'rgba(94,234,212,.35)'},{offset:1,color:'rgba(94,234,212,0)'}])}},
    {name:'回复成功',type:'bar',barWidth:'48%',data:d.daily.map(x=>x.replies),
     itemStyle:{borderRadius:[4,4,0,0],color:new echarts.graphic.LinearGradient(0,1,0,0,
       [{offset:0,color:'rgba(129,140,248,.95)'},{offset:1,color:'rgba(167,139,250,.5)'}])}}]});
  const ve=echarts.init(document.getElementById('ver'));
  ve.setOption({tooltip:{backgroundColor:'rgba(10,14,32,.92)',textStyle:{color:'#e8ecff'}},
   legend:{bottom:0,textStyle:{color:'#aab2e6'}},
   series:[{type:'pie',radius:['45%','72%'],center:['50%','44%'],
     itemStyle:{borderColor:'#0a0e20',borderWidth:3,borderRadius:6},
     label:{color:'#e8ecff',fontSize:12},labelLine:{lineStyle:{color:'rgba(120,140,255,.4)'}},
     data:d.versions.map(v=>({name:v.v||'unknown',value:v.c}))}]});
  iptb.innerHTML = d.by_ip.length ? d.by_ip.map(r=>
    '<tr><td class="ip">'+r.ip+'</td><td>'+r.count+'</td><td>'+r.replies+'</td><td>'+fmt(r.first)+'</td><td>'+fmt(r.last)+'</td></tr>').join('')
   : '<tr><td colspan="5" style="text-align:center;color:#7e86bf;padding:26px">暂无数据</td></tr>';
  status.innerHTML='已更新 · <b>'+tmf.format(new Date())+'</b>';
 }).catch(()=>{status.textContent='连接失败，请检查服务';});
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
