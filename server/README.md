# VisReply 使用统计服务端

零依赖（Python 标准库 + SQLite + ECharts CDN）。仪表盘用 Python + ECharts 实现，比 PHP 好看。

## 功能
- `POST /collect`：软件端静默上报（事件 `startup` / `reply{sent_ok}`）。**IP 由服务器从连接层取，客户端不传。**
- `GET /`：仪表盘页面，展示：总启动次数 / 自动回复成功次数 / 活跃设备数 / 独立 IP 数、近 30 天趋势、版本分布、使用人 IP 排行 Top 20。
- `GET /api/data`：聚合数据 JSON，供前端调用。

## 运行（本地试）
```bash
cd server
python telemetry_server.py --port 8000 --db telemetry.db
# 浏览器打开 http://127.0.0.1:8000/
```

## 宝塔部署（Python 项目 + 反代）

### A. 添加 Python 项目（宝塔「网站 → Python 项目 → 添加」逐字段）
| 字段 | 填法 |
|---|---|
| 项目名称 | `telemetry` |
| Python 环境 | 任意 3.7+（如 Python 3.10.12，本脚本零依赖） |
| 启动方式 | **命令行启动**（⚠️ 不要选 uwsgi/gunicorn：本服务是自带端口的 `ThreadingHTTPServer`，不是 WSGI 应用，选那两个会起不来） |
| 项目路径 | `/www/wwwroot/a.fangzhoui.cn/visreply/telemetry`（`telemetry_server.py` 放这里） |
| 启动命令 | `python telemetry_server.py --port 8000 --db /www/wwwroot/a.fangzhoui.cn/visreply/telemetry/telemetry.db` |
| 环境变量 | 无 |
| 启动用户 | `www` |
| 安装依赖包 | 留空（零依赖） |

> 端口 `8000` 若被占用，换一个（如 `8137`），反代目标同步改。

### B. 反向代理（站点 → 设置 → 反向代理 → 添加）
- 代理名称：`telemetry`
- 目标 URL：`http://127.0.0.1:8000`
- 发送域名：`a.fangzhoui.cn`
- 缓存：关

服务端已做**前缀兼容**：无论反代是否剥掉 `/visreply/telemetry/` 前缀都能命中（`/`、`/api/data`、`/collect` 均按后缀匹配）。
最终 `https://a.fangzhoui.cn/visreply/telemetry/` 打开看板，客户端上报到 `.../visreply/telemetry/collect`。

> 备选：不用 Python 项目管理器，直接 `nohup python3 telemetry_server.py --port 8000 &` 或宝塔进程守护插件。

### C. 生效
客户端改动需重新打包（PYZ 机制）才在真机生效，随下次 OTA 发版带上。

## 安全提示
- 看板已按方哥要求**不鉴权**，任何人猜到 URL 都能看 IP 排行。建议把目录放到不常见路径（如 `visreply/t-<随机串>/`），
  或在宝塔 Nginx 加 `allow/deny` 限制你的 IP。需要 Basic Auth 一行即可补上。
- 数据仅存匿名事件 + 服务器侧 IP，不含账号/聊天内容，合规面已压到最小。

## 重置数据
删除 `telemetry.db` 后重启服务即可清空所有统计。
