# VisReply · 微信桌面端 AI 助手

> 本地运行的微信桌面端 RPA Agent：自动置顶未读会话、本地 OCR 感知聊天列表、调用大模型生成回复草稿，全程数据留在本地。

作者：漩涡鸣人 · QQ 723167066

---

## ✨ 功能特性

- **未读自动置顶**：扫描微信聊天列表红点，将未读会话自动置顶，不错过重要消息。
- **本地视觉感知**：基于 `rapidocr-onnxruntime` + 自研布局解析，离线识别聊天列表首行与未读状态，无需联网 OCR。
- **AI 回复生成**：接入兼容 OpenAI 协议的文本/视觉模型（默认 Aliyun `maas.aliyuncs.com`），根据会话内容生成可编辑的回复草稿。
- **回复策略丰富**：内置 FAQ、关键词、SOP、报价、弱线索流转、跳过联系人等多套回复策略，可配置开关与置信度阈值。
- **话术 / 知识库 / 规则**：支持知识库检索（RAG）、自定义话术规则、好友请求自动处理。
- **桌面 UI（pywebview）**：Telegram 双蓝主题，含工作台、消息队列、知识库、话术与规则、历史记录、设置等模块；左侧底部「官网」按钮直达发布页。
- **新手引导**：首次启动若未配置 API Key，强制引导去设置页配置，保证可用。
- **一键打包**：PyInstaller onedir + Inno Setup 生成带安装向导的 `VisReply_Setup.exe`。

## 🧱 技术架构

感知 → 决策 → 执行 三层管线：

| 层 | 模块（`src/`） | 职责 |
|---|---|---|
| **感知** | `capture/`、`clean_perception/`、`local_vision/`、`ocr/` | 屏幕采集、布局清洗、本地 OCR、微信列表解析 |
| **决策** | `brain/`、`agent/`、`ai/` | 决策引擎、证据门控、对话策略、模型路由与调用 |
| **执行** | `desktop/`、`rpa/`、`message/`、`ui/` | 窗口管理、微信自动化、消息解析、pywebview 界面 |
| **支撑** | `config/`、`common/`、`rag/`、`reply/`、`storage/` | 配置、日志、检索增强、回复组装、存储 |

## 📂 目录结构

```
my_agent/
├── src/                  # 全部源码（agent/ ai/ brain/ capture/ desktop/ ...）
├── resources/           # 前端资源（html/ 界面与桥接）
├── docs/                # 落地页 index.html + 独立下载页 download.html
├── config.example.yaml  # 配置模板（复制为 config.yaml 后填自己的 key）
├── requirements.txt      # Python 依赖
├── WeChatAIAssistant.spec  # PyInstaller 构建脚本
├── installer.iss        # Inno Setup 安装包脚本
├── run.py               # 启动入口
└── main.py              # 应用引导
```

## 🚀 快速开始（源码运行）

环境要求：Windows + Python 3.10。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备配置（复制模板，填入自己的 api_key）
cp config.example.yaml config.yaml
#   用记事本打开 config.yaml，填 text_model / vision_model 的 api_key

# 3. 启动
python run.py
```

> ⚠️ `config.yaml` 不会被提交到仓库（含密钥）。请始终使用 `config.example.yaml` 作为模板。

## 📦 打包为安装包

```bash
# 1. 构建 onedir（PyInstaller）
pyinstaller WeChatAIAssistant.spec --noconfirm --clean

# 2. 编译安装包（需 Inno Setup 7）
#    产物 -> installer_output/VisReply_Setup.exe
ISCC installer.iss
```

构建脚本会自动在打包阶段清空 `api_key`，安装包不含任何密钥明文。

## ⚙️ 配置说明

主要配置段（详见 `config.example.yaml`）：

- `app`：应用名称、版本、启用开关
- `text_model` / `vision_model`：模型服务地址、api_key、模型名
- `wechat`：微信相关开关（如 `enable_rpa_send`）
- `brain/safety`：回复置信度阈值、跳过联系人、失败兜底
- `rag` / `knowledge` / `reply_rules`：知识库与话术规则
- `ui`：界面与官网地址

## ⚠️ 免责声明

本工具仅供个人学习与技术研究，请遵守微信相关使用条款与当地法律法规。使用过程中产生的任何后果由使用者自行承担。请勿用于商业群发、骚扰或违反平台规则的行为。

## 📄 许可证

个人项目，仅供学习与自用。未经作者授权，请勿用于商业用途。
