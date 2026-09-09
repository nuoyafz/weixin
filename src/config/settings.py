import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ================================================================
# 密钥外置（安全修复#1）
# ----------------------------------------------------------------
# config.yaml 里不再存放明文 api_key，只写 `${VAR}` 占位；真实值从
# 环境变量或项目根的 .env 读取。这样：
#   - 源码仓库 / 打包产物里都搜不到 sk- 明文（避免密钥随安装包外泄）
#   - .env 不进版本库、不被打包，密钥只在本地存在
# ================================================================

# 项目根目录（my_agent）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 整值引用 ${VAR}，以及字符串内嵌 ${VAR}
_ENV_WHOLE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_INLINE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# 默认环境变量名：config.yaml 里 api_key 为空时的兜底来源
_DEFAULT_SECRET_KEYS = ("VISREPLY_API_KEY", "ALIYUN_API_KEY")


def _load_dotenv(path: Optional[Path] = None) -> None:
    """加载项目根 .env（简易 loader，零第三方依赖）。

    已存在的系统环境变量优先，不被 .env 覆盖。
    """
    p = Path(path) if path else (_PROJECT_ROOT / ".env")
    try:
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def _expand_env_value(value: Any) -> Any:
    """把配置里的 ${VAR} 引用替换为环境变量真实值（支持 dict/list 递归）。"""
    if isinstance(value, str):
        stripped = value.strip()
        m = _ENV_WHOLE.match(stripped)
        if m:
            return os.environ.get(m.group(1), "")
        return _ENV_INLINE.sub(
            lambda mm: os.environ.get(mm.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    return value


def _inject_secret_from_env(raw: Dict[str, Any]) -> Dict[str, Any]:
    """model 段 api_key 为空时，用环境变量兜底填充。

    保证「config.yaml 里留空」也能正常工作，不必强制写 ${VAR}。
    """
    for section in ("vision_model", "text_model"):
        sec = raw.get(section)
        if not isinstance(sec, dict):
            continue
        cur = str(sec.get("api_key") or "").strip()
        if cur:
            continue
        for env_name in _DEFAULT_SECRET_KEYS:
            val = str(os.environ.get(env_name) or "").strip()
            if val:
                sec["api_key"] = val
                break
    return raw


def expand_env_config(raw: Any) -> Any:
    """对外入口：供其它自行 yaml.safe_load 的模块复用同一套展开逻辑。"""
    _load_dotenv()
    raw = _expand_env_value(raw)
    if isinstance(raw, dict):
        raw = _inject_secret_from_env(raw)
    return raw


@dataclass
class VisionModelConfig:
    provider: str = "local"
    base_url: str = ""
    api_key: str = ""
    api_key_required: bool = False
    api_style: str = "local_ocr"
    model: str = ""
    timeout_seconds: int = 110
    image_input: bool = True
    image_detail: str = "high"
    image_content_format: str = "openai_image_url"
    temperature: float = 0.1
    max_tokens: int = 1800
    ocr_enabled: bool = True
    ocr_text_score: float = 0.5


@dataclass
class TextModelConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    api_key_required: bool = True
    api_style: str = "openai_chat"
    model: str = ""
    timeout_seconds: int = 110
    temperature: float = 0.2
    max_tokens: int = 1200


@dataclass
class WeChatConfig:
    window_title_keywords: List[str] = field(default_factory=lambda: ["微信", "WeChat"])
    system_contacts: List[str] = field(default_factory=lambda: [
        "微信支付", "微信游戏", "服务号", "公众号", "订阅号",
        "微信团队", "QQ邮箱提醒", "妈咪", "爸比", "老爸", "老妈",
        "妈妈", "爸爸", "老公", "老婆", "媳妇", "宝宝", "宝贝",
        "亲爱的", "我的一家人"
    ])
    enable_rpa_send: bool = True
    capture_backend: str = "printwindow"
    background_mode_enabled: bool = True
    background_mode_strategy: str = "managed_display"
    managed_display_required: bool = True
    managed_display_device: str = ""
    managed_display_window_width: int = 900
    managed_display_window_height: int = 680
    background_send_enabled: bool = True
    background_text_clipboard_paste_enabled: bool = True
    foreground_keyboard_mode: str = "efficiency"
    input_box_click_ratio_x: float = 0.68
    input_box_click_ratio_y: float = 0.9
    send_click_delay_seconds: float = 0.15
    between_messages_delay_seconds: float = 1.2
    auto_open_unread: bool = True
    ocr_region_filter_enabled: bool = True
    ocr_region_fallback_enabled: bool = True
    ocr_reply_stabilize_enabled: bool = True
    ocr_reply_stable_frames: int = 1
    ocr_group_sender_split_enabled: bool = True
    contact_blacklist: List[str] = field(default_factory=lambda: [
        "拼多多", "瑞幸咖啡", "美团", "饿了么", "滴滴", "抖音", "快手",
        "京东", "淘宝", "天猫", "订阅号", "服务号", "公众号", "微信团队",
        "微信支付", "微信运动", "文件传输助手", "腾讯新闻", "腾讯文档",
        "折叠的聊天", "群助手", "群消息", "京东快递", "顺丰速运", "菜鸟",
        "中国移动", "中国联通", "中国电信", "10086", "10000", "95588", "支付宝",
    ])
    # 识别模式：
    #   double_click_pin = 双击「聊天」图标把未读会话置顶后点第一行（推荐，能兜住红点漏检）
    #   red_dot          = 红点直点（其他模式，不双击，直接扫列表红点点击，原始行为）
    recognition_mode: str = "double_click_pin"
    # T2 视觉定位（YOLOv8）开关：
    #   rule_based = 走现有规则/比例定位（默认，已稳定）
    #   yolo       = 启用 yolo_locator 目标检测（需先训练模型并放置权重）
    # 模型缺失或推理异常时，调用方必须回退 rule_based，不可中断主流程。
    localization_mode: str = "rule_based"
    yolo_model_path: str = "models/yolo_wechat_loc_n.onnx"


@dataclass
class NaturalReplyConfig:
    enabled: bool = True
    use_casual_you: bool = True
    segment_send_enabled: bool = True
    max_segments: int = 2
    max_send_segments: int = 3
    max_chars_per_segment: int = 64
    segment_delay_min_seconds: float = 0.8
    segment_delay_max_seconds: float = 1.8
    compose_pause_enabled: bool = True
    compose_pause_min_seconds: float = 0.35
    compose_pause_max_seconds: float = 1.5
    compose_pause_per_char_seconds: float = 0.025


@dataclass
class RAGConfig:
    enabled: bool = True
    chunk_chars: int = 900
    chunk_overlap_chars: int = 120
    top_k: int = 5
    min_score: float = 0.18
    embedding_provider: str = "local_hash"
    embedding_dimensions: int = 384
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_timeout_seconds: int = 30


@dataclass
class ReplyRulesConfig:
    """回复规则（对应原版「回复规则」页签）。
    keyword_reply 为规则列表，格式：`关键词,关键词 => 回复`
    skip_contacts 为额外跳过联系人（叠加 wechat.system_contacts）。
    """
    keyword_reply: List[str] = field(default_factory=list)
    skip_contacts: List[str] = field(default_factory=list)


@dataclass
class SafetyConfig:
    send_mode: str = "normal"
    min_confidence_to_reply: float = 0.6
    block_system_contacts: bool = True
    block_ads: bool = True
    block_payment_notices: bool = True


@dataclass
class AssistantConfig:
    auto_loop_enabled: bool = True
    loop_interval_seconds: int = 5
    online_fast_mode_enabled: bool = True
    idle_loop_interval_seconds: float = 2.0
    active_loop_interval_seconds: float = 0.0
    human_like_min_delay_seconds: int = 2
    human_like_max_delay_seconds: int = 8
    skip_cycle_when_no_unread: bool = True


@dataclass
class UIConfig:
    theme: str = "light"
    compact_mode: bool = True
    log_level: str = "normal"


@dataclass
class Settings:
    app_name: str = "VisReply"
    app_version: str = "v1.0.0"
    vision_model: VisionModelConfig = field(default_factory=VisionModelConfig)
    text_model: TextModelConfig = field(default_factory=TextModelConfig)
    wechat: WeChatConfig = field(default_factory=WeChatConfig)
    natural_reply: NaturalReplyConfig = field(default_factory=NaturalReplyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    reply_rules: ReplyRulesConfig = field(default_factory=ReplyRulesConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    knowledge_root: str = "data/knowledge"
    sop_enabled: bool = False


def _pkg_version() -> str:
    """版本唯一真相源：src/__init__.py 的 __version__。

    【2026-09-07 修复左下角 0.0.0】原实现用 Path(__file__) 读磁盘源文件，
    打包后 settings.py 进 PYZ、_internal/src/__init__.py 不存在 → 读文件
    必失败 → 恒 0.0.0。改为：优先 import（源码/打包都通），磁盘读取仅作兜底。
    """
    try:
        from src import __version__ as _v
        return str(_v)
    except Exception:
        pass
    try:
        p = Path(__file__).resolve().parents[1] / "__init__.py"
        t = p.read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', t)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "0.0.0"


def load_settings(config_path: Optional[str] = None) -> Settings:
    settings = Settings()
    if config_path is None:
        # 默认读取项目根目录的 config.yaml（与 UI 写入/读取的是同一份）。
        # 否则 load_config() 无参调用时永远返回空默认配置，导致
        # text_model 的 base_url/api_key/model 全空 -> available()=False ->
        # 本地模式永不调用 LLM -> 机器人永不回复。
        candidates = [
            Path(__file__).resolve().parents[2] / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw:
            # 安全修复#1：加载 .env 并展开 ${VAR}/空值兜底，
            # 使 api_key 可以不以明文出现在 config.yaml 里。
            raw = expand_env_config(raw)
            settings = _apply_override(settings, raw)

    # 回复规则里的额外跳过联系人合并进系统联系人（去重保序），
    # 保证解析与决策使用同一份跳过名单。
    extra_skip = list(settings.reply_rules.skip_contacts)
    if extra_skip:
        seen = set(settings.wechat.system_contacts)
        for name in extra_skip:
            if name and name not in seen:
                settings.wechat.system_contacts.append(name)

    # 联系人黑名单也合并进系统联系人，确保解析层（MessageParser 等）
    # 与决策层使用同一份跳过名单。
    if settings.wechat.contact_blacklist:
        seen = set(settings.wechat.system_contacts)
        for name in settings.wechat.contact_blacklist:
            if name and name not in seen:
                settings.wechat.system_contacts.append(name)

    # 版本号唯一真相源：始终以 src/__init__.py 的 __version__ 为准，
    # 忽略 config.yaml 的 app.version（避免显示过时版本）。
    settings.app_version = "v" + _pkg_version()
    return settings


def _apply_override(settings: Settings, raw: Dict[str, Any]) -> Settings:
    if raw.get("app"):
        settings.app_name = raw["app"].get("name", settings.app_name)
        # 注意：app_version 不再从 config.yaml 读取（易与代码版本脱节），
        # 统一由 src/__init__.py 的 __version__ 驱动，见 load_settings 末尾。

    if raw.get("vision_model"):
        _apply_to(settings.vision_model, raw["vision_model"])

    if raw.get("text_model"):
        _apply_to(settings.text_model, raw["text_model"])

    if raw.get("wechat"):
        wc = raw["wechat"]
        if "capture_backend" in wc:
            settings.wechat.capture_backend = wc["capture_backend"]
        if "background_mode_enabled" in wc:
            settings.wechat.background_mode_enabled = wc["background_mode_enabled"]
        if "background_mode_strategy" in wc:
            settings.wechat.background_mode_strategy = wc["background_mode_strategy"]
        if "managed_display_required" in wc:
            settings.wechat.managed_display_required = wc["managed_display_required"]
        if "ocr_region_filter_enabled" in wc:
            settings.wechat.ocr_region_filter_enabled = wc["ocr_region_filter_enabled"]
        if "ocr_group_sender_split_enabled" in wc:
            settings.wechat.ocr_group_sender_split_enabled = wc["ocr_group_sender_split_enabled"]
        if "localization_mode" in wc:
            settings.wechat.localization_mode = wc["localization_mode"]
        if "yolo_model_path" in wc:
            settings.wechat.yolo_model_path = wc["yolo_model_path"]

    if raw.get("natural_reply"):
        _apply_to(settings.natural_reply, raw["natural_reply"])

    if raw.get("rag"):
        _apply_to(settings.rag, raw["rag"])

    if raw.get("safety"):
        _apply_to(settings.safety, raw["safety"])

    if raw.get("reply_rules"):
        _apply_to(settings.reply_rules, raw["reply_rules"])

    if raw.get("assistant"):
        _apply_to(settings.assistant, raw["assistant"])

    return settings


def _apply_to(target: Any, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if isinstance(current, bool) or isinstance(current, int) or isinstance(current, float) or isinstance(current, str):
                setattr(target, key, value)
            elif isinstance(current, list):
                setattr(target, key, value)
