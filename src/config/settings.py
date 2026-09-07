import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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
    return settings


def _apply_override(settings: Settings, raw: Dict[str, Any]) -> Settings:
    if raw.get("app"):
        settings.app_name = raw["app"].get("name", settings.app_name)
        settings.app_version = raw["app"].get("version", settings.app_version)

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
