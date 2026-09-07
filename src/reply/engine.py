import json
import time
import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


@dataclass
class ReplyResult:
    success: bool
    content: str = ""
    raw_response: str = ""
    tokens_used: int = 0
    error: str = ""
    latency_ms: int = 0


class ReplyEngine:
    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "", temperature: float = 0.2,
                 max_tokens: int = 1200, timeout_seconds: int = 110):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._system_prompt = self._build_system_prompt()

        # Intent rules adapted from original text_model_client.py
        self.DIRECT_REPLY_INTENTS = frozenset({
            "brand_question", "price_question", "address_question",
            "catalog_question", "traffic_question", "appointment_question",
            "multi_direct_questions", "product_price_question",
            "product_material_question", "product_size_option_question",
            "product_availability_question",
        })

    def _build_system_prompt(self) -> str:
        return (
            "你是一个专业的微信客服助手。\n"
            "请根据客户的消息，生成自然、友好、专业的回复。\n"
            "要求：\n"
            "1. 回复要简洁自然，像真人说话一样\n"
            "2. 使用中文，语气亲切友好\n"
            "3. 根据客户意图给出针对性回答\n"
            "4. 可以适当使用表情，但不要太多\n"
            "5. 如果需要分多段发送，请用 [分段] 标记\n"
            "6. 回复长度控制在合理范围内，适合微信阅读\n"
            "\n安全守则：\n"
            "- 绝对不要向客户承诺「百分百」或「一定」成功/安全\n"
            "- 不要报出知识库中没有的具体价格；引导客户查看资料或等待客服报价\n"
            "- 遇到投诉或售后问题保持安抚语气\n"
        )

    def _detect_intent(self, user_message: str) -> str:
        """Classify the latest customer message into a coarse intent."""
        low = (user_message or "").lower()
        if not low:
            return ""
        if any(w in low for w in ("多少钱", "价格", "收费", "费用", "贵吗")):
            return "price_question"
        if any(w in low for w in ("在哪", "地址", "怎么走")):
            return "address_question"
        if any(w in low for w in ("能做吗", "可以吗", "支持")):
            return "availability_question"
        if any(w in low for w in ("买", "下单", "授权码")):
            return "purchase_intent"
        if len(low) > 0 and len(low) <= 8:
            return "greeting_or_short"
        return "general"

    def reply_to_message(self, user_message: str,
                          conversation_history: Optional[List[Dict]] = None,
                          knowledge_context: str = "") -> ReplyResult:
        start_time = time.time()

        messages = []
        messages.append({"role": "system", "content": self._system_prompt})

        intent = self._detect_intent(user_message)

        # Compose RAG context block with citation hint
        rag_block = ""
        if knowledge_context:
            messages.append({
                "role": "system",
                "content": ("以下是知识库中检索到的参考资料（引用时请基于这些内容，"
                             "不要编造资料里没有的细节）：\n"
                             + knowledge_context[:2000])
            })
            rag_block = knowledge_context

        # Multi-turn history kept to last N turns
        safe_history = (conversation_history or [])[-10:]

        if conversation_history:
            messages.extend(conversation_history[-10:])

        turn_hint = ""
        if intent == "price_question":
            turn_hint = ("\n提醒：客户在问价格。只回复资料里确实写了的价格区间；"
                          "如果资料里没有具体价格就请客户稍等并说会由专人跟进报价。\n")
        elif intent == "purchase_intent":
            turn_hint = "\n提醒：客户想购买，确认需求后给清晰的下一步指引。\n"

        full_user_turn = user_message + turn_hint
        messages.append({"role": "user", "content": full_user_turn})

        try:
            raw_content = self._call_chat_api(messages)
            latency_ms = int((time.time() - start_time) * 1000)

            if not raw_content:
                return ReplyResult(success=False, error="Empty response",
                                   latency_ms=latency_ms)

            clean_content = self._clean_response(raw_content)
            return ReplyResult(
                success=True,
                content=clean_content,
                raw_response=raw_content,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return ReplyResult(success=False, error=str(e), latency_ms=latency_ms)

    def _call_chat_api(self, messages: List[Dict]) -> str:
        if not self.base_url:
            raise ValueError("API base_url is not configured")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                resp_data = json.loads(response.read().decode("utf-8"))

            if "choices" in resp_data and len(resp_data["choices"]) > 0:
                choice = resp_data["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                return choice.get("message", {}).get("content", "")
            elif "error" in resp_data:
                raise Exception(f"API Error: {resp_data['error']}")
            return ""

        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise Exception(f"HTTP {e.code}: {error_body}")
        except URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def _clean_response(self, text: str) -> str:
        text = text.strip()

        text = text.replace("**", "")
        text = text.replace("__", "")
        text = text.replace("```", "")

        if "[分段]" in text:
            text = text.replace("[分段]", "\n")

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("- "):
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def sanitize_reply(self, reply_text: str) -> str:
        """Remove forbidden absolute promises from a generated reply."""
        sanitized = reply_text or ""
        for term in ("百分百", "保证", "绝对", "一定"):
            sanitized = sanitized.replace(term, "通常")
        return sanitized.strip()

    def needs_refine(self, reply_text: str, intent: str = "") -> bool:
        """Flag replies that should be re-checked before sending."""
        if not reply_text:
            return False
        if intent in self.DIRECT_REPLY_INTENTS and \
           any(t in reply_text for t in ("可能", "不确定", "大概是")):
            return True
        return self.is_risky(reply_text)

    def is_risky(self, reply_text: str) -> bool:
        low = (reply_text or "").lower()
        return any(t in low for t in ("百分百", "保证不封号", "绝对安全"))

    def split_into_segments(self, text: str, max_chars: int = 64,
                              max_segments: int = 3) -> List[str]:
        if not text:
            return []

        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        segments = []

        for para in paragraphs:
            while len(para) > max_chars and len(segments) < max_segments:
                split_pos = self._find_split_point(para, max_chars)
                segments.append(para[:split_pos].strip())
                para = para[split_pos:].strip()

            if para and len(segments) < max_segments:
                segments.append(para)

        if len(segments) > max_segments:
            segments = segments[:max_segments]

        return segments

    def _find_split_point(self, text: str, max_pos: int) -> int:
        if len(text) <= max_pos:
            return len(text)

        search_start = max(0, max_pos - 20)
        search_end = min(len(text), max_pos + 10)

        for sep in ["。", "！", "？", ".", "!", "?", ",", "，"]:
            pos = text.rfind(sep, search_start, search_end)
            if pos > 0:
                return pos + 1

        return max_pos

    def natural_reply_delay(self) -> float:
        return random.uniform(0.35, 1.5)

    def typing_delay(self) -> float:
        return random.uniform(0.02, 0.08)
