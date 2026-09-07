"""证据门：阻止没有资料支撑的模型回复。

门控与行业无关：它不知道什么是型号、价格、课程、餐厅或维修服务。
它只检查当前轮次传入的已保存资料是否足以支撑回复。
可见的聊天记录不作为证据，因为旧的助手回复可能已经包含错误。
"""
from __future__ import annotations

import re
from typing import Any


class EvidenceGate:

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        safety = self.config.get("safety") or {}
        self.enabled = bool(safety.get("require_reply_basis", True))
        self.min_ratio = float(safety.get("reply_basis_min_ratio", 0.18) or 0.18)

    def check(
        self,
        analysis: dict[str, Any],
        reply: str,
        text_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "reason": "disabled"}

        reply = str(reply or "").strip()
        if not reply:
            return {"ok": True, "reason": "empty_reply"}

        analysis = analysis or {}

        strong_commitments = self._strong_commitment_terms(reply)
        if strong_commitments:
            return {
                "ok": False,
                "reason": "unsafe_strong_commitment",
                "message": "回复包含绝对化承诺，已交给人工确认。",
                "terms": strong_commitments,
            }

        if analysis.get("faq_reply_matched"):
            return {"ok": True, "reason": "faq_reply"}

        if analysis.get("keyword_reply_matched"):
            rules = analysis.get("keyword_reply_rules")
            if not isinstance(rules, list):
                rule = analysis.get("keyword_reply_rule")
                rules = [rule] if isinstance(rule, dict) else []
            sources = {
                str(rule.get("source", ""))
                for rule in rules
                if isinstance(rule, dict)
            }
            if sources and all(s == "keyword_reply" for s in sources):
                return {
                    "ok": True,
                    "reason": "user_approved_keyword_reply",
                    "matched_rules": len(rules),
                }
            if sources and all(s.startswith("dynamic_") for s in sources):
                return {"ok": True, "reason": "structured_business_rule"}

        basis = self._basis_text(analysis)
        if not basis.strip():
            return {
                "ok": False,
                "reason": "no_saved_basis",
                "message": "资料里没有可支撑这条回复的内容。",
            }

        unsupported_terms = self._unsupported_claim_terms(reply, basis)
        if unsupported_terms:
            return {
                "ok": False,
                "reason": "unsupported_claim_terms",
                "message": "回复里包含资料没有写明的技术来源或承诺。",
                "terms": unsupported_terms,
            }

        cited_evidence = self._verified_model_evidence(analysis, reply, basis)
        if cited_evidence:
            return {
                "ok": True,
                "reason": "model_cited_saved_basis",
                "verified_evidence": cited_evidence,
                "basis_ratio": 1,
            }

        verdict = self._grounded(reply, basis)
        if verdict["ok"]:
            verdict["reason"] = "grounded_in_saved_basis"
            return verdict

        return {
            "ok": False,
            "reason": "reply_not_grounded",
            "message": "生成回复没有在资料里找到足够依据。",
            **verdict,
        }

    # ------------------------------------------------------------------
    # 强承诺词检测
    # ------------------------------------------------------------------
    _STRONG_COMMITMENT_PATTERNS = (
        r"保证不[会能]",
        r"保证[没无]有",
        r"保证[可即]",
        r"百分百不",
        r"百分百[没无]",
        r"100%不",
        r"100%[没无]",
        r"绝不[会能]",
        r"永远不会",
        r"绝对不会",
        r"绝对[没无]有",
        r"一定不[会能]",
        r"肯定不会",
        r"肯定不[会能]",
        r"不可能[会]",
        r"从不[会能]",
        r"永久免[费]",
    )

    def _strong_commitment_terms(self, reply: str) -> list[str]:
        hits = []
        for pat in self._STRONG_COMMITMENT_PATTERNS:
            for m in re.finditer(pat, reply):
                span = reply[max(0, m.start() - 2):min(len(reply), m.end() + 2)]
                hits.append(span.strip())
        return hits[:5]

    # ------------------------------------------------------------------
    # 回复依据文本
    # ------------------------------------------------------------------
    def _basis_text(self, analysis: dict[str, Any]) -> str:
        parts: list[str] = []
        rag = analysis.get("rag") if isinstance(analysis.get("rag"), dict) else {}
        for chunk in rag.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            parts.extend([
                str(chunk.get("title") or ""),
                str(chunk.get("content") or ""),
                str(chunk.get("source_label") or ""),
            ])

        for section_name in ("business_profile", "reply_rules", "business"):
            section = self.config.get(section_name)
            if not isinstance(section, dict):
                continue
            parts.extend(str(v or "") for v in section.values())

        learning = self.config.get("learning")
        if isinstance(learning, dict):
            parts.append(str(learning.get("preview_document") or ""))

        for scenario in self.config.get("test_scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            parts.append(str(scenario.get("approved_reply") or ""))
            parts.extend(
                m for m in (scenario.get("messages") or [])
                if str(m or "").strip()
            )

        return "\n".join(p for p in parts if str(p or "").strip())

    # ------------------------------------------------------------------
    # 模型引用证据验证
    # ------------------------------------------------------------------
    def _verified_model_evidence(
        self,
        analysis: dict[str, Any],
        reply: str,
        basis: str,
    ) -> list[str]:
        source = str(analysis.get("reply_source") or "").strip().lower()
        if source not in frozenset({"faq", "business"}):
            return []

        try:
            confidence = float(analysis.get("reply_confidence", 0))
        except Exception:
            confidence = 0
        if confidence < 0.65:
            return []

        raw_evidence = analysis.get("reply_evidence")
        if isinstance(raw_evidence, str):
            candidates = [raw_evidence]
        elif isinstance(raw_evidence, list):
            candidates = raw_evidence
        else:
            return []

        basis_norm = self._normalize(basis)
        verified: list[str] = []
        for item in candidates[:5]:
            quote = str(item or "").strip()
            quote_norm = self._normalize(quote)
            if len(quote_norm) < 4 or quote_norm not in basis_norm:
                continue
            if not self._evidence_relevant_to_reply(reply, quote):
                continue
            verified.append(quote)

        if not verified:
            return []
        if not self._numeric_claims_are_grounded(reply, basis):
            return []
        return verified

    @classmethod
    def _evidence_relevant_to_reply(cls, reply: str, evidence: str) -> bool:
        reply_norm = cls._strip_fillers(cls._normalize(reply))
        evidence_norm = cls._normalize(evidence)
        ignored = frozenset({
            '情况', '那个', '我们', '目前', '一下', '这个', '提供',
            '需要', '你好', '客户', '进行', '可以', '服务', '您好',
            '相关', '的是', '的话',
        })
        return any(
            gram in evidence_norm and gram not in ignored
            for gram in cls._ngrams(reply_norm, 2)
        )

    @classmethod
    def _strip_fillers(cls, text: str) -> str:
        for filler in (
            "你好", "您好", "老板", "亲", "这边", "我们", "你们",
            "可以", "好的", "是的", "嗯嗯", "嗯", "哈", "哈哈",
            "请问", "方便", "具体", "这个", "那个", "一下",
            "了解", "看看", "呀", "吗", "呢", "的",
        ):
            text = text.replace(filler, "")
        return text.strip()

    def _numeric_claims_are_grounded(self, reply: str, basis: str) -> bool:
        reply_norm = self._normalize(reply)
        basis_norm = self._normalize(basis)
        claims = re.findall(
            r'\d+(?:\.\d+)*(?:%|元|天|周|月|年|小时|分钟|秒|台|次|个)?',
            reply_norm,
        )
        return all(claim in basis_norm for claim in claims)

    def _unsupported_claim_terms(self, reply: str, basis: str) -> list[str]:
        guarded_terms = (
            '自研', '自己训练', '训练', '数据采集', '数据标注', '标注',
            '模型迭代', '大模型', 'deepseek', 'gpt', 'openai', 'qwen', 'claude',
        )
        basis_norm = self._normalize(basis)
        reply_norm = self._normalize(reply)
        return [
            term for term in guarded_terms
            if self._normalize(term) in reply_norm
            and self._normalize(term) not in basis_norm
        ]

    # ------------------------------------------------------------------
    # 文本匹配（对齐原版 n-gram + longest_hit）
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r'[\s，,。！？!?；;：:、~～\-_（）()【】\[\]\"\'""''<>《》]+', '', str(text or "")).lower()

    @staticmethod
    def _ngrams(value: str, size: int) -> list[str]:
        if len(value) < size:
            return [value] if value else []
        return [value[i:i + size] for i in range(len(value) - size + 1)]

    @staticmethod
    def _longest_hit(value: str, basis: str) -> str:
        for size in range(min(len(value), 16), 2, -1):
            for i in range(len(value) - size + 1):
                piece = value[i:i + size]
                if piece in basis:
                    return piece
        return ""

    def _grounded(self, reply: str, basis: str) -> dict[str, Any]:
        reply_norm = self._normalize(reply)
        basis_norm = self._normalize(basis)
        factual = self._strip_fillers(reply_norm)
        if not factual:
            return {"ok": True, "basis_ratio": 1.0, "matched": ""}

        grams = self._ngrams(factual, 4)
        if not grams:
            return {"ok": False, "basis_ratio": 0.0, "matched": ""}

        matched = [g for g in grams if g in basis_norm]
        ratio = len(matched) / len(grams)
        ok = ratio >= self.min_ratio

        if not ok:
            longest = self._longest_hit(factual, basis_norm)
            if longest and len(longest) >= 8:
                return {
                    "ok": True,
                    "basis_ratio": round(ratio, 3),
                    "matched": "",
                    "longest_hit": longest,
                }

        return {
            "ok": ok,
            "basis_ratio": round(ratio, 3),
            "matched": "",
        }