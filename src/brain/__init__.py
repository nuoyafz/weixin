"""决策脑包：决定是否回复、如何回复。

包含核心决策与全链路增强模块：
  DecisionEngine      规则决策链（keyword / faq / sop / ai / fallback / skip）
  FaqReplyMatcher     FAQ 匹配
  KeywordReplyMatcher 关键词匹配
  EvidenceGate        OCR 证据门控
  PresalesSOP         售前 SOP 流程（intro/pricing/objection/closing）
  QuoteReplyPolicy    引用回复决策（原版语义）
  PriceQuotePolicy    价格报价策略
  WeakLeadFollowUp    弱线索跟进
  sop_prompts         各销售阶段提示词常量模板
"""
from __future__ import annotations

from .decision_engine import Decision, DecisionEngine
from .fallback_reply import FallbackReply
from .faq_reply import FaqReplyMatch, FaqReplyMatcher
from .keyword_reply import KeywordReplyMatch, KeywordReplyMatcher
from .natural_reply import NaturalReplyPolisher

from .evidence_gate import EvidenceGate
from .presales_sop import PresalesSOP, SopsStageMatch
from .quote_reply_policy import QuoteReply, QuoteReplyPolicy, PriceQuotePolicy
from .weak_lead_flow import WeakLeadFollowUp, FollowUpTodo
from . import sop_prompts  # noqa: F401

__all__ = [
    # 核心决策
    "Decision", "DecisionEngine", "FallbackReply",
    "FaqReplyMatch", "FaqReplyMatcher",
    "KeywordReplyMatch", "KeywordReplyMatcher",
    "NaturalReplyPolisher",
    # 决策增强
    "EvidenceGate",
    "PresalesSOP", "SopsStageMatch",
    "QuoteReply", "QuoteReplyPolicy", "PriceQuotePolicy",
    "WeakLeadFollowUp", "FollowUpTodo",
    "sop_prompts",
]