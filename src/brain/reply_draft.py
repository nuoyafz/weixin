"""ReplyDraft - structured reply draft data class.

Aligned with original app.brain.reply_draft - holds the reply content
along with metadata like decision, confidence, and source.
"""
from __future__ import annotations

from typing import Any, Dict


class ReplyDraft:
    """Structured reply draft for WeChat message responses."""

    def __init__(
        self,
        content: str = "",
        decision: str = "reply",
        confidence: float = 0.0,
        source: str = "",
        contact_key: str = "",
        contact_name: str = "",
        original_message: str = "",
        metadata: Dict[str, Any] = None,
    ):
        self.content = content
        self.decision = decision
        self.confidence = confidence
        self.source = source
        self.contact_key = contact_key
        self.contact_name = contact_name
        self.original_message = original_message
        self.metadata = metadata or {}

    @property
    def is_empty(self) -> bool:
        return not bool(self.content.strip())

    @property
    def should_send(self) -> bool:
        return self.decision in ("reply", "reply_draft", "weak_lead_draft") and not self.is_empty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "decision": self.decision,
            "confidence": self.confidence,
            "source": self.source,
            "contact_key": self.contact_key,
            "contact_name": self.contact_name,
            "original_message": self.original_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplyDraft":
        return cls(
            content=data.get("content", ""),
            decision=data.get("decision", "reply"),
            confidence=data.get("confidence", 0.0),
            source=data.get("source", ""),
            contact_key=data.get("contact_key", ""),
            contact_name=data.get("contact_name", ""),
            original_message=data.get("original_message", ""),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"ReplyDraft(decision={self.decision!r}, "
            f"content={self.content[:50]!r}, "
            f"confidence={self.confidence:.2f})"
        )