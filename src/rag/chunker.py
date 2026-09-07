"""文本分块器。

提供统一的检索用分块数据类 Chunk，以及 TextChunker：按段落 / 句子 / 字符数
逐级分块，支持 chunk_chars（单块最大字符数）与 overlap（相邻块重叠字符数）。
仅依赖标准库，可与 knowledge_base.KnowledgeChunk 共存。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """可检索的文本分块。字段命名与 knowledge_base.KnowledgeChunk 对齐，便于互转。"""
    text: str
    source: str = ""
    offset: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    hash_id: str = ""
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.hash_id:
            self.hash_id = compute_hash_id(
                self.text, self.source, self.offset,
                self.metadata.get("title", ""),
                self.metadata.get("hierarchy", ""),
            )


def compute_hash_id(text: str, source: str = "", offset: int = 0,
                    title: str = "", hierarchy: str = "") -> str:
    """计算分块稳定 id（与 KnowledgeChunk 的 md5 风格一致）。"""
    key = f"{source}:{offset}:{title}:{hierarchy}:{text[:200]}"
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


# 句子边界：中文/英文句末标点，保留分隔符。
_SENT_SPLIT = re.compile(r"([。！？!?；;])")
_ATX_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*?)\s*$")


class TextChunker:
    """纯文本分块器。

    - 先按段落切分（尊重段落边界，段尾强制换块）；
    - 段内按句子贪心合并；
    - 超长内容按字符数硬切，并施加 overlap 滑动重叠。
    """

    def __init__(self, chunk_chars: int = 900, overlap: int = 120) -> None:
        if chunk_chars <= 0:
            chunk_chars = 900
        if overlap < 0:
            overlap = 0
        if overlap >= chunk_chars:
            # 重叠不能大于等于块长，否则退化为跳过。
            overlap = max(0, chunk_chars // 5)
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    # ---- 对外 API ----

    def split_text(self, text: str) -> List[str]:
        """返回切分后的文本片段列表（无重叠处理的纯切分，主要供测试）。"""
        return [c.text for c in self.chunk(text or "")]

    def split(self, text: str) -> List[str]:
        """split_text 的别名。"""
        return self.split_text(text)

    def chunk(self, text: str, source: str = "", offset: int = 0,
              metadata: Optional[dict] = None) -> List[Chunk]:
        """将文本分块并返回 Chunk 对象列表。"""
        text = (text or "").strip("\n \t")
        if not text:
            return []

        blocks = self._split_into_chunks(text)

        chunks: List[Chunk] = []
        cur_offset = offset
        for i, part in enumerate(blocks):
            meta = dict(metadata or {})
            if i > 0 and self.overlap:
                meta["overlap"] = True
            chunks.append(Chunk(
                text=part,
                source=source,
                offset=cur_offset,
                metadata=meta,
            ))
            # offset 仅作近似源内位置，重叠会使其略有偏差，但不影响检索。
            cur_offset += len(part)
        return chunks

    # ---- 内部实现 ----

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        sent = _SENT_SPLIT.split(text)
        sents = []
        for i in range(0, len(sent) - 1, 2):
            sents.append((sent[i] + (sent[i + 1] if i + 1 < len(sent) else "")).strip())
        if sent and len(sent) % 2 == 1:
            sents.append(sent[-1].strip())
        # 处理空文本或纯末尾标点等边界
        return [s for s in sents if s]

    def _overlap_tail(self, text: str) -> str:
        if self.overlap <= 0 or len(text) <= self.overlap:
            return ""
        return text[-self.overlap:]

    def _hard_split(self, text: str) -> List[str]:
        """超长内容按字符滑动切分，连续片段互相重叠 overlap 字符。"""
        if not text:
            return []
        if len(text) <= self.chunk_chars:
            return [text]
        step = max(1, self.chunk_chars - self.overlap)
        parts = []
        i = 0
        while i < len(text):
            parts.append(text[i:i + self.chunk_chars])
            i += step
            if i >= len(text):
                break
        return [p for p in parts if p.strip()]

    def _split_into_chunks(self, text: str) -> List[str]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[str] = []
        buf = ""

        for para in paragraphs:
            for sent in self._split_sentences(para):
                if not sent:
                    continue
                if len(buf) + len(sent) <= self.chunk_chars:
                    buf = (buf + sent).strip()
                    continue
                # 当前 buf 已满：先取重叠尾部（在清空之前），再落块。
                tail = self._overlap_tail(buf)
                chunks.append(buf)
                buf = ""
                if len(sent) <= self.chunk_chars:
                    buf = (tail + sent).strip()
                else:
                    # 单句超长：带上重叠尾部一起硬切。
                    chunks.extend(self._hard_split((tail + sent).strip()))
                    buf = ""
            # 段落结束：强制换块，保持"按段落"语义。
            if buf:
                chunks.append(buf)
                buf = ""

        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]