"""检索器。

对 query 与分块做余弦相似度 Top-K 检索，支持"词法 + 向量"混合打分：
    score = vector_weight * 向量余弦 + (1 - vector_weight) * 词法重叠。
纯标准库，向量部分由 EmbeddingProvider 提供（内部已含本地降级）。
"""
from __future__ import annotations

from typing import List, Optional

from .chunker import Chunk
from .embeddings import EmbeddingProvider


def bigrams(text: str) -> set:
    """生成字符 2-gram 集合（混合分数用的词法特征）。"""
    t = (text or "").strip()
    if not t:
        return set()
    if len(t) == 1:
        return {t}
    return {t[i:i + 2] for i in range(len(t) - 1)}


class Retriever:
    """对已索引分块做 Top-K 检索。

    Args:
        embedder: EmbeddingProvider 实例。
        top_k: 默认返回条数。
        min_score: 过滤阈值（混合后得分）。
        vector_weight: 向量得分权重（0~1），1 为纯向量，0 为纯词法。
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        vector_weight: float = 0.6,
    ) -> None:
        self.embedder = embedder or EmbeddingProvider()
        self.top_k = max(1, int(top_k))
        self.min_score = min_score
        self.vector_weight = max(0.0, min(1.0, vector_weight))

    def query(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None,
              min_score: Optional[float] = None) -> List[Chunk]:
        """对给定分块列表执行混合检索，返回按得分降序的 top-K 分块（原地置 score）。"""
        query = (query or "").strip()
        if not query or not chunks:
            return []

        top_k = self.top_k if top_k is None else max(1, int(top_k))
        min_score = self.min_score if min_score is None else min_score

        q_emb = self.embedder.encode_single(query)
        q_gr = bigrams(query)

        # 批量向量化分块（已带缓存的块不再重复编码）。
        need_embed = []
        for c in chunks:
            if not c.embedding:
                need_embed.append(c)
                continue
            if len(c.embedding) != len(q_emb):
                need_embed.append(c)
        if need_embed:
            texts = [c.text for c in need_embed]
            embs = self.embedder.encode(texts)
            for c, emb in zip(need_embed, embs):
                c.embedding = emb

        scored: List[Chunk] = []
        w = self.vector_weight
        for c in chunks:
            vec_sim = self.embedder.cosine(q_emb, c.embedding) if c.embedding else 0.0
            lex_sim = self._lexical(q_gr, c.text)
            score = w * vec_sim + (1.0 - w) * lex_sim
            if score >= min_score:
                c.score = score
                scored.append(c)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    retrieve = query  # 语义别名

    def _lexical(self, q_gr: set, text: str) -> float:
        t_gr = bigrams(text)
        if not q_gr or not t_gr:
            return 0.0
        return len(q_gr & t_gr) / len(q_gr)

    @staticmethod
    def build_context(results: List[Chunk], max_chars: int = 2000) -> str:
        """把检索结果拼成可直接注入 prompt 的上下文文本。"""
        parts = []
        total = 0
        for c in results:
            source = c.source or "knowledge"
            heading = c.metadata.get("title", "") if c.metadata else ""
            if heading:
                block = f"[{source}|{heading}] {c.text}"
            else:
                block = f"[{source}] {c.text}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)