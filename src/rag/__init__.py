"""RAG 向量检索层。

提供统一的分块 / 解析 / 嵌入 / 检索 / 存储能力，与已有 knowledge_base 共存：

- Chunk / TextChunker      分块器
- DocumentParser           文档解析（txt/md/json/csv，识别标题层级）
- EmbeddingProvider        嵌入（模型优先，无网络降级本地 hash）
- Retriever                词法 + 向量混合 Top-K 检索
- VectorStore              内存索引 + 增量持久化
- knowledge_chunk_to_doc   KnowledgeChunk 适配为可检索 Chunk
- 意图对齐                  align_intent / top_intents

旧接口 KnowledgeBase / KnowledgeChunk 仍可导入，保持向后兼容。
"""
from __future__ import annotations

from .chunker import Chunk, TextChunker, compute_hash_id
from .document_parser import DocumentParser
from .embeddings import EmbeddingProvider
from .intent_keywords import (
    INTENT_KEYWORDS,
    align_intent,
    match_intent_keywords,
    top_intents,
)
from .knowledge_base import KnowledgeBase, KnowledgeChunk
from .retriever import Retriever
from .store import VectorStore, knowledge_chunk_to_doc

__all__ = [
    # 数据与分块
    "Chunk",
    "TextChunker",
    "compute_hash_id",
    # 解析
    "DocumentParser",
    # 嵌入
    "EmbeddingProvider",
    # 意图
    "INTENT_KEYWORDS",
    "align_intent",
    "match_intent_keywords",
    "top_intents",
    # 检索
    "Retriever",
    # 存储 / 适配
    "VectorStore",
    "knowledge_chunk_to_doc",
    # 旧接口兼容
    "KnowledgeBase",
    "KnowledgeChunk",
]