"""RAGIndexer - index documents for RAG retrieval.

Aligned with original app.rag.indexer - indexes documents with embeddings
for semantic search in the RAG pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


class RAGIndexer:
    """Index documents for RAG retrieval."""

    def __init__(self, config=None, embeddings=None):
        self._config = config or {}
        self._embeddings = embeddings
        self._documents = []
        self._index = {}

    def add_document(self, doc_id: str, content: str,
                     metadata: dict[str, Any] = None) -> None:
        """Add a document to the index."""
        self._documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add multiple documents."""
        for doc in documents:
            self.add_document(
                doc.get("id", ""),
                doc.get("content", ""),
                doc.get("metadata"),
            )

    def build_index(self) -> None:
        """Build the search index."""
        self._index = {}
        for doc in self._documents:
            content = doc.get("content", "")
            words = content.lower().split()
            for word in words:
                if word not in self._index:
                    self._index[word] = []
                if doc["id"] not in self._index[word]:
                    self._index[word].append(doc["id"])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed documents."""
        if not self._index:
            self.build_index()

        query_words = query.lower().split()
        scores = {}
        for word in query_words:
            if word in self._index:
                for doc_id in self._index[word]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            for doc in self._documents:
                if doc["id"] == doc_id:
                    results.append({**doc, "score": score})
                    break

        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "documents": self._documents,
                    "index": self._index,
                }, f, ensure_ascii=False)
        except Exception:
            pass

    def load(self, path: str) -> None:
        """Load index from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._documents = data.get("documents", [])
            self._index = data.get("index", {})
        except Exception:
            pass

    def incremental(self) -> None:
        """增量更新索引（对齐原版 RagIndexer.incremental）。"""
        self.build_index()


# 兼容原版类名别名
RagIndexer = RAGIndexer