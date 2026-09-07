"""向量存储。

内存内维护分块文档索引，提供增量构建：新增分块按 hash_id 去重、计算向量，
并持久化到 data/knowledge 下的索引 JSON。可与 knowledge_base 共存，
并提供把已有 KnowledgeChunk 转成可检索 Chunk 的适配函数。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .chunker import Chunk, TextChunker, compute_hash_id
from .document_parser import DocumentParser
from .embeddings import EmbeddingProvider
from .retriever import Retriever

# data/knowledge 索引文件名
DEFAULT_INDEX_NAME = "index.json"


def knowledge_chunk_to_doc(chunk, embedder: Optional[EmbeddingProvider] = None) -> Chunk:
    """把 knowledge_base.KnowledgeChunk 转成可检索的 Chunk（适配层）。

    惰性：保留 text/source/offset/hash_id；embedding 仅在传入 embedder 时计算，
    由检索阶段按需批量补齐，避免整库预热。
    """
    meta: Dict[str, Any] = getattr(chunk, "metadata", None) or {}
    hid = getattr(chunk, "hash_id", "") or ""
    out = Chunk(
        text=getattr(chunk, "text", ""),
        source=getattr(chunk, "source", "") or "",
        offset=getattr(chunk, "offset", 0) or 0,
        metadata=dict(meta) if isinstance(meta, dict) else {"junk": meta},
        hash_id=hid,
    )
    if not out.hash_id:
        out.hash_id = compute_hash_id(out.text, out.source, out.offset)
    if embedder is not None:
        out.embedding = embedder.encode_single(out.text)
    return out


class VectorStore:
    """分块文档的内存内索引 + 增量持久化。

    Args:
        index_path: 索引 JSON 路径（默认 <data/knowledge>/index.json）。
        embedder: EmbeddingProvider；缺省自建（内部可降级到本地 hash）。
        chunker: TextChunker；缺省自建。
        top_k / min_score / vector_weight: 由内部 Retriever 使用。
    """

    def __init__(
        self,
        index_path: str = "",
        root_path: str = "data/knowledge",
        embedder: Optional[EmbeddingProvider] = None,
        chunker: Optional[TextChunker] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        vector_weight: float = 0.6,
    ) -> None:
        self.root_path = Path(root_path)
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.root_path.mkdir(parents=True, exist_ok=True)
            self.index_path = self.root_path / DEFAULT_INDEX_NAME

        self.embedder = embedder or EmbeddingProvider(
            dimensions=384, provider="local_hash")
        self.chunker = chunker or TextChunker()
        self.parser = DocumentParser(self.chunker)
        self.retriever = Retriever(
            self.embedder, top_k=top_k, min_score=min_score,
            vector_weight=vector_weight)

        self._chunks: List[Chunk] = []
        self._hash_to_idx: Dict[str, int] = {}
        self.loaded_from = ""

    # ---- 新增分块（增量，按 hash_id 去重） ----

    def add_chunk(self, chunk: Chunk) -> bool:
        self._compute_embedding(chunk)
        if chunk.hash_id in self._hash_to_idx:
            return False
        idx = len(self._chunks)
        self._chunks.append(chunk)
        self._hash_to_idx[chunk.hash_id] = idx
        return True

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """批量加入，返回实际新增数。接受 Chunk 或 KnowledgeChunk。"""
        added = 0
        for c in chunks:
            if isinstance(c, Chunk):
                ok = self.add_chunk(c)
            else:
                ok = self.add_chunk(knowledge_chunk_to_doc(c, self.embedder))
            if ok:
                added += 1
        return added

    def add_text(self, text: str, source: str = "",
                 metadata: Optional[dict] = None) -> int:
        chunks = self.chunker.chunk(text, source=source, metadata=metadata)
        return self.add_chunks(chunks)

    def add_file(self, filepath: str, source: Optional[str] = None) -> int:
        chunks = self.parser.parse_file(filepath, source=source)
        return self.add_chunks(chunks)

    def add_directory(self, directory: str, extensions: Optional[list] = None) -> int:
        exts = {"*.txt", "*.md", "*.json", "*.csv"} if extensions is None else set(extensions)
        d = Path(directory)
        if not d.exists():
            return 0
        files = []
        for e in exts:
            files.extend(d.rglob(e))
        total = 0
        for f in files:
            try:
                total += self.add_file(str(f))
            except Exception:  # noqa: BLE001
                pass
        return total

    # ---- 兼容接待：直接喂入已分块的 KnowledgeBase ----

    def add_knowledge_chunks(self, kchunks: List[Any]) -> int:
        """一次加入多条 KnowledgeChunk（或 Chunk）。"""
        return self.add_chunks(list(kchunks))

    # ---- 向量 ----

    def _compute_embedding(self, chunk: Chunk) -> None:
        if not chunk.embedding:
            chunk.embedding = self.embedder.encode_single(chunk.text)

    def ensure_embeddings(self) -> None:
        """为缺失向量的分块批量补齐（常在对齐模型后端变更后调用）。"""
        need = [c for c in self._chunks if not c.embedding]
        if need:
            embs = self.embedder.encode([c.text for c in need])
            for c, emb in zip(need, embs):
                c.embedding = emb

    # ---- 检索 ----

    def search(self, query: str, top_k: Optional[int] = None,
               min_score: Optional[float] = None) -> List[Chunk]:
        return self.retriever.query(query, self._chunks, top_k=top_k, min_score=min_score)

    def get_context(self, query: str, max_chars: int = 2000,
                    top_k: Optional[int] = None) -> str:
        results = self.search(query, top_k=top_k)
        return self.retriever.build_context(results, max_chars=max_chars)

    # ---- 持久化（增量构建到 data/knowledge 索引 JSON） ----

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "format": "visionlead-rag-store",
            "version": 1,
            "embedder": self.embedder.backend,
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "offset": c.offset,
                    "metadata": c.metadata or {},
                    "hash_id": c.hash_id,
                    "embedding": c.embedding,
                }
                for c in self._chunks
            ],
        }

    def save(self, path: Optional[str] = None) -> str:
        p = Path(path) if path else self.index_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._snapshot(), f, ensure_ascii=False, indent=2)
        return str(p)

    def load(self, path: Optional[str] = None) -> bool:
        p = Path(path) if path else self.index_path
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:  # noqa: BLE001
            return False

        new_chunks: List[Chunk] = []
        for c in state.get("chunks", []):
            if not isinstance(c, dict):
                continue
            chunk = Chunk(
                text=c.get("text", ""),
                source=c.get("source", ""),
                offset=c.get("offset", 0),
                metadata=c.get("metadata") or {},
                hash_id=c.get("hash_id", ""),
            )
            emb = c.get("embedding") or []
            if isinstance(emb, list) and emb:
                chunk.embedding = [float(x) for x in emb]
            new_chunks.append(chunk)

        self._chunks = new_chunks
        self._rebuild_hash_index()
        self.loaded_from = str(p)
        return True

    def _rebuild_hash_index(self) -> None:
        self._hash_to_idx = {c.hash_id: i for i, c in enumerate(self._chunks)}

    # ---- 统计 ----

    def get_stats(self) -> Dict[str, Any]:
        sources = set(c.source for c in self._chunks)
        return {
            "total_chunks": len(self._chunks),
            "total_sources": len(sources),
            "sources": sorted(sources),
            "backend": self.embedder.backend,
            "storage": str(self.index_path),
            "loaded_from": self.loaded_from,
        }

    def __len__(self) -> int:
        return len(self._chunks)