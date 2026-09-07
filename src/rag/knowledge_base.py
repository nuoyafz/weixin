import json
import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class KnowledgeChunk:
    text: str
    source: str = ""
    offset: int = 0
    embedding: List[float] = field(default_factory=list)
    hash_id: str = ""
    score: float = 0.0

    def __post_init__(self):
        if not self.hash_id:
            content = f"{self.source}:{self.offset}:{self.text[:200]}"
            self.hash_id = hashlib.md5(content.encode()).hexdigest()


class KnowledgeBase:
    def __init__(self, root_path: str = "data/knowledge",
                 chunk_chars: int = 900,
                 chunk_overlap_chars: int = 120,
                 top_k: int = 5,
                 min_score: float = 0.10):
        self.root_path = Path(root_path)
        self.chunk_chars = chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.top_k = top_k
        self.min_score = min_score
        self._chunks: List[KnowledgeChunk] = []
        self._index_built = False
        self._hash_to_idx: dict = {}

    def load_documents(self, directory: Optional[str] = None) -> int:
        target_dir = Path(directory) if directory else self.root_path
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            return 0

        files = []
        for ext in ["*.txt", "*.md", "*.json", "*.csv"]:
            files.extend(target_dir.rglob(ext))

        # 排除说明类文件（README/使用说明）：它们不是业务知识，
        # 混入会让 LLM 上下文里出现"请在此放置知识库文件"之类的废话，
        # 挤占检索窗口并污染回复风格。
        skip_names = {
            "readme.md", "readme.txt", "readme",
            "说明.txt", "说明.md", "使用说明.txt", "使用说明.md",
        }
        count = 0
        for filepath in files:
            if filepath.name.lower() in skip_names:
                continue
            try:
                chunks = self._load_file(filepath)
                self._chunks.extend(chunks)
                count += len(chunks)
            except Exception as e:
                print(f"[RAG] Failed to load {filepath}: {e}")

        self._build_index()
        print(f"[RAG] Loaded {count} chunks from {len(files)} files")
        return count

    def _load_file(self, filepath: Path) -> List[KnowledgeChunk]:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        source = str(filepath.name)
        return self._chunk_text(text, source)

    def _chunk_text(self, text: str, source: str) -> List[KnowledgeChunk]:
        chunks = []
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = ""
        current_offset = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_chars:
                current_chunk = (current_chunk + "\n" + para).strip()
            else:
                if current_chunk:
                    chunk = KnowledgeChunk(
                        text=current_chunk,
                        source=source,
                        offset=current_offset
                    )
                    chunks.append(chunk)
                    current_offset += len(current_chunk)
                current_chunk = para

                if len(para) > self.chunk_chars:
                    sub_chunks = self._split_long_text(para, source, current_offset)
                    chunks.extend(sub_chunks)
                    current_offset += len(para)
                    current_chunk = ""

        if current_chunk:
            chunk = KnowledgeChunk(
                text=current_chunk,
                source=source,
                offset=current_offset
            )
            chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_text(self, text: str, source: str,
                          base_offset: int) -> List[KnowledgeChunk]:
        chunks = []
        sentences = re.split(r'([。！？.!?])', text)
        current = ""
        offset = base_offset

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if len(current) + len(sentence) <= self.chunk_chars:
                current += sentence
            else:
                if current:
                    chunks.append(KnowledgeChunk(
                        text=current, source=source, offset=offset
                    ))
                    offset += len(current)
                current = sentence

        if current:
            chunks.append(KnowledgeChunk(
                text=current, source=source, offset=offset
            ))

        return chunks

    def _build_index(self) -> None:
        self._hash_to_idx = {}
        for i, chunk in enumerate(self._chunks):
            self._hash_to_idx[chunk.hash_id] = i
        self._index_built = True

    def query(self, query: str) -> List[Tuple[KnowledgeChunk, float]]:
        if not self._index_built:
            self.load_documents()

        results = []
        for chunk in self._chunks:
            similarity = self._lexical_similarity(query, chunk.text)
            if similarity >= self.min_score:
                chunk.score = similarity
                results.append((chunk, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            return results[:self.top_k]

        # 兜底：query 与所有 chunk 无关键词重叠（相似度均低于 min_score）时，
        # 返回前 top_k 个 chunk（而非全部），确保知识库内容至少能被 LLM 取到，
        # 同时避免一次性把 10+ 个 chunk 塞进提示词拖慢模型、增加 token 成本
        # （真机曾因返回全部 13 个 chunk 导致单轮 LLM 调用耗时 ~56s）。
        if self._chunks:
            fallback = self._chunks[: max(self.top_k, len(self._chunks))]
            for c in fallback:
                c.score = 0.0
            return [(c, 0.0) for c in fallback]

        return []

    def get_context(self, query: str, max_chars: int = 2000) -> str:
        results = self.query(query)
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for chunk, score in results:
            if total_chars + len(chunk.text) > max_chars:
                break
            context_parts.append(f"[{chunk.source}] {chunk.text}")
            total_chars += len(chunk.text)

        return "\n\n".join(context_parts)

    def _compute_hash_embedding(self, text: str) -> List[float]:
        embedding = []
        for i in range(0, len(text), 2):
            chunk = text[i:i+4]
            if len(chunk) < 2:
                chunk = chunk + " "
            hash_val = int(hashlib.md5(chunk.encode()).hexdigest()[:8], 16)
            embedding.append((hash_val % 10000) / 10000.0 - 0.5)

        if len(embedding) < 384:
            while len(embedding) < 384:
                embedding.append(0.0)
        elif len(embedding) > 384:
            embedding = embedding[:384]

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0

        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _tokenize(self, text: str) -> set:
        """轻量分词（不依赖 jieba，纯标准库）：中文取相邻 2 字 bigram，英文/数字取小写词。
        足以做中文短查询的关键词重叠检索。"""
        if not text:
            return set()
        text = str(text).lower()
        tokens: set = set()
        for seg in re.findall(r'[\u4e00-\u9fff]+', text):
            if len(seg) == 1:
                tokens.add(seg)
            else:
                for i in range(len(seg) - 1):
                    tokens.add(seg[i:i + 2])
        for w in re.findall(r'[a-z0-9]+', text):
            tokens.add(w)
        return tokens

    def _lexical_similarity(self, query: str, doc: str) -> float:
        """关键词命中率（recall）：query 的 token 被 doc 覆盖的比例，范围 [0,1]。
        比 local hash 余弦更适合中文短查询——能稳定命中含相同业务词的 chunk。"""
        qt = self._tokenize(query)
        dt = self._tokenize(doc)
        if not qt or not dt:
            return 0.0
        hit = len(qt & dt)
        return hit / len(qt)

    def add_document(self, text: str, source: str = "manual") -> int:
        chunks = self._chunk_text(text, source)
        for chunk in chunks:
            if chunk.hash_id not in self._hash_to_idx:
                self._chunks.append(chunk)
        self._build_index()
        return len(chunks)

    def save_state(self, path: str) -> None:
        state = {
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "offset": c.offset,
                    "hash_id": c.hash_id
                }
                for c in self._chunks
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._chunks = [
                KnowledgeChunk(
                    text=c["text"],
                    source=c.get("source", ""),
                    offset=c.get("offset", 0),
                    hash_id=c.get("hash_id", "")
                )
                for c in state.get("chunks", [])
            ]
            self._build_index()
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        sources = set(c.source for c in self._chunks)
        return {
            "total_chunks": len(self._chunks),
            "total_sources": len(sources),
            "sources": list(sources),
            "index_built": self._index_built
        }
