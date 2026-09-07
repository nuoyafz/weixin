"""嵌入提供器。

优先尝试加载可选的中文 embedding 模型（sentence-transformers / transformers），
加载任一失败时自动降级为基于字符 n-gram 的本地 hash 向量，保证无网络、无模型
缓存时也能运行。纯标准库实现，不强制依赖 numpy / torch。
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

# 默认中文/多语言句子模型，仅在有缓存/网络时可用。
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingProvider:
    """统一的文本向量化接口。

    Attributes:
        backend: "model"（真实模型）或 "local_hash"（本地降级）。
        dimension: 输出向量维度。
        model_name: 若使用真实模型，为模型名。
    """

    def __init__(
        self,
        dimensions: int = 384,
        model_name: str = "",
        provider: str = "",
        timeout_seconds: int = 30,
    ) -> None:
        self.dimensions = max(64, int(dimensions or 384))
        self.timeout_seconds = timeout_seconds
        self.backend = "local_hash"
        self.model_name = ""
        self._model = None
        self._tokenizer = None
        self._pipe = None
        self._load_error = ""

        # provider 明确要求本地时跳过模型加载（如 config 里 embedding_provider=local_hash）。
        want_model = provider not in ("", "local_hash")
        if want_model:
            self._try_load_model(model_name or DEFAULT_MODEL)
            if self.backend != "model":
                # 模型加载失败，记录原因并降级。
                self._load_error = self._load_error or "sentence-transformers / transformers 不可用"

    # ---- 模型加载（可选） ----

    def _try_load_model(self, model_name: str) -> None:
        try:
            # 优先 sentence-transformers：提供原生 encode，最省事。
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self.backend = "model"
            self.model_name = model_name
            if self.dimensions <= 64:
                self.dimensions = self._model.get_sentence_embedding_dimension() or 384
            return
        except Exception as e:  # noqa: BLE001
            self._load_error = f"sentence-transformers: {e}"

        try:
            # 退而求其次：transformers feature-extraction pipeline + mean pooling。
            from transformers import AutoTokenizer, pipeline  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._pipe = pipeline("feature-extraction", model=model_name,
                                  tokenizer=self._tokenizer,
                                  truncation=True, max_length=256)
            self.backend = "model"
            self.model_name = model_name
            return
        except Exception as e:  # noqa: BLE001
            self._load_error = f"transformers: {e}"

        self.backend = "local_hash"

    @property
    def available(self) -> bool:
        return self.backend == "model"

    # ---- 编码 ----

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def encode(self, texts: List[str]) -> List[List[float]]:
        texts = [t or "" for t in texts]
        if not texts:
            return []
        if self.backend == "model":
            try:
                return self._encode_with_model(texts)
            except Exception:  # noqa: BLE001
                # 运行期模型异常：安静降级，保证调用不中断。
                pass
        return [self._local_hash_embedding(t) for t in texts]

    def _encode_with_model(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            vecs = self._model.encode(texts, batch_size=32, show_progress_bar=False,
                                      convert_to_numpy=True)
            return [[float(x) for x in row] for row in vecs]

        # transformers pipeline 分支：mean pooling。
        if self._pipe is not None:
            out = []
            for t in texts:
                spans = self._pipe(t)
                # spans: [seq_len, hidden]; 若批次则 [1, seq_len, hidden]
                if spans and isinstance(spans[0], list):
                    matrix = spans[0] if len(spans) == 1 else spans
                else:
                    matrix = spans
                hidden = len(matrix[0]) if matrix else 0
                if not matrix or hidden == 0:
                    out.append([0.0] * self.dimensions)
                    continue
                pooled = [
                    sum(row[i] for row in matrix) / len(matrix)
                    for i in range(hidden)
                ]
                out.append(self._pad_to_dim(pooled))
            return out

        return [self._local_hash_embedding(t) for t in texts]

    # ---- 本地 hash 降级嵌入（字符 n-gram） ----

    def _local_hash_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        if not text:
            return vec
        # 1~3-gram 字符量作为特征，体会中文/短语的局部共现。
        for size in (1, 2, 3):
            for i in range(len(text) - size + 1):
                gram = text[i:i + size]
                bucket = int(hashlib.md5(gram.encode("utf-8", errors="replace")).hexdigest()[:4], 16) % self.dimensions
                vec[bucket] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            inv = 1.0 / norm
            return [x * inv for x in vec]
        return vec

    def _pad_to_dim(self, row: List[float]) -> List[float]:
        if len(row) > self.dimensions:
            return row[: self.dimensions]
        if len(row) < self.dimensions:
            return row + [0.0] * (self.dimensions - len(row))
        return row

    # ---- 相似度 ----

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        a = a[:n]
        b = b[:n]
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def similarity(self, a: List[float], b: List[float]) -> float:
        return self.cosine(a, b)