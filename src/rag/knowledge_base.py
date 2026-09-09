import json
import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


def skip_unreviewed_from_config(cfg) -> bool:
    """卡片审核闭环：默认开启——vault 里 status=待审 的实体卡不进检索，
    在软件「知识库」页点「通过审核」后才生效。"""
    try:
        obs = (cfg or {}).get("obsidian") or {}
        return bool(obs.get("review_cards", True))
    except Exception:
        return True


def extra_roots_from_config(cfg) -> List[str]:
    """按配置算出附加知识源：Obsidian vault 的「知识」「待补充」目录。

    对话目录默认不进检索——「嗯」「好的」这类闲聊会把真正有用的内容挤掉，
    只有用户在设置里打开「对话也参与检索」才加进来。
    """
    try:
        obs = (cfg or {}).get("obsidian") or {}
        if not obs.get("enabled"):
            return []
        vault = str(obs.get("vault_path") or "").strip()
        if not vault:
            return []
        base = Path(vault) / str(obs.get("root_folder") or "VisReply")
        roots: List[str] = []
        for key, default in (("knowledge_folder", "知识"),
                             ("pending_folder", "待补充")):
            p = base / str(obs.get(key) or default)
            if p.exists():
                roots.append(str(p))
        if obs.get("index_dialogue"):
            p = base / str(obs.get("dialogue_folder") or "对话")
            if p.exists():
                roots.append(str(p))
        return roots
    except Exception:
        return []


def rag_flags_from_config(cfg) -> dict:
    """UltraRAG 三件套开关（无需向量/embedding，纯 LLM + 现有 bigram 词法）。

    - query_rewrite：客户口语→书面/同义改写 + 多 query 扩展（命中低分时触发，平时零额外延迟）
    - rerank：LLM 精排 top 候选（候选多于 top_k 且非高置信时触发）
    抄 UltraRAG 思路但落地为「不配向量」版本：复用现有 bigram 词法 + aliases + gaps。
    """
    try:
        rag = (cfg or {}).get("rag") or {}
        return {
            "enable_rewrite": bool(rag.get("query_rewrite", True)),
            "rewrite_variants": int(rag.get("query_rewrite_variants", 3) or 3),
            "rewrite_trigger": float(rag.get("query_rewrite_trigger", 0.18) or 0.18),
            "enable_rerank": bool(rag.get("rerank", True)),
            "rerank_trigger": float(rag.get("rerank_trigger", 0.6) or 0.6),
        }
    except Exception:
        return {"enable_rewrite": True, "rewrite_variants": 3, "rewrite_trigger": 0.18,
                "enable_rerank": True, "rerank_trigger": 0.6}


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
                 min_score: float = 0.10,
                 extra_roots: Optional[List[str]] = None,
                 index_state_path: str = "data/knowledge_index.json",
                 gap_score: float = 0.15,
                 gap_repeat: int = 3,
                 gap_log_path: str = "data/knowledge_gaps.jsonl",
                 skip_unreviewed: bool = False,
                 llm_call=None,
                 enable_rewrite: bool = True,
                 rewrite_variants: int = 3,
                 rewrite_trigger: float = 0.18,
                 enable_rerank: bool = True,
                 rerank_trigger: float = 0.6,
                 rerank_top_n: int = 10):
        self.root_path = Path(root_path)
        self.skip_unreviewed = bool(skip_unreviewed)
        self.chunk_chars = chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.top_k = top_k
        self.min_score = min_score
        self._chunks: List[KnowledgeChunk] = []
        self._index_built = False
        self._hash_to_idx: dict = {}

        # --- 多源挂载：data/knowledge + Obsidian vault 的知识目录 ---
        # extra_roots 元素为目录路径字符串，扫描时把路径记成相对该目录的形式
        self.extra_roots: List[Path] = [Path(p) for p in (extra_roots or [])]

        # --- 增量索引：{绝对路径: sha256} + {绝对路径: [chunk,...]} ---
        # 只在文件内容变化时重新解析，vault 上千篇时避免每次全量重扫
        self._file_sig: dict = {}
        self._chunks_by_file: dict = {}
        self.index_state_path = index_state_path

        # --- 知识缺口（gaps）：反复查不到的问题自动记账 ---
        # 借鉴 mdvault 的 `gaps` 命令，用于自动往「待补充」目录沉淀
        self.gap_score = gap_score        # 最高分低于此值视作"没查到"
        self.gap_repeat = gap_repeat      # 同一问题累计这么多次才落盘
        self.gap_log_path = gap_log_path
        self._gap_hits: dict = {}         # query -> 次数（进程内累计）

        # --- UltraRAG 三件套（不配向量）：LLM 改写/多query + LLM 精排 ---
        # llm_call: 由调用方注入的 LLM 回调 (system_prompt, user_prompt) -> dict，
        # 内部用 call_text_json。保持 KB 不反向依赖 text_model_client，避免循环 import。
        self.llm_call = llm_call
        self.enable_rewrite = bool(enable_rewrite)
        self.rewrite_variants = max(1, int(rewrite_variants) or 3)
        self.rewrite_trigger = float(rewrite_trigger) if rewrite_trigger is not None else 0.18
        self.enable_rerank = bool(enable_rerank)
        self.rerank_trigger = float(rerank_trigger) if rerank_trigger is not None else 0.6
        self.rerank_top_n = max(self.top_k, int(rerank_top_n) or 10)

    # ---------------- 多源 / 增量索引 ----------------

    def set_extra_roots(self, roots: List[str]) -> None:
        """替换附加知识源（Obsidian vault 的知识目录等），并让下次加载走增量。"""
        self.extra_roots = [Path(p) for p in (roots or [])]

    def _all_roots(self) -> List[Path]:
        roots = [self.root_path]
        roots.extend(self.extra_roots)
        return roots

    def _scan_files(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        files: List[Path] = []
        for ext in ["*.txt", "*.md", "*.json", "*.csv"]:
            files.extend(root.rglob(ext))
        skip_names = {
            "readme.md", "readme.txt", "readme",
            "说明.txt", "说明.md", "使用说明.txt", "使用说明.md",
        }
        return [f for f in files if f.name.lower() not in skip_names]

    def _rebuild_flat(self) -> None:
        """把 _chunks_by_file 展开成 _chunks 并重建哈希索引。"""
        flat: List[KnowledgeChunk] = []
        for key in sorted(self._chunks_by_file):
            flat.extend(self._chunks_by_file[key])
        self._chunks = flat
        self._hash_to_idx = {c.hash_id: i for i, c in enumerate(self._chunks)}
        self._index_built = True

    def load_documents(self, directory: Optional[str] = None,
                       force: bool = False) -> int:
        """增量加载所有知识源。

        - 默认扫描 root_path + extra_roots（Obsidian vault 的知识目录）
        - 先比 mtime+size，再比 sha256，只有内容真变了才重新切块
        - 先读完整字节再替换旧 chunk，避免在 Obsidian 保存中途读到半截文件
        """
        if self.root_path and not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)
        self._load_index_state()

        roots = [Path(directory)] if directory else self._all_roots()
        seen = set()
        changed = 0

        for root in roots:
            root = Path(root)
            for filepath in self._scan_files(root):
                key = str(filepath.resolve())
                seen.add(key)
                try:
                    st = filepath.stat()
                    cached = self._file_sig.get(key)
                    # 第一级：mtime + size 都没变 → 直接跳过（不读文件）
                    if (not force and cached
                            and cached.get("mtime") == st.st_mtime
                            and cached.get("size") == st.st_size):
                        continue
                    raw = filepath.read_bytes()
                    sig = hashlib.sha256(raw).hexdigest()
                    # 第二级：内容 hash 没变（只是被 touch 了）→ 刷新指纹即可
                    if not force and cached and cached.get("sig") == sig:
                        cached["mtime"] = st.st_mtime
                        cached["size"] = st.st_size
                        continue
                    text = raw.decode("utf-8", errors="replace")
                    self._chunks_by_file[key] = self._load_file(filepath, root)
                    self._file_sig[key] = {
                        "sig": sig, "mtime": st.st_mtime, "size": st.st_size
                    }
                    changed += 1
                except Exception as e:
                    print(f"[RAG] Failed to load {filepath}: {e}")

        # 磁盘上已删除的文件：同步清掉索引，否则检索会命中幽灵内容
        for key in list(self._file_sig):
            if key not in seen:
                self._file_sig.pop(key, None)
                self._chunks_by_file.pop(key, None)
                changed += 1

        self._rebuild_flat()
        self._save_index_state()
        print(f"[RAG] 索引 {len(self._chunks)} chunks / "
              f"{len(self._file_sig)} 文件（本次变动 {changed}）")
        return len(self._chunks)

    # ---------------- 索引状态持久化（跨进程复用 mtime 指纹） ----------------

    def _load_index_state(self) -> None:
        """载回文件指纹**和切好的 chunk**。

        只恢复指纹不恢复 chunk 会出事：第二次启动所有文件都判定"没变"直接跳过，
        结果 _chunks_by_file 是空的，检索全挂。
        """
        p = Path(self.index_state_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            self._file_sig = data.get("files") or {}
            raw_chunks = data.get("chunks") or {}
            restored = {}
            for key, items in raw_chunks.items():
                if key not in self._file_sig:
                    continue  # 文件已不在索引里，别把幽灵 chunk 带回来
                restored[key] = [
                    KnowledgeChunk(
                        text=c.get("text", ""),
                        source=c.get("source", ""),
                        offset=int(c.get("offset", 0)),
                        hash_id=c.get("hash_id", ""),
                    ) for c in items
                ]
            if restored:
                self._chunks_by_file = restored
        except Exception:
            self._file_sig = {}
            self._chunks_by_file = {}

    def _save_index_state(self) -> None:
        try:
            p = Path(self.index_state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "files": self._file_sig,
                "chunks": {
                    key: [{"text": c.text, "source": c.source,
                           "offset": c.offset, "hash_id": c.hash_id}
                          for c in chunks]
                    for key, chunks in self._chunks_by_file.items()
                },
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def invalidate(self) -> None:
        """强制下次全量重建（配置变更 / 用户点了重新索引）。"""
        self._file_sig = {}
        self._chunks_by_file = {}
        self._index_built = False
        try:
            Path(self.index_state_path).unlink(missing_ok=True)
        except Exception:
            pass

    # ---------------- markdown 预处理 ----------------

    @staticmethod
    def _parse_frontmatter(text: str) -> Tuple[dict, str]:
        """拆出 YAML frontmatter（只要 title/tags/aliases），失败则原样返回。"""
        if not text.startswith("---"):
            return {}, text
        end = text.find("\n---", 3)
        if end < 0:
            return {}, text
        block = text[3:end].strip()
        body = text[end + 4:].lstrip("\n")
        meta: dict = {}
        try:
            import yaml
            parsed = yaml.safe_load(block)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            # YAML 不合法就退化为手工抓 aliases/tags，别让一篇坏笔记拖垮整库
            for key in ("aliases", "tags"):
                m = re.search(rf'^{key}\s*:\s*(.+)$', block, re.M)
                if m:
                    meta[key] = [x.strip() for x in
                                 re.split(r'[,，]', m.group(1).strip("[] "))]
            m = re.search(r'^title\s*:\s*(.+)$', block, re.M)
            if m:
                meta["title"] = m.group(1).strip()
        return meta, body

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """去掉 md 语法噪音，避免 [[双链]]/嵌入/Dataview 污染 chunk 与回复。"""
        text = re.sub(r'<!--.*?-->', '', text, flags=re.S)          # HTML 注释
        text = re.sub(r'```+\s*dataview[\s\S]*?```+', '', text)      # Dataview 块
        text = re.sub(r'^\s*%%[\s\S]*?%%\s*$', '', text, flags=re.M)  # Obsidian 注释
        text = re.sub(r'!\[\[[^\]]*\]\]', '', text)                   # ![[附件]] 嵌入
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)              # 外链图片
        text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)    # [[A|B]] → B
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)               # [[A]] → A
        text = re.sub(r'^```+\w*\s*$', '', text, flags=re.M)          # 代码围栏
        return text

    def _load_file(self, filepath: Path,
                   base: Optional[Path] = None) -> List[KnowledgeChunk]:
        """解析单个文件：frontmatter → 清洗 → 相对路径 source → 带前缀切块。"""
        raw = filepath.read_bytes() if filepath.exists() else b""
        text = raw.decode("utf-8", errors="replace") if raw else \
            filepath.read_text(encoding="utf-8", errors="replace")

        meta, body = self._parse_frontmatter(text)
        body = self._clean_markdown(body)

        # 卡片审核闭环（P0）：待审/忽略的实体卡不进检索，审核通过才生效
        if self.skip_unreviewed and \
                str(meta.get("status") or "").strip() in ("待审", "忽略", "ignored"):
            return []

        # source 用相对路径：vault 里同名笔记一大堆，只存文件名溯源不到
        try:
            rel = filepath.resolve().relative_to(
                (base or self.root_path).resolve())
            source = str(rel).replace("\\", "/")
        except Exception:
            source = filepath.name

        title = str(meta.get("title") or filepath.stem)
        aliases = meta.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        # 同义词扩召回：纯词法检索不认"多少钱"="产品价格"，
        # 把 aliases 拼进 chunk 文本，等于给检索加一层人工同义词表
        alias_line = ("同义词: " + "、".join(aliases) + "\n") if aliases else ""
        tag_line = ("标签: " + "、".join(str(t) for t in tags) + "\n") if tags else ""

        return self._chunk_text(
            body, source,
            rel_path=source, title=title,
            alias_line=alias_line, tag_line=tag_line,
        )

    def _chunk_text(self, text: str, source: str,
                    rel_path: str = "", title: str = "",
                    alias_line: str = "", tag_line: str = "") -> List[KnowledgeChunk]:
        """切块，并给每块拼上上下文前缀（Anthropic contextual retrieval 思路）：
        [相对路径 > 笔记标题 > 小节标题] + 同义词/标签行 + 正文。
        标题与别名进索引后，"价格" 能命中标题带"产品价格"的笔记。"""
        use_prefix = bool(rel_path or title)

        def compose(heading: str, body: str) -> str:
            if not use_prefix:
                return body
            parts = [rel_path or source]
            if title:
                parts.append(title)
            if heading and heading != title:
                parts.append(heading)
            head = "[" + " > ".join(p for p in parts if p) + "]"
            return f"{head}\n{alias_line}{tag_line}{body}".rstrip()

        chunks: List[KnowledgeChunk] = []
        sections = self._split_markdown_sections(text)
        current_chunk = ""
        current_heading = ""
        offset = 0

        def flush(body: str, headings: List[str], off: int) -> int:
            if not body.strip():
                return off
            # 一块里跨了多个小节就不标小节名，避免张冠李戴
            heading = headings[0] if len(headings) == 1 else ""
            if len(body) <= self.chunk_chars:
                chunks.append(KnowledgeChunk(
                    text=compose(heading, body), source=source, offset=off))
                return off + len(body)
            for sub, sub_off in self._iter_long_text(body, off):
                chunks.append(KnowledgeChunk(
                    text=compose(heading, sub), source=source, offset=sub_off))
            return off + len(body)

        current_headings: List[str] = []
        for heading, para in sections:
            if len(current_chunk) + len(para) <= self.chunk_chars:
                current_chunk = (current_chunk + "\n" + para).strip()
                if heading:
                    current_headings.append(heading)
            else:
                offset = flush(current_chunk, current_headings, offset)
                current_chunk = para
                current_headings = [heading] if heading else []
        flush(current_chunk, current_headings, offset)
        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
        """按空行分段，并记住每段最近的 markdown 标题，用于生成 chunk 前缀。
        纯 txt 也能用（标题为空，行为与旧版一致）。"""
        out: List[Tuple[str, str]] = []
        heading = ""
        buf: List[str] = []

        def emit():
            if buf:
                para = "\n".join(buf).strip()
                if para:
                    out.append((heading, para))
                buf.clear()

        for line in text.splitlines():
            m = re.match(r'^(#{1,6})\s+(.*)$', line)
            if m:
                emit()
                heading = m.group(2).strip()
                buf.append(heading)  # 标题文字本身也要留在正文里，不能只当前缀
                continue
            if not line.strip():
                emit()
                continue
            buf.append(line)
        emit()
        return out

    def _iter_long_text(self, text: str, base_offset: int):
        """把超长段落按句切成多块，yield (块文本, 起始偏移)。"""
        sentences = re.split(r'([。！？.!?])', text)
        current = ""
        offset = base_offset
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            i += 2
            if len(current) + len(sentence) <= self.chunk_chars:
                current += sentence
            else:
                if current:
                    yield current, offset
                    offset += len(current)
                current = sentence
        if current:
            yield current, offset

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

        q = (query or "").strip()

        # ① 词法基线：对所有 chunk 打分（现有 bigram 词法，不依赖向量）
        scored: dict = {}
        for i, chunk in enumerate(self._chunks):
            scored[i] = self._lexical_similarity(q, chunk.text)
        best = max(scored.values()) if scored else 0.0

        # ② UltraRAG 改写 + 多 query 扩展：仅当词法最高分低于触发阈值才调 LLM，
        #    命中良好的查询零额外延迟；改写覆盖口语/同义，补漏"答非所问"。
        if self.llm_call and self.enable_rewrite and best < self.rewrite_trigger:
            for vq in self._llm_rewrite_query(q):
                for i, chunk in enumerate(self._chunks):
                    s = self._lexical_similarity(vq, chunk.text)
                    if s > scored.get(i, 0.0):
                        scored[i] = s
            best = max(scored.values())

        # 取达阈值的候选并按分排序
        results = [(self._chunks[i], s) for i, s in scored.items() if s >= self.min_score]
        results.sort(key=lambda x: x[1], reverse=True)

        # ③ UltraRAG LLM 精排：候选多于 top_k 且非高置信时，让 LLM 重排挑最相关
        if self.llm_call and self.enable_rerank and best < self.rerank_trigger \
                and len(results) > self.top_k:
            reranked = self._llm_rerank(q, results[:self.rerank_top_n])
            if reranked:
                results = reranked

        # 知识缺口记账（CRAG 自愈 / 转人工 的数据源）：最高分仍够不着 = 知识库没这条
        best = results[0][1] if results else 0.0
        if best < self.gap_score:
            self._record_gap(q, best)

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

    def _llm_rewrite_query(self, query: str) -> List[str]:
        """UltraRAG 改写：把口语/方言问题改写成多个书面/同义问法。

        返回变体列表（含原问题，去重），失败时返回空列表 → 退回纯词法检索。
        """
        try:
            sys_p = (
                "你是客服知识库检索增强器。用户会用口语、方言或简写提问，但知识库文档是书面语。"
                "请把用户问题改写为若干更可能命中知识库书面文档的问法，覆盖用户真实意图，"
                "包含同义词、近义词、不同表述。只输出 JSON："
                "{\"queries\": [\"原始问题\", \"改写1\", \"改写2\"]}，最多 %d 个。"
                % self.rewrite_variants
            )
            usr_p = "用户问题：%s" % query
            out = self.llm_call(sys_p, usr_p) or {}
            variants = [str(x).strip() for x in (out.get("queries") or []) if str(x).strip()]
            seen = set()
            out_list = []
            for v in ([query] + variants):
                if v and v not in seen:
                    seen.add(v)
                    out_list.append(v)
            return out_list[: max(1, self.rewrite_variants + 1)]
        except Exception:
            return []

    def _llm_rerank(self, query: str,
                    candidates: List[Tuple[KnowledgeChunk, float]]
                    ) -> List[Tuple[KnowledgeChunk, float]]:
        """UltraRAG 精排：LLM 按相关性重排 top 候选，返回至多 top_k 个。

        失败时返回空列表 → 退回词法排序结果。
        """
        try:
            items = []
            for i, (chunk, _) in enumerate(candidates):
                snippet = chunk.text[:200].replace("\n", " ")
                items.append("[%d] (%s) %s" % (i, chunk.source, snippet))
            sys_p = (
                "你是检索结果精排器。给定用户问题和若干候选知识片段，"
                "按与用户问题的相关性从高到低排序，返回最相关的至多 %d 个片段编号。"
                "只输出 JSON：{\"ranked\": [2, 0, 5]}，数字为候选下标（从0开始）。"
                % self.top_k
            )
            usr_p = "用户问题：%s\n\n候选片段：\n%s" % (query, "\n".join(items))
            out = self.llm_call(sys_p, usr_p) or {}
            raw = out.get("ranked") or []
            order = []
            for x in raw:
                try:
                    idx = int(x)
                except Exception:
                    continue
                if 0 <= idx < len(candidates) and idx not in order:
                    order.append(idx)
            if not order:
                return []
            result = [candidates[i] for i in order]
            # LLM 返回不足 top_k 时，用剩余候选补齐（保持原顺序）
            if len(result) < self.top_k:
                rest = [c for i, c in enumerate(candidates) if i not in order]
                result.extend(rest[: self.top_k - len(result)])
            return result
        except Exception:
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

    # ---------------- 知识缺口（gaps） ----------------

    def _record_gap(self, query: str, score: float) -> bool:
        """累计"查不到"的问题；每满 gap_repeat 次写一行，避免每条都落盘。"""
        q = (query or "").strip()
        if not q:
            return False
        self._gap_hits[q] = self._gap_hits.get(q, 0) + 1
        n = self._gap_hits[q]
        if n < self.gap_repeat or n % self.gap_repeat:
            return False
        try:
            p = Path(self.gap_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "q": q, "score": round(float(score), 3),
                    "hits": n, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return True

    def get_gaps(self, top_n: int = 20) -> List[dict]:
        """累计的知识缺口，按出现次数降序（供写入 vault「待补充」）。"""
        p = Path(self.gap_log_path)
        if not p.exists():
            return []
        agg: dict = {}
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = str(d.get("q") or "").strip()
                if not q:
                    continue
                cur = agg.setdefault(q, {"q": q, "hits": 0, "last": "", "score": 1.0})
                cur["hits"] = max(cur["hits"], int(d.get("hits") or 0))
                cur["last"] = max(cur["last"], str(d.get("ts") or ""))
                cur["score"] = min(cur["score"], float(d.get("score") or 0))
        except Exception:
            return []
        return sorted(agg.values(), key=lambda x: -x["hits"])[:top_n]

    def clear_gaps(self) -> None:
        """已沉淀到 vault 后清空缺口记录。"""
        try:
            Path(self.gap_log_path).unlink(missing_ok=True)
        except Exception:
            pass
        self._gap_hits = {}

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
