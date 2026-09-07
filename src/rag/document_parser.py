"""文档解析器。

解析 txt / md / json / csv 四种格式，并自动识别 Markdown 标题层级
（h1..h6），把每个分块关联到其所在章节（metadata.hierarchy / title），
以便检索时保留上下文与层级信息。仅依赖标准库。
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import Chunk, TextChunker

_ATX_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*?)\s*$")
_JSON_TEXT_LIMIT = 400


class DocumentParser:
    """根据文件扩展名解析文档为可检索分块。"""

    def __init__(self, chunker: Optional[TextChunker] = None) -> None:
        self.chunker = chunker or TextChunker()

    # ---- 对外 API ----

    def parse_file(self, filepath: str, source: Optional[str] = None) -> List[Chunk]:
        """解析磁盘文件。source 缺省取文件名。"""
        p = Path(filepath)
        text = p.read_text(encoding="utf-8", errors="replace")
        name = source or p.name
        ext = p.suffix.lower().lstrip(".")
        return self.parse(text=text, source=name, filetype=ext)

    def parse(self, text: str, source: str = "", filetype: str = "txt",
              metadata: Optional[dict] = None) -> List[Chunk]:
        """解析文本。filetype ∈ {txt, md, markdown, json, csv}。"""
        text = (text or "").strip()
        if not text:
            return []
        meta_root = dict(metadata or {})
        method = self._dispatcher(filetype)
        return method(text, source, meta_root)

    # ---- 分发 ----

    def _dispatcher(self, filetype: str):
        t = (filetype or "").lower()
        if t in ("md", "markdown"):
            return self._parse_markdown
        if t == "json":
            return self._parse_json
        if t == "csv":
            return self._parse_csv
        return self._parse_txt

    # ---- txt ----

    def _parse_txt(self, text, source, meta_root) -> List[Chunk]:
        return self.chunker.chunk(text, source=source, metadata=meta_root)

    # ---- markdown：识别标题层级 ----

    def _parse_markdown(self, text: str, source: str, meta_root: dict) -> List[Chunk]:
        # 先按标题切分为 (hierarchy, title, body) 章节块。
        sections = self._extract_sections(text)
        chunks: List[Chunk] = []
        for hierarchy, title, body in sections:
            body = body.strip()
            if not body:
                continue
            meta = dict(meta_root)
            meta["title"] = title
            meta["hierarchy"] = hierarchy
            section_text = (f"{title}\n{body}").strip()
            for c in self.chunker.chunk(section_text, source=source, metadata=meta):
                chunks.append(c)
        return chunks

    def _extract_sections(self, text: str) -> List[tuple]:
        """返回 [(hierarchy字符串, 当前标题, 正文)]，标题链用于层级还原。"""
        sections: List[tuple] = []
        # chain[i] 表示第 i 层（0 基）的当前标题文本。
        chain: List[str] = []

        pending_title = ""
        pending_hierarchy = ""
        buf: List[str] = []

        def _flush() -> None:
            nonlocal buf
            body = "\n".join(buf).strip()
            if body:
                sections.append((pending_hierarchy, pending_title, body))
            buf = []

        insert_fence = False  # 简易围栏代码块，避免块内 # 被误判
        for raw in text.splitlines():
            line = raw.rstrip()
            if insert_fence:
                buf.append(line)
                if line.strip().startswith(("```", "~~~")):
                    insert_fence = False
                continue
            if line.strip().startswith(("```", "~~~")):
                insert_fence = True
                buf.append(line)
                continue

            m = _ATX_HEADING.match(line)
            if not m:
                buf.append(line)
                continue

            # 命中标题：先落残块，再维护标题链。
            _flush()
            level = len(m.group(1))  # 1..6
            title = m.group(2).strip()
            if level >= 1 and level <= 6:
                # 弹出更深层标题，设置本层。
                while len(chain) >= level:
                    chain.pop()
                chain.append(title)
            pending_hierarchy = " > ".join(chain)
            pending_title = title

        _flush()

        # 纯标题无正文的节：补一个空正文，避免丢失标题本身。
        if not sections and pending_title and chain:
            sections.append((" > ".join(chain), chain[-1], ""))

        return sections

    # ---- json：抽取文本字段 ----

    def _parse_json(self, text: str, source: str, meta_root: dict) -> List[Chunk]:
        try:
            data = json.loads(text)
        except Exception:
            # 非合法 JSON 时按纯文本兜底。
            return self._parse_txt(text, source, meta_root)

        lines: List[str] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    child_path = f"{path}.{k}" if path else str(k)
                    if isinstance(v, (dict, list)):
                        walk(v, child_path)
                    else:
                        lines.append(f"{child_path}: {v}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]" if path else f"[{i}]")
            else:
                lines.append(f"{path}: {node}")

        walk(data, "")
        content = "\n".join(lines)
        return self.chunker.chunk(content, source=source, metadata=meta_root)

    # ---- csv：每行为一可读行 ----

    def _parse_csv(self, text: str, source: str, meta_root: dict) -> List[Chunk]:
        lines: List[str] = []
        try:
            rows = list(csv.reader(io.StringIO(text)))
        except Exception:
            rows = []
        for row in rows:
            if not row or all(not (c or "").strip() for c in row):
                continue
            lines.append(" | ".join((c or "").strip() for c in row))
        return self.chunker.chunk("\n".join(lines), source=source, metadata=meta_root)