"""FAQ 回复匹配：在调用模型前，先匹配知识库中已保存的常见问答。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List

from .text_intents import detect_text_intents


@dataclass
class FaqReplyMatch:
    matched: bool = False
    question: str = ""
    answer: str = ""
    confidence: float = 0.0
    intents: list = field(default_factory=list)


class FaqReplyMatcher:
    """从知识库 `02_standard_qa.md` 加载问答。文件格式示例：

        ## Q：你们价格多少？
        A：这是 199 一个月。
        ...
    """

    def __init__(self, config: Any = None, path: str = None):
        self.config = config
        self.path = Path(path) if path else self._default_path()
        self._items: List[dict] = []
        self._load_items()

    def _default_path(self) -> Path:
        # 候选根目录：配置指定的知识库根 > 默认 data/knowledge
        candidates = []
        if isinstance(self.config, dict):
            k = self.config.get("knowledge", {}).get("root") or self.config.get("knowledge_root")
            if k:
                candidates.append(Path(str(k)))
        candidates.append(Path("data/knowledge"))
        # 优先用真实存在的问答文件（已有的 09_标准问答FAQ.md，含真实 demo 业务数据）；
        # 找不到再回退到旧约定 text/02_standard_qa.md
        for base in candidates:
            base = base if base.is_absolute() else Path.cwd() / base
            for name in ("09_标准问答FAQ.md", "text/02_standard_qa.md"):
                p = base / name
                if p.exists():
                    return p
        base = candidates[0] if candidates else Path("data/knowledge")
        base = base if base.is_absolute() else Path.cwd() / base
        return base / "09_标准问答FAQ.md"

    def reload(self) -> None:
        self._items = []
        self._load_items()

    def _load_items(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        try:
            content = self.path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            self._items = []
            return
        self._items = self._parse(content)

    @staticmethod
    def _parse(content: str) -> List[dict]:
        items = []
        # 匹配 "Q：..." / "A：..." 成对，或 "问：..." / "答：..."
        q_re = re.compile(r"^\s*#{0,3}\s*(?:Q|问|问题)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
        a_re = re.compile(r"^\s*#{0,3}\s*(?:A|答|回答)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
        qs = list(q_re.finditer(content))
        for i, qm in enumerate(qs):
            q = qm.group(1).strip()
            # 找到其后最近的 A
            next_q = qs[i + 1].start() if i + 1 < len(qs) else len(content)
            segment = content[qm.end():next_q]
            am = a_re.search(segment)
            if am:
                answer = "\n".join(l.strip() for l in am.group(1).splitlines()).strip()
            else:
                # 无 A 标签：取段内非空行首行前几行作为答
                lines = [l.strip() for l in segment.splitlines() if l.strip() and not l.startswith("Q")]
                answer = "\n".join(lines[:3]).strip()
            if q and answer:
                items.append({"question": q, "answer": answer})
        return items

    @staticmethod
    def _bigrams(s: str) -> set:
        s = re.sub(r"\s+", "", s)
        return set(s[i:i + 2] for i in range(len(s) - 1))

    @classmethod
    def _lexical(cls, a: str, b: str) -> float:
        """中文二元文法召回率：a 的二元组被 b 覆盖的比例（对短句更鲁棒）。"""
        a = re.sub(r"\s+", "", a)
        b = re.sub(r"\s+", "", b)
        if len(a) < 2:
            return 0.0
        ba, bb = cls._bigrams(a), cls._bigrams(b)
        if not ba:
            return 0.0
        return len(ba & bb) / len(ba)

    def match(self, text: str, threshold: float = 0.30) -> FaqReplyMatch:
        """模糊匹配常见问答。

        以「问题」为标准用户表述（中文二元文法召回率）为主，辅以对「答案」的折扣相似度。
        弃用 SequenceMatcher 整串比例——它对中文短句会把「你们靠谱吗」与「你们是机器人吗」
        因共享「你们/吗」误判为高相似。问题文案越接近标准表述得分越高，因此建议在
        09_标准问答FAQ.md 里为高频问题补充常见问法变体，让确定性分支覆盖更全。
        """
        if not text:
            return FaqReplyMatch()
        best = None
        best_score = 0.0
        for item in self._items:
            q_lex = self._lexical(text, item["question"])
            a_lex = self._lexical(text, item["answer"])
            score = max(q_lex, 0.5 * a_lex)
            if score > best_score:
                best_score = score
                best = item
        if best and best_score >= threshold:
            intents = detect_text_intents(text) or detect_text_intents(best["question"])
            return FaqReplyMatch(matched=True, question=best["question"],
                                 answer=best["answer"], confidence=round(best_score, 3),
                                 intents=intents)
        return FaqReplyMatch(intents=detect_text_intents(text))