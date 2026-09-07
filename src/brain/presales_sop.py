"""售前 SOP 流程。

识别客户当前处于哪个销售阶段（intro / pricing / objection / closing），
并给出该阶段的对应话术。纯文本规则，不依赖模型。

匹配结果 SopsStageMatch：
  stages      命中的阶段列表（按匹配度降序）
  stage       最佳阶段
  confidence  匹配置信度
  script      最佳阶段对应的直出话术
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .sop_prompts import STAGES, get_stage_script

# ------------------------------------------------------------------ 各阶段关键词
_STAGE_RULES: dict = {
    # 询价 / 报价
    "pricing": (
        "多少钱", "价格", "报价", "费用", "收费", "价钱", "预算", "价位",
        "怎么卖", "咋卖", "多少钱一个月", "多少钱一年", "贵不贵", "贵吗",
        "便宜", "优惠", "折扣", "套餐价", "报价单", "价格表", "费用多少",
    ),
    # 异议 / 犹豫
    "objection": (
        "太贵", "有点贵", "这么贵", "那么贵", "能便宜点吗", "再便宜",
        "考虑一下", "考虑考虑", "再想想", "我看看", "回去商量", "和对象商量",
        "没必要", "不值得", "用不上", "再等等", "过段时间", "先不买",
    ),
    # 促成成交 / 下单
    "closing": (
        "怎么下单", "怎么买", "怎么开通", "怎么购买", "怎么付费", "怎么付款",
        "现在买", "我要买", "直接买", "安排开通", "帮我开通", "办理",
        "怎么签约", "开始用", "马上要", "现在就要",
    ),
    # 开场 / 介绍需求
    "intro": (
        "你好", "您好", "在吗", "在不在", "请问", "咨询", "想了解", "了解一下",
        "怎么用", "是什么", "怎么弄", "介绍", "能做什么", "有什么功能",
    ),
}

_cn_re = re.compile(r"[\w\u4e00-\u9fff]+")


@dataclass
class SopsStageMatch:
    stages: list = field(default_factory=list)   # 命中阶段，按得分降序
    stage: str = ""                              # 最佳阶段
    confidence: float = 0.0
    script: str = ""

    def to_dict(self) -> dict:
        return {
            "stages": self.stages,
            "stage": self.stage,
            "confidence": self.confidence,
            "script": self.script,
        }


class PresalesSOP:
    """售前 SOP：识别销售阶段并给出对应话术。"""

    def __init__(self, config=None, scripts: Optional[dict] = None):
        self._config = config or {}
        # scripts 可用于覆盖默认话术（{stage: template}）
        self._scripts = {s: get_stage_script(s) for s in STAGES}
        if scripts:
            for s, tpl in scripts.items():
                if s in self._scripts and tpl:
                    self._scripts[s] = tpl

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _tokens(text: str) -> List[str]:
        return _cn_re.findall(text or "")

    def match(self, text: str, history: Optional[List[str]] = None) -> SopsStageMatch:
        """识别销售阶段。history 可追加历史消息一起参与，增强识别。"""
        if not text:
            return SopsStageMatch()
        haystack = text
        if history:
            haystack = "\n".join([text] + list(history))

        scored: list = []
        for stage, kws in _STAGE_RULES.items():
            score = sum(1 for k in kws if k in haystack)
            if score > 0:
                scored.append((score, stage))
        scored.sort(reverse=True)

        if not scored:
            return SopsStageMatch()
        best_score, best_stage = scored[0]
        stages = [s for _, s in scored]
        script = self._scripts.get(best_stage, "")
        return SopsStageMatch(stages=stages, stage=best_stage,
                              confidence=best_score / max(_stage_norm(), 1),
                              script=script)

    def process(self, analysis: dict) -> dict:
        """处理 SOP 管线（对齐原版 PresalesSOP.process）。"""
        text = analysis.get("latest_message", "")
        if not text:
            text = analysis.get("raw_content", "")
        match_result = self.match(str(text))
        return {"sop_stage": match_result.stage, "sop_confidence": match_result.confidence}


def _stage_norm() -> int:
    """归一化基数：选取关键词最多的阶段长度，保证 confidence 在 (0,1]。"""
    return max(len(kws) for _, kws in _STAGE_RULES.items())


def grep_stage_hint(text: str) -> List[str]:
    """仅返回命中的阶段名列表（不进类，供轻量调用）。"""
    if not text:
        return []
    return [s for s, kws in _STAGE_RULES.items() if any(k in text for k in kws)]