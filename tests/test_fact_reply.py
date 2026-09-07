"""离线验证：事实敏感分流（FAQ 确定性直出）+ 发送前合规门。

不依赖 LLM / 微信，纯本地运行。
"""
from __future__ import annotations

from src.brain.faq_reply import FaqReplyMatcher
from src.agent.compliance_gate import ComplianceGate


def test_faq_loads_real_data():
    m = FaqReplyMatcher()
    assert len(m._items) > 0, "FAQ 文件未加载（路径或解析失败）"
    print(f"[ok] FAQ 加载 {len(m._items)} 条")


def test_fact_queries_hit_deterministic():
    m = FaqReplyMatcher()
    fact = ["多少钱", "收费多少", "怎么退款", "退款", "客服微信多少",
            "你们是做什么的", "多久能生效", "怎么付款", "有没有试用"]
    for q in fact:
        r = m.match(q)
        assert r.matched, f"事实敏感查询未命中确定性答案: {q!r}"
        assert r.answer, f"命中但答案为空: {q!r}"
    print(f"[ok] 事实敏感查询全部走确定性直出（跳过 LLM）")


def test_open_queries_fall_through_to_llm():
    m = FaqReplyMatcher()
    open_q = ["太贵了", "在吗", "哈哈哈", "你们靠谱吗"]
    for q in open_q:
        r = m.match(q)
        assert not r.matched, f"开放查询不应命中 FAQ（应走 LLM）: {q!r}"
    print(f"[ok] 开放/异议查询正确回退到 LLM 路径")


def test_compliance_allows_canonical():
    g = ComplianceGate()
    ok, reason = g.check("标准版 ¥99/月，专业版 ¥299/月")
    assert ok, f"规范价被误拦截: {reason}"
    ok, _ = g.check("体验版免费，年付 8 折")
    assert ok, "无货币数字的表述不应触发合规门"
    ok, _ = g.check("我们服务过 1000+ 客户")
    assert ok, "非价格数字不应误杀"
    print("[ok] 合规门放行规范价 / 非价格表述")


def test_compliance_blocks_wrong_price():
    g = ComplianceGate()
    ok, reason = g.check("我们收费 199 元每月")
    assert not ok, "错误价格未被拦截"
    ok, _ = g.check("只要 50 元就能用")
    assert not ok, "错误价格未被拦截"
    print(f"[ok] 合规门拦截错误价格: {reason}")


if __name__ == "__main__":
    test_faq_loads_real_data()
    test_fact_queries_hit_deterministic()
    test_open_queries_fall_through_to_llm()
    test_compliance_allows_canonical()
    test_compliance_blocks_wrong_price()
    print("\nALL PASS")
