"""已回复消息去重（修复 #2）的单元/集成测试。

不依赖微信窗口，纯逻辑验证 RepliedDedup 的指纹、命中、TTL、持久化与淘汰。
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.reply_dedup import RepliedDedup


def _tmp_path():
    fd, p = tempfile.mkstemp(suffix=".json", prefix="replied_dedup_test_")
    os.close(fd)
    os.remove(p)
    return p


def test_fingerprint_stable_and_case_insensitive_contact():
    a = RepliedDedup.fingerprint("  Alice ", "在的吗")
    b = RepliedDedup.fingerprint("alice", "在的吗")
    assert a == b, "联系人应大小写/空白不敏感"
    c = RepliedDedup.fingerprint("alice", "在的")
    assert a != c, "消息内容不同应不同指纹"


def test_mark_then_skip():
    p = _tmp_path()
    d = RepliedDedup(p, ttl=100)
    assert not d.should_skip("alice", "你好")
    d.mark_replied("alice", "你好")
    assert d.should_skip("alice", "你好")
    # 不同消息不应被误伤
    assert not d.should_skip("alice", "再见")
    assert not d.should_skip("bob", "你好")


def test_ttl_expiry():
    p = _tmp_path()
    d = RepliedDedup(p, ttl=0.3)
    d.mark_replied("alice", "你好")
    assert d.should_skip("alice", "你好")
    time.sleep(0.4)
    assert not d.should_skip("alice", "你好"), "超过 TTL 应重新允许回复"


def test_persistence_across_instances():
    p = _tmp_path()
    d1 = RepliedDedup(p, ttl=100)
    d1.mark_replied("alice", "你好")
    d1.save()
    # 模拟重启：新实例从磁盘加载
    d2 = RepliedDedup(p, ttl=100)
    assert d2.should_skip("alice", "你好"), "重启后应仍能识别已回复消息"


def test_empty_inputs_never_skip():
    d = RepliedDedup(_tmp_path(), ttl=100)
    assert not d.should_skip("", "你好")
    assert not d.should_skip("alice", "")
    d.mark_replied("", "x")  # 不应写入
    assert not d.should_skip("", "x")


def test_max_entries_eviction():
    p = _tmp_path()
    d = RepliedDedup(p, ttl=10_000, max_entries=3)
    for i in range(10):
        d.mark_replied(f"c{i}", f"m{i}")
    # 超过上限后应淘汰最旧的，总数收敛到上限
    assert len(d._data) <= 3
    # 最新写入的仍在
    assert d.should_skip("c9", "m9")


if __name__ == "__main__":
    test_fingerprint_stable_and_case_insensitive_contact()
    test_mark_then_skip()
    test_ttl_expiry()
    test_persistence_across_instances()
    test_empty_inputs_never_skip()
    test_max_entries_eviction()
    print("ALL_REPLY_DEDUP_TESTS_PASSED")
