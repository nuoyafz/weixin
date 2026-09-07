#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model_check 的离线单元测试（mock 网络，不依赖真实 API）。

运行:
    python tests/test_model_check.py
"""
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reply.model_check import ModelChecker


class _FakeResp:
    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _make_404():
    fp = io.BytesIO(b'{"error":{"message":"model not found"}}')
    return urllib.error.HTTPError(
        url="https://x/chat/completions", code=404,
        msg="Not Found", hdrs=None, fp=fp)


class ModelCheckerTest(unittest.TestCase):
    def test_detect_ok(self):
        with patch("urllib.request.urlopen",
                   return_value=_FakeResp({"choices": [{"message": {"content": "pong"}}]})):
            r = ModelChecker("https://x", "k", "m").detect()
        self.assertTrue(r["ok"])
        self.assertEqual(r["message"], "连接成功，模型可调用")
        self.assertGreaterEqual(r["latency_ms"], 0)

    def test_detect_missing_config(self):
        r = ModelChecker("", "", "").detect()
        self.assertFalse(r["ok"])
        self.assertEqual(r["error_type"], "config")

    def test_speed_check_multiple_rounds(self):
        payload = {"choices": [{"message": {"content": "hi"}}]}
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            sp = ModelChecker("https://x", "k", "m").speed_check(rounds=3)
        self.assertTrue(sp["ok"])
        self.assertEqual(sp["success_rounds"], 3)
        self.assertEqual(len(sp["latencies"]), 3)
        self.assertEqual(sp["min_ms"], min(sp["latencies"]))
        self.assertEqual(sp["max_ms"], max(sp["latencies"]))
        self.assertIn("平均", sp["message"])

    def test_speed_check_all_fail(self):
        with patch("urllib.request.urlopen", side_effect=_make_404()):
            sp = ModelChecker("https://x", "k", "m").speed_check(rounds=2)
        self.assertFalse(sp["ok"])
        self.assertEqual(sp["success_rounds"], 0)
        self.assertIn("失败", sp["message"])

    def test_detect_http_404_model_not_found(self):
        with patch("urllib.request.urlopen", side_effect=_make_404()):
            r = ModelChecker("https://x", "k", "m").detect()
        self.assertFalse(r["ok"])
        self.assertIn("模型名不存在", r["message"])


if __name__ == "__main__":
    unittest.main()
