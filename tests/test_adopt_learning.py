"""验证「保存这条问答」的落盘逻辑（config.yaml 的 test_scenarios 追加）。

做法是：备份 config.yaml → 用假 self 直接调用 HtmlMainWindow._append_test_scenario
→ 校验 YAML 仍可解析且新样本已写入 → 还原备份。不会留下改动。

用法：
    python tools/test_adopt_learning.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG = ROOT / "config.yaml"
BAK = ROOT / "config.yaml.__adopt_test_bak"


class _FakeWindow:
    """只提供 _append_test_scenario 用到的依赖，避免拉起 Qt。"""

    @staticmethod
    def _yaml_scalar(value) -> str:
        from src.ui.html_window import HtmlMainWindow
        return HtmlMainWindow._yaml_scalar(value)


def main() -> int:
    import yaml
    from src.ui.html_window import HtmlMainWindow

    if not CFG.exists():
        print(f"[!] 找不到 {CFG}")
        return 2

    shutil.copy2(CFG, BAK)
    try:
        before = yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}
        n_before = len(before.get("test_scenarios") or [])

        fake = _FakeWindow()
        ok = HtmlMainWindow._append_test_scenario(
            fake, "能不能开发票", "可以的，开票请联系客服提供抬头和税号。")
        print(f"写入返回      : {ok}")

        text = CFG.read_text(encoding="utf-8")
        after = yaml.safe_load(text) or {}
        scenarios = after.get("test_scenarios") or []
        print(f"样本数 {n_before} -> {len(scenarios)}")

        if not scenarios:
            print("[!] 解析后 test_scenarios 为空，写入失败")
            return 1
        last = scenarios[-1]
        print(f"最后一条样本  : {last}")

        good = (
            len(scenarios) == n_before + 1
            and last.get("approved_reply") == "可以的，开票请联系客服提供抬头和税号。"
            and (last.get("messages") or [None])[0] == "能不能开发票"
        )
        # 再追加一条，验证连续追加不会互相覆盖
        HtmlMainWindow._append_test_scenario(fake, "包邮吗", "全国包邮。")
        after2 = yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}
        s2 = after2.get("test_scenarios") or []
        print(f"二次追加后样本数: {len(s2)}")
        good = good and len(s2) == n_before + 2 and s2[-1].get("approved_reply") == "全国包邮。"

        print()
        print("结论：" + ("正常 —— 问答已真正写入 config.yaml 且格式合法。" if good else "异常"))
        return 0 if good else 1
    finally:
        shutil.copy2(BAK, CFG)
        BAK.unlink(missing_ok=True)
        print("（已还原 config.yaml）")


if __name__ == "__main__":
    sys.exit(main())
