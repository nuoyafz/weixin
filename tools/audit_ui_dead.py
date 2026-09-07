"""UI 死按钮 / 失效功能静态审计。

用法：
    python tools/audit_ui_dead.py

只读审计，不修改任何文件。输出三类问题：
  1) 死按钮：HTML 里存在 id 的按钮，但 JS 从未绑定事件（点了没反应）。
  2) 悬空引用：JS 里 getElementById(...) 的 id 在 HTML 中不存在（绑定无效）。
  3) 未接桥：JS 调用的 bridge 方法在 Python 侧没有 @Slot 实现。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "resources" / "html" / "main.html"
PY = ROOT / "src" / "ui" / "html_window.py"


def main() -> int:
    if not HTML.exists():
        print(f"[!] 找不到 {HTML}")
        return 2
    html = HTML.read_text(encoding="utf-8", errors="replace")
    py = PY.read_text(encoding="utf-8", errors="replace") if PY.exists() else ""
    # html_window.py 会通过 runJavaScript 注入补充 JS（如 __bindWindowBtn），
    # 其中的 getElementById 绑定也算数，必须一起参与比对，否则误报死按钮。
    js_all = html + "\n" + py

    # --- 1) HTML 中声明的 id ---
    declared_ids = set(re.findall(r'\bid="([A-Za-z0-9_\-]+)"', html))

    # --- 2) JS 中引用到的 id ---
    used_ids = set()
    # 注入 JS 里的引号可能被 Python 转义（\"），也可能用单引号，两种都要匹配。
    for pat in (
        r'getElementById\(\s*[\\"\']+([A-Za-z0-9_\-]+)[\\"\']+\s*\)',
        r'querySelector\(\s*[\\"\']+#([A-Za-z0-9_\-]+)[\\"\']+\s*\)',
        r'\$\(\s*[\\"\']+#([A-Za-z0-9_\-]+)[\\"\']+\s*\)',
        # 页面内定义了 `const $ = id => document.getElementById(id)` 简写
        r'\$\(\s*[\\"\']+([A-Za-z0-9_\-]+)[\\"\']+\s*\)',
    ):
        used_ids |= set(re.findall(pat, js_all))

    # 辅助函数形式的绑定（如 `function on(id, fn){ getElementById(id).addEventListener... }`
    # + `on("someId", ...)`）会让正则直接搜 getElementById("字面量") 落空而误报。
    # 这里识别这类包装函数，再把它的调用点还原成 id 引用。
    for helper, param in re.findall(
            r'function\s+([A-Za-z_$][\w$]*)\s*\(\s*([A-Za-z_$][\w$]*)\s*,\s*[A-Za-z_$][\w$]*\s*\)\s*\{'
            r'[^}]{0,200}?getElementById\(\s*\2\s*\)', js_all, re.S):
        used_ids |= set(re.findall(
            r'\b%s\(\s*["\']([A-Za-z0-9_\-]+)["\']' % re.escape(helper), js_all))
        used_ids |= set(re.findall(
            r'\b%s\(\s*\\?"([A-Za-z0-9_\-]+)\\?"' % re.escape(helper), js_all))

    # 通用委托绑定（如 querySelectorAll("[data-close]")）会让一批按钮活起来，
    # 这里把被选择器命中的 class / data 属性也统计出来，避免误报。
    generic_selectors = set(re.findall(r'querySelectorAll\(\s*"([^"]+)"\s*\)', js_all))

    dangling = sorted(used_ids - declared_ids)

    # --- 3) 死按钮：button.[id] 从未被引用 ---
    button_ids = set()
    for tag in re.findall(r'<button[^>]*>', html):
        m = re.search(r'\bid="([A-Za-z0-9_\-]+)"', tag)
        if m:
            button_ids.add(m.group(1))
    dead = sorted(button_ids - used_ids)

    # 带 data-close / data-dismiss / data-dialog-close 的按钮由通用委托接管
    closable = set()
    for tag in re.findall(r'<button[^>]*>', html):
        m = re.search(r'\bid="([A-Za-z0-9_\-]+)"', tag)
        if m and re.search(r'data-(close|dismiss|dialog-close)\b', tag):
            closable.add(m.group(1))
    dead = [d for d in dead if d not in closable]

    # --- 4) bridge 方法：JS 调用 vs Python @Slot ---
    js_calls = set(re.findall(r'__bridgeCall\(\s*"([A-Za-z0-9_]+)"', html))
    js_calls |= set(re.findall(r'wepulseBridge\.([A-Za-z0-9_]+)', html))
    js_calls = {c for c in js_calls if not c[0].isdigit()}
    py_slots = set(re.findall(r'\n    def ([A-Za-z0-9_]+)\(', py))
    missing_slots = sorted(c for c in js_calls if c not in py_slots)

    print("=" * 60)
    print("UI 失效功能审计报告")
    print("=" * 60)
    print(f"HTML 按钮总数(带 id): {len(button_ids)}")
    print(f"HTML 声明 id 总数   : {len(declared_ids)}")
    print(f"JS 引用 id 总数     : {len(used_ids)}")
    print()

    print("--- [1] 死按钮（有 id 但 JS 从未绑定，点击无反应）---")
    if dead:
        for b in dead:
            print(f"  x  #{b}")
    else:
        print("  （无）")
    print()

    print("--- [2] 悬空引用（JS 找的 id 在 HTML 里不存在）---")
    if dangling:
        for b in dangling:
            print(f"  x  #{b}")
    else:
        print("  （无）")
    print()

    print("--- [3] 未接桥（JS 调用了但 Python 没有 @Slot 实现）---")
    if missing_slots:
        for b in missing_slots:
            print(f"  x  {b}")
    else:
        print("  （无）")
    print()
    print(f"--- [4] 通用委托选择器（{len(generic_selectors)} 条）---")
    for s in sorted(generic_selectors):
        print(f"  . {s}")
    print()

    # --- 5) 假功能：点击处理器只有一句 showToast，什么都不做 ---
    toast_only = re.findall(
        r'getElementById\("([A-Za-z0-9_\-]+)"\)\?*\.addEventListener\(\s*"click"\s*,\s*'
        r'(?:\(\)\s*=>|function\s*\(\s*\))\s*\{\s*showToast\((.*?)\);\s*\}\s*\)',
        js_all, re.S)
    print("--- [5] 假功能（点击只弹一句提示，无任何实际操作）---")
    if toast_only:
        for eid, msg in toast_only:
            print(f"  x  #{eid}  ->  {msg.strip()[:70]}")
    else:
        print("  （无）")
    print()

    # --- 6) 假保存：关闭弹窗后直接报「已保存」，没有落盘 ---
    fake_save = re.findall(
        r'getElementById\("([A-Za-z0-9_\-]+)"\)\.addEventListener\(\s*"click"\s*,\s*'
        r'(?:\(\)\s*=>|function\s*\(\s*\))\s*\{\s*'
        r'(close\w+Dialog\(\);?\s*showToast\("[^"]*已保存[^"]*"\);?)\s*\}\s*\)',
        js_all, re.S)
    print("--- [6] 假保存（关弹窗 + 提示「已保存」，实际未写入）---")
    if fake_save:
        for eid, body in fake_save:
            print(f"  x  #{eid}  ->  {' '.join(body.split())[:80]}")
    else:
        print("  （无）")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
