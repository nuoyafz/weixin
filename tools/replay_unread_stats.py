"""批量回放历史截图，统计红点识别效果（强化前后对比用）。

用法：
    python tools/replay_unread_stats.py [screenshots_dir]

只读审计。输出：
  - 每张截图 contact_dots / nav_badge 摘要
  - 命中数字徽章（unread>1）的样本（验证多策略 OCR）
  - 汇总：检出红点帧数、可点击帧数
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    from src.rpa.red_dot_detector import RedDotDetector

    shots = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "screenshots")
    files = sorted(shots.glob("*.png"))
    if not files:
        print(f"[!] {shots} 下没有截图")
        return 2

    d = RedDotDetector()
    total_frames = 0
    frames_with_dots = 0
    frames_clickable = 0
    digit_badges = 0

    print(f"共 {len(files)} 张截图，开始回放…\n")
    for f in files:
        try:
            img = np.array(Image.open(f).convert("RGB"))[:, :, ::-1].copy()
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        if img.shape[1] < 400:  # 只回放微信窗口级截图
            continue
        total_frames += 1
        cds = d._scan_contact_dots(img)
        nb = d._scan_nav_badge(img)
        clickable = [c for c in cds if d._dot_clickable(c)]
        if cds:
            frames_with_dots += 1
        if clickable:
            frames_clickable += 1
        # 摘要打印（只打印有红点或 nav_badge 的帧）
        if cds or nb:
            nums = [c["unread_count"] for c in cds if c["unread_count"] is not None]
            digit_badges += len(nums)
            print(f"  {f.name} size={img.shape[1]}x{img.shape[0]}"
                  f" dots={len(cds)} clickable={len(clickable)}"
                  f" nums={nums} nav={nb is not None}"
                  + (f" navunread={nb.get('unread_count')}" if nb else "")
                  + (f" | dots: " + "; ".join(
                      f"({c['center_x']},{c['center_y']})a{c['area']}u{c['unread_count']}"
                      for c in cds[:4]) if cds else ""))

    print("\n==== 汇总 ====")
    print(f"有效帧        : {total_frames}")
    print(f"检出红点帧    : {frames_with_dots} ({100*frames_with_dots//max(total_frames,1)}%)")
    print(f"可点击帧      : {frames_clickable} ({100*frames_clickable//max(total_frames,1)}%)")
    print(f"读到数字徽章  : {digit_badges} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
