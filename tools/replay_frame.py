"""离线回放：对真机截图跑红点检测 + OCR，输出结构化 json 与标注图。

用途
----
真机 RPA 的 bug 大多无法在无头环境复现，但**图像检测逻辑**（红点、OCR、坐标）
完全可以对已有截图离线回放。本脚本把「看截图猜」换成「读数字」：
改检测逻辑后秒级自测，不需要启动微信，也不消耗一次真机取证机会。

用法
----
  python tools/replay_frame.py data/wechat/latest.png
  python tools/replay_frame.py "data/wechat/wait_conv_20260901_*.png" --ocr
  python tools/replay_frame.py data/wechat/latest.png --annotate --json out.json

参数
----
  --ocr        额外跑 OCR（首次会加载模型，较慢；默认关闭）
  --annotate   生成标注图（红点绿框+序号，导航徽章橙框，OCR 蓝框）
  --json PATH  把结果写成 json（默认只打印到终端）

说明
----
检测参数直接读 config.yaml 并传给 RedDotDetector，与生产路径
（ObserveService 里 RedDotDetector(self.config, ...)）保持一致，
避免出现「离线跑得过、真机跑不过」的参数漂移。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402
import yaml  # noqa: E402

from src.rpa.red_dot_detector import RedDotDetector  # noqa: E402


def _load_config() -> dict:
    path = os.path.join(_ROOT, "config.yaml")
    if not os.path.exists(path):
        print("[warn] 未找到 config.yaml，使用空配置（检测参数走代码默认值）")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print("[warn] config.yaml 读取失败: %s" % e)
        return {}


def _rect_of(d: dict):
    """从检测结果里取矩形，兼容 x/y/w/h 与 center_x/center_y 两种字段风格。"""
    if not isinstance(d, dict):
        return None
    if "x" in d and "w" in d:
        return int(d.get("x", 0)), int(d.get("y", 0)), int(d.get("w", 0)), int(d.get("h", 0))
    cx = d.get("center_x", d.get("cx"))
    cy = d.get("center_y", d.get("cy"))
    if cx is None or cy is None:
        return None
    r = int(d.get("radius", d.get("r", 10)))
    return int(cx) - r, int(cy) - r, 2 * r, 2 * r


def _annotate(img, nav_badge, dots, ocr_items, out_path: str) -> None:
    out = img.copy()
    for i, d in enumerate(dots or []):
        rect = _rect_of(d)
        if not rect:
            continue
        x, y, w, h = rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(out, "#%d" % i, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if isinstance(nav_badge, dict):
        rect = _rect_of(nav_badge)
        if rect:
            x, y, w, h = rect
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(out, "NAV", (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    for it in ocr_items or []:
        box = it.get("box")
        if box is None:
            continue
        try:
            pts = box.reshape(-1, 2).astype(int)
            cv2.polylines(out, [pts], True, (255, 120, 0), 1)
        except Exception:
            continue
    cv2.imwrite(out_path, out)


def _run_ocr(img):
    from src.local_vision.ocr_engine import OCREngine
    res = OCREngine().run(img)
    items = []
    for it in res.items:
        items.append({
            "text": it.text,
            "score": round(float(it.score), 3),
            "cx": round(float(it.cx), 1),
            "cy": round(float(it.cy), 1),
            "box": it.box.tolist() if hasattr(it.box, "tolist") else None,
        })
    return items


def _run_view(img_path: str) -> dict:
    """完整复现生产感知链路：WechatScreenReader().analyze()，输出视图判定证据。

    与 observe_service._last_frame_looks_like_list / 正式 OCR 流程用的是同一个
    reader 与同一条 analyze 路径，用于离线复现「视图误判」「latest_message 取错
    文本」等问题。
    """
    from src.clean_perception.reader import WechatScreenReader
    frame = cv2.imread(img_path)
    if frame is None:
        return {"path": img_path, "error": "imread 失败"}
    a = WechatScreenReader().analyze(frame)
    out = {
        "path": img_path,
        "view": a.view,
        "view_confidence": round(float(a.view_confidence), 3),
        "intent": a.intent,
        "current_contact": a.current_contact,
        "is_self_latest": a.is_self_latest,
        "anchors": None,
        "messages": [],
    }
    if a.anchors is not None:
        out["anchors"] = {
            "chat_left": a.anchors.chat_left,
            "chat_right": a.anchors.chat_right,
            "message_top": a.anchors.message_top,
            "message_bottom": a.anchors.message_bottom,
            "header_bottom": a.anchors.header_bottom,
            "input_top": a.anchors.input_top,
            "source": a.anchors.source,
        }
    for m in a.messages[:20]:
        out["messages"].append({
            "text": m.text,
            "side": m.side,
            "x_min": round(float(m.x_min), 1),
            "x_max": round(float(m.x_max), 1),
            "y_center": round(float(m.y_center), 1),
            "conf": round(float(m.confidence), 2),
        })
    return out


def replay(path: str, cfg: dict, do_ocr: bool, do_annotate: bool) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"path": path, "error": "cv2.imread 失败（路径含中文或文件损坏）"}
    h, w = img.shape[:2]

    logs = []
    det = RedDotDetector(cfg, logger=lambda *a, **k: logs.append(str(a)))
    try:
        nav_badge = det._scan_nav_badge(img)
    except Exception as e:
        nav_badge = None
        logs.append("nav_badge 异常: %s" % e)
    try:
        dots = det._scan_contact_dots(img)
    except Exception as e:
        dots = []
        logs.append("contact_dots 异常: %s" % e)

    ocr_items = []
    if do_ocr:
        try:
            ocr_items = _run_ocr(img)
        except Exception as e:
            logs.append("OCR 异常: %s" % e)

    if do_annotate:
        stem, ext = os.path.splitext(path)
        _annotate(img, nav_badge, dots, ocr_items, "%s_annotated%s" % (stem, ext or ".png"))

    return {
        "path": path,
        "size": {"w": int(w), "h": int(h)},
        "nav_badge": nav_badge,
        "contact_dots": dots,
        "contact_dot_count": len(dots or []),
        "ocr_count": len(ocr_items),
        "ocr": ocr_items,
        "detector_logs": logs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="离线回放真机截图：红点检测 + OCR")
    ap.add_argument("images", nargs="+", help="图片路径，支持通配符")
    ap.add_argument("--ocr", action="store_true", help="额外跑 OCR（较慢）")
    ap.add_argument("--annotate", action="store_true", help="生成标注图")
    ap.add_argument("--json", dest="json_path", default="", help="结果写入 json 文件")
    ap.add_argument("--view", action="store_true",
                    help="跑完整感知链路（视图判定+消息提取），复现生产 clean_reader 路径")
    args = ap.parse_args()

    paths = []
    for pat in args.images:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else [pat])

    cfg = _load_config()
    if args.view:
        results = [_run_view(p) for p in paths]
    else:
        results = [replay(p, cfg, args.ocr, args.annotate) for p in paths]

    for r in results:
        print("=" * 60)
        if r.get("error"):
            print("%s\n  ERROR: %s" % (r["path"], r["error"]))
            continue
        if "view" in r and "error" not in r:
            print("%s" % r["path"])
            print("  视图        : %s (conf=%.3f)" % (r.get("view"), r.get("view_confidence", 0.0)))
            print("  意图        : %s" % r.get("intent"))
            print("  联系人      : %r" % r.get("current_contact"))
            print("  自己最新    : %s" % r.get("is_self_latest"))
            anc = r.get("anchors") or {}
            print("  锚点        : chat_left=%s chat_right=%s msg_top=%s msg_bottom=%s src=%s" % (
                anc.get("chat_left"), anc.get("chat_right"), anc.get("message_top"),
                anc.get("message_bottom"), anc.get("source")))
            for m in r.get("messages") or []:
                print("    [%-6s] x=%-7.1f y=%-7.1f %s" % (m["side"], m["x_min"], m["y_center"], m["text"]))
            continue
        nav = r.get("nav_badge")
        print("%s" % r["path"])
        print("  尺寸        : %dx%d" % (r["size"]["w"], r["size"]["h"]))
        print("  导航徽章    : %s" % ("有" if nav else "无"))
        print("  联系人红点  : %d 个" % r["contact_dot_count"])
        for i, d in enumerate(r.get("contact_dots") or []):
            rect = _rect_of(d)
            print("    #%d %s" % (i, rect if rect else d))
        if args.ocr:
            print("  OCR 文本行  : %d" % r["ocr_count"])
            for it in r.get("ocr") or []:
                print("    (%.2f) x=%-7.1f y=%-7.1f %s" % (
                    it["score"], it.get("cx", 0.0), it["cy"], it["text"]))
        for line in r.get("detector_logs") or []:
            print("  [detector] %s" % line)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print("\n已写入 %s" % args.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
