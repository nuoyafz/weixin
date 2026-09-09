#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 VisReply 在线更新清单与增量包。

用法:
  python tools/gen_update_manifest.py \
      --url https://你的服务器/visreply/update \
      --notes "修复若干已知问题，优化回复速度" \
      --setup installer_output/VisReply_Setup.exe

输出 update_dist/：version.json + increment_<ver>.zip（+ 可选整包）。
上传整个 update_dist/ 到服务器静态目录即可。

增量包只包含相对上次构建「sha256 变化的文件」，并刻意排除：
  data/、logs/、venv/、config.yaml、*.db、*.log 等用户数据与密钥。
因此用户更新后其本地配置 / 知识库 / 聊天数据不会被覆盖。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DEFAULT = ROOT / "dist_onedir8" / "WeChatAIAssistant"
OUT_DEFAULT = ROOT / "update_dist"
STATE_PATH = ROOT / "_build_tmp" / "update_last.json"

# 用户数据 / 密钥 / 临时 绝不进增量包
EXCLUDE_DIRS = {"data", "logs", "venv", "_build_tmp", "installer_output",
                "__pycache__", ".git", ".workbuddy", "build", "build8"}
EXCLUDE_NAMES = {"config.yaml", "agent.db", ".gitignore"}
EXCLUDE_SUFFIX = {".log"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def get_version() -> str:
    try:
        t = (ROOT / "src" / "__init__.py").read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', t)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "0.0.0"


def iter_build_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.name in EXCLUDE_NAMES or rel.name.endswith(".db"):
            continue
        if rel.suffix in EXCLUDE_SUFFIX:
            continue
        yield rel, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(BUILD_DEFAULT))
    ap.add_argument("--url", required=True, help="服务器静态基址（如 https://域名/visreply/update）")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true", help="强制用户更新（忽略增量，走整包）")
    ap.add_argument("--full-only", action="store_true", dest="full_only",
                    help="只生成整包兜底，不生成增量包（适合首次发布或简单维护）")
    ap.add_argument("--setup", default="", help="整包兜底用：VisReply_Setup.exe 路径")
    ap.add_argument("--state", default=str(STATE_PATH))
    ap.add_argument("--prev-version", default="")
    ap.add_argument("--baseline-build", default="",
                   help="增量基线构建目录（如 dist_onedir8_164/WeChatAIAssistant）。指定后用其文件哈希作 diff 基线，不依赖 update_last.json。")
    args = ap.parse_args()

    build_root = Path(args.build)
    if not build_root.is_dir():
        print("[错误] 构建目录不存在:", build_root)
        print("       请先打包，或 --build 指定 dist 目录。")
        sys.exit(1)
    base = args.url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)

    version = get_version()
    print("[gen] 新版本号:", version)

    current = {}
    for rel, p in iter_build_files(build_root):
        current[str(rel).replace("\\", "/")] = {
            "sha256": sha256_of(p), "size": p.stat().st_size}
    print("[gen] 当前构建文件数:", len(current))

    prev, prev_version = {}, args.prev_version
    if state_path.exists():
        try:
            s = json.loads(state_path.read_text(encoding="utf-8"))
            prev = s.get("files", {})
            if not prev_version:
                prev_version = s.get("version", "")
        except Exception as e:
            print("[gen][警告] 读取上次状态失败:", e)
    if not prev_version:
        prev_version = "0.0.0"

    # 指定基线构建目录：用其文件哈希作 diff 基线（不依赖 update_last.json 历史）
    if args.baseline_build:
        bb = Path(args.baseline_build)
        if bb.is_dir():
            prev = {}
            for rel, p in iter_build_files(bb):
                prev[str(rel).replace("\\", "/")] = {"sha256": sha256_of(p), "size": p.stat().st_size}
            if not args.prev_version:
                ip = bb / "src" / "__init__.py"
                if not ip.exists():
                    ip = bb / "_internal" / "src" / "__init__.py"
                if ip.exists():
                    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                                  ip.read_text(encoding="utf-8", errors="ignore"))
                    if m:
                        prev_version = m.group(1)
            print(f"[gen] 使用基线构建: {bb} (上一版本 {prev_version})")
        else:
            print(f"[gen][警告] --baseline-build 不存在: {bb}")

    is_first = (len(prev) == 0)  # 首次发布：无任何历史基线
    print("[gen] 上一版本号:", prev_version)
    if is_first:
        print("[gen][信息] 首次发布 → 无 diff 基线，仅生成整包兜底（不生成伪增量包）")

    changed = [k for k, v in current.items()
               if k not in prev or prev[k].get("sha256") != v["sha256"]]
    # 首次发布没有可对比的基线，全量增量 == 整包，毫无意义 → 强制不走增量；
    # --full-only 则任何时候都只出整包，适合不想维护增量的简单场景。
    if is_first or args.full_only:
        changed = []
        if not args.full_only:
            print("[gen][信息] 已跳过增量包（首次发布只出整包兜底）")
    print("[gen] 计入增量的变化文件数:", len(changed))

    # 首次发布必须提供整包，否则用户既无增量也无整包，无法更新
    if is_first and not args.setup:
        print("[错误] 首次发布必须提供 --setup 整包，否则用户无法更新。")
        print("       请加: --setup installer_output/VisReply_Setup.exe")
        sys.exit(1)

    notes = args.notes
    if notes.startswith("@"):
        nf = Path(notes[1:])
        notes = nf.read_text(encoding="utf-8") if nf.exists() else ""
    if not notes:
        notes = "本次更新包含功能优化与问题修复。"

    zip_name = f"increment_{version}.zip"
    zip_path = out / zip_name
    files_manifest = []
    if changed:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in changed:
                arc = rel.replace("\\", "/")
                z.write(build_root / rel, arc)
                files_manifest.append({"path": arc,
                                       "sha256": current[rel]["sha256"],
                                       "size": current[rel]["size"]})
        print(f"[gen] 增量包: {zip_path} ({len(files_manifest)} 文件)")
    else:
        print("[gen][警告] 无变化文件，本次不生成增量包（增量列表为空）")

    full_package = ""
    if args.setup:
        setup = Path(args.setup)
        if setup.exists():
            # 保留文件名中的版本号（如 VisReply_Setup_1.6.3.exe），便于区分
            out_name = setup.name
            shutil.copy(setup, out / out_name)
            full_package = f"{base}/{out_name}"
            print(f"[gen] 整包兜底: {out / out_name}")

    manifest = {
        "version": version,
        "previous_version": prev_version,
        "release_notes": notes,
        "force": bool(args.force),
        "published_at": date.today().isoformat(),
        "increment_url": f"{base}/{zip_name}" if changed else "",
        "full_package": full_package,
        "files": files_manifest,
    }
    (out / "version.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] 清单: {out / 'version.json'}")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"version": version, "files": current}, ensure_ascii=False),
        encoding="utf-8")
    print(f"[gen] 状态已更新: {state_path}")

    iss = ROOT / "installer.iss"
    if iss.exists():
        t = iss.read_text(encoding="utf-8")
        nt = re.sub(r'(#define MyAppVersion ")[^"]+(")', r'\g<1>%s\g<2>' % version, t)
        if nt != t:
            iss.write_text(nt, encoding="utf-8")
            print(f"[gen] installer.iss MyAppVersion -> {version}")

    print(f"\n[gen] 完成。把整个 {out} 目录上传到 {base}/ 即可。")


if __name__ == "__main__":
    main()
