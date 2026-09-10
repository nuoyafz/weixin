"""VisReply 在线更新核心（混合模式：增量优先 + 整包兜底）。

- 本地版本取自 src.__version__（与 installer.iss 的 MyAppVersion 对齐）。
- 服务端只需静态托管 version.json + increment_<ver>.zip（+ 可选整包）。
- 流程：check_for_update() 比对版本 → start_update() 下载 / sha256 校验 /
  解压到临时目录 → 写 update.bat（等主进程退出后 xcopy 覆盖 + 重启 exe）
  → 退出主进程。
- 所有依赖均为 Python 标准库，无需新增第三方包。
"""
import os
import sys
import json
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

# 占位地址；发布前请改为你自己的服务器（或填到 config.yaml 的 app.update_server）。
UPDATE_SERVER = "https://update.example.com/visreply/update"

# CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS：让更新器进程脱离父进程独立运行，
# 父进程（主程序）退出后仍能完成文件替换并拉起新进程。
_DETACHED = 0x00000200 | 0x00000800


def get_local_version() -> str:
    try:
        from src import __version__
        return str(__version__)
    except Exception:
        return "0.0.0"


def _find_config_path():
    """兼容多种运行形态定位 config.yaml：
    - onedir (PyInstaller 6.22+): exe 旁 _internal/config.yaml
    - onefile: sys._MEIPASS/config.yaml
    - 源码开发: src/updater/__init__.py 上两层 = 项目根
    - cwd
    """
    candidates = []
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "config.yaml")
        candidates.append(exe_dir / "_internal" / "config.yaml")
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "config.yaml")
    except Exception:
        pass
    candidates.append(Path.cwd() / "config.yaml")
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "config.yaml")
    seen = set()
    for c in candidates:
        try:
            c = Path(c)
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            if c.exists():
                return c
        except Exception:
            pass
    return None


def _resolve_server() -> str:
    env = os.environ.get("VISREPLY_UPDATE_SERVER")
    if env:
        return env.rstrip("/")
    try:
        import yaml  # 可选：读 config.yaml 的 app.update_server
        p = _find_config_path()
        if p and p.exists():
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s = (d.get("app") or {}).get("update_server")
            if s:
                return str(s).rstrip("/")
    except Exception:
        pass
    return UPDATE_SERVER.rstrip("/")


def _parse_version(v):
    v = (v or "0.0.0").strip().lstrip("vV")
    out = []
    for p in v.split("."):
        try:
            out.append(int(p))
        except Exception:
            out.append(0)
    return tuple(out)


def version_gt(a, b) -> bool:
    return _parse_version(a) > _parse_version(b)


def version_eq(a, b) -> bool:
    return _parse_version(a) == _parse_version(b)


def _http_get_json(url, timeout=15):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "VisReply-Updater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_download(url, dest, timeout=600, progress=None):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "VisReply-Updater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length", "0") or "0")
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)


def _push(win, js):
    try:
        if win is not None and hasattr(win, "evaluate_js"):
            win.evaluate_js(js)
    except Exception:
        pass


def check_for_update():
    """向服务器查询是否有新版本。返回 dict（见各分支）。"""
    local = get_local_version()
    try:
        manifest = _http_get_json(_resolve_server() + "/version.json")
    except Exception as e:
        return {"ok": False, "message": "无法连接更新服务器：%s" % e}
    remote = manifest.get("version", "")
    if not version_gt(remote, local):
        return {"ok": True, "update_available": False, "current": local, "latest": remote}
    files = manifest.get("files") or []
    full = manifest.get("full_package")
    prev_ver = manifest.get("previous_version", "")
    # 增量仅当本地版本正好等于「上一版本」时才安全；跨版本直接套增量会缺中间新增文件 → 走整包
    forced = bool(manifest.get("force", False))
    can_increment = (not forced) and bool(files) and version_eq(local, prev_ver)
    method = "increment" if can_increment else ("full" if full else "none")
    size = sum(int(f.get("size", 0)) for f in files) if files else 0
    return {
        "ok": True, "update_available": True, "current": local,
        "version": remote, "previous_version": manifest.get("previous_version"),
        "release_notes": manifest.get("release_notes", ""), "size": size,
        "force": bool(manifest.get("force", False)), "method": method,
        "full_package": full,
    }


def start_update(win=None, manifest=None):
    """执行更新。win 为 pywebview window 对象（用于推送进度）。"""
    if manifest is None:
        manifest = _http_get_json(_resolve_server() + "/version.json")
    base = _resolve_server()
    files = manifest.get("files") or []
    full = manifest.get("full_package")
    force = bool(manifest.get("force", False))
    prev_ver = manifest.get("previous_version", "")

    # 仅当本地版本正好等于「上一版本」且非强制时走增量；否则（无增量/强制/跨版本）走整包兜底
    if (not files) or force or not version_eq(get_local_version(), prev_ver):
        if not full:
            raise RuntimeError("服务器未提供完整安装包链接，无法更新")
        _do_full_update(win, full)
        return

    ver = manifest.get("version", "latest")
    tmp = Path(tempfile.gettempdir()) / ("visreply_update_%s" % ver)
    staged = tmp / "staged"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / ("increment_%s.zip" % ver)
    zip_url = manifest.get("increment_url") or (base + "/increment_%s.zip" % ver)
    try:
        def prog(done, total):
            pct = int(done / total * 90) if total else 5
            _push(win, "window.__onUpdateProgress(%d);" % pct)
        _http_download(zip_url, str(zip_path), timeout=600, progress=prog)
        _push(win, "window.__onUpdateProgress(92);")
        _verify_and_extract(zip_path, staged, files)
    except Exception:
        if full:
            _push(win, "window.__onUpdateProgress(0);")
            _do_full_update(win, full)
            return
        raise

    _push(win, "window.__onUpdateProgress(96);")
    app_dir = Path(sys.executable).resolve().parent
    exe = Path(sys.executable).resolve()
    bat = tmp / "update.bat"
    bat.write_text(_make_increment_bat(staged, app_dir, exe, ver), encoding="utf-8")
    _push(win, "window.__onUpdateProgress(98);")
    subprocess.Popen(["cmd.exe", "/c", str(bat)], creationflags=_DETACHED, close_fds=True)
    _push(win, "window.__onUpdateRestarting();")
    os._exit(0)


def _verify_and_extract(zip_path, staged, files):
    """逐文件校验 sha256 后解压到 staged；任何不一致立即失败。"""
    expected = {f["path"]: f for f in files if f.get("path")}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            info = expected.get(name)
            if info is None:
                raise RuntimeError("增量包含未声明文件：%s" % name)
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != info.get("sha256"):
                raise RuntimeError("文件校验失败（sha256 不匹配）：%s" % name)
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for name in expected:
        if not (staged / name).exists():
            raise RuntimeError("增量包缺失文件：%s" % name)


def _make_increment_bat(staged, app_dir, exe, ver):
    """生成增量覆盖 bat。
    【2026-09-08 加固】解决「下载/解压成功但 exe 未替换 → 重启仍是旧版、仍提示更新」：
      1) taskkill 强制结束残留进程，释放被锁的 exe/pyd 句柄；
      2) xcopy 覆盖失败时自动重试 3 次（每次间隔等待，避开瞬时锁/杀软扫描）；
      3) fc /B 二进制比对确认 WeChatAIAssistant.exe 真被替换；
      4) 仍失败则写 _update_failed.txt，下次启动由 maybe_fallback_full_update() 自动回退整包。
    全程 echo 留痕到 %TEMP%\\visreply_update.log，失败时可直接定位断点。
    """
    s, a, e = staged.resolve(), app_dir.resolve(), exe.resolve()
    log = Path(tempfile.gettempdir()) / "visreply_update.log"
    return (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        f'set LOG="{log}"\r\n'
        f'set STAGED="{s}"\r\n'
        f'set APP="{a}"\r\n'
        f'set EXE="{e}"\r\n'
        f'set VER="{ver}"\r\n'
        'echo [%date% %time%] increment bat start >> %LOG%\r\n'
        "taskkill /F /IM WeChatAIAssistant.exe >nul 2>&1\r\n"
        "ping -n 3 127.0.0.1 >nul\r\n"
        "set TRIES=0\r\n"
        ":RETRY\r\n"
        "set /a TRIES+=1\r\n"
        'echo [%time%] xcopy try %TRIES% >> %LOG%\r\n'
        'xcopy /Y /E /C /I "%STAGED%\\*" "%APP%" >> %LOG% 2>&1\r\n'
        'echo [%time%] xcopy exit %errorlevel% >> %LOG%\r\n'
        'fc /B "%STAGED%\\WeChatAIAssistant.exe" "%APP%\\WeChatAIAssistant.exe" >nul 2>&1\r\n'
        "if %errorlevel%==0 (\r\n"
        '  echo [%time%] verify OK >> %LOG%\r\n'
        '  echo %VER% > "%APP%\\_just_updated.txt"\r\n'
        '  del "%APP%\\_update_failed.txt" >nul 2>&1\r\n'
        '  echo [%time%] restart >> %LOG%\r\n'
        '  start "" "%EXE%"\r\n'
        "  goto :EOF\r\n"
        ")\r\n"
        "if %TRIES% LSS 3 (\r\n"
        '  echo [%time%] retry after 2s >> %LOG%\r\n'
        "  ping -n 3 127.0.0.1 >nul\r\n"
        "  goto RETRY\r\n"
        ")\r\n"
        'echo [%time%] increment FAILED >> %LOG%\r\n'
        'echo %VER% > "%APP%\\_update_failed.txt"\r\n'
        'start "" "%EXE%"\r\n'
    )


def _do_full_update(win, full_url):
    tmp = Path(tempfile.gettempdir()) / "visreply_update_full"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    setup = tmp / "VisReply_Setup.exe"
    def prog(done, total):
        pct = int(done / total * 90) if total else 5
        _push(win, "window.__onUpdateProgress(%d);" % pct)
    _http_download(full_url, str(setup), timeout=900, progress=prog)
    _push(win, "window.__onUpdateProgress(95);")
    app_dir = Path(sys.executable).resolve().parent
    exe = Path(sys.executable).resolve()
    bat = tmp / "update_full.bat"
    bat.write_text(_make_full_bat(setup, app_dir, exe), encoding="utf-8")
    _push(win, "window.__onUpdateRestarting();")
    subprocess.Popen(["cmd.exe", "/c", str(bat)], creationflags=_DETACHED, close_fds=True)
    os._exit(0)


def _make_full_bat(setup, app_dir, exe):
    """整包安装 bat。
    【2026-09-08 修复】「整包更新后版本号不变、仍提示更新」根因：
      旧逻辑 `setup /VERYSILENT /NORESTART` 且无 CloseApplications。只要 {app} 内任意
      文件被占用（WebView2 渲染进程映射了 _internal 的 DLL、杀软瞬时锁、第二实例残留），
      Inno 无法替换 → 把「替换旧 exe」排成「重启后待执行」(pending rename)；/NORESTART
      又不会重启 → 磁盘 exe 仍是旧的；bat 末尾 `start "" "%EXE%"` 又把那个旧 exe 拉起来
      → 用户一直跑旧版、一直提示更新。
    修复：
      1) 装前 taskkill 结束残留进程，释放被锁句柄；
      2) 加 /CLOSEAPPLICATIONS（配合 installer.iss 的 CloseApplications=yes），由
         RestartManager 强制关闭占用 {app} 文件的进程后再复制；
      3) 判安装退出码：0=成功直接拉起新 exe；3010=需重启，写 _reboot_required.txt 且
         不拉起旧 exe（等用户重启后自然是新版）；其它非0=失败，不拉起。
    """
    a, e = app_dir.resolve(), exe.resolve()
    log = Path(tempfile.gettempdir()) / "visreply_update.log"
    return ("@echo off\r\n"
            "chcp 65001 >nul\r\n"
            f'echo [%date% %time%] full-setup bat start > "{log}"\r\n'
            "taskkill /F /IM WeChatAIAssistant.exe >nul 2>&1\r\n"
            "ping -n 4 127.0.0.1 >nul\r\n"
            f'"{setup.resolve()}" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS >> "{log}" 2>&1\r\n'
            f'echo [%time%] setup exit %errorlevel% >> "{log}"\r\n'
            "if %errorlevel%==3010 (\r\n"
            f'  echo [%time%] REBOOT_REQUIRED >> "{log}"\r\n'
            f'  echo 1 > "{a}\\_reboot_required.txt"\r\n'
            "  goto :EOF\r\n"
            ")\r\n"
            "if not %errorlevel%==0 (\r\n"
            f'  echo [%time%] setup FAILED exit %errorlevel% >> "{log}"\r\n'
            "  goto :EOF\r\n"
            ")\r\n"
            f'echo [%time%] launch new exe >> "{log}"\r\n'
            f'del "{a}\\_just_updated.txt" "{a}\\_update_failed.txt" >nul 2>&1\r\n'
            f'start "" "{e}"\r\n')


def maybe_fallback_full_update():
    """启动自检（在窗口 loaded 事件中调用）：
    若检测到「上次增量落地失败(_update_failed.txt)」或「_just_updated 记录的版本
    与当前实际版本不符」，说明增量更新的 exe 没被真正替换，自动回退整包更新。
    返回 True 表示已触发整包更新（程序将退出并由安装包覆盖重启）。
    """
    try:
        app_dir = Path(sys.executable).resolve().parent
        local = get_local_version()
        failed = app_dir / "_update_failed.txt"
        just = app_dir / "_just_updated.txt"
        need = False
        if failed.exists():
            need = True
        elif just.exists():
            try:
                rec = just.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                rec = ""
            if rec and rec != local:
                need = True
        if not need:
            return False
        # 清理标记，避免重复触发（整包会彻底覆盖并重写 _just_updated.txt）
        for f in (failed, just):
            try:
                if f.exists():
                    f.unlink()
                    print(f"[updater] 已清理残留标记 {f.name}，避免重复触发整包更新")
            except Exception as _e:  # noqa: BLE001
                print(f"[updater] 清理标记失败 {f.name}: {_e}")
        manifest = _http_get_json(_resolve_server() + "/version.json")
        full = manifest.get("full_package")
        if full:
            _do_full_update(win=None, full_url=full)
            return True
    except Exception as e:  # noqa: BLE001
        print("[updater] 自检整包回退异常:", e)
    return False
