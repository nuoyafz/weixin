"""
Launcher — 支持 GUI 模式与屏外挂机模式（无界面后台自动回复）。

用法:
  python run.py                 # GUI 模式（原行为，零改动）
  python run.py --gui          # 同上
  python run.py --offscreen    # 屏外挂机：微信全程屏外(x=-10000)，直接起自动回复循环，不弹窗口
  python run.py --once         # 只跑一个周期后退出（验证/定时任务用）
  python run.py --offscreen --dry-run   # 挂机但只模拟决策、不真发（首次验证不误发）
  python run.py --offscreen --config my.yaml

屏外模式不依赖 Qt/GUI：自动回复循环是守护线程，窗口以「屏外可见」态运行，
用户完全无感，停止时还原为最小化（绝不闪现桌面）。

日志等级：
  [ OK ] 成功/正常动作（绿）
  [FAIL] 失败/异常（红，加粗）
  [WARN] 警告/模拟动作（黄）
  [STATE] 状态切换（青）
  [INFO] 普通信息
控制台带 ANSI 颜色，日志文件为纯文本（无颜色码）。
"""
import sys
import os
import argparse
import signal
import time
import threading
from pathlib import Path
from datetime import datetime


# ================================================================
# 颜色 / 分级
# ================================================================
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


# ================================================================
# 流保护（关键）
# PyInstaller windowed 模式（spec 里 console=False）会把 sys.stdout /
# sys.stderr 置为 None，此时任何 print() / sys.stdout.isatty() 都会抛
# AttributeError 直接崩。这里在模块最早期把它们重定向到 devnull，
# 使整个程序在窗口模式下也能正常运行（日志走 TeeLogger 文件 / UI 面板，
# 本就不需要控制台）；tty / 文件重定向 / nohup 场景下流非 None，原样保留。
# ================================================================
def _ensure_streams() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="ignore")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="ignore")


_ensure_streams()


# 全模块是否用颜色（控制台是 tty 才上色，后台 nohup 不染色）
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    if not _USE_COLOR or not color:
        return text
    return f"{color}{text}{RESET}"


# 日志等级 -> (前缀, 颜色)
LEVELS = {
    "info":  ("[INFO] ", ""),
    "ok":    ("[ OK ] ", GREEN),
    "warn":  ("[WARN] ", YELLOW),
    "fail":  ("[FAIL] ", RED + BOLD),
    "state": ("[STATE]", CYAN),
}


# ================================================================
# 公共初始化
# ================================================================
def setup_paths() -> Path:
    """固定工作目录并把项目根加入 sys.path，供所有模式复用。"""
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    return project_dir


def check_environment() -> bool:
    print("=" * 50)
    print("  VisReply - 环境检查")
    print("=" * 50)
    print()

    checks = []

    try:
        import webview  # noqa: F401
        checks.append(("pywebview", True))
    except ImportError:
        checks.append(("pywebview", False))

    try:
        import numpy
        checks.append(("numpy", True))
    except ImportError:
        checks.append(("numpy", False))

    try:
        import cv2
        checks.append(("opencv-python", True))
    except ImportError:
        checks.append(("opencv-python", False))

    try:
        import yaml
        checks.append(("PyYAML", True))
    except ImportError:
        checks.append(("PyYAML", False))

    try:
        import win32api
        checks.append(("pywin32", True))
    except ImportError:
        checks.append(("pywin32", False))

    try:
        from rapidocr_onnxruntime import RapidOCR
        checks.append(("rapidocr-onnxruntime", True))
    except ImportError:
        checks.append(("rapidocr-onnxruntime", False))

    all_ok = True
    for name, ok in checks:
        status = "正常" if ok else "缺失"
        symbol = "[正常]" if ok else "[缺失]"
        print(f"  {symbol} {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n" + _c("[错误] 部分依赖缺失！", RED + BOLD))
        print("运行: pip install -r requirements.txt")
        print()
        choice = input("Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            return False

    print()
    return True


def load_config(path: Path) -> dict:
    """加载 config.yaml 为 dict；不存在/解析失败则返回空 dict。"""
    try:
        import yaml
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(_c(f"[WARN] 读取配置文件失败: {e}", YELLOW))
    return {}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _echo_pin(logger: "TeeLogger", pin: dict) -> None:
    """双击置顶回显：打印「进入的会话名 / 是否带未读 / 未读数量校验结果」。"""
    if not pin:
        return
    name = pin.get("name") or "(未识别)"
    unread = pin.get("unread")
    method = pin.get("method") or "?"
    verify = pin.get("verify")
    before = pin.get("before")
    after = pin.get("after")
    extra = ""
    if method == "top_row":
        extra = f" 顶行有红点={pin.get('row_had_dot')}"
    # 未读数量校验结果 → 决定颜色等级（绿=成功 / 红=失败 / 黄=无法判定）
    if verify == "success":
        level, vtxt = "ok", "成功"
    elif verify == "fail":
        level, vtxt = "fail", "失败"
    elif verify == "unknown":
        level, vtxt = "warn", "无法判定"
    else:
        level, vtxt = "ok", None
    verify_txt = ""
    if vtxt is not None:
        verify_txt = f" 校验={vtxt}(双击前={before}→点击后={after})"
    logger.log(
        f"双击置顶进入会话: 会话名={name!r} 未读数={unread} 方式={method}{extra}{verify_txt}",
        level=level,
    )


class TeeLogger:
    """把日志同时打到控制台和 logs/offscreen_run.log。

    控制台带 ANSI 颜色分级，日志文件为纯文本（无颜色码、带时间戳前缀）。
    """

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.log_path = project_dir / "logs" / "offscreen_run.log"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.log_path = None

    def log(self, msg: str, level: str = "info") -> None:
        """分级日志：level ∈ {info, ok, warn, fail, state}。"""
        prefix, color = LEVELS.get(level, LEVELS["info"])
        plain = f"[{_ts()}] {prefix} {msg}"
        print(_c(plain, color))
        if self.log_path is not None:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(plain + "\n")
            except Exception:
                pass

    def log_raw(self, msg: str, color: str = "") -> None:
        """无前缀/无时间戳的纯文本（横幅、分隔线用）。"""
        print(_c(msg, color))
        if self.log_path is not None:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass


def _print_banner(logger: TeeLogger, mode_label: str, dry_run: bool,
                  config_path: Path) -> None:
    logger.log_raw("=" * 56)
    logger.log_raw("  微信自动回复 · 屏外挂机", CYAN + BOLD)
    logger.log_raw("-" * 56)
    logger.log_raw(f"  模式     : {mode_label}")
    logger.log_raw(f"  dry-run  : {'是（仅模拟，不真发）' if dry_run else '否'}")
    logger.log_raw(f"  配置     : {config_path}")
    logger.log_raw(f"  日志文件 : {logger.log_path}")
    logger.log_raw("=" * 56)


# ================================================================
# 屏外挂机模式（无 GUI）
# ================================================================
def run_offscreen(config: dict, project_dir: Path, dry_run: bool, once: bool) -> int:
    logger = TeeLogger(project_dir)
    _print_banner(
        logger,
        "单轮验证" if once else "持续挂机",
        dry_run,
        project_dir / "config.yaml",
    )

    # 让 cycle 内部也感知屏外（决策/截图策略可据此微调）
    config.setdefault("wechat", {})["offscreen"] = True

    from src.desktop.wechat_window_manager import WeChatWindowManager
    from src.agent.observe_service import ObserveService

    wm = WeChatWindowManager()
    hw = wm.find_wechat_window()
    if not hw:
        logger.log("未找到微信窗口，请先打开并登录微信。", level="fail")
        return 1

    # ---- 重度方案：虚拟显示器隔离（可选，需先装 usbmmidd_v2 驱动）----
    # 启用且健康时，微信停放在虚拟屏，与用户主屏光标彻底隔离；
    # 任何一步失败都降级回「屏外保活 + 空闲门控」，不崩溃。
    dm = None
    if config.get("vdd_enabled"):
        try:
            from src.desktop.display_manager import ManagedDisplayManager
            dm = ManagedDisplayManager(config)
            if dm.ensure_fixed_window_capacity() and dm.is_vdd_healthy():
                wm.display_manager = dm
                if wm.park_on_virtual_display(hw):
                    logger.log("虚拟显示器已启用，微信已停到虚拟屏（与主屏光标隔离）。",
                               level="ok")
                else:
                    logger.log("虚拟显示器启用命令已执行但停放失败，降级为屏外保活。",
                               level="warn")
                    dm = None
            else:
                logger.log("虚拟显示器容量不足（驱动可能未装/未以管理员运行），"
                           "降级为屏外保活。", level="warn")
                dm = None
        except Exception as e:
            logger.log(f"虚拟显示器初始化失败，降级为屏外保活: {e}", level="warn")
            dm = None

    # 强制屏外/虚拟屏：先最小化再 prepare。桌面态直接 move_offscreen 会被微信弹回，
    # 必须先最小化 → prepare_window（prepare 对最小化窗口转屏外可见态）。
    # 若已停到虚拟屏，prepare 仅接入 offscreen watcher，不强制移回主屏。
    wm.minimize(hw)
    time.sleep(0.3)
    wm.prepare_window(hw)
    time.sleep(0.5)
    if dm is None:
        rect = wm.get_rect(hw)
        logger.log(f"微信已移至屏外 x={rect.x if rect else None} "
                   f"offscreen={wm.is_window_offscreen(hw)}", level="ok")
    else:
        wm.park_on_virtual_display(hw)

    obs = ObserveService(config=config)
    if dm is not None:
        try:
            obs.window_manager.display_manager = dm
        except Exception:
            pass

    # 控制台事件钩子：把状态/失败/识别结果以分级方式呈现
    def on_event(event_type: str, data: dict) -> None:
        if event_type == "status":
            state = data.get("state", "")
            msg = data.get("message", "")
            logger.log(f"{state} · {msg}", level="state")
        elif event_type == "error":
            msg = data.get("message") or "(无错误信息)"
            logger.log(f"运行错误: {msg}", level="fail")
        elif event_type == "recognition":
            cnt = data.get("count", 0)
            if cnt:
                logger.log(f"识别到 {cnt} 个未读红点", level="ok")
        elif event_type == "preview":
            pass  # 预览帧，不输出到控制台
        else:
            logger.log(f"[{event_type}] {data}", level="info")
    obs.on_event(on_event)

    # 拦截发送：在「类级别」patch WeChatSender.send_report。
    # 原因：自动回复循环每轮可能重建 sender 实例（配置热重载路径），
    # 仅包装单个实例会在重建后失效、导致 dry-run 漏拦真实发送。
    # 类级别 patch 对重建后的实例依然生效。
    from src.rpa.wechat_sender import WeChatSender
    _orig_send = WeChatSender.send_report

    def wrapped_send(self_sender, report: dict) -> dict:
        analysis = report.get("analysis", report) if isinstance(report, dict) else {}
        contact = report.get("current_contact", "") or analysis.get("current_contact", "")
        reply = report.get("reply_draft", "") or analysis.get("reply_draft", "")
        if dry_run:
            logger.log(f"拟回复 {contact!r}: {reply[:60]!r}", level="warn")
            return {"ok": True, "reason": "dry-run", "action": "noop", "duration_ms": 0}
        logger.log(f"发送回复 -> {contact!r}: {reply[:60]!r}", level="info")
        res = _orig_send(self_sender, report)
        if res.get("ok"):
            logger.log(f"发送成功 -> {contact!r}", level="ok")
        else:
            reason = res.get("reason", "")
            logger.log(f"发送失败 -> {contact!r} 原因={reason!r}", level="fail")
        return res

    WeChatSender.send_report = wrapped_send

    # ---- 单轮模式：跑一个周期后退出 ----
    if once:
        logger.log("运行单轮...", level="info")
        obs.detect_wechat()
        res = obs.run_one_cycle()
        ok = res.get("ok")
        logger.log(
            f"单轮结果: ok={ok} contact={res.get('contact')!r} "
            f"idle={res.get('idle')} paused={res.get('paused')} "
            f"error={res.get('error', '')!r}",
            level="ok" if ok else "fail",
        )
        _echo_pin(logger, res.get("pin_echo"))
        if res.get("error"):
            logger.log(f"单轮异常详情: {res.get('error')}", level="fail")
        try:
            wm.minimize(hw)
        except Exception:
            pass
        wm.stop_offscreen_watcher()
        logger.log("已还原最小化，退出。", level="state")
        return 0

    # ---- 挂机循环模式 ----
    # 每轮结果汇总：把 run_one_cycle 的 flow / 发送成败浓缩成一行，
    # 让控制台一眼看出「本轮干了什么、成功还是失败」。
    _orig_run_one_cycle = ObserveService.run_one_cycle

    def wrapped_cycle(self_obs):
        res = _orig_run_one_cycle(self_obs)
        try:
            if res.get("error"):
                logger.log(f"周期异常: {res.get('error')}", level="fail")
            elif res.get("paused"):
                pass  # paused 已由 status 事件输出
            elif res.get("idle"):
                pass  # idle 已由 status 事件输出
            else:
                flow = res.get("flow") or []
                summary = " → ".join(
                    f"{s.get('name','?')}:{s.get('status','?')}" for s in flow
                )
                contact = res.get("contact") or ""
                send = res.get("send_result") or {}
                if send:
                    if send.get("ok"):
                        logger.log(
                            f"周期完成 contact={contact!r} 发送成功 | {summary}",
                            level="ok",
                        )
                    else:
                        logger.log(
                            f"周期完成 contact={contact!r} 发送失败 原因={send.get('reason','')!r} | {summary}",
                            level="fail",
                        )
                else:
                    logger.log(
                        f"周期完成 contact={contact!r} | {summary}", level="ok"
                    )
                # 双击置顶进入的会话名 / 是否带未读（置顶模式专属回显）
                _echo_pin(logger, res.get("pin_echo"))
        except Exception:
            pass
        return res

    ObserveService.run_one_cycle = wrapped_cycle

    from src.common.local_logger import LocalLogger
    LocalLogger.start_run()

    # 匿名使用统计（隐蔽）：静默上报一次启动，不影响主流程
    try:
        from src.telemetry.usage import report_startup
        report_startup()
    except Exception:
        pass

    obs.start_loop()
    logger.log("循环已启动（Ctrl+C 停止并还原最小化）。", level="state")

    stop_ev = threading.Event()

    def handle_sig(signum, frame):
        logger.log_raw("")
        logger.log("收到停止信号，正在优雅退出...", level="state")
        stop_ev.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop_ev.is_set():
            time.sleep(0.5)
    finally:
        obs.stop_loop()
        try:
            wm.minimize(hw)
        except Exception:
            pass
        wm.stop_offscreen_watcher()
        logger.log("已停止，微信最小化（未回桌面）。", level="state")
    return 0


# ================================================================
# GUI 模式（原行为，零改动）
# ================================================================
def run_gui() -> int:
    if not check_environment():
        return 1

    print(_c("[启动] 正在启动图形界面...", CYAN))
    print()

    try:
        from src.common.local_logger import LocalLogger
        LocalLogger.start_run()
        from src.config import Settings, load_settings
        from src.ui.webview_window import launch

        config_path = Path(__file__).resolve().parent / "config.yaml"
        settings = load_settings(str(config_path)) if config_path.exists() else Settings()

        # 匿名使用统计（隐蔽）：静默上报一次启动，不影响主流程
        try:
            from src.telemetry.usage import report_startup
            report_startup()
        except Exception:
            pass

        launch(settings)
        return 0

    except Exception as e:
        print(_c(f"\n[致命] 启动失败: {e}", RED + BOLD))
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return 1


# ================================================================
# 入口
# ================================================================
def main():
    setup_paths()

    parser = argparse.ArgumentParser(
        description="My WeChat Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="GUI 模式（默认）")
    mode.add_argument("--offscreen", action="store_true",
                      help="屏外挂机模式（无界面后台自动回复）")
    mode.add_argument("--once", action="store_true",
                      help="只跑一个周期后退出（与 --offscreen 同义但单轮）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只模拟决策、不真实发送（配合 --offscreen/--once）")
    parser.add_argument("--config", type=str, default=None,
                        help="指定配置文件路径（默认 config.yaml）")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else (project_dir / "config.yaml")
    config = load_config(config_path)

    # --once 视为屏外单轮
    if args.once:
        return run_offscreen(config, project_dir, dry_run=args.dry_run, once=True)
    if args.offscreen:
        return run_offscreen(config, project_dir, dry_run=args.dry_run, once=False)
    # 默认 GUI
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
