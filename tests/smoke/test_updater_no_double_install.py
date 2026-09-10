"""验证 updater 修复：更新后不再自动再次重启/装两遍。

根因1：full 更新成功路径必须清理 _just_updated.txt / _update_failed.txt，
        否则下次启动 maybe_fallback 会再次触发整包更新（闭环）。
根因2：去掉 Popen 后 Timer(0.8) 的竞态，改为 os._exit(0) 由 bat 独占接管，
        避免新旧实例短暂双开。
"""
import inspect
from pathlib import Path

import src.updater as U
from src.updater import _make_full_bat, _make_increment_bat


def test_full_bat_clears_markers_before_relaunch():
    s = _make_full_bat(Path("X:/setup.exe"), Path("C:/app"),
                       Path("C:/app/WeChatAIAssistant.exe"))
    assert "_just_updated.txt" in s and "_update_failed.txt" in s
    del_idx = s.find('del "C:\\app\\_just_updated.txt"')
    start_idx = s.find('start "" "C:\\app\\WeChatAIAssistant.exe"')
    assert del_idx != -1, "full bat 未清理残留标记"
    assert start_idx != -1
    assert del_idx < start_idx, "del 标记必须在 start 新 exe 之前执行"


def test_increment_bat_clears_failed_marker():
    s = _make_increment_bat(Path("X:/staged"), Path("C:/app"),
                            Path("C:/app/WeChatAIAssistant.exe"), "1.6.13")
    assert "_update_failed.txt" in s
    assert 'del "%APP%\\_update_failed.txt"' in s


def test_no_race_timer_remaining():
    src = inspect.getsource(U)
    assert "threading.Timer(0.8" not in src, "仍存在 0.8s 竞态定时器"
    assert "os._exit(0)" in src, "应改为立即退出由 bat 接管"
