#!/usr/bin/env python3
"""诊断：点开列表里第一个会话，复现会话视图，看 current_contact 为何抽不出来。

打印 view_gate 证据、anchors、逐行坐标，以及对 _extract_contact 的命中情况。
"""
from __future__ import annotations
import sys, os, time, ctypes, ctypes.wintypes
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)
import cv2
from src.desktop.wechat_window_manager import WeChatWindowManager
from src.capture.screen_capture import ScreenCapture
from src.clean_perception.reader import WechatScreenReader, _normalize_ocr, ChatAnalysis
from src.clean_perception.view_gate import ViewGate

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
def pid_name(pid):
    try:
        h=kernel32.OpenProcess(0x0410,False,pid); buf=ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h,0,buf,260); kernel32.CloseHandle(h); return buf.value.split("\\")[-1].lower()
    except: return "?"
hwnd=None
def enum(hw,_):
    global hwnd
    buf=ctypes.create_unicode_buffer(512); user32.GetWindowTextW(hw,buf,512); t=buf.value
    pid=ctypes.wintypes.DWORD(); user32.GetWindowThreadProcessId(hw,ctypes.byref(pid)); n=pid_name(pid.value)
    if ("weixin.exe" in n or "wechat.exe" in n) and ("微信" in t) and t.strip()=="微信":
        hwnd=hw
    return True
EP=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.wintypes.HWND,ctypes.wintypes.LPARAM)
user32.EnumWindows(EP(enum),0)

wm=WeChatWindowManager(); wm._hwnd_wechat=hwnd; wm.show_window(hwnd); time.sleep(0.6)

# 取窗口矩形（点击前重新获取，避免坐标漂移）
def get_rect():
    r=ctypes.wintypes.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right-r.left, r.bottom-r.top
wx,wy,ww,wh = get_rect()
print(f"window: x={wx} y={wy} w={ww} h={wh}")

# 点开第一个会话：列表第一项"方政"在图像 x≈184-234, y≈140-170
# 屏幕坐标 = 窗口左上 + 图像坐标（1:1 映射）
target_x = wx + 209   # (184+234)/2
target_y = wy + 155   # (140+170)/2
print(f"click conversation at ({target_x},{target_y})")
def click(x,y):
    user32.SetCursorPos(x,y)
    time.sleep(0.05)
    user32.mouse_event(0x0002,0,0,0,0)  # LEFT DOWN
    user32.mouse_event(0x0004,0,0,0,0)  # LEFT UP
user32.SetForegroundWindow(hwnd)
time.sleep(0.3)
click(target_x,target_y)  # 微信桌面版单击列表项即可打开会话
time.sleep(2.0)
# 验证是否进入会话（右侧聊天区出现 x>=460 的内容行）
def seems_open():
    r=ctypes.wintypes.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left,r.top,r.right-r.left,r.bottom-r.top)
print(f"after click rect={seems_open()}")

sc=ScreenCapture(window_manager=wm, config={"data_dir":os.path.join(os.getcwd(),"data")})
res=sc.capture_window(None, hwnd=hwnd, subdir="wechat", prefix="conv")
img = cv2.imread(res) if isinstance(res,str) else (res.image if getattr(res,"image",None) is not None else None)
h,w=img.shape[:2]
print(f"captured {w}x{h}")
cv2.imwrite("data/wechat/debug_conv.png", img)

reader=WechatScreenReader()
arr=reader._to_array(img)
ocr=reader._get_ocr(); enhanced=reader._enhance(arr)
raw=ocr.run(enhanced) if hasattr(ocr,"run") else None
lines=_normalize_ocr(raw)
analysis=ChatAnalysis(image_size=(w,h))
lines=reader._quality_filter(lines, analysis)

vg=ViewGate(); verdict=vg.judge(arr, lines)
anchors=reader.layout.estimate(arr, lines)
print(f"\nview={verdict.view} conf={verdict.confidence} evidence={verdict.evidence}")
print(f"anchors: chat_left={anchors.chat_left} header_bottom={anchors.header_bottom} chat_right={anchors.chat_right}")
print("=" * 100)
print(f"{'#':>2} {'x_min':>5} {'x_max':>5} {'y_min':>5} {'y_max':>5} {'cy':>5} {'scr':>5} {'low':>3}  text")
for i,ln in enumerate(lines):
    print(f"{i:>2} {ln.x_min:>5.0f} {ln.x_max:>5.0f} {ln.y_min:>5.0f} {ln.y_max:>5.0f} {ln.cy:>5.0f} {ln.score:>5.2f} {str(ln.low_conf):>3}  {ln.text}")
print("=" * 100)
contact=reader._extract_contact(lines, anchors, h, w)
print(f"\n_extract_contact -> {contact!r}")
print(f"raw_text_lines count={len(analysis.raw_text_lines)}")
