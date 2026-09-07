# -*- mode: python ; coding: utf-8 -*-
"""VisReply PyInstaller 打包 spec（pywebview 版）。

2026-09-06：UI 从 PySide6/QtWebEngine 迁移到 pywebview(edgechromium)，
本 spec 相应重写：
  1. 不再收集 PySide6 —— run.py/main.py 已不 import Qt，显式 excludes 兜底，
     包体积从 ~200MB(QT DLL) 降为仅 WebView2（系统自带，不打进包）。
  2. hiddenimports 显式带上 webview 的 Windows 两个平台模块：
     winforms（窗口层，基于 pythonnet）+ edgechromium（WebView2 渲染层）。
     pyinstaller-hooks-contrib 的 hook-webview 通常能自动处理，显式声明防漏。
  3. 保留：config.yaml 构建时脱敏（api_key 清空）、ocr_engine.py 源文件
     显式 datas（spec_from_file_location 动态加载绕过 PyInstaller）、
     rapidocr onnx 模型与 config.yaml。
  4. 目标机需 WebView2 Runtime：Win10 21H2+/Win11 预装；极老系统需引导
     安装 Evergreen Bootstrapper（约 1.8MB，微软官网下载）。
"""
import os
import sys

# rapidocr 模型目录（打包环境下 rapidocr_onnxruntime 已安装，动态定位）
try:
    import rapidocr_onnxruntime as _ro
    _RO_DIR = os.path.dirname(_ro.__file__)
    _RO_MODELS = os.path.join(_RO_DIR, 'models')
except Exception:
    _RO_MODELS = None

def _sanitize_config(src, dst):
    """读取 src 的 YAML 文本，把所有 api_key 字段值清空后写到 dst，返回 dst 路径。

    用正则而非 yaml 模块：保留原文件注释与缩进，且 spec 阶段不依赖 PyYAML。
    只匹配 `api_key:` / `xxx_api_key:` 行的值，注释和其他字段不动。
    """
    import re
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(src, 'r', encoding='utf-8') as f:
        text = f.read()
    # api_key: "sk-xxx" / api_key: sk-xxx / embedding_api_key: '...' 全部清空
    sanitized = re.sub(
        r'(?m)^(\s*[\w]*api_key\s*:\s*)(["\']?)[^\s"\']+(["\']?)',
        r'\1\2\3',
        text)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(sanitized)
    return dst


def _verify_config_parity(src, dst):
    """校验脱敏产物与源 config「除 api_key 行外完全一致」。

    方哥 2026-09-06 要求：打包进安装包的 config 必须与当前 config.yaml 一致，
    唯一例外是 api_key 清空（红线：密钥不得随包分发）。
    这里逐行比对（api_key 行只比键名不比值），不一致时把差异行打到构建日志，
    让每次打包都能自证「配置没被悄悄改坏/改少」。不抛异常，避免误伤构建。
    """
    import re as _re

    def _norm(path):
        with open(path, 'r', encoding='utf-8') as f:
            out = []
            for line in f:
                if _re.match(r'^\s*[\w]*api_key\s*:', line):
                    out.append('<api_key>')       # 值不参与比较
                else:
                    out.append(line.rstrip('\n'))
            return out

    try:
        a, b = _norm(src), _norm(dst)
    except Exception as e:  # noqa: BLE001
        print('[spec][警告] config 校验跳过：读文件失败 %s' % e)
        return False
    if len(a) != len(b):
        print('[spec][警告] config 行数不一致：源 %d 行 / 产物 %d 行' % (len(a), len(b)))
        return False
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not diffs:
        print('[spec] config 校验通过：打包配置 = 当前配置（仅 api_key 已清空，%d 行一致）' % len(a))
        return True
    print('[spec][警告] 打包配置与当前 config 存在 %d 处非 api_key 差异：' % len(diffs))
    for i, x, y in diffs[:10]:
        print('   行%-4d 源: %s' % (i + 1, x.strip()[:100]))
        print('   行%-4d 包: %s' % (i + 1, y.strip()[:100]))
    return False


# 构建时生成脱敏产物（放 _build_tmp/release/ 子目录隔离），并立即校验一致性
_CFG_RELEASE = _sanitize_config('config.yaml',
                                os.path.join('_build_tmp', 'release', 'config.yaml'))
_verify_config_parity('config.yaml', _CFG_RELEASE)

_datas = [
    ('resources', 'resources'),   # html + icons
    # RAG 知识库（FAQ/话术/产品资料，2026-09-06 方哥反馈"知识库没打包"）：
    # 缺了它，装完 RAG 无文档可检索、回复退化。config.yaml 的 knowledge.root
    # = "data/knowledge"，运行时按 _project_root()（打包后 = _internal）解析，
    # 故目标目录与源保持一致即可被找到。
    ('data/knowledge', 'data/knowledge'),
    # 运行时配置打进包，但【必须脱敏】：开发者本机 config.yaml 里有真实 API key，
    # 原样打包 = 把密钥分发给所有拿到安装包的人（2026-09-05 方哥明确要求不得泄露）。
    # 构建时生成脱敏产物：所有 api_key 字段清空，装完由用户在 UI 里填自己的。
    # 注意：datas 第二元素是目标【目录】，落地文件名=源文件名，
    # 所以脱敏产物必须叫 config.yaml（放 _build_tmp/release/ 子目录隔离）
    (_CFG_RELEASE, '.'),
    # reader.py / red_dot_detector.py 用 spec_from_file_location 动态加载
    # 该 .py 源文件（绕开 local_vision/__init__.py 的越级相对导入），
    # PyInstaller 只打包 .pyc 不保留 .py，故必须显式带源文件，否则 OCR 失效。
    (os.path.join('src', 'local_vision', 'ocr_engine.py'),
     os.path.join('src', 'local_vision')),
]
if _RO_MODELS:
    _datas.append((_RO_MODELS, 'rapidocr_onnxruntime/models'))
    # rapidocr 的 DEFAULT_CFG_PATH = root_dir/config.yaml，缺了 OCR 初始化即崩。
    # 注意：add-data 第二参数是「目标目录」不是文件路径，写错会多套一层目录
    # （变成 config.yaml/config.yaml）导致 rapidocr open 时报 PermissionError。
    _RO_CONFIG = os.path.join(_RO_DIR, 'config.yaml')
    if os.path.exists(_RO_CONFIG):
        _datas.append((_RO_CONFIG, 'rapidocr_onnxruntime'))

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # PySide6 已整体退役：入口代码零 import，此处 excludes 只是防止
    # 未来有人误加 Qt import 时悄悄把 200MB DLL 打回包里（构建即失败可见）。
    excludes=['PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
              'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.QtWebChannel', 'PySide6.QtNetwork', 'shiboken6',
              'PyQt5', 'PyQt6'],
    noarchive=False,
)

# 排除 OpenCV 冗余的 ffmpeg 视频后端 DLL（opencv_videoio_ffmpeg*.dll，约 +53MB）。
# OCR 管线只用 core/imgcodecs/imgproc：imencode/imdecode/cvtColor/imwrite，
# 完全不碰 VideoCapture/VideoWriter（视频 IO），删掉不影响任何功能。
# 2026-09-07 实测：移走后上述图像操作全部正常。
a.binaries = [b for b in a.binaries if 'opencv_videoio_ffmpeg' not in b[0]]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir 模式：二进制/数据交给下方 COLLECT
    name='WeChatAIAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,         # 纯 GUI：不弹 cmd 黑框
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icons/app.ico',   # exe 图标：资源管理器 + 任务栏 + 标题栏
)

# onedir 模式：产物是 dist_onedir/WeChatAIAssistant/ 完整目录（含 _internal/、
# resources/ 等），配合 installer.iss 打成传统安装包——安装后是
# 「带文件夹和文件的正常软件目录」，且启动无需 onefile 的整包解压，秒开。
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='WeChatAIAssistant',
)
