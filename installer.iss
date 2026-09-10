; ============================================================
; 微信AI助手 安装包脚本（Inno Setup 6）
; 用法：装好 Inno Setup 6 后，右键本文件 -> Compile，
;       或用 ISCC.exe 编译，产出 WeChatAIAssistant_Setup.exe
; ============================================================

#define MyAppName "VisReply"
#define MyAppVersion "1.6.13"
#define MyAppPublisher "漩涡鸣人"
#define MyAppExeName "WeChatAIAssistant.exe"

[Setup]
; 纯用户级安装：不需要管理员权限，装到当前用户目录（与 DefaultDirName 配套）
PrivilegesRequired=lowest
; 在线更新（整包静默安装）时，强制关闭占用 {app} 文件的进程（WebView2 渲染进程、
; 杀软扫描锁、残留实例等），否则 Inno 无法替换文件会排队到「下次重启」，导致更新后版本号不变。
CloseApplications=yes
RestartApplications=no
;GUID 仅用于标识本软件（卸载/升级用），保持不变即可
AppId={{8E5C2A10-6B7D-4F3E-9A21-3C8D5F0B7E44}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; 默认装到用户目录，避开 C:\Program Files 的管理员权限/写入限制
; （本项目运行时要写 data/agent.db 和日志，装 Program Files 会写不进去）
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
; 输出目录和安装包文件名
OutputDir=installer_output
OutputBaseFilename=VisReply_Setup_{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
; 支持简体中文安装向导
ShowLanguageDialog=no
WizardStyle=modern
; 应用图标（NOYA 旷野松，resources/icons/app.ico）
SetupIconFile=resources\icons\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
; onedir 版：安装后是带 _internal/、DLL、resources 的完整软件目录
; （启动快，无需 onefile 每次解压 226MB）
; 【2026-09-08 修复】整包更新(在线更新走 full 时)原会用打包时脱敏(空 key)的
;   config.yaml 整体覆盖用户已配置好的 config → api_key 丢失、全部配置被重置。
;   修复：通配拷贝排除 config.yaml，单独用 onlyifdoesntexist 处理 ——
;   仅在首次安装写入默认配置，之后永远保留用户那份(含 api_key)。
Source: "dist_onedir8\WeChatAIAssistant\*"; DestDir: "{app}"; \
    Excludes: "config.yaml"; Flags: ignoreversion recursesubdirs createallsubdirs
; 【2026-09-09 修复路径】PyInstaller 6.22 把所有 datas 收进 _internal/ 子目录，
;   根目录并没有 config.yaml（旧脚本按根路径引用 → ISCC 报 Source file does not exist）。
;   真实路径是 _internal\config.yaml，仅首次安装写入；已存在则保留用户配置（含 api_key）。
Source: "dist_onedir8\WeChatAIAssistant\_internal\config.yaml"; DestDir: "{app}\_internal"; \
    Flags: onlyifdoesntexist
; 上面 Excludes: "config.yaml" 会按文件名匹配到所有层级，连带排掉
;   _internal\rapidocr_onnxruntime\config.yaml（rapidocr 初始化必需，缺了 OCR 直接崩），
;   这里显式补回，且允许覆盖（它是依赖自带配置，不含用户数据）。
Source: "dist_onedir8\WeChatAIAssistant\_internal\rapidocr_onnxruntime\config.yaml"; DestDir: "{app}\_internal\rapidocr_onnxruntime"; \
    Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; \
    GroupDescription: "附加任务："; Flags: checkedonce
Name: "autostart"; Description: "开机自动启动"; \
    GroupDescription: "附加任务："; Flags: unchecked

[Registry]
; 开机自启（勾选时写入）
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "WeChatAIAssistant"; \
    ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; \
    Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即运行 {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 卸载时保留用户数据不删（data/agent.db 等在 {localappdata} 下自行管理）
