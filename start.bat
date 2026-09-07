@echo off
chcp 65001 >nul
echo ============================================
echo   My WeChat Agent Launcher
echo   Pure Visual OCR WeChat Assistant
echo   (No Paid License, Fully Local)
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found, please install Python 3.11+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "venv" (
    echo [First Run] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Using domestic Tsinghua mirror source to speed up pip install
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    :: 强制补装PySide6，防止requirements漏写
    pip install PySide6 -i https://pypi.tuna.tsinghua.edu.cn/simple
    if not exist "_internal" (
        echo [TIP] Copy _internal directory from VisionLeadAgent to here
        echo       _internal contains RapidOCR and ONNX Runtime
    )
) else (
    call venv\Scripts\activate.bat
    :: 每次启动前校验关键UI依赖，缺失就自动补装
    python -c "import PySide6" 2>nul
    if %%errorlevel%% neq 0 (
        echo [WARN] PySide6 missing, auto installing...
        pip install PySide6 -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
)

echo.
echo [Starting] Launching assistant...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start, check logs
    pause
)
