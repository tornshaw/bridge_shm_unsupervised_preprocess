@echo off
setlocal ENABLEDELAYEDEXPANSION

chcp 65001 >nul

echo ==============================================
echo  桥梁健康监测系统传感器健康状态无监督可视化分析软件
 echo  Windows 一键打包脚本
 echo ==============================================

set APP_NAME=桥梁健康监测系统传感器健康状态无监督可视化分析软件
set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

set PYTHON_EXE=python
if not "%~1"=="" (
  set PYTHON_EXE=%~1
)

echo [1/7] 检查 Python...
%PYTHON_EXE% --version >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python，请安装后重试。
  goto :fail
)

echo [2/7] 创建虚拟环境 .venv_exe_build ...
if exist .venv_exe_build (
  echo [提示] 已存在 .venv_exe_build，复用该环境。
) else (
  %PYTHON_EXE% -m venv .venv_exe_build
  if errorlevel 1 goto :fail
)

call .venv_exe_build\Scripts\activate.bat
if errorlevel 1 goto :fail

echo [3/7] 升级 pip/setuptools/wheel...
python -m pip install -U pip setuptools wheel
if errorlevel 1 goto :fail

echo [4/7] 安装依赖（含 CPU 版 torch）...
python -m pip install -U pyinstaller pandas numpy matplotlib scikit-learn openpyxl pymysql
if errorlevel 1 goto :fail
python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [5/7] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "%APP_NAME%.spec" del /q "%APP_NAME%.spec"

echo [6/7] 执行 PyInstaller...
pyinstaller --noconfirm --clean --windowed ^
  --name "%APP_NAME%" ^
  --collect-all torch ^
  --collect-all sklearn ^
  --collect-all matplotlib ^
  --collect-all pandas ^
  --hidden-import torch ^
  --hidden-import torch._C ^
  run_gui_app.py
if errorlevel 1 goto :fail

echo [7/7] 复制运行时辅助文件...
if exist mapping (
  xcopy /E /I /Y mapping "dist\%APP_NAME%\mapping" >nul
)
if exist data-export-csv.py (
  copy /Y data-export-csv.py "dist\%APP_NAME%\data-export-csv.py" >nul
)
if exist data (
  xcopy /E /I /Y data "dist\%APP_NAME%\data" >nul
)

echo.
echo [成功] 打包完成：
echo dist\%APP_NAME%\%APP_NAME%.exe
echo.
echo 说明：
echo - 首次启动较慢属正常现象。
echo - 若在线模式使用 Doris，请确认 data-export-csv.py 参数与 GUI 约定一致。

goto :eof

:fail
echo.
echo [失败] 打包流程中断，请检查上方日志。
exit /b 1
