@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\pgAdmin 4\python\python.exe" (
        set "PY_CMD=C:\Program Files\pgAdmin 4\python\python.exe"
    )
)

"%PY_CMD%" -m backend.cli %*
