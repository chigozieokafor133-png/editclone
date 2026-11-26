@echo off
setlocal
cd /d "%~dp0"
echo =====================================================================
echo  EditClone - Launching Full-Stack Web Server (http://localhost:8000)
echo =====================================================================

set "PY_CMD=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\pgAdmin 4\python\python.exe" (
        set "PY_CMD=C:\Program Files\pgAdmin 4\python\python.exe"
    )
)

echo Using Python: %PY_CMD%
"%PY_CMD%" backend/server.py
pause
