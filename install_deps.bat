@echo off
setlocal
cd /d "%~dp0"
echo =====================================================================
echo  Installing EditClone Dependencies (User Space - No Admin Required)
echo =====================================================================

set "PY_CMD=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\pgAdmin 4\python\python.exe" (
        set "PY_CMD=C:\Program Files\pgAdmin 4\python\python.exe"
    )
)

echo Using Python: %PY_CMD%
"%PY_CMD%" -m pip install --user -r requirements.txt
echo.
echo Dependencies installation finished!
pause
