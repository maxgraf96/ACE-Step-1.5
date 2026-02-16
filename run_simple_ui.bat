@echo off
chcp 65001 >nul
echo ==========================================
echo   ACE-Step Simple UI
echo   Streamlined interface for executives
echo ==========================================
echo.

:: Change to script directory
cd /d "%~dp0"

:: Check if uv is available
where uv >nul 2>nul
if %errorlevel% == 0 (
    echo Using uv to run Simple UI...
    uv run acestep-simple %*
) else (
    echo Using Python directly...
    python -m acestep.simple_ui %*
)

echo.
pause
