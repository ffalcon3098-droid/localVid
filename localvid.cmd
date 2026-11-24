@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where py >nul 2>&1
if %errorlevel%==0 (
    py scripts\generate.py %*
) else (
    python scripts\generate.py %*
)
set "LV_EXIT=%errorlevel%"

if "%~1"=="" (
    echo.
    pause
)
exit /b %LV_EXIT%
