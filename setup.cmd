@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set "SETUP_EXIT=%errorlevel%"
if "%~1"=="" (
    echo.
    pause
)
exit /b %SETUP_EXIT%
