@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set "PY_CMD=python"
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py"
%PY_CMD% scripts\web_ui.py %*
