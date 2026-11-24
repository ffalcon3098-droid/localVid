@echo off
setlocal
cd /d "%~dp0"
py scripts\web_ui.py %*
