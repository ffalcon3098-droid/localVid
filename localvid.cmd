@echo off
REM Compatibility wrapper: localvid.cmd -> picovid.cmd
call "%~dp0picovid.cmd" %*
exit /b %errorlevel%
