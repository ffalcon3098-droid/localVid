@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

set "PY_CMD=python"
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py"

REM Check required Python UI & H.264 packages (rich + imageio-ffmpeg)
%PY_CMD% -c "import rich, imageio_ffmpeg" >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [PicoVid] Eksik Python paketleri tespit edildi: rich / imageio-ffmpeg
    set /p "LV_INSTALL=requirements.txt simdi kurulsun mu? [E/h]: "
    if /i "%LV_INSTALL%"=="" set "LV_INSTALL=E"
    if /i "%LV_INSTALL%"=="E" (
        %PY_CMD% -m pip install -r requirements.txt
    ) else if /i "%LV_INSTALL%"=="Y" (
        %PY_CMD% -m pip install -r requirements.txt
    )
)

if /i "%~1"=="--web" (
    shift
    %PY_CMD% scripts\web_ui.py %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %errorlevel%
)

%PY_CMD% scripts\generate.py %*
set "LV_EXIT=%errorlevel%"

if "%~1"=="" (
    echo.
    pause
)
exit /b %LV_EXIT%
