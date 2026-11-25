@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py"
if "%PY_CMD%"=="" (
    where python >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python"
)
if "%PY_CMD%"=="" (
    echo.
    echo [PicoVid] Python kurulu degil. Sifirdan otomatik kurulum ^(setup.cmd^) baslatiliyor...
    call "%~dp0setup.cmd"
    exit /b %errorlevel%
)

if not exist "build\sdcpp-vulkan\bin\sd-cli.exe" if not exist "build\sdcpp-cuda\bin\sd-cli.exe" (
    if "%~1"=="" (
        echo.
        echo [PicoVid] Ilk acilis tespit edildi ^(derlenmis motor henuz yok^).
        set /p "PV_SETUP=Sifirdan otomatik kurulumu ^(setup.cmd^) simdi calistiralim mi? [E/h]: "
        if /i "%PV_SETUP%"=="" set "PV_SETUP=E"
        if /i "%PV_SETUP%"=="E" (
            call "%~dp0setup.cmd"
            if errorlevel 1 exit /b %errorlevel%
        )
    )
)

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
