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
    echo [PicoVid] Python is not installed. Launching automatic setup ^(setup.cmd^)...
    call "%~dp0setup.cmd"
    exit /b %errorlevel%
)

if not exist "engine\sd-vulkan-ltx2\sd-cli.exe" if not exist "engine\sd-cuda-ltx2\sd-cli.exe" if not exist "build\sdcpp-vulkan\bin\sd-cli.exe" if not exist "build\sdcpp-cuda\bin\sd-cli.exe" (
    if "%~1"=="" (
        echo.
        echo [PicoVid] First run detected ^(engine binary not installed yet^).
        set /p "PV_SETUP=Run automatic setup ^(setup.cmd^) now? [Y/n]: "
        if /i "%PV_SETUP%"=="" set "PV_SETUP=Y"
        if /i "%PV_SETUP%"=="Y" (
            call "%~dp0setup.cmd"
            if errorlevel 1 exit /b %errorlevel%
        ) else if /i "%PV_SETUP%"=="E" (
            call "%~dp0setup.cmd"
            if errorlevel 1 exit /b %errorlevel%
        )
    )
)

REM Check required Python UI & H.264 packages (rich + imageio-ffmpeg)
%PY_CMD% -c "import rich, imageio_ffmpeg" >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [PicoVid] Missing Python packages detected: rich / imageio-ffmpeg
    set /p "LV_INSTALL=Install requirements.txt now? [Y/n]: "
    if /i "%LV_INSTALL%"=="" set "LV_INSTALL=Y"
    if /i "%LV_INSTALL%"=="Y" (
        %PY_CMD% -m pip install -r requirements.txt
    ) else if /i "%LV_INSTALL%"=="E" (
        %PY_CMD% -m pip install -r requirements.txt
    )
)

if /i "%~1"=="--web" (
    shift
    %PY_CMD% scripts\web_ui.py %1 %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %errorlevel%
)

%PY_CMD% scripts\generate.py %*
set "LV_EXIT=%errorlevel%"

if "%~1"=="" (
    echo.
    pause
)
exit /b %LV_EXIT%
