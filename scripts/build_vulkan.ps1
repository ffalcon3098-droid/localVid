param(
    [string]$SourceDir = "third_party\stable-diffusion.cpp",
    [string]$BuildDir = "build\sdcpp-vulkan",
    [switch]$Clean,
    [switch]$Reconfigure
)

$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = (& $vswhere -latest -products * -property installationPath).Trim()
if (-not $vsPath) { throw "Visual Studio 2022 not found!" }

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$cmake  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

$vkRoot = $env:VULKAN_SDK
if (-not $vkRoot -or -not (Test-Path $vkRoot)) {
    $vkDir = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $vkDir) { throw "Vulkan SDK not found in C:\VulkanSDK!" }
    $vkRoot = $vkDir.FullName
}

Write-Host "VS2022    : $vsPath"
Write-Host "Vulkan SDK: $vkRoot"
Write-Host "Source    : $SourceDir"
Write-Host "Build     : $BuildDir"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

$needConfig = $Clean -or $Reconfigure -or (-not (Test-Path (Join-Path $BuildDir "build.ninja")))
$configLine = if ($needConfig) {
    "`"$cmake`" -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release -DSD_VULKAN=ON -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SERVER=OFF"
} else {
    "echo Skipping CMake configure (build.ninja exists)"
}

$tmpCmd = Join-Path $env:TEMP "localvid_build_vulkan.cmd"
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
set "VULKAN_SDK=$vkRoot"
set "PATH=$vkRoot\Bin;%PATH%"
$configLine
if errorlevel 1 exit /b 1
"$cmake" --build "$BuildDir" --target sd-cli --parallel
if errorlevel 1 exit /b 1
"@ | Set-Content -Path $tmpCmd -Encoding ASCII

cmd.exe /c "$tmpCmd"
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

Write-Host "BUILD_OK: $(Join-Path $BuildDir 'bin\sd-cli.exe')"
