param(
    [ValidateSet("vulkan", "cuda")]
    [string]$Backend = "vulkan",
    [string]$SourceDir = "third_party\stable-diffusion.cpp",
    [string]$BuildDir = "",
    [string]$CudaArch = "",
    [switch]$Clean,
    [switch]$Reconfigure
)

$ErrorActionPreference = "Stop"

if (-not $BuildDir) {
    $BuildDir = "build\sdcpp-$Backend"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = (& $vswhere -latest -products * -property installationPath).Trim()
if (-not $vsPath) { throw "Visual Studio 2022 not found!" }

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$cmake  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

$extraEnv = ""
$cmakeFlags = "-DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SERVER=OFF"

if ($Backend -eq "vulkan") {
    $vkRoot = $env:VULKAN_SDK
    if (-not $vkRoot -or -not (Test-Path $vkRoot)) {
        $vkDir = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $vkDir) { throw "Vulkan SDK not found! Install via: winget install KhronosGroup.VulkanSDK" }
        $vkRoot = $vkDir.FullName
    }
    $extraEnv = "set `"VULKAN_SDK=$vkRoot`"`r`nset `"PATH=$vkRoot\Bin;%PATH%`""
    $cmakeFlags += " -DSD_VULKAN=ON"
    Write-Host "Vulkan SDK: $vkRoot"
} elseif ($Backend -eq "cuda") {
    $cmakeFlags += " -DSD_CUDA=ON"
    if ($CudaArch) {
        $cmakeFlags += " -DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    }
    Write-Host "CUDA build enabled (ensure CUDA Toolkit nvcc is in PATH)"
}

Write-Host "VS2022    : $vsPath"
Write-Host "Backend   : $Backend"
Write-Host "Source    : $SourceDir"
Write-Host "Build     : $BuildDir"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

$needConfig = $Clean -or $Reconfigure -or (-not (Test-Path (Join-Path $BuildDir "build.ninja")))
$configLine = if ($needConfig) {
    "`"$cmake`" -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" $cmakeFlags"
} else {
    "echo Skipping CMake configure (build.ninja exists)"
}

$tmpCmd = Join-Path $env:TEMP "localvid_build_$Backend.cmd"
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
$extraEnv
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
