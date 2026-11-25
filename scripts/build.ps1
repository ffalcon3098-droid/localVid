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
$vsPath = ""
if (Test-Path $vswhere) {
    # Prefer a Visual Studio / BuildTools instance that ACTUALLY has C++ x64 build tools installed
    $vsRaw = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsRaw) {
        $vsPath = (@($vsRaw) | Where-Object { $_ -and $_.Trim() } | Select-Object -First 1).Trim()
    }
}

# Fallback: scan common Visual Studio (2019/2022/18/BuildTools, x64 & x86) locations for vcvars64.bat
if (-not $vsPath -or -not (Test-Path (Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"))) {
    foreach ($baseDir in @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio"
    )) {
        if (Test-Path $baseDir) {
            $foundVc = Get-ChildItem -Path $baseDir -Filter "vcvars64.bat" -Recurse -Depth 5 -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($foundVc) {
                $vsPath = (Resolve-Path (Join-Path $foundVc.DirectoryName "..\..\..")).Path
                break
            }
        }
    }
}

$vcvars = if ($vsPath) { Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" } else { "" }
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    Write-Host ""
    Write-Host "HATA: Visual Studio C++ derleyicisi (vcvars64.bat) bulunamadi!" -ForegroundColor Red
    Write-Host "Cozum (tek komutla C++ Build Tools + CMake kurulumu):" -ForegroundColor Yellow
    Write-Host '  winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"' -ForegroundColor Cyan
    Write-Host "Veya Visual Studio Installer -> Modify -> 'Desktop development with C++' secenegini isaretleyin." -ForegroundColor Yellow
    throw "vcvars64.bat not found (Visual Studio C++ workload missing)."
}

# Locate CMake (prefer VS bundled, fallback to system PATH)
$cmake = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $cmake)) {
    $sysCmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($sysCmake) {
        $cmake = $sysCmake.Source
    } else {
        throw "cmake.exe not found! Install via: winget install Kitware.CMake"
    }
}

# Locate Ninja (prefer VS bundled, fallback to system PATH, else fall back to Visual Studio generator)
$ninja = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $ninja)) {
    $sysNinja = Get-Command ninja.exe -ErrorAction SilentlyContinue
    $ninja = if ($sysNinja) { $sysNinja.Source } else { "" }
}

$extraEnvLines = @()
$cmakeFlags = "-DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SERVER=OFF"

if ($Backend -eq "vulkan") {
    $vkRoot = $env:VULKAN_SDK
    if (-not $vkRoot -or -not (Test-Path $vkRoot)) {
        $vkDir = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $vkDir) { throw "Vulkan SDK not found! Install via: winget install KhronosGroup.VulkanSDK" }
        $vkRoot = $vkDir.FullName
    }
    $extraEnvLines += "set `"VULKAN_SDK=$vkRoot`""
    $extraEnvLines += "set `"PATH=$vkRoot\Bin;%PATH%`""
    $cmakeFlags += " -DSD_VULKAN=ON"
    Write-Host "Vulkan SDK: $vkRoot"
} elseif ($Backend -eq "cuda") {
    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot -or -not (Test-Path $cudaRoot)) {
        $cudaDir = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($cudaDir) { $cudaRoot = $cudaDir.FullName }
    }
    if ($cudaRoot -and (Test-Path (Join-Path $cudaRoot "bin\nvcc.exe"))) {
        $extraEnvLines += "set `"CUDA_PATH=$cudaRoot`""
        $extraEnvLines += "set `"PATH=$cudaRoot\bin;%PATH%`""
        Write-Host "CUDA SDK  : $cudaRoot"
    } else {
        Write-Host "CUDA build enabled (ensure CUDA Toolkit nvcc is in PATH)"
    }
    $cmakeFlags += " -DSD_CUDA=ON"
    if ($CudaArch) {
        $cmakeFlags += " -DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    }
}

$extraEnv = ($extraEnvLines -join "`r`n")

Write-Host "VS C++    : $vsPath"
Write-Host "CMake     : $cmake"
Write-Host "Generator : $(if ($ninja) { "Ninja ($ninja)" } else { 'Visual Studio 17 2022 (x64)' })"
Write-Host "Backend   : $Backend"
Write-Host "Source    : $SourceDir"
Write-Host "Build     : $BuildDir"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

if ($ninja) {
    $needConfig = $Clean -or $Reconfigure -or (-not (Test-Path (Join-Path $BuildDir "build.ninja")))
    $configLine = if ($needConfig) {
        "`"$cmake`" -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" $cmakeFlags"
    } else {
        "echo Skipping CMake configure (build.ninja exists)"
    }
    $buildLine = "`"$cmake`" --build `"$BuildDir`" --target sd-cli --parallel"
} else {
    $needConfig = $Clean -or $Reconfigure -or (-not (Test-Path (Join-Path $BuildDir "CMakeCache.txt")))
    $configLine = if ($needConfig) {
        "`"$cmake`" -S `"$SourceDir`" -B `"$BuildDir`" -G `"Visual Studio 17 2022`" -A x64 $cmakeFlags"
    } else {
        "echo Skipping CMake configure (CMakeCache.txt exists)"
    }
    $buildLine = "`"$cmake`" --build `"$BuildDir`" --config Release --target sd-cli --parallel"
}

$tmpCmd = Join-Path $env:TEMP "picovid_build_$Backend.cmd"
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
$extraEnv
$configLine
if errorlevel 1 exit /b 1
$buildLine
if errorlevel 1 exit /b 1
"@ | Set-Content -Path $tmpCmd -Encoding ASCII

cmd.exe /c "$tmpCmd"
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

# Visual Studio multi-config generator places binary under bin\Release\sd-cli.exe -> copy to bin\sd-cli.exe if needed
$outBin = Join-Path $BuildDir "bin\sd-cli.exe"
$relBin = Join-Path $BuildDir "bin\Release\sd-cli.exe"
if (-not (Test-Path $outBin) -and (Test-Path $relBin)) {
    Copy-Item $relBin $outBin -Force
    Get-ChildItem (Join-Path $BuildDir "bin\Release\*.dll") -ErrorAction SilentlyContinue | Copy-Item -Destination (Join-Path $BuildDir "bin") -Force
}

Write-Host "BUILD_OK: $outBin"
