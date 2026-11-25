param(
    [ValidateSet("auto", "cuda", "vulkan")]
    [string]$Backend = "auto",
    [string]$Profile = "auto",
    [switch]$SkipModels,
    [switch]$CheckOnly,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Refresh-EnvPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = ($machine, $user | Where-Object { $_ }) -join ";"
}

function Ensure-Winget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[UYARI] winget bulunamadi (App Installer). Eksik araclari manuel kurmaniz gerekebilir." -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $ver = & python --version 2>&1
        if ($ver -match "Python 3\.") { return "python" }
    }
    return $null
}

function Find-VCVars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($p) {
            $first = (@($p) | Where-Object { $_ -and $_.Trim() } | Select-Object -First 1).Trim()
            $vc = Join-Path $first "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vc) { return $vc }
        }
    }
    foreach ($root in @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    )) {
        $vc = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vc) { return $vc }
    }
    return $null
}

function Detect-TargetBackend {
    if ($Backend -ne "auto") { return $Backend }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) { return "cuda" }
    }
    return "vulkan"
}

function Find-CudaNvcc {
    if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) { return (Get-Command nvcc.exe).Source }
    $cudaDir = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($cudaDir -and (Test-Path (Join-Path $cudaDir.FullName "bin\nvcc.exe"))) {
        return (Join-Path $cudaDir.FullName "bin\nvcc.exe")
    }
    return $null
}

function Find-VulkanSdk {
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) { return $env:VULKAN_SDK }
    $vkDir = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($vkDir) { return $vkDir.FullName }
    return $null
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PicoVid v0.5.0 · Sifirdan Otomatik Kurulum Sihirbazi (Windows)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$targetBackend = Detect-TargetBackend
$sdCliPath     = Join-Path $RepoRoot "build\sdcpp-$targetBackend\bin\sd-cli.exe"
$hasGit        = [bool](Get-Command git -ErrorAction SilentlyContinue)
$pyCmd         = Find-Python
$vcVars        = Find-VCVars
$sdkPath       = if ($targetBackend -eq "cuda") { Find-CudaNvcc } else { Find-VulkanSdk }
$hasBin        = Test-Path $sdCliPath
$hasRealesrgan = Test-Path (Join-Path $RepoRoot "tools\realesrgan\realesrgan-ncnn-vulkan.exe")

Write-Host ("  Algilanan Backend   : " + $targetBackend.ToUpper()) -ForegroundColor Green
Write-Host ("  [1] Git             : " + $(if ($hasGit) { "HAZIR" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [2] Python 3        : " + $(if ($pyCmd) { "HAZIR ($pyCmd)" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [3] VS C++ Tools    : " + $(if ($vcVars) { "HAZIR" } elseif ($hasBin) { "GEREKMIYOR (sd-cli hazir)" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [4] $($targetBackend.ToUpper()) SDK       : " + $(if ($sdkPath) { "HAZIR ($sdkPath)" } elseif ($hasBin) { "GEREKMIYOR (sd-cli hazir)" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [5] sd-cli.exe      : " + $(if ($hasBin) { "HAZIR ($sdCliPath)" } else { "DERLENECEK" }))
Write-Host ("  [6] Cinema HD Tool  : " + $(if ($hasRealesrgan) { "HAZIR" } else { "INDIRILECEK (43 MB)" }))
Write-Host "=================================================================" -ForegroundColor Cyan

if ($CheckOnly) {
    Write-Host "[CHECK-ONLY] Donanim & on-kosul kontrolu tamamlandi." -ForegroundColor Yellow
    exit 0
}

# 1. Git
if (-not $hasGit) {
    Write-Host "`n[1/7] Git kuruluyor..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
    } else { throw "Git kurulu degil: https://git-scm.com/download/win" }
}

# 2. Python 3.12
if (-not $pyCmd) {
    Write-Host "`n[2/7] Python 3.12 kuruluyor..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
        $pyCmd = Find-Python
    }
    if (-not $pyCmd) { throw "Python kuruldu ama PATH guncellenemedi. Terminali kapatip yeniden acin ve setup.cmd'yi tekrar calistirin." }
}

# 3. Python packages
Write-Host "`n[3/7] Python paketleri (requirements.txt) kontrol ediliyor..." -ForegroundColor Cyan
& $pyCmd -m pip install --upgrade pip --quiet 2>$null
& $pyCmd -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt basarisiz oldu!" }

# 4 & 5. Build prerequisites + sd-cli.exe compilation (only if binary not already present)
if (-not $hasBin) {
    if (-not $vcVars) {
        Write-Host "`n[4/7] Visual Studio 2022 C++ Build Tools + CMake kuruluyor (birkaç dakika surebilir)..." -ForegroundColor Cyan
        if (Ensure-Winget) {
            winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"
            Refresh-EnvPath
            $vcVars = Find-VCVars
        }
        if (-not $vcVars) { throw "Visual Studio C++ Build Tools kurulamadi. Lutfen kurulumdan sonra setup.cmd'yi tekrar calistirin." }
    }

    if (-not $sdkPath) {
        if ($targetBackend -eq "cuda") {
            Write-Host "`n[5/7] NVIDIA CUDA Toolkit kuruluyor..." -ForegroundColor Cyan
            if (Ensure-Winget) {
                winget install -e --id Nvidia.CUDA --accept-package-agreements --accept-source-agreements
                Refresh-EnvPath
            }
        } else {
            Write-Host "`n[5/7] Khronos Vulkan SDK kuruluyor..." -ForegroundColor Cyan
            if (Ensure-Winget) {
                winget install -e --id KhronosGroup.VulkanSDK --accept-package-agreements --accept-source-agreements
                Refresh-EnvPath
            }
        }
    }

    $srcDir = Join-Path $RepoRoot "third_party\stable-diffusion.cpp"
    if (-not (Test-Path (Join-Path $srcDir ".git"))) {
        Write-Host "`n[6/7] stable-diffusion.cpp (3b12781) klonlaniyor..." -ForegroundColor Cyan
        git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git $srcDir
        if ($LASTEXITCODE -ne 0) { throw "git clone basarisiz!" }
        git -C $srcDir fetch --depth 1 origin 3b12781 2>$null
        git -C $srcDir checkout 3b12781
    }

    # Apply 9 PicoVid patches if not yet applied
    powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1") -CheckOnly *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      9 PicoVid yamasi uygulaniyor..." -ForegroundColor Cyan
        powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1")
        if ($LASTEXITCODE -ne 0) { throw "apply_patches.ps1 basarisiz!" }
    }

    Write-Host "      sd-cli.exe ($targetBackend) derleniyor..." -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\build.ps1") -Backend $targetBackend
    if ($LASTEXITCODE -ne 0) { throw "build.ps1 basarisiz!" }
} else {
    Write-Host "`n[4-6/7] Derlenmis sd-cli.exe zaten hazir, derleme atlandi." -ForegroundColor Green
}

# 6. Cinema HD Vulkan Upscaler (43 MB)
if (-not $hasRealesrgan) {
    Write-Host "`n[7/7] Cinema HD Vulkan araci (realesrgan-ncnn-vulkan, 43 MB) indiriliyor..." -ForegroundColor Cyan
    & $pyCmd (Join-Path $RepoRoot "scripts\postprocess.py") --install-only
}

# 7. Models
if (-not $SkipModels) {
    Write-Host "`n[Modeller] Akilli model indirici baslatiliyor..." -ForegroundColor Cyan
    if ($Profile -eq "auto") {
        if ($Yes) {
            & $pyCmd (Join-Path $RepoRoot "scripts\download_models.py") --profile fast-5b
        } else {
            & $pyCmd (Join-Path $RepoRoot "scripts\download_models.py") --interactive
        }
    } elseif ($Profile -ne "skip") {
        & $pyCmd (Join-Path $RepoRoot "scripts\download_models.py") --profile $Profile
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[UYARI] Model indirmede sorun olustu; istediginiz zaman 'py scripts\download_models.py' ile kaldigi yerden devam edebilirsiniz." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  KURULUM TAMAMLANDI! PicoVid kullanima hazir:" -ForegroundColor Green
Write-Host "    Terminal Sihirbazi : .\picovid.cmd" -ForegroundColor White
Write-Host "    Web Studyosu       : .\web-studio.cmd  (veya .\picovid.cmd --web)" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
