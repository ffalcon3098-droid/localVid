param(
    [ValidateSet("auto", "cuda", "vulkan")]
    [string]$Backend = "auto",
    [string]$Profile = "auto",
    [switch]$BuildFromSource,
    [switch]$SkipModels,
    [switch]$CheckOnly,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$ReleaseRepo = "ffalcon3098-droid/localVid"
$ReleaseTag  = "v0.5.1"
$PrebuiltAsset = "picovid-sd-cli-win64-vulkan.zip"
$Ltx2PrebuiltUrl = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1133-52491bd/sd-master-52491bd-bin-win-vulkan-x64.zip"

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
    foreach ($baseDir in @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio"
    )) {
        if (Test-Path $baseDir) {
            $found = Get-ChildItem -Path $baseDir -Filter "vcvars64.bat" -Recurse -Depth 5 -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    }
    return $null
}

function Install-VCToolsSafe {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsInstaller = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe"

    # If VS is already installed without C++ tools, modify it directly instead of failing in winget!
    if ((Test-Path $vswhere) -and (Test-Path $vsInstaller)) {
        $existing = & $vswhere -latest -products * -property installationPath 2>$null
        if ($existing) {
            $instPath = (@($existing) | Where-Object { $_ -and $_.Trim() } | Select-Object -First 1).Trim()
            if ($instPath -and (Test-Path $instPath)) {
                Write-Host "      Mevcut Visual Studio ($instPath) uzerine C++ Build Tools ekleniyor..." -ForegroundColor Cyan
                $p = Start-Process -FilePath $vsInstaller -ArgumentList @(
                    "modify", "--installPath", "`"$instPath`"",
                    "--add", "Microsoft.VisualStudio.Workload.VCTools",
                    "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project",
                    "--includeRecommended", "--passive", "--norestart"
                ) -Wait -PassThru
                Refresh-EnvPath
                $vc = Find-VCVars
                if ($vc) { return $vc }
            }
        }
    }

    if (Ensure-Winget) {
        Write-Host "      Visual Studio 2022 Build Tools (C++ + CMake) kuruluyor..." -ForegroundColor Cyan
        winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended" | Out-Null
        Refresh-EnvPath
        $vc = Find-VCVars
        if ($vc) { return $vc }
    }
    return $null
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

function Detect-TargetBackend {
    if ($Backend -ne "auto") { return $Backend }
    # If CUDA Toolkit + NVIDIA GPU are both ready AND user wants source build, pick CUDA.
    # Otherwise Vulkan works out-of-the-box on AMD, Intel AND NVIDIA GPUs with our prebuilt binary!
    if ($BuildFromSource -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) { return "cuda" }
    }
    if ((Test-Path (Join-Path $RepoRoot "build\sdcpp-cuda\bin\sd-cli.exe"))) { return "cuda" }
    return "vulkan"
}

function Try-DownloadPrebuiltBinary {
    param([string]$DestBinDir, [string]$PyExe = "py")
    New-Item -ItemType Directory -Force -Path $DestBinDir | Out-Null
    $tmpZip = Join-Path $env:TEMP $PrebuiltAsset
    if (Test-Path $tmpZip) { Remove-Item $tmpZip -Force }

    Write-Host "`n[Hizli Kurulum] Hazir derlenmis sd-cli.exe ($ReleaseTag · Vulkan — AMD/Intel/NVIDIA uyumlu) indiriliyor..." -ForegroundColor Cyan
    Write-Host "                (Visual Studio / C++ / SDK derlemesine gerek kalmaz!)" -ForegroundColor Green

    $downloaded = $false

    # Method 1: GitHub CLI (handles both private and public repos seamlessly)
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        try {
            & gh release download $ReleaseTag --repo $ReleaseRepo --pattern $PrebuiltAsset --dir $env:TEMP --clobber 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-Path $tmpZip) -and ((Get-Item $tmpZip).Length -gt 1000000)) {
                $downloaded = $true
            }
        } catch {}
    }

    # Method 2: Pure-LF git credential fill (via Python) + curl.exe -L (strips Auth header on Azure 302 redirect!)
    if (-not $downloaded -and $PyExe) {
        $pyHelper = @"
import json, os, pathlib, shutil, subprocess, sys, urllib.request
repo, tag, asset_name, out_path = sys.argv[1:5]
tok = os.environ.get("GITHUB_TOKEN", "").strip()
if not tok:
    p = subprocess.run(
        ["git", "credential", "fill"],
        input=f"protocol=https\nhost=github.com\npath={repo}.git\n\n".encode("ascii"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("password="):
            tok = line.split("=", 1)[1].strip()
            break
if not tok:
    sys.exit(2)
req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
    headers={"Authorization": f"Bearer {tok}", "User-Agent": "PicoVid/0.5.3", "Accept": "application/vnd.github+json"},
)
rel = json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8"))
assets = [a for a in rel.get("assets", []) if a.get("name") == asset_name]
if not assets:
    sys.exit(3)
asset_url = assets[0]["url"]
curl = shutil.which("curl.exe") or shutil.which("curl")
if curl:
    r = subprocess.run([curl, "-L", "--fail", "-#", "-H", f"Authorization: Bearer {tok}", "-H", "Accept: application/octet-stream", asset_url, "-o", out_path])
    sys.exit(r.returncode)
# Fallback without curl: follow redirect manually without forwarding Authorization header
class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return urllib.request.Request(newurl, headers={"User-Agent": "PicoVid/0.5.3"})
opener = urllib.request.build_opener(_NoAuthRedirect())
dl_req = urllib.request.Request(asset_url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "PicoVid/0.5.3", "Accept": "application/octet-stream"})
with opener.open(dl_req, timeout=120) as resp, open(out_path, "wb") as fh:
    shutil.copyfileobj(resp, fh)
"@
        try {
            & $PyExe -c $pyHelper $ReleaseRepo $ReleaseTag $PrebuiltAsset $tmpZip
            if ($LASTEXITCODE -eq 0 -and (Test-Path $tmpZip) -and ((Get-Item $tmpZip).Length -gt 1000000)) {
                $downloaded = $true
            }
        } catch {}
    }

    # Method 3: Public direct download fallback
    if (-not $downloaded) {
        try {
            $pubUrl = "https://github.com/$ReleaseRepo/releases/download/$ReleaseTag/$PrebuiltAsset"
            & curl.exe -L --fail -# $pubUrl -o $tmpZip 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-Path $tmpZip) -and ((Get-Item $tmpZip).Length -gt 1000000)) {
                $downloaded = $true
            }
        } catch {}
    }

    if (-not $downloaded) {
        Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
        Write-Host "  [UYARI] Hazir binary indirilemedi (private repo ise 'git pull' kimligi veya 'gh auth login' gerekir)." -ForegroundColor Yellow
        return $false
    }

    Expand-Archive -Path $tmpZip -DestinationPath $DestBinDir -Force
    Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
    $exePath = Join-Path $DestBinDir "sd-cli.exe"
    if (Test-Path $exePath) {
        Write-Host "  [OK] Hazir sd-cli.exe indirildi ve kuruldu: $exePath" -ForegroundColor Green
        return $true
    }
    return $false
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PicoVid v0.5.1 · Sifirdan Otomatik Kurulum Sihirbazi (Windows)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$targetBackend = Detect-TargetBackend
$sdCliPath     = Join-Path $RepoRoot "build\sdcpp-$targetBackend\bin\sd-cli.exe"
$hasGit        = [bool](Get-Command git -ErrorAction SilentlyContinue)
$pyCmd         = Find-Python
$vcVars        = Find-VCVars
$sdkPath       = if ($targetBackend -eq "cuda") { Find-CudaNvcc } else { Find-VulkanSdk }
$hasBin        = (Test-Path $sdCliPath) -or (Test-Path (Join-Path $RepoRoot "build\sdcpp-vulkan\bin\sd-cli.exe"))
$hasRealesrgan = Test-Path (Join-Path $RepoRoot "tools\realesrgan\realesrgan-ncnn-vulkan.exe")

Write-Host ("  Hedef Backend       : " + $targetBackend.ToUpper() + $(if (-not $BuildFromSource -and $targetBackend -eq "vulkan") { " (Hazir binary — VS C++ gerekmez!)" } else { "" })) -ForegroundColor Green
Write-Host ("  [1] Git             : " + $(if ($hasGit) { "HAZIR" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [2] Python 3        : " + $(if ($pyCmd) { "HAZIR ($pyCmd)" } else { "EKSIK (kurulacak)" }))
Write-Host ("  [3] sd-cli.exe      : " + $(if ($hasBin) { "HAZIR" } elseif (-not $BuildFromSource -and $targetBackend -eq "vulkan") { "HAZIR BINARY INDIRILECEK (~30 MB)" } else { "KAYNAKTAN DERLENECEK" }))
Write-Host ("  [4] Cinema HD Tool  : " + $(if ($hasRealesrgan) { "HAZIR" } else { "INDIRILECEK (43 MB)" }))
Write-Host "=================================================================" -ForegroundColor Cyan

if ($CheckOnly) {
    Write-Host "[CHECK-ONLY] Donanim & on-kosul kontrolu tamamlandi." -ForegroundColor Yellow
    exit 0
}

# 1. Git
if (-not $hasGit) {
    Write-Host "`n[1/5] Git kuruluyor..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
    } else { throw "Git kurulu degil: https://git-scm.com/download/win" }
}

# 2. Python 3.12
if (-not $pyCmd) {
    Write-Host "`n[2/5] Python 3.12 kuruluyor..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
        $pyCmd = Find-Python
    }
    if (-not $pyCmd) { throw "Python kuruldu ama PATH guncellenemedi. Terminali kapatip yeniden acin ve setup.cmd'yi tekrar calistirin." }
}

# 3. Python packages
Write-Host "`n[3/5] Python paketleri (requirements.txt) kontrol ediliyor..." -ForegroundColor Cyan
& $pyCmd -m pip install --upgrade pip --quiet 2>$null
& $pyCmd -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt basarisiz oldu!" }

# 4. Get sd-cli.exe: try prebuilt binary FIRST (zero VS/C++/SDK needed!), fallback to local build
if (-not $hasBin) {
    $gotPrebuilt = $false
    if (-not $BuildFromSource -and $targetBackend -eq "vulkan") {
        $gotPrebuilt = Try-DownloadPrebuiltBinary -DestBinDir (Join-Path $RepoRoot "build\sdcpp-vulkan\bin") -PyExe $pyCmd
    }

    if (-not $gotPrebuilt) {
        Write-Host "`n[4/5] Hazir binary indirilemedi veya -BuildFromSource secildi; kaynaktan derleniyor..." -ForegroundColor Yellow
        if (-not $vcVars) {
            $vcVars = Install-VCToolsSafe
            if (-not $vcVars) {
                throw "Visual Studio C++ Build Tools (vcvars64.bat) bulunamadi. Visual Studio Installer -> Modify -> 'Desktop development with C++' secip tekrar deneyin."
            }
        }
        if (-not $sdkPath) {
            if ($targetBackend -eq "cuda") {
                if (Ensure-Winget) {
                    winget install -e --id Nvidia.CUDA --accept-package-agreements --accept-source-agreements
                    Refresh-EnvPath
                }
            } else {
                if (Ensure-Winget) {
                    winget install -e --id KhronosGroup.VulkanSDK --accept-package-agreements --accept-source-agreements
                    Refresh-EnvPath
                }
            }
        }
        $srcDir = Join-Path $RepoRoot "third_party\stable-diffusion.cpp"
        if (-not (Test-Path (Join-Path $srcDir ".git"))) {
            git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git $srcDir
            if ($LASTEXITCODE -ne 0) { throw "git clone basarisiz!" }
            git -C $srcDir fetch --depth 1 origin 3b12781 2>$null
            git -C $srcDir checkout 3b12781
        }
        powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1") -CheckOnly *> $null
        if ($LASTEXITCODE -eq 0) {
            powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1")
            if ($LASTEXITCODE -ne 0) { throw "apply_patches.ps1 basarisiz!" }
        }
        powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\build.ps1") -Backend $targetBackend
        if ($LASTEXITCODE -ne 0) { throw "build.ps1 basarisiz!" }
    }
} else {
    Write-Host "`n[4/5] sd-cli.exe zaten hazir." -ForegroundColor Green
}

# 4b. Install LTX-2.3 Vulkan prebuilt engine (29.6 MB, zero compilation needed!)
$ltxDir = Join-Path $RepoRoot "engine\sd-vulkan-ltx2"
$ltxExe = Join-Path $ltxDir "sd-cli.exe"
if (-not (Test-Path $ltxExe)) {
    Write-Host "`n[4b/5] LTX-2.3 Vulkan motoru indiriliyor (~30 MB, derleme gerekmez)..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $ltxDir | Out-Null
    $ltxZip = Join-Path $ltxDir "sd-ltx2-win-vulkan.zip"
    curl.exe -fL --retry 3 -sS -o $ltxZip $Ltx2PrebuiltUrl
    if ($LASTEXITCODE -eq 0 -and (Test-Path $ltxZip)) {
        Expand-Archive -Path $ltxZip -DestinationPath $ltxDir -Force
        Remove-Item $ltxZip -Force -ErrorAction SilentlyContinue
        Write-Host "  [OK] LTX-2.3 Vulkan motoru hazir: $ltxExe" -ForegroundColor Green
    } else {
        Write-Host "  [UYARI] LTX-2.3 Vulkan motoru indirilemedi; tekrar deneyebilirsiniz." -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[4b/5] LTX-2.3 Vulkan motoru zaten hazir." -ForegroundColor Green
}

# 5. Cinema HD Vulkan Upscaler (43 MB)
if (-not $hasRealesrgan) {
    Write-Host "`n[5/5] Cinema HD Vulkan araci (realesrgan-ncnn-vulkan, 43 MB) indiriliyor..." -ForegroundColor Cyan
    & $pyCmd (Join-Path $RepoRoot "scripts\postprocess.py") --install-tools
}

# 6. Models
if (-not $SkipModels) {
    Write-Host "`n[Modeller] Akilli model indirici baslatiliyor..." -ForegroundColor Cyan
    if ($Profile -eq "auto") {
        if ($Yes) {
            & $pyCmd (Join-Path $RepoRoot "scripts\download_models.py") --profile ltx23
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
