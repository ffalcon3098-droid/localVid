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
        Write-Host "[WARN] winget not found (App Installer). You may need to install missing tools manually." -ForegroundColor Yellow
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
                Write-Host "      Adding C++ Build Tools to existing Visual Studio ($instPath)..." -ForegroundColor Cyan
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
        Write-Host "      Installing Visual Studio 2022 Build Tools (C++ + CMake)..." -ForegroundColor Cyan
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

    Write-Host "`n[Fast Setup] Downloading prebuilt sd-cli.exe ($ReleaseTag · Vulkan — AMD/Intel/NVIDIA compatible)..." -ForegroundColor Cyan
    Write-Host "             (No Visual Studio / C++ / SDK compilation needed!)" -ForegroundColor Green

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
        Write-Host "  [WARN] Could not download prebuilt binary (for private repos, ensure git credentials or 'gh auth login')." -ForegroundColor Yellow
        return $false
    }

    Expand-Archive -Path $tmpZip -DestinationPath $DestBinDir -Force
    Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
    $exePath = Join-Path $DestBinDir "sd-cli.exe"
    if (Test-Path $exePath) {
        Write-Host "  [OK] Prebuilt sd-cli.exe downloaded and installed: $exePath" -ForegroundColor Green
        return $true
    }
    return $false
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PicoVid v0.7.0 · Zero-to-Ready Setup Wizard (Windows)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$targetBackend = Detect-TargetBackend
$sdCliPath     = Join-Path $RepoRoot "build\sdcpp-$targetBackend\bin\sd-cli.exe"
$hasGit        = [bool](Get-Command git -ErrorAction SilentlyContinue)
$pyCmd         = Find-Python
$vcVars        = Find-VCVars
$sdkPath       = if ($targetBackend -eq "cuda") { Find-CudaNvcc } else { Find-VulkanSdk }
$hasBin        = (Test-Path $sdCliPath) -or (Test-Path (Join-Path $RepoRoot "build\sdcpp-vulkan\bin\sd-cli.exe"))
$hasRealesrgan = Test-Path (Join-Path $RepoRoot "tools\realesrgan\realesrgan-ncnn-vulkan.exe")

Write-Host ("  Target Backend      : " + $targetBackend.ToUpper() + $(if (-not $BuildFromSource -and $targetBackend -eq "vulkan") { " (Prebuilt binary — no VS C++ needed!)" } else { "" })) -ForegroundColor Green
Write-Host ("  [1] Git             : " + $(if ($hasGit) { "READY" } else { "MISSING (will install)" }))
Write-Host ("  [2] Python 3        : " + $(if ($pyCmd) { "READY ($pyCmd)" } else { "MISSING (will install)" }))
Write-Host ("  [3] sd-cli.exe      : " + $(if ($hasBin) { "READY" } elseif (-not $BuildFromSource -and $targetBackend -eq "vulkan") { "PREBUILT DOWNLOAD (~30 MB)" } else { "BUILD FROM SOURCE" }))
Write-Host ("  [4] Cinema HD Tool  : " + $(if ($hasRealesrgan) { "READY" } else { "WILL DOWNLOAD (43 MB)" }))
Write-Host "=================================================================" -ForegroundColor Cyan

if ($CheckOnly) {
    Write-Host "[CHECK-ONLY] Hardware & prerequisite check complete." -ForegroundColor Yellow
    exit 0
}

# 1. Git
if (-not $hasGit) {
    Write-Host "`n[1/5] Installing Git..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
    } else { throw "Git is not installed: https://git-scm.com/download/win" }
}

# 2. Python 3.12
if (-not $pyCmd) {
    Write-Host "`n[2/5] Installing Python 3.12..." -ForegroundColor Cyan
    if (Ensure-Winget) {
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Refresh-EnvPath
        $pyCmd = Find-Python
    }
    if (-not $pyCmd) { throw "Python installed but PATH did not refresh. Reopen your terminal and run setup.cmd again." }
}

# 3. Python packages
Write-Host "`n[3/5] Checking Python packages (requirements.txt)..." -ForegroundColor Cyan
& $pyCmd -m pip install --upgrade pip --quiet 2>$null
& $pyCmd -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed!" }

# 4. Get sd-cli.exe: try prebuilt binary FIRST (zero VS/C++/SDK needed!), fallback to local build
if (-not $hasBin) {
    $gotPrebuilt = $false
    if (-not $BuildFromSource -and $targetBackend -eq "vulkan") {
        $gotPrebuilt = Try-DownloadPrebuiltBinary -DestBinDir (Join-Path $RepoRoot "build\sdcpp-vulkan\bin") -PyExe $pyCmd
    }

    if (-not $gotPrebuilt) {
        Write-Host "`n[4/5] Prebuilt binary unavailable or -BuildFromSource requested; building from source..." -ForegroundColor Yellow
        if (-not $vcVars) {
            $vcVars = Install-VCToolsSafe
            if (-not $vcVars) {
                throw "Visual Studio C++ Build Tools (vcvars64.bat) not found. Open Visual Studio Installer -> Modify -> select 'Desktop development with C++' and retry."
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
            if ($LASTEXITCODE -ne 0) { throw "git clone failed!" }
            git -C $srcDir fetch --depth 1 origin 3b12781 2>$null
            git -C $srcDir checkout 3b12781
        }
        powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1") -CheckOnly *> $null
        if ($LASTEXITCODE -eq 0) {
            powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\apply_patches.ps1")
            if ($LASTEXITCODE -ne 0) { throw "apply_patches.ps1 failed!" }
        }
        powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\build.ps1") -Backend $targetBackend
        if ($LASTEXITCODE -ne 0) { throw "build.ps1 failed!" }
    }
} else {
    Write-Host "`n[4/5] sd-cli.exe is already installed." -ForegroundColor Green
}

# 4b. Install LTX-2.3 Vulkan prebuilt engine (29.6 MB, zero compilation needed!)
$ltxDir = Join-Path $RepoRoot "engine\sd-vulkan-ltx2"
$ltxExe = Join-Path $ltxDir "sd-cli.exe"
if (-not (Test-Path $ltxExe)) {
    Write-Host "`n[4b/5] Downloading LTX-2.3 Vulkan engine (~30 MB, zero compilation needed)..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $ltxDir | Out-Null
    $ltxZip = Join-Path $ltxDir "sd-ltx2-win-vulkan.zip"
    curl.exe -fL --retry 3 -sS -o $ltxZip $Ltx2PrebuiltUrl
    if ($LASTEXITCODE -eq 0 -and (Test-Path $ltxZip)) {
        Expand-Archive -Path $ltxZip -DestinationPath $ltxDir -Force
        Remove-Item $ltxZip -Force -ErrorAction SilentlyContinue
        Write-Host "  [OK] LTX-2.3 Vulkan engine ready: $ltxExe" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Could not download LTX-2.3 Vulkan engine; you can retry later." -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[4b/5] LTX-2.3 Vulkan engine is already installed." -ForegroundColor Green
}

# 5. Cinema HD Vulkan Upscaler (43 MB)
if (-not $hasRealesrgan) {
    Write-Host "`n[5/5] Downloading Cinema HD Vulkan tool (realesrgan-ncnn-vulkan, 43 MB)..." -ForegroundColor Cyan
    & $pyCmd (Join-Path $RepoRoot "scripts\postprocess.py") --install-tools
}

# 6. Models
if (-not $SkipModels) {
    Write-Host "`n[Models] Starting smart model downloader..." -ForegroundColor Cyan
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
        Write-Host "[WARN] Model download encountered an issue; resume anytime with 'py scripts\download_models.py'." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE! PicoVid is ready to use:" -ForegroundColor Green
Write-Host "    Terminal Wizard : .\picovid.cmd" -ForegroundColor White
Write-Host "    Web Studio      : .\web-studio.cmd  (or .\picovid.cmd --web)" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
