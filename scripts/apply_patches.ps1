param(
    [string]$SourceDir = "third_party\stable-diffusion.cpp",
    [string]$PatchDir = "patches",
    [switch]$CheckOnly,
    [switch]$Reverse
)

$ErrorActionPreference = "Stop"

$seriesFile = Join-Path $PatchDir "SERIES.txt"
if (-not (Test-Path $seriesFile)) { throw "SERIES.txt not found at $seriesFile" }
if (-not (Test-Path $SourceDir)) { throw "Source directory not found: $SourceDir" }

$patches = Get-Content $seriesFile | ForEach-Object {
    $line = ($_ -replace '#.*$', '').Trim()
    if ($line) { $line }
}

if ($Reverse) {
    [array]::Reverse($patches)
}

Write-Host "Target repo : $SourceDir"
Write-Host "Patch count : $($patches.Count) (CheckOnly=$CheckOnly, Reverse=$Reverse)"

$flags = @()
if ($Reverse) { $flags += "-R" }

foreach ($p in $patches) {
    $patchPath = (Resolve-Path (Join-Path $PatchDir $p)).Path
    Write-Host "  [CHECK] $p"
    & git -C $SourceDir apply --check @flags $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "Patch check failed for $p"
    }
    if (-not $CheckOnly) {
        Write-Host "  [APPLY] $p"
        & git -C $SourceDir apply @flags $patchPath
        if ($LASTEXITCODE -ne 0) {
            throw "Patch apply failed for $p"
        }
    }
}

Write-Host "PATCH_SERIES_OK ($($patches.Count) patches verified)"
