# Downloads the OpenFace 2.2.0 Windows binary and its CE-CLM landmark models into tools/.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\setup_openface.ps1

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster

$tools = Join-Path $PSScriptRoot "..\tools"
$dest = Join-Path $tools "OpenFace_2.2.0_win_x64"
$url = "https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip"

if (-not (Test-Path (Join-Path $dest "FeatureExtraction.exe"))) {
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = Join-Path $tools "openface.zip"
    Write-Host "Downloading OpenFace 2.2.0..."
    Invoke-WebRequest $url -OutFile $zip
    Expand-Archive $zip $tools -Force
    Remove-Item $zip
}

if (-not (Test-Path (Join-Path $dest "model\patch_experts\cen_patches_1.00_of.dat"))) {
    Write-Host "Downloading landmark models (~400 MB)..."
    Push-Location $dest
    try { & powershell -ExecutionPolicy Bypass -File .\download_models.ps1 } finally { Pop-Location }
}

Write-Host "OpenFace ready at $dest"
