[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$SkipModels,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Checking Python..." -ForegroundColor Cyan
$versionText = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $versionText.Split(".")
if ([int]$parts[0] -ne 3 -or [int]$parts[1] -lt 11 -or [int]$parts[1] -ge 14) {
    throw "Python 3.11, 3.12, or 3.13 is required. Found $versionText."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..." -ForegroundColor Cyan
    & $Python -m venv .venv
}
$VenvPython = (Resolve-Path ".venv\Scripts\python.exe").Path

Write-Host "Installing dependencies (this may take several minutes)..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip setuptools wheel
if (-not $CpuOnly -and $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "NVIDIA GPU detected; installing the locked CUDA 13.0 PyTorch build..." -ForegroundColor Cyan
    & $VenvPython -m pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
}
& $VenvPython -m pip install -r requirements-lock.txt
& $VenvPython -m pip install -e . --no-deps
& $VenvPython -m pip check

$ExifTool = Get-Command exiftool -ErrorAction SilentlyContinue
if ($null -eq $ExifTool) {
    Write-Warning "ExifTool was not found. Analysis works without it; XMP --write requires it."
    Write-Host "Download ExifTool without administrator rights: https://exiftool.org/" -ForegroundColor Yellow
} else {
    Write-Host "ExifTool: $($ExifTool.Source)" -ForegroundColor Green
}

if (-not $SkipModels) {
    Write-Host "Downloading YOLO and OpenCLIP model weights..." -ForegroundColor Cyan
    & $VenvPython scripts\download_models.py --config config\default.yaml
} else {
    Write-Warning "Model download skipped. analyze will remain unavailable until weights are downloaded."
}

Write-Host "Running installation verification..." -ForegroundColor Cyan
& $VenvPython scripts\verify_install.py

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate this environment in a new PowerShell:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then place references and previews under data\ and run:"
Write-Host "  python -m photo_sorter validate-config"
Write-Host "  python -m photo_sorter analyze --input data\input --output data\output\results.csv"
Write-Host "  streamlit run app.py"
