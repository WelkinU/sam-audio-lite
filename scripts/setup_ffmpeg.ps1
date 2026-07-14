# Download FFmpeg "gpl-shared" for TorchCodec on Windows (full-shared DLL build).
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DestRoot = Join-Path $ProjectRoot "tools\ffmpeg"
$ZipPath = Join-Path $DestRoot "ffmpeg-shared.zip"
$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"

New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

$BinDir = Get-ChildItem -Path $DestRoot -Recurse -Filter "avcodec-*.dll" -ErrorAction SilentlyContinue |
    Select-Object -First 1 |
    ForEach-Object { $_.Directory.FullName }

if ($BinDir) {
    Write-Host "       FFmpeg shared build already present at $BinDir"
    exit 0
}

Write-Host "       Downloading FFmpeg shared build (required by TorchCodec on Windows)..."
Invoke-WebRequest -Uri $Url -OutFile $ZipPath
Expand-Archive -Path $ZipPath -DestinationPath $DestRoot -Force
Remove-Item $ZipPath -Force

$BinDir = Get-ChildItem -Path $DestRoot -Recurse -Filter "avcodec-*.dll" |
    Select-Object -First 1 |
    ForEach-Object { $_.Directory.FullName }

if (-not $BinDir) {
    Write-Error "FFmpeg shared DLLs were not found after extraction."
}
Write-Host "       FFmpeg shared build installed at $BinDir"
