# Stage the app and compile VotionSplat-3.0.0-Setup.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Votion3DGS"
& .venv\Scripts\python.exe tools\make_brand_assets.py
if ($LASTEXITCODE -ne 0) { throw "make_brand_assets.py failed" }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\installer\stage.ps1
if ($LASTEXITCODE -ne 0) { throw "stage.ps1 failed" }
$iscc = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup is not installed. Run: winget install --id JRSoftware.InnoSetup -e"
}
& $iscc tools\installer\VotionSplat.iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit $LASTEXITCODE" }
"Installer written to D:\VotionSplat_build\Output"
