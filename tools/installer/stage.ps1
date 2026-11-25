# Copy the app to D:\VotionSplat_build\stage for the installer.
$ErrorActionPreference = "Stop"
$Root = "D:\Votion3DGS"
$Stage = "D:\VotionSplat_build\stage"
if (Test-Path $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$excludeDirs = @(
    "$Root\.git",
    "$Root\outputs",
    "$Root\.hf_cache",
    "$Root\_downloads",
    "$Root\_hy_fetch",
    "$Root\.pycache_tmp",
    "$Root\docs",
    "$Root\models",
    "$Root\Logo_and_Banners",
    "$Root\site_mockup",
    "$Root\unreal",
    "$Root\.venvs\wheels",
    "$Root\.venvs\src",
    "$Root\vendor\hyworld",
    "$Root\vendor\recastnavigation",
    "$Root\vendor\fused_ssim",
    "$Root\.venvs\hyworld\cuda_wheel",
    "$Root\.venvs\empty_ext",
    "__pycache__"
)
& robocopy $Root $Stage /MIR /XD $excludeDirs /XF *.log expand_fit.json hires_fit.json /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed with exit $LASTEXITCODE"
}
Get-ChildItem -LiteralPath $Stage -File -Filter *.png | Remove-Item -Force

$dirs = @(
    "outputs",
    ".hf_cache\hub",
    ".hf_cache\torch\hub\checkpoints",
    "models\diffusion_models",
    "models\text_encoders",
    "models\vae",
    "models\loras",
    "models\upscale_models",
    "models\MoGe",
    "models\SEEDVR2",
    "models\sam3",
    "models\dinov2-base",
    "models\clip_vision",
    "docs"
)
foreach ($rel in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Stage $rel) | Out-Null
}

Copy-Item -LiteralPath "$Root\models\sam3\sam3.pt" -Destination "$Stage\models\sam3\sam3.pt" -Force
Copy-Item -LiteralPath "$Root\unreal\dist\VotionSplat-UnrealPlugin" -Destination "$Stage\unreal_plugin\VotionSplat-UnrealPlugin" -Recurse -Force
Copy-Item -LiteralPath "$Root\docs\INSTALL.md" -Destination "$Stage\docs\INSTALL.md" -Force
Copy-Item -LiteralPath "$Root\docs\UNREAL_EXPORT.md" -Destination "$Stage\docs\UNREAL_EXPORT.md" -Force
Copy-Item -LiteralPath "$Root\THIRD_PARTY_NOTICES.md" -Destination "$Stage\docs\THIRD_PARTY_NOTICES.md" -Force
Copy-Item -LiteralPath "$Root\LICENSE" -Destination "$Stage\docs\LICENSE" -Force

$python = "$env:LOCALAPPDATA\Programs\Python\Python312"
if (-not (Test-Path "$python\python.exe")) {
    throw "Base Python 3.12 was not found at $python"
}
& robocopy $python "$Stage\runtime\python312" /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) {
    throw "Python copy failed with exit $LASTEXITCODE"
}

$bytes = (Get-ChildItem -LiteralPath $Stage -Recurse -File -Force | Measure-Object -Property Length -Sum).Sum
"{0:N1} GB staged at {1}" -f ($bytes / 1GB), $Stage
