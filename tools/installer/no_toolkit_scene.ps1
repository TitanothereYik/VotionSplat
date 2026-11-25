# One small scene with no CUDA Toolkit and no Visual Studio on PATH.
$ErrorActionPreference = "Stop"
$Root = "D:\Votion3DGS"
Set-Location -LiteralPath $Root
$env:VOTIONSPLAT_NO_DEV_TOOLS = "1"
$env:PYTHONPATH = $Root
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HOME = "$Root\.hf_cache"
$env:HF_HUB_CACHE = "$Root\.hf_cache\hub"
$env:HF_XET_CACHE = "$Root\.hf_cache\xet"
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:TORCH_EXTENSIONS_DIR = "$Root\.venvs\empty_ext"
Remove-Item Env:\CUDA_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\CUDA_HOME -ErrorAction SilentlyContinue
$env:PATH = (($env:PATH -split ";") | Where-Object {
    $_ -and ($_ -notmatch "NVIDIA GPU Computing Toolkit") -and ($_ -notmatch "Microsoft Visual Studio")
}) -join ";"
"PATH_NVCC " + ((Get-Command nvcc -ErrorAction SilentlyContinue).Source)
"PATH_CL " + ((Get-Command cl -ErrorAction SilentlyContinue).Source)

$scene = "gate_30"
$hy = "$Root\outputs\$scene\hyworld"
$py = "$Root\.venv\Scripts\python.exe"
$hyPy = "$Root\.venvs\hyworld\Scripts\python.exe"
$run = "$Root\tools\hyworld\run_stage.py"

function Invoke-Step([string]$Name, [string]$Exe, [string[]]$StepArgs) {
    "STATUS // STEP $Name"
    & $Exe @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "STEP $Name failed with exit $LASTEXITCODE"
    }
    "STATUS // STEP $Name exit 0"
}

if (Test-Path -LiteralPath "$Root\outputs\$scene\pano.png") {
    "STATUS // STEP panorama exists, skip"
} else {
    Invoke-Step "panorama" $py @(
        "-m", "engine.workers.pano", "--mode", "text", "--action", "generate",
        "--scene", $scene, "--sentence", "a quiet alpine meadow", "--hfov", "70"
    )
}
$camera = "$hy\render_results\view0\traj0\camera.json"
if (Test-Path -LiteralPath $camera) {
    "STATUS // STEP plan exists, skip"
    Set-Content -LiteralPath "$Root\.venvs\notk_only.txt" -Value "view0/traj0" -Encoding ascii
} else {
    Invoke-Step "plan" $hyPy @(
        $run, "--stage", "traj_generate", "--scene", $scene, "--scene-type", "auto", "--",
        "--target_path", $hy, "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration",
        "--force_vlm", "--skip_exist"
    )
    & $py "$Root\tools\installer\keep_one_traj.py" $scene
    if ($LASTEXITCODE -ne 0) { throw "selection failed" }
}
$only = (Get-Content -LiteralPath "$Root\.venvs\notk_only.txt" -Raw).Trim()
$guide = "$hy\render_results\view0\traj0\render.mp4"
if (Test-Path -LiteralPath $guide) {
    "STATUS // STEP render exists, skip"
} else {
    Invoke-Step "render" $hyPy @($run, "--stage", "traj_render", "--", "--target_path", $hy)
}
$expanded = "$hy\render_results\view0\traj0\worldstereo-memory-dmd_result.mp4"
if (Test-Path -LiteralPath $expanded) {
    "STATUS // STEP expand exists, skip"
} else {
    Invoke-Step "expand" $py @(
        "-m", "engine.workers.hy_expand", "--scene", $scene, "--only", $only, "--level", "blocks"
    )
}
$aligned = "$hy\render_results\generation_bank_worldstereo-memory-dmd\aligned_pcd.ply"
if (Test-Path -LiteralPath $aligned) {
    "STATUS // STEP align exists, skip"
} else {
    Invoke-Step "align" $hyPy @($run, "--stage", "align_bank", "--", "--target_path", $hy)
}
Invoke-Step "dataset" $hyPy @(
    $run, "--stage", "gen_gs_data", "--", "--root_path", $hy,
    "--save_normal", "--split_sky", "--pano_density_mult", "2", "--polar_up_density_mult", "4"
)
Invoke-Step "splat" $py @(
    "-m", "engine.workers.hy_train_loop", "--scene", $scene, "--steps", "500", "--difix"
)
Invoke-Step "export" $py @(
    "-m", "engine.workers.hy_export", "--scene", $scene, "--color-space", "votion"
)
"STATUS // NOTK DONE"
