# Import the live pipeline in the installed app. Writes install_check.log.
param(
    [Parameter(Mandatory = $true)][string]$AppDir
)
$ErrorActionPreference = "Continue"
$log = New-Object System.Collections.Generic.List[string]
$env:PYTHONPATH = $AppDir
$env:VOTIONSPLAT_NO_DEV_TOOLS = "1"
# Each venv must run on the bundled Python. A build machine still has the original Python, so imports alone would pass.
$env:VOTIONSPLAT_PY_HOME = Join-Path $AppDir "runtime\python312"
$onBundled = "import os, sys; home = os.environ['VOTIONSPLAT_PY_HOME']; assert os.path.samefile(sys.base_prefix, home), sys.base_prefix; "

function Add-Check([string]$Name, [string]$Exe, [string]$Code) {
    & $Exe -c ($onBundled + $Code)
    if ($LASTEXITCODE -eq 0) {
        $script:log.Add("PASS $Name")
    } else {
        $script:log.Add("FAIL $Name")
    }
}

Add-Check "venv" "$AppDir\.venv\Scripts\python.exe" "import torch, PySide6, diffusers, huggingface_hub; assert torch.cuda.is_available()"
Add-Check "hyworld" "$AppDir\.venvs\hyworld\Scripts\python.exe" "import torch, gsplat, fused_ssim, recast, pytorch3d, cupy; assert torch.cuda.is_available()"
Add-Check "seedvr2" "$AppDir\.venvs\seedvr2\Scripts\python.exe" "import torch, diffusers; assert torch.cuda.is_available()"
Add-Check "difix" "$AppDir\.venvs\difix\Scripts\python.exe" "import torch, diffusers; assert torch.cuda.is_available()"

$gpu = & nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
if ($LASTEXITCODE -eq 0) {
    $log.Add("PASS nvidia-smi $gpu")
} else {
    $log.Add("FAIL nvidia-smi")
}

$path = Join-Path $AppDir "install_check.log"
[IO.File]::WriteAllLines($path, $log, [Text.UTF8Encoding]::new($false))
$log | ForEach-Object { $_ }
if ($log | Where-Object { $_.StartsWith("FAIL") }) {
    exit 1
}
exit 0
