param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("moge2", "pager", "dap")]
    [string]$Name
)

$ErrorActionPreference = "Stop"
# The machine pip.ini adds an NVIDIA index that may not resolve; these venvs only need PyPI.
$env:PIP_CONFIG_FILE = "nul"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Venv = Join-Path $Repo ".venvs\depth_$Name"
$Src = Join-Path $Repo ".venvs\src"
New-Item -ItemType Directory -Force -Path $Src | Out-Null

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

# PaGeR pins python < 3.13; one 3.12 lets all three share a torch wheel.
$Base = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Invoke-Checked $Base @("-m", "venv", $Venv)
}
$Py = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
# pip can stall on the 2.7 GB torch wheel; a copy fetched with curl into .venvs\wheels is used first.
$Wheel = Get-ChildItem (Join-Path $Repo ".venvs\wheels") -Filter "torch-*+cu128-cp312-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
$Torch = if ($Wheel) { $Wheel.FullName } else { "torch" }
Invoke-Checked $Py @("-m", "pip", "install", $Torch, "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")

$repos = @{
    moge2 = "https://github.com/microsoft/MoGe.git"
    pager = "https://github.com/prs-eth/PaGeR.git"
    dap   = "https://github.com/Insta360-Research-Team/DAP.git"
}
$dest = Join-Path $Src $Name
if (-not (Test-Path (Join-Path $dest ".git"))) {
    Invoke-Checked "git" @("clone", "--depth", "1", $repos[$Name], $dest)
}
$sha = (git -C $dest rev-parse HEAD).Trim()
Set-Content -Path (Join-Path $Src "$Name.sha") -Value $sha -Encoding ascii

switch ($Name) {
    "moge2" {
        # flex-gemm is MoGe-3 only and needs a CUDA build; gradio is the demo.
        Invoke-Checked $Py @("-m", "pip", "install", "--no-deps", "-e", $dest)
        Invoke-Checked $Py @("-m", "pip", "install",
            "click", "numpy>=2", "opencv-python-headless", "scipy>=1.8", "pillow", "matplotlib",
            "trimesh", "huggingface-hub", "requests", "tqdm",
            "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37")
    }
    "pager" {
        Invoke-Checked $Py @("-m", "pip", "install", "-e", $dest)
    }
    "dap" {
        # DAP has no package; the runner imports it from the repo root.
        Invoke-Checked $Py @("-m", "pip", "install",
            "einops", "pyyaml", "opencv-python", "pillow", "numpy", "matplotlib", "huggingface_hub",
            "ftfy", "omegaconf", "regex", "scikit-learn", "termcolor", "torchmetrics")
    }
}

$licence = Get-ChildItem $dest -File | Where-Object { $_.Name -match "LICENSE" }
foreach ($file in $licence) {
    Write-Output ("licence file: " + $file.FullName)
    Get-Content $file.FullName -TotalCount 4
}
Write-Output "installed $Name at $sha"
