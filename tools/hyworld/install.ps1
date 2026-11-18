# Build .venvs\hyworld from the pinned worldgen copy.
# Usage: powershell -File tools\hyworld\install.ps1
$ErrorActionPreference = "Stop"
$env:PIP_CONFIG_FILE = "nul"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Venv = Join-Path $Repo ".venvs\hyworld"
$World = Join-Path $Venv "worldgen"
$Py = Join-Path $Venv "Scripts\python.exe"
$Log = Join-Path $Repo ".venvs\install_hyworld.log"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    Write-Output (">> " + $Exe + " " + ($Arguments -join " "))
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

if (-not (Test-Path (Join-Path $Repo "vendor\hyworld\traj_generate.py"))) {
    throw "vendor\hyworld is missing. Extract the pinned tarball first."
}
if (-not (Test-Path $Py)) {
    $Base = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
    Invoke-Checked $Base @("-m", "venv", $Venv)
}
Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "ninja")

$Wheel = Get-ChildItem (Join-Path $Repo ".venvs\wheels") -Filter "torch-*+cu128-cp312-*.whl" | Select-Object -First 1
if (-not $Wheel) { throw "cu128 torch wheel is missing under .venvs\wheels" }
Invoke-Checked $Py @("-m", "pip", "install", $Wheel.FullName, "torchvision==0.26.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128")

$Constraints = Join-Path $PSScriptRoot "constraints.txt"
$pkgs = @(
    "numpy==1.26.4", "scipy==1.14.1", "opencv-python==4.10.0.84", "scikit-image==0.25.2", "scikit-learn",
    "matplotlib==3.10.3", "transformers==5.2.0", "diffusers==0.36.0", "peft==0.18.1", "accelerate", "safetensors", "omegaconf", "einops",
    "imageio[ffmpeg]", "imagesize", "decord", "trimesh", "rtree", "embreex", "plyfile", "open3d==0.19.0",
    "pymeshlab==2023.12.post2", "cupy-cuda12x==13.6.0", "timm>=1.0.17", "kornia", "easydict", "torchmetrics",
    "loguru", "tqdm", "tyro==1.0.8", "viser", "tensorboard", "splines", "jaxtyping", "pybind11",
    "onnxruntime-gpu", "zim_anything", "huggingface_hub", "qwen-vl-utils", "iopath", "ftfy==6.1.1",
    "click", "pyyaml", "rich", "triton-windows>=3.7,<3.8", "pycocotools"
)
Invoke-Checked $Py (@("-m", "pip", "install", "-c", $Constraints) + $pkgs)

if (Test-Path $World) { Remove-Item -Recurse -Force $World }
Copy-Item -Recurse (Join-Path $Repo "vendor\hyworld") $World
Invoke-Checked $Py @((Join-Path $PSScriptRoot "apply_patches.py"), $World)

Invoke-Checked $Py @("-m", "pip", "install", "--no-deps", "git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6")
Invoke-Checked $Py @("-m", "pip", "install", "--no-deps", "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38")
$P3dName = "pytorch3d-0.7.9+d9839a9pt2.11.0cu128-cp312-cp312-win_amd64.whl"
$P3d = Join-Path $Repo ".venvs\wheels\$P3dName"
if (-not (Test-Path $P3d)) {
    Invoke-Checked "curl.exe" @("-L", "--retry", "3", "-o", $P3d, "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9%2Bd9839a9/$P3dName")
}
Invoke-Checked $Py @("-m", "pip", "install", $P3d)
Invoke-Checked $Py @("-m", "pip", "install", "--no-deps", "git+https://github.com/nerfstudio-project/nerfview.git@4538024fe0d15fd1a0e4d760f3695fc44ca72787")
Invoke-Checked $Py @( (Join-Path $PSScriptRoot "build_native.py") )
Write-Output "hyworld venv ready: $Py"
