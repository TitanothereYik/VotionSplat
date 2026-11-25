# Votion 3DGS

Native Windows app that turns a picture, a 360 image, or a text prompt into a 3D Gaussian splat. Version **2.0**. It is a desktop program (PySide6), not a browser UI and not a ComfyUI graph.

Owner: Yik. License: [MIT](LICENSE). No telemetry. The first audience is a small private team.

## Pipeline

```
2D image (default), 360 image, or text
  → 8K equirectangular panorama
  → MoGe geometry and Plot Camera rails
  → WAN 2.1 hole-fill (720p) on those rails
  → 8K HiRes composite (WAN only in the holes)
  → SphereSfM, then gnomonic cube faces
  → train the splat
  → splat.ply (full spherical harmonics)
```

Quality profile targets an RTX 3090 (24 GB). Fast and Scout ship as profiles. The window has six stages: Home, Panorama, Geometry, Generate, Reconstruct, Splat.

The UI dummy is at [titanothereyik.github.io/votion-3dgs-2-html](https://titanothereyik.github.io/votion-3dgs-2-html/). Product rules live in [docs/Votion_3DGS_2.0_Index.md](docs/Votion_3DGS_2.0_Index.md) and [PRD/README.md](PRD/README.md).

## Run

Windows, Python **3.12**, CUDA **12.8**. Do not point this app at another project's embedded Python.

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements.lock` records the stack that ran on the dev machine, including `torch==2.11.0+cu128` and `PySide6==6.11.2`. The `+cu128` wheels come from the PyTorch index above.

The HY-World train environment is `.venvs\hyworld`, built with `tools\hyworld\install.ps1`.

Start the window:

```bat
tools\launch.bat
```

That sets the Hugging Face cache to `.hf_cache` in this folder (not the user profile cache) and launches `python -m app.main`. On Home, **Download models** fills `models\` from `tools/model_manifest.json`. The Quality pack is on the order of 45–60 GB.

## What is not in git

Weights (`models\`), scene outputs (`outputs\`), the hub cache (`.hf_cache\`), and the virtualenv (`.venv\`) stay on disk. Vendored libraries are pinned in [tools/vendor_pins.json](tools/vendor_pins.json) and checked out under `vendor\` on the machine that builds them.

The rollback tag for this snapshot is `snapshot/pre-worldstereo`.
