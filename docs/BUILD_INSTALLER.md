# Build the installer

Run this from the repo root on the build machine. The staged tree and the setup files go to `D:\VotionSplat_build`, outside the repo. Expect about an hour for the native rebuild and one to three hours for compression.

## 1. Native extensions

`tools\hyworld\build_native.py` sets `TORCH_CUDA_ARCH_LIST` to `8.6;8.9;12.0+PTX` (RTX 30 / A5000 / A6000, RTX 40, RTX 50).

```
.venvs\hyworld\Scripts\python.exe tools\hyworld\build_native.py
```

Check both compiled files with CUDA 12.8 `cuobjdump --list-elf`. Each must list `sm_86`, `sm_89`, and `sm_120`:

- `.venvs\hyworld\Lib\site-packages\gsplat\csrc.pyd`
- `.venvs\hyworld\Lib\site-packages\fused_ssim_cuda.cp312-win_amd64.pyd`

## 2. No-toolkit check

Set `VOTIONSPLAT_NO_DEV_TOOLS=1`, clear `CUDA_PATH` and `CUDA_HOME`, drop CUDA Toolkit and Visual Studio from `PATH`, and set `TORCH_EXTENSIONS_DIR` to an empty folder. Run one small scene through Panorama, Trajectories, one Expand path, Dataset, 500 splat steps with Difix, and Export. A request for `nvcc`, `cl.exe`, or `ninja` means something still compiles at runtime.

## 3. Stage and compile

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\installer\build_installer.ps1
```

That script rebuilds the icon, runs `tools\installer\stage.ps1`, then Inno Setup. The setup is `D:\VotionSplat_build\Output\VotionSplat-3.0.0-Setup.exe` plus the spanned `.bin` parts. Do not upload it to GitHub. Put the Output folder on the company share.

Install once to `D:\VotionSplat_test` and read `install_check.log`. Every line should be PASS.

## Icon and banners

`tools\make_brand_assets.py` copies the finished art when `Logo_and_Banners` is present:

- `votion-logo-091-rounded.png` is only the icon: `app\assets\icon.png`, `icon.ico`, the window, and the shortcuts.
- `VotionSplat_Banner_Long.png` becomes `app\assets\banner.png`, full width at the top of the window.
- `votion-banner-seed-2.png` becomes `app\assets\repo_banner.png` for the GitHub page. The installer welcome panel uses that same banner scaled down so the whole image fits.

Run `.venv\Scripts\python.exe tools\make_brand_assets.py`, commit `app\assets`, and rebuild the installer.
