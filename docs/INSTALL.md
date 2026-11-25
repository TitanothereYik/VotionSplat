# Install VotionSplat

## Requirements

- Windows 10 or 11, 64-bit
- NVIDIA RTX 30, 40, or 50 series, or RTX A5000 or A6000
- 24 GB of GPU memory recommended
- NVIDIA driver 570 or newer
- 150 GB free on the install drive (the app, the environments, and the models)

## Install

1. Run `VotionSplat-3.0.1-Setup` from the company share. The share path is not fixed in this file yet.
2. Install to `C:\VotionSplat`, or another folder you can write to. Do not use Program Files. The app stores models and outputs inside its own folder.
3. The setup installs the VC++ runtime, points the Python environments at the bundled Python, and writes `install_check.log`.
4. Open VotionSplat from the Start Menu.

## First run

Home shows the model cards. Choose **Core** and press **Download models**. Cards turn green as files arrive. SAM 3 is already installed. Optional Difix and Florence-2 are the second choice.

A full scene then follows the [user guide](USER_GUIDE.md): Panorama, Trajectories, Expand, Dataset, Splat, then export.

## Uninstall

Uninstall removes the app and the Start Menu entries. It leaves `models`, `outputs`, and `.hf_cache` in the install folder. Those are your downloads and scenes. Delete that folder yourself if you want them gone.
