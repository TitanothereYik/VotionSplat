# Changelog

## Unreleased

- Home's model cards keep their full height and scroll on short windows.
- README shows all six pages at the same size and links VotionSplat-UnrealPlugin as the way to get the best result in Unreal.
- HY-World 2.0 carries its required NOTICE, and THIRD_PARTY_NOTICES names its license.

## 3.0.1

- Align and Expand stay offline: DINOv2 loads from `models/dinov2-base`, and the Wan VAE config is a core Home download.
- The install check fails if a venv still points at the build machine's Python.
- The desktop shortcut is a choice in the installer. Uninstall removes the app, the environments, and the Python caches, and keeps `models`, `outputs`, and `.hf_cache`.
- The installer ships `docs/unreal/VS_OCIO.ocio` and the SAM License.
- THIRD_PARTY_NOTICES lists SAM 3 and the downloaded model weights, plus the GPL note for the installer.
- Panorama, Home, and log labels no longer show internal graph and phase names.
- The top banner keeps a fixed 150 px height. On wide windows the space past the art is black.
- README and the user guide show real app screenshots.

## 3.0.0

- The app is VotionSplat 3.0.0. The repository is TitanothereYik/VotionSplat.
- Export keeps Unreal / VotionSplat, sRGB, and linear. The older Unreal presets are gone.
- The old trainer, geometry, reconstruct, and depth bake-off paths are removed.
- Home downloads the models the current pipeline uses. SAM 3 ships in the installer. Other models download on first run.
- Company installs use the Inno Setup package: Python 3.12, four environments, and the prebuilt CUDA extensions.
- The window icon, the top-of-window banner, and the GitHub and installer banner are the finished art.
