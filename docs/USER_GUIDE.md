# VotionSplat user guide

Work left to right. The scene name is the folder under `outputs`.

## Home

Download models, then pick a CUDA device. **Core** is required (about 78 GB). **Core + optional** adds Difix and Florence-2 captions. **Download models** fills anything that is still missing. SAM 3 is already in the install.

## Panorama

Makes `outputs\<scene>\pano.png` (8192×4096) and `outputs\<scene>\prompt.txt`.

- **2D Images:** load a photo, **Detect** the field of view, type the surroundings or **Caption** with Florence-2, then **Generate**.
- **360 Images:** load a 2:1 panorama. **Use as pano.png** keeps it, or **Upscale to 8K**.
- **text:** write the scene and **Generate**.

## Trajectories

Plans walks on the panorama, then paints a guide for each kept path.

- Scene type: **Auto**, **Outdoor**, or **Indoor**.
- **Navigation paths**, **Upward routes**, **Reconstruction loops**.
- **Start over** archives the previous plan.
- **Plan trajectories**, then **Render guides**.

Files land in `outputs\<scene>\hyworld\render_results`.

## Expand

Fills each kept guide with WorldStereo, then can upscale it.

- **Profile first path** times one 21-frame sample.
- **Expand kept paths** runs the list.
- **Regenerate selected** runs one path again.
- **Align** fits the frames to the panorama depth.
- **HiRes selected** and **HiRes all** upscale with SeedVR2. Scale is **2x (1664x960)** or **4x (3328x1920)**.
- **GPU memory** and **HiRes memory** choose how much of the model stays on the card.

Clips stay next to the guides. HiRes frames stay under `outputs\<scene>\hyworld\_votion`.

## Dataset

**Build dataset** writes training images, depth, and normals to `outputs\<scene>\hyworld\gs_data`. **Panorama density** and **Upper polar density** control how many extra views come from the panorama.

## Splat

Trains the Gaussian splat from that dataset.

- **Train**, **Continue training**, **Pause**, **Stop**.
- **Difix3D+ refine** is optional and uses the Difix model.
- **Export splat.ply** writes `outputs\<scene>\hyworld\splat.ply`. The default color space is **Unreal / VotionSplat Export**. The other choices are **3DGS / sRGB** and **Linear Rec.709**.
- **Open result folder** opens `outputs\<scene>\hyworld\gs_result`.

Screenshots: [app-01](media/screenshots/app-01.png), [app-02](media/screenshots/app-02.png), [app-03](media/screenshots/app-03.png).
