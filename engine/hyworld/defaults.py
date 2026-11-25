"""Numbers for the HY-World route. The Qt process may import this."""

from __future__ import annotations

from pathlib import Path

SAM3_ROOT = Path(r"D:\sam3\sam3")
SAM3_CHECKPOINT = Path(r"D:\sam3\checkpoints\sam3.pt")
PIN = "df9988efb87bfc0f4947eb3889411cf957478b06"

HELP_TRAJECTORIES = """TRAJECTORIES
Plan walks the panorama with HY-World. The chips across the top fill in as each step finishes: the panorama, the sky mask, the navigation map, any objects, then the path map. Render guides paints the kept paths. The Guide chip plays the start frame, the guide, and the mask of the selected row.
Scene type Auto asks the local vision model. Indoor and Outdoor skip that question. Navigation, upward routes, and reconstruction loops are on unless you clear them.
Extra views per kind keeps every origin view and the first N views of each other kind, with all of that view's paths. Clear a row to leave it out of Expand. Start over archives the old plan. Nothing is deleted.
"""

HELP_EXPAND = """EXPAND
Expand fills each kept guide with WorldStereo, one path at a time, using the scaled FP8 model. The queue shows retrieve, sample, and done, with seconds and peak memory. The four clips are the guide, the mask, the memory frames, and the result. Align then fits those frames to the panorama depth and shows a top view. WorldMirror stays off.
Profile first path runs a single 21-frame sample so the page can estimate the rest. Regenerate selected runs one path again. A full scene is that sample multiplied by the kept list.
GPU memory picks how much of the model waits in system RAM. Auto reads this card and this RAM, picks the lightest level that fits, and steps down one level if a sample still runs out. The rail then shows the card and the level in use. Expand needs about 11 GB of free card memory at the lightest level. What each level needed is saved in models/expand_fit.json and used next time.
HiRes all upscales every expanded path with SeedVR2 3B. HiRes scale picks 2x (1664x960, saved under _votion/hires) or 4x (3328x1920, saved under _votion/hires_x4); the two are kept apart, and the list and HiRes chip show the scale picked. HiRes selected redoes one path. The HiRes chip plays the result beside the original. HiRes memory picks how the work is split: at 2x the VAE is split into tiles, and at 4x the 21 frames go through the model in smaller overlapping batches, because all 21 at once do not fit a 24 GB card. Auto picks the largest setting that fits this card and steps down if a clip runs out. 4x takes about 3 to 4 minutes per path on a 3090. What each setting needed is saved in models/hires_fit.json. A path whose Expand result changed shows +HR old and is redone by HiRes all. To train on these frames, pick them under Training frames on the Splat page once every path in the dataset has them.
"""

HELP_DATASET = """DATASET
Dataset turns the aligned frames into the training folder: images, depth, normals, cameras, and the point clouds. The panorama and polar views are added so the top and bottom of the sphere are covered.
Image, Depth, and Normal browse the folder. The slider steps through the current group.
"""

HELP_SPLAT = """SPLAT
Train fits a Gaussian splat to the dataset on this GPU and always starts over. Training frames picks the Expand frames or, when every path has them, the HiRes 2x or 4x frames; the panorama and polar views keep the Expand size, and larger frames make each step slower. Continue training picks up the last run after Stop or a crash, from the last save, which is at most 10 minutes back, and it keeps that run's step count. Eight thousand steps is the one-GPU length. The viewport updates while it trains. Pause holds the step and the view still moves. Stop saves a checkpoint first; press Stop again to kill at once. After the run the same view stays open on the newest checkpoint.
Difix3D+ refine is off unless you turn it on. It renders views between the paths, cleans them with NVIDIA Difix, and trains on those frames. The NVIDIA licence is research and evaluation only, not for commercial use. A run does this every 2000 steps and each round adds about 2 minutes.
Export writes hyworld/splat.ply in the panorama's camera frame. The mesh checkbox fuses a surface at the end and is off unless you turn it on.
"""
