# Votion Splat plugin — validation log

## Phase 0

| Check | Result |
| --- | --- |
| Unreal Editor 5.5 | `C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\Win64\UnrealEditor.exe` |
| Build.bat / RunUAT.bat | present |
| `.venv` | present, PyOpenColorIO 2.6.0 |
| `.venvs\hyworld` | present, gsplat 1.5.3, CUDA available |
| OCIO config | `docs/unreal/VS_OCIO.ocio` (Linear Rec.709 (sRGB), sRGB - Display, ACES 1.0 - SDR Video) |
| Scenes | `outputs\alpine_01\splat.ply`, `outputs\alpine_02\splat.ply`, `outputs\alpine_02\hyworld\splat.ply` |

HY export uses `.venv\Scripts\python.exe` (`engine/hyworld/argv.py` `export_argv`). It was not run against `outputs/`, so the existing `splat.ply` files are unchanged.

## Phase 1

- `ply_export` self-test: pass (`ply_export self-test OK`).
- New colour space `votion` writes `votion.version`, `votion.raster`, `votion.near_plane`, `votion.sh_degree`.
- HY export skips soften for that preset. LiteGS export tags `litegs_classic` and near plane 0.2.
- HY viewers round 8-bit values (`+ 0.5`) instead of truncating.

## Phase 2

- Inverse view LUT: N=65 failed the 0.5-level check (97.6% of samples). N=129 passed: 99.70% within 0.5 level, mean 0.036 levels, max 9.59 levels. File: `Plugins/VotionSplat/Resources/VotionSplat_InverseView.bin`.
- gsplat references for all 11 synthetic PLYs wrote `unreal/validation/refs/<name>/{main,side}.npy`.
- T1 main bright centroid is (959.5, 539.5), argmax pixel (959, 539). T2 main centroid is (1343.5, 347.5), argmax (1344, 347).

## Phase 4

- `VotionSplatHostEditor` Win64 Development linked on 2026-11-20: `UnrealEditor-VotionSplat.dll` and `UnrealEditor-VotionSplatEditor.dll`.
- The user's other 5.5 editor (`Votion_AI_Splat_01`) had Live Coding active, which blocks every build of this engine executable. The plugin build used `-NoHotReloadFromIDE` and did not close that editor.

## Phase 3

WallGS baseline image and GPU timings are not measured yet. They need an editor session with WallGS enabled and are recorded when that session is run.

## Phase 5 and 6

- The editor imports a tagged PLY with no dialog. T1 logs `splats=1 degree=0 raster=gsplat_antialiased near=0.01`.
- The after-tonemap composite runs in the editor viewport (mode 0, one proxy).
- Validation captures use `/Engine/Maps/Entry` so the background is black, matching the gsplat references. `take_high_res_screenshot` must be called with the mask disabled; the mask fills the frame with green.
- An axis-aligned ellipse used a vanishing eigenvector, so the quad was clipped to the minor axis. The projector now picks the well-conditioned of the two equivalent axes.

## Phase 7 synthetic

All 11 PLYs, both cameras, 1920×1080. Thresholds are the synthetic profile except the two LiteGS files, which use the litegs profile. Every comparison passed with no pixel shift.

| Test | main mean / max | side mean / max |
| --- | --- | --- |
| T1_center | 0.0003 / 1 | 0.0005 / 1 |
| T2_offaxis | 0.0005 / 1 | 0.0004 / 1 |
| T3_aniso | 0.0026 / 1 | 0.0038 / 1 |
| T4_subpixel | 0.0008 / 1 | 0.0010 / 1 |
| T4_subpixel_litegs | 0.0013 / 1 | 0.0014 / 1 |
| T5_stack | 0.0306 / 1 | 0.0492 / 1 |
| T6_sh3 | 0.0087 / 1 | 0.0136 / 1 |
| T6_sh3_litegs | 0.0088 / 2 | 0.0138 / 2 |
| T7_near | 0.2264 / 1 | 0.1897 / 1 |
| T8_termination | 0.0495 / 1 | 0.0701 / 1 |
| T9_big | 0.1811 / 1 | 0.1840 / 1 |

T8 stays inside 1 level on these cameras, so the synthetic stack does not show a visible early-termination gap. The real-scene check still decides whether an exact termination mode is needed. A depth-cube occlusion check and an `-rdgvalidation` run are not recorded yet.

## Chart

`docs/unreal/Votion_Plugin_Chart.ply` (16,000 splats), front and three-quarter cameras, 1920×1080 and 3840×2160. Profile `gsplat`. All four pass, max error 1 level, no pixel shift.

| View | mean | max |
| --- | --- | --- |
| front | 0.0164 | 1 |
| threequarter | 0.0155 | 1 |
| front 4K | 0.0165 | 1 |
| threequarter 4K | 0.0158 | 1 |

The default level brush is a cube at the origin and writes depth, so it covered the middle of the chart until the capture hid BSP. A high-resolution shot whose scene-depth texture is not the shot's own view skips the depth test instead of sampling that other buffer.

## Phase 7 real scenes

`outputs/alpine_02/hyworld/splat.ply` and `outputs/alpine_01/splat.ply` were not modified. The HY votion export is `unreal/validation/scenes/alpine_02_votion.ply` (2,448,756 splats, `gsplat_antialiased`, near 0.00614, degree 0). alpine_01 was copied and tagged `litegs_classic`, near 0.2, degree 3 (2,978,688 splats). Its header had no postshot or votion comment.

gsplat classic still draws splats whose centres are outside the frame. LiteGS drops a centre when its NDC is outside ±1.3. On alpine_01 that cull changes the picture by about 45 levels, so `render_reference.py` applies it for `litegs_classic` before calling gsplat. Every view passed its profile with no pixel shift.

| Scene | View | mean | p99.9 | max |
| --- | --- | --- | --- | --- |
| alpine_02 1080 | yaw0 | 0.202 | 1 | 16 |
| alpine_02 1080 | yaw90 | 0.159 | 1 | 13 |
| alpine_02 1080 | yaw180 | 0.189 | 1 | 7 |
| alpine_02 1080 | yaw270 | 0.169 | 1 | 12 |
| alpine_02 1080 | yaw45 pitch -20 | 0.169 | 1 | 14 |
| alpine_02 4K | yaw0 | 0.207 | 2 | 52 |
| alpine_02 4K | yaw90 | 0.163 | 1 | 28 |
| alpine_02 4K | yaw180 | 0.192 | 1 | 54 |
| alpine_02 4K | yaw270 | 0.173 | 1 | 36 |
| alpine_02 4K | yaw45 pitch -20 | 0.174 | 2 | 52 |
| alpine_01 1080 | orbit0 | 0.205 | 4 | 56 |
| alpine_01 1080 | orbit90 | 0.144 | 1 | 61 |
| alpine_01 1080 | orbit180 | 0.217 | 10 | 89 |
| alpine_01 1080 | orbit270 | 0.179 | 2 | 44 |
| alpine_01 4K | orbit0 | 0.205 | 4 | 58 |
| alpine_01 4K | orbit90 | 0.144 | 1 | 61 |
| alpine_01 4K | orbit180 | 0.217 | 10 | 90 |
| alpine_01 4K | orbit270 | 0.180 | 2 | 53 |

yaw0 at 1920×1080, FOV 90, is the app viewport size. Mean 0.202, p99.9 1, `frac_gt8` 2.9e-6.

Termination: a 32×32 crop on alpine_01 orbit180 (the p99.9 of 10) matches a full back-to-front blend within 0.62 levels and differs from gsplat by 23 levels. gsplat stops early. alpine_02 stays at p99.9 ≤ 2, so that scene does not need the exact stop. alpine_01's p99.9 above 3 is this stop, so Phase 8 step 6 (ROV) is required and is kept only if it costs 25% or less. The checkpoint-versus-PLY export check was not run.

## Phase 8

`r.VotionSplat.SortReuse` (default 1) keeps the sorted index buffer while the view direction is bitwise unchanged and the splat count is unchanged. A view with no view state still sorts every frame. Entries older than 300 frames are dropped.

After the change, T1 main/side means are 0.00024 / 0.00047 (max 1). alpine_02 yaw0 mean is 0.20191, and a second shot of the same camera is 0.20191. Both pass, no pixel shift, and the mean moved by far less than 0.05. GPU times are not measured yet, so the 16-bit accumulation, compaction and 24-bit keys stay off.

## Phase 8 perf

Host viewport was 2770×1855. Scene depth was 1540×1032. One proxy, one view per frame. `r.ShaderDevelopmentMode=1` was on in this host project (it is not set in `Votion_AI_Splat_01`). Stats over 120 frames: `project` matched `views`, and `sort` was 0 or 1. The render path was otherwise unchanged, so the Phase 7 image numbers still apply.

| Step | Scene | Camera | fps | Project ms | Raster ms | Composite ms |
| --- | --- | --- | --- | --- | --- | --- |
| baseline, debug shaders | alpine_01 | orbit0 | 6.6 | 2.56 | 151.23 | 0.11 |
| baseline, debug shaders | alpine_01 | orbit90 | 4.9 | 2.38 | 208.63 | |
| baseline, debug shaders | alpine_01 | orbit180 | 5.1 | 2.38 | 200.11 | |
| baseline, debug shaders | alpine_01 | orbit270 | 8.4 | 2.46 | 120.62 | |
| baseline, debug shaders | alpine_02 | yaw0 | 18.5 | 0.34 | 54.37 | |
| baseline, debug shaders | alpine_02 | yaw90 | 25.5 | 0.31 | 44.40 | |
| baseline, debug shaders | alpine_02 | yaw180 | 22.2 | 0.30 | 44.20 | |
| baseline, debug shaders | alpine_02 | yaw270 | 14.4 | 0.37 | 69.82 | |
| baseline, debug shaders | alpine_02 | yaw45 pitch -20 | 20.1 | 0.32 | 50.09 | |

Sort did not appear in ProfileGPU while the camera was still. Raster draws every splat (5,957,376 prims on alpine_01, 4,897,512 on alpine_02). The empty-scene passes around it are under 3 ms, so the frame is the raster.

`r.ShaderDevelopmentMode` is now 0 in the host project, matching `Votion_AI_Splat_01`. The renderer now caches the projection per view, compacts to visible splats in the existing order, and draws with `DrawPrimitiveIndirect`.

16-bit accumulation (`r.VotionSplat.AccumFormat 0`) passed every profile threshold, and every synthetic mean stayed within 0.05 of Phase 7. alpine_02 means rose by at most 0.044. alpine_01 did not:

| View | Phase 7 mean | 16-bit mean | 32-bit mean |
| --- | --- | --- | --- |
| orbit0 | 0.205 | 0.279 | 0.205 |
| orbit90 | 0.144 | 0.203 | 0.144 |
| orbit180 | 0.217 | 0.303 | 0.217 |
| orbit270 | 0.179 | 0.254 | 0.179 |

32-bit with compaction and the view cache matches Phase 7. The default stays `r.VotionSplat.AccumFormat 1`.

The same viewport, after the view cache and visible-splat draw, with 32-bit accumulation:

| Scene | Camera | before fps | after fps | before raster ms | after raster ms |
| --- | --- | --- | --- | --- | --- |
| alpine_01 | orbit0 | 6.6 | 6.9 | 151.23 | 144.47 |
| alpine_01 | orbit90 | 4.9 | 5.1 | 208.63 | 200.90 |
| alpine_01 | orbit180 | 5.1 | 5.3 | 200.11 | 194.07 |
| alpine_01 | orbit270 | 8.4 | 8.7 | 120.62 | 114.90 |
| alpine_02 | yaw0 | 18.5 | 19.2 | 54.37 | 51.04 |
| alpine_02 | yaw90 | 25.5 | 26.8 | 44.40 | 36.08 |
| alpine_02 | yaw180 | 22.2 | 23.0 | 44.20 | 42.24 |
| alpine_02 | yaw270 | 14.4 | 14.8 | 69.82 | 68.04 |
| alpine_02 | yaw45 pitch -20 | 20.1 | 20.8 | 50.09 | 47.68 |

While the camera is still, stats show about 1 project, 1 sort and 1 compact per 120 frames. Those passes are not the frame. The raster time stays, because the visible splats still cover the 2770×1855 target.

16-bit accumulation on the same two views: alpine_01 orbit0 32.2 fps, raster 26.72 ms (was 6.9 fps / 144 ms). alpine_02 yaw0 75.4 fps, raster 8.39 ms (was 19.2 fps / 51 ms). That is the fast path, and it is off by default because of the alpine_01 means above. Set `r.VotionSplat.AccumFormat 0` to turn it on.

## Phase 8 tile rasterizer

`r.VotionSplat.TileRaster 1` (default) blends the compacted splats front to back in 16×16 compute tiles. Per pixel it applies the same alpha rule and scene-depth test as the quad shader. It stops when transmittance would fall to `r.VotionSplat.TileStopT` (1e-4, the gsplat rule). It accumulates in registers and writes Accum once.

Tile lists are built from the compacted records: a tile count per splat, then a scan, then emission in record order. A stable 4-bit radix sort on the tile id runs 4 passes at this size, and per-tile ranges come last. The lists are cached per view with the draw records. Pair buffers start at 4×N (at least 4M, at most 64M). They grow from a GPU readback of the pair total. A frame whose pairs exceed the buffer draws with quads, using indirect args written on the GPU. Only the first proxy in a view uses tiles; later proxies blend quads over it.

Image gate, tile path. Every synthetic capture passed (T5 0.0305/0.0491, T8 0.0501/0.0711, T9 0.181/0.184). Every real capture passed, and every real mean improved on Phase 7:

| Scene | View | Phase 7 mean | Tile mean | Tile p99.9 | Tile max |
| --- | --- | --- | --- | --- | --- |
| alpine_02 | yaw0 | 0.202 | 0.189 | 1 | 3 |
| alpine_02 | yaw90 | 0.159 | 0.148 | 1 | 2 |
| alpine_02 | yaw180 | 0.189 | 0.185 | 1 | 1 |
| alpine_02 | yaw270 | 0.169 | 0.153 | 1 | 2 |
| alpine_02 | yaw45 pitch -20 | 0.169 | 0.144 | 1 | 5 |
| alpine_01 | orbit0 | 0.205 | 0.181 | 1 | 55 |
| alpine_01 | orbit90 | 0.144 | 0.131 | 1 | 61 |
| alpine_01 | orbit180 | 0.217 | 0.191 | 10 | 89 |
| alpine_01 | orbit270 | 0.179 | 0.157 | 1 | 44 |

alpine_01 orbit180 keeps a p99.9 of 10, so the remaining gap there is not the early stop.

Perf. The host viewport was 1847×1237 this time, so the quad path was re-measured at that size. 32-bit Accum, still camera, ProfileGPU ms:

| Scene | Camera | quad fps | quad raster ms | tile fps | tile raster ms |
| --- | --- | --- | --- | --- | --- |
| alpine_01 | orbit0 | 14.5 | 69.76 | 120.1 | 4.22 |
| alpine_01 | orbit90 | 10.8 | 94.19 | 114.3 | 7.33 |
| alpine_01 | orbit180 | 11.0 | 91.88 | 120.1 | 1.26 |
| alpine_01 | orbit270 | 18.0 | 56.99 | 80.1 | 9.52 |
| alpine_02 | yaw0 | 40.1 | 23.71 | 120.1 | 1.72 |
| alpine_02 | yaw90 | 54.2 | 17.04 | 120.1 | 1.72 |
| alpine_02 | yaw180 | 47.5 | 19.38 | 120.1 | 1.47 |
| alpine_02 | yaw270 | 31.1 | 31.32 | 120.1 | 2.13 |
| alpine_02 | yaw45 pitch -20 | 42.9 | 22.27 | 120.1 | 1.62 |

120.1 fps is the editor frame cap. alpine_01 needs 18.7M to 23.8M pairs at this size, so the buffer grows once from 16M to 32M.

Moving-camera cost. With `r.VotionSplat.ViewCache 0` and `r.VotionSplat.SortReuse 0`, every frame runs Project, Sort, Compact, the tile build and the tile raster. alpine_01 runs at 40.3–51.3 fps: project 2.5 ms, sort 1.7 ms, compact 1.4 ms, tile build 7.1–9.9 ms. alpine_02 runs at 90.1–111.0 fps, with a tile build of 2.5–3.6 ms. The tile build is the next cost to cut: its radix and range passes dispatch over the whole pair buffer rather than the pair count.
