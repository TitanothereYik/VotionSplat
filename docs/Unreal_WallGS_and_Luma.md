# Unreal colour match — WallGS Pro and Luma

Checked against WallGS Pro 1.1.4 on Unreal Engine 5.5, and Luma AI Plugin 0.4.1 on Unreal Engine 5.3.

Export the PLY with the matching preset in Votion, then use only that plugin. The presets are not interchangeable. **Unreal / WallGS Pro, OCIO ACES view** writes the same file as **Unreal / WallGS Pro**, set up for an OCIO / ACES viewport instead of the LUT; see its section below. The HY-World file is `outputs/<scene>/hyworld/splat.ply`.

Leave the Gaussian actor at scale 1. Both plugins convert coordinates on import.

**Soften like the viewport (no needles)** is on by default for HY-World exports and applies to every color space. The Votion preview fades splats thinner than a pixel (antialiased rendering) and is upscaled by Windows display scaling (150% on a 4K screen). WallGS and Luma draw every splat at full opacity and full resolution, so thin splats show as needles and dots. The export widens each splat slightly in 3D and rescales its opacity so its brightness stays the same, sized for a 1920 px, 90° view from the nearest training camera (`SOFTEN_*` in `engine/splat/defaults.py`). It roughly halves the extra sharpness Unreal shows over the preview, and needles mostly disappear. Unreal stays a little crisper, because a static file can't reproduce the preview's per-view fade and upscale exactly. The bake adds about 30 s to the export. Turn it off to export the trained splats unchanged.

## Why the setup looks like this

Votion trains like standard 3DGS: overlapping Gaussians are blended in display (sRGB-encoded) space. Many trained Gaussians carry strong colours that only cancel out in that blend. Blending the same Gaussians in linear light lets the hidden colours through as purple, blue, and orange blotches, with harder contrast.

- **WallGS** blends untagged PLYs in display space, then converts the result with an inverse film curve. The Votion export is untagged, and the WallGS LUT undoes that curve.
- **Luma** always blends in linear light. The Votion export stores colour so that Luma's linear blend is proportional to the display-space blend. Exposure scales it back and the Luma LUT displays it.

Both LUTs are in `docs/unreal/`. `tools/make_unreal_luts.py` regenerates them from the values in `engine/splat/defaults.py`.

## Import the LUT

1. Drag `docs/unreal/Votion_WallGS_LUT.png` or `docs/unreal/Votion_Luma_LUT.png` into the Content Browser.
2. Open the texture and set **Texture Group** to **ColorLookupTable**. Unreal then turns sRGB off, removes mips, and keeps it uncompressed.
3. Save.

## Post Process Volume

Add one Post Process Volume and set **Infinite Extent (Unbound)**. Give it a higher **Priority** than any other unbound volume, or remove the others. A camera or sequencer post-process override replaces this volume; match the same values there, or set that override's blend weight to 0.

| Setting | Where | WallGS Pro | Luma 0.4.1 |
| --- | --- | --- | --- |
| Infinite Extent (Unbound) | Post Process Volume | on | on |
| Metering Mode | Exposure | Manual | Manual |
| Apply Physical Camera Exposure | Exposure | off | off |
| Exposure Compensation | Exposure | **-1.5** | **2.32** |
| Highlight Contrast | Lens > Local Exposure | 1 | 1 |
| Shadow Contrast | Lens > Local Exposure | 1 | 1 |
| Detail Strength | Lens > Local Exposure | 1 | 1 |
| Bloom Intensity | Lens > Bloom | 0 | 0 |
| Lens Flares Intensity | Lens > Lens Flares | 0 | 0 |
| Vignette Intensity | Lens > Image Effects | 0 | 0 |
| Blue Correction | Color Grading > Misc | 0 | 0 |
| Expand Gamut | Color Grading > Misc | 0 | 0 |
| Tone Curve Amount | Color Grading > Misc | 0 | 0 |
| Gamma (luminance slider) | Color Grading > Global | **1.4** | 1 (default) |
| Color Grading LUT | Color Grading > Misc | `Votion_WallGS_LUT` | `Votion_Luma_LUT` |
| Color Grading LUT Intensity | Color Grading > Misc | 1 | 1 |

Leave every other Color Grading value (Saturation, Contrast, Gain, Offset, White Balance, and the Shadows/Midtones/Highlights groups) at its default.

The exposure and gamma values belong to the LUTs. A different value needs a regenerated LUT, so adjust brightness in Votion, not here.

WallGS's inverse film curve squeezes every display value below about 0.075 into the first of the LUT's 16 steps, where the output falls off steeply and the darks crush toward black. Global Gamma runs before the LUT and spreads those darks over several steps; the WallGS LUT is fitted through it. Without the Gamma setting the WallGS LUT shows wrong colours.

New UE 5.x projects write Local Exposure contrast 0.8 into `Config/DefaultEngine.ini` (`r.DefaultFeature.LocalExposure.HighlightContrastScale` and `...ShadowContrastScale`). The three Local Exposure overrides above set it back to neutral.

Turn Exponential Height Fog off while matching colour. It darkens the distance on its own.

## WallGS Pro (Unreal Engine 5.5)

In Votion, export with **Unreal / WallGS Pro**. The file has no colour tags.

### Project

**Project Settings > Rendering > Forward Shading** stays off. WallGS Pro does not support Forward Shading on Win64.

### Import and place

1. Delete earlier Votion WallGS assets from the level, especially any imported while the export still wrote sRGB tags. A single tagged splat switches every splat in the level to linear blending.
2. Drag `splat.ply` into the Content Browser.
3. In the WallGS dialog, choose **Native Asset (Recommended)** and press **Import**.
4. Drag the asset into the level.
5. Open the asset. **Source Color Space** reads **Legacy Appearance (untagged)**. Leave it there; sRGB (encoded) and the linear options blend in linear light.

### Actor

Select the splat component. Leave Appearance at the plugin defaults:

| Setting | Value |
| --- | --- |
| Splat Scale | 1 |
| Exposure | 0 |
| Contrast | 1 |
| Saturation | 1 |
| Gamma | 1 |
| Tint | white |
| SH Degree | 3 |
| SH Strength | 1 |
| Relighting | off |
| Discard Extremely Transparent Splats | off |

Change these three under **Optimization** (some are under the advanced arrow):

| Setting | Value |
| --- | --- |
| Preset (Spatial LOD) | **Off** |
| Rendering Method | Sorted Alpha (Quality) |
| Splat Budget (0 = Unlimited) | 0 |

The default Spatial LOD preset, Balanced, keeps every splat within 3 m. Beyond that it drops 25% of splats from about 6 m, 40% from 10 m and 55% from 18 m, and makes the survivors more opaque. A Votion splat builds smooth surfaces from many faint overlapping Gaussians, so the dropped ones leave dots and colour grain on walls. Stochastic rendering adds per-frame noise.

Relighting treats the capture as a surface under Unreal lights, so unlit areas go dark. Current HY-World checkpoints store base colour only (degree 0), so SH Degree and SH Strength have nothing further to show on this export.

## WallGS Pro through OCIO / ACES

In Votion, export with **Unreal / WallGS Pro, OCIO ACES view**. It writes the same untagged file as **Unreal / WallGS Pro**, so WallGS blends it in display space exactly like the Votion preview. A tagged ACEScg file would be blended in linear light, which turns the hidden colours described above into glitter. Colour management happens in Unreal's OCIO viewport instead.

WallGS turns each blended pixel into scene colour with a fixed inverse film curve (scene colour 0 to 5.55). `docs/unreal/Votion_VS_OCIO.ocio` is a copy of `VS_OCIO.ocio` with one added colour space, **Votion WallGS Legacy**. It undoes that curve and then applies the inverse of **sRGB - Display / ACES 1.0 - SDR Video**. With it as the OCIO Source, that view shows the Votion colours. Greys come back within 0.0001, and 99.87% of the current scene's splat colours come back exactly; the rest are saturated colours the ACES SDR view can't produce. The frame holds scene-referred ACES values, with display white at about 16, so the config's other views (the OpenVPCal LED wall, HDR displays) get proper scene-referred input. All other colour spaces and views are unchanged. `tools/make_ocio_config.py` regenerates both files from `VS_OCIO.ocio`, or from `VOTION_OCIO_CONFIG`.

1. Copy `docs/unreal/Votion_VS_OCIO.ocio` and `docs/unreal/Votion_WallGS_Legacy.spi1d` into the folder that holds your `VS_OCIO.ocio` and its OpenVPCal `.clf` files. The OpenVPCal view needs `Calibration wall5 - REDWideGamutRGB_EOTF_CS.clf`, which isn't in the repo.
2. Remove every ACEScg- or Linear-tagged WallGS splat from the level. One tagged splat switches every splat in the level to linear blending.
3. Import the PLY as **Native Asset (Recommended)**. **Source Color Space** reads **Legacy Appearance (untagged)**. Leave it there.
4. Set the actor's Appearance and **Optimization** as in the tables above. **Preset Off** matters most.
5. Create an **OpenColorIO Configuration** asset pointing at `Votion_VS_OCIO.ocio`. In the viewport's OCIO settings, enable it with Source **Votion WallGS Legacy**, Display **sRGB - Display** and View **ACES 1.0 - SDR Video**. Keep **Project Settings > Rendering > Working Color Space** at **sRGB / Rec709**.
6. Unbound Post Process Volume: Manual metering, Apply Physical Camera Exposure off and Exposure Compensation **0** (Unreal then multiplies by exactly 1). Use the same Local Exposure, Bloom, Lens Flares, Vignette, Blue Correction, Expand Gamut and Tone Curve Amount values as the table above. Leave out the Color Grading LUT: with OCIO on, Unreal outputs the colour before its LUT and Gamma, so they have no effect.

The OCIO Source applies to the whole frame. Other Unreal content in the level would be read through the WallGS curve too, so this setup suits splat-only levels.

## Luma (Unreal Engine 5.3, plugin 0.4.1)

In Votion, export with **Unreal / Luma**. Luma's material emits `0.4 * pow(colour, 2.2)`. The export stores colour so this equals `0.4 * trained colour / 2`. Exposure Compensation 2.32 multiplies by 5, which brings the blended result back to the trained display colour, and the Luma LUT displays it. Trained colours above 2 clip, which affects about 0.2% of Gaussians in the current scene.

### Import and place

1. Delete the previously generated Luma assets. Luma keeps the old textures inside them.
2. Drag `splat.ply` into the Content Browser and let Luma build the assets.
3. Place the blueprint whose name ends in **`_Baked`**.
4. Leave `_Dynamic`, `_Dynamic_No_TAA`, `_Baked_No_TAA`, and `_Baked_Cropped` unused.
5. Actor scale stays 1.

`_Baked_Cropped` and an enabled crop box keep only the inside of the box. Use the plain `_Baked` actor with cropping off.

### Why the file is larger than the training count

Luma 0.4.1 splits the scene into 64 chunks, but a file of 1,900,000 Gaussians or fewer is imported as the centre chunk alone. Everything outside that box disappears, which reads as a dark surround. Votion pads a smaller file with invisible Gaussians so the import takes all 64 chunks. The extra rows are invisible. Chunks that still hold 100 splats or fewer are dropped by the plugin.

The file always has 45 `f_rest` columns, which is the degree-3 layout Luma 0.4.1 reads. Faint Gaussians below opacity 0.02 are removed, and needle scales are limited to 64:1.

## Accuracy

The full chain was simulated on the CPU for training view `panorama_L00_A0008`: blend, plugin colour handling, Unreal exposure, the 16-point LUT, and Unreal's 32-point log LUT. Mean difference from the Votion trainer image, in display units from 0 to 1:

| Path | Mean difference |
| --- | --- |
| WallGS, untagged + WallGS LUT | 0.005 |
| WallGS, sRGB-tagged export (old) | 0.018 |
| Luma, current export + Luma LUT | 0.004 |
| Luma, previous export, no LUT | 0.0135 |

The remaining difference comes from Unreal's LUT resolution and from both plugins skipping the trainer's antialiasing filter.

Measured in Unreal 5.5 with WallGS Pro, using the test chart `docs/unreal/Votion_WallGS_Chart.ply` (`tools/make_unreal_chart.py`): 32 greys and 8 colours, captured with HighResShot. With Gamma 1.4 and the current LUT, greys from 0.02 to 0.95 land within 0.004 of the Votion value, and the mean error over all patches is 0.003. A full room view from the same camera matched a trainer-rule render of `splat.ply` to within 0.003 per brightness band above 0.08.
