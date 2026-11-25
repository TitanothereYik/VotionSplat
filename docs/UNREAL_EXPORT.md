# Unreal export

For the best result in Unreal, use **[VotionSplat-UnrealPlugin](https://github.com/TitanothereYik/VotionSplat-UnrealPlugin)**. It renders the splats with the same rasterizer rule, near plane, and spherical-harmonic degree they were trained with, which the export writes into the PLY header. A generic 3DGS plugin will load the file but will not match the app viewport.

On the Splat page, **Export splat.ply** defaults to **Unreal / VotionSplat Export**:

> Unreal / VotionSplat Export: drag splat.ply into the Content Browser with VotionSplat-UnrealPlugin. No settings. The file keeps the trained splats, and the plugin draws them with the same rule as this viewport.

The file is `outputs\<scene>\hyworld\splat.ply`.

## Install the plugin

1. Copy `{app}\unreal_plugin\VotionSplat-UnrealPlugin` into `<UEProject>\Plugins\`, or clone [the plugin repository](https://github.com/TitanothereYik/VotionSplat-UnrealPlugin) there and let the editor compile it.
2. The plugin is for Unreal Engine 5.5.
3. Enable it, then drag `splat.ply` into the Content Browser.

`docs/unreal/VS_OCIO.ocio` (`{app}\docs\unreal\VS_OCIO.ocio` in an install) is the config used to bake the plugin's inverse-view lookup. The export does not ask you to pick an OCIO view.

The other two export presets are for tools outside Unreal: **3DGS / sRGB** and **Linear Rec.709**.

Screenshots: [unreal-01](media/screenshots/unreal-01.png), [unreal-02](media/screenshots/unreal-02.png), [unreal-03](media/screenshots/unreal-03.png).
