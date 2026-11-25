# Unreal export

On the Splat page, **Export splat.ply** defaults to **Unreal / VotionSplat Export**:

> Unreal / VotionSplat Export: drag splat.ply into the Content Browser with the Votion Splat plugin. No settings. The file keeps the trained splats, and the plugin draws them with the same rule as this viewport.

The file is `outputs\<scene>\hyworld\splat.ply`.

## Install the plugin

1. Copy `{app}\unreal_plugin\VotionSplat` into `<UEProject>\Plugins\`.
2. The plugin is for Unreal Engine 5.5.
3. Enable it, then drag `splat.ply` into the Content Browser.

`docs/unreal/VS_OCIO.ocio` is the config used to bake the plugin's inverse-view lookup. The export does not ask you to pick an OCIO view.

The other two export presets are for tools outside Unreal: **3DGS / sRGB** and **Linear Rec.709**.

Screenshots: [unreal-01](media/screenshots/unreal-01.png), [unreal-02](media/screenshots/unreal-02.png), [unreal-03](media/screenshots/unreal-03.png).
