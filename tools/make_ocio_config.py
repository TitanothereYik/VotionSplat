"""Write the OCIO config that shows an untagged WallGS Pro splat through an ACES view exactly like Votion.

WallGS blends an untagged (Legacy Appearance) PLY in display space, like the trainer, then writes
scene colour L = inverse-Narkowicz(D^2.2) (D^2.2 clamped to 0.99) into Unreal's working space (sRGB
linear by default). The added colour space "Votion WallGS Legacy" undoes that curve back to the blended
display colour D, then applies the inverse of D.OCIO_DISPLAY / D.OCIO_VIEW to reach scene-referred ACES.
Used as the OCIO Source in Unreal, that view returns D on screen, and every other view of the config
(LED wall, HDR) sees real scene-referred ACES values. The user's config is copied, never edited. Run:
    .venv\\Scripts\\python.exe tools\\make_ocio_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.splat import defaults as D  # noqa: E402

OUT_DIR = ROOT / "docs" / "unreal"
LUT_NAME = "Votion_WallGS_Legacy.spi1d"
LUT_SIZE = 16384
NARKOWICZ = (2.51, 0.03, 2.43, 0.59, 0.14)
LEGACY_CLAMP = 0.99  # WallGSInverseLegacyToneCurve clamps D^2.2 here


def narkowicz(x):
    a, b, c, d, e = NARKOWICZ
    x = np.maximum(x, 0.0)
    return x * (a * x + b) / (x * (c * x + d) + e)


def wallgs_legacy_to_working(display):
    """WallGSLegacyToWorking for an sRGB working space (WallGSColor.ush)."""
    a, b, c, d, e = NARKOWICZ
    y = np.clip(np.power(np.maximum(display, 0.0), 2.2), 0.0, LEGACY_CLAMP)
    qa, qb, qc = a - y * c, b - y * d, -y * e
    return (-qb + np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)


def write_lut(path: Path) -> float:
    top = float(wallgs_legacy_to_working(np.array([1.0]))[0])
    values = np.clip(narkowicz(np.linspace(0.0, top, LUT_SIZE)), 0.0, LEGACY_CLAMP)
    body = "\n".join(f"{v:.9f}" for v in values)
    path.write_text(f"Version 1\nFrom 0.0 {top:.9f}\nLength {LUT_SIZE}\nComponents 1\n{{\n{body}\n}}\n")
    return top


def build(ocio, source: Path):
    config = ocio.Config.CreateFromFile(str(source))
    # getTransform returns the config's own objects; flipping those would invert the view itself
    parts = ocio.Config.CreateFromFile(str(source))
    to_view = parts.getViewTransform(D.OCIO_VIEW).getTransform(ocio.VIEWTRANSFORM_DIR_FROM_REFERENCE)
    to_display = parts.getColorSpace(D.OCIO_DISPLAY).getTransform(ocio.COLORSPACE_DIR_FROM_REFERENCE)
    if to_view is None or to_display is None:
        raise RuntimeError(f"{source.name}: {D.OCIO_VIEW} / {D.OCIO_DISPLAY} need from_*_reference transforms")
    to_view.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    to_display.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    group = ocio.GroupTransform([
        ocio.FileTransform(src=LUT_NAME, interpolation=ocio.INTERP_LINEAR),
        ocio.ExponentTransform(value=[2.2, 2.2, 2.2, 1.0], direction=ocio.TRANSFORM_DIR_INVERSE),
        to_display,
        to_view,
    ])
    space = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name=D.OCIO_WALLGS_SOURCE,
        family="Votion",
        encoding="scene-linear",
        description=(f"Unreal scene colour of an untagged WallGS Pro splat (sRGB working space) back to "
                     f"scene-referred ACES through the inverse of {D.OCIO_DISPLAY} / {D.OCIO_VIEW}. "
                     "Written by Votion tools/make_ocio_config.py."),
    )
    space.setTransform(group, ocio.COLORSPACE_DIR_TO_REFERENCE)
    config.addColorSpace(space)
    config.setName(f"{config.getName()}_votion")
    # The source config's own LUTs (e.g. OpenVPCal calibration) live next to it.
    config.setSearchPath("./")
    config.addSearchPath(str(source.resolve().parent).replace("\\", "/"))
    config.validate()
    return config


def shown_error(ocio, path: Path, display: np.ndarray) -> np.ndarray:
    """Per-colour max channel error between blended display colour ``display`` [N, 3] and what the
    target view shows from WallGS's scene colour through the new source. Saturated colours outside
    what the ACES output transform can produce come back desaturated."""
    config = ocio.Config.CreateFromFile(str(path))
    shown = config.getProcessor(ocio.DisplayViewTransform(
        src=D.OCIO_WALLGS_SOURCE, display=D.OCIO_DISPLAY, view=D.OCIO_VIEW)).getDefaultCPUProcessor()
    scene = np.ascontiguousarray(wallgs_legacy_to_working(display), dtype=np.float32)
    shown.applyRGB(scene)
    return np.abs(scene - np.minimum(display, LEGACY_CLAMP ** (1 / 2.2))).max(axis=1)


def check(ocio, path: Path, source: Path) -> tuple[float, float, float, list[str]]:
    """Grey-ramp max error and random-colour p99 error of the new source through the target view,
    max change of every existing view (ACEScg in), and views skipped for files missing in the source."""
    rng = np.random.default_rng(0)
    grey = float(shown_error(ocio, path, np.linspace(0, 1, 256)[:, None].repeat(3, 1)).max())
    colour = float(np.percentile(shown_error(ocio, path, rng.random((20000, 3))), 99))
    config = ocio.Config.CreateFromFile(str(path))
    original = ocio.Config.CreateFromFile(str(source))
    probe = (rng.random((20000, 3)) * 20).astype(np.float32)
    drift, skipped = 0.0, []
    for disp in original.getDisplays():
        for view in original.getViews(disp):
            try:
                a, b = (cfg.getProcessor(ocio.DisplayViewTransform(src=D.OCIO_SOURCE, display=disp, view=view))
                        .getDefaultCPUProcessor() for cfg in (original, config))
            except (ocio.Exception, ocio.ExceptionMissingFile):
                skipped.append(f"{disp} / {view}")
                continue
            ya, yb = probe.copy(), probe.copy()
            a.applyRGB(ya)
            b.applyRGB(yb)
            drift = max(drift, float(np.nanmax(np.abs(ya - yb))))
    return grey, colour, drift, skipped


def main() -> int:
    import PyOpenColorIO as ocio

    source = D.ocio_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top = write_lut(OUT_DIR / LUT_NAME)
    out = OUT_DIR / D.OCIO_VOTION_CONFIG_NAME
    out.write_text(build(ocio, source).serialize())
    grey, colour, drift, skipped = check(ocio, out, source)
    print(f"wrote {out} and {LUT_NAME} (scene colour 0..{top:.3f}); "
          f"{D.OCIO_WALLGS_SOURCE} -> {D.OCIO_DISPLAY} / {D.OCIO_VIEW}: grey error {grey:.5f}, "
          f"random colours p99 {colour:.4f}; existing views changed by at most {drift:.2e}")
    if skipped:
        print(f"not checked (files missing next to {source.name}): {', '.join(skipped)}")
    return 0 if grey < 2e-3 and drift == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
