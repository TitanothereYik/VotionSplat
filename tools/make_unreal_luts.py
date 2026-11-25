"""Write the Unreal Color Grading LUTs that restore Votion's display-space colours.

WallGS Pro (untagged PLY): WallGS blends in display space, then reconstructs scene colour with an
inverse film curve. The WallGS LUT undoes that curve, so with the tone curve off the screen shows the
blended display colour again. That curve squeezes every display value below ~0.075 into the first
of the user LUT's 16 sRGB-spaced nodes, so the Post Process Volume also sets Color Grading > Global
Gamma (applied before the LUT) to spread the darks over several nodes; the 3D LUT is fitted through it.

Luma: the export stores colour so Luma's linear blend is proportional to the display-space blend.
The Luma LUT turns that value back into display colour.

Both LUTs are fitted through Unreal 5.5's two LUT stages (16-point user LUT in sRGB encoding, baked
into the 32-point log-encoded combined LUT) and quantised to 8 bit. Run:
    .venv\\Scripts\\python.exe tools\\make_unreal_luts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.splat import defaults as D  # noqa: E402

OUT_DIR = ROOT / "docs" / "unreal"
USER_N = 16
COMBINED_N = 32
SRGB_TO_AP1 = np.array([[0.6131, 0.3395, 0.0474], [0.0702, 0.9164, 0.0134], [0.0206, 0.1096, 0.8698]])
AP1_TO_SRGB = np.linalg.inv(SRGB_TO_AP1)
SMOOTH = 0.03  # weight of the neighbour-difference rows that pin LUT nodes no colour reaches


def srgb_oetf(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb_eotf(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def wallgs_inverse_legacy(c):
    """WallGSLegacyToWorking: inverse Narkowicz ACES fit of pow(c, 2.2), input clamped to 0.99."""
    y = np.clip(np.power(np.maximum(c, 0.0), 2.2), 0.0, 0.99)
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    qa, qb, qc = a - y * cc, b - y * d, -y * e
    return (-qb + np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)


def log_to_lin(e):
    return np.exp2((e - 444.0 / 1023.0) * 14.0) * 0.18


def lin_to_log(x):
    return np.log2(x) / 14.0 - np.log2(0.18) / 14.0 + 444.0 / 1023.0


def grade(lin, gamma):
    """Unreal's Global Gamma in ColorCorrect: per-channel pow(x, 1/gamma) in AP1, back to sRGB, clipped."""
    ap1 = np.maximum(lin @ SRGB_TO_AP1.T, 0.0)
    return np.maximum(np.power(ap1, 1.0 / gamma) @ AP1_TO_SRGB.T, 0.0)


def _interp_matrix(pos, n):
    """Rows: linear-interpolation weights of ``pos`` in [0, 1] over ``n`` evenly spaced nodes."""
    t = np.clip(pos, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(t).astype(int), n - 2)
    f = t - i0
    m = np.zeros((pos.size, n))
    m[np.arange(pos.size), i0] = 1 - f
    m[np.arange(pos.size), i0 + 1] = f
    return m


def _trilinear(pos, n):
    """Sparse rows of trilinear weights of ``pos`` (k, 3) in [0, 1]^3 over an n^3 grid, index (b*n+g)*n+r."""
    from scipy.sparse import csr_matrix

    t = np.clip(pos, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(t).astype(int), n - 2)
    f = t - i0
    rows, cols, vals = [], [], []
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = (np.where(dr, f[:, 0], 1 - f[:, 0]) * np.where(dg, f[:, 1], 1 - f[:, 1])
                     * np.where(db, f[:, 2], 1 - f[:, 2]))
                idx = ((i0[:, 2] + db) * n + i0[:, 1] + dg) * n + i0[:, 0] + dr
                rows.append(np.arange(len(pos)))
                cols.append(idx)
                vals.append(w)
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(pos), n**3))


def _pipeline_matrix(scene):
    """Linear map from the 16 user-LUT values to the on-screen value for each scene-linear input."""
    e_nodes = np.arange(COMBINED_N) / (COMBINED_N - 1)
    x_nodes = log_to_lin(e_nodes) - log_to_lin(0.0)
    user = _interp_matrix(srgb_oetf(x_nodes), USER_N)  # combined-LUT node -> user-LUT weights
    e = lin_to_log(np.maximum(scene, 0.0) + log_to_lin(0.0))
    return _interp_matrix(e, COMBINED_N) @ user


def pipeline_matrix_3d(scene_rgb, gamma):
    """Sparse map from the 16^3 user-LUT node values to the on-screen value, Global Gamma included."""
    g = np.arange(COMBINED_N) / (COMBINED_N - 1)
    eb, eg, er = np.meshgrid(g, g, g, indexing="ij")
    e_nodes = np.stack([er.ravel(), eg.ravel(), eb.ravel()], 1)
    lin_nodes = log_to_lin(e_nodes) - log_to_lin(0.0)
    user = _trilinear(srgb_oetf(grade(lin_nodes, gamma)), USER_N)
    e = lin_to_log(np.maximum(scene_rgb, 0.0) + log_to_lin(0.0))
    return (_trilinear(e, COMBINED_N) @ user).tocsr()


def _smooth_rows(n):
    from scipy.sparse import coo_matrix

    idx = np.arange(n**3).reshape(n, n, n)
    pairs = [(idx[1:].ravel(), idx[:-1].ravel()), (idx[:, 1:].ravel(), idx[:, :-1].ravel()),
             (idx[:, :, 1:].ravel(), idx[:, :, :-1].ravel())]
    a = np.concatenate([p[0] for p in pairs])
    b = np.concatenate([p[1] for p in pairs])
    k = np.arange(a.size)
    return coo_matrix((np.r_[np.ones(k.size), -np.ones(k.size)], (np.r_[k, k], np.r_[a, b])),
                      shape=(k.size, n**3)).tocsr()


def fit_lut(scene_of_display, display):
    """Least-squares user-LUT values (8-bit) so the screen reproduces ``display``."""
    from scipy.optimize import lsq_linear

    a = _pipeline_matrix(scene_of_display(display))
    v = lsq_linear(a, display, bounds=(0.0, 1.0)).x
    v8 = np.round(v * 255) / 255
    err = np.abs(a @ v8 - display)
    return v8, err


def fit_lut_3d(scene_of_display, colours, gamma, weights):
    """Per-channel bounded least squares over the 16^3 nodes; returns (16^3, 3) 8-bit values and errors."""
    from scipy.optimize import lsq_linear
    from scipy.sparse import diags, vstack

    a = pipeline_matrix_3d(scene_of_display(colours), gamma)
    aw = diags(weights) @ a
    s = _smooth_rows(USER_N) * SMOOTH * np.sqrt(weights.sum() / USER_N**3)
    big = vstack([aw, s]).tocsr()
    v = np.zeros((USER_N**3, 3))
    for ch in range(3):
        rhs = np.r_[weights * colours[:, ch], np.zeros(s.shape[0])]
        v[:, ch] = lsq_linear(big, rhs, bounds=(0.0, 1.0), lsmr_tol="auto", max_iter=200).x
    v8 = np.round(v * 255) / 255
    return v8, np.abs(a @ v8 - colours), a


def lut_image(v8) -> Image.Image:
    """Unreal 256x16 strip: 16 slices along x (blue), red across a slice, green down."""
    r = np.tile(v8, (USER_N, USER_N))
    g = np.repeat(v8[:, None], USER_N * USER_N, axis=1)
    b = np.repeat(np.repeat(v8, USER_N)[None, :], USER_N, axis=0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.round(rgb * 255).astype(np.uint8), "RGB")


def lut_image_3d(v8) -> Image.Image:
    """Same strip layout from (16^3, 3) node values indexed (b*16+g)*16+r."""
    cube = v8.reshape(USER_N, USER_N, USER_N, 3)  # b, g, r
    strip = np.concatenate([cube[b] for b in range(USER_N)], axis=1)  # rows g, cols b*16+r
    return Image.fromarray(np.round(strip * 255).astype(np.uint8), "RGB")


def wallgs_samples():
    ramp = np.concatenate([np.geomspace(1e-3, 0.05, 120), np.linspace(0.05, 0.99, 240)])
    g = np.concatenate([np.geomspace(2e-3, 0.08, 6), np.linspace(0.12, 0.99, 14)])
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    colours = np.concatenate([np.repeat(ramp[:, None], 3, 1), cube])
    weights = np.r_[np.full(ramp.size, 4.0), np.ones(len(cube))]
    return colours, weights, ramp.size


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wall_scale = 2.0 ** D.WALLGS_UE_EXPOSURE
    colours, weights, n_grey = wallgs_samples()
    v8, err, _ = fit_lut_3d(lambda c: wall_scale * wallgs_inverse_legacy(c), colours, D.WALLGS_UE_GAMMA, weights)
    lut_image_3d(v8).save(OUT_DIR / D.WALLGS_UE_LUT)
    grey = colours[:n_grey, 0]
    ge = err[:n_grey].max(1)
    dark = grey < 0.1
    print(f"WallGS: {OUT_DIR / D.WALLGS_UE_LUT}  gamma {D.WALLGS_UE_GAMMA}  greys mean {ge.mean():.4f} "
          f"max {ge.max():.4f}, darks (<0.1) max {ge[dark].max():.4f} rel {np.max(ge[dark] / grey[dark]):.2f}; "
          f"colours mean {err[n_grey:].mean():.4f} max {err[n_grey:].max():.4f}")

    display = np.concatenate([np.geomspace(1e-3, 0.05, 300), np.linspace(0.05, 0.99, 900)])
    luma_scale = 2.0 ** D.LUMA_UE_EXPOSURE * D.LUMA_EMISSION / D.LUMA_COLOR_MAX
    lv8, lerr = fit_lut(lambda c: luma_scale * c, display)
    lut_image(lv8).save(OUT_DIR / D.LUMA_UE_LUT)
    dark = display < 0.1
    print(f"Luma: {OUT_DIR / D.LUMA_UE_LUT}  mean abs err {lerr.mean():.4f}  max {lerr.max():.4f}  "
          f"darks (<0.1) max {lerr[dark].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
