"""3DGS PLY writer — Unreal WallGS Pro / Luma, Postshot, Brush, SuperSplat.

Input is the LiteGS unclustered layout from ``LiteGSBackend.unclustered_numpy()``:
xyz ``[3,N]``, scale (log) ``[3,N]``, rot (wxyz, normalized) ``[4,N]``, sh_0 ``[1,3,N]``,
sh_rest ``[K-1,3,N]``, opacity (logit) ``[1,N]``. Property order follows the reference
Inria 3DGS ``save_ply``: x y z nx ny nz f_dc_0..2 f_rest_0..(3(K-1)-1) opacity
scale_0..2 rot_0..3. ``f_rest`` is channel-major (all coefficients of R, then G, then B)
— the same layout Inria writes via ``_features_rest.transpose(1,2).flatten``.

Colour spaces (``engine.splat.defaults.COLOR_SPACES``):

* ``srgb`` — leave trained SH alone. Matches Brush / Postshot / SuperSplat / PlayCanvas
  (3DGS blends in gamma).
* ``linear`` — sRGB EOTF on evaluated DC, rest scaled by the Jacobian. For engines that
  treat splat RGB as linear albedo. Tagged linear Rec.709 (WallGS reads "Linear Rec.709").
* ``wallgs`` (default) — WallGS Pro: trained sRGB full SH, every Gaussian, untagged so
  WallGS blends in display space like the trainer.
* ``wallgs_acescg`` — WallGS Pro for an OCIO / ACES viewport: the same untagged file as
  ``wallgs`` (a tagged ACEScg file blends in linear light and glitters). Unreal's OCIO Source
  ``D.OCIO_WALLGS_SOURCE`` (tools/make_ocio_config.py) does the colour management.
* ``luma`` — Luma AI Niagara 0.4.1. Its material decodes colour as
  ``0.4 * pow(clamp(rgb), 2.2)`` and blends in linear light. DC is re-encoded so the
  emission is proportional to the trained display colour (``rgb / LUMA_COLOR_MAX``);
  Luma's linear blend then equals the trainer's display-space blend, and Unreal's
  exposure plus ``LUMA_UE_LUT`` turn it back into display colour. Prune
  ``sigmoid(opacity) < UNREAL_MIN_OPACITY``, clamp scale anisotropy, no count cap. Luma's DLL always cuts the splat into 64 spatial chunks; at or below
  ``LUMA_SINGLE_CHUNK_MAX`` the importer builds one Niagara system from chunk 0 alone
  (a central box) and silently drops the rest, so smaller files are padded with
  invisible Gaussians to reach the 64-system path. The Niagara material only
  shades DC, but the importer indexes a fixed degree-3 record, so the file
  always carries 45 ``f_rest`` columns (zeros when the splat has no higher bands).
  Without them the loader assigns every Gaussian scale 1 and a NaN rotation.

Coordinates stay COLMAP/OpenCV; both UE plugins convert on import. Do not
sRGB→linear bake.

``compact=True`` drops Gaussians with ``sigmoid(opacity) < min_opacity``;
``drop_sh_rest=True`` writes a DC-only file (view-independent colour, ~4x smaller).
Numpy only — no torch, so ``tools`` scripts can import it too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.splat import defaults as D

SH_C0 = 0.28209479177387814
LUMA_F_REST = 45  # degree-3 Inria width. Luma 0.4.1 reads scale and rotation past this.
LUMA_GAMMA = 2.2  # Luma material: tc = pow(clamp(ToneColor, 0, 1), 2.2) * LUMA_EMISSION
LUMA_MIN_COUNT = D.LUMA_SINGLE_CHUNK_MAX + 1
LUMA_PAD_LOG_SCALE = -14.0  # sub-pixel sprite
LUMA_PAD_OPACITY = -20.0  # alpha ~2e-9, zero in Luma's f16 colour texture


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_to_linear_grad(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    hi = (2.4 / 1.055) * (((c + 0.055) / 1.055) ** 1.4)
    return np.where(c <= 0.04045, 1.0 / 12.92, hi)


def flatten_sh_rest_inria(shr: np.ndarray) -> np.ndarray:
    """``[K-1, 3, N]`` → ``[N, 3*(K-1)]`` Inria / 3DGS channel-major."""
    n = int(shr.shape[-1])
    if shr.size == 0:
        return np.zeros((n, 0), dtype=np.float32)
    return np.ascontiguousarray(shr.transpose(2, 1, 0).reshape(n, -1))


def clamp_log_anisotropy(log_scale: np.ndarray, max_ratio: float) -> np.ndarray:
    """Fatten degenerate needles: raise the small axes so max/min <= ``max_ratio``.

    ``log_scale`` is ``[N,3]`` (PLY / Inria). Extreme anisotropy sorts as sheets that
    punch through furniture when the host only does a coarse particle depth sort.
    """
    if max_ratio is None or float(max_ratio) <= 1.0:
        return log_scale
    s = np.exp(np.clip(np.asarray(log_scale, dtype=np.float64), -20.0, 20.0))
    floor = s.max(axis=1, keepdims=True) / float(max_ratio)
    return np.log(np.maximum(s, floor).clip(1e-8, None)).astype(np.float32)


def _luma_pad(xyz, scale, rot, sh0, shr_flat, opa, target: int):
    """Append invisible Gaussians up to ``target`` rows. They reuse real positions so the
    chunk grid Luma derives from the bounds does not move."""
    n = int(xyz.shape[0])
    if n == 0 or n >= target:
        return xyz, scale, rot, sh0, shr_flat, opa
    src = np.arange(target - n) % n
    pad = target - n
    rot_pad = np.zeros((pad, 4), dtype=np.float32)
    rot_pad[:, 0] = 1.0
    return (
        np.concatenate([xyz, xyz[src]]),
        np.concatenate([scale, np.full((pad, 3), LUMA_PAD_LOG_SCALE, dtype=np.float32)]),
        np.concatenate([rot, rot_pad]),
        np.concatenate([sh0, np.zeros((pad, sh0.shape[1]), dtype=np.float32)]),
        np.concatenate([shr_flat, np.zeros((pad, shr_flat.shape[1]), dtype=np.float32)]),
        np.concatenate([opa, np.full(pad, LUMA_PAD_OPACITY, dtype=np.float32)]),
    )


def bake_sh_color(
    sh0: np.ndarray,
    shr: np.ndarray,
    color_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap SH so evaluated RGB matches ``color_space``. ``sh0`` is ``[1,3,N]``,
    ``shr`` is ``[K-1,3,N]``. Identity for ``srgb``, ``wallgs`` and ``wallgs_acescg``."""
    space = D.color_space_from_key(color_space)
    sh0 = np.asarray(sh0, dtype=np.float32)
    shr = np.asarray(shr, dtype=np.float32)
    if space in (D.COLOR_SPACE_SRGB, D.COLOR_SPACE_WALLGS, D.COLOR_SPACE_WALLGS_ACESCG, D.COLOR_SPACE_VOTION):
        return sh0, shr
    if space == D.COLOR_SPACE_LUMA:
        rgb = np.asarray(sh0[0], dtype=np.float64) * SH_C0 + 0.5
        enc = np.clip(rgb / D.LUMA_COLOR_MAX, 0.0, 1.0) ** (1.0 / LUMA_GAMMA)
        return ((enc - 0.5) / SH_C0).astype(np.float32)[None, :, :], shr

    dc = np.asarray(sh0[0], dtype=np.float64)  # [3, N]
    rgb = dc * SH_C0 + 0.5  # 3DGS / LiteGS display-referred eval at DC
    rgb_clip = np.clip(rgb, 0.0, 1.0)
    lin = _srgb_to_linear(rgb_clip)
    jac = _srgb_to_linear_grad(rgb_clip)
    dc_out = ((lin - 0.5) / SH_C0).astype(np.float32)
    sh0_out = dc_out[None, :, :].copy()
    if shr.size == 0:
        return sh0_out, shr
    jac_f = jac.astype(np.float32)  # [3, N]
    shr_out = (shr * jac_f[None, :, :]).astype(np.float32)
    return sh0_out, shr_out


def _luma_rest(shr_flat: np.ndarray) -> np.ndarray:
    """Fit ``[N, *]`` rest coefficients to the 45 columns Luma's loader requires."""
    n, width = shr_flat.shape
    if width == LUMA_F_REST:
        return shr_flat
    out = np.zeros((n, LUMA_F_REST), dtype=np.float32)
    if width:
        out[:, : min(width, LUMA_F_REST)] = shr_flat[:, :LUMA_F_REST]
    return out


def write_splat_ply(
    path: Path,
    arrays: dict,
    *,
    compact: bool = False,
    drop_sh_rest: bool = False,
    min_opacity: float = 0.005,
    color_space: str = D.COLOR_SPACE_DEFAULT,
    votion_raster: str | None = None,
    votion_near_plane: float | None = None,
    votion_sh_degree: int | None = None,
) -> tuple[Path, int]:
    """Write ``path`` and return ``(path, gaussians_written)``, not counting Luma padding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    space = D.color_space_from_key(color_space)
    votion = space == D.COLOR_SPACE_VOTION
    if votion:
        if votion_raster not in D.VOTION_RASTERS:
            raise ValueError(f"votion export needs votion_raster in {D.VOTION_RASTERS}, got {votion_raster!r}")
        near = float("nan") if votion_near_plane is None else float(votion_near_plane)
        if not np.isfinite(near) or near <= 0.0:
            raise ValueError("votion export needs a finite votion_near_plane > 0")
        compact = False
        drop_sh_rest = False

    xyz = np.asarray(arrays["xyz"], dtype=np.float32).T  # [N,3]
    scale = np.asarray(arrays["scale"], dtype=np.float32).T  # [N,3]
    rot = np.asarray(arrays["rot"], dtype=np.float32).T  # [N,4]
    sh0 = np.asarray(arrays["sh_0"], dtype=np.float32)  # [1,3,N]
    shr = np.asarray(arrays["sh_rest"], dtype=np.float32)  # [K-1,3,N]
    opa = np.asarray(arrays["opacity"], dtype=np.float32).reshape(-1)  # [N]
    n_total = int(xyz.shape[0])
    sh0, shr = bake_sh_color(sh0, shr, space)
    luma = space == D.COLOR_SPACE_LUMA
    if luma:
        drop_sh_rest = False  # columns must exist even though Niagara shades DC only

    sh0 = np.ascontiguousarray(sh0.transpose(2, 1, 0).reshape(n_total, -1))  # [N,3]
    if drop_sh_rest or shr.size == 0:
        shr_flat = np.zeros((n_total, 0), dtype=np.float32)
    else:
        shr_flat = flatten_sh_rest_inria(shr)

    votion_deg = 0
    if votion:
        k_minus_1 = int(shr_flat.shape[1] // 3)
        deg_avail = int(round((k_minus_1 + 1) ** 0.5)) - 1
        if (deg_avail + 1) ** 2 - 1 != k_minus_1 or deg_avail < 0:
            raise ValueError(f"votion export has {shr_flat.shape[1]} f_rest columns, not 3*((d+1)^2-1)")
        votion_deg = deg_avail if votion_sh_degree is None else min(int(votion_sh_degree), deg_avail)

    keep = np.ones(n_total, dtype=bool)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1) & np.isfinite(opa)
    keep &= finite
    opa_cut = float(min_opacity)
    prune_opa = compact
    if luma:
        opa_cut = max(opa_cut, float(D.UNREAL_MIN_OPACITY))
        prune_opa = True
    if prune_opa:
        keep &= _sigmoid(opa) >= opa_cut
    if not keep.all():
        xyz, scale, rot, sh0, shr_flat, opa = (
            xyz[keep],
            scale[keep],
            rot[keep],
            sh0[keep],
            shr_flat[keep],
            opa[keep],
        )
    n_real = int(xyz.shape[0])
    if luma:
        scale = clamp_log_anisotropy(scale, D.UNREAL_MAX_ANISOTROPY)
        shr_flat = _luma_rest(shr_flat)
        xyz, scale, rot, sh0, shr_flat, opa = _luma_pad(xyz, scale, rot, sh0, shr_flat, opa, LUMA_MIN_COUNT)
    n = int(xyz.shape[0])

    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = rot / np.clip(norm, 1e-8, None)

    n_rest = int(shr_flat.shape[1])
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(n_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype = np.dtype([(name, "<f4") for name in names])
    verts = np.empty(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["nx"] = 0.0
    verts["ny"] = 0.0
    verts["nz"] = 0.0
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    for i in range(n_rest):
        verts[f"f_rest_{i}"] = shr_flat[:, i]
    verts["opacity"] = opa
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    if votion:
        tags = (
            f"comment votion.version={D.VOTION_TAG_VERSION}",
            f"comment votion.raster={votion_raster}",
            f"comment votion.near_plane={float(votion_near_plane)!r}",
            f"comment votion.sh_degree={votion_deg}",
        )
    else:
        tags = D.COLOR_SPACE_TAGS.get(space, ())
    header = ["ply", "format binary_little_endian 1.0", *tags]
    header.append(f"element vertex {n}")
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(verts.tobytes())
    tmp.replace(path)
    return path, n_real


def _wallgs_source_space(lines: list[str]) -> str:
    """WallGS Pro 1.1.4 ParsePlyColorSpace (GaussianSplatLibrary.cpp), for the self-test."""
    prim = next((ln.split("=", 1)[1].strip() for ln in lines
                 if ln.startswith("comment postshot.color.primaries=")), "")
    eotf = next((ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("comment postshot.color.eotf=")), "")
    if not prim and not eotf:
        return "Legacy Appearance (untagged)"
    rec709 = prim.lower() in ("srgb", "rec709", "rec.709")
    if prim.lower() == "acescg" and eotf.lower() == "linear":
        return "ACEScg (linear)"
    if rec709 and eotf.lower() == "linear":
        return "Linear Rec.709"
    if rec709 and eotf.lower() == "srgb":
        return "sRGB (encoded)"
    return "import error"


def read_ply_header(path: Path) -> dict:
    """Vertex count and whether f_rest bands are present (for the Splat page hint)."""
    info = {"vertices": 0, "f_rest": 0}
    try:
        with Path(path).open("rb") as f:
            for _ in range(200):
                line = f.readline().decode("ascii", errors="replace").strip()
                if line.startswith("element vertex"):
                    info["vertices"] = int(line.split()[-1])
                elif line.startswith("property float f_rest_"):
                    info["f_rest"] += 1
                elif line == "end_header" or not line:
                    break
    except OSError:
        pass
    return info


def load_splat_ply_arrays(path: Path) -> dict:
    """Read an Inria splat PLY into the LiteGS unclustered layout for ``write_splat_ply``."""
    path = Path(path)
    with path.open("rb") as f:
        names: list[str] = []
        n = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float "):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        offset = f.tell()
    mm = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(n, len(names)))
    idx = {name: i for i, name in enumerate(names)}

    def col(*keys: str) -> np.ndarray:
        return np.column_stack([np.asarray(mm[:, idx[k]], dtype=np.float32) for k in keys])

    rest_names = [k for k in names if k.startswith("f_rest_")]
    rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    xyz = col("x", "y", "z").T
    scale = col("scale_0", "scale_1", "scale_2").T
    rot = col("rot_0", "rot_1", "rot_2", "rot_3").T
    dc = col("f_dc_0", "f_dc_1", "f_dc_2")
    sh0 = dc.T[None, :, :].copy()
    if rest_names:
        shr_flat = col(*rest_names)
        bands = shr_flat.shape[1] // 3
        shr = shr_flat.reshape(n, 3, bands).transpose(2, 1, 0).copy()
    else:
        shr = np.zeros((0, 3, n), dtype=np.float32)
    opa = np.asarray(mm[:, idx["opacity"]], dtype=np.float32).reshape(1, n).copy()
    return {"xyz": xyz, "scale": scale, "rot": rot, "sh_0": sh0, "sh_rest": shr, "opacity": opa}


def _self_test() -> None:
    # Inria channel-major: [K-1, 3, N] → R0,R1,... G0,G1,... B0,B1,...
    shr = np.zeros((2, 3, 1), dtype=np.float32)
    shr[0, 0, 0] = 1.0  # rest coeff 0, R
    shr[1, 1, 0] = 2.0  # rest coeff 1, G
    flat = flatten_sh_rest_inria(shr)
    assert flat.shape == (1, 6)
    assert abs(float(flat[0, 0]) - 1.0) < 1e-6  # R coeff 0
    assert abs(float(flat[0, 1]) - 0.0) < 1e-6  # R coeff 1
    assert abs(float(flat[0, 3]) - 2.0) < 1e-6  # G coeff 1
    # sRGB bake is identity
    sh0 = np.zeros((1, 3, 4), dtype=np.float32)
    sh0[0, :, :] = ((np.array([0.2, 0.5, 0.8]) - 0.5) / SH_C0)[:, None]
    shr0 = np.ones((2, 3, 4), dtype=np.float32) * 0.01
    a, b = bake_sh_color(sh0, shr0, "srgb")
    assert np.allclose(a, sh0) and np.allclose(b, shr0)
    wa, wb = bake_sh_color(sh0, shr0, "wallgs")
    assert np.allclose(wa, sh0) and np.allclose(wb, shr0)
    assert D.color_space_from_key("unreal") == D.COLOR_SPACE_WALLGS
    la, lb = bake_sh_color(sh0, shr0, "luma")
    emission = D.LUMA_EMISSION * np.clip(la[0, :, 0] * SH_C0 + 0.5, 0, 1) ** LUMA_GAMMA
    scene = emission * 2.0 ** D.LUMA_UE_EXPOSURE
    assert np.allclose(scene, np.array([0.2, 0.5, 0.8]), atol=3e-3) and np.allclose(lb, shr0)
    # Linear bake of mid-grey DC (sRGB 0.5 → linear ~0.214) packs back darker SH
    sh_mid = np.zeros((1, 3, 1), dtype=np.float32)  # rgb = 0.5
    out, _ = bake_sh_color(sh_mid, np.zeros((0, 3, 1), dtype=np.float32), "linear")
    rgb_lin = float(out[0, 0, 0] * SH_C0 + 0.5)
    assert 0.20 < rgb_lin < 0.23, rgb_lin
    # OCIO ACES view preset: trained colour untouched (WallGS must blend it in display space)
    assert D.color_space_from_key("acescg") == D.COLOR_SPACE_WALLGS_ACESCG
    ashr = np.full((3, 3, 3), 0.02, dtype=np.float32)
    aa, ab = bake_sh_color(sh0, ashr, "wallgs_acescg")
    assert np.array_equal(aa, sh0) and np.array_equal(ab, ashr)
    needle = np.log(np.array([[1.0, 1e-6, 1e-6]], dtype=np.float64))
    fat = clamp_log_anisotropy(needle, 64.0)
    ratio = float(np.exp(fat).max() / np.exp(fat).min())
    assert abs(ratio - 64.0) < 0.1, ratio
    n = 4
    arrays = {
        "xyz": np.zeros((3, n), dtype=np.float32),
        "scale": np.zeros((3, n), dtype=np.float32),
        "rot": np.array([[1, 0, 0, 0]] * n, dtype=np.float32).T,
        "sh_0": np.repeat(sh0[:, :, :1], n, axis=2),
        "sh_rest": np.zeros((15, 3, n), dtype=np.float32),
        "opacity": np.zeros((1, n), dtype=np.float32),
    }
    arrays["sh_rest"][0, 0, 0] = 0.07
    arrays["opacity"][0, 0] = float(np.log(0.001 / 0.999))  # below UNREAL_MIN_OPACITY
    arrays["scale"][:, 1] = np.log(np.array([1.0, 1e-6, 1e-6], dtype=np.float32))
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        wall = Path(td) / "wallgs.ply"
        out, written = write_splat_ply(wall, arrays, color_space="wallgs")
        assert written == n  # every Gaussian kept
        info = read_ply_header(out)
        assert info["vertices"] == n and info["f_rest"] == 45
        head = out.read_bytes()[:2048].decode("ascii", errors="replace").splitlines()
        assert not any(h.startswith("comment ") for h in head)  # tags would switch WallGS to linear blend
        loaded = load_splat_ply_arrays(out)
        rgb_keep = loaded["sh_0"][0, :, 0] * SH_C0 + 0.5
        assert np.allclose(rgb_keep, np.array([0.2, 0.5, 0.8]), atol=1e-5)
        assert np.allclose(loaded["scale"], arrays["scale"])  # no needle clamp
        out_a, written_a = write_splat_ply(Path(td) / "acescg.ply", arrays, color_space="wallgs_acescg")
        assert written_a == n and out_a.read_bytes() == out.read_bytes()  # same file as wallgs
        for space, reads in D.WALLGS_READS.items():
            out_s, _ = write_splat_ply(Path(td) / f"{space}.ply", arrays, color_space=space)
            lines = out_s.read_bytes()[:2048].decode("ascii", errors="replace").splitlines()
            assert _wallgs_source_space(lines) == reads, (space, reads)
        global LUMA_MIN_COUNT
        saved_min = LUMA_MIN_COUNT
        LUMA_MIN_COUNT = 8
        try:
            luma = Path(td) / "luma.ply"
            out_l, written_l = write_splat_ply(luma, arrays, color_space="luma")
            assert written_l == n - 1  # faint Gaussian dropped
            info_l = read_ply_header(out_l)
            assert info_l["vertices"] == 8 and info_l["f_rest"] == LUMA_F_REST
            padded = load_splat_ply_arrays(out_l)
            sc = np.exp(padded["scale"][:, : n - 1])
            assert float((sc.max(axis=0) / sc.min(axis=0)).max()) <= 64.1
            assert np.allclose(padded["xyz"][:, n - 1 :], padded["xyz"][:, : 8 - (n - 1)])
            assert float(_sigmoid(padded["opacity"][0, n - 1 :]).max()) < 1e-8
            bare = dict(arrays)
            bare["sh_rest"] = np.zeros((0, 3, n), dtype=np.float32)
            out_b, _ = write_splat_ply(Path(td) / "luma_dc.ply", bare, color_space="luma")
            assert read_ply_header(out_b)["f_rest"] == LUMA_F_REST
        finally:
            LUMA_MIN_COUNT = saved_min
        rng = np.random.default_rng(0)
        vn = 1000
        v_rot = rng.standard_normal((4, vn)).astype(np.float32)
        v_rot /= np.clip(np.linalg.norm(v_rot, axis=0, keepdims=True), 1e-8, None)
        v_arrays = {
            "xyz": rng.standard_normal((3, vn)).astype(np.float32),
            "scale": (rng.standard_normal((3, vn)) * 0.1).astype(np.float32),
            "rot": v_rot,
            "sh_0": (rng.standard_normal((1, 3, vn)) * 0.1).astype(np.float32),
            "sh_rest": (rng.standard_normal((15, 3, vn)) * 0.05).astype(np.float32),
            "opacity": rng.standard_normal((1, vn)).astype(np.float32),
        }
        vpath, vwritten = write_splat_ply(
            Path(td) / "votion.ply", v_arrays, color_space="votion",
            votion_raster="gsplat_antialiased", votion_near_plane=0.01, votion_sh_degree=3,
        )
        assert vwritten == vn
        vlines = vpath.read_bytes().split(b"end_header")[0].decode("ascii").splitlines()
        assert vlines[:6] == [
            "ply", "format binary_little_endian 1.0",
            "comment votion.version=1",
            "comment votion.raster=gsplat_antialiased",
            "comment votion.near_plane=0.01",
            "comment votion.sh_degree=3",
        ]
        loaded_v = load_splat_ply_arrays(vpath)
        assert np.array_equal(loaded_v["xyz"], v_arrays["xyz"])
        assert np.array_equal(loaded_v["scale"], v_arrays["scale"])
        assert np.array_equal(loaded_v["opacity"], v_arrays["opacity"])
        assert np.array_equal(loaded_v["sh_0"], v_arrays["sh_0"])
        assert np.array_equal(loaded_v["sh_rest"], v_arrays["sh_rest"])
        assert np.allclose(loaded_v["rot"], v_arrays["rot"], atol=1e-6)
        try:
            write_splat_ply(Path(td) / "bad.ply", v_arrays, color_space="votion")
            raise AssertionError("missing votion_raster should raise")
        except ValueError:
            pass
        va, vb = bake_sh_color(v_arrays["sh_0"], v_arrays["sh_rest"], "votion")
        assert np.array_equal(va, v_arrays["sh_0"]) and np.array_equal(vb, v_arrays["sh_rest"])
    print("ply_export self-test OK")


if __name__ == "__main__":
    _self_test()
