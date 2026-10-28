"""3DGS PLY writer — Unreal MLSLabs / Luma, Postshot, Brush, SuperSplat.

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
  treat splat RGB as linear albedo.
* ``unreal`` (default) — MLSLabs CUDA 3DGS: trained sRGB full SH, prune
  ``sigmoid(opacity) < UNREAL_MIN_OPACITY``, clamp scale anisotropy. No count cap.
* ``luma`` — Luma AI Niagara: trained sRGB DC (plugin ignores ``f_rest``), same prune
  and needle clamp, keep at most ``LUMA_MAX_GAUSSIANS`` so Luma uses one Niagara
  system (above 1.9M it splits into 64 spatial chunks that do not depth-sort).

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


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


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


def keep_highest_opacity(logit: np.ndarray, n_max: int) -> np.ndarray:
    """Boolean mask keeping the ``n_max`` largest ``sigmoid(opacity)`` values."""
    n = int(np.asarray(logit).shape[0])
    n_max = int(n_max)
    if n_max <= 0 or n <= n_max:
        return np.ones(n, dtype=bool)
    order = np.argpartition(_sigmoid(logit), n - n_max)
    keep = np.zeros(n, dtype=bool)
    keep[order[n - n_max :]] = True
    return keep


def bake_sh_color(
    sh0: np.ndarray,
    shr: np.ndarray,
    color_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap SH so evaluated RGB matches ``color_space``. ``sh0`` is ``[1,3,N]``,
    ``shr`` is ``[K-1,3,N]``. Identity for ``srgb``, ``unreal``, and ``luma``."""
    space = D.color_space_from_key(color_space)
    sh0 = np.asarray(sh0, dtype=np.float32)
    shr = np.asarray(shr, dtype=np.float32)
    if space in (D.COLOR_SPACE_SRGB, D.COLOR_SPACE_UNREAL, D.COLOR_SPACE_LUMA):
        return sh0, shr

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


def unbake_linear_contrast_dc(
    dc: np.ndarray,
    shr_flat: np.ndarray,
    contrast: float = D.MLSLABS_CONTRAST,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the old Unreal sRGB→linear + contrast bake on PLY ``f_dc`` / ``f_rest``.

    ``dc`` is ``[N,3]``, ``shr_flat`` is Inria channel-major ``[N, 3*(K-1)]``.
    """
    contrast = float(contrast) if float(contrast) > 1e-6 else 1.0
    rgb_out = np.asarray(dc, dtype=np.float64) * SH_C0 + 0.5
    lin = (np.clip(rgb_out, 0.0, 1.0) - 0.5) * contrast + 0.5
    lin = np.maximum(lin, 0.0)
    srgb = _linear_to_srgb(lin)
    jac = _srgb_to_linear_grad(srgb) / contrast
    dc_o = ((srgb - 0.5) / SH_C0).astype(np.float32)
    if shr_flat.size == 0:
        return dc_o, shr_flat
    n, width = shr_flat.shape
    bands = width // 3
    shr = np.asarray(shr_flat, dtype=np.float32).reshape(n, 3, bands)
    shr_o = shr / np.clip(jac.astype(np.float32)[:, :, None], 1e-6, None)
    return dc_o, shr_o.reshape(n, width).astype(np.float32)


def write_splat_ply(
    path: Path,
    arrays: dict,
    *,
    compact: bool = False,
    drop_sh_rest: bool = False,
    min_opacity: float = 0.005,
    color_space: str = D.COLOR_SPACE_DEFAULT,
) -> tuple[Path, int]:
    """Write ``path`` and return ``(path, gaussians_written)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    space = D.color_space_from_key(color_space)

    xyz = np.asarray(arrays["xyz"], dtype=np.float32).T  # [N,3]
    scale = np.asarray(arrays["scale"], dtype=np.float32).T  # [N,3]
    rot = np.asarray(arrays["rot"], dtype=np.float32).T  # [N,4]
    sh0 = np.asarray(arrays["sh_0"], dtype=np.float32)  # [1,3,N]
    shr = np.asarray(arrays["sh_rest"], dtype=np.float32)  # [K-1,3,N]
    opa = np.asarray(arrays["opacity"], dtype=np.float32).reshape(-1)  # [N]
    n_total = int(xyz.shape[0])
    sh0, shr = bake_sh_color(sh0, shr, space)
    ue = space in D.UE_ENGINE_SPACES
    if space == D.COLOR_SPACE_LUMA:
        drop_sh_rest = True  # Luma Niagara only stores DC RGB + opacity

    sh0 = np.ascontiguousarray(sh0.transpose(2, 1, 0).reshape(n_total, -1))  # [N,3]
    if drop_sh_rest or shr.size == 0:
        shr_flat = np.zeros((n_total, 0), dtype=np.float32)
    else:
        shr_flat = flatten_sh_rest_inria(shr)

    keep = np.ones(n_total, dtype=bool)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1) & np.isfinite(opa)
    keep &= finite
    opa_cut = float(min_opacity)
    prune_opa = compact
    if ue:
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
    if space == D.COLOR_SPACE_LUMA:
        cap = keep_highest_opacity(opa, D.LUMA_MAX_GAUSSIANS)
        if not cap.all():
            xyz, scale, rot, sh0, shr_flat, opa = (
                xyz[cap],
                scale[cap],
                rot[cap],
                sh0[cap],
                shr_flat[cap],
                opa[cap],
            )
    if ue:
        scale = clamp_log_anisotropy(scale, D.UNREAL_MAX_ANISOTROPY)
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

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(verts.tobytes())
    tmp.replace(path)
    return path, n


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


def repair_baked_unreal_ply(path: Path) -> tuple[Path, int]:
    """Undo the old linear+contrast SH bake, then rewrite with current Unreal hygiene."""
    arrays = load_splat_ply_arrays(path)
    n = int(arrays["sh_0"].shape[-1])
    dc = arrays["sh_0"][0].T.copy()
    shr_flat = flatten_sh_rest_inria(arrays["sh_rest"])
    dc, shr_flat = unbake_linear_contrast_dc(dc, shr_flat)
    arrays["sh_0"] = dc.T[None, :, :].astype(np.float32)
    if shr_flat.size:
        bands = shr_flat.shape[1] // 3
        arrays["sh_rest"] = shr_flat.reshape(n, 3, bands).transpose(2, 1, 0).astype(np.float32)
    return write_splat_ply(path, arrays, color_space=D.COLOR_SPACE_UNREAL)


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
    ua, ub = bake_sh_color(sh0, shr0, "unreal")
    assert np.allclose(ua, sh0) and np.allclose(ub, shr0)
    la, lb = bake_sh_color(sh0, shr0, "luma")
    assert np.allclose(la, sh0) and np.allclose(lb, shr0)
    # Linear bake of mid-grey DC (sRGB 0.5 → linear ~0.214) packs back darker SH
    sh_mid = np.zeros((1, 3, 1), dtype=np.float32)  # rgb = 0.5
    out, _ = bake_sh_color(sh_mid, np.zeros((0, 3, 1), dtype=np.float32), "linear")
    rgb_lin = float(out[0, 0, 0] * SH_C0 + 0.5)
    assert 0.20 < rgb_lin < 0.23, rgb_lin
    needle = np.log(np.array([[1.0, 1e-6, 1e-6]], dtype=np.float64))
    fat = clamp_log_anisotropy(needle, 64.0)
    ratio = float(np.exp(fat).max() / np.exp(fat).min())
    assert abs(ratio - 64.0) < 0.1, ratio
    cap = keep_highest_opacity(np.array([-2.0, 0.0, 2.0, 4.0, -1.0], dtype=np.float32), 3)
    assert cap.sum() == 3 and bool(cap[1] and cap[2] and cap[3])

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
        mls = Path(td) / "mls.ply"
        out, written = write_splat_ply(mls, arrays, color_space="unreal")
        assert written == n - 1  # faint Gaussian dropped
        info = read_ply_header(out)
        assert info["vertices"] == n - 1 and info["f_rest"] == 45
        loaded = load_splat_ply_arrays(out)
        rgb_keep = loaded["sh_0"][0, :, 0] * SH_C0 + 0.5
        assert np.allclose(rgb_keep, np.array([0.2, 0.5, 0.8]), atol=1e-5)
        sc = np.exp(loaded["scale"][:, 0])
        assert float(sc.max() / max(float(sc.min()), 1e-8)) <= 64.1
        luma = Path(td) / "luma.ply"
        out_l, written_l = write_splat_ply(luma, arrays, color_space="luma")
        assert written_l == n - 1
        info_l = read_ply_header(out_l)
        assert info_l["vertices"] == n - 1 and info_l["f_rest"] == 0
    print("ply_export self-test OK")


if __name__ == "__main__":
    _self_test()
