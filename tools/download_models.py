"""Copy-if-exists from ComfyUI models, else Hugging Face, else MISSING_<id> + Index URL.

Lookup order per docs/Votion_3DGS_2.0_Phase0.md. No invented file sizes.
Free-space check for WAN 14B uses Hugging Face metadata (measured), not a guessed GB.

Progress lines (stdout, for the Home status bars):

    PROGRESS\\t<id>\\tstart\\t<total_bytes>   total 0 if Hub did not report a size
    PROGRESS\\t<id>\\t<bytes>\\t<total>\\t<pct>  pct is -1 when total is unknown
    PROGRESS\\t<id>\\tdone
    PROGRESS\\t<id>\\tfail
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "model_manifest.json"
DEFAULT_COMFY = Path(r"D:\ComfyUI_windows_portable_360")
LEGACY_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_root(root: Path) -> Path:
    return root / ".hf_cache"


def apply_hf_cache_env(root: Path) -> Path:
    """Pin Hub + Xet cache to the repo drive before huggingface_hub is imported.

    Default Hub cache is C:\\Users\\<user>\\.cache\\huggingface — a different
    volume from D:\\Votion3DGS\\models. Hugging Face then warns (or fills C:)
    even when D: has hundreds of GB free.
    """
    cache = hf_cache_root(root)
    (cache / "hub").mkdir(parents=True, exist_ok=True)
    (cache / "xet").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return cache


apply_hf_cache_env(REPO_ROOT)


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(file_id: str, *fields: object) -> None:
    """Machine-readable progress for the Qt status grid. Do not invent totals."""
    parts = [str(file_id), *[str(f) for f in fields]]
    print("PROGRESS\t" + "\t".join(parts), flush=True)


class FileProgressTqdm:
    """Minimal tqdm-compatible bar. Writes PROGRESS lines to stdout only."""

    file_id = "unknown"
    _last_pct = -2
    _last_t = 0.0

    def __init__(self, iterable=None, *, total=None, initial=0, **kwargs) -> None:
        self.iterable = iterable
        self.total = int(total) if total else 0
        self.n = int(initial or 0)
        self.disable = False
        self.desc = kwargs.get("desc") or ""
        self.leave = bool(kwargs.get("leave", False))
        self.postfix = None
        self.unit = kwargs.get("unit") or "B"

    @property
    def format_dict(self) -> dict:
        return {"rate": None, "n": self.n, "total": self.total}

    def refresh(self, *args, **kwargs) -> None:
        total = int(self.total or 0)
        pct = int(100 * self.n / total) if total else -1
        progress(self.file_id, self.n, total, pct)

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            self.update(1)
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def update(self, n: int = 1):
        self.n += int(n or 0)
        total = int(self.total or 0)
        pct = int(100 * self.n / total) if total else -1
        now = time.monotonic()
        if pct != type(self)._last_pct or (now - type(self)._last_t) >= 0.25:
            type(self)._last_pct = pct
            type(self)._last_t = now
            progress(self.file_id, self.n, total, pct)

    def close(self) -> None:
        return None

    def clear(self, *args, **kwargs) -> None:
        return None

    def display(self, *args, **kwargs) -> None:
        return None

    def set_description(self, desc=None, *args, **kwargs) -> None:
        self.desc = desc or ""

    def set_description_str(self, desc=None, *args, **kwargs) -> None:
        self.desc = desc or ""

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def set_postfix_str(self, *args, **kwargs) -> None:
        return None

    def reset(self, total=None) -> None:
        self.n = 0
        if total is not None:
            self.total = int(total) if total else 0
        type(self)._last_pct = -2

    def unpause(self) -> None:
        return None


def _tqdm_for(file_id: str) -> type:
    class Bound(FileProgressTqdm):
        pass

    Bound.file_id = file_id
    Bound._last_pct = -2
    Bound._last_t = 0.0
    return Bound


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dest_path(root: Path, entry: dict) -> Path:
    return root / "models" / entry["local_relative_path"]


def comfy_path(comfy_root: Path, entry: dict) -> Path:
    folder = Path(entry["local_relative_path"]).parts[0]
    return comfy_root / "ComfyUI" / "models" / folder / entry["filename"]


def is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def copy_or_hardlink(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        if src.stat().st_dev == dest.parent.stat().st_dev:
            os.link(src, dest)
            return "hardlink"
    except OSError:
        pass
    shutil.copy2(src, dest)
    return "copy"


def select_entries(manifest: dict, download_set: str) -> list[dict]:
    if download_set == "p1":
        wanted = "p1"
    elif download_set == "p2":
        wanted = "p2"
    else:
        wanted = "all"
    return [e for e in manifest["files"] if wanted in e.get("sets", [])]


def status_token(entry: dict, root: Path) -> str:
    if is_ready(dest_path(root, entry)):
        return "READY"
    return f"MISSING_{entry['id']}"


def hf_size_bytes(entry: dict) -> int | None:
    """Measured size from Hub metadata. Returns None if Hub does not report it."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        url = hf_hub_url(
            repo_id=entry["repo"],
            filename=entry["hf_filename"],
            revision="main",
        )
        meta = get_hf_file_metadata(url)
        size = getattr(meta, "size", None)
        if size is None:
            return None
        return int(size)
    except Exception as exc:
        log(f"size lookup failed for {entry['id']}: {type(exc).__name__}: {exc}")
        return None


def volume_free_bytes(path: Path) -> int:
    """Free bytes on the volume that `path` lives on (or will live on)."""
    probe = path if path.exists() else Path(path.anchor) if path.anchor else path.parent
    return shutil.disk_usage(probe).free


def gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def log_disk_map(root: Path, cache: Path) -> None:
    dest = root / "models"
    dest_free = volume_free_bytes(dest)
    cache_free = volume_free_bytes(cache)
    c_free = volume_free_bytes(Path("C:/"))
    log(f"HF cache={cache} ({gb(cache_free)} free on that volume)")
    log(f"models dest={dest} ({gb(dest_free)} free on that volume)")
    log(
        f"C: {gb(c_free)} free - Hub is not writing to "
        f"{Path.home() / '.cache' / 'huggingface'}"
    )


def ensure_free_space(dest: Path, cache: Path, needed: int, entry_id: str) -> None:
    dest_free = volume_free_bytes(dest)
    cache_free = volume_free_bytes(cache)
    log(
        f"disk dest={gb(dest_free)} cache={gb(cache_free)}; "
        f"Hub size for {entry_id}={needed} bytes ({gb(needed)})"
    )
    if dest_free < needed:
        raise RuntimeError(
            f"not enough free space on the models drive for {entry_id}: "
            f"need {gb(needed)}, have {gb(dest_free)} ({dest})"
        )
    if cache_free < needed:
        raise RuntimeError(
            f"not enough free space on the Hugging Face cache drive for {entry_id}: "
            f"need {gb(needed)}, have {gb(cache_free)} ({cache})"
        )


_orig_showwarning = warnings.showwarning


def _hf_disk_warning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if "Not enough free disk space" in text:
        log(
            "Hugging Face cache warning (check the path in this message — "
            f"it is not the D: models folder): {text}"
        )
        return
    _orig_showwarning(message, category, filename, lineno, file=file, line=line)


def lookup_cached_hf(entry: dict, cache_dirs: list[Path]) -> Path | None:
    """Reuse a blob already on disk (legacy C: cache or this repo's cache)."""
    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    except Exception:
        return None
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        try:
            hit = try_to_load_from_cache(
                repo_id=entry["repo"],
                filename=entry["hf_filename"],
                cache_dir=str(cache_dir),
                revision="main",
            )
        except Exception:
            continue
        if hit is None or hit is _CACHED_NO_EXIST:
            continue
        path = Path(str(hit))
        if is_ready(path):
            return path
    return None


def download_hf(entry: dict, dest: Path, cache: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    hub_dir = cache / "hub"
    previous = warnings.showwarning
    warnings.showwarning = _hf_disk_warning
    try:
        cached = hf_hub_download(
            repo_id=entry["repo"],
            filename=entry["hf_filename"],
            revision="main",
            cache_dir=str(hub_dir),
            tqdm_class=_tqdm_for(entry["id"]),
        )
    finally:
        warnings.showwarning = previous
    kind = copy_or_hardlink(Path(cached), dest)
    log(f"{entry['id']}: HF {kind} -> {dest}")


def fetch_one(entry: dict, root: Path, comfy_root: Path, cache: Path) -> str:
    dest = dest_path(root, entry)

    if is_ready(dest):
        size = dest.stat().st_size
        progress(entry["id"], "start", size)
        progress(entry["id"], size, size, 100)
        progress(entry["id"], "done")
        log(f"{entry['id']}: already READY {dest}")
        return "READY"

    src = comfy_path(comfy_root, entry)
    if is_ready(src):
        size = src.stat().st_size
        progress(entry["id"], "start", size)
        kind = copy_or_hardlink(src, dest)
        progress(entry["id"], size, size, 100)
        progress(entry["id"], "done")
        log(f"{entry['id']}: Comfy {kind} {src} -> {dest}")
        return "READY"

    reuse = lookup_cached_hf(entry, [cache / "hub", LEGACY_HF_HUB])
    if reuse is not None:
        size = reuse.stat().st_size
        progress(entry["id"], "start", size)
        kind = copy_or_hardlink(reuse, dest)
        progress(entry["id"], size, size, 100)
        progress(entry["id"], "done")
        log(f"{entry['id']}: cache {kind} {reuse} -> {dest}")
        return "READY"

    known = hf_size_bytes(entry)
    total = int(known) if known else 0
    progress(entry["id"], "start", total)
    log(f"{entry['id']}: not in Comfy ({src}); Hugging Face {entry['url']}")
    if entry.get("check_free_space"):
        if known is None:
            progress(entry["id"], "fail")
            raise RuntimeError(
                f"WAN 14B size unknown from Hub; aborting {entry['id']} "
                f"(Index URL {entry['url']})"
            )
        ensure_free_space(dest, cache, known, entry["id"])

    try:
        download_hf(entry, dest, cache)
    except Exception as exc:
        progress(entry["id"], "fail")
        log(
            f"MISSING_{entry['id']} {entry['url']} "
            f"({type(exc).__name__}: {exc})"
        )
        return f"MISSING_{entry['id']}"

    if is_ready(dest):
        size = dest.stat().st_size
        progress(entry["id"], size, total or size, 100)
        progress(entry["id"], "done")
        return "READY"
    progress(entry["id"], "fail")
    log(f"MISSING_{entry['id']} {entry['url']}")
    return f"MISSING_{entry['id']}"


def print_status(manifest: dict, root: Path, download_set: str) -> None:
    for entry in select_entries(manifest, download_set):
        log(f"{entry['label']}\t{status_token(entry, root)}")
    if download_set == "p1":
        moge = next(e for e in manifest["files"] if e["id"] == "moge")
        if is_ready(dest_path(root, moge)):
            log(f"{moge['label']}\tREADY")
        else:
            log(f"{moge['label']}\tP2 subset")
    for row in manifest.get("info_rows", []):
        log(f"{row['label']}\t{row['note']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion 3DGS model downloader")
    parser.add_argument("--set", choices=("p1", "p2", "all"), default="p1")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    cache = apply_hf_cache_env(root)
    manifest = load_manifest(args.manifest)
    comfy_root = args.comfy_root

    if args.status_only:
        print_status(manifest, root, args.set)
        return 0

    log(f"STATUS // download set={args.set}")
    log(f"dest={root / 'models'}")
    log(f"comfy={comfy_root / 'ComfyUI' / 'models'}")
    log_disk_map(root, cache)

    worst = 0
    for entry in select_entries(manifest, args.set):
        token = fetch_one(entry, root, comfy_root, cache)
        log(f"{entry['label']}\t{token}")
        if token.startswith("MISSING_"):
            worst = 1

    log("STATUS // download finished")
    return worst


if __name__ == "__main__":
    sys.exit(main())
