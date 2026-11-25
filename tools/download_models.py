"""Download the models in tools/model_manifest.json.

Kinds: file (Hugging Face file, then a hardlink into dest), snapshot (a pinned
Hub revision), url (a direct download), bundled (already in the install).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "model_manifest.json"


def hf_cache_root(root: Path) -> Path:
    return root / ".hf_cache"


def apply_hf_cache_env(root: Path) -> Path:
    """Pin Hub + Xet cache to the chosen install before huggingface_hub is imported."""
    cache = hf_cache_root(root)
    (cache / "hub").mkdir(parents=True, exist_ok=True)
    (cache / "xet").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    mirror = os.environ.get("VOTIONSPLAT_HF_ENDPOINT", "").strip()
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
    return cache


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


def select_entries(manifest: dict, download_set: str) -> list[dict]:
    models = manifest["models"]
    if download_set == "core":
        return [entry for entry in models if entry.get("group") == "core"]
    return list(models)


def snapshot_dir(root: Path, entry: dict) -> Path:
    if entry.get("local_dir"):
        return root / entry["local_dir"]
    org, name = entry["repo"].split("/", 1)
    folder = "models--" + org + "--" + name.replace("/", "--")
    return root / ".hf_cache" / "hub" / folder / "snapshots" / entry["revision"]


def is_ready(root: Path, entry: dict) -> bool:
    kind = entry.get("kind")
    if kind in ("file", "url", "bundled"):
        path = root / entry["dest"]
        if not path.is_file():
            return False
        expected = entry.get("expected_bytes")
        if expected is None:
            return path.stat().st_size > 0
        return path.stat().st_size == int(expected)
    if kind == "snapshot":
        base = snapshot_dir(root, entry)
        return all((base / rel).is_file() for rel in entry.get("check_files", []))
    return False


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


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    return token or None


def _mark_ready(entry: dict, note: str) -> str:
    size = int(entry.get("expected_bytes") or 0)
    progress(entry["id"], "start", size)
    progress(entry["id"], size, size, 100)
    progress(entry["id"], "done")
    log(f"{entry['id']}: {note}")
    return "READY"


def fetch_file(entry: dict, root: Path, cache: Path) -> str:
    dest = root / entry["dest"]
    if is_ready(root, entry):
        return _mark_ready(entry, f"already READY {dest}")
    from huggingface_hub import hf_hub_download

    known = int(entry.get("expected_bytes") or 0)
    progress(entry["id"], "start", known)
    log(f"{entry['id']}: Hugging Face {entry['repo']} {entry['filename']}")
    cached = hf_hub_download(
        repo_id=entry["repo"],
        filename=entry["filename"],
        cache_dir=str(cache / "hub"),
        token=_hf_token(),
        tqdm_class=_tqdm_for(entry["id"]),
    )
    kind = copy_or_hardlink(Path(cached), dest)
    if not is_ready(root, entry):
        progress(entry["id"], "fail")
        log(f"MISSING_{entry['id']} {dest}")
        return f"MISSING_{entry['id']}"
    progress(entry["id"], "done")
    log(f"{entry['id']}: {kind} -> {dest}")
    return "READY"


def fetch_snapshot(entry: dict, root: Path, cache: Path) -> str:
    if is_ready(root, entry):
        return _mark_ready(entry, f"already READY {entry['repo']}@{entry['revision'][:12]}")
    from huggingface_hub import snapshot_download

    known = int(round(float(entry.get("size_gb") or 0) * 1_000_000_000))
    progress(entry["id"], "start", known)
    kwargs = {
        "repo_id": entry["repo"],
        "revision": entry["revision"],
        "cache_dir": str(cache / "hub"),
        "token": _hf_token(),
        "tqdm_class": _tqdm_for(entry["id"]),
    }
    patterns = entry.get("allow_patterns") or None
    if patterns:
        kwargs["allow_patterns"] = patterns
    local_dir = entry.get("local_dir")
    if local_dir:
        kwargs["local_dir"] = str(root / local_dir)
    log(f"{entry['id']}: snapshot {entry['repo']}@{entry['revision'][:12]}")
    snapshot_download(**kwargs)
    if not is_ready(root, entry):
        progress(entry["id"], "fail")
        log(f"MISSING_{entry['id']} {entry['repo']}")
        return f"MISSING_{entry['id']}"
    progress(entry["id"], known, known, 100)
    progress(entry["id"], "done")
    return "READY"


def fetch_url(entry: dict, root: Path) -> str:
    dest = root / entry["dest"]
    if is_ready(root, entry):
        return _mark_ready(entry, f"already READY {dest}")
    part = dest.parent / (dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"{entry['id']}: {entry['url']}")
    request = urllib.request.Request(entry["url"])
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or entry.get("expected_bytes") or 0)
        progress(entry["id"], "start", total)
        done = 0
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else -1
                progress(entry["id"], done, total, pct)
    os.replace(part, dest)
    if not is_ready(root, entry):
        progress(entry["id"], "fail")
        log(f"MISSING_{entry['id']} {entry['url']}")
        return f"MISSING_{entry['id']}"
    progress(entry["id"], "done")
    return "READY"


def fetch_bundled(entry: dict, root: Path) -> str:
    dest = root / entry["dest"]
    if is_ready(root, entry):
        return _mark_ready(entry, f"bundled READY {dest}")
    progress(entry["id"], "fail")
    log(f"MISSING {entry['id']} (reinstall VotionSplat)")
    return f"MISSING_{entry['id']}"


def fetch_one(entry: dict, root: Path, cache: Path) -> str:
    kind = entry.get("kind")
    try:
        if kind == "file":
            return fetch_file(entry, root, cache)
        if kind == "snapshot":
            return fetch_snapshot(entry, root, cache)
        if kind == "url":
            return fetch_url(entry, root)
        if kind == "bundled":
            return fetch_bundled(entry, root)
    except Exception as exc:
        progress(entry["id"], "fail")
        log(f"MISSING_{entry['id']} ({type(exc).__name__}: {exc})")
        return f"MISSING_{entry['id']}"
    progress(entry["id"], "fail")
    log(f"MISSING_{entry['id']} unknown kind {kind}")
    return f"MISSING_{entry['id']}"


def gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def warn_disk(root: Path, entries: list[dict]) -> None:
    needed_gb = sum(float(entry.get("size_gb") or 0) for entry in entries) + 20
    free = shutil.disk_usage(root if root.exists() else Path(root.anchor or root)).free
    log(f"dest={root} ({gb(free)} free on that volume)")
    if free < needed_gb * 1_000_000_000:
        log(
            f"WARNING // {gb(free)} free, this set is about "
            f"{needed_gb - 20:.1f} GB plus 20 GB of working space"
        )


def print_status(manifest: dict, root: Path, download_set: str) -> None:
    for entry in select_entries(manifest, download_set):
        token = "READY" if is_ready(root, entry) else f"MISSING_{entry['id']}"
        log(f"{entry['label']}\t{token}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VotionSplat model downloader")
    parser.add_argument("--set", choices=("core", "all"), default="core")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    cache = apply_hf_cache_env(root)
    manifest = load_manifest(args.manifest)
    if args.status_only:
        print_status(manifest, root, args.set)
        return 0

    entries = select_entries(manifest, args.set)
    log(f"STATUS // download set={args.set}")
    warn_disk(root, entries)
    worst = 0
    for entry in entries:
        token = fetch_one(entry, root, cache)
        log(f"{entry['label']}\t{token}")
        if token.startswith("MISSING"):
            worst = 1
    log("STATUS // download finished")
    return worst


if __name__ == "__main__":
    sys.exit(main())
