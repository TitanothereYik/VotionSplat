"""Write outputs/<scene>/pano.png and prompt.txt. Measure final size."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from engine.job import REPO_ROOT


def scene_dir(scene: str) -> Path:
    name = (scene or "alpine_01").strip() or "alpine_01"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    path = REPO_ROOT / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def pano_path(scene: str) -> Path:
    return scene_dir(scene) / "pano.png"


def prompt_path(scene: str) -> Path:
    return scene_dir(scene) / "prompt.txt"


def wan_prompt_path(scene: str) -> Path:
    """Generate-tab WAN prompt. Independent of P1 ``prompt.txt`` concat."""
    return scene_dir(scene) / "_work" / "wan_prompt.txt"


def write_prompt(scene: str, text: str) -> Path:
    path = prompt_path(scene)
    path.write_text(text if text is not None else "", encoding="utf-8")
    extra = wan_prompt_path(scene)
    if extra.is_file():
        extra.unlink()
    return path


def write_measure(scene: str, width: int, height: int, note: str = "") -> Path:
    """Append measured size to the scene log (PRD §4.7). Does not rewrite env_report."""
    path = scene_dir(scene) / "pano_log.txt"
    extra = f" {note}" if note else ""
    line = f"measured {int(width)}x{int(height)}{extra}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return path


def save_pano_image(scene: str, image: Image.Image) -> Path:
    path = pano_path(scene)
    rgb = image.convert("RGB")
    rgb.save(path, format="PNG")
    write_measure(scene, rgb.size[0], rgb.size[1], str(path))
    print(f"STATUS // measured {rgb.size[0]}x{rgb.size[1]} -> {path}", flush=True)
    print(f"PANO\t{path}", flush=True)
    return path


def copy_as_pano(scene: str, source: Path) -> Path:
    with Image.open(source) as im:
        im.load()
        return save_pano_image(scene, im)
