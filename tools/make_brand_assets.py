"""Copy the finished art into app/assets and rebuild the icon and installer images.

Sources, when present:
  Logo_and_Banners/votion-logo-091-rounded.png   app icon
  Logo_and_Banners/VotionSplat_Banner_Long.png   top of the app window
  Logo_and_Banners/votion-banner-seed-2.png      GitHub page and installer
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"
SOURCES = ROOT / "Logo_and_Banners"
ICON = ASSETS / "icon.png"
BANNER = ASSETS / "banner.png"
REPO_BANNER = ASSETS / "repo_banner.png"
ICO = ASSETS / "icon.ico"
WIZARD = ASSETS / "installer_wizard.png"
SMALL = ASSETS / "installer_small.png"
# Inno's welcome panel keeps a 164:314 frame. 534x1022 is the 250% size.
WIZARD_SIZE = (534, 1022)


def _take(source: Path, dest: Path) -> None:
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    elif not dest.is_file():
        raise SystemExit(f"missing {source}")


def _write_ico() -> None:
    image = Image.open(ICON).convert("RGBA")
    image.save(
        ICO,
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def _write_wizard() -> None:
    banner = Image.open(REPO_BANNER).convert("RGB")
    canvas = Image.new("RGB", WIZARD_SIZE, (0, 0, 0))
    width = WIZARD_SIZE[0]
    height = max(1, round(width * banner.height / banner.width))
    fitted = banner.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (0, (WIZARD_SIZE[1] - height) // 2))
    canvas.save(WIZARD)


def _write_small() -> None:
    image = Image.open(ICON).convert("RGBA")
    image.resize((164, 164), Image.Resampling.LANCZOS).save(SMALL)


def _copy_mockup() -> None:
    media = ROOT / "site_mockup" / "media"
    if not media.is_dir():
        return
    for source in (ICON, BANNER, REPO_BANNER):
        (media / source.name).write_bytes(source.read_bytes())


def main() -> int:
    _take(SOURCES / "votion-logo-091-rounded.png", ICON)
    _take(SOURCES / "VotionSplat_Banner_Long.png", BANNER)
    _take(SOURCES / "votion-banner-seed-2.png", REPO_BANNER)
    _write_ico()
    _write_wizard()
    _write_small()
    _copy_mockup()
    for path in (ICON, BANNER, REPO_BANNER, ICO, WIZARD, SMALL):
        print(f"{path.name} {path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
