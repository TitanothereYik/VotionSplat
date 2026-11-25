"""Build the window icon and the brand banner.

Overwrite app/assets/icon.png (1024x1024) and app/assets/banner.png (1600x400)
with the finished art, then run this script again. It always rebuilds icon.ico.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"
ICON = ASSETS / "icon.png"
BANNER = ASSETS / "banner.png"
ICO = ASSETS / "icon.ico"
FONT = Path(r"C:\Windows\Fonts\arial.ttf")
BG = "#122031"
EDGE = "#ff9f30"
INK = "#f0f0f0"


def _font(size: int) -> ImageFont.ImageFont:
    if FONT.is_file():
        return ImageFont.truetype(str(FONT), size=size)
    return ImageFont.load_default()


def _placeholder(path: Path, size: tuple[int, int], caption: str) -> None:
    if path.is_file():
        return
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, size[0] - 4, size[1] - 4), outline=EDGE, width=6)
    title_font = _font(max(28, size[1] // 8))
    caption_font = _font(max(18, size[1] // 14))
    title = "VotionSplat"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    caption_box = draw.textbbox((0, 0), caption, font=caption_font)
    title_w = title_box[2] - title_box[0]
    title_h = title_box[3] - title_box[1]
    caption_w = caption_box[2] - caption_box[0]
    caption_h = caption_box[3] - caption_box[1]
    gap = size[1] // 18
    block = title_h + gap + caption_h
    top = (size[1] - block) // 2
    draw.text(((size[0] - title_w) // 2, top), title, font=title_font, fill=INK)
    draw.text(
        ((size[0] - caption_w) // 2, top + title_h + gap),
        caption,
        font=caption_font,
        fill=EDGE,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_ico() -> None:
    image = Image.open(ICON).convert("RGBA")
    image.save(ICO, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def _copy_mockup() -> None:
    media = ROOT / "site_mockup" / "media"
    if not media.is_dir():
        return
    for source in (ICON, BANNER):
        target = media / source.name
        target.write_bytes(source.read_bytes())


def main() -> int:
    _placeholder(ICON, (1024, 1024), "ICON PLACEHOLDER")
    _placeholder(BANNER, (1600, 400), "BANNER PLACEHOLDER")
    _write_ico()
    _copy_mockup()
    print(f"icon {ICON}")
    print(f"banner {BANNER}")
    print(f"ico {ICO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
