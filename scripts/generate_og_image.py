"""Generate the social preview (Open Graph) image.

The image is a build artifact, not hand-drawn, so the wording stays tied to
`marketing/POSITIONING.md` instead of drifting independently. Re-run after any
positioning change:

    python scripts/generate_og_image.py

GitHub's social preview is 1280x640 and crops to roughly 1200x630 in most
unfurls, so everything meaningful stays inside a centre-safe area.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "og-image.png"

WIDTH, HEIGHT = 1280, 640
MARGIN = 88

BACKGROUND = "#0B1620"
TEAL = "#0A7B83"
HEADLINE_FILL = "#F2F7F8"
BODY_FILL = "#9FB6BC"
CODE_FILL = "#7FD4DA"

HEADLINE = "Cut AI context cost and\nprove nothing was lost."
SUBHEAD = (
    "Every selection emits a receipt: what was kept, what was omitted,\n"
    "and the handle that recovers the exact original bytes."
)
COMMAND = "pip install -U entroly && entroly go"

# Windows, then common Linux locations, so this runs in CI as well as locally.
FONT_CANDIDATES = {
    "bold": (
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    "regular": (
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ),
    "mono": (
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ),
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES[kind]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise SystemExit(
        f"no {kind} font found; add a path to FONT_CANDIDATES[{kind!r}]"
    )


def main() -> int:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # Accent rule down the left edge, so the card reads as branded even when a
    # platform renders it small.
    draw.rectangle([(0, 0), (14, HEIGHT)], fill=TEAL)

    # Vertical rhythm is tuned so the block sits optically centred: unfurls crop
    # top and bottom before they crop the sides.
    y = MARGIN + 18

    draw.text((MARGIN, y), "ENTROLY", font=_font("bold", 30), fill=TEAL)
    y += 82

    draw.multiline_text(
        (MARGIN, y),
        HEADLINE,
        font=_font("bold", 62),
        fill=HEADLINE_FILL,
        spacing=16,
    )
    y += 202

    draw.multiline_text(
        (MARGIN, y),
        SUBHEAD,
        font=_font("regular", 27),
        fill=BODY_FILL,
        spacing=12,
    )
    y += 122

    # Command chip.
    mono = _font("mono", 26)
    box = draw.textbbox((0, 0), COMMAND, font=mono)
    pad_x, pad_y = 22, 16
    draw.rounded_rectangle(
        [
            (MARGIN, y),
            (MARGIN + (box[2] - box[0]) + pad_x * 2, y + (box[3] - box[1]) + pad_y * 2),
        ],
        radius=10,
        fill="#11242E",
        outline=TEAL,
        width=2,
    )
    draw.text((MARGIN + pad_x, y + pad_y - box[1]), COMMAND, font=mono, fill=CODE_FILL)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
