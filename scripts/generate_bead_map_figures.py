#!/usr/bin/env python3
"""Compose aligned views of the committed photograph and existing color masks.

No bead detection, retouching, morphology, or generated image content is used.
Requires Pillow >= 10.1. Run from any working directory.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"yellow": "#e6ac12", "red": "#cf3445", "black": "#29323b"}
BACKGROUND = "#fafaf7"
# Source-pixel coordinates, shared by both panels; no geometric transformation.
CROPS = {
    "bend": (1030, 1330, 1900, 2200),
    "detail": (1250, 1320, 1850, 1920),
    "sweep": (500, 1200, 1900, 2600),
}


def load_views():
    photo = Image.open(ROOT / "docs/figures/source/beads-photo-2.jpg").convert("RGB")
    combined = Image.new("RGB", photo.size, BACKGROUND)
    # Existing masks use dark foreground on white; 128 suppresses JPEG ringing.
    # Fixed paint order means black wins any overlap, then red, then yellow.
    for name, color in COLORS.items():
        with Image.open(ROOT / f"beads-photo-2-{name}.jpg") as image:
            if image.size != photo.size:
                raise ValueError(f"{name} mask and photograph dimensions differ")
            foreground = image.convert("L").point(lambda value: 255 if value < 128 else 0)
            combined.paste(color, mask=foreground)
    return photo, combined


def compose(photo, combined, crop):
    card = Image.new("RGB", (1200, 660), BACKGROUND)
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=30)
    for x, label, view in ((36, "Photograph", photo), (610, "Color masks", combined)):
        draw.text((x, 26), label, font=font, fill="#29323b")
        panel = view.crop(crop).resize((554, 554), Image.Resampling.LANCZOS)
        card.paste(panel, (x, 82))
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    photo, combined = load_views()
    previews = []
    for name, crop in CROPS.items():
        card = compose(photo, combined, crop)
        preview = card.resize((400, 220), Image.Resampling.LANCZOS)
        previews.append((name, card, preview))
        if name == "bend":
            card.save(args.output_dir / "bead-map-card.png")
            preview.save(args.output_dir / "bead-map-card-400.png")
    # Review sheet shows each composition at full output size and card size.
    sheet = Image.new("RGB", (1640, 3 * 710), "#eceeea")
    draw = ImageDraw.Draw(sheet)
    for index, (name, card, preview) in enumerate(previews):
        y = index * 710
        draw.text((20, y + 10), name, font=ImageFont.load_default(size=22), fill="#29323b")
        sheet.paste(card, (0, y + 42))
        sheet.paste(preview, (1220, y + 42))
    sheet.save(args.output_dir / "bead-map-candidates.png")
    print(f"Wrote card, 400-pixel preview, and candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
