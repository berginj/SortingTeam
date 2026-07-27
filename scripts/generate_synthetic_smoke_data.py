from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _reference(path: Path, jersey: tuple[int, int, int], accent: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (320, 420), (35, 50, 35))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 25, 200, 105), fill=(180, 135, 105))
    draw.rounded_rectangle((75, 95, 245, 330), radius=25, fill=jersey)
    draw.rectangle((75, 150, 245, 185), fill=accent)
    draw.rectangle((95, 330, 135, 415), fill=(40, 40, 40))
    draw.rectangle((185, 330, 225, 415), fill=(40, 40, 40))
    image.save(path, quality=90)


def _game_frame(path: Path, index: int, *, blur: bool = False) -> None:
    image = Image.new("RGB", (960, 640), (45, 100, 45))
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 80):
        draw.line((x, 0, x, image.height), fill=(65, 125, 65), width=2)
    for x, jersey, accent in (
        (230, (25, 35, 95), (230, 230, 230)),
        (560, (20, 100, 155), (240, 200, 30)),
    ):
        draw.ellipse((x, 120, x + 75, 195), fill=(180, 135, 105))
        draw.rounded_rectangle((x - 35, 190, x + 110, 440), radius=20, fill=jersey)
        draw.rectangle((x - 35, 250, x + 110, 285), fill=accent)
        for y in range(300, 430, 15):
            draw.line((x - 25, y, x + 100, y), fill=(235, 235, 235), width=2)
    draw.text((20, 20), f"Synthetic offline smoke frame {index}", fill="white")
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=3))
    image.save(path, quality=85)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "references" / "target"
    other = root / "references" / "other"
    input_dir = root / "input"
    for directory in (target, other, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        _reference(
            target / f"synthetic_target_{index}.jpg",
            (25 + index * 2, 35, 95),
            (235, 235, 235),
        )
        _reference(
            other / f"synthetic_other_{index}.jpg",
            (15, 105 + index * 2, 160),
            (240, 195, 25),
        )
    for index in range(3):
        _game_frame(
            input_dir / f"synthetic_{index:04d}.jpg",
            index,
            blur=index == 2,
        )
    print(f"Synthetic target references: {target}")
    print(f"Synthetic other references: {other}")
    print(f"Synthetic input previews: {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
