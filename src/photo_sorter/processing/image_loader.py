from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageLoadError(ValueError):
    """Raised for unreadable or unsupported images."""


@dataclass(slots=True)
class LoadedImage:
    image: Image.Image
    width: int
    height: int
    capture_time: datetime | None
    camera_id: str | None
    iso: int | None
    file_mtime: float


def _parse_datetime(value: object, subsecond: object, offset: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    if subsecond:
        digits = "".join(character for character in str(subsecond) if character.isdigit())[:6]
        if digits:
            parsed = parsed.replace(microsecond=int(digits.ljust(6, "0")))
    if offset:
        try:
            return datetime.fromisoformat(
                f"{parsed.strftime('%Y-%m-%dT%H:%M:%S.%f')}{str(offset).strip()}"
            )
        except ValueError:
            pass
    return parsed


def load_image(path: Path) -> LoadedImage:
    try:
        with Image.open(path) as source:
            source.load()
            exif = source.getexif()
            capture_time = _parse_datetime(
                exif.get(36867) or exif.get(306),
                exif.get(37521),
                exif.get(36881),
            )
            serial = exif.get(42033)
            model = exif.get(272)
            camera_id = str(serial or model).strip() if serial or model else None
            iso_value = exif.get(34855)
            try:
                iso = int(iso_value) if iso_value is not None else None
            except (TypeError, ValueError):
                iso = None
            oriented = ImageOps.exif_transpose(source).convert("RGB").copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageLoadError(f"Could not decode image {path}: {exc}") from exc
    stat = path.stat()
    return LoadedImage(
        image=oriented,
        width=oriented.width,
        height=oriented.height,
        capture_time=capture_time,
        camera_id=camera_id,
        iso=iso,
        file_mtime=stat.st_mtime,
    )
