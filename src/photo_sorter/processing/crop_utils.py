from __future__ import annotations

from PIL import Image

BBox = tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bounding box has no area after clamping: {bbox}")
    return x1, y1, x2, y2


def pad_bbox(
    bbox: BBox,
    padding: float,
    image_width: int,
    image_height: int,
) -> BBox:
    x1, y1, x2, y2 = bbox
    pad_x = round((x2 - x1) * padding)
    pad_y = round((y2 - y1) * padding)
    return clamp_bbox(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        image_width,
        image_height,
    )


def upper_body_bbox(
    bbox: BBox,
    ratio: float,
    image_width: int,
    image_height: int,
) -> BBox:
    if not 0 < ratio <= 1:
        raise ValueError("upper-body ratio must be in (0, 1]")
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width, image_height)
    upper_y2 = y1 + max(1, round((y2 - y1) * ratio))
    return clamp_bbox((x1, y1, x2, upper_y2), image_width, image_height)


def crop_image(image: Image.Image, bbox: BBox) -> Image.Image:
    valid = clamp_bbox(bbox, image.width, image.height)
    return image.crop(valid)


def crop_quality_score(
    bbox: BBox,
    image_width: int,
    image_height: int,
    *,
    reference_area_fraction: float = 0.10,
    reference_min_dimension: int = 160,
) -> float:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width, image_height)
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(image_width * image_height)
    area_score = min(1.0, (area_ratio / max(reference_area_fraction, 1e-9)) ** 0.5)
    dimension_score = min(1.0, min(x2 - x1, y2 - y1) / reference_min_dimension)
    return float(area_score * dimension_score)


def centrality_score(bbox: BBox, image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width, image_height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    normalized_x = abs(center_x - image_width / 2) / (image_width / 2)
    normalized_y = abs(center_y - image_height / 2) / (image_height / 2)
    distance = ((normalized_x**2 + normalized_y**2) / 2) ** 0.5
    return float(max(0.0, 1.0 - distance))
