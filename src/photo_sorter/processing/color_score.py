from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from photo_sorter.schemas.config_models import UniformColorRange


@dataclass(frozen=True, slots=True)
class ColorScore:
    score: float
    qualifying_percentage: float
    breakdown: dict[str, float]


def calculate_color_score(
    image: Image.Image,
    color_ranges: list[UniformColorRange],
) -> ColorScore:
    if image.width < 2 or image.height < 2:
        return ColorScore(score=0.0, qualifying_percentage=0.0, breakdown={})
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    valid = (value > 10) & ~((value >= 253) & (saturation <= 5))
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return ColorScore(
            score=0.0,
            qualifying_percentage=0.0,
            breakdown={item.name: 0.0 for item in color_ranges},
        )

    total_weight = sum(item.weight for item in color_ranges)
    weighted = 0.0
    union = np.zeros(valid.shape, dtype=np.uint8)
    breakdown: dict[str, float] = {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    for item in color_ranges:
        lower = np.asarray(item.hsv_lower, dtype=np.uint8)
        upper = np.asarray(item.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        qualifies = (mask > 0) & valid
        ratio = float(np.count_nonzero(qualifies) / valid_count)
        breakdown[item.name] = ratio
        weighted += item.weight * ratio
        union[qualifies] = 1
    score = weighted / total_weight if total_weight else 0.0
    percentage = float(np.count_nonzero(union) / valid_count)
    return ColorScore(
        score=float(np.clip(score, 0.0, 1.0)),
        qualifying_percentage=percentage,
        breakdown=breakdown,
    )
