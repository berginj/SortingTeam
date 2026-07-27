from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from photo_sorter.processing.color_score import calculate_color_score
from photo_sorter.schemas.config_models import UniformColorRange


def _hsv_image(h: int, s: int, v: int) -> Image.Image:
    hsv = np.full((100, 100, 3), (h, s, v), dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb, "RGB")


def test_matching_color_scores_high() -> None:
    ranges = [
        UniformColorRange(
            name="navy",
            hsv_lower=(100, 70, 35),
            hsv_upper=(135, 255, 180),
            weight=1.0,
        )
    ]
    score = calculate_color_score(_hsv_image(115, 200, 120), ranges)
    assert score.score > 0.95
    assert score.qualifying_percentage > 0.95


def test_nonmatching_color_scores_zero() -> None:
    ranges = [
        UniformColorRange(
            name="navy",
            hsv_lower=(100, 70, 35),
            hsv_upper=(135, 255, 180),
            weight=1.0,
        )
    ]
    score = calculate_color_score(_hsv_image(20, 200, 120), ranges)
    assert score.score == 0


def test_clipped_white_region_is_ignored() -> None:
    ranges = [
        UniformColorRange(
            name="white",
            hsv_lower=(0, 0, 180),
            hsv_upper=(179, 70, 255),
            weight=1.0,
        )
    ]
    score = calculate_color_score(Image.new("RGB", (20, 20), "white"), ranges)
    assert score.score == 0
