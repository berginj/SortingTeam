from __future__ import annotations

import pytest

from photo_sorter.processing.crop_utils import (
    centrality_score,
    clamp_bbox,
    crop_quality_score,
    pad_bbox,
    upper_body_bbox,
)


def test_clamp_bbox_to_image_boundaries() -> None:
    assert clamp_bbox((-10, -20, 120, 90), 100, 80) == (0, 0, 100, 80)


def test_clamp_bbox_rejects_empty_area() -> None:
    with pytest.raises(ValueError, match="no area"):
        clamp_bbox((10, 10, 10, 20), 100, 100)


def test_padding_and_upper_body_crop_are_clamped() -> None:
    padded = pad_bbox((0, 10, 50, 110), 0.10, 100, 100)
    assert padded == (0, 0, 55, 100)
    assert upper_body_bbox(padded, 0.625, 100, 100) == (0, 0, 55, 62)


def test_crop_quality_and_centrality_are_bounded() -> None:
    quality = crop_quality_score((25, 10, 75, 90), 100, 100)
    centrality = centrality_score((25, 10, 75, 90), 100, 100)
    assert 0 <= quality <= 1
    assert centrality == pytest.approx(1.0)
