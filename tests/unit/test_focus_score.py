from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from photo_sorter.processing.focus_score import (
    FocusEvaluationError,
    calculate_focus_metrics,
    normalize_focus_results,
    percentile_ranks,
    robust_normalize,
)
from photo_sorter.schemas.results import ImageResult


def _checker(size: int = 256) -> Image.Image:
    yy, xx = np.indices((size, size))
    values = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(values[:, :, None], 3, axis=2), "RGB")


def _result(name: str, lap: float, ten: float) -> ImageResult:
    return ImageResult(
        filename=name,
        full_path=name,
        relative_path=name,
        laplacian_focus_score=lap,
        tenengrad_focus_score=ten,
    )


def test_focus_scores_sharp_image_above_blurred(app_config) -> None:
    sharp = _checker()
    blurred = Image.fromarray(cv2.GaussianBlur(np.asarray(sharp), (31, 31), 8), "RGB")
    sharp_metrics = calculate_focus_metrics(sharp, app_config.focus)
    blurred_metrics = calculate_focus_metrics(blurred, app_config.focus)
    assert sharp_metrics.laplacian > blurred_metrics.laplacian
    assert sharp_metrics.tenengrad > blurred_metrics.tenengrad


def test_tiny_crop_is_rejected(app_config) -> None:
    with pytest.raises(FocusEvaluationError, match="too small"):
        calculate_focus_metrics(Image.new("RGB", (20, 100)), app_config.focus)


def test_robust_normalization_handles_ties_and_outlier() -> None:
    tied = robust_normalize([5, 5, 5])
    assert np.allclose(tied, 0.5)
    ranked = percentile_ranks([10, 10, 20, 1000])
    assert ranked[0] == ranked[1]
    assert ranked[-1] == 100


def test_focus_result_normalization_sets_score_and_percentile(app_config) -> None:
    results = [_result("a", 1, 10), _result("b", 5, 50), _result("c", 20, 200)]
    normalize_focus_results(results, app_config.focus)
    assert results[0].focus_percentile == 0
    assert results[2].focus_percentile == 100
    assert all(result.normalized_focus_score is not None for result in results)


def test_singleton_percentile_is_100(app_config) -> None:
    result = _result("only", 3, 7)
    normalize_focus_results([result], app_config.focus)
    assert result.focus_percentile == 100
