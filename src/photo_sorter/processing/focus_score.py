from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from photo_sorter.schemas.config_models import FocusConfig
from photo_sorter.schemas.results import FocusMetrics, ImageResult


class FocusEvaluationError(ValueError):
    """Raised when a crop is too small for meaningful focus evaluation."""


def _resize_for_focus(image: npt.NDArray[np.uint8], long_edge: int) -> npt.NDArray[np.uint8]:
    height, width = image.shape[:2]
    scale = long_edge / max(width, height)
    if scale >= 1:
        return image
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(image, target, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _metrics(gray: npt.NDArray[np.uint8]) -> tuple[float, float]:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_score = float(laplacian.var())
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.asarray(sobel_x * sobel_x + sobel_y * sobel_y, dtype=np.float64)
    tenengrad = float(magnitude.mean())
    return laplacian_score, tenengrad


def _estimate_noise(gray: npt.NDArray[np.uint8]) -> float:
    residual = np.asarray(
        gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0),
        dtype=np.float32,
    )
    median = float(np.median(residual))
    absolute = np.asarray(np.abs(residual - median), dtype=np.float32)
    mad = float(np.median(absolute))
    return 1.4826 * mad


def calculate_focus_metrics(
    image: Image.Image,
    config: FocusConfig,
    *,
    iso: int | None = None,
) -> FocusMetrics:
    if min(image.width, image.height) < config.minimum_crop_dimension:
        raise FocusEvaluationError(
            f"crop is too small for focus evaluation: {image.width}x{image.height}; "
            f"minimum dimension is {config.minimum_crop_dimension}"
        )
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = _resize_for_focus(rgb, config.analysis_long_edge)
    gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    denoised = bool(
        config.denoise.enabled and iso is not None and iso >= config.denoise.iso_threshold
    )
    if denoised:
        gray = np.asarray(
            cv2.bilateralFilter(
                gray,
                config.denoise.diameter,
                config.denoise.sigma_color,
                config.denoise.sigma_space,
            ),
            dtype=np.uint8,
        )
    noise = _estimate_noise(gray)
    full_lap, full_ten = _metrics(gray)

    height, width = gray.shape
    x1, x2 = round(width * 0.2), round(width * 0.8)
    y1, y2 = round(height * 0.1), round(height * 0.7)
    center = gray[y1:y2, x1:x2]
    if min(center.shape[:2]) >= 8:
        center_lap, center_ten = _metrics(center)
    else:
        center_lap, center_ten = full_lap, full_ten

    center_weight = config.center_weight
    laplacian = (1 - center_weight) * full_lap + center_weight * center_lap
    tenengrad = (1 - center_weight) * full_ten + center_weight * center_ten
    if config.noise_correction:
        laplacian = max(0.0, laplacian - config.noise_penalty * noise**2)
        tenengrad = max(0.0, tenengrad - config.noise_penalty * noise**2 * 16)
    return FocusMetrics(
        laplacian=laplacian,
        tenengrad=tenengrad,
        full_laplacian=full_lap,
        full_tenengrad=full_ten,
        center_laplacian=center_lap,
        center_tenengrad=center_ten,
        estimated_noise=noise,
        denoised=denoised,
    )


def robust_normalize(values: Sequence[float]) -> npt.NDArray[np.float64]:
    array = np.log1p(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)
    z = np.clip((array - median) / scale, -12, 12)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


def percentile_ranks(
    values: Sequence[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    count = array.size
    if count == 0:
        return np.asarray([], dtype=np.float64)
    if count == 1:
        return np.asarray([100.0])
    order = np.argsort(array, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start
        while end + 1 < count and array[order[end + 1]] == array[order[start]]:
            end += 1
        average_rank = (start + end) / 2
        ranks[order[start : end + 1]] = 100.0 * average_rank / (count - 1)
        start = end + 1
    return ranks


def normalize_focus_results(results: list[ImageResult], config: FocusConfig) -> None:
    eligible = [
        result
        for result in results
        if result.laplacian_focus_score is not None
        and result.tenengrad_focus_score is not None
        and not result.error
    ]
    if not eligible:
        return
    lap = robust_normalize([result.laplacian_focus_score or 0.0 for result in eligible])
    ten = robust_normalize([result.tenengrad_focus_score or 0.0 for result in eligible])
    combined = config.laplacian_weight * lap + config.tenengrad_weight * ten
    percentiles = percentile_ranks(combined)
    for result, normalized, percentile in zip(eligible, combined, percentiles, strict=True):
        result.normalized_focus_score = float(np.clip(normalized, 0, 1))
        result.focus_percentile = float(np.clip(percentile, 0, 100))
