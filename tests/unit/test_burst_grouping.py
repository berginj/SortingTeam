from __future__ import annotations

from datetime import datetime, timedelta

from photo_sorter.processing.burst_grouping import assign_bursts
from photo_sorter.schemas.results import ImageResult


def _result(name: str, second: float, uniform: float, focus: float) -> ImageResult:
    base = datetime(2026, 1, 1, 12, 0, 0)
    return ImageResult(
        filename=name,
        full_path=name,
        relative_path=name,
        capture_time=base + timedelta(seconds=second),
        camera_id="camera-a",
        uniform_probability=uniform,
        normalized_focus_score=focus,
        person_detection_confidence=0.9,
    )


def test_bursts_split_on_time_gap_and_rank_best(app_config) -> None:
    results = [
        _result("IMG_0001.jpg", 0, 0.8, 0.5),
        _result("IMG_0002.jpg", 1, 0.9, 0.9),
        _result("IMG_0003.jpg", 5, 0.95, 0.95),
    ]
    assign_bursts(results, app_config.burst_grouping)
    assert results[0].burst_id == results[1].burst_id
    assert results[2].burst_id != results[1].burst_id
    assert results[1].burst_rank == 1
    assert results[1].is_best_in_burst


def test_filename_and_mtime_fallback(app_config) -> None:
    results = []
    for index in range(1, 4):
        result = ImageResult(
            filename=f"DSC_{index:04d}.jpg",
            full_path=f"DSC_{index:04d}.jpg",
            relative_path=f"DSC_{index:04d}.jpg",
            file_mtime=1000.0 + index,
        )
        results.append(result)
    assign_bursts(results, app_config.burst_grouping)
    assert len({item.burst_id for item in results}) == 1
    assert results[0].burst_size == 3
