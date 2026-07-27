from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    WRONG_TEAM = "WRONG_TEAM"
    SOFT = "SOFT"
    NO_SUBJECT = "NO_SUBJECT"
    ERROR = "ERROR"


class PersonDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)


class PersonEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    upper_body_bbox: tuple[int, int, int, int]
    detection_confidence: float
    uniform_probability: float
    target_similarity: float
    other_similarity: float | None = None
    similarity_margin: float | None = None
    embedding_similarity: float
    uniform_color_score: float
    color_qualifying_percentage: float
    color_breakdown: dict[str, float]
    crop_quality_score: float
    centrality_score: float
    target_person_score: float


class FocusMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    laplacian: float
    tenengrad: float
    full_laplacian: float
    full_tenengrad: float
    center_laplacian: float
    center_tenengrad: float
    estimated_noise: float
    denoised: bool


class ImageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    full_path: str
    relative_path: str
    image_width: int | None = None
    image_height: int | None = None
    capture_time: datetime | None = None
    camera_id: str | None = None
    file_mtime: float | None = None
    people_detected: int = 0
    selected_person_index: int | None = None
    selected_person_bbox: tuple[int, int, int, int] | None = None
    person_detection_confidence: float | None = None
    uniform_probability: float | None = None
    uniform_embedding_similarity: float | None = None
    uniform_color_score: float | None = None
    laplacian_focus_score: float | None = None
    tenengrad_focus_score: float | None = None
    normalized_focus_score: float | None = None
    focus_percentile: float | None = None
    decision: Decision = Decision.ERROR
    reason: str = ""
    processing_time_ms: int = 0
    error: str = ""
    burst_id: str = ""
    burst_size: int = 1
    burst_rank: int = 1
    is_best_in_burst: bool = True
    analysis_sidecar_path: str = ""


CSV_COLUMNS = [
    "filename",
    "full_path",
    "relative_path",
    "image_width",
    "image_height",
    "capture_time",
    "camera_id",
    "people_detected",
    "selected_person_index",
    "selected_person_bbox",
    "person_detection_confidence",
    "uniform_probability",
    "uniform_embedding_similarity",
    "uniform_color_score",
    "laplacian_focus_score",
    "tenengrad_focus_score",
    "normalized_focus_score",
    "focus_percentile",
    "decision",
    "reason",
    "processing_time_ms",
    "error",
    "burst_id",
    "burst_size",
    "burst_rank",
    "is_best_in_burst",
    "analysis_sidecar_path",
]


def _format_value(value: Any, precision: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def result_to_csv_row(result: ImageResult, precision: int = 6) -> dict[str, Any]:
    raw = result.model_dump()
    raw.pop("file_mtime", None)
    return {column: _format_value(raw.get(column), precision) for column in CSV_COLUMNS}


def write_results_csv(
    results: list[ImageResult],
    output: Path,
    *,
    precision: int = 6,
    overwrite: bool = False,
) -> None:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for result in results:
                writer.writerow(result_to_csv_row(result, precision))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
