from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeConfig(StrictModel):
    device: str = "auto"
    workers: int = Field(default=0, ge=0, le=64)
    chunk_size: int = Field(default=16, ge=1, le=512)
    offline: bool = True
    seed: int = 42
    cache_dir: Path = Path("../.cache/photo_sorter")
    target_references: Path = Path("../data/references/target")
    other_references: Path = Path("../data/references/other")
    allowed_extensions: list[str] = [".jpg", ".jpeg"]
    recursive: bool = True

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, value: list[str]) -> list[str]:
        normalized = [
            item.lower() if item.startswith(".") else f".{item.lower()}" for item in value
        ]
        if not normalized:
            raise ValueError("at least one input extension is required")
        return list(dict.fromkeys(normalized))


class PersonDetectionConfig(StrictModel):
    backend: Literal["ultralytics"] = "ultralytics"
    model_path: Path = Path("../.cache/photo_sorter/models/yolo11n.pt")
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    inference_size: int = Field(default=960, ge=320, le=4096)
    max_detections: int = Field(default=100, ge=1, le=1000)
    crop_padding: float = Field(default=0.08, ge=0, le=0.5)
    upper_body_ratio: float = Field(default=0.625, ge=0.4, le=0.9)
    min_crop_size: int = Field(default=64, ge=8, le=2048)


class ClipConfig(StrictModel):
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    checkpoint_path: Path = Path(
        "../.cache/photo_sorter/models/open_clip_ViT-B-32_laion2b_s34b_b79k.safetensors"
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    precision: Literal["auto", "fp32", "fp16"] = "auto"


class SelectionWeights(StrictModel):
    uniform: float = Field(default=0.70, ge=0, le=1)
    color: float = Field(default=0.20, ge=0, le=1)
    detection: float = Field(default=0.05, ge=0, le=1)
    crop_quality: float = Field(default=0.05, ge=0, le=1)
    centrality: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_total(self) -> SelectionWeights:
        total = self.uniform + self.color + self.detection + self.crop_quality + self.centrality
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"selection weights must sum to 1.0; got {total:.6f}")
        return self


class UniformClassifierConfig(StrictModel):
    mode: Literal["auto", "logistic", "centroid"] = "auto"
    min_examples_per_class: int = Field(default=5, ge=2)
    warning_examples_per_class: int = Field(default=15, ge=2)
    logistic_c: float = Field(default=1.0, gt=0)
    centroid_temperature: float = Field(default=0.07, gt=0)
    target_only_midpoint: float = Field(default=0.25, ge=-1, le=1)
    minimum_target_score: float = Field(default=0.35, ge=0, le=1)
    ambiguous_reference_margin: float = Field(default=0.15, ge=0, le=1)
    selection_weights: SelectionWeights = SelectionWeights()

    @model_validator(mode="after")
    def validate_counts(self) -> UniformClassifierConfig:
        if self.warning_examples_per_class < self.min_examples_per_class:
            raise ValueError("warning_examples_per_class must be >= min_examples_per_class")
        return self


class UniformColorRange(StrictModel):
    name: str = Field(min_length=1)
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    weight: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_hsv(self) -> UniformColorRange:
        for label, values in (("hsv_lower", self.hsv_lower), ("hsv_upper", self.hsv_upper)):
            if not (0 <= values[0] <= 179 and all(0 <= item <= 255 for item in values[1:])):
                raise ValueError(f"{label} must use OpenCV HSV ranges H=0..179, S/V=0..255")
        if any(lower > upper for lower, upper in zip(self.hsv_lower, self.hsv_upper, strict=True)):
            raise ValueError("hsv_lower values must not exceed hsv_upper; split wraparound hues")
        return self


class DenoiseConfig(StrictModel):
    enabled: bool = True
    iso_threshold: int = Field(default=3200, ge=0)
    diameter: int = Field(default=5, ge=1, le=15)
    sigma_color: float = Field(default=25.0, gt=0)
    sigma_space: float = Field(default=25.0, gt=0)


class FocusConfig(StrictModel):
    analysis_long_edge: int = Field(default=512, ge=128, le=2048)
    minimum_crop_dimension: int = Field(default=64, ge=8)
    laplacian_weight: float = Field(default=0.5, ge=0, le=1)
    tenengrad_weight: float = Field(default=0.5, ge=0, le=1)
    center_weight: float = Field(default=0.30, ge=0, le=1)
    noise_correction: bool = True
    noise_penalty: float = Field(default=0.35, ge=0, le=2)
    denoise: DenoiseConfig = DenoiseConfig()

    @model_validator(mode="after")
    def validate_metric_weights(self) -> FocusConfig:
        if not math.isclose(self.laplacian_weight + self.tenengrad_weight, 1.0, abs_tol=1e-6):
            raise ValueError("laplacian_weight and tenengrad_weight must sum to 1.0")
        return self


class BurstGroupingConfig(StrictModel):
    enabled: bool = True
    max_gap_seconds: float = Field(default=2.0, gt=0, le=60)
    uniform_weight: float = Field(default=0.65, ge=0, le=1)
    focus_weight: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self) -> BurstGroupingConfig:
        if not math.isclose(self.uniform_weight + self.focus_weight, 1.0, abs_tol=1e-6):
            raise ValueError("burst ranking weights must sum to 1.0")
        return self


class DecisionThresholdsConfig(StrictModel):
    keep_uniform_probability: float = Field(default=0.85, ge=0, le=1)
    review_uniform_probability: float = Field(default=0.45, ge=0, le=1)
    wrong_team_probability: float = Field(default=0.25, ge=0, le=1)
    keep_focus_percentile: float = Field(default=35.0, ge=0, le=100)
    soft_focus_percentile: float = Field(default=15.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_order(self) -> DecisionThresholdsConfig:
        if not (
            self.wrong_team_probability
            < self.review_uniform_probability
            <= self.keep_uniform_probability
        ):
            raise ValueError("uniform thresholds must satisfy wrong_team < review <= keep")
        if self.soft_focus_percentile >= self.keep_focus_percentile:
            raise ValueError("soft_focus_percentile must be below keep_focus_percentile")
        return self


class MetadataToggle(StrictModel):
    enabled: bool = False


class MetadataDecisionMapping(StrictModel):
    keyword: str | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    color_label: str | None = None


class MetadataMappingConfig(StrictModel):
    keywords: MetadataToggle = MetadataToggle(enabled=True)
    ratings: MetadataToggle = MetadataToggle(enabled=False)
    color_labels: MetadataToggle = MetadataToggle(enabled=False)
    decisions: dict[str, MetadataDecisionMapping]
    raw_extensions: list[str] = [
        ".arw",
        ".cr2",
        ".cr3",
        ".dng",
        ".nef",
        ".orf",
        ".raf",
        ".rw2",
    ]

    @field_validator("raw_extensions")
    @classmethod
    def normalize_raw_extensions(cls, value: list[str]) -> list[str]:
        return [item.lower() if item.startswith(".") else f".{item.lower()}" for item in value]


class OutputConfig(StrictModel):
    audit_dir: Path = Path("../data/output/audit")
    xmp_staging_dir: Path = Path("../data/output/xmp")
    xmp_backup_dir: Path = Path("../data/output/xmp_backups")
    csv_float_precision: int = Field(default=6, ge=1, le=12)
    write_audit_json: bool = True


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file_enabled: bool = True
    file: Path = Path("../data/output/photo_sorter.jsonl")


class AppConfig(StrictModel):
    runtime: RuntimeConfig = RuntimeConfig()
    person_detection: PersonDetectionConfig = PersonDetectionConfig()
    clip: ClipConfig = ClipConfig()
    uniform_classifier: UniformClassifierConfig = UniformClassifierConfig()
    uniform_colors: list[UniformColorRange]
    focus: FocusConfig = FocusConfig()
    burst_grouping: BurstGroupingConfig = BurstGroupingConfig()
    decision_thresholds: DecisionThresholdsConfig = DecisionThresholdsConfig()
    metadata_mapping: MetadataMappingConfig
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def validate_colors(self) -> AppConfig:
        if not self.uniform_colors:
            raise ValueError("at least one uniform color range is required")
        return self
