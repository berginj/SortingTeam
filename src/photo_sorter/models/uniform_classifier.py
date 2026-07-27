from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from photo_sorter.models.clip_embedder import ImageEmbedder
from photo_sorter.models.person_detector import PersonDetector
from photo_sorter.processing.crop_utils import (
    crop_image,
    crop_quality_score,
    pad_bbox,
    upper_body_bbox,
)
from photo_sorter.processing.image_loader import ImageLoadError, load_image
from photo_sorter.schemas.config_models import AppConfig
from photo_sorter.schemas.results import PersonDetection
from photo_sorter.utils.cache import atomic_write_json, sha256_file, stable_fingerprint
from photo_sorter.utils.paths import discover_images

FloatArray = npt.NDArray[np.float32]
LabelArray = npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class UniformEvidence:
    probability: float
    target_similarity: float
    other_similarity: float | None
    similarity_margin: float | None
    embedding_similarity: float


@dataclass(slots=True)
class UniformModel:
    mode: str
    target_centroid: FloatArray
    other_centroid: FloatArray | None
    classifier: Any | None
    temperature: float
    target_only_midpoint: float
    fingerprint: str
    metadata: dict[str, Any]

    def predict(self, embeddings: FloatArray) -> list[UniformEvidence]:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[1] != self.target_centroid.shape[0]:
            raise ValueError("embedding dimension does not match trained uniform model")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-12)
        target_similarity = normalized @ self.target_centroid
        other_similarity = (
            normalized @ self.other_centroid if self.other_centroid is not None else None
        )
        if self.mode == "logistic" and self.classifier is not None:
            probabilities = self.classifier.predict_proba(normalized)[:, 1]
        elif other_similarity is not None:
            delta = np.clip((target_similarity - other_similarity) / self.temperature, -60, 60)
            probabilities = 1.0 / (1.0 + np.exp(-delta))
        else:
            delta = np.clip(
                (target_similarity - self.target_only_midpoint) / self.temperature,
                -60,
                60,
            )
            probabilities = 1.0 / (1.0 + np.exp(-delta))
        evidence: list[UniformEvidence] = []
        for index, target in enumerate(target_similarity):
            other = float(other_similarity[index]) if other_similarity is not None else None
            evidence.append(
                UniformEvidence(
                    probability=float(np.clip(probabilities[index], 0, 1)),
                    target_similarity=float(target),
                    other_similarity=other,
                    similarity_margin=float(target - other) if other is not None else None,
                    embedding_similarity=float(np.clip((target + 1) / 2, 0, 1)),
                )
            )
        return evidence


def _normalize(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero embedding")
    return (vector / norm).astype(np.float32, copy=False)


def _reference_crop(
    image: Image.Image,
    detections: list[PersonDetection],
    config: AppConfig,
    path: Path,
    logger: logging.Logger,
) -> Image.Image | None:
    if not detections:
        logger.warning(
            "Reference has no person detection; embedding the entire image",
            extra={"event": "reference_whole_image", "path": str(path)},
        )
        return image
    scored: list[tuple[float, PersonDetection]] = []
    for detection in detections:
        quality = crop_quality_score(
            detection.bbox,
            image.width,
            image.height,
        )
        scored.append((0.6 * detection.confidence + 0.4 * quality, detection))
    scored.sort(key=lambda item: item[0], reverse=True)
    if (
        len(scored) > 1
        and scored[0][0] - scored[1][0] < config.uniform_classifier.ambiguous_reference_margin
    ):
        logger.warning(
            "Skipping ambiguous multi-person reference; crop it to one dominant player",
            extra={
                "event": "reference_ambiguous",
                "path": str(path),
                "people": len(scored),
            },
        )
        return None
    bbox = pad_bbox(
        scored[0][1].bbox,
        config.person_detection.crop_padding,
        image.width,
        image.height,
    )
    upper = upper_body_bbox(
        bbox,
        config.person_detection.upper_body_ratio,
        image.width,
        image.height,
    )
    return crop_image(image, upper)


def _reference_inventory(
    target_paths: list[Path], other_paths: list[Path], config: AppConfig
) -> tuple[list[dict[str, str]], str]:
    inventory = [
        {"path": str(path.resolve()), "label": "target", "sha256": sha256_file(path)}
        for path in target_paths
    ]
    inventory.extend(
        {"path": str(path.resolve()), "label": "other", "sha256": sha256_file(path)}
        for path in other_paths
    )
    fingerprint = stable_fingerprint(
        {
            "references": inventory,
            "clip_model": config.clip.model_name,
            "clip_pretrained": config.clip.pretrained,
            "clip_checkpoint": str(config.clip.checkpoint_path),
            "detector_model": str(config.person_detection.model_path),
            "detector_confidence": config.person_detection.confidence_threshold,
            "crop_padding": config.person_detection.crop_padding,
            "upper_body_ratio": config.person_detection.upper_body_ratio,
            "classifier": config.uniform_classifier.model_dump(mode="json"),
            "cache_schema": 1,
        }
    )
    return inventory, fingerprint


def _embed_reference_class(
    paths: list[Path],
    detector: PersonDetector,
    embedder: ImageEmbedder,
    config: AppConfig,
    logger: logging.Logger,
) -> tuple[FloatArray, list[str]]:
    crops: list[Image.Image] = []
    accepted: list[str] = []
    for path in paths:
        try:
            loaded = load_image(path)
            detections = detector.detect(loaded.image)
            crop = _reference_crop(loaded.image, detections, config, path, logger)
            if crop is not None:
                crops.append(crop)
                accepted.append(str(path.resolve()))
        except (ImageLoadError, ValueError, OSError) as exc:
            logger.warning(
                "Skipping invalid reference",
                extra={"event": "reference_error", "path": str(path), "error": str(exc)},
            )
    if not crops:
        return np.empty((0, embedder.dimension), dtype=np.float32), accepted
    return embedder.embed(crops), accepted


def _cross_validation_metrics(
    embeddings: FloatArray,
    labels: LabelArray,
    config: AppConfig,
) -> dict[str, float]:
    minimum_class_count = int(min(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)))
    if minimum_class_count < 2:
        return {}
    folds = min(5, minimum_class_count)
    estimator = LogisticRegression(
        class_weight="balanced",
        C=config.uniform_classifier.logistic_c,
        max_iter=1000,
        random_state=config.runtime.seed,
        solver="liblinear",
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.runtime.seed)
    probabilities = cross_val_predict(
        estimator,
        embeddings,
        labels,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    predicted = probabilities >= 0.5
    return {
        "cv_precision": float(precision_score(labels, predicted, zero_division=0)),
        "cv_recall": float(recall_score(labels, predicted, zero_division=0)),
        "cv_f1": float(f1_score(labels, predicted, zero_division=0)),
        "cv_folds": float(folds),
    }


def _load_cached_model(path: Path, fingerprint: str) -> UniformModel | None:
    if not path.is_file():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    target = np.asarray(payload["target_centroid"], dtype=np.float32)
    other_raw = payload.get("other_centroid")
    other = np.asarray(other_raw, dtype=np.float32) if other_raw is not None else None
    return UniformModel(
        mode=str(payload["mode"]),
        target_centroid=target,
        other_centroid=other,
        classifier=payload.get("classifier"),
        temperature=float(payload["temperature"]),
        target_only_midpoint=float(payload["target_only_midpoint"]),
        fingerprint=fingerprint,
        metadata=dict(payload.get("metadata", {})),
    )


def train_or_load_uniform_model(
    detector: PersonDetector,
    embedder: ImageEmbedder,
    config: AppConfig,
    logger: logging.Logger,
    *,
    force: bool = False,
) -> UniformModel:
    extensions = config.runtime.allowed_extensions
    target_paths = discover_images(config.runtime.target_references, extensions, recursive=True)
    other_paths = (
        discover_images(config.runtime.other_references, extensions, recursive=True)
        if config.runtime.other_references.exists()
        else []
    )
    if not target_paths:
        raise ValueError(f"No target reference images found in {config.runtime.target_references}")
    inventory, fingerprint = _reference_inventory(target_paths, other_paths, config)
    cache_root = config.runtime.cache_dir / "uniform"
    classifier_path = cache_root / "uniform_classifier.joblib"
    if not force:
        cached = _load_cached_model(classifier_path, fingerprint)
        if cached is not None:
            logger.info(
                "Reusing cached uniform classifier",
                extra={
                    "event": "uniform_cache_hit",
                    "mode": cached.mode,
                    "fingerprint": fingerprint,
                },
            )
            return cached

    target_embeddings, accepted_target = _embed_reference_class(
        target_paths, detector, embedder, config, logger
    )
    other_embeddings, accepted_other = _embed_reference_class(
        other_paths, detector, embedder, config, logger
    )
    if target_embeddings.shape[0] == 0:
        raise ValueError("No valid target references remained after validation")

    warning_count = config.uniform_classifier.warning_examples_per_class
    if target_embeddings.shape[0] < warning_count:
        logger.warning(
            "Target reference count is below the recommended minimum",
            extra={
                "event": "reference_count_warning",
                "class": "target",
                "count": int(target_embeddings.shape[0]),
                "recommended": warning_count,
            },
        )
    if other_embeddings.shape[0] < warning_count:
        logger.warning(
            "Other-team reference count is below the recommended minimum",
            extra={
                "event": "reference_count_warning",
                "class": "other",
                "count": int(other_embeddings.shape[0]),
                "recommended": warning_count,
            },
        )

    target_centroid = _normalize(np.mean(target_embeddings, axis=0))
    other_centroid = (
        _normalize(np.mean(other_embeddings, axis=0)) if other_embeddings.shape[0] > 0 else None
    )
    minimum = config.uniform_classifier.min_examples_per_class
    may_train = (
        target_embeddings.shape[0] >= minimum
        and other_embeddings.shape[0] >= minimum
        and config.uniform_classifier.mode != "centroid"
    )
    classifier: Any | None = None
    metrics: dict[str, float] = {}
    if may_train:
        combined = np.concatenate([target_embeddings, other_embeddings], axis=0)
        labels = np.concatenate(
            [
                np.ones(target_embeddings.shape[0], dtype=np.int8),
                np.zeros(other_embeddings.shape[0], dtype=np.int8),
            ]
        )
        classifier = LogisticRegression(
            class_weight="balanced",
            C=config.uniform_classifier.logistic_c,
            max_iter=1000,
            random_state=config.runtime.seed,
            solver="liblinear",
        )
        classifier.fit(combined, labels)
        metrics = _cross_validation_metrics(combined, labels, config)
        mode = "logistic"
    else:
        mode = "centroid"
        if config.uniform_classifier.mode == "logistic":
            logger.warning(
                "Logistic mode requested but too few examples; using centroid fallback",
                extra={"event": "classifier_fallback"},
            )
    metadata: dict[str, Any] = {
        "fingerprint": fingerprint,
        "mode": mode,
        "target_count": int(target_embeddings.shape[0]),
        "other_count": int(other_embeddings.shape[0]),
        "accepted_target": accepted_target,
        "accepted_other": accepted_other,
        "inventory": inventory,
        **metrics,
    }
    model = UniformModel(
        mode=mode,
        target_centroid=target_centroid,
        other_centroid=other_centroid,
        classifier=classifier,
        temperature=config.uniform_classifier.centroid_temperature,
        target_only_midpoint=config.uniform_classifier.target_only_midpoint,
        fingerprint=fingerprint,
        metadata=metadata,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_root / "reference_embeddings.npz",
        fingerprint=np.asarray([fingerprint]),
        target_embeddings=target_embeddings,
        other_embeddings=other_embeddings,
    )
    joblib.dump(
        {
            "fingerprint": fingerprint,
            "mode": mode,
            "target_centroid": target_centroid,
            "other_centroid": other_centroid,
            "classifier": classifier,
            "temperature": model.temperature,
            "target_only_midpoint": model.target_only_midpoint,
            "metadata": metadata,
        },
        classifier_path,
    )
    atomic_write_json(cache_root / "uniform_classifier.json", metadata)
    logger.info(
        "Uniform classifier trained",
        extra={
            "event": "uniform_trained",
            "mode": mode,
            "target_count": int(target_embeddings.shape[0]),
            "other_count": int(other_embeddings.shape[0]),
        },
    )
    return model


def target_person_score(
    *,
    uniform_probability: float,
    color_score: float,
    detection_confidence: float,
    crop_quality: float,
    centrality: float,
    config: AppConfig,
) -> float:
    weights = config.uniform_classifier.selection_weights
    score = (
        weights.uniform * uniform_probability
        + weights.color * color_score
        + weights.detection * detection_confidence
        + weights.crop_quality * crop_quality
        + weights.centrality * centrality
    )
    return float(np.clip(score, 0, 1))
