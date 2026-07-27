from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from photo_sorter.models.uniform_classifier import (
    UniformModel,
    train_or_load_uniform_model,
)


class NoDetection:
    def detect(self, image):
        return []

    def detect_batch(self, images):
        return [[] for _ in images]


class ColorEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed(self, images):
        rows = []
        for image in images:
            mean = np.asarray(image).mean(axis=(0, 1))
            rows.append([1.0, 0.0] if mean[0] > mean[2] else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_centroid_predictions_cover_dual_and_target_only() -> None:
    embedding = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dual = UniformModel(
        mode="centroid",
        target_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        other_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
        classifier=None,
        temperature=0.07,
        target_only_midpoint=0.25,
        fingerprint="dual",
        metadata={},
    )
    evidence = dual.predict(embedding)
    assert evidence[0].probability > 0.99
    assert evidence[1].probability < 0.01
    assert evidence[0].similarity_margin == 1.0

    target_only = UniformModel(
        mode="centroid",
        target_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        other_centroid=None,
        classifier=None,
        temperature=0.07,
        target_only_midpoint=0.25,
        fingerprint="target-only",
        metadata={},
    )
    assert target_only.predict(embedding)[0].probability > 0.99


def test_reference_training_and_cache_reuse(tmp_path: Path, app_config) -> None:
    for index in range(5):
        Image.new("RGB", (80, 100), (200, index, 0)).save(
            app_config.runtime.target_references / f"target_{index}.jpg"
        )
        Image.new("RGB", (80, 100), (0, index, 200)).save(
            app_config.runtime.other_references / f"other_{index}.jpg"
        )
    logger = logging.getLogger("uniform-test")
    trained = train_or_load_uniform_model(NoDetection(), ColorEmbedder(), app_config, logger)
    assert trained.mode == "logistic"
    assert trained.metadata["target_count"] == 5
    cached = train_or_load_uniform_model(NoDetection(), ColorEmbedder(), app_config, logger)
    assert cached.fingerprint == trained.fingerprint
    assert (app_config.runtime.cache_dir / "uniform" / "uniform_classifier.json").is_file()
    assert (app_config.runtime.cache_dir / "uniform" / "reference_embeddings.npz").is_file()
