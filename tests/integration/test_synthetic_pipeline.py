from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from photo_sorter.models.uniform_classifier import UniformModel
from photo_sorter.processing.pipeline import PipelineDependencies, analyze_directory
from photo_sorter.schemas.results import PersonDetection


class FakeDetector:
    def detect(self, image: Image.Image) -> list[PersonDetection]:
        marker = image.getpixel((0, 0))
        if marker[0] < 10 and marker[1] < 10 and marker[2] < 10:
            return []
        return [
            PersonDetection(index=0, bbox=(10, 10, 100, 180), confidence=0.90),
            PersonDetection(index=1, bbox=(110, 10, 195, 180), confidence=0.85),
        ]

    def detect_batch(self, images):
        return [self.detect(image) for image in images]


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed(self, images):
        features = []
        for image in images:
            mean = np.asarray(image).mean(axis=(0, 1))
            if mean[0] >= mean[2]:
                features.append([1.0, 0.0])
            else:
                features.append([0.0, 1.0])
        return np.asarray(features, dtype=np.float32)


def _dependencies() -> PipelineDependencies:
    model = UniformModel(
        mode="centroid",
        target_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        other_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
        classifier=None,
        temperature=0.07,
        target_only_midpoint=0.25,
        fingerprint="synthetic",
        metadata={},
    )
    return PipelineDependencies(
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        uniform_model=model,
        device="cpu",
    )


def _photo(path: Path, *, no_subject: bool = False, blurred: bool = False) -> None:
    image = Image.new("RGB", (220, 200), "black" if no_subject else "gray")
    if not no_subject:
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 100, 180), fill=(180, 20, 20))
        draw.rectangle((110, 10, 195, 180), fill=(20, 20, 180))
        for y in range(10, 180, 8 if not blurred else 40):
            draw.line((10, y, 100, y), fill="white", width=2)
    image.save(path, quality=90)


def test_synthetic_pipeline_handles_multiple_people_no_subject_and_corruption(
    tmp_path: Path, app_config
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _photo(input_dir / "IMG_0001.jpg")
    _photo(input_dir / "IMG_0002.jpg", no_subject=True)
    (input_dir / "IMG_0003.jpg").write_bytes(b"not a jpeg")
    output = tmp_path / "output" / "results.csv"
    app_config.output.audit_dir = tmp_path / "output" / "audit"
    app_config.runtime.chunk_size = 2
    summary = analyze_directory(
        input_dir,
        output,
        app_config,
        logging.getLogger("test"),
        dependencies=_dependencies(),
    )
    assert len(summary.results) == 3
    first = next(result for result in summary.results if result.filename == "IMG_0001.jpg")
    assert first.people_detected == 2
    assert first.selected_person_index == 0
    assert first.laplacian_focus_score is not None
    no_subject = next(result for result in summary.results if result.filename == "IMG_0002.jpg")
    assert no_subject.decision.value == "NO_SUBJECT"
    corrupted = next(result for result in summary.results if result.filename == "IMG_0003.jpg")
    assert corrupted.decision.value == "ERROR"
    assert output.is_file()
    assert all(Path(result.analysis_sidecar_path).is_file() for result in summary.results)
