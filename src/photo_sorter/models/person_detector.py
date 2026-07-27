from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from photo_sorter.schemas.config_models import PersonDetectionConfig
from photo_sorter.schemas.results import PersonDetection


class PersonDetector(Protocol):
    def detect(self, image: Image.Image) -> list[PersonDetection]: ...

    def detect_batch(self, images: Sequence[Image.Image]) -> list[list[PersonDetection]]: ...


class ModelNotAvailableError(RuntimeError):
    """Raised when a required local model checkpoint is unavailable."""


class UltralyticsPersonDetector:
    """YOLO person detector with no network fallback."""

    def __init__(self, config: PersonDetectionConfig, device: str) -> None:
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise ModelNotAvailableError(
                f"YOLO model not found at {model_path}. "
                "Run: python scripts/download_models.py --config config/default.yaml"
            )
        try:
            config_dir = model_path.parent / "ultralytics_config"
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO
            settings = ultralytics.settings
            events = importlib.import_module("ultralytics.utils.events").events
        except ImportError as exc:
            raise ModelNotAvailableError(
                "Ultralytics is not installed. Run the platform setup script."
            ) from exc
        settings.update(
            {
                "sync": False,
                "hub": False,
                "clearml": False,
                "comet": False,
                "dvc": False,
                "mlflow": False,
                "neptune": False,
                "raytune": False,
                "wandb": False,
                "weights_dir": str(model_path.parent),
                "runs_dir": str(config_dir / "runs"),
                "datasets_dir": str(config_dir / "datasets"),
            }
        )
        events.enabled = False
        self._config = config
        self._device = device
        self._model = YOLO(str(model_path), task="detect", verbose=False)

    def detect(self, image: Image.Image) -> list[PersonDetection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[Image.Image]) -> list[list[PersonDetection]]:
        if not images:
            return []
        sources = [np.asarray(image.convert("RGB")) for image in images]
        device = 0 if self._device.startswith("cuda") else self._device
        results = self._model.predict(
            source=sources,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.inference_size,
            classes=[0],
            max_det=self._config.max_detections,
            device=device,
            verbose=False,
            stream=False,
        )
        all_detections: list[list[PersonDetection]] = []
        for result in results:
            detections: list[tuple[tuple[int, int, int, int], float]] = []
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()
                for box, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
                    if int(class_id) != 0 or float(confidence) < self._config.confidence_threshold:
                        continue
                    values = [round(float(value)) for value in box]
                    rounded = (values[0], values[1], values[2], values[3])
                    detections.append((rounded, float(confidence)))
            detections.sort(key=lambda item: (item[0][0], item[0][1], -item[1]))
            all_detections.append(
                [
                    PersonDetection(index=index, bbox=bbox, confidence=confidence)
                    for index, (bbox, confidence) in enumerate(detections)
                ]
            )
        return all_detections
