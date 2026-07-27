from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from photo_sorter.config import ConfigError, load_config
from photo_sorter.schemas.results import Decision, ImageResult, write_results_csv


def test_default_configuration_validates() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.uniform_classifier.selection_weights.uniform == 0.70


def test_invalid_hsv_configuration_has_useful_error(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["uniform_colors"][0]["hsv_lower"] = [200, 0, 0]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="OpenCV HSV"):
        load_config(path)


def test_csv_serialization_round_trip_shape(tmp_path: Path) -> None:
    result = ImageResult(
        filename="photo.jpg",
        full_path=str(tmp_path / "photo.jpg"),
        relative_path="photo.jpg",
        selected_person_bbox=(1, 2, 30, 40),
        decision=Decision.REVIEW,
        reason="ambiguous",
    )
    output = tmp_path / "results.csv"
    write_results_csv([result], output)
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["selected_person_bbox"]) == [1, 2, 30, 40]
    assert row["decision"] == "REVIEW"


def test_csv_refuses_overwrite(tmp_path: Path) -> None:
    result = ImageResult(filename="a", full_path="a", relative_path="a")
    output = tmp_path / "results.csv"
    write_results_csv([result], output)
    with pytest.raises(FileExistsError):
        write_results_csv([result], output)
