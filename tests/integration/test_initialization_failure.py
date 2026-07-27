from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from photo_sorter.processing import pipeline


def test_global_model_failure_still_writes_error_rows(
    tmp_path: Path, app_config, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (100, 100), "gray").save(input_dir / "one.jpg")
    Image.new("RGB", (100, 100), "gray").save(input_dir / "two.jpg")
    monkeypatch.setattr(
        pipeline,
        "build_dependencies",
        lambda config, logger: (_ for _ in ()).throw(RuntimeError("weights missing")),
    )
    output = tmp_path / "results.csv"
    summary = pipeline.analyze_directory(
        input_dir, output, app_config, logging.getLogger("failure")
    )
    assert len(summary.results) == 2
    assert all(result.decision.value == "ERROR" for result in summary.results)
    assert output.is_file()
