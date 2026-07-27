from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from photo_sorter.metadata.xmp_writer import apply_xmp
from photo_sorter.review.storage import load_review_frame, save_review_override


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results_csv(path: Path, image: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "full_path", "relative_path", "decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": image.name,
                "full_path": str(image),
                "relative_path": image.name,
                "decision": "KEEP",
            }
        )


def test_xmp_dry_run_does_not_touch_image_or_sidecar(tmp_path: Path, app_config) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"preview bytes")
    results = tmp_path / "results.csv"
    _results_csv(results, image)
    before = _digest(image)
    changes, report = apply_xmp(results, app_config, write=False)
    assert _digest(image) == before
    assert changes[0].status == "CHANGE"
    assert not changes[0].xmp_path.exists()
    assert report.is_file()


def test_review_storage_never_overwrites_raw_results(tmp_path: Path) -> None:
    raw = tmp_path / "results.csv"
    pd.DataFrame([{"filename": "a.jpg", "full_path": "a.jpg", "decision": "REVIEW"}]).to_csv(
        raw, index=False
    )
    reviewed = tmp_path / "reviewed_results.csv"
    before = _digest(raw)
    frame = load_review_frame(raw, reviewed)
    save_review_override(
        frame,
        0,
        reviewed,
        decision="KEEP",
        target_person_index=1,
        uniform=True,
        focus=True,
    )
    assert _digest(raw) == before
    saved = pd.read_csv(reviewed)
    assert saved.loc[0, "effective_decision"] == "KEEP"
