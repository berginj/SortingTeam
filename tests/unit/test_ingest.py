from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from photo_sorter.processing.ingest import (
    IngestError,
    copy_to_archive,
    load_manifest,
    prepare_previews,
    stage_selected_originals,
)


def _image(path: Path, color: tuple[int, int, int] = (10, 40, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color).save(path)


def _results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "relative_path", "decision", "effective_decision"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_ingest_copy_verifies_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "card"
    _image(source / "DCIM" / "IMG_0001.jpg")
    archive = tmp_path / "archive"
    manifest_path = tmp_path / "manifest.json"
    first = copy_to_archive(source, archive, manifest_path, write=True)
    assert first.summary()["COPIED"] == 1
    assert (archive / "DCIM" / "IMG_0001.jpg").is_file()
    assert first.records[0].source_sha256 == first.records[0].destination_sha256
    resumed = copy_to_archive(source, archive, manifest_path, write=True)
    assert resumed.summary()["VERIFIED_EXISTING"] == 1
    assert load_manifest(manifest_path).records[0].status == "VERIFIED_EXISTING"


def test_ingest_refuses_destination_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "card"
    source.mkdir()
    with pytest.raises(IngestError, match="outside"):
        copy_to_archive(source, source / "archive", tmp_path / "manifest.json", write=True)


def test_prepare_previews_from_local_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    _image(source / "event" / "IMG_0002.jpg")
    previews = tmp_path / "previews"
    result = prepare_previews(
        source, previews, tmp_path / "previews.json", max_edge=256, write=True
    )
    assert result.summary()["CREATED"] == 1
    output = previews / "event" / "IMG_0002.jpg"
    with Image.open(output) as image:
        assert max(image.size) <= 256


def test_stage_selected_originals_uses_effective_decision_and_does_not_overwrite(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    original = archive / "day1" / "IMG_0003.ARW"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"raw bytes")
    results = tmp_path / "results.csv"
    _results(
        results,
        [
            {
                "filename": "IMG_0003.jpg",
                "relative_path": "day1/IMG_0003.jpg",
                "decision": "WRONG_TEAM",
                "effective_decision": "KEEP",
            },
            {
                "filename": "skip.jpg",
                "relative_path": "day1/skip.jpg",
                "decision": "NO_SUBJECT",
                "effective_decision": "NO_SUBJECT",
            },
        ],
    )
    destination = tmp_path / "to_import"
    staged = stage_selected_originals(
        results, archive, destination, tmp_path / "stage.json", decisions={"KEEP"}, write=True
    )
    assert staged.summary()["COPIED"] == 1
    assert (destination / "day1" / "IMG_0003.ARW").read_bytes() == b"raw bytes"
    assert original.read_bytes() == b"raw bytes"
