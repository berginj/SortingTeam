from __future__ import annotations

import csv
from pathlib import Path

import pytest

from photo_sorter.metadata import xmp_writer
from photo_sorter.metadata.xmp_writer import XmpError, apply_xmp, plan_xmp_changes

XMP = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="2" xmp:Label="Blue">
      <dc:subject><rdf:Bag><rdf:li>Family</rdf:li><rdf:li>ML_REVIEW</rdf:li></rdf:Bag></dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def _write_results(path: Path, preview: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "full_path", "relative_path", "decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": preview.name,
                "full_path": str(preview),
                "relative_path": f"day1/{preview.name}",
                "decision": "KEEP",
            }
        )


def test_original_mapping_preserves_unrelated_xmp(tmp_path: Path, app_config, monkeypatch) -> None:
    monkeypatch.setattr(xmp_writer, "find_exiftool", lambda: None)
    preview = tmp_path / "IMG_0001.jpg"
    preview.write_bytes(b"preview")
    results = tmp_path / "results.csv"
    _write_results(results, preview)
    raw = tmp_path / "originals" / "day1" / "IMG_0001.ARW"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    raw.with_suffix(".xmp").write_text(XMP, encoding="utf-8")
    changes = plan_xmp_changes(
        results,
        app_config,
        originals_dir=tmp_path / "originals",
        ratings_enabled=True,
        labels_enabled=True,
    )
    change = changes[0]
    assert change.xmp_path == raw.with_suffix(".xmp")
    assert change.keywords_after == ["Family", "ML_KEEP"]
    assert change.rating_before == 2
    assert change.rating_after == 5
    assert change.label_before == "Blue"
    assert change.label_after == "Green"


def test_write_requires_exiftool_after_creating_preview_report(
    tmp_path: Path, app_config, monkeypatch
) -> None:
    monkeypatch.setattr(xmp_writer, "find_exiftool", lambda: None)
    preview = tmp_path / "image.jpg"
    preview.write_bytes(b"preview")
    results = tmp_path / "results.csv"
    _write_results(results, preview)
    with pytest.raises(XmpError, match="ExifTool"):
        apply_xmp(results, app_config, write=True)
    assert results.with_name("results_xmp_changes.csv").is_file()


def test_ambiguous_original_stem_is_skipped(tmp_path: Path, app_config) -> None:
    preview = tmp_path / "same.jpg"
    preview.write_bytes(b"x")
    results = tmp_path / "results.csv"
    _write_results(results, preview)
    originals = tmp_path / "originals"
    for folder in ("a", "b"):
        raw = originals / folder / "same.ARW"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw")
    changes = plan_xmp_changes(results, app_config, originals_dir=originals)
    assert changes[0].status == "SKIP"
    assert "ambiguous" in changes[0].message


def test_staging_path_cannot_escape_output(tmp_path: Path, app_config) -> None:
    preview = tmp_path / "safe.jpg"
    preview.write_bytes(b"x")
    results = tmp_path / "results.csv"
    with results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "full_path", "relative_path", "decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "safe.jpg",
                "full_path": str(preview),
                "relative_path": "../../outside.jpg",
                "decision": "KEEP",
            }
        )
    changes = plan_xmp_changes(results, app_config, staging_dir=tmp_path / "staging")
    assert changes[0].xmp_path == (tmp_path / "staging" / "safe.xmp").resolve()
