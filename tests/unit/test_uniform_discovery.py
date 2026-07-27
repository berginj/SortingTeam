from __future__ import annotations

from pathlib import Path

from PIL import Image

from photo_sorter.processing.uniform_discovery import (
    discover_uniform_groups,
    promote_reference_images,
)


def _image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 60), colour).save(path)


def test_discovers_groups_and_promotes_selected_previews(tmp_path: Path) -> None:
    previews = tmp_path / "previews"
    previews.mkdir()
    for index in range(3):
        _image(previews / f"blue-{index}.jpg", (15, 35, 160))
        _image(previews / f"red-{index}.jpg", (170, 25, 20))

    discovery = discover_uniform_groups(previews, group_count=2, samples_per_group=2)

    assert discovery.scanned_files == 6
    assert len(discovery.groups) == 2
    assert sum(len(group.files) for group in discovery.groups) == 6
    assert all(group.samples for group in discovery.groups)

    references = tmp_path / "references"
    copied = promote_reference_images(discovery.groups[0].samples, references)

    assert len(copied) == len(discovery.groups[0].samples)
    assert all(path.is_file() for path in copied)
    assert promote_reference_images(discovery.groups[0].samples, references) == copied


def test_discovery_requires_jpeg_previews(tmp_path: Path) -> None:
    (tmp_path / "not-an-image.txt").write_text("nope", encoding="utf-8")

    try:
        discover_uniform_groups(tmp_path)
    except ValueError as exc:
        assert "No JPEG previews" in str(exc)
    else:
        raise AssertionError("Expected missing preview error")
