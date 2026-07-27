from __future__ import annotations

from pathlib import Path

from PIL import Image

from photo_sorter.processing.image_loader import _parse_datetime, load_image
from photo_sorter.utils.cache import (
    atomic_write_json,
    clear_runtime_cache,
    sha256_file,
    stable_fingerprint,
)
from photo_sorter.utils.paths import discover_images, safe_artifact_stem


def test_cache_helpers_and_model_preservation(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    models = cache / "models"
    models.mkdir(parents=True)
    (models / "weight.pt").write_bytes(b"weight")
    generated = cache / "uniform"
    generated.mkdir()
    source = generated / "value.bin"
    source.write_bytes(b"abc")
    assert len(sha256_file(source)) == 64
    assert stable_fingerprint({"b": 2, "a": 1}) == stable_fingerprint({"a": 1, "b": 2})
    atomic_write_json(generated / "manifest.json", {"ok": True})
    removed = clear_runtime_cache(cache)
    assert removed
    assert models.is_dir()
    clear_runtime_cache(cache, include_models=True)
    assert not models.exists()


def test_image_loading_discovery_and_datetime(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "Photo 01.JPG"
    Image.new("RGB", (32, 24), "red").save(path)
    loaded = load_image(path)
    assert (loaded.width, loaded.height) == (32, 24)
    assert discover_images(tmp_path, [".jpg"]) == [path]
    assert safe_artifact_stem(Path("nested/Photo 01.JPG")).startswith("nested_Photo_01")
    parsed = _parse_datetime("2026:01:02 03:04:05", "12", "-05:00")
    assert parsed is not None
    assert parsed.microsecond == 120000
    assert parsed.utcoffset() is not None
    assert _parse_datetime("not a date", None, None) is None
