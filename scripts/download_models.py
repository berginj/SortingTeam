from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from photo_sorter.config import load_config
from photo_sorter.utils.cache import sha256_file


def _download_yolo(destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"YOLO already present: {destination}")
        return
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_cache = destination.parent / ".download" / "yolo"
    download_cache.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(download_cache)
        downloaded = Path(attempt_download_asset("yolo11n.pt")).resolve()
    finally:
        os.chdir(previous)
    if not downloaded.is_file():
        raise RuntimeError("Ultralytics did not return a downloaded YOLO checkpoint.")
    if downloaded != destination.resolve():
        shutil.copy2(downloaded, destination)
    print(f"YOLO downloaded: {destination}")


def _download_open_clip(config_path: Path, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"OpenCLIP already present: {destination}")
        return
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open_clip_torch is not installed.") from exc
    config = load_config(config_path)
    pretrained_config = open_clip.get_pretrained_cfg(config.clip.model_name, config.clip.pretrained)
    if not pretrained_config:
        raise RuntimeError(
            f"Unknown OpenCLIP checkpoint: {config.clip.model_name}/{config.clip.pretrained}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_cache = destination.parent / ".download"
    download_cache.mkdir(parents=True, exist_ok=True)
    downloaded_value = open_clip.download_pretrained(
        pretrained_config, cache_dir=str(download_cache)
    )
    downloaded = Path(downloaded_value).resolve()
    if not downloaded.is_file():
        raise RuntimeError("OpenCLIP did not return a downloaded checkpoint.")
    shutil.copy2(downloaded, destination)
    print(f"OpenCLIP downloaded: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download third-party model weights for offline photo sorting."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    _download_yolo(config.person_detection.model_path, args.force)
    _download_open_clip(config_path, config.clip.checkpoint_path, args.force)
    transient_cache = config.person_detection.model_path.parent / ".download"
    if transient_cache.is_dir():
        shutil.rmtree(transient_cache)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "models": [
            {
                "name": "YOLO11n",
                "path": str(config.person_detection.model_path),
                "sha256": sha256_file(config.person_detection.model_path),
                "source": "Ultralytics assets",
                "license_note": "Ultralytics package/model terms apply; see THIRD_PARTY_NOTICES.md",
            },
            {
                "name": f"{config.clip.model_name}/{config.clip.pretrained}",
                "path": str(config.clip.checkpoint_path),
                "sha256": sha256_file(config.clip.checkpoint_path),
                "source": "OpenCLIP pretrained registry",
                "license_note": "OpenCLIP code and pretrained dataset/model terms apply.",
            },
        ],
    }
    manifest_path = config.runtime.cache_dir / "models" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
