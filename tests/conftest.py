from __future__ import annotations

from pathlib import Path

import pytest

from photo_sorter.config import load_config
from photo_sorter.schemas.config_models import AppConfig


@pytest.fixture()
def app_config(tmp_path: Path) -> AppConfig:
    config = load_config(Path("config/default.yaml"))
    config.runtime.cache_dir = tmp_path / "cache"
    config.runtime.target_references = tmp_path / "references" / "target"
    config.runtime.other_references = tmp_path / "references" / "other"
    config.person_detection.model_path = tmp_path / "models" / "yolo11n.pt"
    config.clip.checkpoint_path = tmp_path / "models" / "clip.pt"
    config.output.audit_dir = tmp_path / "output" / "audit"
    config.output.xmp_staging_dir = tmp_path / "output" / "xmp"
    config.output.xmp_backup_dir = tmp_path / "output" / "backups"
    config.logging.file = tmp_path / "output" / "log.jsonl"
    config.logging.file_enabled = False
    config.runtime.target_references.mkdir(parents=True)
    config.runtime.other_references.mkdir(parents=True)
    return config
