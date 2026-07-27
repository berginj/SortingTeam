from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from photo_sorter.schemas.config_models import AppConfig


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


def _resolve_path(value: Path, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()


def _resolve_config_paths(config: AppConfig, config_path: Path) -> AppConfig:
    base_dir = config_path.parent.resolve()
    config.runtime.cache_dir = _resolve_path(config.runtime.cache_dir, base_dir)
    config.runtime.target_references = _resolve_path(config.runtime.target_references, base_dir)
    config.runtime.other_references = _resolve_path(config.runtime.other_references, base_dir)
    config.person_detection.model_path = _resolve_path(config.person_detection.model_path, base_dir)
    config.clip.checkpoint_path = _resolve_path(config.clip.checkpoint_path, base_dir)
    config.output.audit_dir = _resolve_path(config.output.audit_dir, base_dir)
    config.output.xmp_staging_dir = _resolve_path(config.output.xmp_staging_dir, base_dir)
    config.output.xmp_backup_dir = _resolve_path(config.output.xmp_backup_dir, base_dir)
    config.logging.file = _resolve_path(config.logging.file, base_dir)
    return config


def load_config(path: Path | str) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read YAML configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        details = "\n".join(
            f"- {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"Invalid configuration {config_path}:\n{details}") from exc
    return _resolve_config_paths(config, config_path)


def dump_config(config: AppConfig) -> str:
    return yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
