from __future__ import annotations

import hashlib
import re
from pathlib import Path


def discover_images(
    input_path: Path,
    extensions: list[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    root = input_path.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    extension_set = {item.lower() for item in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in extension_set),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def safe_artifact_stem(relative_path: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.with_suffix("").as_posix())
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{digest}"


def ensure_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".photo_sorter_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise PermissionError(f"Directory is not writable: {path}: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)
