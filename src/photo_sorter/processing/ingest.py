"""Non-destructive local ingest, preview preparation, and original staging.

This module deliberately only copies files.  It is designed for a card-to-archive
workflow where the archive remains the immutable source of truth and Lightroom
receives a smaller, selected copy later.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from photo_sorter.metadata.exif_reader import find_exiftool
from photo_sorter.schemas.results import Decision
from photo_sorter.utils.cache import sha256_file

RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rwl",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}
PREVIEW_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class IngestError(RuntimeError):
    """Raised when an ingest operation is unsafe or its inputs are invalid."""


@dataclass(slots=True)
class TransferRecord:
    relative_path: str
    source_path: str
    destination_path: str
    size_bytes: int
    source_sha256: str = ""
    destination_sha256: str = ""
    status: str = "PENDING"
    message: str = ""
    completed_at: str = ""


@dataclass(slots=True)
class TransferManifest:
    operation: str
    source_root: str
    destination_root: str
    created_at: str
    updated_at: str
    records: list[TransferRecord] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        values: dict[str, int] = {"total": len(self.records)}
        for record in self.records:
            values[record.status] = values.get(record.status, 0) + 1
        return values


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_directory(value: Path, label: str, *, create: bool = False) -> Path:
    path = value.expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise IngestError(f"{label} is not a directory: {path}")
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_distinct_roots(source: Path, destination: Path) -> tuple[Path, Path]:
    """Resolve roots and prevent copying a folder into itself or its child."""
    source_root = _resolve_directory(source, "Source")
    destination_root = _resolve_directory(destination, "Destination", create=True)
    if source_root == destination_root or _is_within(destination_root, source_root):
        raise IngestError("Destination must be outside the source directory.")
    return source_root, destination_root


def discover_media(root: Path, extensions: Iterable[str] | None = None) -> list[Path]:
    root = _resolve_directory(root, "Source")
    extension_set = {
        extension.casefold() for extension in (extensions or RAW_EXTENSIONS | PREVIEW_EXTENSIONS)
    }
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in extension_set
        ),
        key=lambda path: str(path).casefold(),
    )


def _write_manifest(manifest: TransferManifest, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = _now()
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_manifest(path: Path) -> TransferManifest:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
        records = [TransferRecord(**record) for record in data.pop("records", [])]
        return TransferManifest(records=records, **data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IngestError(f"Could not read transfer manifest {path}: {exc}") from exc


def _copy_and_verify(source: Path, destination: Path) -> tuple[str, str]:
    """Copy atomically and verify the final destination against the source hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        shutil.copy2(source, temporary)
        destination_hash = sha256_file(temporary)
        if source_hash != destination_hash:
            raise IngestError(f"Checksum mismatch while copying {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash, destination_hash


def copy_to_archive(
    source: Path,
    destination: Path,
    manifest_path: Path,
    *,
    write: bool = False,
) -> TransferManifest:
    """Plan or execute a verified, resumable copy from a card/source to an archive."""
    source_root, destination_root = validate_distinct_roots(source, destination)
    manifest = TransferManifest(
        operation="ingest_copy",
        source_root=str(source_root),
        destination_root=str(destination_root),
        created_at=_now(),
        updated_at=_now(),
    )
    for item in discover_media(source_root):
        relative = item.relative_to(source_root)
        target = destination_root / relative
        record = TransferRecord(
            relative_path=str(relative),
            source_path=str(item),
            destination_path=str(target),
            size_bytes=item.stat().st_size,
        )
        if not write:
            record.status = "PLANNED"
            record.message = "Dry run; pass --write to copy and verify files."
        elif target.exists():
            if not target.is_file():
                record.status = "ERROR"
                record.message = "Destination exists but is not a regular file."
            else:
                record.source_sha256 = sha256_file(item)
                record.destination_sha256 = sha256_file(target)
                if record.source_sha256 == record.destination_sha256:
                    record.status = "VERIFIED_EXISTING"
                    record.message = "Existing destination matches source; nothing copied."
                else:
                    record.status = "ERROR"
                    record.message = "Destination exists with different content; no overwrite."
        else:
            try:
                record.source_sha256, record.destination_sha256 = _copy_and_verify(item, target)
                record.status = "COPIED"
                record.message = "Copied and SHA-256 verified."
            except (IngestError, OSError) as exc:
                record.status = "ERROR"
                record.message = str(exc)
        record.completed_at = _now()
        manifest.records.append(record)
        _write_manifest(manifest, manifest_path)
    return manifest


def _save_preview(image: Image.Image, output: Path, max_edge: int) -> None:
    if max_edge < 256:
        raise IngestError("Preview long edge must be at least 256 pixels.")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = ImageOps.exif_transpose(image).convert("RGB")
    rendered.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    temporary = output.with_name(f".{output.name}.partial")
    try:
        rendered.save(temporary, format="JPEG", quality=85, optimize=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_raw_preview(source: Path, exiftool: Path) -> Image.Image:
    errors: list[str] = []
    for tag in ("PreviewImage", "JpgFromRaw"):
        process = subprocess.run(
            [str(exiftool), "-b", f"-{tag}", str(source)],
            check=False,
            capture_output=True,
        )
        if process.returncode == 0 and process.stdout:
            try:
                with Image.open(io.BytesIO(process.stdout)) as image:
                    return image.copy()
            except OSError as exc:
                errors.append(f"{tag}: {exc}")
        else:
            errors.append(process.stderr.decode("utf-8", errors="replace").strip())
    detail = "; ".join(error for error in errors if error) or "no embedded JPEG found"
    raise IngestError(f"Could not extract an embedded preview from {source.name}: {detail}")


def prepare_previews(
    source: Path,
    destination: Path,
    manifest_path: Path,
    *,
    max_edge: int = 2560,
    write: bool = False,
) -> TransferManifest:
    """Create analysis JPEGs from local JPEGs or ExifTool-extracted RAW previews."""
    source_root, destination_root = validate_distinct_roots(source, destination)
    tool = find_exiftool()
    manifest = TransferManifest(
        operation="prepare_previews",
        source_root=str(source_root),
        destination_root=str(destination_root),
        created_at=_now(),
        updated_at=_now(),
    )
    for item in discover_media(source_root):
        relative = item.relative_to(source_root)
        target = (destination_root / relative).with_suffix(".jpg")
        record = TransferRecord(str(relative), str(item), str(target), item.stat().st_size)
        if not write:
            record.status = "PLANNED"
            record.message = "Dry run; pass --write to create temporary JPEG previews."
        elif target.exists():
            record.status = "SKIPPED_EXISTING"
            record.message = "Preview already exists; no overwrite."
        else:
            try:
                if item.suffix.casefold() in PREVIEW_EXTENSIONS:
                    with Image.open(item) as image:
                        _save_preview(image, target, max_edge)
                elif tool is None:
                    raise IngestError(
                        "ExifTool is required to extract embedded previews from RAW files."
                    )
                else:
                    _save_preview(_extract_raw_preview(item, tool), target, max_edge)
                record.destination_sha256 = sha256_file(target)
                record.status = "CREATED"
                record.message = "Local analysis preview created."
            except (IngestError, OSError, subprocess.SubprocessError) as exc:
                record.status = "ERROR"
                record.message = str(exc)
        record.completed_at = _now()
        manifest.records.append(record)
        _write_manifest(manifest, manifest_path)
    return manifest


def _effective_decision(row: pd.Series) -> str:
    for column in ("effective_decision", "override_decision", "decision"):
        value = row.get(column)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip().upper()
    return Decision.ERROR.value


def _original_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for item in discover_media(root, RAW_EXTENSIONS | PREVIEW_EXTENSIONS):
        index.setdefault(item.stem.casefold(), []).append(item)
    return index


def _map_original(
    row: pd.Series, root: Path, index: dict[str, list[Path]]
) -> tuple[Path | None, str]:
    relative = Path(str(row.get("relative_path", "") or ""))
    filename = Path(str(row.get("filename", "")))
    if relative.is_absolute() or ".." in relative.parts:
        relative = filename
    stem = filename.stem or relative.stem
    candidates = [
        path
        for path in (root / relative.parent).glob(f"{stem}.*")
        if path.is_file() and path.suffix.casefold() in RAW_EXTENSIONS | PREVIEW_EXTENSIONS
    ]
    if len(candidates) == 1:
        return candidates[0], "relative path"
    matches = index.get(stem.casefold(), [])
    if len(matches) == 1:
        return matches[0], "unique filename stem"
    if not matches:
        return None, "no original with a matching filename stem"
    return None, "original filename stem is ambiguous"


def stage_selected_originals(
    results_path: Path,
    originals_root: Path,
    destination: Path,
    manifest_path: Path,
    *,
    decisions: set[str] | None = None,
    write: bool = False,
) -> TransferManifest:
    """Copy selected originals for Lightroom import; source files are never moved."""
    if not results_path.is_file():
        raise IngestError(f"Results CSV does not exist: {results_path}")
    source_root, destination_root = validate_distinct_roots(originals_root, destination)
    try:
        results = pd.read_csv(results_path)
    except Exception as exc:
        raise IngestError(f"Could not read results CSV {results_path}: {exc}") from exc
    missing = {"filename", "decision"} - set(results.columns)
    if missing:
        raise IngestError(f"Results CSV is missing columns: {', '.join(sorted(missing))}")
    selected = {
        value.upper() for value in (decisions or {Decision.KEEP.value, Decision.REVIEW.value})
    }
    manifest = TransferManifest(
        "stage_selected", str(source_root), str(destination_root), _now(), _now()
    )
    index = _original_index(source_root)
    for _, row in results.iterrows():
        decision = _effective_decision(row)
        if decision not in selected:
            continue
        original, mapping = _map_original(row, source_root, index)
        relative = str(row.get("relative_path", row.get("filename", "")))
        if original is None:
            record = TransferRecord(relative, "", "", 0, status="ERROR", message=mapping)
        else:
            target_relative = original.relative_to(source_root)
            target = destination_root / target_relative
            record = TransferRecord(
                str(target_relative), str(original), str(target), original.stat().st_size
            )
            if not write:
                record.status = "PLANNED"
                record.message = f"Dry run; matched by {mapping}."
            elif target.exists():
                record.source_sha256 = sha256_file(original)
                record.destination_sha256 = sha256_file(target) if target.is_file() else ""
                if record.source_sha256 == record.destination_sha256:
                    record.status = "VERIFIED_EXISTING"
                    record.message = "Existing staged copy matches original."
                else:
                    record.status = "ERROR"
                    record.message = "Destination collision; no overwrite."
            else:
                try:
                    record.source_sha256, record.destination_sha256 = _copy_and_verify(
                        original, target
                    )
                    record.status = "COPIED"
                    record.message = f"Copied for Lightroom import; matched by {mapping}."
                except (IngestError, OSError) as exc:
                    record.status = "ERROR"
                    record.message = str(exc)
        record.completed_at = _now()
        manifest.records.append(record)
        _write_manifest(manifest, manifest_path)
    return manifest
