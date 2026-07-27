from __future__ import annotations

import csv
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from photo_sorter.metadata.exif_reader import find_exiftool, read_xmp_fields
from photo_sorter.schemas.config_models import AppConfig
from photo_sorter.schemas.results import Decision


class XmpError(RuntimeError):
    """Raised when safe XMP preparation or writing fails."""


@dataclass(slots=True)
class XmpChange:
    filename: str
    source_path: Path
    xmp_path: Path | None
    decision: str
    status: str
    message: str
    keywords_before: list[str]
    keywords_after: list[str]
    rating_before: int | None
    rating_after: int | None
    label_before: str | None
    label_after: str | None

    def csv_row(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "source_path": str(self.source_path),
            "xmp_path": str(self.xmp_path) if self.xmp_path else "",
            "decision": self.decision,
            "status": self.status,
            "message": self.message,
            "keywords_before": ";".join(self.keywords_before),
            "keywords_after": ";".join(self.keywords_after),
            "rating_before": self.rating_before if self.rating_before is not None else "",
            "rating_after": self.rating_after if self.rating_after is not None else "",
            "label_before": self.label_before or "",
            "label_after": self.label_after or "",
        }


def _parse_xmp_fallback(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"keywords": [], "rating": None, "label": None}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {"keywords": [], "rating": None, "label": None}
    namespaces = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "xmp": "http://ns.adobe.com/xap/1.0/",
    }
    keywords = [
        item.text or ""
        for item in root.findall(".//dc:subject/rdf:Bag/rdf:li", namespaces)
        if item.text
    ]
    rating_text = root.findtext(".//xmp:Rating", namespaces=namespaces)
    label = root.findtext(".//xmp:Label", namespaces=namespaces)
    if rating_text is None or label is None:
        description = root.find(".//rdf:Description", namespaces)
        if description is not None:
            rating_text = rating_text or description.get(f"{{{namespaces['xmp']}}}Rating")
            label = label or description.get(f"{{{namespaces['xmp']}}}Label")
    try:
        rating = int(rating_text) if rating_text is not None else None
    except ValueError:
        rating = None
    return {"keywords": keywords, "rating": rating, "label": label}


def _existing_metadata(path: Path, exiftool: Path | None) -> dict[str, Any]:
    if not path.is_file():
        return {"keywords": [], "rating": None, "label": None}
    if exiftool is not None:
        try:
            return read_xmp_fields(path, exiftool)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return _parse_xmp_fallback(path)


def _raw_index(originals_dir: Path, extensions: list[str]) -> dict[str, list[Path]]:
    extension_set = {item.casefold() for item in extensions}
    index: dict[str, list[Path]] = {}
    for path in originals_dir.rglob("*"):
        if path.is_file() and path.suffix.casefold() in extension_set:
            index.setdefault(path.stem.casefold(), []).append(path.resolve())
    return index


def _map_original(
    row: pd.Series,
    originals_dir: Path,
    raw_index: dict[str, list[Path]],
    extensions: list[str],
) -> tuple[Path | None, str]:
    relative_value = str(row.get("relative_path", "") or "")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        relative = Path(str(row.get("filename", "")))
    parent = originals_dir / relative.parent
    stem = Path(str(row.get("filename", relative.name))).stem
    exact = [
        (parent / f"{stem}{extension}").resolve()
        for extension in extensions
        if (parent / f"{stem}{extension}").is_file()
    ]
    if len(exact) == 1:
        return exact[0], "relative path"
    if len(exact) > 1:
        return None, "multiple RAW files match the relative path and stem"
    candidates = raw_index.get(stem.casefold(), [])
    if len(candidates) == 1:
        return candidates[0], "unique filename stem"
    if not candidates:
        return None, "no RAW file with a matching stem was found"
    return None, "filename stem is ambiguous under the originals directory"


def _effective_decision(row: pd.Series) -> str:
    for column in ("effective_decision", "override_decision", "decision"):
        value = row.get(column)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip().upper()
    return Decision.ERROR.value


def plan_xmp_changes(
    results_path: Path,
    config: AppConfig,
    *,
    originals_dir: Path | None = None,
    staging_dir: Path | None = None,
    keywords_enabled: bool | None = None,
    ratings_enabled: bool | None = None,
    labels_enabled: bool | None = None,
) -> list[XmpChange]:
    if not results_path.is_file():
        raise XmpError(f"Results CSV does not exist: {results_path}")
    try:
        frame = pd.read_csv(results_path)
    except Exception as exc:
        raise XmpError(f"Could not read results CSV {results_path}: {exc}") from exc
    required = {"filename", "full_path", "decision"}
    missing = required - set(frame.columns)
    if missing:
        raise XmpError(f"Results CSV is missing columns: {', '.join(sorted(missing))}")

    tool = find_exiftool()
    use_keywords = (
        config.metadata_mapping.keywords.enabled if keywords_enabled is None else keywords_enabled
    )
    use_ratings = (
        config.metadata_mapping.ratings.enabled if ratings_enabled is None else ratings_enabled
    )
    use_labels = (
        config.metadata_mapping.color_labels.enabled if labels_enabled is None else labels_enabled
    )
    original_root = originals_dir.expanduser().resolve() if originals_dir else None
    raw_index = (
        _raw_index(original_root, config.metadata_mapping.raw_extensions) if original_root else {}
    )
    stage_root = (staging_dir or config.output.xmp_staging_dir).expanduser().resolve()
    controlled_keywords = {
        mapping.keyword for mapping in config.metadata_mapping.decisions.values() if mapping.keyword
    }
    changes: list[XmpChange] = []
    for _, row in frame.iterrows():
        decision = _effective_decision(row)
        source = Path(str(row["full_path"])).expanduser().resolve()
        mapping = config.metadata_mapping.decisions.get(decision)
        if mapping is None:
            changes.append(
                XmpChange(
                    filename=str(row["filename"]),
                    source_path=source,
                    xmp_path=None,
                    decision=decision,
                    status="SKIP",
                    message=f"No metadata mapping exists for decision {decision}",
                    keywords_before=[],
                    keywords_after=[],
                    rating_before=None,
                    rating_after=None,
                    label_before=None,
                    label_after=None,
                )
            )
            continue
        if mapping.keyword is None and mapping.rating is None and mapping.color_label is None:
            changes.append(
                XmpChange(
                    filename=str(row["filename"]),
                    source_path=source,
                    xmp_path=None,
                    decision=decision,
                    status="SKIP",
                    message=f"Decision {decision} has no metadata mapping",
                    keywords_before=[],
                    keywords_after=[],
                    rating_before=None,
                    rating_after=None,
                    label_before=None,
                    label_after=None,
                )
            )
            continue
        if original_root:
            raw_source, mapping_message = _map_original(
                row,
                original_root,
                raw_index,
                config.metadata_mapping.raw_extensions,
            )
            if raw_source is None:
                changes.append(
                    XmpChange(
                        filename=str(row["filename"]),
                        source_path=source,
                        xmp_path=None,
                        decision=decision,
                        status="SKIP",
                        message=mapping_message,
                        keywords_before=[],
                        keywords_after=[],
                        rating_before=None,
                        rating_after=None,
                        label_before=None,
                        label_after=None,
                    )
                )
                continue
            source = raw_source
            xmp_path = raw_source.with_suffix(".xmp")
            message = f"Mapped by {mapping_message}"
        else:
            relative_value = str(row.get("relative_path", row["filename"]))
            relative = Path(relative_value)
            if relative.is_absolute() or ".." in relative.parts:
                relative = Path(str(row["filename"]))
            relative = relative.with_suffix(".xmp")
            xmp_path = (stage_root / relative).resolve()
            message = "Staged sidecar"

        current = _existing_metadata(xmp_path, tool)
        keywords_before = [str(value) for value in current.get("keywords", [])]
        preserved = [value for value in keywords_before if value not in controlled_keywords]
        keywords_after = preserved.copy()
        if use_keywords and mapping.keyword:
            keywords_after.append(mapping.keyword)
        elif not use_keywords:
            keywords_after = keywords_before.copy()
        rating_before = current.get("rating")
        rating_after = (
            mapping.rating if use_ratings and mapping.rating is not None else rating_before
        )
        label_before = current.get("label")
        label_after = (
            mapping.color_label if use_labels and mapping.color_label is not None else label_before
        )
        changed = (
            keywords_after != keywords_before
            or rating_after != rating_before
            or label_after != label_before
        )
        changes.append(
            XmpChange(
                filename=str(row["filename"]),
                source_path=source,
                xmp_path=xmp_path,
                decision=decision,
                status="CHANGE" if changed else "UNCHANGED",
                message=message,
                keywords_before=keywords_before,
                keywords_after=keywords_after,
                rating_before=rating_before,
                rating_after=rating_after,
                label_before=label_before,
                label_after=label_after,
            )
        )
    return changes


def _write_preview(changes: list[XmpChange], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(changes[0].csv_row())
        if changes
        else [
            "filename",
            "source_path",
            "xmp_path",
            "decision",
            "status",
            "message",
            "keywords_before",
            "keywords_after",
            "rating_before",
            "rating_after",
            "label_before",
            "label_after",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for change in changes:
            writer.writerow(change.csv_row())


def _backup_sidecar(path: Path, backup_root: Path, timestamp: str) -> Path | None:
    if not path.is_file():
        return None
    safe_parent = path.parent.as_posix().replace(":", "").lstrip("/")
    destination = backup_root / timestamp / safe_parent / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


def _write_one(change: XmpChange, tool: Path) -> None:
    if change.xmp_path is None:
        return
    target = change.xmp_path
    target.parent.mkdir(parents=True, exist_ok=True)
    arguments = [str(tool)]
    if target.exists():
        arguments.extend(["-overwrite_original", "-P"])
    else:
        arguments.extend(["-o", str(target)])

    arguments.append("-XMP-dc:Subject=")
    for keyword in change.keywords_after:
        arguments.append(f"-XMP-dc:Subject+={keyword}")
    if change.rating_after is not None:
        arguments.append(f"-XMP-xmp:Rating={change.rating_after}")
    if change.label_after is not None:
        arguments.append(f"-XMP-xmp:Label={change.label_after}")
    arguments.append(str(target if target.exists() else change.source_path))
    completed = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not target.is_file():
        raise XmpError(
            f"ExifTool failed for {target}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def apply_xmp(
    results_path: Path,
    config: AppConfig,
    *,
    write: bool = False,
    originals_dir: Path | None = None,
    staging_dir: Path | None = None,
    preview_path: Path | None = None,
    keywords_enabled: bool | None = None,
    ratings_enabled: bool | None = None,
    labels_enabled: bool | None = None,
) -> tuple[list[XmpChange], Path]:
    changes = plan_xmp_changes(
        results_path,
        config,
        originals_dir=originals_dir,
        staging_dir=staging_dir,
        keywords_enabled=keywords_enabled,
        ratings_enabled=ratings_enabled,
        labels_enabled=labels_enabled,
    )
    report = preview_path or results_path.with_name(f"{results_path.stem}_xmp_changes.csv")
    _write_preview(changes, report)
    if not write:
        return changes, report
    tool = find_exiftool()
    if tool is None:
        raise XmpError(
            "ExifTool is required for --write but was not found on PATH. No sidecars were changed."
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for change in changes:
        if change.status != "CHANGE" or change.xmp_path is None:
            continue
        try:
            _backup_sidecar(change.xmp_path, config.output.xmp_backup_dir, timestamp)
            _write_one(change, tool)
            change.status = "WRITTEN"
        except (OSError, subprocess.SubprocessError, XmpError) as exc:
            change.status = "ERROR"
            change.message = f"{change.message}; {exc}"
    _write_preview(changes, report)
    return changes, report
