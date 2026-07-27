"""Reference-free grouping of recurring uniform colour and pattern signatures."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

REFERENCE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True, slots=True)
class UniformGroup:
    """A visually similar group of preview images."""

    index: int
    files: tuple[Path, ...]
    samples: tuple[Path, ...]
    mean_rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class UniformDiscovery:
    """The usable preview inventory and its suggested visual groups."""

    scanned_files: int
    groups: tuple[UniformGroup, ...]


def _preview_paths(root: Path, max_images: int) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Preview folder does not exist: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in REFERENCE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    if not paths:
        raise ValueError(f"No JPEG previews found in {root}")
    if len(paths) <= max_images:
        return paths
    indexes = np.linspace(0, len(paths) - 1, max_images, dtype=int)
    return [paths[index] for index in indexes]


def _colour_pattern(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    width, height = image.size
    left, top = round(width * 0.2), round(height * 0.2)
    right, bottom = round(width * 0.8), round(height * 0.78)
    crop = image.crop((left, top, right, bottom)).resize((96, 96))
    rgb = np.asarray(crop, dtype=np.uint8)
    hsv = np.asarray(crop.convert("HSV"), dtype=np.uint8)
    # Six spatial regions retain simple stripe/panel patterns while the HSV histogram
    # makes the result less sensitive to lighting than an RGB average alone.
    features: list[np.ndarray] = []
    for y_start, y_end in ((0, 48), (48, 96)):
        for x_start, x_end in ((0, 32), (32, 64), (64, 96)):
            tile = hsv[y_start:y_end, x_start:x_end]
            histogram, _ = np.histogram(tile[..., 0], bins=12, range=(0, 256))
            features.append(histogram.astype(np.float32) / max(1, tile.shape[0] * tile.shape[1]))
    mean = rgb.mean(axis=(0, 1))
    return np.concatenate(features), (int(mean[0]), int(mean[1]), int(mean[2]))


def discover_uniform_groups(
    root: Path,
    *,
    group_count: int = 4,
    max_images: int = 400,
    samples_per_group: int = 6,
) -> UniformDiscovery:
    """Cluster central preview colour/pattern signatures without loading ML models.

    This intentionally produces candidate groups, not team labels. The user confirms
    which group represents their team before its previews become reference images.
    """

    if group_count < 1 or max_images < 1 or samples_per_group < 1:
        raise ValueError("group_count, max_images, and samples_per_group must be positive")
    candidates = _preview_paths(root, max_images)
    paths: list[Path] = []
    features: list[np.ndarray] = []
    colours: list[tuple[int, int, int]] = []
    for path in candidates:
        try:
            feature, colour = _colour_pattern(path)
        except (OSError, ValueError):
            continue
        paths.append(path)
        features.append(feature)
        colours.append(colour)
    if not paths:
        raise ValueError("No readable JPEG previews were available for uniform discovery")
    matrix = np.stack(features)
    clusters = min(group_count, len(paths))
    if clusters == 1:
        labels = np.zeros(len(paths), dtype=int)
        centers = matrix.mean(axis=0, keepdims=True)
    else:
        model = KMeans(n_clusters=clusters, n_init="auto", random_state=42)
        labels = model.fit_predict(matrix)
        centers = model.cluster_centers_
    groups: list[UniformGroup] = []
    for label in range(clusters):
        members = [index for index, assigned in enumerate(labels) if assigned == label]
        ranked = sorted(members, key=lambda index: float(np.linalg.norm(matrix[index] - centers[label])))
        mean_rgb = (
            round(np.mean([colours[index][0] for index in members])),
            round(np.mean([colours[index][1] for index in members])),
            round(np.mean([colours[index][2] for index in members])),
        )
        groups.append(
            UniformGroup(
                index=label + 1,
                files=tuple(paths[index] for index in members),
                samples=tuple(paths[index] for index in ranked[:samples_per_group]),
                mean_rgb=mean_rgb,
            )
        )
    groups.sort(key=lambda group: (-len(group.files), group.index))
    return UniformDiscovery(scanned_files=len(paths), groups=tuple(groups))


def promote_reference_images(files: tuple[Path, ...], destination: Path) -> list[Path]:
    """Copy selected JPEG previews to a reference folder without overwriting files."""

    if not files:
        raise ValueError("Choose a uniform group before promoting reference images")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in files:
        token = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
        output = destination / f"auto_reference_{token}{source.suffix.lower()}"
        if not output.exists():
            temporary = output.with_suffix(f"{output.suffix}.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(output)
        copied.append(output)
    return copied
