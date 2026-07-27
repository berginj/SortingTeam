from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from photo_sorter.schemas.config_models import BurstGroupingConfig
from photo_sorter.schemas.results import ImageResult

_SEQUENCE_RE = re.compile(r"^(.*?)(\d+)$")


def _timestamp(result: ImageResult) -> float | None:
    if result.capture_time is None:
        return None
    try:
        return result.capture_time.timestamp()
    except (OSError, ValueError):
        naive_epoch = datetime(1970, 1, 1)
        return (result.capture_time.replace(tzinfo=None) - naive_epoch).total_seconds()


def _fallback_parts(result: ImageResult) -> tuple[str, int] | None:
    stem = Path(result.filename).stem
    match = _SEQUENCE_RE.match(stem)
    if not match:
        return None
    return match.group(1).casefold(), int(match.group(2))


def _burst_hash(results: list[ImageResult], mode: str) -> str:
    seed = f"{mode}|{results[0].relative_path}|{len(results)}"
    return f"B-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:10].upper()}"


def _split_timestamp_groups(items: list[ImageResult], max_gap: float) -> list[list[ImageResult]]:
    grouped: list[list[ImageResult]] = []
    buckets: dict[tuple[str, str], list[ImageResult]] = defaultdict(list)
    for item in items:
        parent = Path(item.relative_path).parent.as_posix().casefold()
        buckets[(item.camera_id or "", parent)].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (_timestamp(item) or 0, item.relative_path.casefold()))
        current: list[ImageResult] = []
        previous: float | None = None
        for item in bucket:
            current_time = _timestamp(item)
            if (
                current
                and previous is not None
                and current_time is not None
                and current_time - previous > max_gap
            ):
                grouped.append(current)
                current = []
            current.append(item)
            previous = current_time
        if current:
            grouped.append(current)
    return grouped


def _split_fallback_groups(items: list[ImageResult], max_gap: float) -> list[list[ImageResult]]:
    grouped: list[list[ImageResult]] = []
    buckets: dict[tuple[str, str], list[tuple[int, ImageResult]]] = defaultdict(list)
    singletons: list[ImageResult] = []
    for item in items:
        parts = _fallback_parts(item)
        if parts is None or item.file_mtime is None:
            singletons.append(item)
            continue
        parent = Path(item.relative_path).parent.as_posix().casefold()
        prefix, sequence = parts
        buckets[(parent, prefix)].append((sequence, item))
    for bucket in buckets.values():
        bucket.sort(key=lambda entry: (entry[0], entry[1].relative_path.casefold()))
        current: list[ImageResult] = []
        previous_sequence: int | None = None
        previous_mtime: float | None = None
        for sequence, item in bucket:
            new_group = current and (
                previous_sequence is None
                or sequence != previous_sequence + 1
                or previous_mtime is None
                or item.file_mtime is None
                or abs(item.file_mtime - previous_mtime) > max_gap
            )
            if new_group:
                grouped.append(current)
                current = []
            current.append(item)
            previous_sequence = sequence
            previous_mtime = item.file_mtime
        if current:
            grouped.append(current)
    grouped.extend([[item] for item in singletons])
    return grouped


def assign_bursts(results: list[ImageResult], config: BurstGroupingConfig) -> None:
    if not results:
        return
    if not config.enabled:
        groups = [[item] for item in results]
    else:
        with_timestamp = [item for item in results if item.capture_time is not None]
        without_timestamp = [item for item in results if item.capture_time is None]
        groups = _split_timestamp_groups(with_timestamp, config.max_gap_seconds)
        groups.extend(_split_fallback_groups(without_timestamp, config.max_gap_seconds))

    for group in groups:
        mode = "time" if group[0].capture_time is not None else "fallback"
        burst_id = _burst_hash(group, mode)

        def ranking_key(item: ImageResult) -> tuple[float, float, str]:
            uniform = item.uniform_probability if item.uniform_probability is not None else -1.0
            focus = item.normalized_focus_score if item.normalized_focus_score is not None else -1.0
            composite = config.uniform_weight * uniform + config.focus_weight * focus
            confidence = (
                item.person_detection_confidence
                if item.person_detection_confidence is not None
                else -1.0
            )
            return (-composite, -confidence, item.relative_path.casefold())

        ranked = sorted(group, key=ranking_key)
        for rank, item in enumerate(ranked, start=1):
            item.burst_id = burst_id
            item.burst_size = len(group)
            item.burst_rank = rank
            item.is_best_in_burst = rank == 1
