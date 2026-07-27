from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from photo_sorter.models.clip_embedder import ImageEmbedder, OpenClipEmbedder, choose_device
from photo_sorter.models.person_detector import (
    PersonDetector,
    UltralyticsPersonDetector,
)
from photo_sorter.models.uniform_classifier import (
    UniformModel,
    target_person_score,
    train_or_load_uniform_model,
)
from photo_sorter.processing.burst_grouping import assign_bursts
from photo_sorter.processing.color_score import calculate_color_score
from photo_sorter.processing.crop_utils import (
    centrality_score,
    crop_image,
    crop_quality_score,
    pad_bbox,
    upper_body_bbox,
)
from photo_sorter.processing.decision_engine import apply_decisions
from photo_sorter.processing.focus_score import (
    FocusEvaluationError,
    calculate_focus_metrics,
    normalize_focus_results,
)
from photo_sorter.processing.image_loader import ImageLoadError, LoadedImage, load_image
from photo_sorter.schemas.config_models import AppConfig
from photo_sorter.schemas.results import (
    Decision,
    ImageResult,
    PersonDetection,
    PersonEvaluation,
    write_results_csv,
)
from photo_sorter.utils.cache import atomic_write_json
from photo_sorter.utils.paths import discover_images, safe_artifact_stem
from photo_sorter.utils.timing import Timer


@dataclass(slots=True)
class PipelineDependencies:
    detector: PersonDetector
    embedder: ImageEmbedder
    uniform_model: UniformModel
    device: str


@dataclass(slots=True)
class WorkImage:
    path: Path
    relative_path: Path
    result: ImageResult
    timer: Timer
    loaded: LoadedImage | None = None
    detections: list[PersonDetection] = field(default_factory=list)
    people: list[PersonEvaluation] = field(default_factory=list)
    focus_details: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    results: list[ImageResult]
    output_path: Path
    collection_csv_path: Path
    device: str
    classifier_mode: str


def build_dependencies(
    config: AppConfig,
    logger: logging.Logger,
    *,
    force_uniform_training: bool = False,
) -> PipelineDependencies:
    if config.runtime.offline:
        ultralytics_config_dir = (config.runtime.cache_dir / "ultralytics").resolve()
        ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config_dir)
    device = choose_device(config.runtime.device)
    logger.info(
        "Loading local models",
        extra={"event": "models_loading", "device": device, "offline": config.runtime.offline},
    )
    detector = UltralyticsPersonDetector(config.person_detection, device)
    embedder = OpenClipEmbedder(config.clip, device)
    uniform_model = train_or_load_uniform_model(
        detector,
        embedder,
        config,
        logger,
        force=force_uniform_training,
    )
    return PipelineDependencies(
        detector=detector,
        embedder=embedder,
        uniform_model=uniform_model,
        device=device,
    )


def _new_result(path: Path, root: Path) -> ImageResult:
    relative = path.relative_to(root)
    return ImageResult(
        filename=path.name,
        full_path=str(path.resolve()),
        relative_path=relative.as_posix(),
        decision=Decision.ERROR,
    )


def _safe_load(work: WorkImage) -> None:
    try:
        loaded = load_image(work.path)
        work.loaded = loaded
        work.result.image_width = loaded.width
        work.result.image_height = loaded.height
        work.result.capture_time = loaded.capture_time
        work.result.camera_id = loaded.camera_id
        work.result.file_mtime = loaded.file_mtime
    except (ImageLoadError, OSError, ValueError) as exc:
        work.result.error = str(exc)


def _detect_chunk(works: list[WorkImage], detector: PersonDetector, logger: logging.Logger) -> None:
    valid = [work for work in works if work.loaded is not None and not work.result.error]
    if not valid:
        return
    try:
        batches = detector.detect_batch([work.loaded.image for work in valid if work.loaded])
        if len(batches) != len(valid):
            raise RuntimeError("detector returned an unexpected batch size")
        for work, detections in zip(valid, batches, strict=True):
            work.detections = detections
            work.result.people_detected = len(detections)
    except Exception as batch_exc:
        logger.warning(
            "Batch detection failed; retrying images individually",
            extra={"event": "detector_batch_retry", "error": str(batch_exc)},
        )
        for work in valid:
            try:
                assert work.loaded is not None
                work.detections = detector.detect(work.loaded.image)
                work.result.people_detected = len(work.detections)
            except Exception as exc:
                work.result.error = f"Person detection failed: {exc}"


def _evaluate_people(
    works: list[WorkImage],
    dependencies: PipelineDependencies,
    config: AppConfig,
    logger: logging.Logger,
) -> None:
    crop_entries: list[
        tuple[
            WorkImage,
            PersonDetection,
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            Image.Image,
        ]
    ] = []
    for work in works:
        if work.loaded is None or work.result.error:
            continue
        for detection in work.detections:
            try:
                padded = pad_bbox(
                    detection.bbox,
                    config.person_detection.crop_padding,
                    work.loaded.width,
                    work.loaded.height,
                )
                upper = upper_body_bbox(
                    padded,
                    config.person_detection.upper_body_ratio,
                    work.loaded.width,
                    work.loaded.height,
                )
                crop_entries.append(
                    (
                        work,
                        detection,
                        padded,
                        upper,
                        crop_image(work.loaded.image, upper),
                    )
                )
            except ValueError as exc:
                work.warnings.append(f"Detection {detection.index} ignored: {exc}")
    if not crop_entries:
        return
    crops = [entry[4] for entry in crop_entries]
    try:
        embeddings = dependencies.embedder.embed(crops)
        evidence = dependencies.uniform_model.predict(embeddings)
    except Exception as batch_exc:
        logger.warning(
            "Batch embedding failed; retrying crops individually",
            extra={"event": "embedding_batch_retry", "error": str(batch_exc)},
        )
        evidence = []
        valid_entries = []
        for entry in crop_entries:
            try:
                embedding = dependencies.embedder.embed([entry[4]])
                evidence.append(dependencies.uniform_model.predict(embedding)[0])
                valid_entries.append(entry)
            except Exception as exc:
                entry[0].warnings.append(
                    f"Uniform evaluation failed for person {entry[1].index}: {exc}"
                )
        crop_entries = valid_entries

    for entry, uniform in zip(crop_entries, evidence, strict=True):
        work, detection, padded, upper, upper_crop = entry
        assert work.loaded is not None
        color = calculate_color_score(upper_crop, config.uniform_colors)
        quality = crop_quality_score(padded, work.loaded.width, work.loaded.height)
        centrality = centrality_score(padded, work.loaded.width, work.loaded.height)
        score = target_person_score(
            uniform_probability=uniform.probability,
            color_score=color.score,
            detection_confidence=detection.confidence,
            crop_quality=quality,
            centrality=centrality,
            config=config,
        )
        work.people.append(
            PersonEvaluation(
                index=detection.index,
                bbox=detection.bbox,
                padded_bbox=padded,
                upper_body_bbox=upper,
                detection_confidence=detection.confidence,
                uniform_probability=uniform.probability,
                target_similarity=uniform.target_similarity,
                other_similarity=uniform.other_similarity,
                similarity_margin=uniform.similarity_margin,
                embedding_similarity=uniform.embedding_similarity,
                uniform_color_score=color.score,
                color_qualifying_percentage=color.qualifying_percentage,
                color_breakdown=color.breakdown,
                crop_quality_score=quality,
                centrality_score=centrality,
                target_person_score=score,
            )
        )


def _select_and_focus(work: WorkImage, config: AppConfig) -> None:
    if work.loaded is None or work.result.error or not work.people:
        return
    selected = max(
        work.people,
        key=lambda person: (person.target_person_score, person.detection_confidence),
    )
    if selected.target_person_score < config.uniform_classifier.minimum_target_score:
        return
    work.result.selected_person_index = selected.index
    work.result.selected_person_bbox = selected.bbox
    work.result.person_detection_confidence = selected.detection_confidence
    work.result.uniform_probability = selected.uniform_probability
    work.result.uniform_embedding_similarity = selected.embedding_similarity
    work.result.uniform_color_score = selected.uniform_color_score
    person_crop = crop_image(work.loaded.image, selected.padded_bbox)
    try:
        focus = calculate_focus_metrics(person_crop, config.focus, iso=work.loaded.iso)
        work.result.laplacian_focus_score = focus.laplacian
        work.result.tenengrad_focus_score = focus.tenengrad
        work.focus_details = focus.model_dump()
    except FocusEvaluationError as exc:
        work.warnings.append(str(exc))


def _write_audits(
    works: list[WorkImage],
    config: AppConfig,
    model: UniformModel | None,
) -> None:
    if not config.output.write_audit_json:
        return
    for work in works:
        path = Path(work.result.analysis_sidecar_path)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "result": work.result.model_dump(mode="json"),
                "people": [person.model_dump(mode="json") for person in work.people],
                "focus": work.focus_details,
                "warnings": work.warnings,
                "uniform_model": {
                    "mode": model.mode if model else None,
                    "fingerprint": model.fingerprint if model else None,
                },
            },
        )


def _write_collection_csv(results: Sequence[ImageResult], path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    controlled = config.metadata_mapping.decisions
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename",
                    "full_path",
                    "decision",
                    "rating",
                    "color_label",
                    "keywords",
                ],
            )
            writer.writeheader()
            for result in results:
                mapping = controlled.get(result.decision.value)
                writer.writerow(
                    {
                        "filename": result.filename,
                        "full_path": result.full_path,
                        "decision": result.decision.value,
                        "rating": mapping.rating if mapping and mapping.rating is not None else "",
                        "color_label": mapping.color_label if mapping else "",
                        "keywords": mapping.keyword if mapping and mapping.keyword else "",
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure_results(
    paths: list[Path], root: Path, error: str, audit_dir: Path
) -> tuple[list[ImageResult], list[WorkImage]]:
    results: list[ImageResult] = []
    works: list[WorkImage] = []
    for path in paths:
        result = _new_result(path, root)
        result.error = error
        result.reason = f"Image could not be processed: {error}"
        result.analysis_sidecar_path = str(
            (audit_dir / f"{safe_artifact_stem(path.relative_to(root))}.json").resolve()
        )
        results.append(result)
        works.append(
            WorkImage(path=path, relative_path=path.relative_to(root), result=result, timer=Timer())
        )
    return results, works


def analyze_directory(
    input_path: Path,
    output_path: Path,
    config: AppConfig,
    logger: logging.Logger,
    *,
    overwrite: bool = False,
    dependencies: PipelineDependencies | None = None,
) -> AnalysisSummary:
    root = input_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    paths = discover_images(
        root,
        config.runtime.allowed_extensions,
        recursive=config.runtime.recursive,
    )
    if not paths:
        raise ValueError(
            f"No supported images found in {root}. "
            f"Expected: {', '.join(config.runtime.allowed_extensions)}"
        )
    audit_dir = config.output.audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    global_error: str | None = None
    if dependencies is None:
        try:
            dependencies = build_dependencies(config, logger)
        except Exception as exc:
            global_error = f"Analysis initialization failed: {exc}"
            logger.exception(
                "Analysis initialization failed",
                extra={"event": "analysis_initialization_error"},
            )
    if global_error:
        results, works = _failure_results(paths, root, global_error, audit_dir)
        apply_decisions(results, config.decision_thresholds)
        assign_bursts(results, config.burst_grouping)
        _write_audits(works, config, None)
        write_results_csv(
            results,
            output,
            precision=config.output.csv_float_precision,
            overwrite=overwrite,
        )
        collection_path = output.with_name(f"{output.stem}_lightroom.csv")
        _write_collection_csv(results, collection_path, config)
        return AnalysisSummary(
            results=results,
            output_path=output,
            collection_csv_path=collection_path,
            device="unavailable",
            classifier_mode="unavailable",
        )
    assert dependencies is not None

    all_works: list[WorkImage] = []
    worker_count = config.runtime.workers or min(4, os.cpu_count() or 1)
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ]
    with Progress(*progress_columns) as progress:
        task = progress.add_task("Analyzing photos", total=len(paths))
        for start in range(0, len(paths), config.runtime.chunk_size):
            chunk_paths = paths[start : start + config.runtime.chunk_size]
            works = []
            for path in chunk_paths:
                relative = path.relative_to(root)
                result = _new_result(path, root)
                result.analysis_sidecar_path = str(
                    (audit_dir / f"{safe_artifact_stem(relative)}.json").resolve()
                )
                works.append(
                    WorkImage(
                        path=path,
                        relative_path=relative,
                        result=result,
                        timer=Timer(),
                    )
                )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(_safe_load, works))
            _detect_chunk(works, dependencies.detector, logger)
            _evaluate_people(works, dependencies, config, logger)
            for work in works:
                try:
                    _select_and_focus(work, config)
                except Exception as exc:
                    work.result.error = f"Image evaluation failed: {exc}"
                work.result.processing_time_ms = work.timer.elapsed_ms
                if work.loaded is not None:
                    work.loaded.image.close()
                progress.advance(task)
            all_works.extend(works)

    results = [work.result for work in all_works]
    normalize_focus_results(results, config.focus)
    apply_decisions(results, config.decision_thresholds)
    assign_bursts(results, config.burst_grouping)
    _write_audits(all_works, config, dependencies.uniform_model)
    write_results_csv(
        results,
        output,
        precision=config.output.csv_float_precision,
        overwrite=overwrite,
    )
    collection_path = output.with_name(f"{output.stem}_lightroom.csv")
    _write_collection_csv(results, collection_path, config)
    logger.info(
        "Analysis complete",
        extra={
            "event": "analysis_complete",
            "images": len(results),
            "errors": sum(bool(result.error) for result in results),
            "output": str(output),
            "device": dependencies.device,
            "classifier_mode": dependencies.uniform_model.mode,
        },
    )
    return AnalysisSummary(
        results=results,
        output_path=output,
        collection_csv_path=collection_path,
        device=dependencies.device,
        classifier_mode=dependencies.uniform_model.mode,
    )
