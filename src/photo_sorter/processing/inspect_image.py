from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw

from photo_sorter.processing.burst_grouping import assign_bursts
from photo_sorter.processing.decision_engine import apply_decisions
from photo_sorter.processing.focus_score import normalize_focus_results
from photo_sorter.processing.image_loader import load_image
from photo_sorter.processing.pipeline import (
    PipelineDependencies,
    WorkImage,
    _detect_chunk,
    _evaluate_people,
    _select_and_focus,
)
from photo_sorter.schemas.config_models import AppConfig
from photo_sorter.schemas.results import Decision, ImageResult
from photo_sorter.utils.cache import atomic_write_json
from photo_sorter.utils.timing import Timer


def inspect_image(
    image_path: Path,
    output_dir: Path,
    config: AppConfig,
    dependencies: PipelineDependencies,
    logger: object,
) -> ImageResult:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_image(image_path)
    result = ImageResult(
        filename=image_path.name,
        full_path=str(image_path),
        relative_path=image_path.name,
        image_width=loaded.width,
        image_height=loaded.height,
        capture_time=loaded.capture_time,
        camera_id=loaded.camera_id,
        file_mtime=loaded.file_mtime,
        decision=Decision.ERROR,
        analysis_sidecar_path=str((output_dir / "analysis.json").resolve()),
    )
    work = WorkImage(
        path=image_path,
        relative_path=Path(image_path.name),
        result=result,
        timer=Timer(),
        loaded=loaded,
    )
    _detect_chunk([work], dependencies.detector, logger)  # type: ignore[arg-type]
    _evaluate_people([work], dependencies, config, logger)  # type: ignore[arg-type]
    _select_and_focus(work, config)
    result.processing_time_ms = work.timer.elapsed_ms
    normalize_focus_results([result], config.focus)
    apply_decisions([result], config.decision_thresholds)
    assign_bursts([result], config.burst_grouping)

    annotated = loaded.image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(max(annotated.size) / 500))
    for person in work.people:
        selected = person.index == result.selected_person_index
        color = "#00FF00" if selected else "#FFD700"
        draw.rectangle(person.bbox, outline=color, width=line_width)
        draw.text(
            (person.bbox[0] + 3, person.bbox[1] + 3),
            f"#{person.index} u={person.uniform_probability:.2f} s={person.target_person_score:.2f}",
            fill=color,
        )
        loaded.image.crop(person.padded_bbox).save(
            output_dir / f"person_{person.index}.jpg", quality=92
        )
        loaded.image.crop(person.upper_body_bbox).save(
            output_dir / f"person_{person.index}_upper.jpg", quality=92
        )
    annotated.save(output_dir / "annotated.jpg", quality=92)
    atomic_write_json(
        output_dir / "analysis.json",
        {
            "schema_version": 1,
            "note": "Focus percentile is relative to a one-image batch for inspect-image.",
            "result": result.model_dump(mode="json"),
            "people": [person.model_dump(mode="json") for person in work.people],
            "focus": work.focus_details,
            "warnings": work.warnings,
            "uniform_model": {
                "mode": dependencies.uniform_model.mode,
                "fingerprint": dependencies.uniform_model.fingerprint,
            },
        },
    )
    loaded.image.close()
    return result
