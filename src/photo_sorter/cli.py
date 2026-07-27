from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from photo_sorter import __version__
from photo_sorter.calibration import CalibrationError, calibrate
from photo_sorter.config import ConfigError, dump_config, load_config
from photo_sorter.logging_config import configure_logging
from photo_sorter.metadata.xmp_writer import XmpError, apply_xmp
from photo_sorter.processing.inspect_image import inspect_image as inspect_one_image
from photo_sorter.processing.pipeline import (
    analyze_directory,
    build_dependencies,
)
from photo_sorter.schemas.config_models import AppConfig
from photo_sorter.utils.cache import clear_runtime_cache
from photo_sorter.utils.paths import ensure_writable_directory

app = typer.Typer(
    name="photo-sorter",
    help="Fully local sports-photo uniform and focus sorting.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
console = Console()
DEFAULT_CONFIG = Path("config/default.yaml")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"photo-sorter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Analyze, review, calibrate, and safely tag sports-photo previews."""


def _load_and_log(config_path: Path) -> tuple[AppConfig, logging.Logger]:
    config = load_config(config_path)
    logger = configure_logging(config.logging)
    return config, logger


def _fatal(message: str, code: int = 1) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)


@app.command()
def analyze(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing exported JPEG previews."),
    ] = Path("data/input"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination results CSV."),
    ] = Path("data/output/results.csv"),
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="YAML configuration.")
    ] = DEFAULT_CONFIG,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing results CSV.")
    ] = False,
    recursive: Annotated[
        bool | None,
        typer.Option("--recursive/--no-recursive", help="Override recursive discovery."),
    ] = None,
    workers: Annotated[int | None, typer.Option("--workers", min=0, max=64)] = None,
) -> None:
    """Analyze every supported preview in a directory."""
    try:
        config, logger = _load_and_log(config_path)
        if recursive is not None:
            config.runtime.recursive = recursive
        if workers is not None:
            config.runtime.workers = workers
        summary = analyze_directory(
            input_path,
            output,
            config,
            logger,
            overwrite=overwrite,
        )
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        _fatal(str(exc))
    table = Table(title="Analysis complete")
    table.add_column("Photos", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Device")
    table.add_column("Uniform mode")
    table.add_row(
        str(len(summary.results)),
        str(sum(bool(item.error) for item in summary.results)),
        summary.device,
        summary.classifier_mode,
    )
    console.print(table)
    console.print(f"Results: [cyan]{summary.output_path}[/cyan]")
    console.print(f"Lightroom CSV: [cyan]{summary.collection_csv_path}[/cyan]")
    if any(item.error for item in summary.results):
        raise typer.Exit(code=1)


@app.command("train-uniform")
def train_uniform(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force", help="Ignore cached classifier.")] = False,
) -> None:
    """Build reference embeddings and the local uniform classifier."""
    try:
        config, logger = _load_and_log(config_path)
        dependencies = build_dependencies(config, logger, force_uniform_training=force)
        model = dependencies.uniform_model
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        _fatal(str(exc))
    console.print(
        Panel.fit(
            json.dumps(model.metadata, indent=2, default=str),
            title=f"Uniform model: {model.mode}",
        )
    )


@app.command("validate-config")
def validate_config(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    show_resolved: Annotated[
        bool, typer.Option("--show-resolved", help="Print the validated effective YAML.")
    ] = False,
) -> None:
    """Validate configuration without loading model weights."""
    try:
        config = load_config(config_path)
        ensure_writable_directory(config.output.audit_dir)
        ensure_writable_directory(config.output.xmp_staging_dir)
    except (ConfigError, OSError, ValueError) as exc:
        _fatal(str(exc), code=2)
    warnings: list[str] = []
    if not config.runtime.target_references.exists():
        warnings.append(
            f"Target reference directory is missing: {config.runtime.target_references}"
        )
    if not config.person_detection.model_path.is_file():
        warnings.append(f"YOLO weight is not downloaded: {config.person_detection.model_path}")
    if not config.clip.checkpoint_path.is_file():
        warnings.append(f"OpenCLIP weight is not downloaded: {config.clip.checkpoint_path}")
    console.print("[bold green]Configuration is valid.[/bold green]")
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if show_resolved:
        console.print(dump_config(config))


@app.command("inspect-image")
def inspect_image(
    image: Annotated[Path, typer.Option("--image", "-i")],
    debug_output: Annotated[Path, typer.Option("--debug-output", "-o")] = Path(
        "data/output/debug/inspect"
    ),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
) -> None:
    """Write an annotated image, crops, and all intermediate scores."""
    try:
        config, logger = _load_and_log(config_path)
        dependencies = build_dependencies(config, logger)
        result = inspect_one_image(image, debug_output, config, dependencies, logger)
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        _fatal(str(exc))
    console.print(f"Decision: [bold]{result.decision.value}[/bold] — {result.reason}")
    console.print(f"Debug artifacts: [cyan]{debug_output.resolve()}[/cyan]")


@app.command("apply-xmp")
def apply_xmp_command(
    results: Annotated[Path, typer.Option("--results", "-r")],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview only (the default).")] = False,
    write: Annotated[
        bool, typer.Option("--write", help="Actually create or update XMP sidecars.")
    ] = False,
    originals_dir: Annotated[
        Path | None,
        typer.Option(
            "--originals-dir",
            help="RAW root. If omitted, sidecars are written to staging.",
        ),
    ] = None,
    staging_dir: Annotated[Path | None, typer.Option("--staging-dir")] = None,
    keywords: Annotated[bool | None, typer.Option("--keywords/--no-keywords")] = None,
    ratings: Annotated[bool | None, typer.Option("--ratings/--no-ratings")] = None,
    color_labels: Annotated[bool | None, typer.Option("--color-labels/--no-color-labels")] = None,
) -> None:
    """Preview or write Lightroom-compatible XMP sidecars."""
    if write and dry_run:
        _fatal("--write and --dry-run are mutually exclusive.", code=2)
    try:
        config = load_config(config_path)
        changes, report = apply_xmp(
            results,
            config,
            write=write,
            originals_dir=originals_dir,
            staging_dir=staging_dir,
            keywords_enabled=keywords,
            ratings_enabled=ratings,
            labels_enabled=color_labels,
        )
    except (ConfigError, XmpError, OSError, ValueError) as exc:
        _fatal(str(exc))
    counts: dict[str, int] = {}
    for change in changes:
        counts[change.status] = counts.get(change.status, 0) + 1
    mode = "WRITE" if write else "DRY RUN"
    console.print(f"[bold]{mode}[/bold] — {counts}")
    console.print(f"Change report: [cyan]{report.resolve()}[/cyan]")
    if counts.get("ERROR"):
        raise typer.Exit(code=1)


@app.command("calibrate")
def calibrate_command(
    reviewed_results: Annotated[Path, typer.Option("--reviewed-results", "-r")] = Path(
        "data/output/reviewed_results.csv"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "config/calibrated_thresholds.yaml"
    ),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
) -> None:
    """Compare predictions with human labels and recommend thresholds."""
    try:
        config = load_config(config_path)
        report = calibrate(reviewed_results, config, output)
    except (ConfigError, CalibrationError, OSError, ValueError) as exc:
        _fatal(str(exc))
    table = Table(title="Calibration")
    table.add_column("Reviewed")
    table.add_column("Uniform labels")
    table.add_column("Focus labels")
    table.add_column("Decision labels")
    table.add_row(
        str(report.reviewed_rows),
        str(report.uniform_labeled_rows),
        str(report.focus_labeled_rows),
        str(report.decision_labeled_rows),
    )
    console.print(table)
    console.print_json(data=report.metrics)
    console.print(f"Recommendations: [cyan]{report.output_path.resolve()}[/cyan]")
    if report.low_confidence:
        console.print("[yellow]Low confidence: collect at least 30 diverse reviews.[/yellow]")


@app.command("clear-cache")
def clear_cache(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    include_models: Annotated[
        bool, typer.Option("--include-models", help="Also remove downloaded weights.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Remove embeddings/classifiers; model weights are preserved by default."""
    if include_models and not yes:
        _fatal("--include-models requires --yes.", code=2)
    try:
        config = load_config(config_path)
        removed = clear_runtime_cache(config.runtime.cache_dir, include_models=include_models)
    except (ConfigError, OSError, ValueError) as exc:
        _fatal(str(exc))
    console.print(f"Removed {len(removed)} cache entries.")


if __name__ == "__main__":
    app()
