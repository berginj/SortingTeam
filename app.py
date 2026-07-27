from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from photo_sorter.config import ConfigError, load_config
from photo_sorter.logging_config import configure_logging
from photo_sorter.processing.ingest import (
    IngestError,
    copy_to_archive,
    discover_media,
    load_manifest,
    prepare_previews,
    stage_selected_originals,
)
from photo_sorter.processing.pipeline import analyze_directory
from photo_sorter.processing.uniform_discovery import (
    UniformDiscovery,
    discover_uniform_groups,
    promote_reference_images,
)
from photo_sorter.review.storage import load_review_frame, save_review_override
from photo_sorter.schemas.results import Decision

st.set_page_config(
    page_title="Sports Photo Sorter",
    page_icon=":material/filter_alt:",
    layout="wide",
)
st.title("Sports Photo Sorter — Local Ingest, Cull, and Review")
st.caption(
    "Local-only workflow: copy and verify a card/source first, analyze temporary previews, "
    "then copy selected originals for Lightroom. Nothing is deleted or moved automatically."
)

WORKFLOW_STATE_PATH = Path("data/output/workflow_state.json")
REFERENCE_EXTENSIONS = {".jpg", ".jpeg"}
WORKFLOW_DEFAULTS = {
    "target_reference_dir": "data/references/target",
    "other_reference_dir": "data/references/other",
    "ingest_source": "D:\\DCIM",
    "ingest_archive": "D:\\PhotoArchive\\Incoming",
    "preview_source": "D:\\PhotoArchive\\Incoming",
    "preview_output": "data/input",
    "discovery_source": "data/input",
    "analysis_input": "data/input",
    "analysis_output": "data/output/results.csv",
    "analysis_config": "config/default.yaml",
    "review_raw": "data/output/results.csv",
    "reviewed_output": "data/output/reviewed_results.csv",
    "stage_results": "data/output/reviewed_results.csv",
    "stage_archive": "D:\\PhotoArchive\\Incoming",
    "stage_destination": "D:\\PhotoArchive\\ToImport",
}


def _load_workflow_state() -> dict[str, str]:
    try:
        payload = json.loads(WORKFLOW_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        key: str(value)
        for key, value in payload.items()
        if key in WORKFLOW_DEFAULTS and isinstance(value, str)
    }


def _initialize_workflow_state() -> None:
    saved = _load_workflow_state()
    for key, fallback in WORKFLOW_DEFAULTS.items():
        st.session_state.setdefault(key, saved.get(key, fallback))


def _save_workflow_state() -> None:
    WORKFLOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: str(st.session_state.get(key, fallback)) for key, fallback in WORKFLOW_DEFAULTS.items()}
    temporary = WORKFLOW_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(WORKFLOW_STATE_PATH)


def _update_workflow(**values: Path | str) -> None:
    for key, value in values.items():
        st.session_state[key] = str(value)
    _save_workflow_state()


def _reference_files(path: Path, extensions: list[str] | set[str]) -> list[Path]:
    if not path.is_dir():
        return []
    return discover_media(path, extensions)


_initialize_workflow_state()


def _session_path(key: str, fallback: str) -> Path:
    return Path(str(st.session_state.get(key, fallback)))


def _path_badge(path: Path, present_label: str, missing_label: str) -> None:
    if path.exists():
        st.badge(present_label, icon=":material/check_circle:", color="green")
    else:
        st.badge(missing_label, icon=":material/pending:", color="orange")


def _render_overview() -> None:
    st.header("Start here")
    with st.container(border=True):
        st.subheader("What you get at the end")
        st.markdown(
            "A **Lightroom import folder** containing verified copies of only the photos "
            "you choose to keep or review. Your full archive remains untouched."
        )
        st.info(
            "This app is a culling assistant, not Lightroom itself. It first makes a safe "
            "archive copy, then creates disposable previews for analysis, then stages your "
            "best original files for import.",
            icon=":material/info:",
        )

    archive = _session_path("ingest_archive", "D:\\PhotoArchive\\Incoming")
    previews = _session_path("preview_output", "data/input")
    results = _session_path("analysis_output", "data/output/results.csv")
    reviewed = _session_path("reviewed_output", "data/output/reviewed_results.csv")
    import_folder = _session_path("stage_destination", "D:\\PhotoArchive\\ToImport")
    target_references = _session_path("target_reference_dir", "data/references/target")
    cards = st.columns(5)
    with cards[0].container(border=True):
        st.markdown(":material/groups: **0. Set the target team**")
        _path_badge(
            target_references,
            "Reference folder selected",
            "Target references required",
        )
        st.caption(str(target_references))
    with cards[1].container(border=True):
        st.markdown(":material/inventory_2: **1. Archive every original**")
        _path_badge(archive, "Archive available", "Archive not created")
        st.caption(str(archive))
    with cards[2].container(border=True):
        st.markdown(":material/photo_library: **2. Analyze previews**")
        _path_badge(previews, "Preview folder available", "Previews not created")
        st.caption("Temporary JPEGs; safe to regenerate.")
    with cards[3].container(border=True):
        st.markdown(":material/rule: **3. Confirm choices**")
        _path_badge(
            reviewed if reviewed.is_file() else results, "Results available", "Analysis not run"
        )
        st.caption("Review adjusts the AI recommendations.")
    with cards[4].container(border=True):
        st.markdown(":material/folder_open: **4. Import selected originals**")
        _path_badge(import_folder, "Import folder available", "Nothing staged yet")
        st.caption(str(import_folder))

    st.subheader("The workflow")
    st.markdown(
        "0. **Team setup:** Add target-uniform reference photos and save the team context.\n"
        "1. **Ingest:** Copy the whole card to a local archive and verify every byte.\n"
        "2. **Previews:** Create smaller JPEGs used only for AI analysis.\n"
        "2a. **Discover uniforms:** Compare recurring colour/pattern groups and choose your team.\n"
        "3. **Analyze:** Rank the previews as Keep, Review, Wrong Team, Soft, or No Subject.\n"
        "4. **Review:** Correct any decisions you disagree with.\n"
        "5. **Stage for Lightroom:** Copy `KEEP` and `REVIEW` originals into one clean import folder."
    )
    st.success(
        "Start with **0 · Team setup**, then **1 · Ingest**. Do not format the card until the ingest manifest shows "
        "each file as `COPIED` or `VERIFIED_EXISTING`.",
        icon=":material/check_circle:",
    )


def _render_team_setup() -> None:
    st.subheader("0. Set the target team before analysis")
    st.write(
        "Add clear, close photos of the uniform you want to keep. This context is saved locally "
        "and carried into analysis, review, and staging."
    )
    st.caption(
        "Use JPEGs with the target player prominent. One target image enables analysis; 5 target "
        "and 5 other-team images improve the classifier; 15 per group is recommended."
    )
    with st.form("team_setup", border=True):
        target = Path(st.text_input("Target-uniform reference folder", key="target_reference_dir"))
        other = Path(
            st.text_input(
                "Optional other-team reference folder",
                key="other_reference_dir",
            )
        )
        saved = st.form_submit_button(
            "Save team setup",
            type="primary",
            icon=":material/save:",
        )
    target_files = _reference_files(target, REFERENCE_EXTENSIONS)
    other_files = _reference_files(other, REFERENCE_EXTENSIONS)
    metrics = st.columns(2)
    metrics[0].metric("Target reference photos", len(target_files))
    metrics[1].metric("Other-team reference photos", len(other_files))
    if not target_files:
        st.error(
            "Analysis is blocked until this folder contains at least one .jpg or .jpeg target-uniform reference photo.",
            icon=":material/block:",
        )
    elif len(target_files) < 5:
        st.warning("Target-only matching will run, but add at least 5 other-team references for stronger separation.")
    elif len(other_files) < 5:
        st.warning("Add at least 5 other-team references to enable the stronger two-class classifier.")
    else:
        st.success("Team context is ready for analysis.", icon=":material/check_circle:")
    if saved:
        _save_workflow_state()
        st.success(f"Saved local workflow context to {WORKFLOW_STATE_PATH}.")


def _optional_bool(label: str, key: str) -> bool | None:
    value = st.selectbox(label, ["Unreviewed", "Yes", "No"], key=key)
    return {"Yes": True, "No": False}.get(value)


def _load_audit(path_value: Any) -> dict[str, Any]:
    if not path_value or str(path_value).lower() == "nan":
        return {}
    try:
        return json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _annotated(
    image: Image.Image, people: list[dict[str, Any]], selected: int | None
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    width = max(2, round(max(rendered.size) / 500))
    for person in people:
        index = int(person["index"])
        bbox = tuple(person["bbox"])
        color = "#32CD32" if index == selected else "#FFD700"
        draw.rectangle(bbox, outline=color, width=width)
        draw.text((bbox[0] + 3, bbox[1] + 3), f"#{index}", fill=color)
    return rendered


def _manifest_status(path: Path) -> None:
    if not path.is_file():
        st.caption("No manifest has been created yet.")
        return
    try:
        manifest = load_manifest(path)
    except IngestError as exc:
        st.warning(str(exc))
        return
    st.success(f"Tracked manifest: {path.resolve()}")
    st.json(
        {"operation": manifest.operation, **manifest.summary(), "updated_at": manifest.updated_at}
    )
    if manifest.records:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "file": record.relative_path,
                        "status": record.status,
                        "message": record.message,
                    }
                    for record in manifest.records[-20:]
                ]
            ),
            hide_index=True,
        )


def _run_transfer(action: Any, success: str) -> bool:
    try:
        with st.spinner("Working locally; this may take a while for a card-sized batch..."):
            manifest = action()
    except (IngestError, OSError, ValueError) as exc:
        st.error(str(exc))
        return False
    summary = manifest.summary()
    if summary.get("ERROR", 0):
        st.warning(f"{success} finished with errors. See the manifest for every file.")
    else:
        st.success(success)
    st.json(summary)
    return not bool(summary.get("ERROR", 0))


def _render_ingest() -> None:
    st.subheader("1. Copy card/source to a local archive")
    st.write("This is a verified copy, not a move. Keep the card untouched until you verify it.")
    st.caption("Outcome: a complete local archive plus an ingest manifest proving what copied.")
    source = Path(st.text_input("Card or source folder", "D:\\DCIM", key="ingest_source"))
    archive = Path(
        st.text_input("Local archive folder", "D:\\PhotoArchive\\Incoming", key="ingest_archive")
    )
    manifest = Path(
        st.text_input(
            "Ingest manifest",
            str(archive / ".photo_sorter" / "ingest_manifest.json"),
            key="ingest_manifest",
        )
    )
    columns = st.columns(2)
    if columns[0].button("Scan source", key="scan_source"):
        try:
            files = discover_media(source)
            st.info(
                f"Found {len(files):,} supported photo files totaling {sum(item.stat().st_size for item in files) / 1_000_000_000:.2f} GB."
            )
        except IngestError as exc:
            st.error(str(exc))
    confirm = st.checkbox(
        "I understand this will copy files and never delete the source.",
        key="confirm_ingest",
        width="stretch",
    )
    if confirm:
        st.success("Confirmation received. You can now start the verified archive copy.")
    else:
        st.caption("Tick the confirmation box before starting the copy.")
    if columns[1].button(
        "Copy and verify to archive",
        key="run_ingest",
        type="primary",
        icon=":material/content_copy:",
    ):
        if not confirm:
            st.warning("Confirm that this is a copy-only operation before starting.")
        else:
            completed = _run_transfer(
                lambda: copy_to_archive(source, archive, manifest, write=True),
                "Archive copy complete.",
            )
            if completed:
                _update_workflow(
                    preview_source=archive,
                    stage_archive=archive,
                )
                st.info("The archive folder has been carried forward to Previews and Staging.")
    _manifest_status(manifest)


def _render_previews() -> None:
    st.subheader("2. Prepare temporary local analysis previews")
    st.write(
        "JPEGs are resized locally. RAW files use embedded previews through ExifTool when available."
    )
    st.caption(
        "Outcome: small, disposable JPEGs for analysis. Your archive originals are not changed."
    )
    source = Path(
        st.text_input("Archive source folder", "D:\\PhotoArchive\\Incoming", key="preview_source")
    )
    output = Path(st.text_input("Temporary preview folder", "data/input", key="preview_output"))
    max_edge = st.number_input(
        "Preview long edge", min_value=256, max_value=10000, value=2560, step=128
    )
    manifest = Path(
        st.text_input(
            "Preview manifest",
            str(output / ".photo_sorter" / "preview_manifest.json"),
            key="preview_manifest",
        )
    )
    if st.button(
        "Create local JPEG previews",
        key="run_previews",
        type="primary",
        icon=":material/photo_library:",
    ):
        completed = _run_transfer(
            lambda: prepare_previews(source, output, manifest, max_edge=int(max_edge), write=True),
            "Preview preparation complete.",
        )
        if completed:
            _update_workflow(analysis_input=output, discovery_source=output)
            st.info("The preview folder has been carried forward to Uniform discovery and Analyze.")
    _manifest_status(manifest)


def _render_uniform_discovery() -> None:
    st.subheader("2a. Discover likely uniform groups")
    st.write(
        "This compares recurring central colour and pattern signatures in your JPEG previews. "
        "It does not identify a team by name; you choose the group that represents the team you care about."
    )
    st.caption(
        "This is a fast local clustering pass. It does not load YOLO or CLIP, and it copies only "
        "representative previews when you promote a group to target references."
    )
    source = Path(st.text_input("Preview folder to inspect", key="discovery_source"))
    controls = st.columns(2)
    group_count = int(
        controls[0].number_input(
            "Suggested groups",
            min_value=2,
            max_value=12,
            value=4,
            key="discovery_group_count",
        )
    )
    max_images = int(
        controls[1].number_input(
            "Maximum previews to sample",
            min_value=20,
            max_value=2000,
            value=400,
            step=20,
            key="discovery_max_images",
        )
    )
    if st.button(
        "Find uniform groups",
        type="primary",
        icon=":material/auto_awesome:",
        key="run_uniform_discovery",
    ):
        try:
            with st.spinner("Comparing local preview colour and pattern signatures..."):
                st.session_state["uniform_discovery"] = discover_uniform_groups(
                    source,
                    group_count=group_count,
                    max_images=max_images,
                    samples_per_group=12,
                )
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    discovery = st.session_state.get("uniform_discovery")
    if not isinstance(discovery, UniformDiscovery):
        st.info("Create previews, then run this step to see candidate uniform groups.")
        return
    st.success(
        f"Compared {discovery.scanned_files} previews and found {len(discovery.groups)} candidate groups."
    )
    for group in discovery.groups:
        with st.container(border=True):
            red, green, blue = group.mean_rgb
            st.markdown(
                f"#### Group {group.index} · {len(group.files)} similar previews "
                f"(average colour RGB {red}, {green}, {blue})"
            )
            images = st.columns(min(4, len(group.samples)))
            for column, sample in zip(images, group.samples[: len(images)], strict=False):
                column.image(str(sample), caption=sample.name, width="stretch")
    group_options = {group.index: group for group in discovery.groups}
    selected_index = st.selectbox(
        "Which group is the target team?",
        list(group_options),
        format_func=lambda index: f"Group {index} ({len(group_options[index].files)} previews)",
        key="discovery_selected_group",
    )
    target_references = _session_path("target_reference_dir", "data/references/target")
    st.caption(f"Selected representative previews will be copied to: {target_references}")
    if st.button(
        "Use this group as target references",
        type="primary",
        icon=":material/groups:",
        key="promote_uniform_group",
    ):
        try:
            copied = promote_reference_images(group_options[selected_index].samples, target_references)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            _save_workflow_state()
            st.success(f"Copied {len(copied)} representative previews into the target reference folder.")
            st.rerun()


def _render_analysis() -> None:
    st.subheader("3. Analyze previews")
    st.caption("Outcome: results.csv ranks every preview and identifies the likely target player.")
    source = Path(st.text_input("Preview folder", "data/input", key="analysis_input"))
    output = Path(
        st.text_input("Analysis results CSV", "data/output/results.csv", key="analysis_output")
    )
    config_path = Path(st.text_input("Configuration", "config/default.yaml", key="analysis_config"))
    target_references = _session_path("target_reference_dir", "data/references/target")
    other_references = _session_path("other_reference_dir", "data/references/other")
    st.text(f"Target references: {target_references}")
    st.text(f"Other-team references: {other_references}")
    target_count = len(_reference_files(target_references, REFERENCE_EXTENSIONS))
    other_count = len(_reference_files(other_references, REFERENCE_EXTENSIONS))
    st.caption(f"Team setup: {target_count} target references · {other_count} other-team references")
    if not target_count:
        st.error(
            "Blocked: add at least one target reference image in 0 · Team setup before running analysis. "
            "No model will be loaded until this is resolved.",
            icon=":material/block:",
        )
    overwrite = st.checkbox("Replace an existing results CSV", key="analysis_overwrite")
    if st.button(
        "Run local analysis",
        key="run_analysis",
        type="primary",
        icon=":material/play_arrow:",
    ):
        try:
            if not target_count:
                return
            config = load_config(config_path)
            config.runtime.target_references = target_references.expanduser().resolve()
            config.runtime.other_references = other_references.expanduser().resolve()
            target_count = len(_reference_files(config.runtime.target_references, config.runtime.allowed_extensions))
            if not target_count:
                st.error(
                    "Blocked: the configured target reference folder contains no supported reference images.",
                    icon=":material/block:",
                )
                return
            with st.spinner("Loading local models and analyzing previews..."):
                logger = configure_logging(config.logging)
                summary = analyze_directory(source, output, config, logger, overwrite=overwrite)
            st.success(
                f"Analysis complete: {len(summary.results)} photos, {sum(bool(row.error) for row in summary.results)} errors."
            )
            st.caption(f"Results: {summary.output_path.resolve()}")
            _update_workflow(review_raw=output, stage_results=output)
            st.info("The results file has been carried forward to Review and Staging.")
        except (ConfigError, OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))


def _render_stage() -> None:
    st.subheader("5. Copy selected originals into a Lightroom import folder")
    st.write("This creates verified copies. Your archive originals stay in place.")
    st.caption(
        "Outcome: this folder is the one you import into Lightroom. By default it contains "
        "both KEEP and REVIEW originals."
    )
    results = Path(
        st.text_input(
            "Results or reviewed-results CSV",
            "data/output/reviewed_results.csv",
            key="stage_results",
        )
    )
    archive = Path(
        st.text_input("Original archive folder", "D:\\PhotoArchive\\Incoming", key="stage_archive")
    )
    destination = Path(
        st.text_input(
            "Lightroom import folder", "D:\\PhotoArchive\\ToImport", key="stage_destination"
        )
    )
    choices = st.multiselect(
        "Decisions to stage",
        [item.value for item in Decision],
        default=["KEEP", "REVIEW"],
        key="stage_decisions",
    )
    manifest = Path(
        st.text_input(
            "Staging manifest",
            str(destination / ".photo_sorter" / "stage_manifest.json"),
            key="stage_manifest",
        )
    )
    confirm = st.checkbox(
        "I understand this copies selected originals; it never deletes archive files.",
        key="confirm_stage",
        width="stretch",
    )
    if confirm:
        st.success("Confirmation received. You can now create the Lightroom import copy.")
    else:
        st.caption("Tick the confirmation box before staging originals.")
    if st.button(
        "Copy selected originals for Lightroom",
        key="run_stage",
        type="primary",
        icon=":material/folder_open:",
    ):
        if not confirm:
            st.warning("Confirm that this is a copy-only operation before staging originals.")
        else:
            _run_transfer(
                lambda: stage_selected_originals(
                    results, archive, destination, manifest, decisions=set(choices), write=True
                ),
                "Original staging complete.",
            )
    _manifest_status(manifest)


def _render_review() -> None:
    st.subheader("4. Review analysis results")
    st.caption(
        "Outcome: reviewed_results.csv replaces only the recommendations you change; "
        "the original results.csv remains unchanged."
    )
    raw_path = Path(st.text_input("Raw results CSV", "data/output/results.csv", key="review_raw"))
    reviewed_path = Path(
        st.text_input(
            "Reviewed output CSV", "data/output/reviewed_results.csv", key="reviewed_output"
        )
    )
    try:
        frame = load_review_frame(raw_path, reviewed_path)
    except (OSError, ValueError) as exc:
        st.info(f"Run analysis first, then review here. {exc}")
        return
    if not frame.empty and frame["decision"].eq("ERROR").all():
        errors = frame.get("error", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()
        st.error(
            "This results file contains only processing errors, not reviewable recommendations. "
            "Return to 0 · Team setup, add target references, then rerun analysis with “Replace an existing results CSV” selected.",
            icon=":material/error:",
        )
        if errors:
            st.code("\n".join(errors[:3]), language=None)
        return
    decisions = sorted(frame["decision"].dropna().astype(str).unique())
    decision_filter = st.multiselect("Decision", decisions, default=decisions, key="review_filter")
    sort_column = st.selectbox(
        "Sort by",
        [
            column
            for column in ("uniform_probability", "focus_percentile", "burst_rank", "filename")
            if column in frame.columns
        ],
        key="review_sort",
    )
    ascending = st.checkbox("Ascending", value=False, key="review_ascending")
    filtered = frame[frame["decision"].astype(str).isin(decision_filter)].sort_values(
        sort_column, ascending=ascending, na_position="last"
    )
    if filtered.empty:
        st.info("No photos match the current filter.")
        return
    if "position" not in st.session_state:
        st.session_state.position = 0
    st.session_state.position = min(st.session_state.position, len(filtered) - 1)
    navigation = st.columns([1, 1, 5])
    if navigation[0].button("← Previous", key="review_previous"):
        st.session_state.position = max(0, st.session_state.position - 1)
    if navigation[1].button("Next →", key="review_next"):
        st.session_state.position = min(len(filtered) - 1, st.session_state.position + 1)
    navigation[2].write(f"{st.session_state.position + 1} of {len(filtered)}")
    row_index = int(filtered.index[st.session_state.position])
    row = frame.loc[row_index]
    try:
        with Image.open(Path(str(row["full_path"]))) as source:
            image = source.convert("RGB").copy()
    except OSError as exc:
        st.error(f"Could not open preview: {exc}")
        return
    audit = _load_audit(row.get("analysis_sidecar_path"))
    people = list(audit.get("people", []))
    selected_raw = row.get("override_target_person_index")
    try:
        selected_index = (
            int(float(selected_raw))
            if pd.notna(selected_raw) and str(selected_raw).strip()
            else int(float(row["selected_person_index"]))
            if pd.notna(row.get("selected_person_index"))
            else None
        )
    except (TypeError, ValueError):
        selected_index = None
    st.markdown(f"### {row['filename']} · {row['decision']}")
    left, center, right = st.columns([2, 1, 1])
    left.image(_annotated(image, people, selected_index), width="stretch")
    selected_person = next(
        (person for person in people if int(person["index"]) == selected_index), None
    )
    if selected_person:
        center.image(
            image.crop(tuple(selected_person["padded_bbox"])),
            caption="Selected person",
            width="stretch",
        )
        right.image(
            image.crop(tuple(selected_person["upper_body_bbox"])),
            caption="Uniform crop",
            width="stretch",
        )
    score_columns = [
        "people_detected",
        "person_detection_confidence",
        "uniform_probability",
        "uniform_color_score",
        "normalized_focus_score",
        "focus_percentile",
        "burst_rank",
        "reason",
        "error",
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {"field": column, "value": "" if pd.isna(row.get(column)) else str(row.get(column))}
                for column in score_columns
            ]
        ),
        hide_index=True,
    )
    person_options: list[int | None] = [None] + [int(person["index"]) for person in people]
    correct_person = st.selectbox(
        "Correct target person",
        person_options,
        index=person_options.index(selected_index) if selected_index in person_options else 0,
        format_func=lambda value: "No target person" if value is None else f"Person #{value}",
    )
    uniform_label = _optional_bool("Target uniform?", f"uniform_{row_index}")
    focus_label = _optional_bool("Focus acceptable?", f"focus_{row_index}")
    note = st.text_input("Review note", key=f"note_{row_index}")
    buttons = st.columns(5)
    chosen: str | None = None
    for column, (label, value) in zip(
        buttons,
        [
            ("K · Keep", "KEEP"),
            ("R · Review", "REVIEW"),
            ("W · Wrong Team", "WRONG_TEAM"),
            ("S · Soft", "SOFT"),
            ("N · No Subject", "NO_SUBJECT"),
        ],
        strict=True,
    ):
        if column.button(label, width="stretch", key=f"decision_{value}"):
            chosen = value
    if chosen:
        save_review_override(
            frame,
            row_index,
            reviewed_path,
            decision=chosen,
            target_person_index=correct_person,
            uniform=uniform_label,
            focus=focus_label,
            note=note,
        )
        st.success(f"Saved {chosen} to {reviewed_path}")
        if st.session_state.position < len(filtered) - 1:
            st.session_state.position += 1
        st.rerun()


overview_tab, setup_tab, ingest_tab, previews_tab, discovery_tab, analysis_tab, review_tab, stage_tab = st.tabs(
    [
        ":material/home: Start here",
        "0 · Team setup",
        "1 · Ingest",
        "2 · Previews",
        "2a · Discover uniforms",
        "3 · Analyze",
        "4 · Review",
        "5 · Stage for Lightroom",
    ]
)
with overview_tab:
    _render_overview()
with setup_tab:
    _render_team_setup()
with ingest_tab:
    _render_ingest()
with previews_tab:
    _render_previews()
with discovery_tab:
    _render_uniform_discovery()
with analysis_tab:
    _render_analysis()
with review_tab:
    _render_review()
with stage_tab:
    _render_stage()
