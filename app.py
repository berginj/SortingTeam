from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from photo_sorter.config import load_config
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
from photo_sorter.review.storage import load_review_frame, save_review_override
from photo_sorter.schemas.results import Decision

st.set_page_config(page_title="Sports Photo Sorter", layout="wide")
st.title("Sports Photo Sorter — Local Ingest, Cull, and Review")
st.caption(
    "Local-only workflow: copy and verify a card/source first, analyze temporary previews, "
    "then copy selected originals for Lightroom. Nothing is deleted or moved automatically."
)


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
            use_container_width=True,
        )


def _run_transfer(action: Any, success: str) -> None:
    try:
        with st.spinner("Working locally; this may take a while for a card-sized batch..."):
            manifest = action()
    except (IngestError, OSError, ValueError) as exc:
        st.error(str(exc))
        return
    summary = manifest.summary()
    if summary.get("ERROR", 0):
        st.warning(f"{success} finished with errors. See the manifest for every file.")
    else:
        st.success(success)
    st.json(summary)


def _render_ingest() -> None:
    st.subheader("1. Copy card/source to a local archive")
    st.write("This is a verified copy, not a move. Keep the card untouched until you verify it.")
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
        "I understand this will copy files and never delete the source.", key="confirm_ingest"
    )
    if columns[1].button("Copy and verify to archive", disabled=not confirm, key="run_ingest"):
        _run_transfer(
            lambda: copy_to_archive(source, archive, manifest, write=True), "Archive copy complete."
        )
    _manifest_status(manifest)


def _render_previews() -> None:
    st.subheader("2. Prepare temporary local analysis previews")
    st.write(
        "JPEGs are resized locally. RAW files use embedded previews through ExifTool when available."
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
    if st.button("Create local JPEG previews", key="run_previews"):
        _run_transfer(
            lambda: prepare_previews(source, output, manifest, max_edge=int(max_edge), write=True),
            "Preview preparation complete.",
        )
    _manifest_status(manifest)


def _render_analysis() -> None:
    st.subheader("3. Analyze previews")
    source = Path(st.text_input("Preview folder", "data/input", key="analysis_input"))
    output = Path(
        st.text_input("Analysis results CSV", "data/output/results.csv", key="analysis_output")
    )
    config_path = Path(st.text_input("Configuration", "config/default.yaml", key="analysis_config"))
    overwrite = st.checkbox("Replace an existing results CSV", key="analysis_overwrite")
    if st.button("Run local analysis", key="run_analysis"):
        try:
            with st.spinner("Loading local models and analyzing previews..."):
                config = load_config(config_path)
                logger = configure_logging(config.logging)
                summary = analyze_directory(source, output, config, logger, overwrite=overwrite)
            st.success(
                f"Analysis complete: {len(summary.results)} photos, {sum(bool(row.error) for row in summary.results)} errors."
            )
            st.caption(f"Results: {summary.output_path.resolve()}")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))


def _render_stage() -> None:
    st.subheader("5. Copy selected originals into a Lightroom import folder")
    st.write("This creates verified copies. Your archive originals stay in place.")
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
    )
    if st.button("Copy selected originals for Lightroom", disabled=not confirm, key="run_stage"):
        _run_transfer(
            lambda: stage_selected_originals(
                results, archive, destination, manifest, decisions=set(choices), write=True
            ),
            "Original staging complete.",
        )
    _manifest_status(manifest)


def _render_review() -> None:
    st.subheader("4. Review analysis results")
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
    left.image(_annotated(image, people, selected_index), use_container_width=True)
    selected_person = next(
        (person for person in people if int(person["index"]) == selected_index), None
    )
    if selected_person:
        center.image(
            image.crop(tuple(selected_person["padded_bbox"])),
            caption="Selected person",
            use_container_width=True,
        )
        right.image(
            image.crop(tuple(selected_person["upper_body_bbox"])),
            caption="Uniform crop",
            use_container_width=True,
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
        pd.DataFrame([{"field": column, "value": row.get(column, "")} for column in score_columns]),
        hide_index=True,
        use_container_width=True,
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
        if column.button(label, use_container_width=True, key=f"decision_{value}"):
            chosen = value
    components.html(
        """<script>const map={k:'K · Keep',r:'R · Review',w:'W · Wrong Team',s:'S · Soft',n:'N · No Subject',ArrowLeft:'← Previous',ArrowRight:'Next →'};window.parent.document.onkeydown=(event)=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;const wanted=map[event.key]||map[event.key.toLowerCase()];const button=wanted&&[...window.parent.document.querySelectorAll('button')].find((el)=>el.innerText.includes(wanted));if(button){event.preventDefault();button.click();}};</script>""",
        height=0,
    )
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


ingest_tab, previews_tab, analysis_tab, review_tab, stage_tab = st.tabs(
    ["1 · Ingest", "2 · Previews", "3 · Analyze", "4 · Review", "5 · Stage for Lightroom"]
)
with ingest_tab:
    _render_ingest()
with previews_tab:
    _render_previews()
with analysis_tab:
    _render_analysis()
with review_tab:
    _render_review()
with stage_tab:
    _render_stage()
