from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from photo_sorter.review.storage import load_review_frame, save_review_override
from photo_sorter.schemas.results import Decision

st.set_page_config(page_title="Sports Photo Sorter Review", layout="wide")
st.title("Sports Photo Sorter — Local Review")
st.caption("All files stay on this computer. Overrides never replace the raw analysis CSV.")


def _optional_bool(label: str, key: str) -> bool | None:
    value = st.selectbox(label, ["Unreviewed", "Yes", "No"], key=key)
    return {"Yes": True, "No": False}.get(value)


def _load_audit(path_value: Any) -> dict[str, Any]:
    if not path_value or str(path_value).lower() == "nan":
        return {}
    path = Path(str(path_value))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


raw_value = st.sidebar.text_input("Raw results CSV", "data/output/results.csv")
reviewed_value = st.sidebar.text_input("Reviewed output CSV", "data/output/reviewed_results.csv")
raw_path = Path(raw_value)
reviewed_path = Path(reviewed_value)

try:
    frame = load_review_frame(raw_path, reviewed_path)
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

decisions = sorted(frame["decision"].dropna().astype(str).unique())
decision_filter = st.sidebar.multiselect("Decision", decisions, default=decisions)
sort_column = st.sidebar.selectbox(
    "Sort by",
    [
        column
        for column in ("uniform_probability", "focus_percentile", "burst_rank", "filename")
        if column in frame.columns
    ],
)
ascending = st.sidebar.checkbox("Ascending", value=False)
filtered = frame[frame["decision"].astype(str).isin(decision_filter)].sort_values(
    sort_column, ascending=ascending, na_position="last"
)
if filtered.empty:
    st.info("No photos match the current filter.")
    st.stop()

if "position" not in st.session_state:
    st.session_state.position = 0
st.session_state.position = min(st.session_state.position, len(filtered) - 1)
navigation = st.columns([1, 1, 5])
if navigation[0].button("← Previous"):
    st.session_state.position = max(0, st.session_state.position - 1)
if navigation[1].button("Next →"):
    st.session_state.position = min(len(filtered) - 1, st.session_state.position + 1)
navigation[2].write(f"{st.session_state.position + 1} of {len(filtered)}")

row_index = int(filtered.index[st.session_state.position])
row = frame.loc[row_index]
image_path = Path(str(row["full_path"]))
try:
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()
except OSError as exc:
    st.error(f"Could not open preview {image_path}: {exc}")
    st.stop()

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

st.subheader(f"{row['filename']} · {row['decision']}")
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
else:
    center.info("No target person selected.")

score_columns = [
    "people_detected",
    "person_detection_confidence",
    "uniform_probability",
    "uniform_embedding_similarity",
    "uniform_color_score",
    "laplacian_focus_score",
    "tenengrad_focus_score",
    "normalized_focus_score",
    "focus_percentile",
    "burst_id",
    "burst_rank",
    "burst_size",
    "reason",
    "error",
]
st.dataframe(
    pd.DataFrame([{"field": column, "value": row.get(column, "")} for column in score_columns]),
    hide_index=True,
    use_container_width=True,
)
if people:
    st.expander("All detected people").json(people)

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

st.write("Decision override")
buttons = st.columns(5)
button_map = [
    ("K · Keep", Decision.KEEP.value),
    ("R · Review", Decision.REVIEW.value),
    ("W · Wrong Team", Decision.WRONG_TEAM.value),
    ("S · Soft", Decision.SOFT.value),
    ("N · No Subject", Decision.NO_SUBJECT.value),
]
chosen: str | None = None
for column, (label, value) in zip(buttons, button_map, strict=True):
    if column.button(label, use_container_width=True):
        chosen = value

components.html(
    """
    <script>
      const map = {k:'K · Keep', r:'R · Review', w:'W · Wrong Team',
                   s:'S · Soft', n:'N · No Subject',
                   ArrowLeft:'← Previous', ArrowRight:'Next →'};
      window.parent.document.onkeydown = (event) => {
        if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
        const wanted = map[event.key] || map[event.key.toLowerCase()];
        if (!wanted) return;
        const button = [...window.parent.document.querySelectorAll('button')]
          .find(el => el.innerText.includes(wanted));
        if (button) { event.preventDefault(); button.click(); }
      };
    </script>
    """,
    height=0,
)

if chosen:
    frame = save_review_override(
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
