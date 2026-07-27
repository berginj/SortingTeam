from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REVIEW_COLUMNS = [
    "override_decision",
    "override_target_person_index",
    "override_uniform",
    "override_focus",
    "review_note",
    "reviewed_at",
    "effective_decision",
]


def load_review_frame(raw_results: Path, reviewed_results: Path) -> pd.DataFrame:
    source = reviewed_results if reviewed_results.is_file() else raw_results
    if not source.is_file():
        raise FileNotFoundError(f"Results CSV not found: {source}")
    frame = pd.read_csv(source)
    for column in REVIEW_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def save_review_override(
    frame: pd.DataFrame,
    row_index: int,
    output: Path,
    *,
    decision: str | None,
    target_person_index: int | None,
    uniform: bool | None,
    focus: bool | None,
    note: str = "",
) -> pd.DataFrame:
    if row_index not in frame.index:
        raise IndexError(f"Review row index does not exist: {row_index}")
    updated = frame.copy()
    updated.at[row_index, "override_decision"] = decision or ""
    updated.at[row_index, "override_target_person_index"] = (
        target_person_index if target_person_index is not None else ""
    )
    updated.at[row_index, "override_uniform"] = str(uniform).lower() if uniform is not None else ""
    updated.at[row_index, "override_focus"] = str(focus).lower() if focus is not None else ""
    updated.at[row_index, "review_note"] = note
    updated.at[row_index, "reviewed_at"] = datetime.now(UTC).isoformat()
    raw_decision = str(updated.at[row_index, "decision"])
    updated.at[row_index, "effective_decision"] = decision or raw_decision
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        updated.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return updated
