from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from photo_sorter.schemas.config_models import AppConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class CalibrationError(ValueError):
    """Raised when reviewed results cannot support calibration."""


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    reviewed_rows: int
    uniform_labeled_rows: int
    focus_labeled_rows: int
    decision_labeled_rows: int
    metrics: dict[str, Any]
    recommendations: dict[str, float]
    low_confidence: bool
    output_path: Path


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _best_binary_threshold(
    probabilities: FloatArray,
    labels: BoolArray,
    current: float,
) -> float:
    candidates = np.arange(0.05, 0.951, 0.05)
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        score = f1_score(labels, predicted, zero_division=0)
        scored.append((float(score), -abs(float(threshold) - current), float(threshold)))
    return max(scored)[2]


def _high_precision_threshold(
    probabilities: FloatArray,
    labels: BoolArray,
    minimum: float,
    current: float,
) -> float:
    candidates: list[float] = []
    for threshold in np.arange(max(0.05, minimum), 0.951, 0.05):
        predicted = probabilities >= threshold
        if int(np.count_nonzero(predicted)) < 3:
            continue
        precision = precision_score(labels, predicted, zero_division=0)
        if precision >= 0.9:
            candidates.append(float(threshold))
    return min(candidates) if candidates else current


def _negative_precision_threshold(
    probabilities: FloatArray,
    labels: BoolArray,
    upper: float,
    current: float,
) -> float:
    candidates: list[float] = []
    for threshold in np.arange(0.05, upper, 0.05):
        predicted_negative = probabilities < threshold
        if int(np.count_nonzero(predicted_negative)) < 3:
            continue
        negative_precision = float(np.mean(labels[predicted_negative] == 0))
        if negative_precision >= 0.9:
            candidates.append(float(threshold))
    return max(candidates) if candidates else current


def _focus_threshold(
    percentiles: FloatArray,
    labels: BoolArray,
    current: float,
) -> float:
    scored: list[tuple[float, float, float]] = []
    for threshold in np.arange(5, 51, 5):
        predicted = percentiles >= threshold
        score = f1_score(labels, predicted, zero_division=0)
        scored.append((float(score), -abs(float(threshold) - current), float(threshold)))
    return max(scored)[2]


def calibrate(
    reviewed_results: Path,
    config: AppConfig,
    output_path: Path,
) -> CalibrationReport:
    if not reviewed_results.is_file():
        raise CalibrationError(f"Reviewed results do not exist: {reviewed_results}")
    frame = pd.read_csv(reviewed_results)
    required = {"uniform_probability", "focus_percentile", "decision"}
    missing = required - set(frame.columns)
    if missing:
        raise CalibrationError(
            f"Reviewed results are missing columns: {', '.join(sorted(missing))}"
        )
    if "override_uniform" not in frame.columns:
        raise CalibrationError(
            "No override_uniform column found. Save labels in the review app first."
        )

    uniform_labels = frame["override_uniform"].map(_optional_bool)
    uniform_mask = uniform_labels.notna() & frame["uniform_probability"].notna()
    uniform_frame = frame.loc[uniform_mask]
    uniform_truth = np.asarray(
        uniform_labels.loc[uniform_mask].astype(bool).to_numpy(), dtype=np.bool_
    )
    uniform_probability = np.asarray(
        uniform_frame["uniform_probability"].astype(float).to_numpy(),
        dtype=np.float64,
    )
    current = config.decision_thresholds
    metrics: dict[str, Any] = {}
    recommendations = {
        "keep_uniform_probability": current.keep_uniform_probability,
        "review_uniform_probability": current.review_uniform_probability,
        "wrong_team_probability": current.wrong_team_probability,
        "keep_focus_percentile": current.keep_focus_percentile,
        "soft_focus_percentile": current.soft_focus_percentile,
    }

    if uniform_truth.size > 0 and len(np.unique(uniform_truth)) == 2:
        predicted = uniform_probability >= current.review_uniform_probability
        matrix = confusion_matrix(uniform_truth, predicted, labels=[False, True])
        tn, fp, fn, tp = matrix.ravel()
        metrics["uniform"] = {
            "confusion_matrix": matrix.tolist(),
            "precision": float(precision_score(uniform_truth, predicted, zero_division=0)),
            "recall": float(recall_score(uniform_truth, predicted, zero_division=0)),
            "f1": float(f1_score(uniform_truth, predicted, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        }
        review_threshold = _best_binary_threshold(
            uniform_probability,
            uniform_truth,
            current.review_uniform_probability,
        )
        recommendations["review_uniform_probability"] = review_threshold
        recommendations["keep_uniform_probability"] = _high_precision_threshold(
            uniform_probability,
            uniform_truth,
            review_threshold,
            current.keep_uniform_probability,
        )
        recommendations["wrong_team_probability"] = min(
            _negative_precision_threshold(
                uniform_probability,
                uniform_truth,
                review_threshold,
                current.wrong_team_probability,
            ),
            review_threshold - 0.05,
        )
    else:
        metrics["uniform"] = {"warning": "Both yes and no uniform labels are required."}

    focus_labeled = (
        frame["override_focus"].map(_optional_bool)
        if "override_focus" in frame.columns
        else pd.Series([None] * len(frame))
    )
    focus_mask = focus_labeled.notna() & frame["focus_percentile"].notna()
    focus_truth = np.asarray(focus_labeled.loc[focus_mask].astype(bool).to_numpy(), dtype=np.bool_)
    focus_values = np.asarray(
        frame.loc[focus_mask, "focus_percentile"].astype(float).to_numpy(),
        dtype=np.float64,
    )
    if focus_truth.size > 0 and len(np.unique(focus_truth)) == 2:
        soft_threshold = _focus_threshold(focus_values, focus_truth, current.soft_focus_percentile)
        recommendations["soft_focus_percentile"] = soft_threshold
        recommendations["keep_focus_percentile"] = max(
            soft_threshold + 5,
            _focus_threshold(focus_values, focus_truth, current.keep_focus_percentile),
        )
        focus_predicted = focus_values >= soft_threshold
        metrics["focus"] = {
            "accuracy": float(accuracy_score(focus_truth, focus_predicted)),
            "f1": float(f1_score(focus_truth, focus_predicted, zero_division=0)),
        }
    else:
        metrics["focus"] = {
            "warning": "Both acceptable and unacceptable focus labels are required."
        }

    decision_mask = (
        frame["override_decision"].notna()
        & frame["override_decision"].astype(str).str.strip().ne("")
        if "override_decision" in frame.columns
        else pd.Series([False] * len(frame))
    )
    if int(decision_mask.sum()) > 0:
        truth = frame.loc[decision_mask, "override_decision"].astype(str).str.upper()
        decision_predicted = frame.loc[decision_mask, "decision"].astype(str).str.upper()
        categories = sorted(set(truth) | set(decision_predicted))
        metrics["decision"] = {
            "accuracy": float(accuracy_score(truth, decision_predicted)),
            "categories": categories,
            "confusion_matrix": confusion_matrix(
                truth, decision_predicted, labels=categories
            ).tolist(),
        }
    else:
        metrics["decision"] = {"warning": "No human decision overrides are available."}

    reviewed_mask = uniform_labels.notna() | focus_labeled.notna() | decision_mask
    reviewed_count = int(reviewed_mask.sum())
    low_confidence = reviewed_count < 30 or uniform_truth.size < 10
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(reviewed_results.resolve()),
        "low_confidence": low_confidence,
        "sample_counts": {
            "reviewed": reviewed_count,
            "uniform": int(uniform_mask.sum()),
            "focus": int(focus_mask.sum()),
            "decision": int(decision_mask.sum()),
        },
        "decision_thresholds": recommendations,
        "metrics": metrics,
        "note": (
            "Low-confidence recommendation; collect at least 30 diverse reviewed examples."
            if low_confidence
            else "Review these values before copying them into the active configuration."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return CalibrationReport(
        reviewed_rows=reviewed_count,
        uniform_labeled_rows=int(uniform_mask.sum()),
        focus_labeled_rows=int(focus_mask.sum()),
        decision_labeled_rows=int(decision_mask.sum()),
        metrics=metrics,
        recommendations=recommendations,
        low_confidence=low_confidence,
        output_path=output_path,
    )
