from __future__ import annotations

import pytest

from photo_sorter.models.uniform_classifier import target_person_score
from photo_sorter.processing.decision_engine import decide
from photo_sorter.schemas.results import Decision, ImageResult


def _base_result(**updates):
    payload = {
        "filename": "a.jpg",
        "full_path": "a.jpg",
        "relative_path": "a.jpg",
        "people_detected": 1,
        "selected_person_index": 0,
        "uniform_probability": 0.9,
        "focus_percentile": 50.0,
    }
    payload.update(updates)
    return ImageResult(**payload)


def test_target_score_uses_configured_weights(app_config) -> None:
    score = target_person_score(
        uniform_probability=0.8,
        color_score=0.5,
        detection_confidence=0.9,
        crop_quality=0.6,
        centrality=0.1,
        config=app_config,
    )
    assert score == pytest.approx(0.735)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"error": "broken"}, Decision.ERROR),
        ({"people_detected": 0, "selected_person_index": None}, Decision.NO_SUBJECT),
        ({"selected_person_index": None, "uniform_probability": None}, Decision.NO_SUBJECT),
        ({"uniform_probability": 0.9, "focus_percentile": 5}, Decision.SOFT),
        ({"uniform_probability": 0.9, "focus_percentile": 35}, Decision.KEEP),
        ({"uniform_probability": 0.7, "focus_percentile": 50}, Decision.REVIEW),
        ({"uniform_probability": 0.7, "focus_percentile": None}, Decision.REVIEW),
        ({"uniform_probability": 0.2, "focus_percentile": 5}, Decision.WRONG_TEAM),
        ({"uniform_probability": 0.3, "focus_percentile": 80}, Decision.REVIEW),
    ],
)
def test_decision_rule_precedence(app_config, updates, expected) -> None:
    decision, reason = decide(_base_result(**updates), app_config.decision_thresholds)
    assert decision is expected
    assert reason
