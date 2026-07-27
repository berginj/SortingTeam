from __future__ import annotations

from photo_sorter.schemas.config_models import DecisionThresholdsConfig
from photo_sorter.schemas.results import Decision, ImageResult


def decide(result: ImageResult, thresholds: DecisionThresholdsConfig) -> tuple[Decision, str]:
    if result.error:
        return Decision.ERROR, f"Image could not be processed: {result.error}"
    if result.people_detected == 0:
        return Decision.NO_SUBJECT, "No person detected above confidence threshold."
    if result.selected_person_index is None or result.uniform_probability is None:
        noun = "person" if result.people_detected == 1 else "people"
        return (
            Decision.NO_SUBJECT,
            f"{result.people_detected} {noun} detected; none matched target uniform.",
        )

    probability = result.uniform_probability
    percentile = result.focus_percentile
    if (
        probability >= thresholds.review_uniform_probability
        and percentile is not None
        and percentile < thresholds.soft_focus_percentile
    ):
        return (
            Decision.SOFT,
            f"Target uniform likely, but subject focus is in bottom {round(percentile)}% of batch.",
        )
    if (
        probability >= thresholds.keep_uniform_probability
        and percentile is not None
        and percentile >= thresholds.keep_focus_percentile
    ):
        return (
            Decision.KEEP,
            f"Target uniform likely; focus percentile is {round(percentile)}.",
        )
    if probability >= thresholds.review_uniform_probability:
        if percentile is None:
            return Decision.REVIEW, "Target uniform likely; focus could not be ranked reliably."
        return (
            Decision.REVIEW,
            f"Target uniform possible; focus percentile is {round(percentile)}.",
        )
    if probability < thresholds.wrong_team_probability:
        return (
            Decision.WRONG_TEAM,
            f"Selected player is unlikely to wear the target uniform ({probability:.0%}).",
        )
    return (
        Decision.REVIEW,
        f"Uniform evidence is ambiguous ({probability:.0%}); human review recommended.",
    )


def apply_decisions(results: list[ImageResult], thresholds: DecisionThresholdsConfig) -> None:
    for result in results:
        result.decision, result.reason = decide(result, thresholds)
