# Calibration

Calibration consumes `reviewed_results.csv`, which preserves raw predictions and adds
human overrides.

## Labels

- `override_uniform`: the selected/corrected person wears the target uniform.
- `override_focus`: the selected/corrected person is acceptably in focus.
- `override_decision`: the desired final category.

Unreviewed/blank values are excluded independently, so a row may contribute to one
metric without contributing to another.

## Metrics and recommendations

At the active Review uniform threshold, calibration reports a binary confusion
matrix, precision, recall, F1, false-positive rate, and false-negative rate.

Candidate probability thresholds are evaluated in 0.05 steps:

- Review threshold maximizes binary F1.
- Keep threshold chooses the lowest threshold at or above Review that attains at
  least 90% precision with three or more predicted positives.
- Wrong Team threshold chooses the largest threshold below Review that attains at
  least 90% precision for predicted negatives with three or more examples.

Focus candidates use 5-percentile steps and maximize agreement/F1 with human focus
labels while maintaining `soft < keep`. Human decision overrides also produce
category accuracy and a confusion matrix.

Fewer than 30 diverse reviewed rows, too few uniform labels, or only one class marks
the output low-confidence. In that case unsupported thresholds retain their active
values.

`config/calibrated_thresholds.yaml` is an advisory artifact. Review it and manually
copy chosen values into a team configuration; the command never modifies active
configuration.

