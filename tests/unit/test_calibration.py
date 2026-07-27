from __future__ import annotations

from pathlib import Path

import pandas as pd

from photo_sorter.calibration import calibrate


def test_calibration_writes_separate_recommendations(tmp_path: Path, app_config) -> None:
    rows = []
    for index in range(40):
        target = index >= 20
        focus_ok = index % 4 != 0
        rows.append(
            {
                "uniform_probability": 0.8 if target else 0.15,
                "focus_percentile": 60 if focus_ok else 5,
                "decision": "KEEP" if target and focus_ok else "REVIEW",
                "override_uniform": target,
                "override_focus": focus_ok,
                "override_decision": "KEEP" if target and focus_ok else "REVIEW",
            }
        )
    reviewed = tmp_path / "reviewed.csv"
    pd.DataFrame(rows).to_csv(reviewed, index=False)
    output = tmp_path / "calibrated.yaml"
    report = calibrate(reviewed, app_config, output)
    assert report.reviewed_rows == 40
    assert not report.low_confidence
    assert output.is_file()
