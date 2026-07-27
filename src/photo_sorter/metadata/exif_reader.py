from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def find_exiftool() -> Path | None:
    executable = shutil.which("exiftool") or shutil.which("exiftool.exe")
    return Path(executable).resolve() if executable else None


def exiftool_version(executable: Path | None = None) -> str | None:
    tool = executable or find_exiftool()
    if tool is None:
        return None
    try:
        completed = subprocess.run(
            [str(tool), "-ver"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def read_xmp_fields(path: Path, executable: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            "-j",
            "-XMP-dc:Subject",
            "-XMP-xmp:Rating",
            "-XMP-xmp:Label",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed = json.loads(completed.stdout)
    if not parsed:
        return {}
    item = parsed[0]
    subject = item.get("Subject", [])
    if isinstance(subject, str):
        subject = [subject]
    return {
        "keywords": [str(value) for value in subject],
        "rating": item.get("Rating"),
        "label": item.get("Label"),
    }
