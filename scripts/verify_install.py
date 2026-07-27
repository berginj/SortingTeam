from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

from photo_sorter.config import ConfigError, load_config
from photo_sorter.metadata.exif_reader import exiftool_version, find_exiftool
from photo_sorter.utils.paths import ensure_writable_directory


def _status(label: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker:4}] {label}: {detail}")


def main() -> int:
    failures = 0
    version_ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    _status("Python", version_ok, sys.version.split()[0])
    failures += not version_ok

    modules = {
        "OpenCV": "cv2",
        "Torch": "torch",
        "OpenCLIP": "open_clip",
        "Ultralytics YOLO": "ultralytics",
        "Pillow": "PIL",
        "Pydantic": "pydantic",
        "Streamlit": "streamlit",
    }
    imported = {}
    for label, module_name in modules.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
            version = getattr(imported[module_name], "__version__", "imported")
            _status(label, True, str(version))
        except Exception as exc:
            _status(label, False, str(exc))
            failures += 1

    torch = imported.get("torch")
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend and mps_backend.is_available())
        detail = "CUDA" if cuda else "Apple MPS" if mps else "CPU"
        _status("Compute device", True, detail)

    try:
        config = load_config(Path("config/default.yaml"))
    except ConfigError as exc:
        _status("Configuration", False, str(exc))
        return 1
    for label, path in (
        ("YOLO cache", config.person_detection.model_path),
        ("OpenCLIP cache", config.clip.checkpoint_path),
    ):
        ok = path.is_file() and path.stat().st_size > 0
        _status(label, ok, str(path))
        failures += not ok
    try:
        ensure_writable_directory(Path("data/output").resolve())
        _status("Output directory", True, str(Path("data/output").resolve()))
    except OSError as exc:
        _status("Output directory", False, str(exc))
        failures += 1

    exiftool = find_exiftool()
    if exiftool:
        _status("ExifTool", True, f"{exiftool} ({exiftool_version(exiftool)})")
    else:
        _status(
            "ExifTool",
            True,
            "not found (optional for analysis; required only for --write)",
        )
    cache_parent = config.runtime.cache_dir
    try:
        cache_parent.mkdir(parents=True, exist_ok=True)
        probe = cache_parent / ".verify"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        _status("Model cache writable", True, str(cache_parent))
    except OSError as exc:
        _status("Model cache writable", False, str(exc))
        failures += 1

    if shutil.which("nvidia-smi"):
        _status("NVIDIA tooling", True, "nvidia-smi found")
    else:
        _status("NVIDIA tooling", True, "not present; CPU/MPS is supported")
    print(f"\nVerification {'passed' if failures == 0 else 'failed'} ({failures} failure(s)).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
