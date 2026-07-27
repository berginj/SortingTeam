#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"

"$PYTHON" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.11, 3.12, or 3.13 is required; found {sys.version.split()[0]}")
PY

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${PHOTO_SORTER_CPU_ONLY:-0}" != "1" ]]; then
  echo "NVIDIA GPU detected; installing the locked CUDA 13.0 PyTorch build..."
  .venv/bin/python -m pip install \
    "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130
fi
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pip check

if ! command -v exiftool >/dev/null 2>&1; then
  echo "Warning: ExifTool is optional for analysis but required for XMP --write."
fi

.venv/bin/python scripts/download_models.py --config config/default.yaml
.venv/bin/python scripts/verify_install.py

echo
echo "Activate with: source .venv/bin/activate"
echo "Then run:"
echo "  python -m photo_sorter validate-config"
echo "  python -m photo_sorter analyze --input data/input --output data/output/results.csv"
echo "  streamlit run app.py"
