# SortingTeam Photo Sorter

SortingTeam is a fully local sports-photo assistant for photographers who need to
find one team's players and rank technically usable frames. It detects every person,
compares each upper-body crop with your target and other-team references, scores
configured uniform colors, measures focus on the selected player, and assigns one
recommendation:

- `KEEP`
- `REVIEW`
- `WRONG_TEAM`
- `SOFT`
- `NO_SUBJECT`
- `ERROR`

It never deletes or rejects an image. The first result is an auditable CSV, not an
irreversible edit.

## Privacy and safety

- Inference occurs on this computer with open-source libraries and locally cached
  model files.
- Photos, metadata, embeddings, and prompts are not uploaded.
- No cloud API or face/player identity recognition is used.
- Normal analysis is offline. The setup/download script needs a network connection
  once to obtain third-party model weights.
- Original images are never overwritten or deleted.
- The Lightroom catalog database is never opened or modified.
- XMP writing is a separate, default-dry-run command. It writes sidecar files only.
- Results are recommendations. Review them before rejecting or deleting anything in
  Lightroom.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency/model licensing.
In particular, Ultralytics has AGPL/commercial licensing terms that matter if you
redistribute this application.

## Windows quick start

Python 3.11, 3.12, or 3.13 is supported. No administrator rights are required.

```powershell
Set-Location C:\path\to\SortingTeam
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m photo_sorter validate-config
```

If PowerShell policy prevents direct script execution, start it for this process:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The setup script creates `.venv`, installs the tested dependency set, reports whether
ExifTool is available, downloads YOLO11n and OpenCLIP weights, and runs installation
verification. ExifTool is optional for analysis and required only for actual XMP
writing. When `nvidia-smi` is present it installs the locked CUDA 13.0 PyTorch wheel;
pass `-CpuOnly` to force the smaller CPU build.

## macOS/Linux quick start

```bash
cd /path/to/SortingTeam
bash scripts/setup_macos_linux.sh
source .venv/bin/activate
python -m photo_sorter validate-config
```

CUDA is selected automatically when PyTorch can use it. Apple Silicon uses MPS when
available; otherwise processing uses the CPU. `requirements-lock.txt` captures the
validated Windows/Python 3.13 environment; the macOS/Linux script resolves the
bounded cross-platform ranges in `pyproject.toml`.

## Recommended ingest-first workflow

For sports events, the recommended sequence is now:

```text
SD card/source folder
  → verified copy into a local archive
  → temporary local JPEG previews
  → analyze and human review
  → verified copies of KEEP/REVIEW originals into a Lightroom import folder
  → Lightroom import
```

The archive is the source of truth. The application never deletes, overwrites, or
moves card/archive originals. Do not format the card until you have independently
checked the archive copy and its transfer manifest.

The guided local interface saves the team setup and carries its selected paths through six stages. Start with **0 · Team setup**: choose a folder containing at least one target-uniform JPEG (five target and five other-team examples are preferred). The app blocks analysis before models load if target references are absent, preventing an all-`ERROR` results CSV.

```powershell
streamlit run app.py
```

Each copy operation writes an atomic JSON manifest with source/destination paths,
per-file status, SHA-256 hashes, timestamps, and errors. Restarting an ingest is
safe: an existing destination with the same hash is recorded as
`VERIFIED_EXISTING`; a different file is an error and is never overwritten.

### Command-line ingest alternative

Preview the copy first, then explicitly write it:

```powershell
python -m photo_sorter ingest `
  --source E:\DCIM `
  --destination D:\PhotoArchive\Incoming `
  --write
```

Create temporary previews from the local archive:

```powershell
python -m photo_sorter prepare-previews `
  --source D:\PhotoArchive\Incoming `
  --destination data\input `
  --max-edge 2560 `
  --write
```

JPEG/TIFF/PNG sources are rendered locally with Pillow. For RAW sources, the command
extracts a locally embedded JPEG preview using ExifTool; if ExifTool is unavailable
or the camera format has no usable embedded preview, it records an error and leaves
the RAW untouched. You can instead export temporary Lightroom JPEG previews as
described below.

After analysis and review, stage copies for Lightroom import:

```powershell
python -m photo_sorter stage-originals `
  --results data\output\reviewed_results.csv `
  --originals-dir D:\PhotoArchive\Incoming `
  --destination D:\PhotoArchive\ToImport `
  --decisions KEEP,REVIEW `
  --write
```

This preserves relative folders, verifies every staged copy, and writes a staging
manifest. It copies rather than moves originals. Import `D:\PhotoArchive\ToImport`
into Lightroom when you are satisfied with the reviewed selection.

## Lightroom preview export (alternative)

In Lightroom Classic, select the photos and export temporary previews using:

- **Image format:** JPEG
- **Color space:** sRGB
- **Long edge:** about 2048–3000 pixels
- **Quality:** about 80
- **Filename:** preserve the original filename
- **Metadata:** include capture time (including subseconds when available)

Export them into `data/input/`. Resizing applies only to temporary previews—never
resize or alter the originals for this workflow.

The sorter applies EXIF orientation before detection. Capture time and camera
identifier are used for burst grouping when present.

## Prepare uniform references

Put positive images here:

```text
data/references/target/
```

Put opponents, officials, alternate jerseys, and visually confusing negatives here:

```text
data/references/other/
```

Good references:

- show one dominant player or a tight upper-body/uniform view;
- cover home/away uniforms, lighting conditions, poses, and partial occlusion;
- have at least 15 examples per class when possible;
- do not mix the target and another team at similar sizes in one reference.

The minimum for logistic regression is five valid images per class. With fewer, the
tool warns and falls back to centroid similarity. If no person is detected, a
reference is treated as an already-cropped uniform image. Ambiguous multi-player
references are skipped.

Build or inspect the cached classifier:

```powershell
python -m photo_sorter train-uniform
python -m photo_sorter train-uniform --force
```

Reference contents and relevant config/model settings are fingerprinted. Unchanged
references reuse the local classifier and embeddings.

## Configure team colors

Copy `config/example_team.yaml`, then edit `uniform_colors`. Values use OpenCV HSV:
Hue `0–179`, Saturation/Value `0–255`.

```yaml
uniform_colors:
  - name: navy
    hsv_lower: [100, 70, 35]
    hsv_upper: [135, 255, 180]
    weight: 0.7
  - name: white
    hsv_lower: [0, 0, 180]
    hsv_upper: [179, 70, 255]
    weight: 0.3
```

Red usually crosses the end of the hue scale; express it as two ranges rather than
using a lower hue greater than the upper hue. Color is supporting evidence and is
never the sole classifier.

Validate before a long run:

```powershell
python -m photo_sorter validate-config --config config\my_team.yaml
```

## Analyze photos

```powershell
python -m photo_sorter analyze `
  --input data\input `
  --output data\output\results.csv `
  --config config\default.yaml
```

macOS/Linux:

```bash
python -m photo_sorter analyze \
  --input data/input \
  --output data/output/results.csv \
  --config config/default.yaml
```

An existing CSV is protected unless `--overwrite` is explicit. One corrupt image or
model inference error creates an `ERROR` row and does not stop the remaining batch.
Outputs include:

- `results.csv`, with the documented scores and decision;
- `results_lightroom.csv`, a filename/decision/rating/label/keyword import aid;
- one audit JSON per photo under `data/output/audit/`;
- structured logs in `data/output/photo_sorter.jsonl`.

Focus percentiles are relative to valid selected-player crops in the current batch.
Re-analyzing a different batch can therefore change the percentile and final decision.

Inspect one image in detail:

```powershell
python -m photo_sorter inspect-image `
  --image data\input\example.jpg `
  --debug-output data\output\debug\example
```

This writes an annotated full image, every person crop, every upper-body crop, and
`analysis.json`.

## Review results

```powershell
streamlit run app.py
```

The local app filters by decision; sorts by uniform probability, focus percentile,
or burst rank; displays boxes and crops; and records:

- decision override;
- corrected target-person index;
- target-uniform yes/no;
- focus acceptable yes/no;
- optional note.

Use the visible decision and navigation buttons to review each image.

Overrides are saved atomically to `data/output/reviewed_results.csv`. The raw
`results.csv` is never changed.

## Test and verify the workflow

Run the full automated suite after installation or an upgrade:

```powershell
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python scripts\verify_install.py
```

For a safe manual smoke test, use a small copied folder rather than a live card:

1. Run `ingest` without `--write` and inspect its manifest.
2. Run it again with `--write`; confirm every record is `COPIED` or
   `VERIFIED_EXISTING`.
3. Create previews, run analysis, and inspect a few decisions in the review UI.
4. Run `stage-originals` without `--write`, inspect the staging manifest, then run
   it with `--write` against a disposable Lightroom-import test folder.

## Write XMP metadata safely

First preview all changes:

```powershell
python -m photo_sorter apply-xmp `
  --results data\output\reviewed_results.csv `
  --dry-run
```

This produces `reviewed_results_xmp_changes.csv` and changes no metadata. Without an
originals directory, actual writes create staged `.xmp` files under
`data/output/xmp/`:

```powershell
python -m photo_sorter apply-xmp `
  --results data\output\reviewed_results.csv `
  --write
```

To update sidecars beside proprietary RAW originals, first use Lightroom's
**Metadata > Save Metadata to File** so current catalog metadata exists on disk, then:

```powershell
python -m photo_sorter apply-xmp `
  --results data\output\reviewed_results.csv `
  --originals-dir D:\Photos\Event `
  --write
```

Matching uses relative path/stem first and a unique filename stem second. Missing or
ambiguous originals are skipped and reported. Existing XMP files are backed up under
`data/output/xmp_backups/<timestamp>/` before each write. RAW/JPEG binaries are never
modified.

Keywords are enabled by default:

- `ML_KEEP`
- `ML_REVIEW`
- `ML_WRONG_TEAM`
- `ML_SOFT`
- `ML_NO_SUBJECT`

Unrelated keywords and XMP fields are preserved. Rating and color-label writes are
disabled by default and may be enabled independently:

```powershell
python -m photo_sorter apply-xmp `
  --results data\output\reviewed_results.csv `
  --originals-dir D:\Photos\Event `
  --ratings --color-labels --write
```

After examining sidecars and backups, use Lightroom's **Metadata > Read Metadata from
File** on the intended originals. Resolve any Lightroom metadata-conflict warnings
before continuing. Sidecars beside exported JPEG previews do not reliably update
cataloged RAW originals, which is why staging and explicit original mapping are
separate.

## Recalibrate thresholds

After reviewing a representative set:

```powershell
python -m photo_sorter calibrate `
  --reviewed-results data\output\reviewed_results.csv
```

The command reports target-uniform precision/recall/F1, false-positive and
false-negative rates, focus agreement, decision confusion/accuracy, and writes
recommendations to `config/calibrated_thresholds.yaml`. It never overwrites
`config/default.yaml`. Treat recommendations based on fewer than 30 diverse reviews
as low confidence.

See [docs/calibration.md](docs/calibration.md) for details.

## Other commands

```text
python -m photo_sorter --help
python -m photo_sorter validate-config
python -m photo_sorter train-uniform
python -m photo_sorter inspect-image --image ... --debug-output ...
python -m photo_sorter clear-cache
python -m photo_sorter clear-cache --include-models --yes
```

`clear-cache` preserves downloaded model weights unless both model flags are given.

## Common failure cases

- **Missing weight:** run `python scripts/download_models.py`.
- **No target references:** add JPEGs under `data/references/target/`.
- **Most references skipped:** crop them to one dominant player.
- **No people detected:** lower detector confidence carefully or increase inference
  size; distant/occluded players remain difficult.
- **Uniform false positives:** add confusing examples to `other/`, cover alternate
  uniforms, and recalibrate.
- **Noise looks sharp:** ensure ISO metadata is exported and tune `focus.denoise` /
  `noise_penalty`.
- **XMP write unavailable:** install ExifTool and put it on `PATH`.
- **XMP original skipped:** preserve original filename/relative paths or remove
  duplicate stems under the originals directory.
- **Out of memory:** reduce `runtime.chunk_size`, `clip.batch_size`, and YOLO
  `inference_size`.

More diagnostics are in [docs/troubleshooting.md](docs/troubleshooting.md).

## Expected limitations

- A team uniform classifier is only as representative as its references.
- Small, back-facing, heavily occluded, or motion-blurred players may be missed.
- Similar opponent colors, warm arena lighting, and alternate jerseys can confuse
  both CLIP and HSV evidence.
- Laplacian/Tenengrad are useful technical signals, not aesthetic judgments; noise,
  compression, and patterned backgrounds affect them.
- Batch focus percentiles are relative, not a universal optical-quality measurement.
- Burst grouping is heuristic when capture timestamps/subseconds are absent.
- The system recognizes a target uniform, not a person's identity.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest --cov=photo_sorter --cov-report=term-missing
```

Architecture is documented in [docs/architecture.md](docs/architecture.md).
