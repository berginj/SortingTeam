# Troubleshooting

## Installation

Run:

```powershell
python scripts\verify_install.py
```

Python must be 3.11–3.13. Re-run the setup script from a normal user PowerShell if
imports are missing. A missing ExifTool is not an analysis failure.

If PyTorch cannot see an NVIDIA GPU, confirm the installed PyTorch build and driver
with the official PyTorch selector. CPU execution remains supported.

## Missing or unusable models

```powershell
python scripts\download_models.py --config config\default.yaml --force
```

Normal analysis never downloads a missing model automatically. Model manifests and
SHA-256 digests are stored under `.cache/photo_sorter/models/`.

## Bad detections

- Increase `person_detection.inference_size` for distant players.
- Lower `confidence_threshold` in small increments; inspect the resulting false
  positives.
- Use `inspect-image` to verify clamped boxes and crops.
- Do not make largest-player assumptions in game photos; improve uniform references
  instead.

## Bad uniform results

- Add hard negatives: officials, crowd members, opponent alternates, warmup gear.
- Ensure each reference has one dominant player.
- Cover home/away and lighting variations.
- Check HSV ranges against OpenCV's hue scale.
- Run calibration after a representative human review.

## Bad focus results

- Focus is measured on the selected person, so first verify person selection.
- Export ISO metadata to allow high-ISO denoising.
- Patterned jerseys and noise can inflate edge metrics; tune noise correction.
- Compare photos in coherent batches because percentile is batch-relative.

## XMP

- Always run dry-run and inspect its CSV first.
- ExifTool must be on `PATH` for `--write`.
- Duplicate RAW stems are skipped unless relative directory mapping is unique.
- Sidecars for JPEG previews do not reliably update RAW originals.
- Existing sidecars are backed up before writes. If a write reports `ERROR`, restore
  the backup and investigate before retrying.

## Performance

For memory pressure, lower:

```yaml
runtime:
  chunk_size: 8
clip:
  batch_size: 8
person_detection:
  inference_size: 640
```

CPU time depends strongly on preview size and people per image. The application
releases decoded images after each chunk and never loads a model per image.

