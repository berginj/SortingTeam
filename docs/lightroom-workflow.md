# Lightroom Classic workflow

## 1. Export previews

Export selected RAW photos to `data/input/` as JPEG, sRGB, quality about 80, long edge
2048–3000 pixels. Preserve filenames and capture-time metadata.

## 2. Analyze and review

```powershell
python -m photo_sorter analyze --input data\input --output data\output\results.csv
streamlit run app.py
```

Use the raw CSV for audit and `reviewed_results.csv` for accepted overrides.

## 3. Preview metadata changes

```powershell
python -m photo_sorter apply-xmp --results data\output\reviewed_results.csv --dry-run
```

Examine every `SKIP`, `CHANGE`, and destination path in the generated report.

## 4. Protect existing Lightroom metadata

Before touching sidecars beside originals:

1. Back up the photo folder and Lightroom catalog.
2. In Lightroom, select the intended files.
3. Use **Metadata > Save Metadata to File**.
4. Confirm existing `.xmp` files are present where expected.

The sorter creates another timestamped backup for each existing sidecar before it
writes.

## 5. Write and import

```powershell
python -m photo_sorter apply-xmp `
  --results data\output\reviewed_results.csv `
  --originals-dir D:\Photos\Event `
  --write
```

No RAW binary is modified. Back in Lightroom, select the same images and use
**Metadata > Read Metadata from File**. Resolve metadata-conflict warnings
deliberately; do not blindly overwrite newer Lightroom edits.

The companion `_lightroom.csv` is intended for filename-based filtering or external
collection tooling. It is not a direct Lightroom catalog writer.

