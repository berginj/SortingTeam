# Architecture

## Data flow

1. `image_loader` decodes a bounded preview, applies EXIF orientation, and extracts
   capture/camera/ISO metadata.
2. A `PersonDetector` returns all confident person boxes.
3. Crop utilities clamp a padded person box and its upper 62.5% uniform region.
4. OpenCLIP embeds every upper-body crop in batches.
5. The uniform model returns logistic probability or centroid evidence; the HSV
   scorer adds team-color evidence.
6. Target selection combines uniform, color, detection confidence, crop quality, and
   optional centrality. It may deliberately select no one.
7. Laplacian and Tenengrad focus metrics operate only on the selected padded person.
8. After all images have raw focus values, robust batch normalization assigns
   normalized scores and percentiles.
9. The decision engine applies explicit precedence, then burst grouping ranks frames.
10. Atomic CSV and per-image audit JSON outputs feed Streamlit, calibration, and XMP.

The pipeline keeps only one configurable chunk of decoded images/crops in memory.
Models are loaded once. A single inference coordinator owns GPU/model calls; bounded
threads handle decoding and CPU operations.

## Interfaces

- `PersonDetector.detect/detect_batch` isolates Ultralytics.
- `ImageEmbedder.embed` isolates OpenCLIP.
- `UniformModel.predict` exposes normalized evidence independently of training mode.
- Pydantic `ImageResult` and `PersonEvaluation` models define durable output data.
- Pure functions implement crop, color, focus normalization, selection, decisions,
  and burst logic for deterministic tests.

Tests inject fake detector/embedder implementations. Ordinary `pytest` therefore
needs no network, model weights, or GPU. The `model` marker is reserved for optional
real-weight smoke checks.

## Uniform model and cache

Reference contents and relevant model/crop/classifier settings form a SHA-256
fingerprint. The fitted classifier, normalized centroids, accepted-reference list,
and validation metrics are stored under `.cache/photo_sorter/uniform/`.

Logistic regression is used with at least five accepted examples in each class.
Otherwise:

- target + other centroids use a temperature-scaled cosine difference;
- target-only mode maps target cosine through a configurable sigmoid.

The CSV exposes target similarity mapped to `[0,1]`. Audit JSON retains raw target,
other, and margin values.

## Focus normalization

The selected crop is downscaled to a common long edge. Optional bilateral denoising
activates only above the configured ISO threshold. A robust high-frequency residual
estimates noise, which can be subtracted from edge metrics.

Each metric is transformed with `log1p`, centered by its median, scaled by median
absolute deviation, and mapped through a sigmoid. The two normalized metrics are
combined and mid-ranked from 0–100. This avoids pretending that one raw threshold is
portable across cameras, export sizes, and sports.

## Privacy boundary

Runtime model paths must resolve to local files. The analysis process supplies no
URLs and needs no cloud SDK. Only `scripts/download_models.py` intentionally accesses
upstream model hosting. Audit artifacts and embeddings remain in ignored local
directories.

Before model imports, runtime enables Hugging Face/transformer offline modes, sets
`YOLO_OFFLINE`, disables Ultralytics event synchronization/integrations, and isolates
Ultralytics settings under the project cache.

No module reads SQLite/Lightroom catalogs, recognizes faces, deletes images, or writes
image binaries.
