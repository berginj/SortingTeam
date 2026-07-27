from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from photo_sorter.models.person_detector import ModelNotAvailableError
from photo_sorter.schemas.config_models import ClipConfig


class ImageEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, images: Sequence[Image.Image]) -> npt.NDArray[np.float32]: ...


def choose_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class OpenClipEmbedder:
    """OpenCLIP image embedding backed by an explicitly local checkpoint."""

    def __init__(self, config: ClipConfig, device: str) -> None:
        checkpoint = Path(config.checkpoint_path)
        if not checkpoint.is_file():
            raise ModelNotAvailableError(
                f"OpenCLIP checkpoint not found at {checkpoint}. "
                "Run: python scripts/download_models.py --config config/default.yaml"
            )
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ModelNotAvailableError(
                "OpenCLIP/PyTorch is not installed. Run the platform setup script."
            ) from exc

        self._torch = torch
        self._config = config
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=str(checkpoint),
            device=device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._dimension = int(getattr(model.visual, "output_dim", 512))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, images: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        if not images:
            return np.empty((0, self._dimension), dtype=np.float32)
        batches: list[npt.NDArray[np.float32]] = []
        use_fp16 = self._config.precision == "fp16" or (
            self._config.precision == "auto" and self._device.startswith("cuda")
        )
        for start in range(0, len(images), self._config.batch_size):
            batch_images = images[start : start + self._config.batch_size]
            tensor = self._torch.stack(
                [self._preprocess(image.convert("RGB")) for image in batch_images]
            ).to(self._device)
            autocast: Any
            if use_fp16 and self._device.startswith("cuda"):
                autocast = self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            else:
                autocast = nullcontext()
            with self._torch.inference_mode(), autocast:
                features = self._model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            batches.append(np.asarray(features.float().cpu().numpy(), dtype=np.float32))
        return np.asarray(np.concatenate(batches, axis=0), dtype=np.float32)
