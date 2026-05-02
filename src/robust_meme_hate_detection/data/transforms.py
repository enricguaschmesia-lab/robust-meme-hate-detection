"""Simple image transforms used for dataloader smoke tests.

The CLIP baseline will later use the model-specific processor. This file keeps a
minimal tensor transform available for Phase 1 dataloader validation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


class ResizeToTensor:
    """Resize a PIL image and convert it to a CHW float tensor in [0, 1]."""

    def __init__(self, size: int | tuple[int, int] = 224) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image: Image.Image):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyTorch is required for ResizeToTensor.") from exc

        resized = image.resize(self.size, Image.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class NormalizeTensor:
    """Normalize a CHW tensor with per-channel mean and standard deviation."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyTorch is required for NormalizeTensor.") from exc

        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class Compose:
    """Small transform composition helper to avoid a torchvision dependency here."""

    def __init__(self, transforms) -> None:
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value
