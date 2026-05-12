"""Image and text transforms for the CLIP fusion model.

For the robustness study, the contract is: the image branch sees a raw
[0, 1] tensor of shape (3, 224, 224); Normalize lives **inside** the model so
that FGSM/PGD can backprop to pixel space. See model_architecture.md §4.2.
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


# ---------------------------------------------------------------------------
# CLIP-specific helpers used by the model fusion classifier.
# ---------------------------------------------------------------------------


# OpenAI CLIP normalization statistics. Matches OpenCLIP/laion2b checkpoints.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipImage01Transform:
    """PIL -> resize 224 (bicubic, antialias) -> CenterCrop -> tensor in [0, 1].

    The output tensor is intentionally NOT normalized; Normalize lives inside
    the model so that adversarial attacks can operate on raw pixels.
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, image: Image.Image):
        try:
            import torch
            from torchvision import transforms as T
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch + torchvision are required for ClipImage01Transform.") from exc

        ops = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(self.size),
            T.ToTensor(),  # -> [0, 1] CHW float
        ])
        return ops(image)


class ClipTokenize:
    """Wrap an OpenCLIP tokenizer so the dataset returns (77,) long tensors."""

    def __init__(self, arch: str = "ViT-B-32") -> None:
        try:
            import open_clip
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("open_clip_torch is required for ClipTokenize.") from exc
        self._tok = open_clip.get_tokenizer(arch)

    def __call__(self, text: str):
        # OpenCLIP tokenizer expects a list and returns a (B, 77) tensor;
        # the dataset returns one example at a time so we squeeze.
        ids = self._tok([text])
        return ids[0]
