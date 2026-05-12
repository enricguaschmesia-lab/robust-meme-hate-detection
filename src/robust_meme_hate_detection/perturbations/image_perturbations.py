"""Image perturbation functions for adversarial robustness evaluation.

Each ``ImagePerturbation`` instance applies a single named transformation on
a PIL image with a configurable severity. Three named severity levels are
provided via ``SEVERITY_PRESETS`` and ``ImagePerturbation.from_preset``;
the numbers are chosen to match the roadmap's examples (Gaussian σ
0.01/0.03/0.05, JPEG quality 80/50/25, brightness/contrast ±10/25/40 %,
translation/crop 2/5/10 %) and to keep low and medium severities label-
preserving on 224×224 inputs.

For reproducibility, each instance optionally owns its own
``random.Random`` and ``numpy.random.Generator`` seeded from a single
``seed`` integer. When ``seed`` is ``None`` the class falls back to the
module-level ``random`` and ``numpy.random`` so legacy callers keep
working.

Supported modes:
    gaussian_noise, blur, compression, brightness, contrast, translation,
    crop, occlusion.
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_MODES = (
    'gaussian_noise',
    'blur',
    'compression',
    # Directional brightness/contrast modes for unambiguous report cells —
    # use these in the Phase-3 benchmark (one signed direction per cell so
    # the cell-level metric isn't averaged over a bimodal population).
    'brightness_up',
    'brightness_down',
    'contrast_up',
    'contrast_down',
    # Bidirectional aliases kept for augmentation use (random sign per
    # sample). Not recommended for the benchmark.
    'brightness',
    'contrast',
    'translation',
    'crop',
    'occlusion',
)

# The "benchmark" subset: directional brightness/contrast, no random-sign
# aliases. Use this in eval/run_perturbed.py and scripts/inspect_perturbations.py.
IMAGE_MODES_BENCHMARK = (
    'gaussian_noise',
    'blur',
    'compression',
    'brightness_up',
    'brightness_down',
    'contrast_up',
    'contrast_down',
    'translation',
    'crop',
    'occlusion',
)

SeverityLevel = Literal['low', 'medium', 'high']

# Per-mode severity presets. The ``severity`` float is interpreted differently
# per mode; the chosen values reproduce the roadmap examples:
#   gaussian_noise — std as fraction of [0, 255]: 0.01 / 0.03 / 0.05
#   blur — PIL Gaussian radius in px; the roadmap's "kernel 3/5/7" maps to
#          radius (kernel-1)/2 = 1/2/3. So severity*10 (rounded) → radius
#          1/2/3 for low/medium/high.
#   compression — quality = 95 - severity*85: 0.176 / 0.529 / 0.824 → q≈80/50/25
#   brightness_{up,down} / contrast_{up,down} — directional shift of magnitude
#          severity*0.4: 0.25 / 0.625 / 1.0 → ±10/25/40 %
#   translation — max shift as fraction of min(W, H): 0.02 / 0.05 / 0.10
#   crop — symmetric crop fraction per edge: 0.02 / 0.05 / 0.10
#   occlusion — patch side as fraction of min(W, H): 0.05 / 0.10 / 0.20
SEVERITY_PRESETS: dict[str, dict[SeverityLevel, float]] = {
    'gaussian_noise':    {'low': 0.01,  'medium': 0.03,  'high': 0.05},
    'blur':              {'low': 0.10,  'medium': 0.20,  'high': 0.30},
    'compression':       {'low': 0.176, 'medium': 0.529, 'high': 0.824},
    'brightness_up':     {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    'brightness_down':   {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    'contrast_up':       {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    'contrast_down':     {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    # Bidirectional aliases share the same severity numbers but the per-call
    # sign is random; intended for training augmentation, not for benchmark cells.
    'brightness':        {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    'contrast':          {'low': 0.25,  'medium': 0.625, 'high': 1.00},
    'translation':       {'low': 0.02,  'medium': 0.05,  'high': 0.10},
    'crop':              {'low': 0.02,  'medium': 0.05,  'high': 0.10},
    'occlusion':         {'low': 0.05,  'medium': 0.10,  'high': 0.20},
}


class ImagePerturbation:
    """Apply a single image perturbation with configurable severity and seed.

    Operates on PIL ``Image`` objects (RGB) and returns a same-size PIL
    image. Brightness and contrast are bidirectional: the sign of the
    deviation from 1.0 is drawn from the instance RNG.
    """

    def __init__(
        self,
        mode: str = 'gaussian_noise',
        probability: float = 1.0,
        severity: float = 0.1,
        *,
        seed: int | None = None,
    ) -> None:
        if mode not in IMAGE_MODES:
            raise ValueError(f"Unknown image perturbation mode: {mode}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")

        self.mode = mode
        self.probability = probability
        self.severity = severity
        self.seed = seed
        if seed is not None:
            self._py_rng: random.Random | None = random.Random(seed)
            self._np_rng: np.random.Generator | None = np.random.default_rng(seed)
        else:
            self._py_rng = None
            self._np_rng = None

    @classmethod
    def from_preset(
        cls,
        mode: str,
        level: SeverityLevel,
        *,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> "ImagePerturbation":
        """Construct using a named severity level from ``SEVERITY_PRESETS``."""
        if mode not in SEVERITY_PRESETS:
            raise ValueError(f"No severity preset for mode: {mode}")
        if level not in SEVERITY_PRESETS[mode]:
            raise ValueError(f"Unknown severity level {level!r} for {mode!r}")
        return cls(
            mode=mode,
            probability=probability,
            severity=SEVERITY_PRESETS[mode][level],
            seed=seed,
        )

    # ------------------------------------------------------------------ rng
    def _rand(self) -> random.Random:
        return self._py_rng if self._py_rng is not None else random

    def _nprand(self):
        # Returns either the seeded Generator or the legacy module.
        return self._np_rng if self._np_rng is not None else np.random

    # ----------------------------------------------------------------- main
    def __call__(self, image: Image.Image) -> Image.Image:
        rng = self._rand()
        if rng.random() > self.probability:
            return image

        dispatch = {
            'gaussian_noise':  self._gaussian_noise,
            'blur':            self._blur,
            'compression':     self._compression,
            'brightness':      self._brightness,
            'brightness_up':   lambda im: self._brightness_directional(im, +1.0),
            'brightness_down': lambda im: self._brightness_directional(im, -1.0),
            'contrast':        self._contrast,
            'contrast_up':     lambda im: self._contrast_directional(im, +1.0),
            'contrast_down':   lambda im: self._contrast_directional(im, -1.0),
            'translation':     self._translation,
            'crop':            self._crop,
            'occlusion':       self._occlusion,
        }
        return dispatch[self.mode](image)

    # --------------------------------------------------------------- ops
    def _gaussian_noise(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image, dtype=np.float32)
        std_dev = self.severity * 255.0
        npr = self._nprand()
        if isinstance(npr, np.random.Generator):
            noise = npr.normal(0.0, std_dev, arr.shape)
        else:
            noise = npr.normal(0.0, std_dev, arr.shape)
        noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy)

    def _blur(self, image: Image.Image) -> Image.Image:
        """PIL Gaussian blur. severity*10 (rounded) = radius in px.

        Roadmap axis is "kernel 3/5/7" which corresponds to radius (kernel-1)/2
        = 1/2/3. The low/medium/high presets pick severity 0.1/0.2/0.3
        accordingly.
        """
        radius = max(1, int(round(self.severity * 10)))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _compression(self, image: Image.Image) -> Image.Image:
        # severity 0.0–1.0 → JPEG quality 95–10. Higher severity → harsher.
        quality = max(10, min(95, int(round(95 - self.severity * 85))))
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def _bidirectional_factor(self) -> float:
        """Factor 1 ± severity*0.4, sign chosen by RNG.

        severity=0 → 1.0 (no change). severity=1 → 0.6 or 1.4 (±40 %).
        """
        rng = self._rand()
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return 1.0 + sign * self.severity * 0.4

    def _directional_factor(self, sign: float) -> float:
        """Factor 1 + sign*severity*0.4. severity=0 → 1.0; severity=1 → 1.4 or 0.6."""
        return 1.0 + sign * self.severity * 0.4

    def _brightness(self, image: Image.Image) -> Image.Image:
        factor = self._bidirectional_factor()
        return ImageEnhance.Brightness(image).enhance(factor)

    def _brightness_directional(self, image: Image.Image, sign: float) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self._directional_factor(sign))

    def _contrast(self, image: Image.Image) -> Image.Image:
        factor = self._bidirectional_factor()
        return ImageEnhance.Contrast(image).enhance(factor)

    def _contrast_directional(self, image: Image.Image, sign: float) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self._directional_factor(sign))

    def _translation(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        arr = np.asarray(image)
        max_shift = int(round(self.severity * min(width, height)))
        if max_shift == 0:
            return image
        rng = self._rand()
        shift_x = rng.randint(-max_shift, max_shift)
        shift_y = rng.randint(-max_shift, max_shift)

        output = np.full_like(arr, 128)  # gray pad
        # Source slice (region of input that survives the shift)
        sx0 = max(0, shift_x); sx1 = min(width,  width  + shift_x)
        sy0 = max(0, shift_y); sy1 = min(height, height + shift_y)
        # Destination slice
        dx0 = max(0, -shift_x); dx1 = dx0 + (sx1 - sx0)
        dy0 = max(0, -shift_y); dy1 = dy0 + (sy1 - sy0)
        output[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
        return Image.fromarray(output.astype(np.uint8))

    def _crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_crop = int(round(self.severity * min(width, height)))
        if max_crop <= 0:
            return image
        rng = self._rand()
        left   = rng.randint(0, max_crop)
        top    = rng.randint(0, max_crop)
        right  = width  - rng.randint(0, max_crop)
        bottom = height - rng.randint(0, max_crop)
        if right - left < 2 or bottom - top < 2:
            return image
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.BILINEAR)

    def _occlusion(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        arr = np.asarray(image, dtype=np.uint8).copy()
        mask_size = max(1, int(round(self.severity * min(width, height))))
        # 1 to 3 masks, scaling with severity.
        n_masks = max(1, min(3, int(round(self.severity * 3))))
        rng = self._rand()
        for _ in range(n_masks):
            x = rng.randint(0, max(0, width  - mask_size))
            y = rng.randint(0, max(0, height - mask_size))
            colour = rng.randint(100, 155)
            arr[y:y + mask_size, x:x + mask_size] = colour
        return Image.fromarray(arr)
