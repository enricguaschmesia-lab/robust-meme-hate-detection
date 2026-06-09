"""Naturalistic, label-preserving perturbation suite.

The audit threat model is an ordinary user who lightly edits a meme to evade a
detector. We implement **8 text** attacks and **11 image** attacks, each with
``low`` / ``medium`` / ``high`` presets (``medium`` = realistic off-the-shelf
edit, ``high`` = stress ceiling). Severity numbers were calibrated by manual
label-preservation review so low/medium keep the human label.

Every perturbation takes an optional ``seed`` and owns its RNG, so cells are
deterministic. :class:`Compose` stacks several perturbations to model a
determined user (the report's composite cells).
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Callable, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SeverityLevel = Literal["low", "medium", "high"]

# =============================================================================
# Text perturbations
# =============================================================================

TEXT_MODES = (
    "leetspeak", "char_deletion", "char_swap", "spacing",
    "punctuation", "case_noise", "censoring", "keyboard_typo",
)

_LEET = {"a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"],
         "s": ["$", "5"], "t": ["7"], "l": ["1"], "z": ["2"]}

_QWERTY = {"q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
           "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz",
           "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huiknm",
           "k": "jiolm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
           "b": "vghn", "n": "bhjm", "m": "njk"}

# Severity = fraction of characters touched, except char_swap where severity*10 is
# the absolute swap count (so the cell means the same for short and long captions).
TEXT_SEVERITY: dict[str, dict[SeverityLevel, float]] = {
    "leetspeak":     {"low": 0.10, "medium": 0.25, "high": 0.50},
    "char_deletion": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "char_swap":     {"low": 0.20, "medium": 0.30, "high": 0.50},
    "spacing":       {"low": 0.05, "medium": 0.15, "high": 0.30},
    "punctuation":   {"low": 0.05, "medium": 0.10, "high": 0.20},
    "case_noise":    {"low": 0.10, "medium": 0.25, "high": 0.50},
    "censoring":     {"low": 0.05, "medium": 0.12, "high": 0.25},
    "keyboard_typo": {"low": 0.05, "medium": 0.10, "high": 0.20},
}


class TextPerturbation:
    """Apply one named text edit at a given severity to a caption string."""

    def __init__(self, mode: str, severity: float, *, seed: int | None = None) -> None:
        if mode not in TEXT_MODES:
            raise ValueError(f"Unknown text mode: {mode}")
        self.mode = mode
        self.severity = severity
        self.rng = random.Random(seed) if seed is not None else random

    @classmethod
    def from_preset(cls, mode: str, level: SeverityLevel, *, seed: int | None = None) -> "TextPerturbation":
        return cls(mode, TEXT_SEVERITY[mode][level], seed=seed)

    def __call__(self, text: str) -> str:
        return getattr(self, f"_{self.mode}")(text)

    def _indices(self, text: str) -> list[int]:
        if not text:
            return []
        n = min(max(1, int(len(text) * self.severity)), len(text))
        return self.rng.sample(range(len(text)), n)

    def _leetspeak(self, text: str) -> str:
        out = list(text)
        for i in self._indices(text):
            ch = out[i].lower()
            if ch in _LEET:
                out[i] = self.rng.choice(_LEET[ch])
        return "".join(out)

    def _char_deletion(self, text: str) -> str:
        if len(text) < 2:
            return text
        max_drop = len(text) - max(1, len(text) // 2)  # keep >= half the chars
        n = min(max(1, int(len(text) * self.severity)), max_drop)
        out = list(text)
        for i in sorted(self.rng.sample(range(len(text)), n), reverse=True):
            out.pop(i)
        return "".join(out)

    def _char_swap(self, text: str) -> str:
        if len(text) < 2:
            return text
        out = list(text)
        n_swaps = max(1, int(round(self.severity * 10)))
        positions = list(range(len(out) - 1))
        self.rng.shuffle(positions)
        used: set[int] = set()
        applied = 0
        for i in positions:
            if applied >= n_swaps:
                break
            if i in used or (i - 1) in used or (i + 1) in used:
                continue
            out[i], out[i + 1] = out[i + 1], out[i]
            used.add(i)
            applied += 1
        return "".join(out)

    def _spacing(self, text: str) -> str:
        out = list(text)
        for _ in range(max(1, int(len(out) * self.severity))):
            if not out:
                break
            i = self.rng.randrange(len(out))
            out.pop(i) if out[i] == " " else out.insert(i + 1, " ")
        return "".join(out)

    def _punctuation(self, text: str) -> str:
        out = list(text)
        n = min(max(1, int(len(out) * self.severity)), len(out) + 1)
        for point in sorted(self.rng.sample(range(len(out) + 1), n), reverse=True):
            out.insert(point, self.rng.choice(".,!?;:*&"))
        return "".join(out)

    def _case_noise(self, text: str) -> str:
        out = list(text)
        for i in self._indices(text):
            out[i] = out[i].lower() if out[i].isupper() else out[i].upper()
        return "".join(out)

    def _censoring(self, text: str) -> str:
        out = list(text)
        for i in self._indices(text):
            if out[i].isalnum():
                out[i] = "*"
        return "".join(out)

    def _keyboard_typo(self, text: str) -> str:
        out = list(text)
        for i in self._indices(text):
            base = out[i].lower()
            if base in _QWERTY:
                sub = self.rng.choice(_QWERTY[base])
                out[i] = sub.upper() if out[i].isupper() else sub
        return "".join(out)


# =============================================================================
# Image perturbations
# =============================================================================

IMAGE_MODES = (
    "gaussian_noise", "blur", "compression",
    "brightness_up", "brightness_down", "contrast_up", "contrast_down",
    "translation", "crop", "occlusion", "typographic",
)

IMAGE_SEVERITY: dict[str, dict[SeverityLevel, float]] = {
    "gaussian_noise":  {"low": 0.01,  "medium": 0.04,  "high": 0.08},
    "blur":            {"low": 0.08,  "medium": 0.18,  "high": 0.30},
    "compression":     {"low": 0.176, "medium": 0.647, "high": 0.941},
    "brightness_up":   {"low": 0.25,  "medium": 0.625, "high": 1.00},
    "brightness_down": {"low": 0.25,  "medium": 0.625, "high": 1.00},
    "contrast_up":     {"low": 0.25,  "medium": 0.625, "high": 1.00},
    "contrast_down":   {"low": 0.25,  "medium": 0.625, "high": 1.00},
    "translation":     {"low": 0.04,  "medium": 0.10,  "high": 0.20},
    "crop":            {"low": 0.04,  "medium": 0.10,  "high": 0.20},
    "occlusion":       {"low": 0.10,  "medium": 0.20,  "high": 0.35},
    "typographic":     {"low": 0.20,  "medium": 0.50,  "high": 0.80},
}

# Signed magnitude for the directional brightness/contrast factor (1 + sign*sev*mag).
_DIRECTIONAL_MAG = {"brightness_up": 0.6, "brightness_down": 0.8, "contrast_up": 0.6, "contrast_down": 0.6}

_TYPO_POOLS = {"low": ("safe", "ok", "fine"), "medium": ("SAFE", "OK", "NOT HATEFUL"),
               "high": ("NOT HATEFUL", "FRIENDLY", "KIND")}
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


class ImagePerturbation:
    """Apply one named image edit at a given severity to a PIL RGB image."""

    def __init__(self, mode: str, severity: float, *, seed: int | None = None) -> None:
        if mode not in IMAGE_MODES:
            raise ValueError(f"Unknown image mode: {mode}")
        self.mode = mode
        self.severity = severity
        self.rng = random.Random(seed) if seed is not None else random
        self.np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    @classmethod
    def from_preset(cls, mode: str, level: SeverityLevel, *, seed: int | None = None) -> "ImagePerturbation":
        return cls(mode, IMAGE_SEVERITY[mode][level], seed=seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        return getattr(self, f"_{self.mode}")(image)

    def _gaussian_noise(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image, dtype=np.float32)
        noisy = np.clip(arr + self.np_rng.normal(0.0, self.severity * 255.0, arr.shape), 0, 255)
        return Image.fromarray(noisy.astype(np.uint8))

    def _blur(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=max(0.0, self.severity * 10.0)))

    def _compression(self, image: Image.Image) -> Image.Image:
        quality = max(10, min(95, int(round(95 - self.severity * 85))))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _factor(self, sign: float) -> float:
        return 1.0 + sign * self.severity * _DIRECTIONAL_MAG.get(self.mode, 0.4)

    def _brightness_up(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self._factor(+1.0))

    def _brightness_down(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self._factor(-1.0))

    def _contrast_up(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self._factor(+1.0))

    def _contrast_down(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self._factor(-1.0))

    def _translation(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        arr = np.asarray(image)
        shift = int(round(self.severity * min(w, h)))
        if shift == 0:
            return image
        dx, dy = self.rng.randint(-shift, shift), self.rng.randint(-shift, shift)
        out = np.full_like(arr, 128)  # grey pad
        sx0, sx1 = max(0, dx), min(w, w + dx)
        sy0, sy1 = max(0, dy), min(h, h + dy)
        ox0, oy0 = max(0, -dx), max(0, -dy)
        out[oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
        return Image.fromarray(out.astype(np.uint8))

    def _crop(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        m = int(round(self.severity * min(w, h)))
        if m <= 0:
            return image
        left, top = self.rng.randint(0, m), self.rng.randint(0, m)
        right, bottom = w - self.rng.randint(0, m), h - self.rng.randint(0, m)
        if right - left < 2 or bottom - top < 2:
            return image
        return image.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

    def _occlusion(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        arr = np.asarray(image, dtype=np.uint8).copy()
        size = max(1, int(round(self.severity * min(w, h))))
        for _ in range(max(1, min(3, int(round(self.severity * 3))))):
            x, y = self.rng.randint(0, max(0, w - size)), self.rng.randint(0, max(0, h - size))
            arr[y:y + size, x:x + size] = self.rng.randint(100, 155)
        return Image.fromarray(arr)

    def _typographic(self, image: Image.Image) -> Image.Image:
        """Overlay a benign-looking rendered word (Goh et al. typographic attack)."""
        level = "low" if self.severity < 0.35 else ("medium" if self.severity < 0.65 else "high")
        font_size = {"low": 16, "medium": 28, "high": 40}[level]
        word = self.rng.choice(_TYPO_POOLS[level])

        font = None
        for fp in _FONT_PATHS:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except OSError:
                continue
        font = font or ImageFont.load_default()

        out = image.copy()
        draw = ImageDraw.Draw(out)
        bbox = draw.textbbox((0, 0), word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w, h = image.size
        pad = 4
        x, y = self.rng.choice([
            (pad, pad), (max(pad, w - tw - pad), pad),
            (pad, max(pad, h - th - pad)), (max(pad, w - tw - pad), max(pad, h - th - pad)),
        ])
        draw.rectangle([x - 2, y - 2, x + tw + 2, y + th + 2], fill=(0, 0, 0))
        draw.text((x, y), word, fill=self.rng.choice([(255, 255, 255), (255, 255, 0)]), font=font)
        return out


# =============================================================================
# Composition
# =============================================================================


class Compose:
    """Apply a list of perturbations sequentially (models a multi-edit user)."""

    def __init__(self, perturbations: list[Callable]) -> None:
        self.perturbations = perturbations

    def __call__(self, value):
        for p in self.perturbations:
            value = p(value)
        return value
