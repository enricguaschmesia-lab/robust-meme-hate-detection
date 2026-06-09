"""On-the-fly perturbed view builder for naturalistic robust training.

For each example in a batch the augmenter independently samples a perturbation
kind from ``{text, image, both}`` (default weights 0.5 / 0.3 / 0.2), a concrete
attack, and a severity, applies it to the raw caption / PIL image, and re-runs
the project transforms so the perturbed view is shape-compatible with the clean
batch. A single seeded RNG makes the stream reproducible.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import torch
from PIL import Image

from robust_meme_hate_detection.perturbations import ImagePerturbation, TextPerturbation


class RobustAugmenter:
    def __init__(
        self,
        *,
        seed: int,
        text_fraction: float = 0.5,
        image_fraction: float = 0.3,
        both_fraction: float = 0.2,
        text_attacks: Sequence[str] = ("leetspeak", "char_deletion", "char_swap", "censoring", "keyboard_typo"),
        image_attacks: Sequence[str] = ("gaussian_noise", "blur", "compression", "brightness_down", "occlusion", "typographic"),
        severity_weights_text: Sequence[float] = (1.0, 2.0, 2.0),
        severity_weights_image: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        if abs(text_fraction + image_fraction + both_fraction - 1.0) > 1e-3:
            raise ValueError("text/image/both fractions must sum to 1")
        self.rng = random.Random(seed)
        self.kinds = ("text", "image", "both")
        self.kind_weights = (text_fraction, image_fraction, both_fraction)
        self.text_attacks = tuple(text_attacks)
        self.image_attacks = tuple(image_attacks)
        self.levels = ("low", "medium", "high")
        self.sev_text = tuple(severity_weights_text)
        self.sev_image = tuple(severity_weights_image)

    def __call__(
        self,
        pil_images: Sequence[Image.Image],
        raw_texts: Sequence[str],
        *,
        image_tfm: Callable,
        text_tfm: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, toks = [], []
        for pil, raw in zip(pil_images, raw_texts):
            kind = self.rng.choices(self.kinds, weights=self.kind_weights, k=1)[0]
            pil_p = self._image(pil) if kind in ("image", "both") else pil
            raw_p = self._text(raw) if kind in ("text", "both") else raw
            imgs.append(image_tfm(pil_p))
            toks.append(text_tfm(raw_p))
        return torch.stack(imgs), torch.stack(toks)

    def _seed(self) -> int:
        return self.rng.randrange(0, 2**31 - 1)

    def _text(self, raw: str) -> str:
        attack = self.rng.choice(self.text_attacks)
        level = self.rng.choices(self.levels, weights=self.sev_text, k=1)[0]
        return TextPerturbation.from_preset(attack, level, seed=self._seed())(raw)

    def _image(self, pil: Image.Image) -> Image.Image:
        attack = self.rng.choice(self.image_attacks)
        level = self.rng.choices(self.levels, weights=self.sev_image, k=1)[0]
        return ImagePerturbation.from_preset(attack, level, seed=self._seed())(pil)
