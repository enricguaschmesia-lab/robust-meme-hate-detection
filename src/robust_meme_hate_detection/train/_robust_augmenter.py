"""Phase 5: per-batch on-the-fly perturbation builder for robust training.

For each example in a batch, the augmenter independently samples a
perturbation type from ``{text, image, both}`` (configurable weights),
applies it to the *raw* PIL image / caption string, and runs the result
through the given ``image_tfm`` / ``text_tfm`` so the perturbed view is
shape-compatible with the clean batch.

Determinism: a single ``random.Random`` is seeded at construction. Each
per-example perturbation draws a fresh integer seed from this RNG and
passes it to ``TextPerturbation`` / ``ImagePerturbation`` (which create
their own seeded RNGs), so runs are reproducible across resumes.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import torch
from PIL import Image

from robust_meme_hate_detection.perturbations.image_perturbations import (
    ImagePerturbation,
)
from robust_meme_hate_detection.perturbations.text_perturbations import (
    TextPerturbation,
)

PerturbationKind = str  # 'text' | 'image' | 'both'
SeverityLevel = str     # 'low' | 'medium' | 'high'


class RobustAugmenter:
    """Build a perturbed view of a batch.

    Parameters mirror the ``robust:`` config block in
    ``configs/stage1_robust_*.yaml``.
    """

    def __init__(
        self,
        *,
        seed: int,
        text_fraction: float = 0.5,
        image_fraction: float = 0.3,
        both_fraction: float = 0.2,
        text_attacks: Sequence[str] = (
            "leetspeak", "char_deletion", "char_swap", "censoring", "keyboard_typo",
        ),
        image_attacks: Sequence[str] = (
            "gaussian_noise", "blur", "compression",
            "brightness_down", "occlusion", "typographic",
        ),
        severity_weights_text: Sequence[float] = (1.0, 2.0, 2.0),
        severity_weights_image: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        self._rng = random.Random(seed)
        s = float(text_fraction + image_fraction + both_fraction)
        if not (0.999 < s < 1.001):
            raise ValueError(
                f"text/image/both fractions must sum to 1, got {s:.4f}"
            )
        self.kinds = ("text", "image", "both")
        self.kind_weights = (float(text_fraction), float(image_fraction), float(both_fraction))
        self.text_attacks = tuple(text_attacks)
        self.image_attacks = tuple(image_attacks)
        self.severity_levels = ("low", "medium", "high")
        if len(severity_weights_text) != 3 or len(severity_weights_image) != 3:
            raise ValueError("severity_weights_{text,image} must have length 3")
        self.severity_weights_text = tuple(float(x) for x in severity_weights_text)
        self.severity_weights_image = tuple(float(x) for x in severity_weights_image)

    # -------------------------------------------------------------- main api
    def __call__(
        self,
        pil_images: Sequence[Image.Image],
        raw_texts: Sequence[str],
        *,
        image_tfm: Callable[[Image.Image], torch.Tensor],
        text_tfm: Callable[[str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(pil_images) != len(raw_texts):
            raise ValueError(
                f"pil_images ({len(pil_images)}) and raw_texts ({len(raw_texts)}) "
                "must have the same length"
            )
        pert_imgs: list[torch.Tensor] = []
        pert_toks: list[torch.Tensor] = []
        for pil, raw in zip(pil_images, raw_texts):
            kind: PerturbationKind = self._rng.choices(self.kinds, weights=self.kind_weights, k=1)[0]
            pil_p, raw_p = self._apply(kind, pil, raw)
            pert_imgs.append(image_tfm(pil_p))
            pert_toks.append(text_tfm(raw_p))
        return torch.stack(pert_imgs, dim=0), torch.stack(pert_toks, dim=0)

    # ----------------------------------------------------------- private ops
    def _draw_seed(self) -> int:
        return self._rng.randrange(0, 2**31 - 1)

    def _draw_severity(self, weights: Sequence[float]) -> SeverityLevel:
        return self._rng.choices(self.severity_levels, weights=weights, k=1)[0]

    def _apply(
        self, kind: PerturbationKind, pil: Image.Image, raw: str,
    ) -> tuple[Image.Image, str]:
        if kind == "text":
            return pil, self._apply_text(raw)
        if kind == "image":
            return self._apply_image(pil), raw
        if kind == "both":
            return self._apply_image(pil), self._apply_text(raw)
        raise ValueError(f"Unknown perturbation kind: {kind!r}")

    def _apply_text(self, raw: str) -> str:
        attack = self._rng.choice(self.text_attacks)
        level = self._draw_severity(self.severity_weights_text)
        pert = TextPerturbation.from_preset(attack, level, seed=self._draw_seed())
        return pert(raw)

    def _apply_image(self, pil: Image.Image) -> Image.Image:
        attack = self._rng.choice(self.image_attacks)
        level = self._draw_severity(self.severity_weights_image)
        pert = ImagePerturbation.from_preset(attack, level, seed=self._draw_seed())
        return pert(pil)
