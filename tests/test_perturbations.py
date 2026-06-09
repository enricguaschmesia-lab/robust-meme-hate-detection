"""Perturbation-suite tests: determinism, presets, and shape preservation."""

from __future__ import annotations

from PIL import Image

from robust_meme_hate_detection.perturbations import (
    IMAGE_MODES,
    TEXT_MODES,
    Compose,
    ImagePerturbation,
    TextPerturbation,
)

CAPTION = "the quick brown fox jumps over the lazy dog 1234"


def test_every_text_mode_has_three_presets_and_runs():
    for mode in TEXT_MODES:
        for level in ("low", "medium", "high"):
            out = TextPerturbation.from_preset(mode, level, seed=0)(CAPTION)
            assert isinstance(out, str)


def test_text_perturbation_is_seed_deterministic():
    a = TextPerturbation.from_preset("leetspeak", "high", seed=7)(CAPTION)
    b = TextPerturbation.from_preset("leetspeak", "high", seed=7)(CAPTION)
    assert a == b and a != CAPTION


def test_every_image_mode_preserves_size_and_mode():
    img = Image.new("RGB", (224, 224), (120, 130, 140))
    for mode in IMAGE_MODES:
        out = ImagePerturbation.from_preset(mode, "medium", seed=0)(img)
        assert out.size == (224, 224) and out.mode == "RGB"


def test_image_perturbation_is_seed_deterministic():
    img = Image.new("RGB", (64, 64), (10, 200, 50))
    import numpy as np

    a = ImagePerturbation.from_preset("gaussian_noise", "high", seed=3)(img)
    b = ImagePerturbation.from_preset("gaussian_noise", "high", seed=3)(img)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_compose_stacks_text_edits():
    composed = Compose([
        TextPerturbation.from_preset("leetspeak", "medium", seed=1),
        TextPerturbation.from_preset("char_deletion", "medium", seed=2),
    ])
    assert isinstance(composed(CAPTION), str)
