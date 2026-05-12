"""Adversarial perturbation module for robustness evaluation.

Public API:
    TextPerturbation, ImagePerturbation, ComposePerturbation: callable
        single-perturbation / pipeline classes.
    TEXT_MODES, IMAGE_MODES: the supported mode names per modality.
    TEXT_SEVERITY_PRESETS, IMAGE_SEVERITY_PRESETS: per-mode severity
        presets keyed by 'low' | 'medium' | 'high'.
"""

from robust_meme_hate_detection.perturbations.compose import ComposePerturbation
from robust_meme_hate_detection.perturbations.image_perturbations import (
    IMAGE_MODES,
    IMAGE_MODES_BENCHMARK,
    ImagePerturbation,
)
from robust_meme_hate_detection.perturbations.image_perturbations import (
    SEVERITY_PRESETS as IMAGE_SEVERITY_PRESETS,
)
from robust_meme_hate_detection.perturbations.text_perturbations import (
    TEXT_MODES,
    TextPerturbation,
)
from robust_meme_hate_detection.perturbations.text_perturbations import (
    SEVERITY_PRESETS as TEXT_SEVERITY_PRESETS,
)

__all__ = [
    'TextPerturbation',
    'ImagePerturbation',
    'ComposePerturbation',
    'TEXT_MODES',
    'IMAGE_MODES',
    'IMAGE_MODES_BENCHMARK',
    'TEXT_SEVERITY_PRESETS',
    'IMAGE_SEVERITY_PRESETS',
]
