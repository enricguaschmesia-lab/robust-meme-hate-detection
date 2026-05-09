"""Adversarial perturbation module for robustness evaluation.

Classes:
    TextPerturbation: Apply text-based adversarial perturbations.
    ImagePerturbation: Apply image-based adversarial perturbations.
    ComposePerturbation: Stack multiple perturbations together.
"""

from robust_meme_hate_detection.perturbations.compose import ComposePerturbation
from robust_meme_hate_detection.perturbations.image_perturbations import ImagePerturbation
from robust_meme_hate_detection.perturbations.text_perturbations import TextPerturbation

__all__ = ['TextPerturbation', 'ImagePerturbation', 'ComposePerturbation']
