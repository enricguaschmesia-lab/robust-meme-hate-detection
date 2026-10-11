"""Composition utilities for stacking multiple perturbations."""

from __future__ import annotations

from typing import Callable


class ComposePerturbation:
    """Stack multiple perturbations and apply them sequentially.
    
    Each perturbation in the list is applied independently with its own
    probability, allowing for diverse combinations of transformations.
    """

    def __init__(self, perturbations: list[Callable]) -> None:
        """
        Args:
            perturbations: List of callable perturbation objects (TextPerturbation,
                ImagePerturbation, etc.)
        """
        self.perturbations = perturbations

    def __call__(self, value):
        """Apply all perturbations sequentially."""
        for perturbation in self.perturbations:
            value = perturbation(value)
        return value
