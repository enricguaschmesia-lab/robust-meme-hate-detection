"""Robust hateful-meme detection (EE-559 mini-project, Group 49).

A CLIP-fusion hateful-meme classifier and the tooling to audit and improve its
robustness under two threat models:

* **naturalistic** label-preserving edits (text + image perturbations), and
* **white-box** L-inf adversarial attacks (FGSM / PGD).

The public surface is intentionally small:

* :class:`~robust_meme_hate_detection.model.CLIPHateMemeClassifier` -- the model.
* :mod:`robust_meme_hate_detection.perturbations` -- the naturalistic suite.
* :mod:`robust_meme_hate_detection.attacks` -- FGSM / PGD on raw ``[0,1]`` pixels.
* :mod:`robust_meme_hate_detection.losses` -- the unified robust loss.
* the ``train`` / ``evaluate`` / ``demo`` entry points at the repo root.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
