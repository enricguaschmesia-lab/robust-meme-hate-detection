"""White-box image attacks (FGSM, PGD)."""

from .pgd import fgsm_image, pgd_image

__all__ = ["fgsm_image", "pgd_image"]
