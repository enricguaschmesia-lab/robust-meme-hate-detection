"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python ``random``, NumPy, and PyTorch (if available).

    Returns the seed for chaining.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass
    return seed
