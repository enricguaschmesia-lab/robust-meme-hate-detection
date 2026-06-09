"""White-box L-inf image attacks: FGSM and PGD on the raw ``[0, 1]`` tensor.

Both rely on the model contract that ``forward(images01, token_ids)`` is
differentiable w.r.t. the pixel tensor (normalisation lives inside the model).
The caption (``token_ids``) is held fixed; only pixels are attacked. These are
used both to *evaluate* worst-case robustness and, inside the trainer, to
generate the PGD view for Madry-style adversarial training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label.float())


def fgsm_image(
    model: torch.nn.Module,
    images01: torch.Tensor,
    token_ids: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Single-step Fast Gradient Sign Method on the image tensor."""
    images01 = images01.clone().detach().requires_grad_(True)
    grad = torch.autograd.grad(_bce(model(images01, token_ids), label), images01)[0]
    return (images01 + epsilon * grad.sign()).clamp(0, 1).detach()


def pgd_image(
    model: torch.nn.Module,
    images01: torch.Tensor,
    token_ids: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    *,
    random_start: bool = True,
) -> torch.Tensor:
    """Iterative L-inf Projected Gradient Descent on the image tensor."""
    if random_start:
        delta = torch.empty_like(images01).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(images01)
    delta = delta.detach().requires_grad_(True)

    for _ in range(steps):
        loss = _bce(model((images01 + delta).clamp(0, 1), token_ids), label)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon).detach().requires_grad_(True)

    return (images01 + delta).clamp(0, 1).detach()
