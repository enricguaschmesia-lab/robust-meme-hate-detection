"""FGSM and PGD attacks on the [0, 1] image tensor.

The attacks rely on the model's contract that ``forward(images01, token_ids)``
is differentiable through the image branch (Normalize lives inside the model;
see model_architecture.md §4.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _bce_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label.float())


def fgsm_image(
    model: torch.nn.Module,
    images01: torch.Tensor,
    token_ids: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    loss_fn=None,
) -> torch.Tensor:
    """Single-step Fast Gradient Sign Method attack on the image tensor."""
    if loss_fn is None:
        loss_fn = _bce_loss
    images01 = images01.clone().detach().requires_grad_(True)
    logit = model(images01, token_ids)
    loss = loss_fn(logit, label)
    grad = torch.autograd.grad(loss, images01)[0]
    return (images01 + epsilon * grad.sign()).clamp(0, 1).detach()


def pgd_image(
    model: torch.nn.Module,
    images01: torch.Tensor,
    token_ids: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    loss_fn=None,
    random_start: bool = True,
) -> torch.Tensor:
    """Iterative Projected Gradient Descent (L∞) attack on the image tensor."""
    if loss_fn is None:
        loss_fn = _bce_loss
    if random_start:
        delta = torch.empty_like(images01).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(images01)
    delta = delta.detach().requires_grad_(True)

    for _ in range(steps):
        x_adv = (images01 + delta).clamp(0, 1)
        logit = model(x_adv, token_ids)
        loss = loss_fn(logit, label)
        g = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * g.sign()).clamp(-epsilon, epsilon).detach().requires_grad_(True)

    return (images01 + delta).clamp(0, 1).detach()
