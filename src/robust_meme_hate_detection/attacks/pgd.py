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
    norm: str = 'linf',
) -> torch.Tensor:
    """Single-step Fast Gradient Sign Method attack on the image tensor."""
    if loss_fn is None:
        loss_fn = _bce_loss
    images01 = images01.clone().detach().requires_grad_(True)
    logit = model(images01, token_ids)
    loss = loss_fn(logit, label)
    grad = torch.autograd.grad(loss, images01)[0]
    if norm == 'linf':
        return (images01 + epsilon * grad.sign()).clamp(0, 1).detach()
    elif norm == 'l2':
        g_flat = grad.view(grad.size(0), -1)
        g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        g_unit = (g_flat / g_norm).view_as(grad)
        return (images01 + epsilon * g_unit).clamp(0, 1).detach()
    else:
        raise ValueError(f"Unsupported norm: {norm!r}")


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
    norm: str = 'linf',
) -> torch.Tensor:
    """Iterative Projected Gradient Descent attack on the image tensor.

    Supports `norm='linf'` (default) and `norm='l2'`.
    """
    if loss_fn is None:
        loss_fn = _bce_loss
    # initialise delta according to the chosen norm
    if random_start:
        if norm == 'linf':
            delta = torch.empty_like(images01).uniform_(-epsilon, epsilon)
        elif norm == 'l2':
            delta = torch.randn_like(images01)
            d_flat = delta.view(delta.size(0), -1)
            d_norm = torch.norm(d_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            r = torch.rand(d_flat.size(0), 1, device=delta.device)
            d_flat = (d_flat / d_norm) * (r * epsilon)
            delta = d_flat.view_as(images01)
        else:
            raise ValueError(f"Unsupported norm: {norm!r}")
    else:
        delta = torch.zeros_like(images01)

    delta = delta.detach().requires_grad_(True)

    for _ in range(steps):
        x_adv = (images01 + delta).clamp(0, 1)
        logit = model(x_adv, token_ids)
        loss = loss_fn(logit, label)
        g = torch.autograd.grad(loss, delta)[0]

        if norm == 'linf':
            delta = (delta + alpha * g.sign()).clamp(-epsilon, epsilon).detach().requires_grad_(True)
        elif norm == 'l2':
            g_flat = g.view(g.size(0), -1)
            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            step = (g_flat / g_norm) * alpha

            delta_flat = (delta.view(delta.size(0), -1) + step)
            delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            # project onto L2 ball of radius epsilon
            factor = (epsilon / delta_norm).clamp(max=1.0)
            delta_flat = delta_flat * factor
            delta = delta_flat.view_as(delta).detach().requires_grad_(True)
        else:
            raise ValueError(f"Unsupported norm: {norm!r}")

    return (images01 + delta).clamp(0, 1).detach()
