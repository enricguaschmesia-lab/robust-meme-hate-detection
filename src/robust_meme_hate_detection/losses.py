"""The unified robust-training loss.

One function assembles every term used in the report, and each term collapses to
zero when its weight is 0 or its logits are ``None`` -- so the same loss serves
the clean, naturalistic, adversarial and combined recipes selected by config::

    L = w_clean * BCE(p_clean, y)
        + alpha * BCE(p_pert, y) + beta * KL(p_clean || p_pert)
                                 + gamma * KL(p_clean_img || p_pert_img)   (naturalistic)
        + delta * BCE(p_adv, y)  + eps   * KL(p_clean || p_adv)            (adversarial)

The clean-side probability inside every KL term is **detached**: consistency
pulls the perturbed/adversarial prediction toward the clean one, never the
reverse.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise binary KL ``p*log(p/q) + (1-p)*log((1-p)/(1-q))`` on probabilities."""
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()


def robust_loss(
    *,
    logit_clean: torch.Tensor,
    labels: torch.Tensor,
    logit_pert: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    image_logit_clean: Optional[torch.Tensor] = None,
    image_logit_pert: Optional[torch.Tensor] = None,
    logit_adv: Optional[torch.Tensor] = None,
    delta: float = 0.0,
    epsilon_kl: float = 0.0,
    clean_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return a dict with ``total`` plus each component, for per-step logging."""
    zero = logit_clean.new_zeros(())

    def bce(logit):
        return F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_weight)

    def kl(detached_logit, live_logit):
        p = torch.sigmoid(detached_logit.float()).detach()
        q = torch.sigmoid(live_logit.float())
        return binary_kl(p, q).mean()

    ce_clean = bce(logit_clean)
    ce_pert = bce(logit_pert) if logit_pert is not None else zero
    kl_full = kl(logit_clean, logit_pert) if (beta > 0.0 and logit_pert is not None) else zero
    kl_image = (
        kl(image_logit_clean, image_logit_pert)
        if (gamma > 0.0 and image_logit_clean is not None and image_logit_pert is not None)
        else zero
    )
    ce_adv = bce(logit_adv) if logit_adv is not None else zero
    kl_adv = kl(logit_clean, logit_adv) if (epsilon_kl > 0.0 and logit_adv is not None) else zero

    total = (
        clean_weight * ce_clean
        + alpha * ce_pert + beta * kl_full + gamma * kl_image
        + delta * ce_adv + epsilon_kl * kl_adv
    )
    return {
        "total": total, "ce_clean": ce_clean, "ce_pert": ce_pert,
        "kl_full": kl_full, "kl_image": kl_image, "ce_adv": ce_adv, "kl_adv": kl_adv,
    }
