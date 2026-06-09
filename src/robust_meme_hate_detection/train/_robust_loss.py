"""Phase 5: robust-training loss utilities.

Pure functions (no I/O, no globals) so the training loop can call them and
unit tests can exercise them on tiny CPU tensors. Two pieces:

* ``binary_kl(p, q, eps)`` — element-wise binary KL on probability tensors,
  returns a per-example scalar so the caller can choose pos-weighting /
  reduction.
* ``robust_loss(...)`` — assembles the four-term Phase 5 loss
  ``BCE(p_clean) + α·BCE(p_pert) + β·KL_full + γ·KL_image_branch`` and
  returns each component so the train loop can log them individually.

The clean-side target inside both KL terms is detached: the KL terms are
"pull the perturbed prediction toward the clean one", not the reverse,
so the clean branch must not be gradient-affected by the perturbed view.
This mirrors the design rationale in
``project_planning/Phase3_4_Completion_Report.md`` §6.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise binary KL divergence on probability tensors.

    Inputs are *probabilities* in ``(0, 1)`` (not logits). Returns a tensor
    of the same shape as ``p``/``q`` with the per-example KL contribution
    ``p·log(p/q) + (1-p)·log((1-p)/(1-q))``.

    Numerical safety: clamps both ``p`` and ``q`` to ``[eps, 1 - eps]``
    before the log to avoid ``log(0)`` / division-by-zero on saturated
    sigmoids.
    """
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
    """Compose the unified robust loss for naturalistic and/or adversarial views.

    ``L = w_clean · BCE(p_clean, y)
          + α · BCE(p_pert, y) + β · KL_full + γ · KL_image           (naturalistic)
          + δ · BCE(p_adv, y)  + ε · KL(p_clean ‖ p_adv)``            (adversarial)

    Every extra term collapses to zero when its weight is ``0`` or its logits
    are ``None``, so this single function serves naturalistic-only,
    adversarial-only, and both-enabled regimes:

    * Naturalistic terms (``ce_pert``, ``kl_full``, ``kl_image``) require
      ``logit_pert`` (and, for ``kl_image``, both image-only logits).
    * Adversarial terms (``ce_adv``, ``kl_adv``) require ``logit_adv``.
      ``ce_adv`` is the Madry-style cross-entropy on the PGD view; ``kl_adv``
      is the TRADES-style consistency term pulling the perturbed prediction
      toward the detached clean one.

    The clean-side target inside every KL term is detached: the KL terms pull
    the perturbed/adversarial prediction toward the clean one, never the
    reverse. Returns a dict with each component for per-step logging.
    """
    zero = logit_clean.new_zeros(())

    ce_clean = F.binary_cross_entropy_with_logits(
        logit_clean, labels, pos_weight=pos_weight
    )

    # ---------------------------------------------------------- naturalistic
    if logit_pert is not None:
        ce_pert = F.binary_cross_entropy_with_logits(
            logit_pert, labels, pos_weight=pos_weight
        )
    else:
        ce_pert = zero

    if beta > 0.0 and logit_pert is not None:
        p_clean = torch.sigmoid(logit_clean.float()).detach()
        q_pert = torch.sigmoid(logit_pert.float())
        kl_full = binary_kl(p_clean, q_pert).mean()
    else:
        kl_full = zero

    if (
        gamma > 0.0
        and image_logit_clean is not None
        and image_logit_pert is not None
    ):
        p_img_clean = torch.sigmoid(image_logit_clean.float()).detach()
        q_img_pert = torch.sigmoid(image_logit_pert.float())
        kl_image = binary_kl(p_img_clean, q_img_pert).mean()
    else:
        kl_image = zero

    # ----------------------------------------------------------- adversarial
    if logit_adv is not None:
        ce_adv = F.binary_cross_entropy_with_logits(
            logit_adv, labels, pos_weight=pos_weight
        )
    else:
        ce_adv = zero

    if epsilon_kl > 0.0 and logit_adv is not None:
        p_clean_adv = torch.sigmoid(logit_clean.float()).detach()
        q_adv = torch.sigmoid(logit_adv.float())
        kl_adv = binary_kl(p_clean_adv, q_adv).mean()
    else:
        kl_adv = zero

    total = (
        clean_weight * ce_clean
        + alpha * ce_pert
        + beta * kl_full
        + gamma * kl_image
        + delta * ce_adv
        + epsilon_kl * kl_adv
    )
    return {
        "total": total,
        "ce_clean": ce_clean,
        "ce_pert": ce_pert,
        "kl_full": kl_full,
        "kl_image": kl_image,
        "ce_adv": ce_adv,
        "kl_adv": kl_adv,
    }
