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
    logit_pert: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    image_logit_clean: Optional[torch.Tensor] = None,
    image_logit_pert: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compose the Phase-5 robust loss.

    ``L = BCE(p_clean, y) + α · BCE(p_pert, y) + β · KL_full + γ · KL_image``

    ``KL_full`` is the binary KL from ``detach(sigmoid(logit_clean))`` to
    ``sigmoid(logit_pert)``. ``KL_image`` is the analogous term computed on
    the image-only branch logits, when both are supplied; if either is
    ``None`` the term collapses to zero and ``γ`` is ignored.

    Returns a dict with keys ``total``, ``ce_clean``, ``ce_pert``,
    ``kl_full``, ``kl_image`` so the train loop can log each term.
    """
    ce_clean = F.binary_cross_entropy_with_logits(
        logit_clean, labels, pos_weight=pos_weight
    )
    ce_pert = F.binary_cross_entropy_with_logits(
        logit_pert, labels, pos_weight=pos_weight
    )

    if beta > 0.0:
        p_clean = torch.sigmoid(logit_clean.float()).detach()
        q_pert = torch.sigmoid(logit_pert.float())
        kl_full = binary_kl(p_clean, q_pert).mean()
    else:
        kl_full = logit_clean.new_zeros(())

    if (
        gamma > 0.0
        and image_logit_clean is not None
        and image_logit_pert is not None
    ):
        p_img_clean = torch.sigmoid(image_logit_clean.float()).detach()
        q_img_pert = torch.sigmoid(image_logit_pert.float())
        kl_image = binary_kl(p_img_clean, q_img_pert).mean()
    else:
        kl_image = logit_clean.new_zeros(())

    total = ce_clean + alpha * ce_pert + beta * kl_full + gamma * kl_image
    return {
        "total": total,
        "ce_clean": ce_clean,
        "ce_pert": ce_pert,
        "kl_full": kl_full,
        "kl_image": kl_image,
    }
