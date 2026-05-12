"""Synthetic-tensor smoke test for the CLIP fusion model and PGD attack.

Runs entirely on CPU with random weights — no dataset, no network. Verifies:

1. The model can be instantiated.
2. Forward + backward through the fusion head works.
3. PGD on the [0, 1] image tensor produces a non-zero gradient and a valid
   adversarial image (still in [0, 1]).
"""

from __future__ import annotations

import sys

import torch

from robust_meme_hate_detection.attacks.pgd import pgd_image
from robust_meme_hate_detection.models.clip_fusion import (
    CLIPHateMemeClassifier,
    build_random_clip_for_testing,
)


def main() -> int:
    torch.manual_seed(0)

    clip = build_random_clip_for_testing(arch="ViT-B-32")
    model = CLIPHateMemeClassifier(
        arch="ViT-B-32", pretrained=None, freeze_encoders=True, clip_model=clip
    )
    model.eval()

    batch = 4
    images01 = torch.rand(batch, 3, 224, 224)
    token_ids = torch.randint(low=0, high=49000, size=(batch, 77), dtype=torch.long)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    # Forward
    logit = model(images01, token_ids)
    assert logit.shape == (batch,), f"unexpected logit shape: {logit.shape}"

    # Backward through the head only
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels)
    loss.backward()
    head_grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.head.parameters()]
    assert any(head_grads), "Head parameters did not receive gradients"

    # PGD: gradient must flow back to the image tensor.
    model.zero_grad()
    images01_adv = pgd_image(model, images01, token_ids, labels, epsilon=4 / 255, alpha=1 / 255, steps=2)
    assert images01_adv.shape == images01.shape
    assert images01_adv.min() >= 0.0 and images01_adv.max() <= 1.0
    delta = (images01_adv - images01).abs()
    assert delta.max().item() > 0.0, "PGD did not perturb the input — gradient path is broken"

    print(
        "smoke_synthetic OK: forward, backward, and PGD gradient flow all verified.",
        f"max delta={delta.max().item():.5f}, mean delta={delta.mean().item():.5f}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
