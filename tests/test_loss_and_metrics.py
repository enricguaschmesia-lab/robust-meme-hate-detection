"""Tests for the unified robust loss, metrics, and the augmenter shape contract."""

from __future__ import annotations

import torch
from PIL import Image

from robust_meme_hate_detection.augment import RobustAugmenter
from robust_meme_hate_detection.losses import robust_loss
from robust_meme_hate_detection.metrics import attack_success_rate, classification_metrics


def _logits():
    torch.manual_seed(0)
    return (
        torch.randn(8, requires_grad=True),
        torch.randn(8, requires_grad=True),
        torch.randn(8, requires_grad=True),
        torch.randint(0, 2, (8,)).float(),
    )


def test_clean_only_collapses_other_terms():
    lc, _, _, y = _logits()
    parts = robust_loss(logit_clean=lc, labels=y)
    for k in ("ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv"):
        assert float(parts[k]) == 0.0
    assert torch.allclose(parts["total"], parts["ce_clean"])


def test_all_terms_active_when_provided():
    lc, lp, la, y = _logits()
    parts = robust_loss(
        logit_clean=lc, labels=y, logit_pert=lp, alpha=1.0, beta=0.5, gamma=0.25,
        image_logit_clean=lc, image_logit_pert=lp, logit_adv=la, delta=1.0, epsilon_kl=0.5,
    )
    for k in ("ce_clean", "ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv"):
        assert float(parts[k].detach()) > 0.0
    parts["total"].backward()  # loss is differentiable


def test_classification_metrics_and_asr():
    m = classification_metrics([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    assert m.accuracy == 1.0 and m.auroc >= 0.99
    # All clean-correct; attack flips 2 of 3 -> ASR = 2/3.
    asr = attack_success_rate([0.1, 0.9, 0.9], [0.9, 0.1, 0.9], [0, 1, 1])
    assert abs(asr - 2 / 3) < 1e-6


def test_augmenter_returns_batch_shaped_views():
    images = [Image.new("RGB", (224, 224), (i * 20, 100, 150)) for i in range(3)]
    texts = ["a hateful looking caption", "benign text here", "another one"]

    def image_tfm(im):
        return torch.from_numpy(__import__("numpy").asarray(im, dtype="float32")).permute(2, 0, 1) / 255.0

    def text_tfm(_t):
        return torch.zeros(77, dtype=torch.long)

    aug = RobustAugmenter(seed=0)
    imgs, toks = aug(images, texts, image_tfm=image_tfm, text_tfm=text_tfm)
    assert imgs.shape == (3, 3, 224, 224) and toks.shape == (3, 77)
