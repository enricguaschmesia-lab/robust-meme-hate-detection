"""Synthetic integration test for the unified trainer's per-batch step.

Exercises the real moving parts of ``train.stage1_robust_unified`` — the
``RobustAugmenter`` naturalistic view, the PGD adversarial view, the shared
modality-dropout fusion, and the composite ``robust_loss`` + backward — on a
random-weight model and tiny synthetic PIL images. No dataset, no network
download (uses the random CLIP builder), CPU only.
"""

from __future__ import annotations

import unittest


class UnifiedStepTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

    def _build(self):
        import torch
        from PIL import Image

        from robust_meme_hate_detection.data.transforms import (
            ClipImage01Transform,
            ClipTokenize,
        )
        from robust_meme_hate_detection.models.clip_fusion import (
            CLIPHateMemeClassifier,
            build_random_clip_for_testing,
        )

        torch.manual_seed(0)
        clip = build_random_clip_for_testing(arch="ViT-B-32")
        model = CLIPHateMemeClassifier(
            arch="ViT-B-32", pretrained=None, freeze_encoders=True, clip_model=clip
        )
        image_tfm = ClipImage01Transform(size=224)
        text_tfm = ClipTokenize(arch="ViT-B-32")

        pil_images = [Image.new("RGB", (64, 48), color=(i * 30 % 255, 100, 150)) for i in range(2)]
        raw_texts = ["a caption about something", "another meme caption here"]
        images = torch.stack([image_tfm(p) for p in pil_images], dim=0)
        tokens = torch.stack([text_tfm(t) for t in raw_texts], dim=0)
        labels = torch.tensor([0.0, 1.0])
        return model, image_tfm, text_tfm, pil_images, raw_texts, images, tokens, labels

    def test_both_mode_step_runs_and_backprops(self):
        import torch

        from robust_meme_hate_detection.attacks.pgd import pgd_image
        from robust_meme_hate_detection.train._robust_augmenter import RobustAugmenter
        from robust_meme_hate_detection.train._robust_loss import robust_loss
        from robust_meme_hate_detection.train.stage1_robust import _fuse_with_dropout

        (model, image_tfm, text_tfm, pil_images, raw_texts,
         images, tokens, labels) = self._build()

        augmenter = RobustAugmenter(seed=0)

        # Naturalistic view.
        pert_img, pert_tok = augmenter(pil_images, raw_texts, image_tfm=image_tfm, text_tfm=text_tfm)
        self.assertEqual(pert_img.shape, images.shape)
        self.assertEqual(pert_tok.shape, tokens.shape)

        # Adversarial view (PGD on pixels).
        adv = pgd_image(model, images, tokens, labels, epsilon=8 / 255, alpha=2 / 255, steps=2)
        self.assertEqual(adv.shape, images.shape)
        self.assertTrue(torch.all(adv >= 0) and torch.all(adv <= 1))

        # Shared modality-dropout masks.
        drop_text = (torch.rand(2) < 0.3).view(-1, 1)
        drop_image = None

        logit_clean = _fuse_with_dropout(model, images, tokens, drop_text, drop_image)
        logit_pert = _fuse_with_dropout(model, pert_img, pert_tok, drop_text, drop_image)
        logit_adv = _fuse_with_dropout(model, adv, tokens, drop_text, drop_image)
        image_logit_clean = model.forward_image_only(images)
        image_logit_pert = model.forward_image_only(pert_img)

        parts = robust_loss(
            logit_clean=logit_clean, labels=labels,
            logit_pert=logit_pert, alpha=1.0, beta=0.5, gamma=0.25,
            image_logit_clean=image_logit_clean, image_logit_pert=image_logit_pert,
            logit_adv=logit_adv, delta=1.0, epsilon_kl=0.0, clean_weight=1.0,
        )
        self.assertTrue(torch.isfinite(parts["total"]))
        self.assertGreater(float(parts["ce_pert"]), 0.0)
        self.assertGreater(float(parts["ce_adv"]), 0.0)

        parts["total"].backward()
        head_grad = sum(
            float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None
        )
        self.assertGreater(head_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
