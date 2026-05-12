"""Unit tests for the CLIP fusion model wiring (no network, no dataset)."""

from __future__ import annotations

import unittest


class FusionHeadTests(unittest.TestCase):
    def test_concat_and_logit_shape(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        from robust_meme_hate_detection.models.clip_fusion import FusionHead

        head = FusionHead(dim=8, hidden=16, dropout=0.0)
        t = torch.randn(3, 8)
        v = torch.randn(3, 8)
        logit = head(t, v)
        self.assertEqual(logit.shape, (3,))

    def test_normalize_buffer_shapes(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        from robust_meme_hate_detection.models.clip_fusion import Normalize

        n = Normalize()
        x = torch.rand(2, 3, 4, 4)
        y = n(x)
        self.assertEqual(y.shape, x.shape)
        # mean/std are registered buffers, so .device should follow .to(...)
        self.assertEqual(n.mean.shape, (1, 3, 1, 1))


class CLIPHateMemeClassifierTests(unittest.TestCase):
    def test_forward_with_minimal_clip_stub(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        from robust_meme_hate_detection.models.clip_fusion import (
            CLIPHateMemeClassifier,
            _MinimalClipStub,
        )

        stub = _MinimalClipStub(embed_dim=64)
        model = CLIPHateMemeClassifier(arch="stub", pretrained=None, clip_model=stub)
        images01 = torch.rand(2, 3, 224, 224)
        token_ids = torch.randint(0, 49000, (2, 77), dtype=torch.long)

        logit = model(images01, token_ids)
        self.assertEqual(logit.shape, (2,))

        # Gradient must flow back to the image tensor (this is the whole point of
        # keeping Normalize inside the model).
        images01.requires_grad_(True)
        logit = model(images01, token_ids)
        logit.sum().backward()
        self.assertIsNotNone(images01.grad)
        self.assertGreater(images01.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
