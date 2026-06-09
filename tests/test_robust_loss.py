"""Unit tests for the unified robust loss (CPU, no network, no dataset).

Verifies that every term gates to zero when its weight / logits are absent, so
the single ``robust_loss`` serves naturalistic-only, adversarial-only, and
both-enabled regimes.
"""

from __future__ import annotations

import unittest


class RobustLossTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

    def _logits(self):
        import torch

        torch.manual_seed(0)
        logit_clean = torch.randn(8, requires_grad=True)
        logit_pert = torch.randn(8, requires_grad=True)
        logit_adv = torch.randn(8, requires_grad=True)
        labels = torch.randint(0, 2, (8,)).float()
        return logit_clean, logit_pert, logit_adv, labels

    def test_clean_only(self):
        import torch

        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, _, _, y = self._logits()
        parts = robust_loss(logit_clean=lc, labels=y)
        # No perturbed/adversarial logits -> only the clean term is active.
        for k in ("ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv"):
            self.assertEqual(float(parts[k]), 0.0, msg=k)
        self.assertAlmostEqual(float(parts["total"]), float(parts["ce_clean"]), places=5)

    def test_clean_weight_scales(self):
        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, _, _, y = self._logits()
        base = robust_loss(logit_clean=lc, labels=y, clean_weight=1.0)
        scaled = robust_loss(logit_clean=lc, labels=y, clean_weight=2.0)
        self.assertAlmostEqual(float(scaled["total"]), 2.0 * float(base["total"]), places=5)

    def test_naturalistic_terms_gate(self):
        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, lp, _, y = self._logits()
        # beta=0 -> kl_full off; alpha>0 -> ce_pert on.
        parts = robust_loss(logit_clean=lc, labels=y, logit_pert=lp, alpha=1.0, beta=0.0, gamma=0.0)
        self.assertGreater(float(parts["ce_pert"]), 0.0)
        self.assertEqual(float(parts["kl_full"]), 0.0)
        self.assertEqual(float(parts["ce_adv"]), 0.0)
        # beta>0 -> kl_full active and non-negative.
        parts2 = robust_loss(logit_clean=lc, labels=y, logit_pert=lp, alpha=1.0, beta=0.5)
        self.assertGreaterEqual(float(parts2["kl_full"]), 0.0)
        self.assertGreater(float(parts2["kl_full"]), 0.0)

    def test_adversarial_terms_gate(self):
        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, _, la, y = self._logits()
        # delta>0 -> ce_adv on; epsilon_kl=0 -> kl_adv off; naturalistic terms off.
        parts = robust_loss(logit_clean=lc, labels=y, logit_adv=la, delta=1.0, epsilon_kl=0.0)
        self.assertGreater(float(parts["ce_adv"]), 0.0)
        self.assertEqual(float(parts["kl_adv"]), 0.0)
        self.assertEqual(float(parts["ce_pert"]), 0.0)
        # epsilon_kl>0 -> TRADES KL term active.
        parts2 = robust_loss(logit_clean=lc, labels=y, logit_adv=la, delta=1.0, epsilon_kl=1.0)
        self.assertGreater(float(parts2["kl_adv"]), 0.0)

    def test_both_modes_compose(self):
        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, lp, la, y = self._logits()
        parts = robust_loss(
            logit_clean=lc, labels=y,
            logit_pert=lp, alpha=1.0, beta=0.5, gamma=0.0,
            logit_adv=la, delta=1.0, epsilon_kl=0.5,
        )
        expected = (
            float(parts["ce_clean"]) + 1.0 * float(parts["ce_pert"]) + 0.5 * float(parts["kl_full"])
            + 1.0 * float(parts["ce_adv"]) + 0.5 * float(parts["kl_adv"])
        )
        self.assertAlmostEqual(float(parts["total"]), expected, places=5)

    def test_kl_zero_for_identical_logits(self):
        import torch

        from robust_meme_hate_detection.train._robust_loss import binary_kl

        p = torch.rand(16).clamp(0.05, 0.95)
        kl = binary_kl(p, p)
        self.assertLess(float(kl.abs().max()), 1e-5)

    def test_total_is_differentiable(self):
        import torch

        from robust_meme_hate_detection.train._robust_loss import robust_loss

        lc, lp, la, y = self._logits()
        parts = robust_loss(
            logit_clean=lc, labels=y,
            logit_pert=lp, alpha=1.0, beta=0.5,
            logit_adv=la, delta=1.0, epsilon_kl=0.5,
        )
        parts["total"].backward()
        self.assertIsNotNone(lc.grad)
        self.assertGreater(float(lc.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
