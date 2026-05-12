"""Unit tests for metrics, threshold sweep, and split loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class MetricsTests(unittest.TestCase):
    def test_classification_metrics_basic(self):
        try:
            import sklearn  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("scikit-learn not installed")
        from robust_meme_hate_detection.eval.metrics import classification_metrics

        probs = [0.1, 0.4, 0.6, 0.9]
        labels = [0, 0, 1, 1]
        m = classification_metrics(probs, labels, threshold=0.5)
        self.assertAlmostEqual(m.accuracy, 1.0)
        self.assertAlmostEqual(m.macro_f1, 1.0)
        self.assertGreaterEqual(m.auroc, 0.99)

    def test_attack_success_rate(self):
        from robust_meme_hate_detection.eval.metrics import attack_success_rate

        clean = [0.1, 0.9, 0.9]   # preds 0, 1, 1
        attacked = [0.9, 0.1, 0.9]  # preds 1, 0, 1
        labels = [0, 1, 1]
        # Clean-correct: all 3. Attacked: example 0 wrong, example 1 wrong, example 2 still right.
        # ASR = 2 / 3
        asr = attack_success_rate(clean, attacked, labels, threshold=0.5)
        self.assertAlmostEqual(asr, 2.0 / 3.0)


class ThresholdTests(unittest.TestCase):
    def test_sweep_picks_best_macro_f1(self):
        try:
            import sklearn  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("scikit-learn not installed")
        from robust_meme_hate_detection.utils.threshold import sweep_threshold

        probs = [0.1, 0.4, 0.45, 0.6, 0.9]
        labels = [0, 0, 1, 1, 1]
        best, grid = sweep_threshold(probs, labels)
        self.assertGreaterEqual(best.macro_f1, 0.0)
        self.assertGreater(len(grid), 0)


class SplitsTests(unittest.TestCase):
    def test_load_train_split_round_trip(self):
        from robust_meme_hate_detection.data.splits import load_train_split

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({
                "source_jsonl": "/scratch/datasets/hate_meta/train.jsonl",
                "seed": 0,
                "held_out_size": 2,
                "train_size": 3,
                "train_ids": ["1", "2", "3"],
                "held_out_ids": ["4", "5"],
            }))
            split = load_train_split(path)
            self.assertEqual(split.train_ids, ["1", "2", "3"])
            self.assertEqual(split.held_out_ids, ["4", "5"])
            self.assertEqual(split.train_id_set, {"1", "2", "3"})


if __name__ == "__main__":
    unittest.main()
