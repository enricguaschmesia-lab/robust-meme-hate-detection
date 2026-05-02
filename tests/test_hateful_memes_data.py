from pathlib import Path
import unittest

from PIL import Image

from robust_meme_hate_detection.data.hateful_memes import (
    HatefulMemesDataset,
    compute_split_stats,
    load_hateful_memes_records,
)
from robust_meme_hate_detection.data.transforms import Compose, NormalizeTensor, ResizeToTensor


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "data"


@unittest.skipUnless(DATA_ROOT.exists(), "Hateful Memes data not found under data/raw/data")
class HatefulMemesDataTests(unittest.TestCase):
    def test_split_stats_match_expected_official_counts(self):
        stats = compute_split_stats(DATA_ROOT, splits=("train", "dev", "test"))

        self.assertEqual(stats["train"]["num_examples"], 8500)
        self.assertEqual(stats["dev"]["num_examples"], 500)
        self.assertEqual(stats["test"]["num_examples"], 1000)
        self.assertEqual(stats["train"]["missing_images"], 0)
        self.assertEqual(stats["dev"]["missing_images"], 0)
        self.assertEqual(stats["test"]["missing_images"], 0)

    def test_records_have_existing_images_and_labels(self):
        records = load_hateful_memes_records(DATA_ROOT, "train")

        self.assertGreater(len(records), 0)
        first = records[0]
        self.assertIn(first.label, (0, 1))
        self.assertTrue(first.text)
        self.assertTrue(first.image_path.exists())

    def test_dataset_returns_pil_image_text_label_and_metadata(self):
        dataset = HatefulMemesDataset(DATA_ROOT, split="dev")
        sample = dataset[0]

        self.assertIsInstance(sample["image"], Image.Image)
        self.assertIsInstance(sample["text"], str)
        self.assertIn(sample["label"], (0, 1))
        self.assertEqual(sample["split"], "dev")
        self.assertTrue(Path(sample["image_path"]).exists())

    def test_dataloader_builds_tensor_batch_when_torch_is_available(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in this local environment")

        from robust_meme_hate_detection.data.hateful_memes import build_hateful_memes_dataloader

        transform = Compose([ResizeToTensor(224), NormalizeTensor()])
        loader = build_hateful_memes_dataloader(
            DATA_ROOT,
            split="train",
            batch_size=4,
            image_transform=transform,
            num_workers=0,
        )
        batch = next(iter(loader))

        self.assertEqual(tuple(batch["image"].shape), (4, 3, 224, 224))
        self.assertEqual(len(batch["text"]), 4)
        self.assertEqual(batch["label"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
