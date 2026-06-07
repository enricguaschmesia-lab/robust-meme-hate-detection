"""Convert the MultiOFF dataset into the JSONL + img/ layout expected by
load_hateful_memes_records.

Input layout (--raw-dir should point at the MultiOFF_Dataset root):
  MultiOFF_Dataset/
    Split Dataset/
      Training_meme_dataset.csv
      Validation_meme_dataset.csv
      Testing_meme_dataset.csv
    Labelled Images/
      <image files>

Output layout (written to --out-dir, suitable as dataset_root in the config):
  <out-dir>/
    train.jsonl
    dev.jsonl
    test.jsonl
    img/
      <image files>   (symlinked by default, copied with --copy)

Usage:
  python scripts/prepare_multioff.py \\
      --raw-dir data/raw/MultiOFF_Dataset \\
      --out-dir data/prepared/multioff
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SPLIT_FILES = {
    "train": "Training_meme_dataset.csv",
    "dev": "Validation_meme_dataset.csv",
    "test": "Testing_meme_dataset.csv",
}


def label_to_int(raw: str) -> int:
    """Map MultiOFF label strings to 0/1."""
    normalized = raw.strip().lower()
    if normalized.startswith("non"):
        return 0
    if normalized == "offensive":
        return 1
    raise ValueError(f"Unrecognised label: {raw!r}")


def convert_split(
    csv_path: Path,
    images_dir: Path,
    out_jsonl: Path,
    img_out_dir: Path,
    copy: bool,
    missing_ok: bool,
) -> tuple[int, int]:
    """Write one JSONL split and link/copy its images. Returns (n_records, n_missing)."""
    records: list[dict] = []
    missing: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_name = row["image_name"].strip()
            text = row["sentence"].strip()
            label = label_to_int(row["label"])
            src = images_dir / image_name
            dst = img_out_dir / image_name

            if not src.exists():
                missing.append(image_name)
                if not missing_ok:
                    raise FileNotFoundError(f"Image not found: {src}")
            elif not dst.exists():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())

            # Use the stem (filename without extension) as the id
            record_id = Path(image_name).stem
            records.append({
                "id": record_id,
                "img": f"img/{image_name}",
                "label": label,
                "text": text,
            })

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records), len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MultiOFF for hateful-memes loader")
    parser.add_argument("--raw-dir", required=True, help="Path to MultiOFF_Dataset root")
    parser.add_argument("--out-dir", required=True, help="Output dataset_root directory")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument(
        "--missing-ok",
        action="store_true",
        help="Skip missing images instead of raising an error",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    images_dir = raw_dir / "Labelled Images"
    split_dir = raw_dir / "Split Dataset"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = out_dir / "img"
    img_out_dir.mkdir(exist_ok=True)

    for split, csv_name in SPLIT_FILES.items():
        csv_path = split_dir / csv_name
        if not csv_path.exists():
            print(f"[SKIP] {csv_name} not found, skipping split '{split}'")
            continue

        out_jsonl = out_dir / f"{split}.jsonl"
        n, n_missing = convert_split(
            csv_path=csv_path,
            images_dir=images_dir,
            out_jsonl=out_jsonl,
            img_out_dir=img_out_dir,
            copy=args.copy,
            missing_ok=args.missing_ok,
        )
        status = f"  ({n_missing} images missing)" if n_missing else ""
        print(f"[{split:5s}] {n:4d} records → {out_jsonl}{status}")

    print(f"\nDone. Set dataset_root: {out_dir}")


if __name__ == "__main__":
    main()
