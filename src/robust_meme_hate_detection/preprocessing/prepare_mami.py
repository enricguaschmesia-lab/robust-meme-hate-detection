"""Convert the MAMI dataset into the JSONL + img/ layout expected by
load_hateful_memes_records.

Expected raw layout (--raw-dir should point at the "MAMI DATASET" root):
  MAMI DATASET/
    TRAINING/
      training.csv          (TSV with BOM; columns: file_name, misogynous,
                             shaming, stereotype, objectification, violence,
                             Text Transcription)
      1.jpg, 2.jpg, ...
    test/
      Test.csv              (TSV; columns: file_name, Text Transcription)
      15001.jpg, ...
    test_labels.txt         (TSV, no header; columns: file_name, misogynous,
                             shaming, stereotype, objectification, violence)
    Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/
      trial.csv             (TSV with BOM; same columns as training)
      28.jpg, 104.jpg, ...

Output layout (--out-dir becomes dataset_root in the config):
  <out-dir>/
    train.jsonl
    dev.jsonl              (converted from trial)
    test.jsonl
    img/
      <all image files>    (symlinked by default, copied with --copy)

Label rule: label=1 if any of misogynous/shaming/stereotype/objectification/
violence equals 1, else label=0.

Usage:
  python -m robust_meme_hate_detection.data.prepare_mami \\
      --raw-dir data/raw/MAMI\\ DATASET \\
      --out-dir data/prepared/mami
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

LABEL_COLS = ["misogynous", "shaming", "stereotype", "objectification", "violence"]

SPLITS = {
    "train": {
        "tsv": ("TRAINING", "training.csv"),
        "images": ("TRAINING",),
        "has_labels": True,
    },
    "dev": {
        "tsv": ("Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET", "trial.csv"),
        "images": ("Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET"),
        "has_labels": True,
    },
    "test": {
        "tsv": ("test", "Test.csv"),
        "images": ("test",),
        "has_labels": False,  # labels come from test_labels.txt
    },
}


def _any_positive(row: dict[str, str]) -> int:
    return int(any(row.get(col, "0").strip() == "1" for col in LABEL_COLS))


def _load_test_labels(raw_dir: Path) -> dict[str, int]:
    """Parse test_labels.txt (no header, tab-separated)."""
    labels: dict[str, int] = {}
    label_file = raw_dir / "test_labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"test_labels.txt not found at {label_file}")
    with label_file.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            file_name = parts[0].strip()
            fake_row = dict(zip(LABEL_COLS, parts[1:6]))
            labels[file_name] = _any_positive(fake_row)
    return labels


def convert_split(
    split: str,
    raw_dir: Path,
    img_out_dir: Path,
    copy: bool,
    missing_ok: bool,
    test_labels: dict[str, int] | None,
) -> list[dict]:
    cfg = SPLITS[split]
    tsv_path = raw_dir.joinpath(*cfg["tsv"])
    images_dir = raw_dir.joinpath(*cfg["images"])

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    records: list[dict] = []
    with tsv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            file_name = row["file_name"].strip()
            text = row.get("Text Transcription", "").strip()

            if cfg["has_labels"]:
                label = _any_positive(row)
            else:
                if test_labels is None or file_name not in test_labels:
                    raise KeyError(
                        f"No label found for test image '{file_name}' in test_labels.txt"
                    )
                label = test_labels[file_name]

            src = images_dir / file_name
            dst = img_out_dir / file_name
            if not src.exists():
                if missing_ok:
                    print(f"  [WARN] missing image, skipping record: {src}")
                    continue
                else:
                    raise FileNotFoundError(f"Image not found: {src}")
            if not dst.exists():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())

            records.append({
                "id": Path(file_name).stem,
                "img": f"img/{file_name}",
                "label": label,
                "text": text,
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MAMI for hateful-memes loader")
    parser.add_argument("--raw-dir", required=True, help='Path to "MAMI DATASET" root')
    parser.add_argument("--out-dir", required=True, help="Output dataset_root directory")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument(
        "--missing-ok", action="store_true", help="Warn on missing images instead of crashing"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = out_dir / "img"
    img_out_dir.mkdir(exist_ok=True)

    test_labels = _load_test_labels(raw_dir)

    for split in ("train", "dev", "test"):
        records = convert_split(
            split=split,
            raw_dir=raw_dir,
            img_out_dir=img_out_dir,
            copy=args.copy,
            missing_ok=args.missing_ok,
            test_labels=test_labels if split == "test" else None,
        )
        out_jsonl = out_dir / f"{split}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_pos = sum(r["label"] for r in records)
        print(f"[{split:5s}] {len(records):4d} records  ({n_pos} positive)  → {out_jsonl}")

    print(f"\nDone. Set dataset_root: {out_dir}")


if __name__ == "__main__":
    main()
