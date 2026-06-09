"""Convert the HarMeme dataset (Harm-C + Harm-P) into the JSONL + img/ layout
expected by load_hateful_memes_records.

Raw layout (--raw-dir should point at the "HarMeme" root):
  HarMeme/
    Annotations/
      Harm-C/   train.jsonl  val.jsonl  test.jsonl
      Harm-P/   train_v1.jsonl  val_v1.jsonl  test_v1.jsonl
    HarMeme_Images/
      harmeme_images_covid_19/   covid_memes_*.png
      harmeme_images_us_pol/     memes_*.png

Output layout (--out-dir becomes dataset_root in the config):
  <out-dir>/
    train.jsonl   (Harm-C train  + Harm-P train_v1)
    dev.jsonl     (Harm-C val   + Harm-P val_v1)
    test.jsonl    (Harm-C test  + Harm-P test_v1)
    img/          (all images symlinked/copied here)

Label rule (labels[0]):
  "not harmful"      → 0
  "somewhat harmful" → 1
  "very harmful"     → 1
  anything else      → skipped with a warning

Usage:
  python -m robust_meme_hate_detection.data.prepare_harmeme \\
      --raw-dir data/raw/HarMeme \\
      --out-dir data/prepared/harmeme
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

# labels[0] → binary label; any other value triggers a skip + warning
LABEL_MAP = {
    "not harmful": 0,
    "somewhat harmful": 1,
    "very harmful": 1,
}

# output split → list of (subset, filename) pairs to concatenate
SPLIT_SOURCES: dict[str, list[tuple[str, str]]] = {
    "train": [("Harm-C", "train.jsonl"), ("Harm-P", "train_v1.jsonl")],
    "dev":   [("Harm-C", "val.jsonl"),   ("Harm-P", "val_v1.jsonl")],
    "test":  [("Harm-C", "test.jsonl"),  ("Harm-P", "test_v1.jsonl")],
}


def _build_image_index(images_root: Path) -> dict[str, Path]:
    """Recursively index all image files under images_root by filename."""
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            name = path.name
            if name in index:
                duplicates.append(name)
            index[name] = path
    if duplicates:
        print(f"[WARN] {len(duplicates)} duplicate image filenames found (last path wins):")
        for d in duplicates[:5]:
            print(f"       {d}")
    return index


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sanity_check(all_records: list[dict]) -> None:
    """Print label distribution and flag anything unexpected."""
    label_counts: Counter = Counter()
    for r in all_records:
        label_counts[r["labels"][0]] += 1

    known = set(LABEL_MAP.keys())
    unknown = {k for k in label_counts if k not in known}

    print("\n--- Label distribution (labels[0]) across all source files ---")
    for val, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        tag = "" if val in known else "  ← UNKNOWN"
        print(f"  {count:5d}  {val!r}{tag}")

    if unknown:
        print(f"\n[WARN] {len(unknown)} unknown label value(s) will be skipped: {unknown}")
    else:
        print("\n[OK] All label values are recognised.")

    ids = [r["id"] for r in all_records]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        print(f"[WARN] {len(dupes)} duplicate IDs across Harm-C + Harm-P: {dupes[:5]}")
    else:
        print(f"[OK] All {len(ids)} IDs are unique across Harm-C and Harm-P.")
    print()


def convert_split(
    split: str,
    annotations_dir: Path,
    image_index: dict[str, Path],
    img_out_dir: Path,
    copy: bool,
    missing_ok: bool,
) -> list[dict]:
    source_records: list[dict] = []
    for subset, filename in SPLIT_SOURCES[split]:
        src_path = annotations_dir / subset / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {src_path}")
        source_records.extend(_load_jsonl(src_path))

    records: list[dict] = []
    n_skipped = 0
    n_missing = 0

    for r in source_records:
        raw_label = r["labels"][0]
        if raw_label not in LABEL_MAP:
            print(f"  [SKIP] id={r['id']!r} — unrecognised label {raw_label!r}")
            n_skipped += 1
            continue

        image_name = r["image"]
        src_img = image_index.get(image_name)
        if src_img is None:
            if missing_ok:
                print(f"  [WARN] image not found in index, skipping record: {image_name!r}")
                n_missing += 1
                continue
            else:
                raise FileNotFoundError(
                    f"Image '{image_name}' not found under HarMeme_Images/. "
                    "Use --missing-ok to skip."
                )

        dst_img = img_out_dir / image_name
        if not dst_img.exists():
            if copy:
                shutil.copy2(src_img, dst_img)
            else:
                dst_img.symlink_to(src_img.resolve())

        records.append({
            "id": r["id"],
            "img": f"img/{image_name}",
            "label": LABEL_MAP[raw_label],
            "text": r.get("text", "").strip(),
        })

    extra = []
    if n_skipped:
        extra.append(f"{n_skipped} skipped")
    if n_missing:
        extra.append(f"{n_missing} missing images")
    suffix = f"  ({', '.join(extra)})" if extra else ""

    n_pos = sum(rec["label"] for rec in records)
    print(f"[{split:5s}] {len(records):5d} records  ({n_pos} positive / {len(records)-n_pos} negative){suffix}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HarMeme for hateful-memes loader")
    parser.add_argument("--raw-dir", required=True, help='Path to "HarMeme" root')
    parser.add_argument("--out-dir", required=True, help="Output dataset_root directory")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument(
        "--missing-ok", action="store_true", help="Warn on missing images instead of crashing"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    annotations_dir = raw_dir / "Annotations"
    images_root = raw_dir / "HarMeme_Images"

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations dir not found: {annotations_dir}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = out_dir / "img"
    img_out_dir.mkdir(exist_ok=True)

    # Load all source records upfront for sanity checks
    all_source: list[dict] = []
    for sources in SPLIT_SOURCES.values():
        for subset, filename in sources:
            path = annotations_dir / subset / filename
            if path.exists():
                all_source.extend(_load_jsonl(path))
    _sanity_check(all_source)

    print("Building image index ...")
    image_index = _build_image_index(images_root)
    print(f"Found {len(image_index)} images.\n")

    for split in ("train", "dev", "test"):
        records = convert_split(
            split=split,
            annotations_dir=annotations_dir,
            image_index=image_index,
            img_out_dir=img_out_dir,
            copy=args.copy,
            missing_ok=args.missing_ok,
        )
        out_jsonl = out_dir / f"{split}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Set dataset_root: {out_dir}")


if __name__ == "__main__":
    main()
