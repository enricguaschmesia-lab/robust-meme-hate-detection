"""Manual-inspection helper for the Phase-3 perturbation suite.

For each (attack, severity_level) cell, save N sample images (image attacks)
or N (original, perturbed) text pairs (text attacks) to a folder hierarchy
the reviewer can scroll through. Used to satisfy the DoD requirement:

    Manually check 50 to 100 examples for label preservation.

Output layout:

    <out>/
      manifest.csv                          # one row per (cell, sample, …)
      text/<attack>/<level>/pairs.csv       # original vs perturbed text
      image/<attack>/<level>/<id>.png       # perturbed image (one per sample)
      image/<attack>/<level>/_orig/<id>.png # original image for comparison

The script does not require a GPU or a trained checkpoint. It works directly
from the staged dataset and is intended to run on the jumphost or, more
commonly, as a short Run:AI batch job so its output lands on shared scratch
where teammates can review it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _derive_seed(base: int, attack: str, level: str) -> int:
    """Stable per-cell seed (SHA-256 keyed; survives across Python processes)."""
    payload = f"{base}|{attack}|{level}".encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31 - 1)


def _load_records(dataset_root: Path, split: str):
    from robust_meme_hate_detection.data.hateful_memes import (
        load_hateful_memes_records,
    )
    return load_hateful_memes_records(dataset_root, split)


def _pick(records, n_samples: int, seed: int) -> list:
    import random as _random
    rng = _random.Random(seed)
    labelled = [r for r in records if r.label is not None]
    if len(labelled) <= n_samples:
        return labelled
    return rng.sample(labelled, n_samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--n-samples', type=int, default=80,
                        help="Sample count per (attack, level) cell.")
    parser.add_argument(
        '--severities', default='low,medium,high',
        help="Subset of {low, medium, high}",
    )
    parser.add_argument(
        '--attacks', default='all',
        help="'all' | 'text' | 'image' | csv of attack names",
    )
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    from robust_meme_hate_detection.perturbations import (
        IMAGE_MODES,
        IMAGE_MODES_BENCHMARK,
        IMAGE_SEVERITY_PRESETS,
        ImagePerturbation,
        TEXT_MODES,
        TEXT_SEVERITY_PRESETS,
        TextPerturbation,
    )

    severities = [s.strip() for s in args.severities.split(',') if s.strip()]
    # Inspect the same image modes the benchmark uses (directional brightness/
    # contrast instead of the bidirectional random-sign aliases).
    if args.attacks == 'all':
        attacks: list[tuple[str, str]] = (
            [(m, 'text') for m in TEXT_MODES]
            + [(m, 'image') for m in IMAGE_MODES_BENCHMARK]
        )
    elif args.attacks == 'text':
        attacks = [(m, 'text') for m in TEXT_MODES]
    elif args.attacks == 'image':
        attacks = [(m, 'image') for m in IMAGE_MODES_BENCHMARK]
    else:
        wanted = [a.strip() for a in args.attacks.split(',') if a.strip()]
        attacks = []
        for a in wanted:
            if a in TEXT_MODES:
                attacks.append((a, 'text'))
            elif a in IMAGE_MODES_BENCHMARK:
                attacks.append((a, 'image'))
            elif a in IMAGE_MODES:
                raise ValueError(
                    f"{a!r} mixes random +/- per-sample. Use the directional "
                    f"variant ('{a}_up' or '{a}_down') for the inspection set."
                )
            else:
                raise ValueError(f"Unknown attack name: {a}")

    records = _load_records(dataset_root, args.split)
    samples = _pick(records, args.n_samples, args.seed)

    manifest_rows: list[dict[str, Any]] = []
    for attack, modality in attacks:
        for level in severities:
            seed = _derive_seed(args.seed, attack, level)
            cell_dir = out / modality / attack / level
            cell_dir.mkdir(parents=True, exist_ok=True)

            if modality == 'text':
                tp = TextPerturbation.from_preset(attack, level, probability=1.0, seed=seed)
                pairs_csv = cell_dir / 'pairs.csv'
                with pairs_csv.open('w', encoding='utf-8', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['id', 'label', 'original', 'perturbed'])
                    for rec in samples:
                        perturbed = tp(rec.text)
                        w.writerow([rec.id, rec.label, rec.text, perturbed])
                        manifest_rows.append({
                            'attack': attack, 'modality': modality, 'level': level,
                            'severity_float': TEXT_SEVERITY_PRESETS[attack][level],
                            'seed': seed, 'id': rec.id, 'label': rec.label,
                            'kind': 'text', 'path': str(pairs_csv.relative_to(out)),
                        })
            else:
                ip = ImagePerturbation.from_preset(attack, level, probability=1.0, seed=seed)
                orig_dir = cell_dir / '_orig'; orig_dir.mkdir(exist_ok=True)
                for rec in samples:
                    img = Image.open(rec.image_path).convert('RGB').resize(
                        (args.image_size, args.image_size), Image.BICUBIC,
                    )
                    perturbed = ip(img)
                    perturbed.save(cell_dir / f"{rec.id}.png")
                    img.save(orig_dir / f"{rec.id}.png")
                    manifest_rows.append({
                        'attack': attack, 'modality': modality, 'level': level,
                        'severity_float': IMAGE_SEVERITY_PRESETS[attack][level],
                        'seed': seed, 'id': rec.id, 'label': rec.label,
                        'kind': 'image', 'path': str((cell_dir / f"{rec.id}.png").relative_to(out)),
                    })
            print(f"wrote {modality}/{attack}/{level}: {len(samples)} samples (seed={seed})",
                  flush=True)

    manifest_path = out / 'manifest.csv'
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    summary = {
        'split': args.split,
        'n_samples_per_cell': len(samples),
        'attacks': [a for a, _ in attacks],
        'severities': severities,
        'global_seed': args.seed,
        'total_rows': len(manifest_rows),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {out / 'summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
