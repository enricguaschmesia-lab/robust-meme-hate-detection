"""Phase-4 white-box adversarial benchmark.

Sweeps an L∞ epsilon grid for FGSM and PGD against a trained checkpoint and
emits a single ``whitebox_eval.json`` whose schema mirrors
``perturbed_eval.json`` so the Phase-4 aggregator can treat both files
uniformly. The attack operates on ``[0,1]`` pixel tensors; Normalize lives
inside the model (see ``models/clip_fusion.py``), so gradients flow through it.

Usage (on the cluster, via ``scripts/cluster_entrypoint.sh``):

    python -m robust_meme_hate_detection.eval.run_whitebox \\
        --ckpt   /scratch/.../stage1-seed0/ckpt/best.pt \\
        --config configs/stage1.yaml \\
        --out    /scratch/.../whitebox-stage1-seed0 \\
        [--modality multimodal|image] \\
        [--attacks fgsm,pgd] \\
        [--epsilons 1,2,4,8]  (numerators over 255) \\
        [--pgd-steps 10] [--pgd-alpha-frac 0.25] \\
        [--seed 0] [--batch-size 64] [--max-batches 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml


class _ImageOnlyAdapter:
    """Adapter so ``fgsm_image`` / ``pgd_image`` can attack the image-only model.

    The attack API expects ``model(images, tokens) -> logit``; for the
    image-only baseline we route through ``forward_image_only`` and ignore
    the token argument.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, images, _tokens):
        return self._model.forward_image_only(images)

    def parameters(self):
        return self._model.parameters()

    def eval(self):
        self._model.eval()
        return self


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument(
        '--dataset-root', default=None,
        help='Override data.dataset_root from --config (for held-out test eval).',
    )
    parser.add_argument(
        '--modality', default=None, choices=('multimodal', 'image'),
        help="Forward path used during eval. Defaults to model.modality from the config. "
             "Text-only models have no pixel input and are rejected.",
    )
    parser.add_argument(
        '--attacks', default='fgsm,pgd',
        help="Comma-separated subset of {fgsm,pgd}",
    )
    parser.add_argument(
        '--epsilons', default='1,2,4,8',
        help="Comma-separated numerators; each value e yields epsilon=e/255.",
    )
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument(
        '--pgd-alpha-frac', type=float, default=0.25,
        help="alpha = pgd_alpha_frac * epsilon (PGD step size relative to epsilon).",
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--max-batches', type=int, default=0,
        help="0 = attack the full split. >0 caps attack batches per cell (smoke).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    from robust_meme_hate_detection.attacks.pgd import fgsm_image, pgd_image
    from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
    from robust_meme_hate_detection.data.transforms import ClipImage01Transform, ClipTokenize
    from robust_meme_hate_detection.eval.metrics import (
        attack_success_rate,
        classification_metrics,
        robustness_gap,
    )
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.seeding import seed_everything
    from robust_meme_hate_detection.utils.threshold import sweep_threshold

    seed_everything(args.seed)

    model_cfg = cfg['model']
    data_cfg = cfg['data']
    modality = args.modality or str(model_cfg.get('modality', 'multimodal'))
    if modality not in ('multimodal', 'image'):
        raise ValueError(
            f"Unsupported modality {modality!r}: white-box image attacks need a vision "
            "branch. Use 'multimodal' or 'image'."
        )

    attacks = [a.strip() for a in args.attacks.split(',') if a.strip()]
    for a in attacks:
        if a not in ('fgsm', 'pgd'):
            raise ValueError(f"Unknown attack: {a!r}")
    epsilon_nums = [int(e.strip()) for e in args.epsilons.split(',') if e.strip()]
    if any(e <= 0 for e in epsilon_nums):
        raise ValueError("--epsilons numerators must be positive integers (over 255)")

    image_tfm = ClipImage01Transform(size=224)
    text_tfm = ClipTokenize(arch=model_cfg['arch'])
    dataset_root = args.dataset_root or data_cfg['dataset_root']
    records = load_hateful_memes_records(Path(dataset_root), args.split)
    records = [r for r in records if r.label is not None]

    class _DS:
        def __init__(self, recs):
            self.records = list(recs)
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            from PIL import Image
            r = self.records[i]
            img = Image.open(r.image_path).convert('RGB')
            return {
                'id': r.id,
                'image': image_tfm(img),
                'text': text_tfm(r.text),
                'label': int(r.label),
            }

    loader = DataLoader(_DS(records), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPHateMemeClassifier(
        arch=model_cfg['arch'],
        pretrained=model_cfg.get('pretrained', 'laion2b_s34b_b79k'),
        freeze_encoders=bool(model_cfg.get('freeze_encoders', True)),
        head_hidden=int(model_cfg.get('head_hidden', 512)),
        head_dropout=float(model_cfg.get('head_dropout', 0.2)),
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()

    # The attack functions call ``model(images, tokens)``; for an image-only
    # checkpoint, route through forward_image_only via the adapter.
    attack_callable = _ImageOnlyAdapter(model) if modality == 'image' else model

    # --------- collect [0,1] images, tokens, labels, ids in batches ---------
    # We pre-materialise the dev split into a list of batches so each
    # (attack, eps) cell sees exactly the same examples and the clean
    # reference matches per-example.
    batches: list[dict[str, Any]] = []
    for b in loader:
        batches.append({
            'images': b['image'],
            'tokens': b['text'],
            'labels': b['label'],
            'ids': list(b['id']),
        })

    # --------- clean reference pass ---------
    t0 = time.time()
    clean_probs: list[float] = []
    labels_flat: list[int] = []
    ids_flat: list[str] = []
    with torch.no_grad():
        for b in batches:
            images = b['images'].to(device, non_blocking=True)
            tokens = b['tokens'].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                if modality == 'multimodal':
                    logit = model(images, tokens)
                else:
                    logit = model.forward_image_only(images)
            clean_probs.extend(torch.sigmoid(logit.float()).detach().cpu().tolist())
            labels_flat.extend(int(x) for x in b['labels'].tolist())
            ids_flat.extend(b['ids'])

    clean_at_05 = classification_metrics(clean_probs, labels_flat, threshold=0.5)
    best_t, threshold_grid = sweep_threshold(clean_probs, labels_flat)
    clean_at_best = classification_metrics(
        clean_probs, labels_flat, threshold=best_t.threshold,
    )

    # --------- per-cell attack passes ---------
    cells: list[dict[str, Any]] = []
    for attack in attacks:
        for eps_num in epsilon_nums:
            epsilon = eps_num / 255.0
            alpha = args.pgd_alpha_frac * epsilon
            steps_used = args.pgd_steps if attack == 'pgd' else 1
            attacked_probs: list[float] = []
            attacked_labels: list[int] = []
            attacked_ids: list[str] = []

            for bi, b in enumerate(batches):
                if args.max_batches and bi >= args.max_batches:
                    break
                images = b['images'].to(device, non_blocking=True)
                tokens = b['tokens'].to(device, non_blocking=True)
                lab_t = b['labels'].to(device).float()

                if attack == 'fgsm':
                    adv = fgsm_image(attack_callable, images, tokens, lab_t, epsilon=epsilon)
                else:
                    adv = pgd_image(
                        attack_callable, images, tokens, lab_t,
                        epsilon=epsilon, alpha=alpha, steps=steps_used,
                        random_start=True,
                    )
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda',
                    ):
                        if modality == 'multimodal':
                            logit_adv = model(adv, tokens)
                        else:
                            logit_adv = model.forward_image_only(adv)
                attacked_probs.extend(torch.sigmoid(logit_adv.float()).detach().cpu().tolist())
                attacked_labels.extend(int(x) for x in b['labels'].tolist())
                attacked_ids.extend(b['ids'])

            clean_subset = clean_probs[: len(attacked_probs)]
            attacked = classification_metrics(
                attacked_probs, attacked_labels, threshold=best_t.threshold,
            )
            clean_subset_metrics = classification_metrics(
                clean_subset, attacked_labels, threshold=best_t.threshold,
            )

            examples: list[dict[str, Any]] = [
                {
                    'id': eid,
                    'label': int(lab),
                    'clean_prob': float(cp),
                    'attacked_prob': float(ap),
                }
                for eid, lab, cp, ap in zip(
                    attacked_ids, attacked_labels, clean_subset, attacked_probs,
                )
            ]

            cell = {
                'attack': attack,
                'modality': 'image',
                'epsilon': epsilon,
                'epsilon_numerator': eps_num,
                'pgd_steps': steps_used,
                'pgd_alpha': alpha,
                'n': len(attacked_probs),
                'attacked_at_best_threshold': attacked.to_dict(),
                'clean_subset_at_best_threshold': clean_subset_metrics.to_dict(),
                'robustness_gap': robustness_gap(clean_subset_metrics, attacked),
                'attack_success_rate': attack_success_rate(
                    clean_subset, attacked_probs, attacked_labels,
                    threshold=best_t.threshold,
                ),
                'examples': examples,
            }
            cells.append(cell)
            print(
                f"{attack:<4s}  eps={eps_num}/255  steps={steps_used:>2d}  "
                f"AUROC={attacked.auroc:.4f}  F1={attacked.macro_f1:.4f}  "
                f"ASR={cell['attack_success_rate']:.4f}",
                flush=True,
            )

    payload = {
        'ckpt': str(args.ckpt),
        'split': args.split,
        'modality': modality,
        'n': len(labels_flat),
        'global_seed': args.seed,
        'attacks': attacks,
        'epsilons_over_255': epsilon_nums,
        'pgd_steps': args.pgd_steps,
        'pgd_alpha_frac': args.pgd_alpha_frac,
        'clean': {
            'at_0.5': clean_at_05.to_dict(),
            'best_threshold': best_t.threshold,
            'at_best_threshold': clean_at_best.to_dict(),
        },
        'threshold_grid': [r.to_dict() for r in threshold_grid],
        'cells': cells,
        'elapsed_seconds': time.time() - t0,
    }
    out_path = out_dir / 'whitebox_eval.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {out_path}  ({len(cells)} cells)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
