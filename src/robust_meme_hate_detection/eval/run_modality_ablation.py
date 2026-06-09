"""Phase 4-S1: modality ablation.

Run a trained checkpoint through three forward paths on the same clean
split and record AUROC, macro-F1 and best-τ for each:

    - multimodal:  model(images, tokens)
    - text_only:   model.forward_text_only(tokens)         (image zero-padded)
    - image_only:  model.forward_image_only(images)        (text zero-padded)

For a multimodal checkpoint, the gap between ``multimodal`` and
``text_only`` quantifies how much the *vision branch* contributes to the
final decision; the gap between ``multimodal`` and ``image_only``
quantifies the text branch's contribution. This is the direct test of
the Phase-4 "text dominates the fusion" hypothesis.

Usage (cluster, via ``scripts/cluster_entrypoint.sh``):

    python -m robust_meme_hate_detection.eval.run_modality_ablation \\
        --ckpt   /scratch/.../stage1-seed0/ckpt/best.pt \\
        --config configs/stage1.yaml \\
        --out    /scratch/.../modality-ablation-stage1-seed0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
    from robust_meme_hate_detection.data.transforms import ClipTokenize
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    from robust_meme_hate_detection.eval.run_perturbed import (
        _build_pil_loader, _eval_cell, _forward, _to_tensor_batch,
    )
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.seeding import seed_everything
    from robust_meme_hate_detection.utils.threshold import sweep_threshold

    seed_everything(args.seed)

    model_cfg = cfg['model']
    data_cfg = cfg['data']

    text_tfm = ClipTokenize(arch=model_cfg['arch'])
    dataset_root = args.dataset_root or data_cfg['dataset_root']
    records = load_hateful_memes_records(Path(dataset_root), args.split)
    records = [r for r in records if r.label is not None]
    dataset = _build_pil_loader(records)

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

    payload: dict = {
        'ckpt': str(args.ckpt),
        'split': args.split,
        'n': len(records),
        'global_seed': args.seed,
    }
    t0 = time.time()
    for modality in ('multimodal', 'text', 'image'):
        probs, labels, _ids, _ = _eval_cell(
            model=model, modality=modality, dataset=dataset, text_tfm=text_tfm,
            device=device, batch_size=args.batch_size,
        )
        at_05 = classification_metrics(probs, labels, threshold=0.5)
        best_t, _grid = sweep_threshold(probs, labels)
        at_best = classification_metrics(probs, labels, threshold=best_t.threshold)
        # Use the key the report uses: 'multimodal' / 'text_only' / 'image_only'.
        key = {'multimodal': 'multimodal', 'text': 'text_only', 'image': 'image_only'}[modality]
        payload[key] = {
            'at_0.5': at_05.to_dict(),
            'best_threshold': best_t.threshold,
            'at_best_threshold': at_best.to_dict(),
            # Full macro-F1/Acc grid so a dev-selected threshold can be read off
            # post-hoc (the report fixes thresholds on clean dev, not on test).
            'grid': [r.to_dict() for r in _grid],
        }
        print(
            f"{key:<12s}  AUROC={at_best.auroc:.4f}  F1={at_best.macro_f1:.4f}  "
            f"best_t={best_t.threshold:.2f}",
            flush=True,
        )

    payload['elapsed_seconds'] = time.time() - t0
    out_path = out_dir / 'modality_ablation.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
