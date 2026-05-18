"""Phase-3 perturbation benchmark.

Walk the (attack × severity) grid against a trained checkpoint and emit one
JSON per checkpoint. Each cell records attack name, modality, severity
level, severity float, RNG seed, clean and attacked metrics, robustness gap,
and attack success rate — i.e. the metadata required by Phase 3 of
``project_planning/ProjectIdeaHenrik_Feasibility_Roadmap.md`` (§"DoD"):

    The benchmark records attack name, modality, severity, and random seed.

Usage (on the cluster, via ``scripts/cluster_entrypoint.sh``):

    python -m robust_meme_hate_detection.eval.run_perturbed \\
        --ckpt   /scratch/.../experiments/stage1-seed0/ckpt/best.pt \\
        --config configs/stage1.yaml \\
        --out    /scratch/.../experiments/perturbed-stage1-seed0 \\
        [--split dev] [--modality multimodal|text|image] \\
        [--attacks all|text|image|<csv of attack names>] \\
        [--severities low,medium,high] [--seed 0] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml


def _derive_seed(base: int, attack: str, level: str) -> int:
    """Deterministic int seed for a (attack, level) cell.

    Uses SHA-256 so the value is stable across Python processes (the built-in
    ``hash()`` is salted with PYTHONHASHSEED by default and would otherwise
    change between Run:AI jobs).
    """
    payload = f"{base}|{attack}|{level}".encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31 - 1)


def _derive_sample_seed(cell_seed: int, sample_id: str) -> int:
    """Deterministic sub-seed for one (cell, sample) composite draw."""
    payload = f"{cell_seed}|{sample_id}".encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31 - 1)


# Phase 5c-2: composite-attack design.
COMPOSITE_TYPES: tuple[str, ...] = (
    'composite_2text',
    'composite_2image',
    'composite_text_image',
    'composite_2text_2image',
)
COMPOSITE_K: dict[str, tuple[int, int]] = {
    # (K_text, K_image)
    'composite_2text': (2, 0),
    'composite_2image': (0, 2),
    'composite_text_image': (1, 1),
    'composite_2text_2image': (2, 2),
}


def _build_pil_loader(records, image_size: int = 224):
    from PIL import Image

    class _PILDS:
        def __init__(self, recs):
            self.records = list(recs)
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            r = self.records[i]
            img = Image.open(r.image_path).convert('RGB')
            # Resize at PIL level so perturbations and the model see the same
            # canvas. This is what ClipImage01Transform does (resize → tensor).
            img = img.resize((image_size, image_size), Image.BICUBIC)
            return {'id': r.id, 'image': img, 'text': r.text, 'label': int(r.label)}

    return _PILDS(records)


def _to_tensor_batch(samples, *, text_tfm, device):
    """Convert a list of {image: PIL, text: str, label: int} into batch tensors."""
    import numpy as np
    import torch

    images = []
    for s in samples:
        arr = np.asarray(s['image'], dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr).permute(2, 0, 1))
    img_t = torch.stack(images, dim=0).to(device, non_blocking=True)
    tok_list = [text_tfm(s['text']) for s in samples]
    tok_t = torch.stack(tok_list, dim=0).to(device, non_blocking=True)
    labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)
    return img_t, tok_t, labels


def _forward(model, modality: str, images, tokens):
    if modality == 'multimodal':
        return model(images, tokens)
    if modality == 'text':
        return model.forward_text_only(tokens)
    if modality == 'image':
        return model.forward_image_only(images)
    raise ValueError(f"Unknown modality: {modality}")


def _apply_composite(
    *,
    text: str,
    image,
    sample_seed: int,
    text_pool,
    image_pool,
    severity: str,
    k_text: int,
    k_image: int,
) -> tuple[str, "object", list[dict[str, Any]]]:
    """Apply K_text + K_image perturbations sequentially. Deterministic given
    ``sample_seed``.

    Components are sampled *without replacement* per modality (we want
    distinct attacks per composite, e.g. two different text edits — not the
    same one twice). Severity is fixed to the cell level for every component
    so the cell is interpretable at a single severity.

    Returns (perturbed_text, perturbed_image, components_log).
    """
    import random as _r

    from robust_meme_hate_detection.perturbations import (
        ImagePerturbation,
        TextPerturbation,
    )

    rng = _r.Random(sample_seed)
    components: list[dict[str, Any]] = []

    text_pool = list(text_pool)
    image_pool = list(image_pool)
    text_picks = rng.sample(text_pool, k=min(k_text, len(text_pool))) if k_text else []
    image_picks = rng.sample(image_pool, k=min(k_image, len(image_pool))) if k_image else []

    out_text = text
    for atk in text_picks:
        comp_seed = rng.randrange(0, 2 ** 31 - 1)
        pert = TextPerturbation.from_preset(atk, severity, probability=1.0, seed=comp_seed)
        out_text = pert(out_text)
        components.append({'attack': atk, 'modality': 'text', 'severity_level': severity})

    out_image = image
    for atk in image_picks:
        comp_seed = rng.randrange(0, 2 ** 31 - 1)
        pert = ImagePerturbation.from_preset(atk, severity, probability=1.0, seed=comp_seed)
        out_image = pert(out_image)
        components.append({'attack': atk, 'modality': 'image', 'severity_level': severity})

    return out_text, out_image, components


def _eval_composite_cell(
    *, model, modality, dataset, text_tfm, device, batch_size,
    composite_type: str, severity: str, cell_seed: int,
    text_pool, image_pool,
) -> tuple[list[float], list[int], list[str], list[str | None], list[list[dict[str, Any]]]]:
    """Single pass over dataset applying a composite perturbation per sample.

    Per-sample components are drawn deterministically from
    ``sha256(cell_seed | sample_id)``. Returns probs/labels/ids/perturbed_texts
    plus the per-sample components log (for the first record of each cell we
    write a representative entry into the cell JSON).
    """
    import torch

    k_text, k_image = COMPOSITE_K[composite_type]
    probs: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    perturbed_texts: list[str | None] = []
    all_components: list[list[dict[str, Any]]] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples_raw = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            samples = []
            for s in samples_raw:
                sample_seed = _derive_sample_seed(cell_seed, str(s['id']))
                new_text, new_image, components = _apply_composite(
                    text=s['text'], image=s['image'],
                    sample_seed=sample_seed,
                    text_pool=text_pool, image_pool=image_pool,
                    severity=severity, k_text=k_text, k_image=k_image,
                )
                samples.append({**s, 'text': new_text, 'image': new_image})
                all_components.append(components)

            img_t, tok_t, lab_t = _to_tensor_batch(samples, text_tfm=text_tfm, device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda',
            ):
                logit = _forward(model, modality, img_t, tok_t)
            probs.extend(torch.sigmoid(logit.float()).detach().cpu().tolist())
            labels.extend(int(x) for x in lab_t.tolist())
            ids.extend(str(s['id']) for s in samples)
            perturbed_texts.extend(str(s['text']) for s in samples)
    return probs, labels, ids, perturbed_texts, all_components


def _eval_cell(
    *, model, modality, dataset, text_tfm, device, batch_size,
    text_perturb=None, image_perturb=None, record_perturbed_text: bool = False,
) -> tuple[list[float], list[int], list[str], list[str | None]]:
    """Single pass over the dataset, optionally applying perturbations.

    Returns four parallel lists: (probs, labels, ids, perturbed_texts). When
    ``record_perturbed_text`` is False, the last list is filled with ``None``
    so callers can still zip without a special case.
    """
    import torch

    probs: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    perturbed_texts: list[str | None] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            if text_perturb is not None:
                samples = [{**s, 'text': text_perturb(s['text'])} for s in samples]
            if image_perturb is not None:
                samples = [{**s, 'image': image_perturb(s['image'])} for s in samples]

            img_t, tok_t, lab_t = _to_tensor_batch(samples, text_tfm=text_tfm, device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda',
            ):
                logit = _forward(model, modality, img_t, tok_t)
            probs.extend(torch.sigmoid(logit.float()).detach().cpu().tolist())
            labels.extend(int(x) for x in lab_t.tolist())
            ids.extend(str(s['id']) for s in samples)
            if record_perturbed_text:
                perturbed_texts.extend(str(s['text']) for s in samples)
            else:
                perturbed_texts.extend([None] * len(samples))
    return probs, labels, ids, perturbed_texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument(
        '--dataset-root', default=None,
        help='Override data.dataset_root from --config (e.g. for held-out test eval against a labelled-test mirror).',
    )
    parser.add_argument(
        '--modality', default=None,
        choices=('multimodal', 'text', 'image'),
        help='Forward mode used during eval. Defaults to model.modality from the config.',
    )
    parser.add_argument(
        '--attacks', default='all',
        help="'all' | 'text' | 'image' | comma-separated subset of attack names",
    )
    parser.add_argument(
        '--severities', default='low,medium,high',
        help='Comma-separated subset of {low,medium,high}',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--composites', default='none',
        help=(
            "'none' (default, back-compat), 'all', or comma-separated subset of "
            f"{{{','.join(COMPOSITE_TYPES)}}}. Each composite cell applies K "
            "perturbations per sample (drawn deterministically from "
            "sha256(cell_seed|sample_id)). Components are sampled from the full "
            "eval pool (TEXT_MODES + IMAGE_MODES_BENCHMARK), not the training "
            "pool, so composites also probe OOD generalisation."
        ),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
    from robust_meme_hate_detection.data.transforms import ClipTokenize
    from robust_meme_hate_detection.eval.metrics import (
        attack_success_rate,
        classification_metrics,
        robustness_gap,
    )
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.perturbations import (
        IMAGE_MODES,
        IMAGE_MODES_BENCHMARK,
        IMAGE_SEVERITY_PRESETS,
        ImagePerturbation,
        TEXT_MODES,
        TEXT_SEVERITY_PRESETS,
        TextPerturbation,
    )
    from robust_meme_hate_detection.utils.seeding import seed_everything
    from robust_meme_hate_detection.utils.threshold import sweep_threshold

    seed_everything(args.seed)

    model_cfg = cfg['model']
    data_cfg = cfg['data']
    modality = args.modality or str(model_cfg.get('modality', 'multimodal'))

    text_tfm = ClipTokenize(arch=model_cfg['arch'])
    dataset_root = args.dataset_root or data_cfg['dataset_root']
    records = load_hateful_memes_records(Path(dataset_root), args.split)
    records = [r for r in records if r.label is not None]
    dataset = _build_pil_loader(records)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mirror stage1.py's model construction so non-default head_hidden /
    # head_dropout / freeze_encoders checkpoints load cleanly.
    model = CLIPHateMemeClassifier(
        arch=model_cfg['arch'],
        pretrained=model_cfg.get('pretrained', 'laion2b_s34b_b79k'),
        freeze_encoders=bool(model_cfg.get('freeze_encoders', True)),
        head_hidden=int(model_cfg.get('head_hidden', 512)),
        head_dropout=float(model_cfg.get('head_dropout', 0.2)),
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])

    # ---------------- clean reference pass ----------------
    t0 = time.time()
    clean_probs, labels, ids, _ = _eval_cell(
        model=model, modality=modality, dataset=dataset, text_tfm=text_tfm,
        device=device, batch_size=args.batch_size,
    )
    clean_at_05 = classification_metrics(clean_probs, labels, threshold=0.5)
    best_t, threshold_grid = sweep_threshold(clean_probs, labels)
    clean_at_best = classification_metrics(clean_probs, labels, threshold=best_t.threshold)

    # ---------------- pick (attack, severity) cells ----------------
    severities = [s.strip() for s in args.severities.split(',') if s.strip()]
    for s in severities:
        if s not in ('low', 'medium', 'high'):
            raise ValueError(f"Unknown severity level: {s}")

    # Benchmark uses the directional brightness/contrast modes, not the
    # bidirectional aliases (the latter are for training-augmentation only).
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
                # bidirectional alias (e.g. 'brightness') — reject in the
                # benchmark path so cell metrics aren't averaged over a
                # bimodal +/- distribution.
                raise ValueError(
                    f"{a!r} mixes random +/- per-sample. Use the directional "
                    f"variant ('{a}_up' or '{a}_down') in the benchmark."
                )
            else:
                raise ValueError(f"Unknown attack name: {a}")

    # ---------------- run each cell ----------------
    cells: list[dict[str, Any]] = []
    for attack, attack_modality in attacks:
        for level in severities:
            cell_seed = _derive_seed(args.seed, attack, level)
            if attack_modality == 'text':
                tp = TextPerturbation.from_preset(attack, level, probability=1.0, seed=cell_seed)
                severity_float = TEXT_SEVERITY_PRESETS[attack][level]
                attacked_probs, attacked_labels, attacked_ids, perturbed_texts = _eval_cell(
                    model=model, modality=modality, dataset=dataset, text_tfm=text_tfm,
                    device=device, batch_size=args.batch_size,
                    text_perturb=tp, image_perturb=None, record_perturbed_text=True,
                )
            else:
                ip = ImagePerturbation.from_preset(attack, level, probability=1.0, seed=cell_seed)
                severity_float = IMAGE_SEVERITY_PRESETS[attack][level]
                attacked_probs, attacked_labels, attacked_ids, perturbed_texts = _eval_cell(
                    model=model, modality=modality, dataset=dataset, text_tfm=text_tfm,
                    device=device, batch_size=args.batch_size,
                    text_perturb=None, image_perturb=ip, record_perturbed_text=False,
                )

            attacked = classification_metrics(attacked_probs, attacked_labels,
                                              threshold=best_t.threshold)
            # Per-example records. For text attacks the perturbed string is
            # stored verbatim. For image attacks the perturbed image is fully
            # reproducible from (global_seed, cell_seed, sample_index) given
            # the deterministic RNG; we therefore record clean+attacked
            # probabilities and the example id, which is enough to identify
            # the per-example outcome and feed Phase-6 failure analysis.
            examples: list[dict[str, Any]] = []
            for i, (eid, lab, cp, ap) in enumerate(zip(
                attacked_ids, attacked_labels, clean_probs, attacked_probs,
            )):
                entry: dict[str, Any] = {
                    'id': eid,
                    'label': int(lab),
                    'clean_prob': float(cp),
                    'attacked_prob': float(ap),
                }
                if attack_modality == 'text':
                    entry['perturbed_text'] = perturbed_texts[i]
                examples.append(entry)

            cells.append({
                'attack': attack,
                'modality': attack_modality,
                'severity_level': level,
                'severity_float': severity_float,
                'seed': cell_seed,
                'n': len(attacked_probs),
                'attacked_at_best_threshold': attacked.to_dict(),
                'robustness_gap': robustness_gap(clean_at_best, attacked),
                'attack_success_rate': attack_success_rate(
                    clean_probs, attacked_probs, attacked_labels,
                    threshold=best_t.threshold,
                ),
                'examples': examples,
            })
            print(
                f"{attack:<16s} {attack_modality:<5s} {level:<6s} "
                f"AUROC={attacked.auroc:.4f}  F1={attacked.macro_f1:.4f}  "
                f"ASR={cells[-1]['attack_success_rate']:.4f}",
                flush=True,
            )

    # ---------------- composite cells (Phase 5c-2) ----------------
    if args.composites and args.composites != 'none':
        if args.composites == 'all':
            composite_types: list[str] = list(COMPOSITE_TYPES)
        else:
            composite_types = []
            for c in (s.strip() for s in args.composites.split(',') if s.strip()):
                if c not in COMPOSITE_TYPES:
                    raise ValueError(f"Unknown composite type: {c!r}")
                composite_types.append(c)

        # Component pool: full eval pool (NOT the training pool).
        text_pool = list(TEXT_MODES)
        image_pool = list(IMAGE_MODES_BENCHMARK)

        for ctype in composite_types:
            for level in severities:
                cell_seed = _derive_seed(args.seed, ctype, level)
                attacked_probs, attacked_labels, attacked_ids, perturbed_texts, all_components = (
                    _eval_composite_cell(
                        model=model, modality=modality, dataset=dataset,
                        text_tfm=text_tfm, device=device, batch_size=args.batch_size,
                        composite_type=ctype, severity=level, cell_seed=cell_seed,
                        text_pool=text_pool, image_pool=image_pool,
                    )
                )
                attacked = classification_metrics(
                    attacked_probs, attacked_labels, threshold=best_t.threshold,
                )
                examples: list[dict[str, Any]] = []
                for i, (eid, lab, cp, ap) in enumerate(zip(
                    attacked_ids, attacked_labels, clean_probs, attacked_probs,
                )):
                    examples.append({
                        'id': eid,
                        'label': int(lab),
                        'clean_prob': float(cp),
                        'attacked_prob': float(ap),
                        'perturbed_text': perturbed_texts[i],
                        'components': all_components[i],
                    })
                k_text, k_image = COMPOSITE_K[ctype]
                cells.append({
                    'attack': ctype,
                    'modality': 'composite',
                    'severity_level': level,
                    'severity_float': float('nan'),
                    'seed': cell_seed,
                    'n': len(attacked_probs),
                    'composite': {
                        'k_text': k_text, 'k_image': k_image,
                        'text_pool': sorted(text_pool),
                        'image_pool': sorted(image_pool),
                    },
                    'attacked_at_best_threshold': attacked.to_dict(),
                    'robustness_gap': robustness_gap(clean_at_best, attacked),
                    'attack_success_rate': attack_success_rate(
                        clean_probs, attacked_probs, attacked_labels,
                        threshold=best_t.threshold,
                    ),
                    'examples': examples,
                })
                print(
                    f"{ctype:<24s} comp  {level:<6s} "
                    f"AUROC={attacked.auroc:.4f}  F1={attacked.macro_f1:.4f}  "
                    f"ASR={cells[-1]['attack_success_rate']:.4f}",
                    flush=True,
                )

    payload = {
        'ckpt': str(args.ckpt),
        'split': args.split,
        'modality': modality,
        'n': len(labels),
        'global_seed': args.seed,
        'clean': {
            'at_0.5': clean_at_05.to_dict(),
            'best_threshold': best_t.threshold,
            'at_best_threshold': clean_at_best.to_dict(),
        },
        'threshold_grid': [r.to_dict() for r in threshold_grid],
        'cells': cells,
        'elapsed_seconds': time.time() - t0,
    }
    out_path = out_dir / 'perturbed_eval.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {out_path}  ({len(cells)} cells)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
