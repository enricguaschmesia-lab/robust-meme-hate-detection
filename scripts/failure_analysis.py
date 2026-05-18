"""Phase-6 failure analysis: multi-seed, multi-recipe, composite-aware.

Reads per-example records inside ``cluster-results/{perturbed,whitebox}-{recipe}-seed{N}/``
for every (recipe, seed) requested, joins them with the eval split's captions,
and writes a narrative-ready markdown to ``project_planning/phase4/failure_analysis.md``.

What it produces (vs the original Phase-4-S3 single-seed/single-recipe script):

- Per-recipe bucket counts (B1..B5) averaged across seeds with ± σ.
- "Consensus residual hotspots" — examples that every requested recipe
  fails on naturally in ≥ majority of seeds.
- "Where ``kldrop`` beats ``kl``" / "Where ``kl`` beats ``kldrop``" — per-example
  disagreements that hold in ≥ majority of seeds.
- B5 composite-only bucket — examples that survive every single-cell natural
  attack but get flipped by ≥ 1 composite cell.

Bucket definitions (per recipe-seed):

    B1  clean-correct AND survives all single-cell natural AND all composite AND PGD
    B2  clean-wrong
    B3  clean-correct AND flipped by ≥ 1 single-cell natural attack
    B4  clean-correct AND robust to single+composite naturals, broken ONLY by PGD
    B5  clean-correct AND robust to single-cell naturals, flipped by ≥ 1 composite

No GPU, no cluster — uses files already on disk.
"""

from __future__ import annotations

import argparse
import json
import random as _random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'cluster-results'
OUT_DIR = REPO / 'project_planning' / 'phase4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- attack taxonomy

TEXT_ATTACKS = {
    'leetspeak', 'char_deletion', 'char_swap', 'spacing',
    'punctuation', 'case_noise', 'censoring', 'keyboard_typo',
}
IMAGE_ATTACKS = {
    'gaussian_noise', 'blur', 'compression',
    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
    'translation', 'crop', 'occlusion', 'typographic',
}
COMPOSITE_TYPES = {
    'composite_2text', 'composite_2image',
    'composite_text_image', 'composite_2text_2image',
}


def _attack_family(name: str) -> str:
    if name in TEXT_ATTACKS:
        return 'text'
    if name == 'typographic':
        return 'typographic'
    if name in ('gaussian_noise', 'blur', 'compression'):
        return 'image-pixel'
    if name in ('brightness_up', 'brightness_down', 'contrast_up', 'contrast_down'):
        return 'image-photometric'
    if name in ('translation', 'crop', 'occlusion'):
        return 'image-geometric'
    return 'other'


def _is_composite(name: str) -> bool:
    return name in COMPOSITE_TYPES


# ---------------------------------------------------------------- caption loading

def _caption_paths(split: str) -> list[Path]:
    if split == 'dev':
        return [REPO / 'data' / 'processed' / 'dev_meta' / 'dev.jsonl']
    if split == 'test_seen':
        return [REPO / 'data' / 'processed' / 'splits' / 'test_seen_labels.jsonl']
    if split == 'test_unseen':
        return [REPO / 'data' / 'processed' / 'splits' / 'test_unseen_labels.jsonl']
    raise ValueError(f"unknown split: {split!r}")


def _load_captions(split: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in _caption_paths(split):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r['id']).zfill(5)
            out[rid] = {
                'label': int(r.get('label', -1)),
                'text': str(r.get('text', '')),
                'img': str(r.get('img', '')),
            }
    return out


# ---------------------------------------------------------------- per (recipe, seed) loaders

def _job_dir(kind: str, recipe: str, seed: int, split: str = 'dev') -> Path | None:
    """Resolve cluster-results/{kind}-{recipe-resolved}{-split-tag}-seed{N}/.

    `recipe='stage1'`  -> e.g. `perturbed-stage1-seed0`            (dev)
    `recipe='kl'`      -> e.g. `perturbed-robust-kl-seed0`         (dev)
    `recipe='kldrop'`, split='test_seen' ->
                          `perturbed-robust-kldrop-test-seen-seed0`
    """
    if recipe in ('stage1', 'clean'):
        base = f'stage1-seed{seed}'
        if split != 'dev':
            split_tag = split.replace('_', '-')
            base = f'stage1-{split_tag}-seed{seed}'
    else:
        base = f'robust-{recipe}-seed{seed}'
        if split != 'dev':
            split_tag = split.replace('_', '-')
            base = f'robust-{recipe}-{split_tag}-seed{seed}'
    p = RESULTS / f'{kind}-{base}'
    return p if p.exists() else None


def _load_perturbed(recipe: str, seed: int, split: str = 'dev') -> dict[str, Any] | None:
    d = _job_dir('perturbed', recipe, seed, split)
    if d is None:
        return None
    fp = d / 'perturbed_eval.json'
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def _load_whitebox(recipe: str, seed: int, split: str = 'dev') -> dict[str, Any] | None:
    d = _job_dir('whitebox', recipe, seed, split)
    if d is None:
        return None
    fp = d / 'whitebox_eval.json'
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


# ---------------------------------------------------------------- per-example seed data

def _build_seed_table(perturbed: dict[str, Any], whitebox: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return {rid -> {clean_prob, best_threshold, per_family_worst, per_composite_worst, pgd}}.

    Worst = farthest from true-label side of best_t (lowest prob if label=1).
    """
    best_t = perturbed['clean']['best_threshold']
    table: dict[str, dict[str, Any]] = {}
    if not perturbed['cells']:
        return table

    # Seed from first cell's example list (every cell carries clean_prob per example)
    for ex in perturbed['cells'][0]['examples']:
        rid = str(ex['id']).zfill(5)
        table[rid] = {
            'label': int(ex['label']),
            'clean_prob': float(ex['clean_prob']),
            'best_threshold': best_t,
            'per_family_worst': {},
            'per_composite_worst': {},
            'pgd_at_4_over_255': None,
        }

    def _worse(prev: dict[str, Any] | None, ex: dict[str, Any], cell: dict[str, Any], label: int) -> dict[str, Any]:
        ap = float(ex['attacked_prob'])
        if prev is None:
            return {'attack': cell['attack'], 'severity': cell['severity_level'], 'attacked_prob': ap}
        if label == 1 and ap < prev['attacked_prob']:
            return {'attack': cell['attack'], 'severity': cell['severity_level'], 'attacked_prob': ap}
        if label == 0 and ap > prev['attacked_prob']:
            return {'attack': cell['attack'], 'severity': cell['severity_level'], 'attacked_prob': ap}
        return prev

    for cell in perturbed['cells']:
        attack = cell['attack']
        is_comp = _is_composite(attack)
        fam = _attack_family(attack) if not is_comp else None
        if not is_comp and fam == 'other':
            continue
        bucket_key = attack if is_comp else fam
        slot_name = 'per_composite_worst' if is_comp else 'per_family_worst'
        for ex in cell['examples']:
            rid = str(ex['id']).zfill(5)
            if rid not in table:
                continue
            entry = table[rid]
            entry[slot_name][bucket_key] = _worse(entry[slot_name].get(bucket_key), ex, cell, entry['label'])

    if whitebox is not None:
        for cell in whitebox.get('cells', []):
            if cell.get('attack') != 'pgd' or cell.get('epsilon_numerator') != 4:
                continue
            for ex in cell['examples']:
                rid = str(ex['id']).zfill(5)
                if rid in table:
                    table[rid]['pgd_at_4_over_255'] = float(ex['attacked_prob'])
    return table


def _bucket_for(entry: dict[str, Any]) -> str:
    """Derive B1..B5 bucket for one (rid, recipe, seed) entry."""
    label = entry['label']
    t = entry['best_threshold']
    cp = entry['clean_prob']
    clean_pred = 1 if cp >= t else 0
    clean_correct = clean_pred == label
    if not clean_correct:
        return 'B2_clean_wrong'

    flipped_natural = any(
        (1 if info['attacked_prob'] >= t else 0) != label
        for info in entry['per_family_worst'].values()
    )
    flipped_composite = any(
        (1 if info['attacked_prob'] >= t else 0) != label
        for info in entry['per_composite_worst'].values()
    )
    pgd = entry.get('pgd_at_4_over_255')
    flipped_pgd = pgd is not None and (1 if pgd >= t else 0) != label

    if flipped_natural:
        return 'B3_natural_attack_flipped'
    if flipped_composite:
        return 'B5_composite_only_failure'
    if flipped_pgd:
        return 'B4_pgd_only'
    return 'B1_robust'


# ---------------------------------------------------------------- aggregation

def _build_full_table(
    recipes: list[str],
    seeds: list[int],
    captions: dict[str, dict[str, Any]],
    split: str = 'dev',
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int]]]:
    """Return ({rid -> {label, text, img, by_seed: {(recipe,seed) -> entry+bucket}}}, available_pairs)."""
    available: list[tuple[str, int]] = []
    table: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        for seed in seeds:
            pert = _load_perturbed(recipe, seed, split)
            if pert is None:
                continue
            wb = _load_whitebox(recipe, seed, split)
            seed_tab = _build_seed_table(pert, wb)
            if not seed_tab:
                continue
            available.append((recipe, seed))
            for rid, ent in seed_tab.items():
                ent['bucket'] = _bucket_for(ent)
                meta = captions.get(rid, {})
                if rid not in table:
                    table[rid] = {
                        'id': rid,
                        'label': ent['label'],
                        'text': meta.get('text', ''),
                        'img': meta.get('img', ''),
                        'by_seed': {},
                    }
                table[rid]['by_seed'][(recipe, seed)] = ent
    return table, available


# ---------------------------------------------------------------- analysis helpers

def _bucket_counts_per_recipe_seed(
    table: dict[str, dict[str, Any]],
    pairs: list[tuple[str, int]],
) -> dict[tuple[str, int], Counter]:
    out: dict[tuple[str, int], Counter] = {p: Counter() for p in pairs}
    for entry in table.values():
        for (recipe, seed), seed_ent in entry['by_seed'].items():
            out[(recipe, seed)][seed_ent['bucket']] += 1
    return out


def _bucket_mean_std(
    counts: dict[tuple[str, int], Counter],
    recipes: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Recipe -> bucket -> (mean, std) across the available seeds."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for recipe in recipes:
        per_seed: dict[str, list[int]] = defaultdict(list)
        seed_counts = [c for (r, _), c in counts.items() if r == recipe]
        if not seed_counts:
            continue
        for b in ('B1_robust', 'B2_clean_wrong', 'B3_natural_attack_flipped',
                  'B5_composite_only_failure', 'B4_pgd_only'):
            vals = [c.get(b, 0) for c in seed_counts]
            per_seed[b] = vals
        out[recipe] = {
            b: (mean(v), pstdev(v) if len(v) > 1 else 0.0)
            for b, v in per_seed.items()
        }
    return out


def _per_recipe_majority_bucket(entry: dict[str, Any], recipe: str, seeds: list[int]) -> str | None:
    """Most-common bucket for a recipe across available seeds. None if no data."""
    seen = [entry['by_seed'][(recipe, s)]['bucket']
            for s in seeds if (recipe, s) in entry['by_seed']]
    if not seen:
        return None
    return Counter(seen).most_common(1)[0][0]


def _is_natural_failure(bucket: str | None) -> bool:
    """True if the example is naturally broken under the project's threat model.

    Includes single-cell natural failures (B3), composite failures (B5), and
    clean-wrong (B2). PGD-only failures (B4) and fully-robust (B1) are *not*
    natural failures — PGD is a worst-case oracle reported separately.
    """
    return bucket in ('B2_clean_wrong', 'B3_natural_attack_flipped',
                      'B5_composite_only_failure')


def _is_naturally_robust(bucket: str | None) -> bool:
    """Complement of `_is_natural_failure` for the recipe-level vote."""
    return bucket in ('B1_robust', 'B4_pgd_only')


def _consensus_natural_failures(
    table: dict[str, dict[str, Any]],
    recipes: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Examples where every recipe naturally fails (majority across its seeds)."""
    out = []
    for entry in table.values():
        majorities = {r: _per_recipe_majority_bucket(entry, r, seeds) for r in recipes}
        if all(_is_natural_failure(b) for b in majorities.values()):
            entry_copy = dict(entry)
            entry_copy['per_recipe_bucket'] = majorities
            out.append(entry_copy)
    return out


def _bootstrap_ci(samples: list[int], n_boot: int = 2000, ci: float = 0.95,
                  rng_seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of a 0/1-valued sample.

    samples: list of 0/1 values (e.g. label == 0 indicator for each disagreement
    example). Returns (mean, lower, upper) of the bootstrap distribution at the
    given CI level. Deterministic given rng_seed.
    """
    if not samples:
        return 0.0, 0.0, 0.0
    rng = _random.Random(rng_seed)
    n = len(samples)
    means: list[float] = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower = means[int((1 - ci) / 2 * n_boot)]
    upper = means[int((1 + ci) / 2 * n_boot) - 1]
    return sum(samples) / n, lower, upper


def _disagreement_examples(
    table: dict[str, dict[str, Any]],
    recipe_a: str,
    recipe_b: str,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(a_beats_b, b_beats_a): examples where a is B1 and b is naturally-failing, and vice versa."""
    a_wins, b_wins = [], []
    for entry in table.values():
        ba = _per_recipe_majority_bucket(entry, recipe_a, seeds)
        bb = _per_recipe_majority_bucket(entry, recipe_b, seeds)
        if _is_naturally_robust(ba) and _is_natural_failure(bb):
            entry_copy = dict(entry)
            entry_copy['_a_bucket'] = ba
            entry_copy['_b_bucket'] = bb
            a_wins.append(entry_copy)
        elif _is_naturally_robust(bb) and _is_natural_failure(ba):
            entry_copy = dict(entry)
            entry_copy['_a_bucket'] = ba
            entry_copy['_b_bucket'] = bb
            b_wins.append(entry_copy)
    return a_wins, b_wins


def _composite_only_per_recipe(
    table: dict[str, dict[str, Any]],
    recipe: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    out = []
    for entry in table.values():
        b = _per_recipe_majority_bucket(entry, recipe, seeds)
        if b == 'B5_composite_only_failure':
            entry_copy = dict(entry)
            entry_copy['_recipe_bucket'] = b
            # Surface the worst composite hit
            seed_ents = [entry['by_seed'][(recipe, s)] for s in seeds if (recipe, s) in entry['by_seed']]
            if seed_ents:
                comp_worst: dict[str, list[float]] = defaultdict(list)
                for se in seed_ents:
                    for ck, info in se.get('per_composite_worst', {}).items():
                        comp_worst[ck].append(info['attacked_prob'])
                entry_copy['_composite_worst_mean'] = {
                    ck: sum(v) / len(v) for ck, v in comp_worst.items()
                }
            out.append(entry_copy)
    return out


def _suggest_category(per_family_worst: dict[str, Any]) -> str:
    pf = per_family_worst
    if 'text' in pf:
        worst = pf['text']
        if worst.get('severity') == 'low':
            return 'text obfuscation (highly fragile — flips even at low severity)'
        if worst['attack'] in ('censoring', 'char_deletion'):
            return 'text obfuscation (caption-erasing)'
        if worst['attack'] in ('leetspeak', 'keyboard_typo', 'char_swap'):
            return 'text obfuscation (character-level)'
        return 'text-side'
    if 'typographic' in pf:
        return 'rendered-text overlay (typographic)'
    if any(k.startswith('image-') for k in pf):
        return 'image corruption'
    return 'unclear — manual review'


# ---------------------------------------------------------------- formatting

def _fmt_caption(caption: str, max_len: int = 90) -> str:
    if len(caption) <= max_len:
        return caption
    return caption[: max_len - 1] + '…'


def _format_consensus_entry(entry: dict[str, Any], recipes: list[str]) -> str:
    lines = [f"#### id={entry['id']}  (label={entry['label']})"]
    lines.append(f"- Caption: `{_fmt_caption(entry['text'])}`")
    lines.append(f"- Image: `{entry['img']}`")
    lines.append(f"- Bucket by recipe: " + ", ".join(
        f"`{r}`→{entry['per_recipe_bucket'].get(r, '—')}" for r in recipes
    ))
    # Pick a representative seed entry (first available) to show worst-cell info
    for r in recipes:
        seed_ents = [e for (rr, _), e in entry['by_seed'].items() if rr == r]
        if seed_ents:
            pf = seed_ents[0]['per_family_worst']
            if pf:
                worst_text = pf.get('text')
                if worst_text:
                    lines.append(f"- `{r}` worst text attack: "
                                 f"`{worst_text['attack']}@{worst_text['severity']}` "
                                 f"→ prob `{worst_text['attacked_prob']:.3f}`")
            break
    lines.append(f"- Suggested category: {_suggest_category(seed_ents[0]['per_family_worst']) if seed_ents else 'n/a'}")
    return '\n'.join(lines)


def _format_disagreement_entry(entry: dict[str, Any], recipe_a: str, recipe_b: str, seeds: list[int]) -> str:
    a_ent = next((entry['by_seed'][(recipe_a, s)] for s in seeds if (recipe_a, s) in entry['by_seed']), None)
    b_ent = next((entry['by_seed'][(recipe_b, s)] for s in seeds if (recipe_b, s) in entry['by_seed']), None)
    lines = [f"#### id={entry['id']}  (label={entry['label']})"]
    lines.append(f"- Caption: `{_fmt_caption(entry['text'])}`")
    if a_ent:
        lines.append(f"- `{recipe_a}` clean prob `{a_ent['clean_prob']:.3f}` (τ={a_ent['best_threshold']:.2f}), bucket `{entry['_a_bucket']}`")
    if b_ent:
        lines.append(f"- `{recipe_b}` clean prob `{b_ent['clean_prob']:.3f}` (τ={b_ent['best_threshold']:.2f}), bucket `{entry['_b_bucket']}`")
        if b_ent['per_family_worst'].get('text'):
            wt = b_ent['per_family_worst']['text']
            lines.append(f"- `{recipe_b}` worst text attack: `{wt['attack']}@{wt['severity']}` → prob `{wt['attacked_prob']:.3f}`")
    return '\n'.join(lines)


def _format_composite_only_entry(entry: dict[str, Any], recipe: str) -> str:
    lines = [f"#### id={entry['id']}  (label={entry['label']})"]
    lines.append(f"- Caption: `{_fmt_caption(entry['text'])}`")
    if entry.get('_composite_worst_mean'):
        for ck, p in sorted(entry['_composite_worst_mean'].items(), key=lambda kv: kv[1] if entry['label'] == 1 else -kv[1]):
            lines.append(f"  - `{ck}` mean attacked prob `{p:.3f}`")
    return '\n'.join(lines)


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds (default 0,1,2)')
    ap.add_argument('--recipes', default='stage1,augonly,kl,kldrop,kldrop-p015,kldrop-p050,kllowmed',
                    help='Comma-separated recipes (default all 5). Use "stage1" or "clean" for the baseline.')
    ap.add_argument('--split', default='dev', choices=('dev', 'test_seen', 'test_unseen'),
                    help='Which eval split this analysis is for. Affects caption-file lookup. Default dev.')
    ap.add_argument('--out', default=None,
                    help='Output markdown path. Default: project_planning/phase4/failure_analysis[.split].md')
    ap.add_argument(
        '--pairs',
        default='kldrop-p015:kl,kldrop-p015:kldrop,kldrop:kl',
        help='Comma-separated recipe_a:recipe_b pairs. For each pair, emit '
             '"Where A beats B" / "Where B beats A" sections with bootstrap '
             'CIs on the label distribution of the disagreement set. Default '
             'covers the Phase 10 refresh pairs (kldrop-p015 vs kl + vs kldrop, '
             'and the original kldrop vs kl for comparison).',
    )
    ap.add_argument('--bootstrap-n', type=int, default=2000,
                    help='Bootstrap resamples for the disagreement CIs (default 2000).')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    recipes = [r.strip() for r in args.recipes.split(',') if r.strip()]
    # Normalise 'clean' alias to 'stage1' so internal keys are consistent
    recipes = ['stage1' if r == 'clean' else r for r in recipes]

    captions = _load_captions(args.split)
    table, available = _build_full_table(recipes, seeds, captions, split=args.split)
    if not available:
        print("No eval JSONs found for any (recipe, seed) — nothing to do.")
        return 1
    counts = _bucket_counts_per_recipe_seed(table, available)
    bucket_stats = _bucket_mean_std(counts, recipes)

    # Cross-recipe analyses
    consensus = _consensus_natural_failures(table, recipes, seeds)
    composite_only_by_recipe: dict[str, list[dict[str, Any]]] = {
        r: _composite_only_per_recipe(table, r, seeds) for r in recipes
    }

    # Phase-10 multi-pair disagreement analysis with bootstrap CIs
    raw_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    pair_results: list[dict[str, Any]] = []
    for pair_spec in raw_pairs:
        if ':' not in pair_spec:
            print(f"  skipping malformed --pairs entry: {pair_spec!r}")
            continue
        ra, rb = (s.strip() for s in pair_spec.split(':', 1))
        if ra not in recipes or rb not in recipes:
            print(f"  skipping pair {ra}:{rb} (one or both recipes not in --recipes)")
            continue
        a_wins, b_wins = _disagreement_examples(table, ra, rb, seeds)
        a_lbl0 = [1 if e['label'] == 0 else 0 for e in a_wins]
        b_lbl1 = [1 if e['label'] == 1 else 0 for e in b_wins]
        ci_a = _bootstrap_ci(a_lbl0, n_boot=args.bootstrap_n)
        ci_b = _bootstrap_ci(b_lbl1, n_boot=args.bootstrap_n)
        pair_results.append({
            'a': ra, 'b': rb,
            'a_wins': a_wins, 'b_wins': b_wins,
            'ci_a_label0': ci_a, 'ci_b_label1': ci_b,
        })

    # ----- write -----
    out: list[str] = []
    out.append(f'# Phase 6 — Failure-case narrative ({args.split} split)\n')
    out.append(f'Aggregates {len(available)} (recipe, seed) pairs across {len(recipes)} recipes '
               f'and seeds {seeds}. Generated by `scripts/failure_analysis.py`. '
               f'Total dev examples: {len(table)}.\n')
    out.append(f'**Bucket definitions** (per recipe-seed): '
               f'`B1_robust` = clean-correct, survives all single-cell, all composite, and PGD; '
               f'`B2_clean_wrong` = wrong before any attack; '
               f'`B3_natural_attack_flipped` = clean-correct, flipped by a single-cell natural attack; '
               f'`B5_composite_only_failure` = clean-correct, robust to single-cell naturals, '
               f'flipped by a composite cell; '
               f'`B4_pgd_only` = clean-correct, robust to all naturals (single+composite), broken only by PGD ε=4/255.\n')

    # 1. Bucket counts per recipe (mean ± σ)
    out.append('## 1. Bucket counts per recipe (mean ± σ across seeds)\n')
    out.append('| Bucket | ' + ' | '.join(f'`{r}`' for r in recipes) + ' |')
    out.append('|---' + '|---' * len(recipes) + '|')
    for b, b_desc in [
        ('B1_robust', 'B1 (fully robust)'),
        ('B2_clean_wrong', 'B2 (clean-wrong)'),
        ('B3_natural_attack_flipped', 'B3 (single-cell broken)'),
        ('B5_composite_only_failure', 'B5 (composite-only broken)'),
        ('B4_pgd_only', 'B4 (PGD-only broken)'),
    ]:
        row = [b_desc]
        for r in recipes:
            stats = bucket_stats.get(r, {}).get(b, (0.0, 0.0))
            row.append(f"{stats[0]:.1f} ± {stats[1]:.1f}")
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')
    out.append('**Reading guide.** B1 + B4 = naturalistic-fully-robust examples '
               '(survives every single-cell AND every composite natural attack — matches the '
               '`Naturalistic /500` column in `phase4/robust_vs_clean.md`). '
               'B1 + B4 + B5 = the looser "single-cell only" definition '
               '(survives all 57 single-cell cells but composites may break it). '
               'B3 + B5 quantify the *naturalistic attack surface*; B2 is the irreducible '
               'clean error; B4 is the headline robustness improvement target.\n')

    # 2. Consensus residual hotspots
    out.append(f'## 2. Consensus residual hotspots — natural failures across every recipe\n')
    out.append(f'Examples whose *majority* bucket across seeds is B2 or B3 for **every** of '
               f'`{",".join(recipes)}`. These are the residual failures no recipe in our '
               f'sweep solves. n={len(consensus)}\n')
    if consensus:
        # Sort by label first, then by id for determinism
        consensus_sorted = sorted(consensus, key=lambda e: (e['label'], e['id']))
        # Show up to 15
        for e in consensus_sorted[:15]:
            out.append(_format_consensus_entry(e, recipes))
            out.append('')
        if len(consensus_sorted) > 15:
            out.append(f'(+{len(consensus_sorted) - 15} more in the consensus-failure set; '
                       f'full list available in the JSON dump below.)\n')

    # 3. Per-pair disagreement analysis with bootstrap CIs (Phase 10 refresh)
    sec_num = 3
    for pr in pair_results:
        ra, rb = pr['a'], pr['b']
        a_wins = pr['a_wins']; b_wins = pr['b_wins']
        m_a, lo_a, hi_a = pr['ci_a_label0']
        m_b, lo_b, hi_b = pr['ci_b_label1']
        out.append(f"## {sec_num}. Where `{ra}` beats `{rb}` (n={len(a_wins)})\n")
        out.append(f"Examples where `{ra}`'s majority bucket is naturally-robust (B1 or B4) "
                   f"and `{rb}`'s is naturally-failing (B2/B3/B5).")
        if a_wins:
            out.append(f"label=0 share: **{m_a * 100:.0f} %** "
                       f"(95 % CI: {lo_a * 100:.0f}–{hi_a * 100:.0f} %, n={len(a_wins)})")
        out.append('')
        a_wins_sorted = sorted(a_wins, key=lambda e: e['id'])
        for e in a_wins_sorted[:8]:
            out.append(_format_disagreement_entry(e, ra, rb, seeds))
            out.append('')
        if len(a_wins_sorted) > 8:
            out.append(f'(+{len(a_wins_sorted) - 8} more.)\n')
        sec_num += 1

        out.append(f"## {sec_num}. Where `{rb}` beats `{ra}` (n={len(b_wins)})\n")
        out.append(f"Reverse direction — `{rb}`-specific fixes.")
        if b_wins:
            out.append(f"label=1 share: **{m_b * 100:.0f} %** "
                       f"(95 % CI: {lo_b * 100:.0f}–{hi_b * 100:.0f} %, n={len(b_wins)})")
        out.append('')
        b_wins_sorted = sorted(b_wins, key=lambda e: e['id'])
        for e in b_wins_sorted[:8]:
            out.append(_format_disagreement_entry(e, ra, rb, seeds))
            out.append('')
        if len(b_wins_sorted) > 8:
            out.append(f'(+{len(b_wins_sorted) - 8} more.)\n')
        sec_num += 1

    # Composite-only failures per recipe
    out.append(f'## {sec_num}. Composite-only failures (B5) — examples robust to single-cell, broken by composites\n')
    out.append('Per recipe count (majority across seeds):\n')
    out.append('| Recipe | B5 count |')
    out.append('|---|---:|')
    for r in recipes:
        out.append(f"| `{r}` | {len(composite_only_by_recipe[r])} |")
    out.append('')
    out.append('Representative B5 examples for each recipe (up to 5):\n')
    for r in recipes:
        lst = composite_only_by_recipe[r]
        if not lst:
            continue
        out.append(f'### `{r}` — B5 examples (n={len(lst)})')
        for e in sorted(lst, key=lambda e: e['id'])[:5]:
            out.append(_format_composite_only_entry(e, r))
            out.append('')

    # Limitations
    sec_num += 1
    out.append(f'## {sec_num}. Limitations of this analysis\n')
    out.append('- Single eval split (`%s`); recipe-seed counts are point estimates per seed averaged across '
               '3 seeds — small samples.' % args.split)
    out.append('- `B5_composite_only_failure` and `B4_pgd_only` are *strict* — they require survival of '
               'every single-cell natural attack first. An example flipped by both a single-cell and a '
               'composite goes into B3, not B5.')
    out.append('- "Where kldrop beats kl" uses majority bucket per recipe (so 2/3 seeds count). Seed-level '
               'noise inside the per-recipe vote is hidden.')
    out.append('- Suggested categories are heuristic tags from `_suggest_category`; the narrative report '
               'should refine them by hand.')
    out.append('- Clean labels themselves are noisy on a small fraction of the dev set; some "B2" entries '
               'may be label disputes, not model failures.\n')

    # ----- emit -----
    out_path = Path(args.out) if args.out else OUT_DIR / (
        'failure_analysis.md' if args.split == 'dev' else f'failure_analysis.{args.split}.md'
    )
    out_path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"  (recipe, seed) pairs available: {available}")
    print(f"  total dev examples: {len(table)}")
    print(f"  consensus natural failures: {len(consensus)}")
    for pr in pair_results:
        m_a = pr['ci_a_label0'][0]
        m_b = pr['ci_b_label1'][0]
        print(f"  {pr['a']} beats {pr['b']}: {len(pr['a_wins'])} "
              f"(label=0 share {m_a * 100:.0f}%) | "
              f"{pr['b']} beats {pr['a']}: {len(pr['b_wins'])} "
              f"(label=1 share {m_b * 100:.0f}%)")
    for r in recipes:
        print(f"  B5 ({r}): {len(composite_only_by_recipe[r])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
