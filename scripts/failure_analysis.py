"""Phase 4-S3: per-example failure analysis.

Reads the per-example records inside ``cluster-results/perturbed-stage1-seed0/``
and ``cluster-results/whitebox-stage1-seed0/``, joins them with the dev-split
captions (pulled to ``data/processed/dev_meta/dev.jsonl``), and writes a
single curated markdown file under ``project_planning/phase4/`` listing 5
representative examples in each of four buckets:

    B1  clean-correct                       — model gets it right pre-attack
    B2  clean-wrong                         — model fails before any attack
    B3  natural-attack-flipped              — flipped by ≥ 1 text/image cell
    B4  PGD-only                            — only the white-box attack flips it

For each example we record id, label, original caption, image path, the
clean prob, the attacked probs by family (worst-cell per family), and a
*candidate* failure-category tag. The human curates final tags in the
markdown afterwards.

No GPU, no cluster — uses files already on disk.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'cluster-results'
DEV_META = REPO / 'data' / 'processed' / 'dev_meta' / 'dev.jsonl'
OUT_DIR = REPO / 'project_planning' / 'phase4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_ATTACKS = {
    'leetspeak', 'char_deletion', 'char_swap', 'spacing',
    'punctuation', 'case_noise', 'censoring', 'keyboard_typo',
}
IMAGE_ATTACKS = {
    'gaussian_noise', 'blur', 'compression',
    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
    'translation', 'crop', 'occlusion', 'typographic',
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


def _load_dev_captions() -> dict[str, dict[str, Any]]:
    """Map zero-padded 5-digit id → {label, text, img_path}."""
    out: dict[str, dict[str, Any]] = {}
    for line in DEV_META.read_text().splitlines():
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


def _load_jsons() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    perturbed = json.loads((RESULTS / 'perturbed-stage1-seed0' / 'perturbed_eval.json').read_text())
    whitebox  = json.loads((RESULTS / 'whitebox-stage1-seed0'  / 'whitebox_eval.json').read_text())
    typo_path = RESULTS / 'perturbed-typographic-stage1-seed0' / 'perturbed_eval.json'
    typographic = json.loads(typo_path.read_text()) if typo_path.exists() else None
    return perturbed, whitebox, typographic


def _load_robust_jsons(variant: str = 'kl') -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Try to load robust seed-0 perturbed + whitebox JSONs. Returns (perturbed, whitebox) or (None, None)."""
    for prefix in (f'perturbed-robust-{variant}-seed0', f'perturbed-train-robust-{variant}-seed0'):
        p = RESULTS / prefix / 'perturbed_eval.json'
        if p.exists():
            pert = json.loads(p.read_text())
            break
    else:
        pert = None
    for prefix in (f'whitebox-robust-{variant}-seed0', f'whitebox-train-robust-{variant}-seed0'):
        p = RESULTS / prefix / 'whitebox_eval.json'
        if p.exists():
            wb = json.loads(p.read_text())
            break
    else:
        wb = None
    return pert, wb


def _build_example_table(perturbed, whitebox, typographic, captions) -> dict[str, dict[str, Any]]:
    """For each example id, collect {clean_prob, worst attacked prob per family,
    PGD attacked prob at eps=4/255}.
    """
    table: dict[str, dict[str, Any]] = {}
    best_t = perturbed['clean']['best_threshold']
    # clean probs are recorded in every cell's examples list — pick from first cell.
    first_cell = perturbed['cells'][0]
    for ex in first_cell['examples']:
        rid = str(ex['id']).zfill(5)
        table[rid] = {
            'id': rid,
            'label': int(ex['label']),
            'clean_prob': float(ex['clean_prob']),
            'caption': captions.get(rid, {}).get('text', ''),
            'img_path': captions.get(rid, {}).get('img', ''),
            'best_threshold': best_t,
            'per_family_worst': {},  # family -> (attack, severity, attacked_prob)
            'pgd_at_4_over_255': None,
        }
    # Walk perturbed cells: track worst attacked prob (farthest from clean side)
    for cell in perturbed['cells']:
        fam = _attack_family(cell['attack'])
        if fam == 'other':
            continue
        for ex in cell['examples']:
            rid = str(ex['id']).zfill(5)
            if rid not in table:
                continue
            entry = table[rid]
            label = entry['label']
            ap = float(ex['attacked_prob'])
            # "Worst" = furthest from the true label's side of best_t
            # For label=1 we want the lowest attacked_prob; for label=0 the highest.
            current = entry['per_family_worst'].get(fam)
            if current is None:
                entry['per_family_worst'][fam] = {
                    'attack': cell['attack'], 'severity': cell['severity_level'],
                    'attacked_prob': ap,
                }
            else:
                if (label == 1 and ap < current['attacked_prob']) or (
                    label == 0 and ap > current['attacked_prob']
                ):
                    entry['per_family_worst'][fam] = {
                        'attack': cell['attack'], 'severity': cell['severity_level'],
                        'attacked_prob': ap,
                    }
    # Walk typographic if present
    if typographic is not None:
        for cell in typographic['cells']:
            for ex in cell['examples']:
                rid = str(ex['id']).zfill(5)
                if rid not in table:
                    continue
                entry = table[rid]
                label = entry['label']
                ap = float(ex['attacked_prob'])
                current = entry['per_family_worst'].get('typographic')
                if current is None or (
                    label == 1 and ap < current['attacked_prob']
                ) or (label == 0 and ap > current['attacked_prob']):
                    entry['per_family_worst']['typographic'] = {
                        'attack': cell['attack'], 'severity': cell['severity_level'],
                        'attacked_prob': ap,
                    }
    # Walk whitebox: pull pgd at eps=4/255
    for cell in whitebox['cells']:
        if cell['attack'] != 'pgd' or cell['epsilon_numerator'] != 4:
            continue
        for ex in cell['examples']:
            rid = str(ex['id']).zfill(5)
            if rid in table:
                table[rid]['pgd_at_4_over_255'] = float(ex['attacked_prob'])
    return table


def _annotate_robust(
    table: dict[str, dict[str, Any]],
    robust_perturbed: dict[str, Any] | None,
    robust_whitebox: dict[str, Any] | None,
) -> None:
    """Mutates ``table``: adds ``robust_clean_prob``, ``robust_natural_flipped``,
    ``robust_pgd_at_4_over_255``, and a derived ``robust_status`` per entry.

    ``robust_status`` is one of:
        - 'fixed'         : clean-baseline classed it as B2/B3/B4 but robust kl passes the same threshold suite.
        - 'still_failed'  : clean-baseline classed it as B2/B3/B4 and robust kl also fails the same suite.
        - 'unchanged'     : robust kl bucket matches clean baseline bucket (e.g. both B1, or both B4).
        - 'new_failure'   : was B1 in clean baseline, fails under robust kl.
        - 'no_data'       : robust ckpt not yet available.
    """
    if robust_perturbed is None:
        for entry in table.values():
            entry['robust_status'] = 'no_data'
        return
    r_best_t = robust_perturbed['clean']['best_threshold']
    # Pull robust clean probs (per-example) from first cell's examples.
    if not robust_perturbed['cells']:
        for entry in table.values():
            entry['robust_status'] = 'no_data'
        return
    robust_clean_probs: dict[str, float] = {}
    for ex in robust_perturbed['cells'][0]['examples']:
        rid = str(ex['id']).zfill(5)
        robust_clean_probs[rid] = float(ex['clean_prob'])
    # Track per-example natural-attack flip under robust ckpt
    flipped_natural: dict[str, bool] = {rid: False for rid in robust_clean_probs}
    for cell in robust_perturbed['cells']:
        for ex in cell['examples']:
            rid = str(ex['id']).zfill(5)
            label = int(ex['label'])
            ap = float(ex['attacked_prob'])
            pred = 1 if ap >= r_best_t else 0
            if pred != label:
                flipped_natural[rid] = True
    # Whitebox PGD ε=4 if present
    pgd_4: dict[str, float] = {}
    if robust_whitebox is not None:
        for cell in robust_whitebox['cells']:
            if cell['attack'] != 'pgd' or cell['epsilon_numerator'] != 4:
                continue
            for ex in cell['examples']:
                rid = str(ex['id']).zfill(5)
                pgd_4[rid] = float(ex['attacked_prob'])
    for rid, entry in table.items():
        label = entry['label']
        rcp = robust_clean_probs.get(rid)
        entry['robust_clean_prob'] = rcp
        entry['robust_best_threshold'] = r_best_t
        entry['robust_natural_flipped'] = flipped_natural.get(rid, None)
        entry['robust_pgd_at_4_over_255'] = pgd_4.get(rid)
        if rcp is None:
            entry['robust_status'] = 'no_data'
            continue
        robust_clean_pred = 1 if rcp >= r_best_t else 0
        robust_clean_correct = robust_clean_pred == label
        robust_natural_ok = (entry['robust_natural_flipped'] is False)
        if entry['robust_pgd_at_4_over_255'] is not None:
            pgd_pred = 1 if entry['robust_pgd_at_4_over_255'] >= r_best_t else 0
            robust_pgd_ok = pgd_pred == label
        else:
            robust_pgd_ok = None
        # Clean-baseline bucket of this entry
        clean_t = entry['best_threshold']
        cp = entry['clean_prob']
        clean_baseline_b1 = (
            (1 if cp >= clean_t else 0) == label
            and not any(
                (1 if info['attacked_prob'] >= clean_t else 0) != label
                for info in entry['per_family_worst'].values()
            )
            and (entry['pgd_at_4_over_255'] is None or (1 if entry['pgd_at_4_over_255'] >= clean_t else 0) == label)
        )
        robust_b1 = robust_clean_correct and robust_natural_ok and (robust_pgd_ok is None or robust_pgd_ok)
        if clean_baseline_b1 and robust_b1:
            entry['robust_status'] = 'unchanged'
        elif clean_baseline_b1 and not robust_b1:
            entry['robust_status'] = 'new_failure'
        elif (not clean_baseline_b1) and robust_b1:
            entry['robust_status'] = 'fixed'
        else:
            entry['robust_status'] = 'still_failed'


def _bucketize(table: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in table.values():
        label = entry['label']
        cp = entry['clean_prob']
        t = entry['best_threshold']
        clean_pred = 1 if cp >= t else 0
        clean_correct = clean_pred == label
        # Any natural-attack flipped?
        flipped_by_natural = False
        for fam, info in entry['per_family_worst'].items():
            ap = info['attacked_prob']
            attacked_pred = 1 if ap >= t else 0
            if attacked_pred != label:
                flipped_by_natural = True
                break
        # PGD flipped?
        pgd_prob = entry['pgd_at_4_over_255']
        pgd_flipped = pgd_prob is not None and (
            (1 if pgd_prob >= t else 0) != label
        )
        if not clean_correct:
            buckets['B2_clean_wrong'].append(entry)
        elif flipped_by_natural:
            buckets['B3_natural_attack_flipped'].append(entry)
        elif pgd_flipped:
            buckets['B4_pgd_only'].append(entry)
        else:
            buckets['B1_clean_correct_robust'].append(entry)
    return buckets


def _suggest_category(entry: dict[str, Any]) -> str:
    """Heuristic tag — to be hand-edited in the markdown afterwards."""
    pf = entry['per_family_worst']
    if 'text' in pf:
        worst_text = pf['text']
        if worst_text['severity'] == 'low':
            return 'text obfuscation (highly fragile — even low severity flips)'
        if worst_text['attack'] in ('censoring', 'char_deletion'):
            return 'text obfuscation (caption-erasing)'
        if worst_text['attack'] in ('leetspeak', 'keyboard_typo', 'char_swap'):
            return 'text obfuscation (character-level)'
    if 'typographic' in pf:
        return 'typographic / rendered-text attack'
    if any(k.startswith('image-') for k in pf):
        return 'image corruption'
    return 'unclear — needs manual review'


def _format_entry(entry: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"#### id={entry['id']}  (label={entry['label']})")
    lines.append('')
    lines.append(f"- **Caption**: `{entry['caption']}`")
    lines.append(f"- **Image**: `{entry['img_path']}`")
    lines.append(f"- **Clean prob**: `{entry['clean_prob']:.4f}` (best-τ={entry['best_threshold']:.2f})")
    if entry['per_family_worst']:
        lines.append('- **Worst attacked prob per family**:')
        for fam, info in sorted(entry['per_family_worst'].items()):
            lines.append(
                f"    - {fam}: `{info['attacked_prob']:.4f}` "
                f"(via `{info['attack']}` @ {info['severity']})"
            )
    if entry['pgd_at_4_over_255'] is not None:
        lines.append(f"- **PGD ε=4/255 attacked prob**: `{entry['pgd_at_4_over_255']:.4f}`")
    lines.append(f"- **Suggested category**: {_suggest_category(entry)}")
    rstatus = entry.get('robust_status')
    if rstatus and rstatus != 'no_data':
        lines.append(f"- **Robust-kl clean prob**: `{entry['robust_clean_prob']:.4f}` "
                     f"(best-τ={entry['robust_best_threshold']:.2f})")
        rpgd = entry.get('robust_pgd_at_4_over_255')
        if rpgd is not None:
            lines.append(f"- **Robust-kl PGD ε=4/255**: `{rpgd:.4f}`")
        lines.append(f"- **Robust-kl status**: `{rstatus}`")
    return '\n'.join(lines)


def main() -> int:
    captions = _load_dev_captions()
    perturbed, whitebox, typographic = _load_jsons()
    table = _build_example_table(perturbed, whitebox, typographic, captions)
    robust_perturbed, robust_whitebox = _load_robust_jsons(variant='kl')
    _annotate_robust(table, robust_perturbed, robust_whitebox)
    buckets = _bucketize(table)
    # Robust-status aggregate counts (only meaningful when robust data exists)
    robust_status_counts: dict[str, int] = defaultdict(int)
    for entry in table.values():
        robust_status_counts[entry.get('robust_status', 'no_data')] += 1

    rng = random.Random(0)
    bucket_order = [
        ('B1_clean_correct_robust',   'Clean-correct, naturally robust'),
        ('B2_clean_wrong',            'Clean-wrong (intrinsic miss)'),
        ('B3_natural_attack_flipped', 'Clean-correct → flipped by ≥ 1 natural attack'),
        ('B4_pgd_only',               'Clean-correct, natural-robust, broken by PGD only'),
    ]
    out_lines: list[str] = []
    out_lines.append('# Phase 4-S3 — Failure-case analysis (multimodal seed-0)')
    out_lines.append('')
    out_lines.append('5 representative examples per bucket, drawn deterministically '
                     'with `random.Random(0)`. Phase 6 will append "robust-fixed" / '
                     '"robust-still-failed" columns once Phase 5 ships.')
    out_lines.append('')
    out_lines.append('## Bucket counts')
    out_lines.append('')
    out_lines.append('| Bucket | Description | Count |')
    out_lines.append('|---|---|---:|')
    for k, desc in bucket_order:
        out_lines.append(f"| `{k}` | {desc} | {len(buckets.get(k, []))} |")
    out_lines.append('')

    # --- Naturalistic-only survival (primary threat model) ---
    n_clean_natural_survivors = len(buckets.get('B1_clean_correct_robust', [])) + \
                                len(buckets.get('B4_pgd_only', []))
    out_lines.append('## Naturalistic-only survival (primary threat model)')
    out_lines.append('')
    out_lines.append('Count of dev examples that are clean-correct **and** survive every naturalistic perturbation cell. '
                     'PGD ε=4/255 is *excluded* — it is a worst-case oracle attack reported separately. '
                     'This is the metric aligned with the project\'s "typical adversarial user" threat model.')
    out_lines.append('')
    out_lines.append(f"- Clean baseline (seed 0): **{n_clean_natural_survivors} / {len(table)}** "
                     f"({n_clean_natural_survivors * 100.0 / max(1, len(table)):.1f} %).")
    if robust_perturbed is not None:
        # Compute robust natural-only survivors from the annotated table.
        r_best_t = robust_perturbed['clean']['best_threshold']
        nat_survivors = 0
        for entry in table.values():
            rcp = entry.get('robust_clean_prob')
            if rcp is None:
                continue
            r_clean_pred = 1 if rcp >= r_best_t else 0
            if r_clean_pred == entry['label'] and entry.get('robust_natural_flipped') is False:
                nat_survivors += 1
        out_lines.append(f"- Robust-kl (seed 0): **{nat_survivors} / {len(table)}** "
                         f"({nat_survivors * 100.0 / max(1, len(table)):.1f} %). "
                         f"Δ vs clean: {nat_survivors - n_clean_natural_survivors:+d} examples.")
    out_lines.append('')

    if robust_status_counts.get('no_data', 0) < len(table):
        out_lines.append('## Robust-kl status counts (seed 0, natural+PGD definition)')
        out_lines.append('')
        out_lines.append('Each dev example is re-bucketed under the robust-kl checkpoint and compared to '
                         'its clean-baseline bucket. *Bucket definition includes PGD ε=4/255 survival;* '
                         'see the previous section for the naturalistic-only headline.')
        out_lines.append('')
        out_lines.append('| Robust status | Count |')
        out_lines.append('|---|---:|')
        for status in ('unchanged', 'fixed', 'still_failed', 'new_failure', 'no_data'):
            count = robust_status_counts.get(status, 0)
            if count:
                out_lines.append(f"| `{status}` | {count} |")
        out_lines.append('')

    for k, desc in bucket_order:
        examples = buckets.get(k, [])
        if not examples:
            continue
        # Sort by clean_prob distance from threshold (most-confident-then-flipped first)
        examples_sorted = sorted(
            examples,
            key=lambda e: abs(e['clean_prob'] - e['best_threshold']),
            reverse=True,
        )
        sample = examples_sorted[:5] if len(examples_sorted) >= 5 else examples_sorted
        out_lines.append(f"## {desc} (`{k}`)")
        out_lines.append('')
        out_lines.append(f"Showing top-5 by clean-confidence (n={len(examples)} total in bucket).")
        out_lines.append('')
        for entry in sample:
            out_lines.append(_format_entry(entry))
            out_lines.append('')

    out_path = OUT_DIR / 'failure_analysis.md'
    out_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    print(f"wrote {out_path.relative_to(REPO)}")
    for k, desc in bucket_order:
        print(f"  {k}: {len(buckets.get(k, []))}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
