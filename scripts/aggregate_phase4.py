"""Phase-4 aggregator: turn raw eval JSON files into tables and figures.

Reads every ``cluster-results/perturbed-*/perturbed_eval.json`` and
``cluster-results/whitebox-*/whitebox_eval.json`` and writes:

    project_planning/phase4/
        perturbed_table.md   # (model, attack, severity) cells
        whitebox_table.md    # (model, attack, epsilon) cells
        worst_case.md        # DoD bullets 5 + 6
        figures/heatmap_perturbed_seed{0,1,2}.png
        figures/curve_whitebox.png
        figures/severity_curves.png

Multimodal perturbation cells are aggregated across the 3 ``stage1-seed*``
runs (mean ± σ). Unimodal cells are single-seed point estimates.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'cluster-results'
OUT = REPO / 'project_planning' / 'phase4'
FIGS = OUT / 'figures'

# Stable ordering for output tables / plots.
TEXT_ATTACKS = [
    'leetspeak', 'char_deletion', 'char_swap', 'spacing',
    'punctuation', 'case_noise', 'censoring', 'keyboard_typo',
]
IMAGE_ATTACKS = [
    'gaussian_noise', 'blur', 'compression',
    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
    'translation', 'crop', 'occlusion',
]
SEVERITIES = ['low', 'medium', 'high']


def _extract_split(key: str) -> tuple[str, str]:
    """Split a dirname-derived key (e.g. 'robust-kl-test-seen-seed0' or 'stage1-seed0')
    into (split, model_key) where split ∈ {dev, test_seen, test_unseen}.

    Phase-7 job naming convention: ``{kind}-{recipe-key}-{split-hyphen}-seed{N}``,
    where split-hyphen is 'test-seen' or 'test-unseen'. Legacy (Phase 5/5c) dev
    jobs have no split tag and resolve to split='dev'.
    """
    m = re.match(r'^(.+?)-test-(seen|unseen)-seed(\d+)$', key)
    if m:
        prefix, kind, seed = m.group(1), m.group(2), m.group(3)
        return f'test_{kind}', f'{prefix}-seed{seed}'
    return 'dev', key


def _load_perturbed() -> dict[tuple[str, str], dict[str, Any]]:
    """{(split, model_key): payload}.

    `split ∈ {dev, test_seen, test_unseen}`. `model_key` is e.g. 'stage1-seed0',
    'robust-kl-seed0', 'baseline-text' — independent of split.
    """
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for d in sorted(RESULTS.glob('perturbed-*')):
        if 'smoke' in d.name or 'inspect' in d.name:
            continue
        f = d / 'perturbed_eval.json'
        if not f.exists():
            continue
        m = re.match(r'perturbed-(.+)', d.name)
        raw_key = m.group(1) if m else d.name
        split, key = _extract_split(raw_key)
        by_pair[(split, key)] = json.loads(f.read_text())
    return by_pair


def _load_modality_ablation() -> dict[tuple[str, str], dict[str, Any]]:
    """{(split, model_key): payload} for ``cluster-results/modality-ablation-*``."""
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for d in sorted(RESULTS.glob('modality-ablation-*')):
        if 'smoke' in d.name:
            continue
        f = d / 'modality_ablation.json'
        if not f.exists():
            continue
        m = re.match(r'modality-ablation-(.+)', d.name)
        raw_key = m.group(1) if m else d.name
        split, key = _extract_split(raw_key)
        by_pair[(split, key)] = json.loads(f.read_text())
    return by_pair


def _filter_by_split(payloads: dict[tuple[str, str], dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    """Slice the (split, key) dict down to a flat {key: payload} for one split."""
    return {k: v for (s, k), v in payloads.items() if s == split}


def _available_splits(*payload_dicts: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    """Stable-ordered list of splits present in any payload dict."""
    order = ['dev', 'test_seen', 'test_unseen']
    seen = {s for d in payload_dicts for (s, _) in d}
    return [s for s in order if s in seen]


def _split_suffix(split: str) -> str:
    """File-name suffix for outputs of a given split. Dev keeps its original name."""
    return '' if split == 'dev' else f'.{split}'


def _is_robust_key(key: str) -> tuple[bool, str | None]:
    """Identify a robust-trained run; return (is_robust, variant).

    Recipes: 'augonly', 'kl' (KL_full + KL_image_branch),
    'kldrop' (KL variant + per-example text-modality dropout p=0.30),
    'kllowmed' (Phase 5c: kl with severity restricted to low+medium),
    'kldrop-p015' / 'kldrop-p050' (Phase 9b dropout-rate sweep — p=0.15
    and p=0.50 alongside the default p=0.30 in `kldrop`)."""
    if key.startswith('robust-kldrop-p015-seed') or key.startswith('train-robust-kldrop-p015-seed'):
        return True, 'kldrop-p015'
    if key.startswith('robust-kldrop-p050-seed') or key.startswith('train-robust-kldrop-p050-seed'):
        return True, 'kldrop-p050'
    if key.startswith('robust-kllowmed-seed') or key.startswith('train-robust-kllowmed-seed'):
        return True, 'kllowmed'
    if key.startswith('robust-kldrop-seed') or key.startswith('train-robust-kldrop-seed'):
        return True, 'kldrop'
    if key.startswith('robust-augonly-seed') or key.startswith('train-robust-augonly-seed'):
        return True, 'augonly'
    if key.startswith('robust-kl-seed') or key.startswith('train-robust-kl-seed'):
        return True, 'kl'
    return False, None


def _load_whitebox() -> dict[tuple[str, str], dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for d in sorted(RESULTS.glob('whitebox-*')):
        if 'smoke' in d.name:
            continue
        f = d / 'whitebox_eval.json'
        if not f.exists():
            continue
        m = re.match(r'whitebox-(.+)', d.name)
        raw_key = m.group(1) if m else d.name
        split, key = _extract_split(raw_key)
        by_pair[(split, key)] = json.loads(f.read_text())
    return by_pair


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float('nan'), float('nan')
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def _fmt(mean: float, std: float | None = None, decimals: int = 4) -> str:
    if isinstance(mean, float) and math.isnan(mean):
        return 'nan'
    if std is None or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ----------------------------------------------------------------------------
# Naturalistic perturbation table
# ----------------------------------------------------------------------------

def _model_class(key: str) -> str:
    if key.startswith('stage1-seed'):
        return 'multimodal'
    if 'text' in key:
        return 'text'
    if 'image' in key:
        return 'image'
    return 'unknown'


def _aggregate_multimodal_cells(payloads: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Group cells by (attack, severity), return mean ± σ across seeds."""
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    clean_aurocs: list[float] = []
    clean_f1s: list[float] = []
    for p in payloads:
        clean_aurocs.append(p['clean']['at_best_threshold']['auroc'])
        clean_f1s.append(p['clean']['at_best_threshold']['macro_f1'])
        for c in p['cells']:
            bucket[(c['attack'], c['severity_level'])].append(c)
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    cm, cs = _mean_std(clean_aurocs)
    fm, fs = _mean_std(clean_f1s)
    agg['__clean__'] = {'auroc': (cm, cs), 'macro_f1': (fm, fs)}
    for key, cells in bucket.items():
        agg[key] = {
            'attacked_auroc': _mean_std([c['attacked_at_best_threshold']['auroc'] for c in cells]),
            'attacked_f1':    _mean_std([c['attacked_at_best_threshold']['macro_f1'] for c in cells]),
            'gap_auroc':      _mean_std([c['robustness_gap']['auroc'] for c in cells]),
            'gap_f1':         _mean_std([c['robustness_gap']['macro_f1'] for c in cells]),
            'asr':            _mean_std([c['attack_success_rate'] for c in cells]),
            'n':              cells[0]['n'],
        }
    return agg


def _write_perturbed_table(perturbed: dict[str, dict[str, Any]], split: str = 'dev') -> Path:
    out = OUT / f'perturbed_table{_split_suffix(split)}.md'
    lines: list[str] = []
    lines.append('# Phase 4 — Naturalistic perturbation benchmark')
    lines.append('')
    lines.append('All numbers on the 500-example dev split at the per-run best macro-F1 threshold.')
    lines.append('Multimodal rows aggregate `stage1-seed{0,1,2}` (mean ± σ).')
    lines.append('Unimodal rows are single-seed point estimates.')
    lines.append('')

    # --- Multimodal ---
    multimodal_payloads = [p for k, p in perturbed.items() if k.startswith('stage1-seed')]
    if multimodal_payloads:
        agg = _aggregate_multimodal_cells(multimodal_payloads)
        clean = agg.pop('__clean__')
        lines.append(f"## Multimodal (n={len(multimodal_payloads)} seeds)")
        lines.append('')
        lines.append(f"Clean AUROC: **{_fmt(*clean['auroc'])}** · Clean macro-F1: **{_fmt(*clean['macro_f1'])}**")
        lines.append('')
        lines.append('| Attack | Severity | Attacked AUROC | Attacked F1 | ΔAUROC | ΔF1 | ASR |')
        lines.append('|---|---|---:|---:|---:|---:|---:|')
        for atk in TEXT_ATTACKS + IMAGE_ATTACKS:
            for sev in SEVERITIES:
                if (atk, sev) not in agg:
                    continue
                a = agg[(atk, sev)]
                lines.append(
                    f"| {atk} | {sev} | {_fmt(*a['attacked_auroc'])} | {_fmt(*a['attacked_f1'])} "
                    f"| {_fmt(*a['gap_auroc'])} | {_fmt(*a['gap_f1'])} | {_fmt(*a['asr'])} |"
                )
        lines.append('')

    # --- Unimodal ---
    for key, p in perturbed.items():
        if key.startswith('stage1-seed'):
            continue
        cls = _model_class(key)
        clean = p['clean']['at_best_threshold']
        lines.append(f"## {cls.capitalize()}-only baseline (`{key}`)")
        lines.append('')
        lines.append(
            f"Clean AUROC: **{clean['auroc']:.4f}** · Clean macro-F1: **{clean['macro_f1']:.4f}** "
            f"· best-τ = {p['clean']['best_threshold']:.2f}"
        )
        lines.append('')
        lines.append('| Attack | Severity | Attacked AUROC | Attacked F1 | ΔAUROC | ΔF1 | ASR |')
        lines.append('|---|---|---:|---:|---:|---:|---:|')
        for c in p['cells']:
            a = c['attacked_at_best_threshold']
            lines.append(
                f"| {c['attack']} | {c['severity_level']} | {a['auroc']:.4f} | {a['macro_f1']:.4f} "
                f"| {c['robustness_gap']['auroc']:+.4f} | {c['robustness_gap']['macro_f1']:+.4f} "
                f"| {c['attack_success_rate']:.4f} |"
            )
        lines.append('')

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


# ----------------------------------------------------------------------------
# White-box table
# ----------------------------------------------------------------------------

def _aggregate_multimodal_whitebox(payloads: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    bucket: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    clean_aurocs: list[float] = []
    clean_f1s: list[float] = []
    for p in payloads:
        clean_aurocs.append(p['clean']['at_best_threshold']['auroc'])
        clean_f1s.append(p['clean']['at_best_threshold']['macro_f1'])
        for c in p['cells']:
            bucket[(c['attack'], c['epsilon_numerator'])].append(c)
    agg: dict[tuple[str, int], dict[str, Any]] = {}
    agg['__clean__'] = {
        'auroc': _mean_std(clean_aurocs),
        'macro_f1': _mean_std(clean_f1s),
    }
    for key, cells in bucket.items():
        agg[key] = {
            'attacked_auroc': _mean_std([c['attacked_at_best_threshold']['auroc'] for c in cells]),
            'attacked_f1':    _mean_std([c['attacked_at_best_threshold']['macro_f1'] for c in cells]),
            'gap_auroc':      _mean_std([c['robustness_gap']['auroc'] for c in cells]),
            'asr':            _mean_std([c['attack_success_rate'] for c in cells]),
            'pgd_steps':      cells[0].get('pgd_steps'),
        }
    return agg


def _write_whitebox_table(whitebox: dict[str, dict[str, Any]], split: str = 'dev') -> Path:
    out = OUT / f'whitebox_table{_split_suffix(split)}.md'
    lines: list[str] = []
    lines.append('# Phase 4 — White-box image attacks (FGSM / PGD)')
    lines.append('')
    lines.append('All numbers on the 500-example dev split at the per-run best macro-F1 threshold.')
    lines.append('PGD uses random L∞ start, step size α = ε/4, 10 iterations.')
    lines.append('Multimodal rows aggregate `stage1-seed{0,1,2}` (mean ± σ).')
    lines.append('')

    # --- Multimodal ---
    mm_payloads = [p for k, p in whitebox.items() if k.startswith('stage1-seed')]
    if mm_payloads:
        agg = _aggregate_multimodal_whitebox(mm_payloads)
        clean = agg.pop('__clean__')
        lines.append(f"## Multimodal (n={len(mm_payloads)} seeds)")
        lines.append('')
        lines.append(f"Clean AUROC: **{_fmt(*clean['auroc'])}** · Clean macro-F1: **{_fmt(*clean['macro_f1'])}**")
        lines.append('')
        lines.append('| Attack | ε (×255) | Attacked AUROC | Attacked F1 | ΔAUROC | ASR |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for atk in ('fgsm', 'pgd'):
            for eps_num in sorted({k[1] for k in agg if k[0] == atk}):
                a = agg[(atk, eps_num)]
                lines.append(
                    f"| {atk.upper()} | {eps_num} | {_fmt(*a['attacked_auroc'])} | {_fmt(*a['attacked_f1'])} "
                    f"| {_fmt(*a['gap_auroc'])} | {_fmt(*a['asr'])} |"
                )
        lines.append('')

    # --- Image-only baseline ---
    for key, p in whitebox.items():
        if key.startswith('stage1-seed'):
            continue
        clean = p['clean']['at_best_threshold']
        lines.append(f"## Image-only baseline (`{key}`)")
        lines.append('')
        lines.append(
            f"Clean AUROC: **{clean['auroc']:.4f}** · Clean macro-F1: **{clean['macro_f1']:.4f}**"
        )
        lines.append('')
        lines.append('| Attack | ε (×255) | Attacked AUROC | Attacked F1 | ΔAUROC | ASR |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for c in p['cells']:
            a = c['attacked_at_best_threshold']
            lines.append(
                f"| {c['attack'].upper()} | {c['epsilon_numerator']} | {a['auroc']:.4f} | "
                f"{a['macro_f1']:.4f} | {c['robustness_gap']['auroc']:+.4f} | {c['attack_success_rate']:.4f} |"
            )
        lines.append('')

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


# ----------------------------------------------------------------------------
# Worst-case analysis (DoD bullets 5 + 6)
# ----------------------------------------------------------------------------

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


def _write_worst_case(perturbed: dict[str, dict[str, Any]], whitebox: dict[str, dict[str, Any]], split: str = 'dev') -> Path:
    out = OUT / f'worst_case{_split_suffix(split)}.md'
    lines: list[str] = []
    lines.append('# Phase 4 — Worst-case + modality fragility')
    lines.append('')
    lines.append('Answers DoD bullets 5 ("worst-case across families") and 6 ("which modality is most fragile").')
    lines.append('')

    # --- Worst (attack, severity) per family, per model ---
    for key, p in perturbed.items():
        if key.startswith('stage1-seed') and key != 'stage1-seed0':
            continue  # one representative seed for the worst-cell table; the table itself averages
        cls = _model_class(key)
        title = 'Multimodal (seed 0)' if key == 'stage1-seed0' else f"{cls.capitalize()}-only (`{key}`)"
        lines.append(f"## Worst cells — {title}")
        lines.append('')
        lines.append('| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |')
        lines.append('|---|---|---:|---:|---:|')
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in p['cells']:
            by_family[_attack_family(c['attack'])].append(c)
        for fam in ('text', 'image-pixel', 'image-photometric', 'image-geometric'):
            cells = by_family.get(fam, [])
            if not cells:
                continue
            worst = min(cells, key=lambda c: c['attacked_at_best_threshold']['auroc'])
            a = worst['attacked_at_best_threshold']
            lines.append(
                f"| {fam} | {worst['attack']} ({worst['severity_level']}) | {a['auroc']:.4f} | "
                f"{worst['robustness_gap']['auroc']:+.4f} | {worst['attack_success_rate']:.4f} |"
            )
        lines.append('')

    # --- Per-modality fragility ranking (mean ΔAUROC across all attacks-of-its-relevance, high severity) ---
    lines.append('## Per-modality fragility — average ΔAUROC at high severity')
    lines.append('')
    lines.append('Average drop in AUROC (clean − attacked) across high-severity cells of each family.')
    lines.append('A larger drop ⇒ more fragile on that attack family. "n/a" means the model has no')
    lines.append("input modality the attack family touches (e.g. text-only model under image attacks).")
    lines.append('')
    lines.append('| Model | Text high-sev | Image-pixel high-sev | Image-photometric high-sev | Image-geometric high-sev |')
    lines.append('|---|---:|---:|---:|---:|')

    def _avg_gap_family(p: dict[str, Any], family: str, severity: str) -> float | None:
        gaps = [
            c['robustness_gap']['auroc']
            for c in p['cells']
            if _attack_family(c['attack']) == family and c['severity_level'] == severity
        ]
        if not gaps:
            return None
        return sum(gaps) / len(gaps)

    rows: list[tuple[str, list[float | None]]] = []
    # multimodal: average across seeds
    mm = [p for k, p in perturbed.items() if k.startswith('stage1-seed')]
    if mm:
        per_family: list[float | None] = []
        for fam in ('text', 'image-pixel', 'image-photometric', 'image-geometric'):
            vals = [v for v in (_avg_gap_family(p, fam, 'high') for p in mm) if v is not None]
            per_family.append(sum(vals) / len(vals) if vals else None)
        rows.append((f"multimodal ({len(mm)}-seed mean)", per_family))

    for key, p in perturbed.items():
        if key.startswith('stage1-seed'):
            continue
        per_family = [_avg_gap_family(p, fam, 'high')
                      for fam in ('text', 'image-pixel', 'image-photometric', 'image-geometric')]
        rows.append((key, per_family))

    for model_name, per_family in rows:
        cells = [f"{v:+.4f}" if v is not None else 'n/a' for v in per_family]
        lines.append(f"| {model_name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append('')

    # --- White-box per-modality fragility at ε=4/255 PGD ---
    lines.append('## Per-modality fragility — white-box PGD ε=4/255')
    lines.append('')
    lines.append('| Model | Clean AUROC | PGD AUROC | ΔAUROC | ASR |')
    lines.append('|---|---:|---:|---:|---:|')

    def _whitebox_row(label: str, payloads: list[dict[str, Any]]) -> str | None:
        if not payloads:
            return None
        cleans = [p['clean']['at_best_threshold']['auroc'] for p in payloads]
        atts: list[float] = []
        gaps: list[float] = []
        asrs: list[float] = []
        for p in payloads:
            for c in p['cells']:
                if c['attack'] == 'pgd' and c['epsilon_numerator'] == 4:
                    atts.append(c['attacked_at_best_threshold']['auroc'])
                    gaps.append(c['robustness_gap']['auroc'])
                    asrs.append(c['attack_success_rate'])
        if not atts:
            return None
        return (
            f"| {label} | {_fmt(*_mean_std(cleans))} | {_fmt(*_mean_std(atts))} "
            f"| {_fmt(*_mean_std(gaps))} | {_fmt(*_mean_std(asrs))} |"
        )

    mm_wb = [p for k, p in whitebox.items() if k.startswith('stage1-seed')]
    row = _whitebox_row(f"multimodal ({len(mm_wb)}-seed mean)", mm_wb)
    if row:
        lines.append(row)
    for key, p in whitebox.items():
        if key.startswith('stage1-seed'):
            continue
        row = _whitebox_row(key, [p])
        if row:
            lines.append(row)
    lines.append('')

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _make_figures(perturbed: dict[str, dict[str, Any]], whitebox: dict[str, dict[str, Any]], split: str = 'dev') -> list[Path]:
    import numpy as np
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- Heatmap of ΔAUROC per (attack × severity) for each multimodal seed ----
    for key, p in perturbed.items():
        if not key.startswith('stage1-seed'):
            continue
        attacks_order = TEXT_ATTACKS + IMAGE_ATTACKS
        mat = np.full((len(attacks_order), len(SEVERITIES)), np.nan)
        for c in p['cells']:
            if c['attack'] not in attacks_order:
                continue
            r = attacks_order.index(c['attack'])
            col = SEVERITIES.index(c['severity_level'])
            mat[r, col] = c['robustness_gap']['auroc']
        fig, ax = plt.subplots(figsize=(4.2, 7.0))
        # Symmetric colour scale on the *positive-gap* side dominated by text attacks.
        vmax = float(np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels(SEVERITIES)
        ax.set_yticks(range(len(attacks_order)))
        ax.set_yticklabels(attacks_order)
        ax.set_title(f"ΔAUROC (clean − attacked) — {key}")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:+.2f}", ha='center', va='center',
                            color='white' if abs(v) > vmax * 0.5 else 'black', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='ΔAUROC')
        fig.tight_layout()
        path = FIGS / f"heatmap_perturbed_{key}{_split_suffix(split)}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # ---- White-box curve: AUROC vs ε for FGSM and PGD ----
    if whitebox:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        mm = [p for k, p in whitebox.items() if k.startswith('stage1-seed')]
        if mm:
            for attack in ('fgsm', 'pgd'):
                eps_to_aurocs: dict[int, list[float]] = defaultdict(list)
                for p in mm:
                    for c in p['cells']:
                        if c['attack'] == attack:
                            eps_to_aurocs[c['epsilon_numerator']].append(c['attacked_at_best_threshold']['auroc'])
                xs = sorted(eps_to_aurocs)
                means = [statistics.mean(eps_to_aurocs[x]) for x in xs]
                stds = [statistics.pstdev(eps_to_aurocs[x]) if len(eps_to_aurocs[x]) > 1 else 0.0 for x in xs]
                ax.errorbar(xs, means, yerr=stds, marker='o', capsize=4,
                            label=f"multimodal {attack.upper()}")
            clean_aurocs = [p['clean']['at_best_threshold']['auroc'] for p in mm]
            ax.axhline(statistics.mean(clean_aurocs), linestyle='--', alpha=0.5,
                       label=f"multimodal clean ({statistics.mean(clean_aurocs):.3f})")
        for k, p in whitebox.items():
            if k.startswith('stage1-seed'):
                continue
            for attack in ('fgsm', 'pgd'):
                xs = sorted({c['epsilon_numerator'] for c in p['cells'] if c['attack'] == attack})
                ys = []
                for x in xs:
                    cell = next(c for c in p['cells']
                                if c['attack'] == attack and c['epsilon_numerator'] == x)
                    ys.append(cell['attacked_at_best_threshold']['auroc'])
                ax.plot(xs, ys, marker='s', linestyle=':', label=f"image-only {attack.upper()}")
            ax.axhline(p['clean']['at_best_threshold']['auroc'], linestyle=':', alpha=0.5,
                       label=f"image-only clean ({p['clean']['at_best_threshold']['auroc']:.3f})")
        ax.set_xlabel("ε (×255)")
        ax.set_ylabel("AUROC")
        ax.set_title("White-box robustness curves (dev, n=500)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        fig.tight_layout()
        path = FIGS / f'curve_whitebox{_split_suffix(split)}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # ---- Severity curves per family (multimodal mean) ----
    mm = [p for k, p in perturbed.items() if k.startswith('stage1-seed')]
    if mm:
        families = [
            ('text-character', ['leetspeak', 'char_deletion', 'char_swap', 'keyboard_typo']),
            ('text-format',    ['spacing', 'punctuation', 'case_noise', 'censoring']),
            ('image-pixel',    ['gaussian_noise', 'blur', 'compression']),
            ('image-photometric', ['brightness_up', 'brightness_down', 'contrast_up', 'contrast_down']),
            ('image-geometric',   ['translation', 'crop', 'occlusion']),
        ]
        fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 3.2), sharey=True)
        for ax, (fam, names) in zip(axes, families):
            for name in names:
                pts: list[tuple[int, float]] = []
                for s_i, sev in enumerate(SEVERITIES):
                    aurocs = [
                        c['attacked_at_best_threshold']['auroc']
                        for p in mm for c in p['cells']
                        if c['attack'] == name and c['severity_level'] == sev
                    ]
                    if aurocs:
                        pts.append((s_i, statistics.mean(aurocs)))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker='o', label=name)
            cm = statistics.mean(p['clean']['at_best_threshold']['auroc'] for p in mm)
            ax.axhline(cm, linestyle='--', alpha=0.4, color='black')
            ax.set_xticks(range(len(SEVERITIES)))
            ax.set_xticklabels(SEVERITIES, rotation=20)
            ax.set_title(fam)
            ax.set_ylim(0.5, 0.85)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, loc='lower left')
        axes[0].set_ylabel("AUROC (multimodal mean)")
        fig.tight_layout()
        path = FIGS / f'severity_curves{_split_suffix(split)}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    return paths


# ----------------------------------------------------------------------------
# Phase 5: robust-vs-clean comparison
# ----------------------------------------------------------------------------

VARIANT_ORDER = ('clean', 'augonly', 'kl', 'kldrop', 'kldrop-p015', 'kldrop-p050', 'kllowmed')


def _group_by_variant(payloads: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition by training recipe: clean / augonly / kl / kldrop / kllowmed."""
    groups: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANT_ORDER}
    for k, p in payloads.items():
        is_robust, variant = _is_robust_key(k)
        if is_robust:
            groups[variant].append(p)
        elif k.startswith('stage1-seed'):
            groups['clean'].append(p)
    return groups


def _fully_robust_count(p: dict[str, Any], whitebox_p: dict[str, Any] | None = None) -> int:
    """Count dev examples that survive every natural cell AND PGD ε=4 (if given).

    *Mixes* naturalistic and white-box attacks when ``whitebox_p`` is provided.
    For naturalistic-only survival, pass ``whitebox_p=None`` (see also
    ``_natural_fully_robust_count``).
    """
    # Examples are recorded per-cell; first build clean threshold + label/clean-prob.
    best_t = p['clean']['best_threshold']
    if not p['cells']:
        return 0
    table: dict[str, dict[str, Any]] = {}
    for ex in p['cells'][0]['examples']:
        rid = str(ex['id']).zfill(5)
        table[rid] = {
            'label': int(ex['label']),
            'clean_prob': float(ex['clean_prob']),
            'flipped': False,
        }
    for cell in p['cells']:
        for ex in cell['examples']:
            rid = str(ex['id']).zfill(5)
            entry = table.get(rid)
            if entry is None:
                continue
            ap = float(ex['attacked_prob'])
            pred = 1 if ap >= best_t else 0
            if pred != entry['label']:
                entry['flipped'] = True
    if whitebox_p is not None:
        for cell in whitebox_p['cells']:
            if cell['attack'] != 'pgd' or cell['epsilon_numerator'] != 4:
                continue
            for ex in cell['examples']:
                rid = str(ex['id']).zfill(5)
                entry = table.get(rid)
                if entry is None:
                    continue
                ap = float(ex['attacked_prob'])
                pred = 1 if ap >= best_t else 0
                if pred != entry['label']:
                    entry['flipped'] = True
    survivors = 0
    for rid, entry in table.items():
        cp_pred = 1 if entry['clean_prob'] >= best_t else 0
        if cp_pred == entry['label'] and not entry['flipped']:
            survivors += 1
    return survivors


def _natural_fully_robust_count(p: dict[str, Any]) -> int:
    """Count dev examples that are clean-correct AND survive every natural cell.

    Aligned to the project's primary threat model: a typical adversarial user
    editing the image/caption with an off-the-shelf tool, not a gradient-aware
    attacker. White-box PGD is reported separately in the whitebox table; it
    is a worst-case oracle attack and should not contaminate the headline
    naturalistic robustness number.
    """
    return _fully_robust_count(p, whitebox_p=None)


def _write_robust_vs_clean(
    perturbed: dict[str, dict[str, Any]],
    whitebox: dict[str, dict[str, Any]],
    modality_ablation: dict[str, dict[str, Any]],
    split: str = 'dev',
) -> Path:
    out = OUT / f'robust_vs_clean{_split_suffix(split)}.md'
    lines: list[str] = []
    lines.append('# Phase 5 — Robust vs clean training comparison')
    lines.append('')
    lines.append('All multimodal numbers aggregate the 3 seeds (mean ± σ). Each row is one training recipe.')
    lines.append('')

    pert_groups = _group_by_variant(perturbed)
    wb_groups = _group_by_variant(whitebox)
    ma_groups = _group_by_variant(modality_ablation)

    # --- Clean accuracy + naturalistic worst-cell text ΔAUROC per recipe ---
    lines.append('## Clean accuracy & naturalistic worst-case')
    lines.append('')
    lines.append('| Recipe | n seeds | Clean AUROC | Clean F1 | Worst-cell text ΔAUROC | Worst-cell image ΔAUROC |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for variant in VARIANT_ORDER:
        seeds = pert_groups.get(variant, [])
        if not seeds:
            continue
        cleans = [p['clean']['at_best_threshold']['auroc'] for p in seeds]
        f1s = [p['clean']['at_best_threshold']['macro_f1'] for p in seeds]
        worst_text: list[float] = []
        worst_img: list[float] = []
        for p in seeds:
            text_gaps = [c['robustness_gap']['auroc'] for c in p['cells'] if _attack_family(c['attack']) == 'text']
            img_gaps = [c['robustness_gap']['auroc'] for c in p['cells'] if _attack_family(c['attack']).startswith('image-') or _attack_family(c['attack']) == 'typographic']
            if text_gaps:
                worst_text.append(max(text_gaps))
            if img_gaps:
                worst_img.append(max(img_gaps))
        lines.append(
            f"| {variant} | {len(seeds)} | {_fmt(*_mean_std(cleans))} | {_fmt(*_mean_std(f1s))} "
            f"| {_fmt(*_mean_std(worst_text))} | {_fmt(*_mean_std(worst_img))} |"
        )
    lines.append('')

    # --- Per-family mean ΔAUROC at high severity ---
    families = ('text', 'image-pixel', 'image-photometric', 'image-geometric', 'typographic')
    lines.append('## Mean ΔAUROC at high severity, per family')
    lines.append('')
    lines.append('| Recipe | ' + ' | '.join(families) + ' |')
    lines.append('|---|' + '|'.join(['---:'] * len(families)) + '|')
    for variant in VARIANT_ORDER:
        seeds = pert_groups.get(variant, [])
        if not seeds:
            continue
        cells: list[str] = [variant]
        for fam in families:
            vals = []
            for p in seeds:
                gaps = [c['robustness_gap']['auroc'] for c in p['cells']
                        if _attack_family(c['attack']) == fam and c['severity_level'] == 'high']
                if gaps:
                    vals.append(sum(gaps) / len(gaps))
            cells.append(_fmt(*_mean_std(vals)) if vals else 'n/a')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # --- White-box ε-grid (PGD) ---
    lines.append('## White-box PGD AUROC vs ε')
    lines.append('')
    epsilons = sorted({c['epsilon_numerator'] for p in whitebox.values() for c in p['cells'] if c['attack'] == 'pgd'})
    if epsilons:
        lines.append('| Recipe | clean AUROC | ' + ' | '.join(f"PGD ε={e}/255" for e in epsilons) + ' |')
        lines.append('|---|---:|' + '|'.join(['---:'] * len(epsilons)) + '|')
        for variant in VARIANT_ORDER:
            seeds = wb_groups.get(variant, [])
            if not seeds:
                continue
            cleans = [p['clean']['at_best_threshold']['auroc'] for p in seeds]
            cells = [variant, _fmt(*_mean_std(cleans))]
            for eps in epsilons:
                vals = []
                for p in seeds:
                    for c in p['cells']:
                        if c['attack'] == 'pgd' and c['epsilon_numerator'] == eps:
                            vals.append(c['attacked_at_best_threshold']['auroc'])
                cells.append(_fmt(*_mean_std(vals)) if vals else 'n/a')
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # --- Modality-ablation triplet ---
    lines.append('## Modality ablation (clean dev forwards)')
    lines.append('')
    lines.append('Direct test of the "dead image branch" failure mode: did `KL_image_branch` lift `image_only` AUROC?')
    lines.append('')
    lines.append('| Recipe | multimodal AUROC | text_only AUROC | image_only AUROC |')
    lines.append('|---|---:|---:|---:|')
    for variant in VARIANT_ORDER:
        seeds = ma_groups.get(variant, [])
        if not seeds:
            continue
        def _grab(p, key):
            return p[key]['at_best_threshold']['auroc']
        mm = [_grab(p, 'multimodal') for p in seeds]
        to = [_grab(p, 'text_only') for p in seeds]
        io = [_grab(p, 'image_only') for p in seeds]
        lines.append(
            f"| {variant} | {_fmt(*_mean_std(mm))} | {_fmt(*_mean_std(to))} | {_fmt(*_mean_std(io))} |"
        )
    lines.append('')

    # --- Per-example fully-robust counts ---
    lines.append('## Per-example fully-robust counts (out of 500 dev)')
    lines.append('')
    lines.append('Two threat models, reported separately:')
    lines.append('')
    lines.append('* **Naturalistic-only**: example is clean-correct AND survives every '
                 'naturalistic perturbation cell (8 text × 3 + 11 image × 3 = 57 cells).')
    lines.append('* **Natural + white-box PGD**: same, plus survives PGD ε=4/255. '
                 'PGD has near-100 % ASR by design (worst-case oracle) so this number is dominated by it; '
                 'reported only for completeness.')
    lines.append('')
    lines.append('| Recipe | Naturalistic seed0 | seed1 | seed2 | mean | Nat + PGD seed0 | seed1 | seed2 | mean |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')

    def _seed_lookup(payloads: dict[str, dict[str, Any]], variant: str, seed: int) -> dict[str, Any] | None:
        if variant == 'clean':
            return payloads.get(f"stage1-seed{seed}")
        # try both naming conventions
        for prefix in (f"robust-{variant}-seed", f"train-robust-{variant}-seed"):
            k = f"{prefix}{seed}"
            if k in payloads:
                return payloads[k]
        return None

    for variant in VARIANT_ORDER:
        nat_counts: list[int] = []
        full_counts: list[int] = []
        for seed in (0, 1, 2):
            p = _seed_lookup(perturbed, variant, seed)
            wb = _seed_lookup(whitebox, variant, seed)
            if p is None:
                continue
            nat_counts.append(_natural_fully_robust_count(p))
            full_counts.append(_fully_robust_count(p, wb))
        if not nat_counts:
            continue
        nat_cells = [str(c) for c in nat_counts] + ['—'] * (3 - len(nat_counts))
        full_cells = [str(c) for c in full_counts] + ['—'] * (3 - len(full_counts))
        nat_mean = sum(nat_counts) / len(nat_counts) if nat_counts else 0.0
        full_mean = sum(full_counts) / len(full_counts) if full_counts else 0.0
        lines.append(
            f"| {variant} | {nat_cells[0]} | {nat_cells[1]} | {nat_cells[2]} | {nat_mean:.2f} "
            f"| {full_cells[0]} | {full_cells[1]} | {full_cells[2]} | {full_mean:.2f} |"
        )
    lines.append('')

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def _make_robust_figures(
    perturbed: dict[str, dict[str, Any]],
    whitebox: dict[str, dict[str, Any]],
    split: str = 'dev',
) -> list[Path]:
    """Two extra plots: a robust-kl seed-0 ΔAUROC heatmap + a clean/augonly/kl
    white-box AUROC-vs-ε curve overlay."""
    import numpy as np
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- robust-kl seed-0 heatmap (when available) ---
    kl_key = next((k for k in perturbed if _is_robust_key(k)[1] == 'kl' and k.endswith('seed0')), None)
    if kl_key is not None:
        p = perturbed[kl_key]
        attacks_order = TEXT_ATTACKS + IMAGE_ATTACKS + ['typographic']
        mat = np.full((len(attacks_order), len(SEVERITIES)), np.nan)
        for c in p['cells']:
            if c['attack'] not in attacks_order:
                continue
            r = attacks_order.index(c['attack'])
            col = SEVERITIES.index(c['severity_level'])
            mat[r, col] = c['robustness_gap']['auroc']
        fig, ax = plt.subplots(figsize=(4.4, 7.4))
        vmax = float(np.nanmax(np.abs(mat)))
        if not (vmax > 0):
            vmax = 0.1
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels(SEVERITIES)
        ax.set_yticks(range(len(attacks_order)))
        ax.set_yticklabels(attacks_order)
        ax.set_title(f"ΔAUROC — {kl_key}")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:+.2f}", ha='center', va='center',
                            color='white' if abs(v) > vmax * 0.5 else 'black', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='ΔAUROC')
        fig.tight_layout()
        path = FIGS / f'heatmap_robust_kl_seed0{_split_suffix(split)}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # --- White-box AUROC vs ε overlay: clean / augonly / kl ---
    wb_groups = _group_by_variant(whitebox)
    if any(wb_groups.get(v) for v in ('clean', 'augonly', 'kl')):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        styles = {
            'clean': '-', 'augonly': '--', 'kl': '-.', 'kldrop': ':',
            'kldrop-p015': (0, (1, 1)), 'kldrop-p050': (0, (5, 2)),
            'kllowmed': (0, (3, 1, 1, 1)),
        }
        for variant in VARIANT_ORDER:
            seeds = wb_groups.get(variant, [])
            if not seeds:
                continue
            for attack in ('fgsm', 'pgd'):
                eps_to_aurocs: dict[int, list[float]] = defaultdict(list)
                for p in seeds:
                    for c in p['cells']:
                        if c['attack'] == attack:
                            eps_to_aurocs[c['epsilon_numerator']].append(c['attacked_at_best_threshold']['auroc'])
                xs = sorted(eps_to_aurocs)
                if not xs:
                    continue
                means = [statistics.mean(eps_to_aurocs[x]) for x in xs]
                stds = [statistics.pstdev(eps_to_aurocs[x]) if len(eps_to_aurocs[x]) > 1 else 0.0 for x in xs]
                ax.errorbar(xs, means, yerr=stds, marker='o', capsize=3,
                            linestyle=styles[variant],
                            label=f"{variant} {attack.upper()}")
        ax.set_xlabel("ε (×255)")
        ax.set_ylabel("AUROC")
        ax.set_title("White-box robustness — clean vs augonly vs kl (dev)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='lower left')
        fig.tight_layout()
        path = FIGS / f'curve_whitebox_robust{_split_suffix(split)}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    return paths


# ----------------------------------------------------------------------------
# Phase 5c: in-pool vs OOD, per-severity, composite-attack tables.
# ----------------------------------------------------------------------------

def _in_pool_attacks() -> dict[str, set[str]]:
    """Canonical training augmentation pool, read from `stage1_robust_kl.yaml`.

    Every Phase-5/5b/5c robust config uses the same `robust.{text,image}_attacks`
    block; reading one is enough. Falls back to literals if the file is missing
    (e.g. running the aggregator outside the repo). Held-out attacks are the
    full eval pool minus this set, per modality.
    """
    cfg_path = REPO / 'configs' / 'stage1_robust_kl.yaml'
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        robust = cfg.get('robust', {})
        text_pool = set(robust.get('text_attacks', []))
        image_pool = set(robust.get('image_attacks', []))
        if text_pool and image_pool:
            return {'text': text_pool, 'image': image_pool}
    except Exception:
        pass
    return {
        'text': {'leetspeak', 'char_deletion', 'char_swap', 'censoring', 'keyboard_typo'},
        'image': {'gaussian_noise', 'blur', 'compression',
                  'brightness_down', 'occlusion', 'typographic'},
    }


def _is_composite_cell(cell: dict[str, Any]) -> bool:
    return str(cell.get('attack', '')).startswith('composite_')


def _write_pool_vs_ood_table(perturbed: dict[str, dict[str, Any]], split: str = 'dev') -> Path:
    """Per-recipe mean ΔAUROC split by training-pool membership.

    Single-perturbation cells only (composite cells are reported separately).
    Caveat noted in the table: held-out text attacks are also the lowest-impact
    ones on the clean ckpt, so the text-side OOD column has limited signal.
    """
    out = OUT / f'robust_vs_clean{_split_suffix(split)}.md'
    pools = _in_pool_attacks()
    pert_groups = _group_by_variant(perturbed)

    def _mean_gap(seeds: list[dict[str, Any]], names: set[str]) -> tuple[float, float]:
        per_seed: list[float] = []
        for p in seeds:
            gaps = [
                c['robustness_gap']['auroc']
                for c in p['cells']
                if not _is_composite_cell(c) and c['attack'] in names
            ]
            if gaps:
                per_seed.append(sum(gaps) / len(gaps))
        return _mean_std(per_seed)

    text_all = set(TEXT_ATTACKS)
    image_all = set(IMAGE_ATTACKS) | {'typographic'}
    text_in, text_ood = pools['text'] & text_all, text_all - pools['text']
    image_in, image_ood = pools['image'] & image_all, image_all - pools['image']

    lines: list[str] = []
    lines.append('')
    lines.append('## Attack-type generalisation: in-pool vs held-out (OOD)')
    lines.append('')
    lines.append('Training pool (text, n=' + str(len(text_in)) + '): `'
                 + ', '.join(sorted(text_in)) + '`  ')
    lines.append('Held-out text (n=' + str(len(text_ood)) + '): `'
                 + ', '.join(sorted(text_ood)) + '`  ')
    lines.append('Training pool (image, n=' + str(len(image_in)) + '): `'
                 + ', '.join(sorted(image_in)) + '`  ')
    lines.append('Held-out image (n=' + str(len(image_ood)) + '): `'
                 + ', '.join(sorted(image_ood)) + '`')
    lines.append('')
    lines.append('Mean ΔAUROC (clean − attacked) across all severities of the cells in each '
                 'partition. Lower is better. *Caveat*: held-out text attacks '
                 '(`spacing/punctuation/case_noise`) are the weakest text attacks on the '
                 'clean ckpt (ΔAUROC ≤ 0.02), so the text-side OOD column has limited signal. '
                 'The image-side OOD column is the meaningful generalisation test.')
    lines.append('')
    lines.append('| Recipe | text in-pool | text OOD | Δ(OOD-in) text | image in-pool | image OOD | Δ(OOD-in) image |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for variant in VARIANT_ORDER:
        seeds = pert_groups.get(variant, [])
        if not seeds:
            continue
        t_in_m, t_in_s = _mean_gap(seeds, text_in)
        t_oo_m, t_oo_s = _mean_gap(seeds, text_ood)
        i_in_m, i_in_s = _mean_gap(seeds, image_in)
        i_oo_m, i_oo_s = _mean_gap(seeds, image_ood)
        dt = t_oo_m - t_in_m
        di = i_oo_m - i_in_m
        lines.append(
            f"| {variant} | {_fmt(t_in_m, t_in_s)} | {_fmt(t_oo_m, t_oo_s)} | {dt:+.4f} "
            f"| {_fmt(i_in_m, i_in_s)} | {_fmt(i_oo_m, i_oo_s)} | {di:+.4f} |"
        )
    lines.append('')
    lines.append('Reading: Δ(OOD−in) > 0 means the recipe defends in-pool attacks more than '
                 'held-out ones — i.e. augmentation has *memorised* its training pool rather '
                 'than learning a transferable defence. Δ near 0 (or negative) is the '
                 'generalisation signal we want.')
    lines.append('')

    with out.open('a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out


def _write_per_severity_table(perturbed: dict[str, dict[str, Any]], split: str = 'dev') -> Path:
    """Per-recipe mean ΔAUROC split by severity (low / medium / high).

    Single-perturbation cells only. The medium column is the headline for the
    "internet-realistic threat" framing (high is at the label-preserving limit).
    """
    out = OUT / f'robust_vs_clean{_split_suffix(split)}.md'
    pert_groups = _group_by_variant(perturbed)
    families = ('text', 'image-pixel', 'image-photometric', 'image-geometric', 'typographic')

    def _mean_per_sev(seeds: list[dict[str, Any]], family: str, sev: str) -> tuple[float, float]:
        per_seed: list[float] = []
        for p in seeds:
            gaps = [
                c['robustness_gap']['auroc']
                for c in p['cells']
                if not _is_composite_cell(c)
                and _attack_family(c['attack']) == family
                and c['severity_level'] == sev
            ]
            if gaps:
                per_seed.append(sum(gaps) / len(gaps))
        return _mean_std(per_seed)

    lines: list[str] = []
    lines.append('')
    lines.append('## Per-severity sensitivity (single-perturbation cells)')
    lines.append('')
    lines.append('Mean ΔAUROC at each severity, per family. **Medium severity is the '
                 'internet-realistic threat number** (the level a typical adversarial user '
                 'would reach with an off-the-shelf editor); high severity is a stress-test '
                 'ceiling that often borders on label-preserving limits; low is included for '
                 'monotonicity sanity-checking.')
    lines.append('')
    for family in families:
        lines.append(f"### {family}")
        lines.append('')
        lines.append('| Recipe | low | **medium** (realistic) | high (stress) |')
        lines.append('|---|---:|---:|---:|')
        for variant in VARIANT_ORDER:
            seeds = pert_groups.get(variant, [])
            if not seeds:
                continue
            lo_m, lo_s = _mean_per_sev(seeds, family, 'low')
            md_m, md_s = _mean_per_sev(seeds, family, 'medium')
            hi_m, hi_s = _mean_per_sev(seeds, family, 'high')
            if all(math.isnan(x) for x in (lo_m, md_m, hi_m)):
                continue
            lines.append(
                f"| {variant} | {_fmt(lo_m, lo_s)} | **{_fmt(md_m, md_s)}** | {_fmt(hi_m, hi_s)} |"
            )
        lines.append('')

    with out.open('a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out


def _write_composite_table(perturbed: dict[str, dict[str, Any]], split: str = 'dev') -> Path | None:
    """Per-recipe per-severity stats for composite-attack cells (Phase 5c-2).

    Returns None when no composite cells are present (e.g. before the 5c-2
    cluster runs land); existing single-perturbation tables are unaffected.
    """
    has_any = any(
        _is_composite_cell(c)
        for p in perturbed.values()
        for c in p.get('cells', [])
    )
    if not has_any:
        return None

    out = OUT / f'robust_vs_clean{_split_suffix(split)}.md'
    pert_groups = _group_by_variant(perturbed)
    composite_types = (
        'composite_2text', 'composite_2image',
        'composite_text_image', 'composite_2text_2image',
    )

    def _cell_stat(seeds, ctype: str, sev: str, field: str) -> tuple[float, float]:
        per_seed = []
        for p in seeds:
            vals: list[float] = []
            for c in p['cells']:
                if c.get('attack') == ctype and c.get('severity_level') == sev:
                    if field == 'gap':
                        vals.append(c['robustness_gap']['auroc'])
                    elif field == 'asr':
                        vals.append(c['attack_success_rate'])
                    elif field == 'auroc':
                        vals.append(c['attacked_at_best_threshold']['auroc'])
            if vals:
                per_seed.append(sum(vals) / len(vals))
        return _mean_std(per_seed)

    lines: list[str] = []
    lines.append('')
    lines.append('## Composite attacks (multiple perturbations per sample)')
    lines.append('')
    lines.append('Each composite cell applies *K* random perturbations to every sample, '
                 'drawn deterministically per `(cell_seed, sample_id)` from the full eval pool '
                 '(not the training pool — composite is itself an OOD test). '
                 'All component perturbations use the cell-level severity.')
    lines.append('')
    lines.append('| Composite | K_text | K_image |')
    lines.append('|---|---:|---:|')
    lines.append('| `composite_2text` | 2 | 0 |')
    lines.append('| `composite_2image` | 0 | 2 |')
    lines.append('| `composite_text_image` | 1 | 1 |')
    lines.append('| `composite_2text_2image` | 2 | 2 |')
    lines.append('')

    # Detect whether any composite cells use severity='mixed' (Phase 9a);
    # if so, surface it as an extra column so the realistic-user threat
    # appears alongside the fixed-severity columns.
    has_mixed = any(
        c.get('attack', '').startswith('composite_') and c.get('severity_level') == 'mixed'
        for p in perturbed.values()
        for c in p.get('cells', [])
    )

    for ctype in composite_types:
        lines.append(f"### {ctype}")
        lines.append('')
        if has_mixed:
            lines.append('| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|')
        else:
            lines.append('| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | high ASR |')
            lines.append('|---|---:|---:|---:|---:|')
        for variant in VARIANT_ORDER:
            seeds = pert_groups.get(variant, [])
            if not seeds:
                continue
            lo = _cell_stat(seeds, ctype, 'low', 'gap')
            md = _cell_stat(seeds, ctype, 'medium', 'gap')
            hi = _cell_stat(seeds, ctype, 'high', 'gap')
            asr = _cell_stat(seeds, ctype, 'high', 'asr')
            if has_mixed:
                mx = _cell_stat(seeds, ctype, 'mixed', 'gap')
                mxasr = _cell_stat(seeds, ctype, 'mixed', 'asr')
                if all(math.isnan(x[0]) for x in (lo, md, hi, mx)):
                    continue
                lines.append(
                    f"| {variant} | {_fmt(*lo)} | {_fmt(*md)} | {_fmt(*hi)} "
                    f"| **{_fmt(*mx)}** | {_fmt(*asr)} | {_fmt(*mxasr)} |"
                )
            else:
                if all(math.isnan(x[0]) for x in (lo, md, hi)):
                    continue
                lines.append(
                    f"| {variant} | {_fmt(*lo)} | **{_fmt(*md)}** | {_fmt(*hi)} | {_fmt(*asr)} |"
                )
        lines.append('')

    with out.open('a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out


# ----------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    perturbed_all = _load_perturbed()
    whitebox_all = _load_whitebox()
    modality_ablation_all = _load_modality_ablation()
    splits = _available_splits(perturbed_all, whitebox_all, modality_ablation_all)
    print(f"splits found: {splits}")

    paths: list[Path] = []
    for split in splits:
        perturbed = _filter_by_split(perturbed_all, split)
        whitebox = _filter_by_split(whitebox_all, split)
        modality_ablation = _filter_by_split(modality_ablation_all, split)
        print(f"\n=== split={split} | perturbed={len(perturbed)} whitebox={len(whitebox)} ma={len(modality_ablation)} ===")
        print(f"  perturbed: {sorted(perturbed)}")
        print(f"  whitebox:  {sorted(whitebox)}")
        print(f"  modality:  {sorted(modality_ablation)}")
        if perturbed:
            paths.append(_write_perturbed_table(perturbed, split=split))
        if whitebox:
            paths.append(_write_whitebox_table(whitebox, split=split))
        if perturbed or whitebox:
            paths.append(_write_worst_case(perturbed, whitebox, split=split))
        if perturbed or whitebox or modality_ablation:
            paths.append(_write_robust_vs_clean(perturbed, whitebox, modality_ablation, split=split))
        if perturbed:
            paths.append(_write_pool_vs_ood_table(perturbed, split=split))
            paths.append(_write_per_severity_table(perturbed, split=split))
            comp = _write_composite_table(perturbed, split=split)
            if comp is not None:
                paths.append(comp)
        paths.extend(_make_figures(perturbed, whitebox, split=split))
        paths.extend(_make_robust_figures(perturbed, whitebox, split=split))

    for p in paths:
        print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
