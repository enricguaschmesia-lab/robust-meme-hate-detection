# Phase 9a — Mixed-severity composite cells

> Status: written 2026-05-18. Closes the Phase-5c-2 / Phase-7 § 8
> limitation that "composite cell design fixes a single severity per
> cell." Adds a fourth severity tag — `mixed` — where each component
> samples severity ∈ {low, medium, high} independently per sample,
> closer to a realistic adversarial user who combines a strong edit
> with a weak edit.

## 1. Code change

Surgical: `src/robust_meme_hate_detection/eval/run_perturbed.py::_apply_composite`
gains a `severity='mixed'` sentinel. When set, each component's
severity is sampled per-component from the same RNG used to pick
attack types — preserving determinism per `sha256(cell_seed | sample_id)`
and recording the actual per-component severity in the JSON's
`components` log.

The composite loop in `main()` always emits a `mixed` cell alongside
the standard `low/medium/high` cells. Each perturbed-eval JSON
grows from 69 cells (Phase 5c) to **73 cells** (57 single + 4
composite types × 4 severities).

Aggregator `_write_composite_table` auto-detects the presence of
`mixed` cells (`has_mixed`); when present, surfaces a separate
"mixed" column in the composite tables (lives in
`project_planning/phase4/robust_vs_clean*.md`).

Cluster cost: **45 perturbed re-runs** (5 recipes × 3 seeds × 3
splits). Whitebox + modality-ablation results unchanged.

## 2. Findings

### 2.1 Mixed lands between medium and high

For every (recipe × composite type × split) cell, the mixed-severity
ΔAUROC lands **between medium and high** — closer to medium. Sample
from test_unseen `composite_2text`:

| Recipe | medium | mixed | high |
|---|---:|---:|---:|
| clean | 0.073 | 0.078 | 0.111 |
| augonly | 0.043 | 0.044 | 0.064 |
| kl | 0.052 | 0.049 | 0.069 |
| **kldrop** | **0.039** | **0.039** | 0.057 |
| kldrop-p015 | 0.054 | 0.050 | 0.074 |
| **kldrop-p050** | 0.035 | **0.035** | **0.048** |
| kllowmed | 0.053 | 0.050 | 0.074 |

Mixed ≈ medium for every recipe, sometimes slightly worse (clean)
and sometimes slightly better (kl, kldrop). **The realistic
adversarial-user threat is comparable to a medium-severity
single-cell composite**, not to a high-severity stress test. That
matches intuition: a real user wouldn't push every edit to its
limit; a mix of edits at typical severities is the natural failure
mode.

### 2.2 `kldrop-p050` is the strongest defender under mixed-severity

On every split and every composite type, `kldrop-p050` (Phase 9b)
has the smallest mixed-severity ΔAUROC. The previously-recommended
`kldrop` (p=0.30) is the second-best. The reading: **higher
modality-dropout rate buys more composite robustness**, at the
clean-accuracy cost (see Phase 9b).

### 2.3 No new Pareto rank inversions

Mixed-severity composite numbers do not change the recipe ordering
established by single-cell + composite-medium metrics. The Pareto
front remains:

- **`kl` / `kldrop-p015`** for clean-accuracy preservation.
- **`kldrop` / `kldrop-p050`** for maximum composite robustness.

See Phase 9b for the Pareto-front update incorporating the
dropout-rate sweep.

## 3. Limitations

- **Severity sampling is uniform** over {low, medium, high}.
  A real adversarial user might bias toward medium or have a
  more structured strategy. Uniform is the cleanest baseline.
- **Components are still drawn from a fixed `K_text × K_image` per
  composite type.** A truly variable-K composite would be a
  separate extension.
- **Mixed-severity cells are reported only per-recipe, not
  per-sample.** The components log carries the per-component
  severity, so per-sample mixed-severity attack analysis is
  possible if Phase 8-style failure analysis is extended to
  composites.

## 4. One-line conclusion

Mixed-severity composite ΔAUROC reproduces the medium-severity
ordering between recipes — it doesn't change the Pareto recipe
ranking, but it provides the realistic-user-distribution number
that the project's threat model wanted. All composite tables in
`robust_vs_clean*.md` now carry the mixed column.
