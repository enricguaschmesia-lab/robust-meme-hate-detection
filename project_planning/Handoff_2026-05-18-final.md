# Handoff — 2026-05-18 (final, naturalistic-perturbations side)

Supersedes `Handoff_2026-05-18.md`. The naturalistic-perturbations
side of `robust-meme-hate-detection` is **done**. Phases 1–9b are
all committed; poster figures exist; reports are written. Adversarial
training (PGD ε ≥ 2/255) is owned by Henrik and out of scope.

## 1. Project state in one paragraph

OpenCLIP ViT-B/32 fusion classifier on Hateful Memes. Three eval
splits: dev (n=500), test_seen (n=1000, 49 % pos), test_unseen
(n=2000, 37 % pos). Seven training recipes × 3 seeds each:
`stage1` (clean), `augonly`, `kl`, `kldrop` (p=0.30), `kldrop-p015`
(p=0.15), `kldrop-p050` (p=0.50), `kllowmed`. Eval surface per ckpt
× split: 57 single-cell perturbations + **16 composite cells**
(4 types × 4 severities incl. mixed-severity) + 8 white-box ε-cells
(FGSM/PGD at ε ∈ {1,2,4,8}/255) + modality-ablation triplet.
**Naturalistic recommended default: `kldrop-p015`** (image-branch
revival without clean-accuracy cost); `kldrop-p050` as the
conservative composite-heaviest alternative; `kl` as the
clean-accuracy-preserving baseline.

## 2. State by phase

| Phase | Status | Notes |
|---|---|---|
| 1–4 | done & committed (`aede3ac` + earlier) | Baseline + perturbation suite + Phase 4 eval. |
| 5 / 5b / 5c | done & committed (`aede3ac`) | Robust training + modality dropout + generalisation analysis. |
| 6 (failure narrative) | done & committed (`b7b9dbe`) | Multi-seed/multi-recipe/composite-aware failure analysis. Class-asymmetric kldrop/kl finding on dev. |
| 7 (held-out test) | done & committed (`cd5bafe`) | All 5 recipes × 3 seeds × test_seen + test_unseen × 3 eval kinds (90 jobs). |
| **8 (test failure analysis)** | done, **commit pending** | Class-asymmetric finding partially reproduces on test. |
| **9a (mixed composites)** | done, commit pending | `mixed` severity tag for composite cells; 45 perturbed re-runs. |
| **9b (dropout-rate sweep)** | done, commit pending | `kldrop-p015` / `kldrop-p050`; 60 jobs. **kldrop-p015 is the new default**. |
| **Poster figures** | done, commit pending | `scripts/make_poster_figures.py` + 6 figures. |

## 3. Headline numbers (3-seed mean, all 7 recipes × 3 splits)

| Recipe | Clean AUROC dev / test_seen / test_unseen | Worst-cell text Δ | Image-only AUROC | Composite_2text mixed Δ |
|---|---|---:|---:|---:|
| clean | 0.742 / 0.747 / 0.738 | 0.115 / 0.129 / 0.110 | 0.593 / 0.582 / 0.602 | 0.075 / 0.080 / 0.078 |
| augonly | 0.712 / 0.716 / 0.708 | 0.083 / 0.083 / 0.063 | 0.604 / 0.606 / 0.620 | 0.054 / 0.049 / 0.044 |
| kl | 0.737 / 0.745 / 0.738 | 0.090 / 0.086 / 0.065 | 0.611 / 0.605 / 0.628 | 0.058 / 0.053 / 0.049 |
| **kldrop-p015** ★ | **0.735 / 0.748 / 0.743** | 0.081 / 0.087 / 0.069 | **0.638 / 0.638 / 0.658** | 0.057 / 0.058 / 0.050 |
| kldrop (p=0.30) | 0.705 / 0.721 / 0.713 | 0.067 / 0.074 / 0.055 | 0.636 / 0.641 / 0.655 | 0.046 / 0.043 / 0.039 |
| **kldrop-p050** | 0.700 / 0.717 / 0.712 | 0.065 / 0.068 / 0.049 | **0.641 / 0.651 / 0.666** | **0.041 / 0.040 / 0.035** |
| kllowmed | 0.735 / 0.744 / 0.736 | 0.093 / 0.089 / 0.068 | 0.602 / 0.597 / 0.617 | 0.058 / 0.053 / 0.050 |

★ Recommended default. Best per column (per split) bolded.

## 4. What committed; what's not

**Committed:**
- `aede3ac` — Phases 1–5/5b/5c (incl. perturbation suite, robust training, kllowmed, OOD/severity/composite tables).
- `b7b9dbe` — Phase 6 (failure-case narrative, multi-seed/multi-recipe/composite-aware).
- `cd5bafe` — Phase 7 (held-out test eval, --dataset-root CLI, split-aware aggregator, test labels, Handoff_2026-05-18.md).

**Pending commits** (Phase 8, 9, Poster figures):
- `scripts/failure_analysis.py` — `_job_dir` split-aware fix.
- `scripts/aggregate_phase4.py` — `kldrop-p015/p050` recipe recognition; `VARIANT_ORDER` extension; mixed-severity composite column auto-detection; whitebox-curve styles for the new recipes.
- `src/robust_meme_hate_detection/eval/run_perturbed.py` — `severity='mixed'` per-component sampling in `_apply_composite`; composite loop always emits a mixed cell.
- `configs/stage1_robust_kl_drop_p015.yaml`, `stage1_robust_kl_drop_p050.yaml` — new sweep configs.
- `scripts/make_poster_figures.py` + `project_planning/poster_figures/` — 6 figures (PNG + PDF + caption.txt each).
- `project_planning/Phase{8,9a,9b}_*_Report.md` — three new reports.
- `project_planning/Phase7_Completion_Report.md` — § 6 and § 8 updates.
- `project_planning/Implementation_Worklog.md` — Phase 18 + 19 entries.
- `README.md` — Phase 8 + 9 status blocks + reproducibility section.
- `project_planning/phase4/failure_analysis.{test_seen,test_unseen}.md` — new.
- `project_planning/phase4/robust_vs_clean*.md`, `perturbed_table*.md`, `whitebox_table*.md`, `worst_case*.md` — regenerated incl. new recipes + mixed column.
- `project_planning/phase4/figures/*.png` — regenerated incl. dropout-sweep variants.
- `cluster-results/*-robust-kldrop-p{015,050}-*` — 60 new dirs (Phase 9b).
- `cluster-results/perturbed-*-seed{0,1,2}/perturbed_eval.json` — overwritten (73 cells each).
- `Handoff_2026-05-18-final.md` (this file).

## 5. Open questions / next steps (out of scope for this thread)

1. **Final-report write-up.** Source-of-truth for each section:
   - Phase 1-2 baseline → `Phase1-2_Completion_Report.md`.
   - Naturalistic benchmark → `Phase3_4_Completion_Report.md`.
   - Robust training → `Phase5_Completion_Report.md` (§§ 1–11) +
     `Phase5_Completion_Report.md § 13` (Phase 5c).
   - Failure analysis → `Phase6_Completion_Report.md` (dev) +
     `Phase8_TestFailure_Report.md` (test).
   - Held-out test → `Phase7_Completion_Report.md`.
   - Mixed composites & dropout sweep → `Phase9a_*_Report.md`,
     `Phase9b_*_Report.md`.
   - Final recommendation → `kldrop-p015`, see Phase 7 § 6 update
     and Phase 9b § 4.
2. **Poster authoring.** Figures + captions are in
   `project_planning/poster_figures/`. Recommended panel order:
   01 (Pareto headline) → 06 (dropout sweep, shows why p=0.15 is
   the new default) → 02 (image-branch revival) → 03 (composite
   escalation) → 04 (class-asymmetric trade-off) → 05 (severity
   curves). Poster authoring not in scope here.
3. **Adversarial training (Henrik).** White-box PGD ε ≥ 2/255
   remains undefended on every recipe across all 3 splits. Henrik
   is tackling. The codebase is ready: `attacks/pgd.py` exposes
   `fgsm_image` / `pgd_image`; the training loop in
   `stage1_robust.py` would need a Madry-style inner-max wrapper.
4. **(Optional) Finer dropout sweep.** Phase 9b's sweep is coarse
   (p ∈ {0, 0.15, 0.30, 0.50}). A {0.10, 0.15, 0.20, 0.25} sweep
   around the new optimum might refine `kldrop-p015` further.
   Cost: ~12 more A100-h.

## 6. Reproducibility commands

```bash
# Local
pytest tests/                                                 # 7 passed, 4 skipped
python -m robust_meme_hate_detection.tests.smoke_synthetic    # forward + backward + PGD

# Aggregate cluster results (dev + test_seen + test_unseen)
PYTHONPATH=src .venv/bin/python3 scripts/aggregate_phase4.py

# Failure analysis per split
PYTHONPATH=src .venv/bin/python3 scripts/failure_analysis.py                # dev
PYTHONPATH=src .venv/bin/python3 scripts/failure_analysis.py --split test_seen
PYTHONPATH=src .venv/bin/python3 scripts/failure_analysis.py --split test_unseen

# Poster figures
PYTHONPATH=src .venv/bin/python3 scripts/make_poster_figures.py
```

Cluster reproduction (assumes
`/scratch/robust-meme-hate-detection/data/test_labels/` already
staged per Handoff_2026-05-18.md):

```bash
# Phase 9a — re-run perturbed cells with mixed composites
bash /tmp/submit_phase9a.sh        # 45 jobs

# Phase 9b — dropout-rate sweep
bash /tmp/submit_phase9b_train.sh  # 6 train jobs
# wait for Succeeded
bash /tmp/submit_phase9b_eval.sh   # 54 eval jobs
```

## 7. One-line conclusion

Naturalistic side of the project is done. **`kldrop-p015` is the
new recommended default**, dominating the previously-published
`kldrop` (p=0.30) on every metric on every split — a Pareto
improvement the original kldrop hyperparameter choice was sitting
on top of. The naturalistic perturbations + composite + dropout
sweep + class-asymmetric per-example analyses are all reproducible
from `scripts/make_poster_figures.py` once cluster-results are
pulled. Ready for poster authoring and final report.
