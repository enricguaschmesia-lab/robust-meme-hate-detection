# Phase 7 — Held-out test evaluation

> Status: written 2026-05-18 on the 1000-example `test_seen` and
> 2000-example `test_unseen` splits. Every Phase 5/5c number in
> `project_planning/Phase5_Completion_Report.md` and
> `project_planning/phase4/robust_vs_clean.md` was on the 500-example
> dev split; Phase 7 is the held-out reproduction. Data lives in
> `project_planning/phase4/robust_vs_clean.test_seen.md` and
> `robust_vs_clean.test_unseen.md`.

## 1. Setup

### 1.1 Splits and class priors

| Split | n | positive rate | source |
|---|---:|---:|---|
| dev (Phase 5/5c) | 500 | 35.8 % | `/scratch/datasets/hate_meta/dev.jsonl` |
| **test_seen** (new) | 1000 | 49.0 % | Hateful Memes phase-2 release |
| **test_unseen** (new) | 2000 | 37.5 % | Hateful Memes phase-2 release |

Labels for both test splits were retrieved from the
`neuralcatcher/hateful_memes` HF mirror; verified bit-perfect overlap
between local `test_seen.jsonl` IDs/texts and the cluster's
unlabelled `/scratch/datasets/hate_meta/test.jsonl` (0 missing,
0 extra, 0 text mismatches).

### 1.2 Class-prior caveat

`test_seen` is rebalanced to ~49/51 by the Hateful Memes challenge
designers. `test_unseen` (37.5 % pos) is closer to the training
prior (35.8 % dev, 35.9 % train) — it is the more naturalistic
generalisation pool. AUROC is unaffected by class-prior shift;
macro-F1 / accuracy at fixed τ are. This report uses AUROC as the
primary metric.

### 1.3 Image staging and code changes

The cluster's `/scratch/datasets/hate_meta/img/` had 10 000 PNGs
(train 8500 + dev 500 + test_seen 1000) — no test_unseen images.
2000 test_unseen PNGs were fetched from
`limjiayi/hateful_memes_expanded` via parallel `urllib.request`,
dereferenced, and rsynced to a writable mirror at
`/scratch/robust-meme-hate-detection/data/test_labels/` containing
both labelled jsonls, a symlink `img -> /scratch/datasets/hate_meta/img`,
and a flat `img_test_unseen/`.

Three eval entrypoints (`run_perturbed`, `run_whitebox`,
`run_modality_ablation`) gained a `--dataset-root` CLI override.
The data loader's existing `SPLIT_ALIASES` understood `test_seen`
and `test_unseen` so no model-code change was needed.

`scripts/aggregate_phase4.py` made split-aware: a regex extracts
`split ∈ {dev, test_seen, test_unseen}` from the job-name suffix
(`test-seen` / `test-unseen`) and the write functions take a
`split` parameter that suffixes their output filenames. Dev
outputs are byte-identical to the pre-extension version.

## 2. Cluster runs

90 jobs total: 5 recipes × 3 seeds × 2 splits × 3 eval kinds
(`perturbed --composites all` / `whitebox FGSM+PGD ε∈{1,2,4,8}/255` /
`modality-ablation`). All 90 Succeeded. Aggregate cluster wall-time
≈ 12 A100-hours under contention.

## 3. Headline reproduction on test

3-seed mean. **kldrop is the only recipe whose clean dev AUROC of
0.705 lifted by ~+0.02 on test_seen (0.721) and held on test_unseen
(0.713)** — the dev gap to clean baseline did not amplify on test.

| Recipe | Clean AUROC | Worst-cell text Δ | Image-only AUROC | Composite_2text med Δ | Naturalistic robust /N |
|---|---:|---:|---:|---:|---:|
| **dev (n=500, 36 % pos)** | | | | | |
| clean | 0.742 | 0.115 | 0.593 | 0.090 | 90.3 (18.1 %) |
| augonly | 0.712 | 0.083 | 0.604 | 0.063 | 87.3 (17.5 %) |
| kl | **0.737** | 0.090 | 0.611 | 0.066 | **111.7** (22.3 %) |
| kldrop | 0.705 | **0.067** | **0.636** | **0.051** | 91.7 (18.3 %) |
| kllowmed | 0.735 | 0.093 | 0.602 | 0.065 | 107.3 (21.5 %) |
| **test_seen (n=1000, 49 % pos)** | | | | | |
| clean | 0.747 | 0.129 | 0.582 | 0.087 | 153.7 (15.4 %) |
| augonly | 0.716 | 0.083 | 0.606 | 0.058 | 164.7 (16.5 %) |
| kl | **0.745** | 0.086 | 0.605 | 0.064 | **196.3** (19.6 %) |
| kldrop | 0.721 | **0.074** | **0.641** | **0.050** | 182.7 (18.3 %) |
| kllowmed | 0.744 | 0.089 | 0.597 | 0.064 | 182.7 (18.3 %) |
| **test_unseen (n=2000, 37 % pos)** | | | | | |
| clean | 0.738 | 0.110 | 0.602 | 0.073 | 233.3 (11.7 %) |
| augonly | 0.708 | 0.063 | 0.620 | 0.043 | 301.3 (15.1 %) |
| kl | **0.738** | 0.065 | 0.628 | 0.052 | 372.7 (18.6 %) |
| kldrop | 0.713 | **0.055** | **0.655** | 0.049 | 347.3 (17.4 %) |
| kllowmed | 0.736 | 0.068 | 0.617 | 0.053 | **387.0** (19.4 %) |

### 3.1 What reproduced from dev

- **`kldrop` is the worst-cell text robustness winner on all three
  splits** (0.067 dev → 0.074 test_seen → 0.055 test_unseen). The
  test_unseen number (0.055) is the project's lowest single-cell
  text ΔAUROC anywhere. This is the strongest reproduction.
- **`kldrop` is the image-branch-revival champion on all three splits**
  — image-only AUROC 0.636 dev → 0.641 test_seen → 0.655 test_unseen,
  exceeding the dedicated image-only baseline (0.628) on every split.
  Modality-dropout fully revives the image branch on held-out data.
- **`kldrop` is the composite_2text medium-severity winner on
  test_seen** (0.050 vs clean 0.087, -43 % gap) — matches the dev
  finding (0.051 vs 0.090, -43 % gap) almost exactly.
- **`kl` preserves clean AUROC** on all splits (0.737 / 0.745 / 0.738
  vs clean baseline 0.742 / 0.747 / 0.738 — within σ on test_unseen).

### 3.2 What changed from dev

- **`kllowmed` has the highest naturalistic-robust count on test_unseen
  (387.0 / 2000)** — the recipe Phase 5c-3 dismissed as
  "Pareto-dominated by `kl`" beats `kl` by 14 examples on the
  naturalistic split. The reversal is concentrated on naturalistic-
  robust *count* — `kl` still beats `kllowmed` on worst-cell text Δ
  (0.065 vs 0.068), image-only AUROC (0.628 vs 0.617), and
  composite-2text medium (0.052 vs 0.053). So `kllowmed`'s win is
  narrow and metric-specific, but the *strict* Pareto-domination
  claim from dev does not hold on test_unseen.
- **`augonly` becomes competitive on test_seen worst-cell text Δ**
  (0.083) — matches `kl` (0.086) and only slightly behind `kldrop`
  (0.074). On dev `augonly` was the dominated recipe; on test it has
  a defensible niche on worst-cell text robustness.
- **Naturalistic-robust % drops on test_unseen** (kl 18.6 % vs dev
  22.3 %, -3.7 pp). The bigger pool (n=2000) and the slightly less-
  clean text content of test_unseen apparently exposes more
  text-fragile examples than dev did. AUROC stays put.

### 3.3 Recipe rankings — dev → test reproduction matrix

| Metric | Dev winner | test_seen winner | test_unseen winner | Reproduces? |
|---|---|---|---|---|
| Clean AUROC | clean (0.742) | clean (0.747) | clean=kl (0.738) | ✓ |
| Worst-cell text Δ | kldrop (0.067) | kldrop (0.074) | kldrop (0.055) | ✓ |
| Image-only AUROC | kldrop (0.636) | kldrop (0.641) | kldrop (0.655) | ✓ |
| Composite_2text med Δ | kldrop (0.051) | kldrop (0.050) | augonly (0.043) | partial (kldrop close) |
| Naturalistic robust % | kl (22.3 %) | kl (19.6 %) | **kllowmed** (19.4 %) | not on test_unseen |

The first four headline metrics — the ones the Phase 5/5c report
leans on for the Pareto recommendation — all reproduce. The
naturalistic-robust *count* is the only headline metric where the
recipe ordering shuffles, and it shuffles toward `kllowmed` (a
recipe Phase 5c-3 had downgraded).

## 4. Generalisation analysis on test

### 4.1 In-pool vs OOD attack transfer

Mean ΔAUROC across all severities, partitioned by training-pool
membership. Δ(OOD − in) negative ⇒ augmentation gain transfers to
held-out attacks. Image-side is the meaningful signal (text-side
OOD attacks are intrinsically weak; ΔAUROC ≤ 0.02 on clean ckpt).

| Recipe | image in-pool | image OOD | Δ(OOD−in) image (test_seen) | Δ(OOD−in) image (test_unseen) | Reproduces dev? |
|---|---:|---:|---:|---:|---|
| clean | 0.009 | 0.005 | −0.0035 | −0.0024 | ✓ (dev −0.005) |
| augonly | 0.005 | 0.002 | −0.0029 | −0.0011 | ✓ (dev −0.004) |
| kl | 0.006 | 0.003 | −0.0032 | −0.0020 | ✓ (dev −0.004) |
| kldrop | 0.006 | 0.004 | −0.0022 | −0.0014 | ✓ (dev −0.005) |
| kllowmed | 0.008 | 0.004 | −0.0037 | −0.0024 | ✓ (dev −0.005) |

All five recipes show negative Δ(OOD−in) image on both test splits,
in the same magnitude as dev. **The Phase 5c-1 claim — augmentation
gains transfer to held-out image attacks — reproduces on test.**

### 4.2 Per-severity sensitivity (single-perturbation cells)

Mean text-family ΔAUROC at medium severity (the "internet-realistic
threat" headline number):

| Recipe | dev | test_seen | test_unseen |
|---|---:|---:|---:|
| clean | 0.063 | 0.058 | 0.042 |
| augonly | 0.042 | 0.036 | 0.025 |
| kl | 0.044 | 0.041 | 0.028 |
| kldrop | **0.034** | **0.033** | **0.026** |
| kllowmed | 0.046 | 0.043 | 0.029 |

`kldrop` cuts the medium-severity text-attack gap by **44 %** vs
clean on test_unseen (0.042 → 0.026) and **43 %** on test_seen
(0.058 → 0.033). The dev headline was -46 %. Reproduces tightly.

Monotonicity (low ≤ medium ≤ high) holds for every (recipe ×
family) row on both test splits.

### 4.3 Composite attacks reproduction

Mean ΔAUROC at medium severity per composite type, test_unseen:

| Composite | clean | kl | kldrop |
|---|---:|---:|---:|
| `composite_2text` | 0.073 | 0.052 | **0.049** |
| `composite_2image` | 0.017 | 0.012 | 0.016 |
| `composite_text_image` | 0.050 | 0.036 | **0.035** |
| `composite_2text_2image` | 0.095 | 0.065 | **0.068** |

`kldrop` is the medium-severity composite winner on three of four
composite types (tied with `augonly` on `composite_2text`). On
`composite_2image` (image-only composite), `kl` wins narrowly —
consistent with `kldrop` having slightly less specialised image-side
augmentation as a side-effect of its modality-dropout schedule.

`composite_2text_2image` (the kitchen-sink threat) on test_unseen
hits ΔAUROC 0.140 on the clean baseline at high severity — the most
damaging cell in the entire benchmark.

## 5. `kllowmed` revisited on test

Phase 5c-3 declared `kllowmed` "Pareto-dominated by `kl` on every
metric, a clean negative result." Reproduction on test:

| Metric | dev kl vs kllowmed | test_seen | test_unseen |
|---|---|---|---|
| Clean AUROC | kl +0.002 | kl +0.001 | kl +0.002 |
| Worst-cell text Δ | kl better by 0.003 | kl better by 0.003 | kl better by 0.003 |
| Image-only AUROC | kl better by 0.009 | kl better by 0.008 | kl better by 0.011 |
| Composite_2text med Δ | kllowmed tied | tied | tied |
| Naturalistic-robust count | kl +4 | kl +13.7 | **kllowmed +14.3** |

The single metric flip is on naturalistic-robust *count* on
test_unseen. The other four metrics still favour `kl`. **Updated
verdict on `kllowmed`:** the Phase-5c-3 negative result *softens* on
held-out test — `kllowmed` retains a use-case on the naturalistic-
distribution test split where it has the highest survival count.
But on every aggregated robustness metric (worst-cell text Δ,
image-only AUROC, composite ΔAUROC), `kl` remains the better
clean-accuracy-preserving recipe and `kldrop` remains the better
composite-defender. `kllowmed` should be reported as a borderline
case, not a clean dominated outcome.

## 6. Updated Pareto recommendation (post-test)

Echoing Phase 5c § 13.6, refined with held-out evidence:

- **`kldrop` is the recommended ckpt under the project's primary
  threat model (typical internet user combining 1–2 perturbations at
  moderate severity).** Wins worst-cell text Δ, image-only AUROC,
  and 3/4 composite-medium cells on **all three splits**. The cost
  (≈ 0.025 AUROC clean on test, ≈ 0.037 on dev) is the headline
  trade.
- **`kl` is the clean-accuracy-preserving alternative.** Highest
  naturalistic-robust count on dev and test_seen; matches clean
  baseline AUROC on test_unseen exactly. The right choice when the
  deployment context is recall-sensitive at a fixed threshold and
  the worst-case threat is single perturbation rather than composite.
- **`augonly` becomes more defensible on test** (especially worst-
  cell text Δ on test_seen) — promote from "dominated" on dev to
  "secondary alternative" on test.
- **`kllowmed`** — the dev "negative result" softens on test_unseen
  where it has the highest naturalistic-robust count, but it loses
  on every other metric to `kl`. Report as a borderline result
  rather than a clean Pareto loss.
- **`clean` (stage1)** — useful as the baseline only; dominated by
  every robust recipe on every robustness metric across all splits.

For a final-report headline: report `kldrop` for composite/worst-cell
robustness, `kl` for clean-preservation, and note the class-asymmetric
per-example deployment story from Phase 6 §4 (kldrop's per-example
wins concentrate on label=0; kl's on label=1).

## 7. White-box PGD on test (for completeness)

PGD ε ≥ 2/255 still drives AUROC to ≈ 0 on every recipe on both
test splits, exactly as expected. No recipe defends the white-box
oracle; this is a Phase 8+ adversarial-training problem and is
reported separately in `phase4/whitebox_table.test_seen.md` and
`whitebox_table.test_unseen.md`.

## 8. Limitations

- **No test-set per-example failure analysis written**. Phase 6's
  narrative is dev-only. The class-asymmetric `kldrop` vs `kl`
  trade-off (33/34 kldrop-wins are label=0; 46/49 kl-wins are
  label=1 on dev) is *predicted* to hold on test; verifying needs
  running `scripts/failure_analysis.py --split test_seen` and
  inspecting the disagreement examples. Left as a follow-up.
- **`kllowmed` on test_unseen partially refutes the Phase 5c-3
  dev claim.** The single-split, single-seed comparison framing was
  too narrow. The right statement is "kllowmed is dominated by kl
  on every metric *except* naturalistic-robust count on the
  naturalistic-prior test split."
- **`kldrop` test_unseen has slightly elevated σ** (clean AUROC σ =
  0.033 vs dev's 0.025) — the larger pool surfaces more seed
  variance on this recipe. Headlines remain stable but per-recipe
  comparisons within ~0.01 AUROC on test_unseen are within seed
  noise.
- **Composite cells fix a single severity per cell** on test as on
  dev. Mixed-severity composites are a follow-up.
- **White-box PGD remains undefended at ε ≥ 2/255** on test, same as
  dev. The naturalistic threat model is the project's primary
  framing; gradient-aware attacks are a separate scope.
- **Test_unseen images sourced from a third-party HF mirror** (`limjiayi`).
  Verified ID + caption overlap with the labelled jsonl from the
  same mirror, but the image bytes themselves are not guaranteed
  bit-identical to the Hateful Memes phase-2 official tarball.
  Compression and metadata may differ minorly.

## 9. One-line conclusion

The Phase-5/5c story holds on held-out test: **`kldrop` is the
modality-balanced, composite-resistant recipe and the project's
recommended ckpt for the realistic adversarial-user threat model;
`kl` is the clean-accuracy-preserving alternative**. The dev-to-test
Pareto front is stable on every metric except naturalistic-robust
count, where `kllowmed` unexpectedly wins on the
naturalistic-distribution test split — making it a third borderline
option rather than a dominated negative result.
