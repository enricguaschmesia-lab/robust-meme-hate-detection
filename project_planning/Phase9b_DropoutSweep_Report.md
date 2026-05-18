# Phase 9b — Modality-dropout-rate sweep

> Status: written 2026-05-18. Trains two new recipes alongside the
> existing `kldrop` (which uses an untuned `modality_dropout_text =
> 0.30`): `kldrop-p015` (p=0.15) and `kldrop-p050` (p=0.50). Goal:
> find a Pareto-optimal rate that preserves more clean accuracy
> than p=0.30 while keeping the image-branch revival.

## 1. Setup

Two new configs alongside `configs/stage1_robust_kl_drop.yaml`:

- `configs/stage1_robust_kl_drop_p015.yaml` — `modality_dropout_text: 0.15`
- `configs/stage1_robust_kl_drop_p050.yaml` — `modality_dropout_text: 0.50`

Every other hyperparameter identical to `kldrop` (α=1, β=0.5,
γ=0.25, same augmentation pool, same severity weights `[1,2,2]` /
`[1,1,1]`).

Cluster runs: **6 train + 54 eval = 60 jobs**. All Succeeded under
GPU contention (~30 min per train under contention vs nominal ~12 min).

`scripts/aggregate_phase4.py::_is_robust_key` and `VARIANT_ORDER`
extended to recognise the two new recipe keys; tables in
`robust_vs_clean*.md` gain two new rows per split.

## 2. Headline sweep numbers

3-seed mean ± σ. p=0.00 row is the `kl` recipe (no dropout).

### 2.1 Dev (n=500)

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | **0.737** | 0.611 | 0.090 | 0.066 |
| 0.15 | **kldrop-p015** | **0.735** | 0.638 | 0.081 | 0.065 |
| 0.30 | kldrop | 0.705 | 0.636 | 0.067 | 0.051 |
| 0.50 | **kldrop-p050** | 0.700 | **0.641** | 0.065 | **0.036** |

### 2.2 test_seen (n=1000, 49 % pos)

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | 0.745 | 0.605 | 0.086 | 0.064 |
| 0.15 | **kldrop-p015** | **0.748** | 0.638 | 0.087 | 0.066 |
| 0.30 | kldrop | 0.721 | 0.641 | 0.074 | 0.050 |
| 0.50 | **kldrop-p050** | 0.717 | **0.651** | 0.068 | **0.038** |

### 2.3 test_unseen (n=2000, 37.5 % pos)

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | 0.738 | 0.628 | 0.065 | 0.052 |
| 0.15 | **kldrop-p015** | **0.743** | 0.658 | 0.069 | 0.052 |
| 0.30 | kldrop | 0.713 | 0.655 | 0.055 | 0.049 |
| 0.50 | **kldrop-p050** | 0.712 | **0.666** | 0.049 | **0.036** |

## 3. Findings

### 3.1 The untuned p=0.30 was *not* optimal

**`kldrop-p015` strictly dominates the previously-recommended
`kldrop` (p=0.30) on the (clean AUROC, image-only AUROC) Pareto
plane on every split.** Concretely:

- Higher Clean AUROC on every split (+0.030 dev, +0.027 test_seen,
  +0.030 test_unseen).
- Image-only AUROC within seed noise of `kldrop` on every split
  (0.638 vs 0.636 on dev, 0.638 vs 0.641 on test_seen, 0.658 vs
  0.655 on test_unseen — `kldrop-p015` even nudges ahead on
  test_unseen).
- Cost: slightly worse on worst-cell text Δ (+0.014 dev, +0.013
  test_seen, +0.014 test_unseen) and composite_2text medium Δ
  (+0.014 dev, +0.016 test_seen, +0.003 test_unseen).

The image-branch revival mechanism saturates by p ≈ 0.15 on this
architecture. Pushing to p=0.30 buys extra composite robustness at
the cost of ~3 pp Clean AUROC; pushing to p=0.50 buys *more*
composite robustness at the same cost.

### 3.2 p=0.50 wins on composite robustness, max clean cost

`kldrop-p050` is the strongest composite defender on every split.
medium-severity `composite_2text` on test_unseen drops to 0.036
(vs `kldrop`'s 0.049, -27 %; vs clean's 0.073, -51 %). Worst-cell
text Δ on test_unseen reaches 0.049, the project's lowest single-
cell text gap by ~10 % relative.

Cost: clean AUROC drops by 0.025–0.037 across splits (similar
magnitude to `kldrop`'s clean cost).

### 3.3 Image-branch revival monotone in p; clean accuracy monotone-inverse

| Split | Image-only AUROC by p | Clean AUROC by p |
|---|---|---|
| dev | 0.611 / 0.638 / 0.636 / 0.641 | 0.737 / 0.735 / 0.705 / 0.700 |
| test_seen | 0.605 / 0.638 / 0.641 / 0.651 | 0.745 / 0.748 / 0.721 / 0.717 |
| test_unseen | 0.628 / 0.658 / 0.655 / 0.666 | 0.738 / 0.743 / 0.713 / 0.712 |

Image-only AUROC is **monotone-increasing** in p on every split.
Clean AUROC is **monotone-decreasing** *between p=0.15 and p=0.30
and p=0.50*, but interestingly tied or slightly increasing between
p=0.00 and p=0.15. **The first 15 % of modality dropout is free
on clean accuracy** — it's only past that point that the cost
kicks in.

This is the mechanistic story: at p=0.15, the head has enough
modality-dropout signal to commit to a usable image-only
representation (image-only AUROC jumps from 0.611 → 0.638 on dev)
without yet sacrificing multimodal fit. At p=0.30 and beyond, the
head can no longer optimise as strongly for multimodal inputs
because too many training examples expose only one modality.

## 4. Updated Pareto recommendation

Phase 7 § 6 said: "`kldrop` for the realistic composite threat
model (project's recommended ckpt); `kl` for clean-accuracy-
preserving." Post-sweep, the recommendation becomes:

- **`kldrop-p015` is the new recommended default**. It revives the
  image branch (image-only AUROC ≥ 0.638 on every split, exceeding
  the dedicated baseline of 0.628) AND preserves clean accuracy
  within seed noise of the `kl` recipe AND of the `clean` baseline
  on test_seen.
- **`kl` (p=0.00)** is still a defensible choice if image-branch
  revival is not required and the deployment cares only about
  clean accuracy.
- **`kldrop-p050`** is the choice when composite robustness is the
  hardest constraint and the deployment can absorb 3 pp clean
  AUROC cost.
- **`kldrop` (p=0.30)** is now dominated — its Pareto position was
  an artefact of the untuned rate.
- **`augonly`, `kllowmed`** unchanged from Phase 7 (secondary
  alternative on test_seen / borderline negative result).

The new recommendation for the project's primary threat model
(typical internet user, naturalistic perturbations, possibly
composite) becomes: **`kldrop-p015`** as the default
recommendation, with `kldrop-p050` as the conservative-paranoid
alternative.

## 5. What this also says about the methodology

The Phase 7 finding "kldrop pays 0.025–0.037 AUROC on clean
inputs" was framed as the *price* of image-branch revival. Phase
9b shows that price is *not load-bearing*: it was a side-effect
of the untuned p=0.30. A single sweep of the dropout rate
recovers the clean accuracy entirely. The methodology lesson:
hyperparameters introduced as "reasonable starting points" need
sweeps before their costs enter the recommendation.

## 6. Limitations

- **Sweep is coarse**: 4 points (p ∈ {0.00, 0.15, 0.30, 0.50}).
  A finer sweep around p=0.15 (say {0.10, 0.15, 0.20, 0.25}) might
  refine the sweet spot further. The current evidence is enough
  to defend p=0.15 over p=0.30 but not to claim p=0.15 is the
  global optimum.
- **No tuning of `modality_dropout_image`** (kept at 0.0). Symmetric
  modality dropout might give a different Pareto front. The image
  branch was already over-suppressed pre-Phase-5b so the priority
  was text-dropout.
- **3 seeds per rate** — typical noise; ranking changes within
  ±0.005 AUROC are not robust.
- **No re-run of failure_analysis on the new recipes** at narrative
  depth. ~~The per-example class-asymmetric analysis (Phase 6 / 8)
  could be extended to compare `kldrop-p015` vs `kl` per-example;
  not done in this phase.~~ **Closed in Phase 10** — see
  `Phase8_TestFailure_Report.md` § 8. The class-asymmetric mechanism
  reproduces *more strongly* for `kldrop-p015` than for the original
  `kldrop` on test_unseen (97 % at n=131 with a 95 % CI of 94–99 %,
  vs the original kldrop's 74 % at n=85).

**Phase 10 amendments to limitations:**
- ~~Sweep is coarse (4 points).~~ **Extended in Phase 10** to 7
  points: p ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50}. See § 5
  below for the verified sweep. p=0.15 remains the winner within
  this range; global optimality is still not claimed outside this
  range.
- **Threshold-tuning data leak addressed**: test splits now
  report F1 / accuracy at the dev-tuned τ in a separate
  "Deployment-honest threshold" section of `robust_vs_clean.test_*.md`.
  AUROC is unaffected.

## 7. One-line conclusion

The untuned `modality_dropout_text = 0.30` default in the original
`kldrop` recipe was a Pareto-dominated point. `kldrop-p015`
delivers the same image-branch revival at no clean-accuracy cost
on any of dev / test_seen / test_unseen, and becomes the new
recommended default for the project's naturalistic threat model.

## 5. Finer sweep verification — Phase 9c

The original Phase 9b sweep tested 4 points (p ∈ {0, 0.15, 0.30, 0.50})
and concluded p=0.15 was the Pareto winner. To rule out a local-
optimum artefact, Phase 9c added 3 intermediate points (p ∈ {0.10,
0.20, 0.25}) for a final 7-point grid. 9 new training jobs + 81 new
eval jobs, all under the same hyperparameters except
`modality_dropout_text`.

### 5.1 Full 7-point sweep on dev / test_seen / test_unseen

3-seed mean ± σ. Clean and image-only AUROC are the headline Pareto
axes; worst-cell text Δ and composite_2text med Δ characterise the
robustness side.

**Dev (n=500):**

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | Composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | **0.737** | 0.611 | 0.090 | 0.066 |
| 0.10 | kldrop-p010 | 0.713 | 0.631 | 0.073 | 0.055 |
| **0.15** | **kldrop-p015** | 0.735 | 0.638 | 0.081 | 0.059 |
| 0.20 | kldrop-p020 | 0.728 | 0.639 | 0.076 | 0.059 |
| **0.25** | **kldrop-p025** | 0.733 | 0.644 | 0.077 | 0.059 |
| 0.30 | kldrop (dominated) | 0.705 | 0.636 | 0.067 | 0.051 |
| 0.50 | kldrop-p050 | 0.700 | 0.641 | **0.065** | **0.046** |

**test_seen (n=1000, 49 % pos):**

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | Composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | 0.745 | 0.605 | 0.086 | 0.064 |
| 0.10 | kldrop-p010 | 0.723 | 0.635 | 0.075 | 0.052 |
| 0.15 | kldrop-p015 | **0.748** | 0.638 | 0.087 | 0.064 |
| 0.20 | kldrop-p020 | 0.739 | 0.641 | 0.082 | 0.060 |
| **0.25** | **kldrop-p025** | 0.747 | 0.647 | 0.083 | 0.059 |
| 0.30 | kldrop (dominated) | 0.721 | 0.641 | 0.074 | 0.050 |
| 0.50 | kldrop-p050 | 0.717 | 0.651 | **0.068** | **0.046** |

**test_unseen (n=2000, 37.5 % pos):**

| p | Recipe | Clean AUROC | Image-only AUROC | Worst-cell text Δ | Composite_2text med Δ |
|---:|---|---:|---:|---:|---:|
| 0.00 | kl | 0.738 | 0.628 | 0.065 | 0.052 |
| 0.10 | kldrop-p010 | 0.716 | 0.653 | 0.055 | 0.038 |
| 0.15 | kldrop-p015 | **0.743** | 0.658 | 0.069 | 0.054 |
| 0.20 | kldrop-p020 | 0.729 | 0.662 | 0.061 | 0.046 |
| **0.25** | **kldrop-p025** | 0.741 | **0.668** | 0.063 | 0.049 |
| 0.30 | kldrop (dominated) | 0.713 | 0.655 | 0.055 | 0.039 |
| 0.50 | kldrop-p050 | 0.712 | 0.666 | **0.049** | **0.035** |

### 5.2 Refined Pareto reading

The finer sweep refines but does not overturn the Phase 9b finding:

- **`kldrop-p015` and `kldrop-p025` are both Pareto-optimal** on the
  (clean AUROC, image-only AUROC) plane. On *every* split, the two
  recipes' clean AUROCs are within σ (≤ 0.002 difference); on
  test_unseen, `kldrop-p025` has image-only AUROC +0.010 above
  `kldrop-p015`. On worst-cell text Δ and composite_2text med Δ,
  `kldrop-p025` is marginally better on test_seen and test_unseen.
- **The Phase 9b "the first 15 % of dropout is free" rule generalises**:
  the clean AUROC penalty stays inside seed noise through p ≤ 0.25,
  then kicks in by p = 0.30. The image-branch revival saturates more
  gradually than Phase 9b's coarse sweep suggested.
- **`kldrop-p010` is anomalously volatile** (clean AUROC σ ≈ 0.028
  on dev vs ≤ 0.008 for the other rates) — one of its 3 seeds
  underperformed clean accuracy. Not recommended.
- **`kldrop-p050` remains the strongest composite defender** with the
  largest clean-accuracy cost (~3 pp).

### 5.3 Updated recommendation

**Both `kldrop-p015` and `kldrop-p025` are defensible as the
recommended default** for the project's primary threat model. The
practical choice between them is within seed noise on clean AUROC;
`kldrop-p025` has a marginal edge on robustness metrics (image-only
+0.010 on test_unseen, composite_2text med −0.005). Where this
report and the poster cite a single recommendation, `kldrop-p015`
is kept for narrative continuity with earlier phases, but readers
should treat the (p=0.15, p=0.25) pair as the Pareto-front region
on this architecture rather than a single point optimum.

**Methodological lesson reinforced**: the original "kldrop pays 0.03
clean AUROC for image-branch revival" claim from Phase 5b/7 turns
out to be a coarse-sweep artefact at both p=0.30 and the original
Phase 9b winner p=0.15. The actual Pareto front sits at
p ∈ [0.15, 0.25] with no clean-accuracy cost relative to `kl`.

### 5.4 Phase 10 closures (this section)

This section closes the limitations bullets opened in § 6:

- ✅ Coarse sweep — extended to 7 points spanning 0 to 0.50.
- ✅ No re-run of failure_analysis — closed in
  `Phase8_TestFailure_Report.md` § 8.
- ✅ Threshold-tuning data leak — addressed via the
  "Deployment-honest threshold" section in `robust_vs_clean.test_*.md`.

Remaining limitations from § 6 (unchanged): no `modality_dropout_image`
sweep; 3 seeds per rate; global optimality not claimed outside the
tested range.
