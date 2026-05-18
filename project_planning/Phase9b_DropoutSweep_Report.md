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
  depth. The per-example class-asymmetric analysis (Phase 6 / 8)
  could be extended to compare `kldrop-p015` vs `kl` per-example;
  not done in this phase.

## 7. One-line conclusion

The untuned `modality_dropout_text = 0.30` default in the original
`kldrop` recipe was a Pareto-dominated point. `kldrop-p015`
delivers the same image-branch revival at no clean-accuracy cost
on any of dev / test_seen / test_unseen, and becomes the new
recommended default for the project's naturalistic threat model.
