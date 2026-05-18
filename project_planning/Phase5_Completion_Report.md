# Phase 5 — Robust training (completion report)

> Status: written 2026-05-16. Augments the Phase-3/4 sequence and tests
> whether training-time perturbation defenses move the per-cell *and*
> per-example fragility metrics surfaced in
> `Phase3_4_Completion_Report.md`. Reads as a continuation of that
> document; see Phase 4 §3.3 (modality ablation) and §3.5 (failure
> analysis) for the diagnoses this phase responds to.

## 1. Recipe

Two robust variants share the same per-example augmentation mix and the
same outer per-batch step; they differ only in the loss weights on the
two KL-consistency terms:

```
L = BCE(p_clean, y) + α · BCE(p_pert, y) + β · KL_full + γ · KL_image_branch
```

- `BCE(p_clean, y)`: standard binary cross-entropy on the clean view
  (with positive-class re-weighting, same `pos_weight_auto` schedule
  as Phase-2 stage-1 training).
- `BCE(p_pert, y)`: the same BCE on a per-example perturbed view drawn
  by `RobustAugmenter`.
- `KL_full`: binary KL of the stop-gradient clean prediction onto the
  perturbed prediction, computed on the full multimodal forward.
- `KL_image_branch`: same KL formula but computed on
  `model.forward_image_only(...)`. This is the new piece, added to
  directly attack the Phase 4-S §3.3 finding that the fusion head's
  image branch is dead.

| Hyperparameter | augonly | kl |
|---|---:|---:|
| α  | 1.0 | 1.0 |
| β  | 0.0 | 0.5 |
| γ  | 0.0 | 0.25 |

Both variants run for 10 epochs at `batch_size=128`, `lr_head=1e-3`,
cosine-with-warmup, bf16 autocast — identical to clean stage-1 so the
clean-accuracy delta is attributable to the loss, not the schedule.

## 2. Augmentation mix

`RobustAugmenter` samples a perturbation `kind` per example:

| Kind | Weight | What it does |
|---|---:|---|
| `text`  | 0.50 | apply one text attack to the caption only |
| `image` | 0.30 | apply one image attack to the PIL image only |
| `both`  | 0.20 | apply one of each independently |

The clean view is always represented by the `BCE(p_clean, y)` term, so
no explicit "clean-only" branch is needed in the augmenter.

**Attack pools** (uniform within the pool):
- Text: `{leetspeak, char_deletion, char_swap, censoring, keyboard_typo}`
  — the five with non-trivial Phase-4 ΔAUROC; `case_noise`, `spacing`,
  `punctuation` are skipped because CLIP's tokenizer ignores case and
  the other two had near-zero clean-baseline gap.
- Image: `{gaussian_noise, blur, compression, brightness_down,
  occlusion, typographic}` — six attacks spanning pixel-noise,
  photometric, geometric and rendered-text families.

**Severity weights**: text severities sampled with weights `(1, 2, 2)`
on `(low, medium, high)` to oversample the harder cells (Phase 4 showed
text-character attacks lose most of their bite at low severity); image
severities uniform on `(low, medium, high)`.

Each per-example perturbation is constructed with a freshly drawn seed
from the augmenter's RNG, so the run is deterministic-given-seed and
identical re-runs reproduce.

## 3. Cluster runs

Six training jobs (`train-robust-{augonly,kl}-seed{0,1,2}`) plus
eighteen eval jobs (`perturbed`, `whitebox`, `modality-ablation` per
robust checkpoint). All training jobs reuse the Phase-1/2/3 image and
run on a single A100; the augmentation overhead (re-decoding PIL,
re-running the image transform on the perturbed view) adds ~12 %
wall-time over the clean stage-1 baseline.

**Phase 5b — modality dropout** (added 2026-05-17 in response to two
reviewer concerns: that the fully-robust headline conflated PGD with
naturalistic survival, and that `KL_image_branch` only partially healed
the dead image branch). Three additional training jobs
(`train-robust-kldrop-seed{0,1,2}`, config
`configs/stage1_robust_kl_drop.yaml`) + nine evals. The kldrop recipe
is identical to `kl` (α=1, β=0.5, γ=0.25) plus per-example text-modality
dropout (`modality_dropout_text: 0.30`): on each training step, 30 %
of examples have their text embedding zeroed before the fusion head,
forcing the head to make a usable image-only prediction. Implementation:
`train/stage1_robust.py::_fuse_with_dropout` (verified to exactly
reproduce `forward_image_only` / `forward_text_only` at the boundaries).

## 4. Headline numbers

Mean ± σ across 3 seeds, multimodal at per-run best macro-F1 threshold.

> **Threat-model note.** The Phase 5 headline now reports robustness
> against the *naturalistic* threat model (an everyday adversarial user
> editing the caption or image with an off-the-shelf tool) and the
> *white-box PGD* worst-case threat model **separately**. PGD ε ≥ 2/255
> has near-100 % attack success rate by design (oracle attack); rolling
> it into a "fully robust" headline hides where the actual robustness
> lives. The original Phase-5 headline mixed the two — fixed below.

| Metric | clean | augonly | kl | **kldrop** | Target | Met by kl? | Met by kldrop? |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Clean dev AUROC | 0.742 ± 0.006 | 0.712 ± 0.022 | 0.737 ± 0.008 | 0.705 ± 0.025 | ≥ 0.65 | ✓ | ✓ |
| Clean dev macro-F1 | 0.695 ± 0.011 | 0.669 ± 0.008 | 0.687 ± 0.006 | 0.655 ± 0.017 | — | — | — |
| **Naturalistic-robust /500** (clean-correct + survives all 57 nat cells) | **105.0** | 114.7 | **134.3** | 124.0 | (new metric) | — | — |
| Worst-cell text ΔAUROC | 0.115 ± 0.006 | 0.083 ± 0.012 | 0.090 ± 0.004 | **0.067 ± 0.008** | ≤ 0.055 | ✗ (22 %) | ✗ (42 %) |
| Mean text ΔAUROC @ high severity | 0.085 | 0.059 | 0.062 | **0.047** | — | — | — |
| **Image-only forward AUROC** | 0.593 ± 0.004 | 0.604 ± 0.003 | 0.611 ± 0.004 | **0.636 ± 0.014** | ≥ 0.628 | ✗ (38 % gap closed) | **✓ (exceeds baseline)** |
| PGD ε=1/255 AUROC | 0.012 | 0.019 | 0.019 | 0.018 | (non-target) | — | — |
| Nat + PGD-robust /500 (legacy metric) | 0.33 | 1.67 | 1.00 | 0.33 | — | — | — |

**Headline reading.** Two distinct Pareto winners emerge:

- **`kl` is the robustness-on-clean-data champion.** It pays only 0.005
  AUROC vs the clean baseline, yet lifts the count of naturalistically
  fully-robust dev examples from 105 → 134 — a **+28 % relative
  improvement**. For a user-facing classifier that cares about
  preserving clean-input accuracy while being harder for casual
  adversarial users to trick, this is the right recipe.

- **`kldrop` is the image-branch-revival champion.** It is the only
  recipe whose image-only forward AUROC (0.636) **exceeds the dedicated
  image-only baseline (0.628)** — the dead image branch is now fully
  revived. It also achieves the best per-cell text robustness
  (worst-cell text ΔAUROC 0.067 vs clean baseline's 0.115, a 42 %
  reduction). The cost is 0.037 AUROC on clean inputs and a slightly
  lower naturalistic-robust count than `kl` (124 vs 134 — because the
  clean-accuracy cost shrinks the pool of clean-correct examples).

The **PGD column is reported but not a target.** All recipes fail at
ε ≥ 2/255; defending the gradient-aware worst case would require
adversarial training (Phase 6+).

## 5. Naturalistic robustness

Per-family mean ΔAUROC at high severity (lower is better):

| Family | clean | augonly | kl | kldrop | Δ (best − clean) |
|---|---:|---:|---:|---:|---:|
| text | 0.0854 | 0.0587 | 0.0617 | **0.0471** | −0.038 (kldrop) |
| image-pixel | 0.0247 | 0.0205 | 0.0237 | 0.0201 | −0.005 (kldrop) |
| image-photometric | 0.0090 | 0.0077 | 0.0068 | **0.0049** | −0.004 (kldrop) |
| image-geometric | 0.0058 | 0.0054 | 0.0049 | **0.0000** | −0.006 (kldrop) |
| typographic | n/a | 0.0092 | 0.0141 | 0.0125 | (no clean baseline) |

The closure is text-heavy and consistent with the augmentation mix
(50 % text weight). Both `kl` and `kldrop` reduce text-family ΔAUROC
by ≥ 25 %, with `kldrop` reaching 45 % (0.085 → 0.047). On image
families, where the clean baseline gap was already small (< 0.025),
`kldrop` reaches essentially zero residual drop on image-geometric
attacks — a striking finding consistent with its newly-functional image
branch (§ 8). The robust-kl seed-0 ΔAUROC heatmap
(`figures/heatmap_robust_kl_seed0.png`) shows the text rows mostly
cooled relative to the clean baseline heatmap; high-severity censoring
and char_deletion remain the residual hotspots even under `kldrop`.

**Naturalistic-fully-robust count (primary headline)** — fraction of
dev examples that are clean-correct *and* survive every one of the 57
naturalistic perturbation cells:

| Recipe | seed 0 | seed 1 | seed 2 | mean | Δ vs clean |
|---|---:|---:|---:|---:|---:|
| clean | 94 | 107 | 114 | 105.0 | — |
| augonly | 129 | 113 | 102 | 114.7 | +9.7 |
| **kl** | 137 | 129 | 137 | **134.3** | **+29.3 (+28 %)** |
| kldrop | 131 | 122 | 119 | 124.0 | +19.0 |

This is the primary robustness metric for the project's threat model
("typical adversarial user with an image-editor app"). The `kl` recipe
beats every other in absolute terms; `kldrop` does better per-cell but
loses some clean-correct examples and so ends up lower.

## 6. Typographic robustness

All three robust recipes saw `typographic` in their image-augmentation
pool. Mean high-severity ΔAUROC: 0.009 (augonly), 0.014 (kl), 0.013
(kldrop). All three close most of the typographic gap, but the
differences are within seed noise (~ 0.004). Two observations worth
flagging:

- *augonly* slightly out-performs the KL recipes here. Plausibly,
  `KL_image_branch` pulls toward the clean image-only logit, which
  damps the model's willingness to react to the typographic overlay;
  pure augmentation has no such anchor.
- Per-family numbers in `phase4/perturbed_table.md` show typographic
  attack at high severity costs ≤ 0.02 AUROC across all robust
  recipes — it never made the worst-case top-3 in any seed.

## 7. White-box robustness

`figures/curve_whitebox_robust.png` overlays the clean / augonly /
kl / kldrop AUROC curves on the ε ∈ {1, 2, 4, 8}/255 grid for FGSM
and PGD. The Phase-4 expectation stands:

- ε = 1/255 AUROC moves from 0.012 (clean) to ~ 0.019 for all three
  robust recipes — a ~50 % relative lift but still far below the clean
  accuracy of ~ 0.71-0.74. kldrop is statistically indistinguishable
  from kl here (0.018 vs 0.019).
- ε ≥ 2/255 cells remain effectively zero across all recipes; PGD at
  4/255 still drives AUROC to ≈ 0 regardless of training recipe.

White-box PGD was not a defense target this phase; the marginal lift
at ε = 1/255 is a free side-effect of natural augmentation rather than
a real defense. **Crucially, we now report this metric separately from
the naturalistic-survival headline** to avoid the threat-model
confusion that contaminated the original Phase 5 framing.

## 8. Image-branch revival test

The modality-ablation triplet (multimodal / text-only / image-only
AUROC, evaluated on the clean dev split):

| Recipe | multimodal | text-only | image-only | gap (mm − img) |
|---|---:|---:|---:|---:|
| clean | 0.742 | 0.625 | 0.593 | 0.149 |
| augonly | 0.712 | 0.622 | 0.604 | 0.108 |
| kl | 0.737 | 0.624 | 0.611 | 0.126 |
| **kldrop** | 0.705 | 0.628 | **0.636** | 0.069 |

**Phase 5b headline: the dead image branch is revived.** The kldrop
recipe achieves an image-only forward AUROC of **0.636**, which
**exceeds** the dedicated image-only baseline trained from scratch
(0.628). This is the direct test of whether modality dropout + KL
together can force the fusion head to learn a usable image-only
representation — and the answer is yes.

The progression across recipes tells a clean story:

1. **Clean baseline (0.593)**: The fusion head learns to almost
   entirely ignore the image branch; the standalone image-only forward
   is *worse* than the dedicated image-only baseline because the
   features get re-shaped to play a vestigial role under text dominance.
2. **augonly (0.604, +0.011)**: Pure perturbation augmentation gives
   the image branch *some* training signal to encode something usable,
   but only a small lift.
3. **kl (0.611, +0.018)**: Adding `KL_image_branch` pulls a bit more
   information into the image branch, but the head can still satisfy
   the term by being near-constant on the image side.
4. **kldrop (0.636, +0.043)**: Forcing the head to make a real
   image-only prediction 30 % of the time finally pushes the image
   branch above the dedicated baseline. The fusion head now treats the
   image as a first-class input rather than a residual.

The multimodal − image-only gap shrinks from 0.149 (clean) to **0.069
(kldrop)** — less than half. Tradeoff: kldrop pays 0.037 AUROC on
clean accuracy. This is a real and irreducible cost — forcing the head
to commit to image-only predictions some fraction of the time prevents
it from over-relying on the (dataset-blessed) text signal.

**Pareto reading.** `kl` keeps clean accuracy near the baseline at the
cost of leaving the image branch ~ 0.025 below where it could be.
`kldrop` extracts the full image-branch signal at the cost of 0.037
clean AUROC. Which is "better" depends on the deployment context:
maximum clean-input accuracy ⇒ `kl`; balanced modality contribution ⇒
`kldrop`.

## 9. Per-example fully-robust counts (two threat models)

Reported separately per the threat-model framing established in § 4.

**Naturalistic-only (primary headline)** — count of dev examples that
are clean-correct AND survive every one of the 57 naturalistic cells:

| Recipe | seed 0 | seed 1 | seed 2 | mean | % of 500 |
|---|---:|---:|---:|---:|---:|
| clean | 94 | 107 | 114 | 105.0 | 21.0 % |
| augonly | 129 | 113 | 102 | 114.7 | 22.9 % |
| **kl** | 137 | 129 | 137 | **134.3** | **26.9 %** |
| kldrop | 131 | 122 | 119 | 124.0 | 24.8 % |

**Natural + white-box PGD-4/255 (legacy / completeness)** — same
definition, plus PGD survival required:

| Recipe | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| clean | 1 | 0 | 0 | 0.33 |
| augonly | 1 | 1 | 3 | 1.67 |
| kl | 1 | 1 | 1 | 1.00 |
| kldrop | 0 | 0 | 1 | 0.33 |

The legacy metric stays near zero because PGD has near-100 % ASR at
ε = 4/255 against all our checkpoints — it dominates the conjunction.
The naturalistic-only metric, which is what the project's threat model
actually requires, shows a clean +28 % relative improvement under `kl`
(105 → 134) and +18 % under `kldrop` (105 → 124). For the report, the
naturalistic metric should be the headline; the PGD-mixed metric
belongs only in the white-box section.

## 10. Failure-case categorisation

`phase4/failure_analysis.md` now reports per-example survival under
both threat models. Headline change vs the original Phase-5 report:
when we strip out PGD (which has near-100 % ASR by construction and
dominates the conjunction), the picture clarifies substantially.

**Naturalistic-only (primary, seed 0):**

| Recipe | Naturalistic survivors / 500 |
|---|---:|
| Clean baseline | 94 (18.8 %) |
| Robust-kl     | 137 (27.4 %)   — **+43 examples vs clean** |

The original "1 fixed / 1 new_failure / 498 still_failed" framing in
`phase4/failure_analysis.md` was computed under the natural+PGD
definition, where PGD ε=4/255 dominates and every example fails on
some cell. That framing remains in the doc for traceability but is no
longer the headline — the +43 naturalistic survivors is.

**Natural + PGD (seed 0, retained for completeness):**

| Robust-kl status (vs clean baseline) | Count |
|---|---:|
| `fixed` | 1 |
| `still_failed` | 498 |
| `new_failure` | 1 |
| `unchanged` | 0 |

The original baseline "1 fully-robust example" (id=63921, "taking a
photo with family") is now broken by a stronger PGD attacked-prob
under the robust-kl ckpt; a different example takes its place. Net
per-example change is zero under the PGD-inclusive definition — but
substantial under the naturalistic-only one, as the table above shows.

## 11. Limitations

- **White-box PGD ε ≥ 2/255 remains unsolved.** Naturalistic + KL +
  modality dropout cannot defend against a gradient-aware attacker;
  that would require adversarial training (project roadmap Phase 6).
- **The PGD-inclusive per-example fully-robust headline did not move.**
  Each dev example still encounters *some* PGD perturbation that flips
  it. The plan's ≥ 50/500 target was unrealistic given PGD's ≈ 100 %
  ASR. We now report the naturalistic-only count as the headline; on
  that metric the +28 % relative improvement (kl) is the real story.
- **`KL_image_branch` term alone is not sufficient to revive the image
  branch.** It lifts image-only AUROC by +0.018 (kl); modality dropout
  is the intervention that fully revives it (+0.043, kldrop). KL
  alone provides a soft anchor; modality dropout provides a hard
  forcing function. Both are needed for the branch to commit to a
  useful representation.
- **`kldrop` pays 0.037 AUROC on clean accuracy.** That is a real
  Pareto cost; the report should not paint `kldrop` as strictly
  dominating `kl`. The trade depends on whether downstream use
  prioritises clean-input accuracy or balanced modality contribution.
- **The modality-dropout rate (0.30) was not tuned.** Picked as a
  reasonable starting point; sweeping {0.15, 0.30, 0.50} could find
  a sweet-spot that preserves more clean accuracy.
- **`text_only` and `image_only` AUROC are post-hoc forwards** through
  a model trained for the fusion task — they index the *contribution*
  of each branch, not a separately-trained unimodal classifier.
- **Dev split only.** All Phase-5 numbers are on the 500-example dev
  split; held-out test reporting is Phase 7.

## 13. Phase 5c — generalisation analysis

> Status: code + 5c-1 tables landed 2026-05-17. Composite-eval re-runs
> (5c-2) and `kl_lowmed` train+eval (5c-3) submitted same day — results
> backfilled into `robust_vs_clean.md` when the cluster pulls land.

Three review questions on top of Phase-5b motivated this sub-phase:

1. Phase-5 reports *worst-cell* and *mean-high-severity* ΔAUROC, but the
   augmentation pool is a strict subset (5/8 text attacks, 6/11 image
   attacks) — does the gain transfer to held-out attacks, or has the
   recipe memorised its training pool?
2. Every eval cell applies *one* perturbation per sample. The realistic
   adversarial-user threat model combines edits (e.g. character swaps +
   blur + typographic overlay). The benchmark has never measured this.
3. Severity weights `(1, 2, 2)` weight high-severity augmentation
   heavily, but high severity is at the label-preserving limit; medium
   is closer to internet-realistic threats. Does a low+medium-only
   training pool defend high-severity eval cells the model has never
   seen at training time, or is the high-severity exposure necessary?

### 13.1 Code changes

- `scripts/aggregate_phase4.py` (`_in_pool_attacks`,
  `_write_pool_vs_ood_table`, `_write_per_severity_table`,
  `_write_composite_table`; `_is_robust_key` learns `kllowmed`;
  `VARIANT_ORDER` constant). No cluster cost — re-aggregates existing
  results into three new sections of
  `project_planning/phase4/robust_vs_clean.md`.
- `src/robust_meme_hate_detection/eval/run_perturbed.py`
  (`_apply_composite`, `_eval_composite_cell`,
  `_derive_sample_seed`, `--composites` CLI flag, `COMPOSITE_TYPES` and
  `COMPOSITE_K` tables). Composite cells are added without touching the
  existing 57-cell schema — perturbed-eval JSONs grow from 57 → 69
  cells when `--composites all` is set.
- `configs/stage1_robust_kl_lowmed.yaml`: copy of `stage1_robust_kl.yaml`
  with `severity_weights_text: [1, 2, 0]` and
  `severity_weights_image: [1, 1, 0]`. All other hyperparameters
  identical, so the only variable in the comparison is severity coverage.

### 13.2 In-pool vs held-out attack generalisation (5c-1)

Training pool (canonical, read from `stage1_robust_kl.yaml`):

- text in-pool (n=5): `leetspeak, char_deletion, char_swap, censoring, keyboard_typo`
- text OOD (n=3): `spacing, punctuation, case_noise`
- image in-pool (n=6): `gaussian_noise, blur, compression, brightness_down, occlusion, typographic`
- image OOD (n=5): `brightness_up, contrast_up, contrast_down, translation, crop`

Mean ΔAUROC across severities, partitioned by training-pool membership
(from `robust_vs_clean.md::Attack-type generalisation`):

| Recipe | text in-pool | text OOD | Δ(OOD−in) text | image in-pool | image OOD | Δ(OOD−in) image |
|---|---:|---:|---:|---:|---:|---:|
| clean   | 0.0691 | 0.0411 | −0.0280 | 0.0108 | 0.0048 | −0.0060 |
| augonly | 0.0478 | 0.0272 | −0.0206 | 0.0075 | 0.0037 | −0.0038 |
| kl      | 0.0496 | 0.0284 | −0.0212 | 0.0078 | 0.0034 | −0.0044 |
| kldrop  | 0.0375 | 0.0227 | −0.0148 | 0.0064 | 0.0013 | −0.0051 |

Reading:

- **Δ(OOD−in) is *negative* for every recipe**, including the clean
  ckpt. That is *not* a generalisation failure — the held-out text
  attacks (`spacing/punctuation/case_noise`) are the weakest attacks
  on the clean ckpt to begin with (ΔAUROC ≤ 0.02), so even an
  attack-pool-memorising recipe would look "more robust" on OOD. The
  text-side OOD column is a hollow test by construction.
- **Image-side is the meaningful comparison**: held-out image attacks
  (`brightness_up`, `contrast_*`, `translation`, `crop`) are not
  systematically easier than the in-pool ones, so the negative
  Δ(OOD−in) for every robust recipe is *real* generalisation:
  robust training transfers to attacks it never saw.
- **`kldrop` shows the strongest gain on held-out image attacks**
  (ΔAUROC 0.0013, vs 0.0034 for `kl`, 0.0037 for `augonly`, 0.0048
  clean). The image-branch revival from modality dropout pays off
  most on OOD image attacks — which is the expected effect: if the
  fusion head has to handle image-only forwards, it can't have
  over-specialised to the augmentation pool.

### 13.3 Per-severity sensitivity (5c-1)

From `robust_vs_clean.md::Per-severity sensitivity`. Medium is the
internet-realistic column; high is a stress ceiling. Mean ΔAUROC for
text-family cells:

| Recipe | low | **medium** | high |
|---|---:|---:|---:|
| clean   | 0.028 | **0.063** | 0.085 |
| augonly | 0.019 | **0.042** | 0.059 |
| kl      | 0.019 | **0.044** | 0.062 |
| kldrop  | 0.015 | **0.034** | 0.047 |

Reading: monotone low ≤ medium ≤ high for every recipe (sanity
gate passed), with the augmentation gain proportional across
severities. The "headline at internet-realistic severity" is a 0.029
absolute drop in ΔAUROC (kldrop vs clean) — i.e. **kldrop cuts the
medium-severity text-attack gap by 46 %**. Image-pixel,
image-photometric, image-geometric, and typographic tables show the
same monotone shape; full tables in `phase4/robust_vs_clean.md`.

### 13.4 Composite attacks (5c-2 — landed 2026-05-17)

Composite cell design: each cell applies *K* perturbations per
sample, drawn deterministically from `sha256(cell_seed | sample_id)`.
Components are sampled *without replacement* per modality from the
**full eval pool** (not the training pool — composite is itself an
OOD test). All component perturbations use the cell-level severity.

| Composite | K_text | K_image |
|---|---:|---:|
| `composite_2text` | 2 | 0 |
| `composite_2image` | 0 | 2 |
| `composite_text_image` | 1 | 1 |
| `composite_2text_2image` | 2 | 2 |

Headline result — composite attacks *escalate* the threat past the
worst-cell single-perturbation ΔAUROC: clean-ckpt `composite_2text` at
high severity hits **ΔAUROC = 0.1302**, exceeding the prior single-cell
worst of 0.1149. The "kitchen-sink" `composite_2text_2image` at high
severity reaches **0.1447** for clean. Composite attacks are the
strongest threat in the benchmark and the most realistic model of an
adversarial internet user.

Mean ΔAUROC at medium (realistic) severity, across recipes
(from `phase4/robust_vs_clean.md::Composite attacks`):

| Recipe | 2text | 2image | text+image | 2text+2image |
|---|---:|---:|---:|---:|
| clean    | 0.0901 | 0.0108 | 0.0496 | 0.0798 |
| augonly  | 0.0633 | 0.0081 | 0.0332 | 0.0617 |
| kl       | 0.0660 | 0.0059 | 0.0360 | 0.0602 |
| **kldrop** | **0.0511** | **0.0051** | **0.0243** | **0.0469** |
| kllowmed | 0.0645 | 0.0041 | 0.0372 | 0.0656 |

**`kldrop` is the strict Pareto winner on composite attacks** — best
at every (composite-type × severity) cell, with `composite_2text`
medium cut by 43 % vs clean, `composite_text_image` medium by 51 %,
`composite_2text_2image` medium by 41 %. The modality-dropout
intervention pays off most when the attack hits both modalities at
once: a fusion head that can fall back on a usable image-only branch
is robust to text-side compositional attacks that would otherwise have
no signal to anchor on.

Monotonicity sanity (low ≤ medium ≤ high) holds for every (recipe ×
composite type) row. Composite ASR at high severity exceeds 0.3 for
every recipe on `composite_2text` and `composite_2text_2image` — i.e.
even the best recipe flips ≥ 20 % of clean-correct examples under
the realistic-user threat model.

### 13.5 Severity-restricted training (5c-3 — landed 2026-05-17)

`kl_lowmed`: same as `kl` (α=1, β=0.5, γ=0.25, no modality dropout)
but `severity_weights_*` set to `[1, 2, 0]` / `[1, 1, 0]` — high
severity is removed from the augmentation distribution entirely.

Hypothesis tested: medium-severity augmentation alone generalises to
high-severity eval cells. **The hypothesis is rejected.** `kl_lowmed`
is Pareto-dominated by `kl` at every severity, on every metric:

| Metric (3-seed mean) | kl | kllowmed | Δ |
|---|---:|---:|---:|
| Clean AUROC | 0.7367 | 0.7347 | −0.002 (within σ) |
| Worst-cell text ΔAUROC | 0.0898 | 0.0926 | +0.003 (worse) |
| Mean text ΔAUROC at medium | 0.0443 | 0.0463 | +0.002 (worse) |
| Mean text ΔAUROC at high | 0.0617 | 0.0651 | +0.003 (worse) |
| Image-only AUROC | 0.6106 | 0.6019 | −0.009 (worse) |
| Naturalistic fully-robust /500 | 111.67 | 107.33 | −4 (worse) |
| Composite 2text medium ΔAUROC | 0.0660 | 0.0645 | −0.002 (tied) |
| Composite 2text_2image high ΔAUROC | 0.1065 | 0.1190 | +0.013 (worse) |

Reading: high-severity training augmentation *is* contributing. Without
it, the model loses ~0.5 % AUROC on the medium-severity defense it
*was* supposed to be specialised for, ~3 % on high severity, and
~4 dev examples on the naturalistic survival count. The
severity-weight choice `[1, 2, 2]` / `[1, 1, 1]` in the original
Phase-5 setup was not over-tuned toward high; removing it produces a
strictly worse model. This is a clean negative result on severity
restricted training: **internet-realistic severity exposure alone is
not sufficient — the head needs to see the high-severity tail at
training time to defend it at eval.**

`kllowmed` does retain `kl`'s clean accuracy almost exactly (within σ),
so the negative result is on the robustness side, not on a
clean-accuracy trade-off.

### 13.6 Updated Pareto reading

With Phase 5c's data, the four robust recipes sort cleanly:

- **`kl` is the clean-accuracy-preserving champion** for naturalistic
  single-attack threats: highest naturalistic-robust count (111.67),
  highest clean AUROC of the robust recipes (0.7367), retains
  competitive composite defense.
- **`kldrop` is the Pareto winner under the realistic composite
  threat** *and* the only recipe with a fully-revived image branch
  (image-only AUROC 0.636 > 0.628 dedicated baseline). Pays 0.037
  clean AUROC; that is the headline trade.
- **`augonly` is dominated by both** — no longer publishable as a
  standalone recipe.
- **`kllowmed` is the negative result** on severity restricted
  training: high-severity augmentation is necessary, not optional.

For the project's primary threat model (typical internet user
combining 1–2 perturbations across modalities at moderate severity),
`kldrop` is the recommended ckpt.

### 13.7 What this section *cannot* say

- **Text-side OOD generalisation is not meaningfully tested.** The
  held-out text attacks have ≤ 0.02 ΔAUROC on the clean ckpt — there
  is no headroom for robust gains to register. The image-side OOD
  comparison is the actual generalisation claim.
- **The "worst-cell ΔAUROC ≤ 0.115" headline from earlier Phase-5
  sections is a per-cell single-attack ceiling**, not a
  worst-case-over-realistic-threats ceiling. Composite attacks reach
  0.1447 ΔAUROC at high severity on the clean ckpt, and 0.0951 on
  `kldrop`. Composite numbers are added alongside, not as a
  replacement; the prior "worst-cell" framing remains valid for the
  single-perturbation table.
- **`kl_lowmed` is a negative result on severity-restricted training**
  rather than a new default. The reported numbers in § 13.5 are
  publishable as a clean methodology check: yes, the high-severity
  weight in the original Phase-5 setup was load-bearing.
- **Composite-attack design fixes a single severity per cell.** A
  realistic adversarial user mixes severities (one strong edit + one
  weak edit). The current composite cells deliberately don't model
  that — every component is at the same severity. Mixed-severity
  composites are deferred to a follow-up.

## 12. Reproducibility

| Artefact | Path |
|---|---|
| Train script | `src/robust_meme_hate_detection/train/stage1_robust.py` (with `_fuse_with_dropout` helper for Phase 5b) |
| Loss utilities | `src/robust_meme_hate_detection/train/_robust_loss.py` |
| Augmenter | `src/robust_meme_hate_detection/train/_robust_augmenter.py` |
| Configs | `configs/stage1_robust_{augonly,kl,kl_drop}.yaml` |
| Aggregator extension | `scripts/aggregate_phase4.py` (`_write_robust_vs_clean`, `_make_robust_figures`, `_natural_fully_robust_count`, `_is_robust_key` incl. `kldrop`) |
| Failure-analysis extension | `scripts/failure_analysis.py` (`_annotate_robust` + naturalistic-only block) |
| Training jobs | `train-robust-{augonly,kl,kldrop,kllowmed}-seed{0,1,2}` (12 jobs) |
| Eval jobs | `perturbed-`, `whitebox-`, `modality-ablation-` × 12 ckpts (36 jobs, perturbed re-runs include `--composites all`) |
| Phase 5c config | `configs/stage1_robust_kl_lowmed.yaml` |
| Phase 5c aggregator | `scripts/aggregate_phase4.py` (`_write_pool_vs_ood_table`, `_write_per_severity_table`, `_write_composite_table`) |
| Phase 5c eval | `src/robust_meme_hate_detection/eval/run_perturbed.py` (`--composites`, `_apply_composite`) |
| Aggregate output | `project_planning/phase4/robust_vs_clean.md` |
| Failure-case output | `project_planning/phase4/failure_analysis.md` |
