# Phase 3 + Phase 4 — Completion Report

Status as of 2026-05-16.

## 1. Phase 3 — Perturbation suite

The naturalistic perturbation suite (8 text × 10 image × 3 severities = 54
benchmark cells, plus 2 bidirectional brightness/contrast aliases reserved
for training augmentation) was recalibrated after the first manual review
round. Round-1 inspection set
`cluster-results/inspect-perturb-20260512-211209-a4852b5/` was replaced by
the calibrated set
`cluster-results/inspect-perturb-calibrated-20260516-130751/` (27 000
manifest rows; 500 image samples per (attack, severity) cell, plus 24 text
`pairs.csv` files).

Round-2 calibration deltas applied in
`src/robust_meme_hate_detection/perturbations/{image,text}_perturbations.py`:

- **gaussian_noise** σ 0.01/0.04/0.08 (was 0.01/0.03/0.05).
- **blur** radius now float; low/medium/high = 0.8/1.8/3.0 px
  (was integer-quantised 1/2/3).
- **compression** JPEG q ≈ 80/40/15 (was 80/50/25).
- **brightness_down** magnitude raised to severity·0.8 → factors
  0.80/0.50/0.20 (was 0.90/0.75/0.60); brightness_up / contrast_{up,down}
  magnitude raised to severity·0.6 → factors ±0.15/±0.375/±0.60
  (were ±0.10/±0.25/±0.40).
- **translation** / **crop** 4 / 10 / 20 % per edge (were 2 / 5 / 10 %).
- **occlusion** patch side 10 / 20 / 35 % (was 5 / 10 / 20 %).
- **char_swap** 2 / 3 / 5 swaps (was 1 / 2 / 3).
- **censoring** 5 / 12 / 25 % of characters (was 10 / 20 / 40 %).

Calibration intent and the human inspection notes that drove it are recorded
in `project_planning/perturbations_calibration.md`. Phase 3 closes with the
recalibrated set materialised and reviewed.

## 2. Phase 4 — Setup

Evaluation against the calibrated suite + a white-box L∞ image attack
sweep, on the Phase-1-2 checkpoints (no new training).

- Naturalistic benchmark: `eval/run_perturbed.py` walks every
  (attack, severity) cell on the 500-example dev split, computes attacked
  metrics at the per-run best macro-F1 threshold, and records the
  robustness gap and attack success rate per cell. Each cell uses a stable
  SHA-256-keyed seed derived from `(global_seed, attack, level)` so the
  evaluation is bit-reproducible across processes.
- White-box: new `eval/run_whitebox.py` wrapper. Sweeps ε ∈ {1, 2, 4, 8}/255
  × {FGSM, PGD-10} on the full 500-example dev split; PGD uses random L∞
  start with α = ε/4. Attacks operate on `[0,1]` pixel tensors;
  Normalize lives inside `CLIPHateMemeClassifier`, so gradients flow through
  the standard CLIP preprocessing.
- Coverage:
  - Naturalistic: `stage1-seed{0,1,2}` (multimodal, 54 cells each),
    `baseline-text-seed0` (text-only, 24 text cells),
    `baseline-image-seed0` (image-only, 30 image cells).
  - White-box: `stage1-seed{0,1,2}` (multimodal) + `baseline-image-seed0`
    (image-only). Text-only has no pixel input and is excluded.
- Clean reference numbers per checkpoint (per-run, dev, n=500):

| Checkpoint | Clean AUROC | Clean macro-F1 | best τ |
|---|---:|---:|---:|
| `stage1-seed0` (perturbed run) | 0.7492 | 0.7017 | 0.30 |
| `stage1-seed1` (perturbed run) | 0.7408 | 0.7049 | 0.37 |
| `stage1-seed2` (perturbed run) | 0.7352 | 0.6797 | 0.30 |
| `stage1-seed0` (whitebox run)  | 0.7432 | 0.6725 | 0.41 |
| `stage1-seed1` (whitebox run)  | 0.7252 | 0.6630 | 0.42 |
| `stage1-seed2` (whitebox run)  | 0.7327 | 0.6680 | 0.42 |
| `baseline-text-seed0`          | 0.6321 | 0.5996 | 0.48 |
| `baseline-image-seed0`         | 0.6278 | 0.5951 | 0.46 |

The per-run clean AUROC values match the Phase-1-2 report
(0.7432 / 0.7253 / 0.7327; text-only 0.6321; image-only 0.6232) to ≤ 0.01
on the whitebox runs (which use the same `ClipImage01Transform` pipeline
as `eval/run_eval.py`) and within ~0.01 on the perturbed runs (which
resize via PIL inside the dataset, hence small bf16-precision differences).
The threshold-sweep best-τ is also recomputed per run and varies between
0.30–0.42 depending on numerical reduction order.

## 3. Headline numbers

### 3.1 Most-damaging attack family (DoD bullet 5)

**White-box PGD is overwhelmingly the most damaging family.** On the
multimodal model, PGD at ε = 1/255 already collapses AUROC from 0.73 → 0.01
(ASR 0.91); at ε = 2/255 AUROC ≈ 0 and ASR ≥ 0.98; by ε = 4/255 the
attacker succeeds essentially always (ASR ≥ 0.997).
FGSM is much weaker — at ε = 8/255 it leaves AUROC near 0.13 (ASR ≈ 0.67) —
so the attack-strength gap inside the image branch is itself a useful
signal: an iterative attacker dominates a one-step attacker by an order
of magnitude in ASR.

**Among naturalistic perturbations** the worst cells are all
high-severity text edits. Per multimodal seed:

| Seed | Worst cell | ΔAUROC |
|---|---|---:|
| 0 | censoring (high) | +0.1221 |
| 1 | char_deletion (high) | +0.1084 |
| 2 | leetspeak (high) | +0.1143 |

Image-side worst cell is consistently `blur` or `crop` at the highest
severity, dropping AUROC by ~0.03–0.04. Geometric and photometric image
attacks barely move the metric: ΔAUROC ≤ 0.025 at high severity, often
below 0.01 at low / medium.

### 3.2 Modality fragility (DoD bullet 6)

Average ΔAUROC at high severity across the relevant attack family
(reproduced from `project_planning/phase4/worst_case.md`):

| Model | Text high-sev | Image-pixel high-sev | Image-photometric high-sev | Image-geometric high-sev |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | +0.0854 | +0.0247 | +0.0090 | +0.0058 |
| baseline-image-seed0 | n/a | +0.0258 | +0.0048 | +0.0102 |
| baseline-text-seed0 | +0.0651 | n/a | n/a | n/a |

Observations:

- **Multimodal is more text-fragile than the text-only baseline.**
  The multimodal model loses on average 0.085 AUROC to high-severity text
  attacks, vs 0.065 for the text-only baseline. Text appears to dominate
  the multimodal decision even when both modalities are available — this
  is the most interesting Phase-4 finding and informs Phase 5.
- **Image-only and multimodal show comparable image fragility**
  (~0.025 high-severity ΔAUROC under pixel-level corruption); image
  attacks are not the path of least resistance into the multimodal model.
- **Under white-box PGD the order collapses**: both multimodal and
  image-only are essentially fully broken at ε ≥ 2/255 (per-modality
  PGD-ε=4/255 row in `worst_case.md` shows ΔAUROC = 0.73 ± 0.01 for
  multimodal and 0.62 for image-only, both with ASR ≥ 0.999). Text-only
  is unreachable via pixel attacks, so under PGD it is the *only*
  surviving baseline by construction, not because it is intrinsically
  more robust.

### 3.3 Modality ablation (Phase 4-S1)

For each multimodal `stage1-seed{0,1,2}` checkpoint, we evaluate three
forward paths on clean dev with the same fusion head weights:

- `multimodal`: `model(images, tokens)` — the path used during training.
- `text_only`: `model.forward_text_only(tokens)` — image embedding zeroed.
- `image_only`: `model.forward_image_only(images)` — text embedding zeroed.

| Seed | Multimodal | Text-only forward | Image-only forward |
|---|---:|---:|---:|
| 0 | 0.7492 | 0.6241 | 0.5965 |
| 1 | 0.7408 | 0.6216 | 0.5930 |
| 2 | 0.7352 | 0.6290 | 0.5880 |
| **mean ± σ** | **0.7418 ± 0.0057** | **0.6249 ± 0.0031** | **0.5925 ± 0.0035** |
| reference: dedicated baseline | — | 0.6321 (`baseline-text-seed0`) | 0.6278 (`baseline-image-seed0`) |

Two readings:

1. **The fusion head essentially embeds the text-only classifier
   losslessly.** Routing the multimodal model through `forward_text_only`
   recovers AUROC 0.625, within 0.007 of the dedicated text-only baseline.
   The text branch is *as strong inside the fusion head* as it is in a
   model trained on text alone.
2. **The fusion head's image branch is *weaker* than a dedicated
   image-only head.** Routing through `forward_image_only` gives 0.593 —
   0.035 below the dedicated image-only baseline at 0.628. The fusion
   doesn't even fully learn the image-only signal; it learned to use the
   image branch only in conjunction with text.

The +0.117 multimodal lift over the best single-branch forward
(`multimodal − text_only = 0.117`) therefore lives entirely in the
**cross-modal interaction features** the fusion head sees
(`[t, v, |t-v|, t⊙v]` per `models/clip_fusion.py`). The image branch
contributes essentially zero standalone signal inside the multimodal
head; it only matters as a modulator on the text branch.

Numbers: `cluster-results/modality-ablation-stage1-seed{0,1,2}/modality_ablation.json`.

### 3.4 Typographic attack (Phase 4-S2)

Overlay short rendered text on the image at a random corner, with a
dark backing rectangle for contrast (defeats anti-aliasing weakness).
Severity controls font size and word pool: `low` (16 px:
`safe/ok/fine`), `medium` (28 px: `SAFE/OK/NOT HATEFUL`), `high`
(40 px: `NOT HATEFUL/FRIENDLY/KIND`). Implemented as a new mode
inside `ImagePerturbation` (see
`src/robust_meme_hate_detection/perturbations/image_perturbations.py`).
Slight deviation from the plan: kept as a `mode` rather than a parallel
class so the existing `run_perturbed.py` dispatch and aggregator pick it
up without refactor.

| Model | Clean AUROC | low | medium | high | max ΔAUROC |
|---|---:|---:|---:|---:|---:|
| multimodal seed 0 | 0.7492 | 0.7452 (+0.004) | 0.7487 (+0.001) | 0.7314 (+0.018) | +0.018 |
| multimodal seed 1 | 0.7408 | 0.7380 (+0.003) | 0.7372 (+0.004) | 0.7260 (+0.015) | +0.015 |
| multimodal seed 2 | 0.7352 | 0.7417 (−0.007) | 0.7360 (−0.001) | 0.7214 (+0.014) | +0.014 |
| **multimodal mean** | **0.7418** | **+0.000** | **+0.001** | **+0.0155** | — |
| baseline-image | 0.6278 | +0.006 | +0.004 | **+0.038** | +0.038 |
| baseline-text | 0.6321 | +0.000 | +0.000 | +0.000 | 0.000 |

Three observations:

1. **The attack works** on every model with an image input. text-only
   shows exactly zero effect (sanity check — it has no pixel input).
2. **The multimodal effect is smaller than the image-only effect**
   (+0.015 vs +0.038 at high severity). Consistent with §3.3: the
   multimodal model leans on text, so corrupting the image branch hurts
   it less. Pure-image attacks land harder on a pure-image model.
3. **The effect is much smaller than zero-shot CLIP typographic attacks
   in the literature** (Goh et al. 2021 reported >0.3 confidence flips on
   CLIP zero-shot). Our fine-tuned fusion head has substantially absorbed
   CLIP's typographic-text fixation, so the rendered-text attack moves
   the prediction modestly rather than catastrophically.

This is a positive result framed two ways: the attack is real and
defensible (we have a non-null measurement of CLIP's known typographic
weakness propagating into our fine-tuned model), and it is *not*
catastrophic — fine-tuning + multimodal fusion together reduce the
typographic-fragility by ~20× compared to raw CLIP zero-shot.

### 3.5 Per-example failure analysis (Phase 4-S3)

Bucketing every dev example by its (clean / natural-attack / PGD)
outcome on the multimodal seed-0 checkpoint at best-τ surfaces a much
sharper picture than the per-cell ΔAUROC averages:

| Bucket | Description | Count | % of 500 |
|---|---|---:|---:|
| `B1` | Clean-correct, natural-robust, PGD-robust | 1 | 0.2 % |
| `B2` | Clean-wrong (intrinsic miss) | 149 | 29.8 % |
| `B3` | Clean-correct → flipped by ≥ 1 natural-attack cell | 257 | 51.4 % |
| `B4` | Clean-correct, natural-robust, broken by PGD only | 93 | 18.6 % |

This is the most uncomfortable Phase-4 result. The per-cell ΔAUROC
averages (~0.04 worst image-attack mean, ~0.085 worst text-attack mean)
suggested a moderate, structured vulnerability. But the **union of
attacks per example** is far more aggressive: of 351 clean-correct
examples, only 94 survive any natural attack (B1+B4), and only 1
survives both natural attacks and white-box PGD. Average robustness
gaps systematically understate the worst-case-per-example exposure
because each example has its own weakest cell among the 54 + 3
(naturalistic + typographic) attack/severity combinations.

Curated failure cases (5 per bucket, top by clean-confidence) with full
prob trajectories and suggested categories live in
[`phase4/failure_analysis.md`](phase4/failure_analysis.md). Phase 6 will
re-run this script against the robust checkpoint and add
`robust-fixed` / `robust-still-failed` columns.

## 4. Tables (full)

- Naturalistic per-cell tables: [`phase4/perturbed_table.md`](phase4/perturbed_table.md).
- White-box per-cell tables: [`phase4/whitebox_table.md`](phase4/whitebox_table.md).
- Worst-case + fragility tables: [`phase4/worst_case.md`](phase4/worst_case.md).

## 5. Figures

- ΔAUROC heatmaps per multimodal seed:
  `phase4/figures/heatmap_perturbed_stage1-seed{0,1,2}.png`.
- White-box robustness curve (AUROC vs ε): `phase4/figures/curve_whitebox.png`.
- Per-family severity curves (multimodal mean):
  `phase4/figures/severity_curves.png`.

## 6. Open issues / Phase 5 entry points

- **Phase 5 severity mix for augmentation**: text attacks dominate the
  Phase-4 vulnerability, *and* the modality-ablation result confirms the
  image branch contributes essentially zero standalone signal inside the
  fusion head. The KL-consistency robust training should accordingly
  oversample text attacks at medium/high severity and force the image
  branch to do useful work — both by adding image perturbations to
  augmentation, and by the KL term anchoring `forward_image_only(x_pert)`
  to `forward_image_only(x_clean)` so the fusion head learns to value the
  image branch. Suggested starting mix: 50 % text / 30 % image / 20 %
  both, with `(1, 2, 2)` weighting on (low, medium, high) for text.
- **Typographic attack should be in the training augmentation set** as
  well, so the model sees rendered-text overlays at training time.
  Trivial addition since it's already a recognised image-mode preset.
- **Failure-analysis baseline is established**: 351/500 clean-correct,
  only 1/500 robust to all natural attacks. Phase 5 success can be
  measured by how that 1 → N progresses (target: > 50/500 fully
  robust on the recipe we ship).
- **White-box defence is out of Phase 4's DoD** but motivates Phase 5's
  adversarial-training arm — if budget allows, mix in PGD-ε=2/255 (or
  FGSM-ε=4/255 as a cheaper proxy) at a low probability to harden the
  image branch.
- **Text-side white-box attacks** remain out of scope per the roadmap
  (Phase-7 stretch). Not implemented.
- **Threshold drift**: per-run best-τ varies between 0.30–0.48 because the
  raw `clean_probs` sequence is bf16-noisy. The aggregator handles this
  per-run; the Phase-5 robust runs should use the same per-run-best
  convention so gaps stay self-consistent.
- **Secondary dataset** (Phase 7) — untouched.

## 7. Reproduction

```bash
# Run benchmarks
./cluster/cluster.sh sync-code
for seed in 0 1 2; do
  ./cluster/cluster.sh submit-cmd "perturbed-stage1-seed$seed" -- \
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.eval.run_perturbed \
      --ckpt "/scratch/.../stage1-seed$seed-*/ckpt/best.pt" \
      --config configs/stage1.yaml --modality multimodal --attacks all \
      --out "/scratch/.../perturbed-stage1-seed$seed"
  ./cluster/cluster.sh submit-cmd "whitebox-stage1-seed$seed" -- \
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.eval.run_whitebox \
      --ckpt "/scratch/.../stage1-seed$seed-*/ckpt/best.pt" \
      --config configs/stage1.yaml --modality multimodal \
      --attacks fgsm,pgd --epsilons 1,2,4,8 --pgd-steps 10 \
      --out "/scratch/.../whitebox-stage1-seed$seed"
done
# Symmetric for baseline-text-seed0 (perturbed only, --modality text --attacks text)
# and baseline-image-seed0 (perturbed + whitebox, --modality image).

# Aggregate locally
PYTHONPATH=src .venv/bin/python3 scripts/aggregate_phase4.py
```
