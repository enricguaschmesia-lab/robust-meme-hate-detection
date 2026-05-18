# Phase 5 — Robust vs clean training comparison

All multimodal numbers aggregate the 3 seeds (mean ± σ). Each row is one training recipe.

## Clean accuracy & naturalistic worst-case

| Recipe | n seeds | Clean AUROC | Clean F1 | Worst-cell text ΔAUROC | Worst-cell image ΔAUROC |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 0.7471 ± 0.0056 | 0.7012 ± 0.0065 | 0.1288 ± 0.0018 | 0.0317 ± 0.0060 |
| augonly | 3 | 0.7158 ± 0.0310 | 0.6737 ± 0.0263 | 0.0832 ± 0.0219 | 0.0159 ± 0.0039 |
| kl | 3 | 0.7454 ± 0.0081 | 0.6927 ± 0.0046 | 0.0855 ± 0.0014 | 0.0194 ± 0.0019 |
| kldrop | 3 | 0.7211 ± 0.0312 | 0.6726 ± 0.0257 | 0.0736 ± 0.0162 | 0.0250 ± 0.0081 |
| kllowmed | 3 | 0.7444 ± 0.0054 | 0.6943 ± 0.0078 | 0.0885 ± 0.0062 | 0.0244 ± 0.0018 |

## Mean ΔAUROC at high severity, per family

| Recipe | text | image-pixel | image-photometric | image-geometric | typographic |
|---|---:|---:|---:|---:|---:|
| clean | 0.0887 ± 0.0016 | 0.0210 ± 0.0021 | 0.0098 ± 0.0007 | 0.0072 ± 0.0032 | 0.0113 ± 0.0039 |
| augonly | 0.0576 ± 0.0169 | 0.0116 ± 0.0041 | 0.0050 ± 0.0034 | 0.0010 ± 0.0038 | 0.0103 ± 0.0054 |
| kl | 0.0614 ± 0.0006 | 0.0151 ± 0.0019 | 0.0082 ± 0.0012 | 0.0040 ± 0.0006 | 0.0132 ± 0.0030 |
| kldrop | 0.0497 ± 0.0137 | 0.0159 ± 0.0051 | 0.0063 ± 0.0018 | 0.0060 ± 0.0043 | 0.0109 ± 0.0062 |
| kllowmed | 0.0639 ± 0.0045 | 0.0185 ± 0.0007 | 0.0092 ± 0.0010 | 0.0049 ± 0.0020 | 0.0127 ± 0.0028 |

## White-box PGD AUROC vs ε

| Recipe | clean AUROC | PGD ε=1/255 | PGD ε=2/255 | PGD ε=4/255 | PGD ε=8/255 |
|---|---:|---:|---:|---:|---:|
| clean | 0.7488 ± 0.0042 | 0.0126 ± 0.0007 | 0.0003 ± 0.0000 | 0.0000 | 0.0000 |
| augonly | 0.7140 ± 0.0291 | 0.0168 ± 0.0019 | 0.0006 ± 0.0002 | 0.0000 | 0.0000 |
| kl | 0.7458 ± 0.0102 | 0.0200 ± 0.0045 | 0.0009 ± 0.0005 | 0.0000 ± 0.0000 | 0.0000 |
| kldrop | 0.7161 ± 0.0294 | 0.0149 ± 0.0014 | 0.0004 ± 0.0003 | 0.0000 | 0.0000 |
| kllowmed | 0.7461 ± 0.0059 | 0.0213 ± 0.0009 | 0.0010 ± 0.0003 | 0.0000 | 0.0000 |

## Modality ablation (clean dev forwards)

Direct test of the "dead image branch" failure mode: did `KL_image_branch` lift `image_only` AUROC?

| Recipe | multimodal AUROC | text_only AUROC | image_only AUROC |
|---|---:|---:|---:|
| clean | 0.7471 ± 0.0056 | 0.6440 ± 0.0055 | 0.5822 ± 0.0029 |
| augonly | 0.7158 ± 0.0310 | 0.6344 ± 0.0081 | 0.6059 ± 0.0110 |
| kl | 0.7454 ± 0.0081 | 0.6429 ± 0.0016 | 0.6045 ± 0.0074 |
| kldrop | 0.7211 ± 0.0312 | 0.6438 ± 0.0089 | 0.6411 ± 0.0047 |
| kllowmed | 0.7444 ± 0.0054 | 0.6427 ± 0.0018 | 0.5968 ± 0.0010 |

## Per-example fully-robust counts (out of 500 dev)

Two threat models, reported separately:

* **Naturalistic-only**: example is clean-correct AND survives every naturalistic perturbation cell (8 text × 3 + 11 image × 3 = 57 cells).
* **Natural + white-box PGD**: same, plus survives PGD ε=4/255. PGD has near-100 % ASR by design (worst-case oracle) so this number is dominated by it; reported only for completeness.

| Recipe | Naturalistic seed0 | seed1 | seed2 | mean | Nat + PGD seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 174 | 140 | 147 | 153.67 | 0 | 2 | 0 | 0.67 |
| augonly | 171 | 176 | 147 | 164.67 | 1 | 3 | 1 | 1.67 |
| kl | 193 | 207 | 189 | 196.33 | 2 | 4 | 0 | 2.00 |
| kldrop | 179 | 179 | 190 | 182.67 | 3 | 0 | 2 | 1.67 |
| kllowmed | 185 | 183 | 180 | 182.67 | 2 | 2 | 4 | 2.67 |


## Attack-type generalisation: in-pool vs held-out (OOD)

Training pool (text, n=5): `censoring, char_deletion, char_swap, keyboard_typo, leetspeak`  
Held-out text (n=3): `case_noise, punctuation, spacing`  
Training pool (image, n=6): `blur, brightness_down, compression, gaussian_noise, occlusion, typographic`  
Held-out image (n=5): `brightness_up, contrast_down, contrast_up, crop, translation`

Mean ΔAUROC (clean − attacked) across all severities of the cells in each partition. Lower is better. *Caveat*: held-out text attacks (`spacing/punctuation/case_noise`) are the weakest text attacks on the clean ckpt (ΔAUROC ≤ 0.02), so the text-side OOD column has limited signal. The image-side OOD column is the meaningful generalisation test.

| Recipe | text in-pool | text OOD | Δ(OOD-in) text | image in-pool | image OOD | Δ(OOD-in) image |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0686 ± 0.0013 | 0.0431 ± 0.0009 | -0.0255 | 0.0089 ± 0.0001 | 0.0054 ± 0.0010 | -0.0035 |
| augonly | 0.0467 ± 0.0119 | 0.0267 ± 0.0091 | -0.0200 | 0.0045 ± 0.0028 | 0.0016 ± 0.0016 | -0.0029 |
| kl | 0.0493 ± 0.0008 | 0.0302 ± 0.0010 | -0.0192 | 0.0064 ± 0.0012 | 0.0032 ± 0.0005 | -0.0032 |
| kldrop | 0.0409 ± 0.0105 | 0.0233 ± 0.0066 | -0.0175 | 0.0063 ± 0.0027 | 0.0041 ± 0.0023 | -0.0022 |
| kllowmed | 0.0507 ± 0.0036 | 0.0307 ± 0.0021 | -0.0200 | 0.0078 ± 0.0007 | 0.0040 ± 0.0012 | -0.0037 |

Reading: Δ(OOD−in) > 0 means the recipe defends in-pool attacks more than held-out ones — i.e. augmentation has *memorised* its training pool rather than learning a transferable defence. Δ near 0 (or negative) is the generalisation signal we want.


## Per-severity sensitivity (single-perturbation cells)

Mean ΔAUROC at each severity, per family. **Medium severity is the internet-realistic threat number** (the level a typical adversarial user would reach with an off-the-shelf editor); high severity is a stress-test ceiling that often borders on label-preserving limits; low is included for monotonicity sanity-checking.

### text

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0308 ± 0.0003 | **0.0577 ± 0.0017** | 0.0887 ± 0.0016 |
| augonly | 0.0200 ± 0.0064 | **0.0399 ± 0.0093** | 0.0576 ± 0.0169 |
| kl | 0.0226 ± 0.0012 | **0.0424 ± 0.0007** | 0.0614 ± 0.0006 |
| kldrop | 0.0181 ± 0.0052 | **0.0351 ± 0.0084** | 0.0497 ± 0.0137 |
| kllowmed | 0.0225 ± 0.0018 | **0.0431 ± 0.0030** | 0.0639 ± 0.0045 |

### image-pixel

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0079 ± 0.0011 | **0.0117 ± 0.0017** | 0.0210 ± 0.0021 |
| augonly | 0.0036 ± 0.0022 | **0.0058 ± 0.0025** | 0.0116 ± 0.0041 |
| kl | 0.0042 ± 0.0010 | **0.0069 ± 0.0014** | 0.0151 ± 0.0019 |
| kldrop | 0.0041 ± 0.0027 | **0.0080 ± 0.0035** | 0.0159 ± 0.0051 |
| kllowmed | 0.0055 ± 0.0012 | **0.0098 ± 0.0004** | 0.0185 ± 0.0007 |

### image-photometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0017 ± 0.0001 | **0.0045 ± 0.0005** | 0.0098 ± 0.0007 |
| augonly | 0.0002 ± 0.0008 | **0.0011 ± 0.0019** | 0.0050 ± 0.0034 |
| kl | 0.0010 ± 0.0008 | **0.0031 ± 0.0011** | 0.0082 ± 0.0012 |
| kldrop | 0.0007 ± 0.0007 | **0.0024 ± 0.0017** | 0.0063 ± 0.0018 |
| kllowmed | 0.0010 ± 0.0008 | **0.0035 ± 0.0012** | 0.0092 ± 0.0010 |

### image-geometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0005 ± 0.0011 | **0.0057 ± 0.0006** | 0.0072 ± 0.0032 |
| augonly | -0.0020 ± 0.0003 | **0.0036 ± 0.0012** | 0.0010 ± 0.0038 |
| kl | -0.0019 ± 0.0012 | **0.0046 ± 0.0016** | 0.0040 ± 0.0006 |
| kldrop | -0.0001 ± 0.0015 | **0.0067 ± 0.0022** | 0.0060 ± 0.0043 |
| kllowmed | -0.0005 ± 0.0007 | **0.0056 ± 0.0010** | 0.0049 ± 0.0020 |

### typographic

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0015 ± 0.0010 | **0.0033 ± 0.0043** | 0.0113 ± 0.0039 |
| augonly | -0.0008 ± 0.0010 | **0.0009 ± 0.0024** | 0.0103 ± 0.0054 |
| kl | -0.0013 ± 0.0004 | **0.0035 ± 0.0026** | 0.0132 ± 0.0030 |
| kldrop | -0.0002 ± 0.0016 | **0.0040 ± 0.0013** | 0.0109 ± 0.0062 |
| kllowmed | -0.0007 ± 0.0012 | **0.0016 ± 0.0031** | 0.0127 ± 0.0028 |


## Composite attacks (multiple perturbations per sample)

Each composite cell applies *K* random perturbations to every sample, drawn deterministically per `(cell_seed, sample_id)` from the full eval pool (not the training pool — composite is itself an OOD test). All component perturbations use the cell-level severity.

| Composite | K_text | K_image |
|---|---:|---:|
| `composite_2text` | 2 | 0 |
| `composite_2image` | 0 | 2 |
| `composite_text_image` | 1 | 1 |
| `composite_2text_2image` | 2 | 2 |

### composite_2text

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | high ASR |
|---|---:|---:|---:|---:|
| clean | 0.0616 ± 0.0017 | **0.0870 ± 0.0012** | 0.1202 ± 0.0043 | 0.3383 ± 0.0127 |
| augonly | 0.0378 ± 0.0098 | **0.0582 ± 0.0148** | 0.0747 ± 0.0174 | 0.2333 ± 0.0173 |
| kl | 0.0432 ± 0.0016 | **0.0635 ± 0.0038** | 0.0804 ± 0.0015 | 0.2322 ± 0.0019 |
| kldrop | 0.0345 ± 0.0106 | **0.0498 ± 0.0136** | 0.0640 ± 0.0168 | 0.2053 ± 0.0145 |
| kllowmed | 0.0423 ± 0.0050 | **0.0637 ± 0.0044** | 0.0857 ± 0.0036 | 0.2537 ± 0.0072 |

### composite_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | high ASR |
|---|---:|---:|---:|---:|
| clean | 0.0024 ± 0.0028 | **0.0043 ± 0.0036** | 0.0344 ± 0.0027 | 0.1711 ± 0.0065 |
| augonly | 0.0018 ± 0.0023 | **0.0037 ± 0.0023** | 0.0206 ± 0.0110 | 0.1659 ± 0.0210 |
| kl | 0.0025 ± 0.0013 | **0.0021 ± 0.0043** | 0.0271 ± 0.0018 | 0.1533 ± 0.0154 |
| kldrop | 0.0023 ± 0.0003 | **0.0037 ± 0.0043** | 0.0254 ± 0.0083 | 0.1720 ± 0.0175 |
| kllowmed | 0.0032 ± 0.0015 | **0.0041 ± 0.0029** | 0.0307 ± 0.0027 | 0.1607 ± 0.0141 |

### composite_text_image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | high ASR |
|---|---:|---:|---:|---:|
| clean | 0.0356 ± 0.0027 | **0.0575 ± 0.0027** | 0.1035 ± 0.0086 | 0.3090 ± 0.0073 |
| augonly | 0.0274 ± 0.0013 | **0.0357 ± 0.0149** | 0.0708 ± 0.0205 | 0.2380 ± 0.0125 |
| kl | 0.0294 ± 0.0006 | **0.0416 ± 0.0063** | 0.0725 ± 0.0026 | 0.2312 ± 0.0067 |
| kldrop | 0.0260 ± 0.0023 | **0.0352 ± 0.0104** | 0.0626 ± 0.0173 | 0.2242 ± 0.0128 |
| kllowmed | 0.0310 ± 0.0017 | **0.0418 ± 0.0059** | 0.0768 ± 0.0084 | 0.2454 ± 0.0164 |

### composite_2text_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | high ASR |
|---|---:|---:|---:|---:|
| clean | 0.0627 ± 0.0006 | **0.1019 ± 0.0047** | 0.1394 ± 0.0041 | 0.3805 ± 0.0274 |
| augonly | 0.0391 ± 0.0153 | **0.0629 ± 0.0227** | 0.0946 ± 0.0290 | 0.3195 ± 0.0138 |
| kl | 0.0428 ± 0.0022 | **0.0716 ± 0.0048** | 0.1100 ± 0.0027 | 0.3106 ± 0.0158 |
| kldrop | 0.0366 ± 0.0118 | **0.0608 ± 0.0175** | 0.0982 ± 0.0225 | 0.2994 ± 0.0112 |
| kllowmed | 0.0441 ± 0.0053 | **0.0763 ± 0.0058** | 0.1140 ± 0.0012 | 0.3204 ± 0.0050 |

