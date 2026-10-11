# Phase 5 — Robust vs clean training comparison

All multimodal numbers aggregate the 3 seeds (mean ± σ). Each row is one training recipe.

## Clean accuracy & naturalistic worst-case

| Recipe | n seeds | Clean AUROC | Clean F1 | Worst-cell text ΔAUROC | Worst-cell image ΔAUROC |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 0.7381 ± 0.0030 | 0.6757 ± 0.0047 | 0.1102 ± 0.0065 | 0.0342 ± 0.0028 |
| augonly | 3 | 0.7082 ± 0.0282 | 0.6517 ± 0.0212 | 0.0630 ± 0.0168 | 0.0212 ± 0.0026 |
| kl | 3 | 0.7384 ± 0.0104 | 0.6760 ± 0.0098 | 0.0653 ± 0.0037 | 0.0259 ± 0.0006 |
| kldrop-p015 | 3 | 0.7432 ± 0.0092 | 0.6817 ± 0.0099 | 0.0692 ± 0.0097 | 0.0326 ± 0.0003 |
| kldrop-p050 | 3 | 0.7118 ± 0.0338 | 0.6561 ± 0.0299 | 0.0486 ± 0.0168 | 0.0369 ± 0.0046 |
| kllowmed | 3 | 0.7355 ± 0.0048 | 0.6747 ± 0.0049 | 0.0678 ± 0.0041 | 0.0307 ± 0.0008 |

## Mean ΔAUROC at high severity, per family

| Recipe | text | image-pixel | image-photometric | image-geometric | typographic |
|---|---:|---:|---:|---:|---:|
| clean | 0.0734 ± 0.0036 | 0.0236 ± 0.0017 | 0.0110 ± 0.0022 | 0.0074 ± 0.0025 | 0.0104 ± 0.0014 |
| augonly | 0.0424 ± 0.0132 | 0.0151 ± 0.0041 | 0.0071 ± 0.0029 | 0.0073 ± 0.0002 | 0.0092 ± 0.0013 |
| kl | 0.0474 ± 0.0032 | 0.0179 ± 0.0006 | 0.0091 ± 0.0008 | 0.0060 ± 0.0016 | 0.0111 ± 0.0017 |
| kldrop-p015 | 0.0486 ± 0.0043 | 0.0208 ± 0.0013 | 0.0097 ± 0.0009 | 0.0086 ± 0.0019 | 0.0121 ± 0.0039 |
| kldrop-p050 | 0.0317 ± 0.0125 | 0.0221 ± 0.0052 | 0.0084 ± 0.0022 | 0.0115 ± 0.0004 | 0.0133 ± 0.0029 |
| kllowmed | 0.0499 ± 0.0033 | 0.0208 ± 0.0007 | 0.0108 ± 0.0015 | 0.0064 ± 0.0004 | 0.0110 ± 0.0007 |

## White-box PGD AUROC vs ε

| Recipe | clean AUROC | PGD ε=1/255 | PGD ε=2/255 | PGD ε=4/255 | PGD ε=8/255 |
|---|---:|---:|---:|---:|---:|
| clean | 0.7309 ± 0.0009 | 0.0150 ± 0.0006 | 0.0005 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 |
| augonly | 0.7014 ± 0.0271 | 0.0201 ± 0.0015 | 0.0009 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| kl | 0.7337 ± 0.0140 | 0.0221 ± 0.0039 | 0.0011 ± 0.0004 | 0.0000 ± 0.0000 | 0.0000 |
| kldrop-p015 | 0.7354 ± 0.0121 | 0.0163 ± 0.0017 | 0.0005 ± 0.0001 | 0.0000 | 0.0000 |
| kldrop-p050 | 0.7012 ± 0.0307 | 0.0142 ± 0.0022 | 0.0005 ± 0.0003 | 0.0000 ± 0.0000 | 0.0000 |
| kllowmed | 0.7299 ± 0.0069 | 0.0236 ± 0.0007 | 0.0012 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 |

## Modality ablation (clean dev forwards)

Direct test of the "dead image branch" failure mode: did `KL_image_branch` lift `image_only` AUROC?

| Recipe | multimodal AUROC | text_only AUROC | image_only AUROC |
|---|---:|---:|---:|
| clean | 0.7381 ± 0.0030 | 0.6118 ± 0.0010 | 0.6020 ± 0.0029 |
| augonly | 0.7082 ± 0.0282 | 0.6056 ± 0.0071 | 0.6200 ± 0.0031 |
| kl | 0.7384 ± 0.0104 | 0.6127 ± 0.0025 | 0.6276 ± 0.0078 |
| kldrop-p015 | 0.7432 ± 0.0092 | 0.6174 ± 0.0027 | 0.6580 ± 0.0086 |
| kldrop-p050 | 0.7118 ± 0.0338 | 0.6093 ± 0.0086 | 0.6657 ± 0.0166 |
| kllowmed | 0.7355 ± 0.0048 | 0.6129 ± 0.0026 | 0.6171 ± 0.0064 |

## Per-example fully-robust counts (out of 500 dev)

Two threat models, reported separately:

* **Naturalistic-only**: example is clean-correct AND survives every naturalistic perturbation cell (8 text × 3 + 11 image × 3 = 57 cells).
* **Natural + white-box PGD**: same, plus survives PGD ε=4/255. PGD has near-100 % ASR by design (worst-case oracle) so this number is dominated by it; reported only for completeness.

| Recipe | Naturalistic seed0 | seed1 | seed2 | mean | Nat + PGD seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 217 | 205 | 244 | 222.00 | 0 | 3 | 0 | 1.00 |
| augonly | 313 | 287 | 246 | 282.00 | 1 | 3 | 2 | 2.00 |
| kl | 351 | 357 | 348 | 352.00 | 2 | 3 | 1 | 2.00 |
| kldrop-p015 | 424 | 322 | 428 | 391.33 | 0 | 0 | 0 | 0.00 |
| kldrop-p050 | 333 | 308 | 351 | 330.67 | 0 | 0 | 5 | 1.67 |
| kllowmed | 379 | 363 | 343 | 361.67 | 3 | 1 | 2 | 2.00 |


## Attack-type generalisation: in-pool vs held-out (OOD)

Training pool (text, n=5): `censoring, char_deletion, char_swap, keyboard_typo, leetspeak`  
Held-out text (n=3): `case_noise, punctuation, spacing`  
Training pool (image, n=6): `blur, brightness_down, compression, gaussian_noise, occlusion, typographic`  
Held-out image (n=5): `brightness_up, contrast_down, contrast_up, crop, translation`

Mean ΔAUROC (clean − attacked) across all severities of the cells in each partition. Lower is better. *Caveat*: held-out text attacks (`spacing/punctuation/case_noise`) are the weakest text attacks on the clean ckpt (ΔAUROC ≤ 0.02), so the text-side OOD column has limited signal. The image-side OOD column is the meaningful generalisation test.

| Recipe | text in-pool | text OOD | Δ(OOD-in) text | image in-pool | image OOD | Δ(OOD-in) image |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0555 ± 0.0022 | 0.0315 ± 0.0009 | -0.0240 | 0.0087 ± 0.0008 | 0.0063 ± 0.0012 | -0.0024 |
| augonly | 0.0328 ± 0.0101 | 0.0191 ± 0.0068 | -0.0136 | 0.0055 ± 0.0022 | 0.0044 ± 0.0007 | -0.0011 |
| kl | 0.0366 ± 0.0028 | 0.0213 ± 0.0016 | -0.0152 | 0.0065 ± 0.0002 | 0.0044 ± 0.0006 | -0.0020 |
| kldrop-p015 | 0.0377 ± 0.0036 | 0.0219 ± 0.0014 | -0.0158 | 0.0078 ± 0.0005 | 0.0064 ± 0.0011 | -0.0013 |
| kldrop-p050 | 0.0251 ± 0.0098 | 0.0139 ± 0.0058 | -0.0112 | 0.0078 ± 0.0024 | 0.0063 ± 0.0012 | -0.0015 |
| kllowmed | 0.0377 ± 0.0031 | 0.0223 ± 0.0019 | -0.0154 | 0.0077 ± 0.0004 | 0.0053 ± 0.0006 | -0.0024 |

Reading: Δ(OOD−in) > 0 means the recipe defends in-pool attacks more than held-out ones — i.e. augmentation has *memorised* its training pool rather than learning a transferable defence. Δ near 0 (or negative) is the generalisation signal we want.


## Per-severity sensitivity (single-perturbation cells)

Mean ΔAUROC at each severity, per family. **Medium severity is the internet-realistic threat number** (the level a typical adversarial user would reach with an off-the-shelf editor); high severity is a stress-test ceiling that often borders on label-preserving limits; low is included for monotonicity sanity-checking.

### text

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0245 ± 0.0000 | **0.0416 ± 0.0015** | 0.0734 ± 0.0036 |
| augonly | 0.0156 ± 0.0041 | **0.0249 ± 0.0093** | 0.0424 ± 0.0132 |
| kl | 0.0172 ± 0.0016 | **0.0280 ± 0.0023** | 0.0474 ± 0.0032 |
| kldrop-p015 | 0.0181 ± 0.0014 | **0.0287 ± 0.0027** | 0.0486 ± 0.0043 |
| kldrop-p050 | 0.0122 ± 0.0041 | **0.0188 ± 0.0083** | 0.0317 ± 0.0125 |
| kllowmed | 0.0175 ± 0.0019 | **0.0285 ± 0.0027** | 0.0499 ± 0.0033 |

### image-pixel

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0046 ± 0.0007 | **0.0117 ± 0.0005** | 0.0236 ± 0.0017 |
| augonly | 0.0013 ± 0.0017 | **0.0060 ± 0.0034** | 0.0151 ± 0.0041 |
| kl | 0.0017 ± 0.0009 | **0.0069 ± 0.0008** | 0.0179 ± 0.0006 |
| kldrop-p015 | 0.0036 ± 0.0008 | **0.0089 ± 0.0014** | 0.0208 ± 0.0013 |
| kldrop-p050 | 0.0028 ± 0.0021 | **0.0094 ± 0.0039** | 0.0221 ± 0.0052 |
| kllowmed | 0.0025 ± 0.0010 | **0.0092 ± 0.0005** | 0.0208 ± 0.0007 |

### image-photometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0014 ± 0.0002 | **0.0057 ± 0.0012** | 0.0110 ± 0.0022 |
| augonly | 0.0005 ± 0.0003 | **0.0031 ± 0.0015** | 0.0071 ± 0.0029 |
| kl | 0.0007 ± 0.0002 | **0.0040 ± 0.0001** | 0.0091 ± 0.0008 |
| kldrop-p015 | 0.0013 ± 0.0002 | **0.0053 ± 0.0007** | 0.0097 ± 0.0009 |
| kldrop-p050 | 0.0006 ± 0.0005 | **0.0041 ± 0.0013** | 0.0084 ± 0.0022 |
| kllowmed | 0.0009 ± 0.0003 | **0.0051 ± 0.0006** | 0.0108 ± 0.0015 |

### image-geometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0011 ± 0.0008 | **0.0076 ± 0.0013** | 0.0074 ± 0.0025 |
| augonly | -0.0000 ± 0.0008 | **0.0078 ± 0.0013** | 0.0073 ± 0.0002 |
| kl | 0.0002 ± 0.0003 | **0.0071 ± 0.0022** | 0.0060 ± 0.0016 |
| kldrop-p015 | 0.0025 ± 0.0011 | **0.0090 ± 0.0005** | 0.0086 ± 0.0019 |
| kldrop-p050 | 0.0020 ± 0.0014 | **0.0095 ± 0.0011** | 0.0115 ± 0.0004 |
| kllowmed | 0.0006 ± 0.0006 | **0.0077 ± 0.0009** | 0.0064 ± 0.0004 |

### typographic

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | -0.0017 ± 0.0005 | **0.0028 ± 0.0018** | 0.0104 ± 0.0014 |
| augonly | -0.0020 ± 0.0015 | **0.0013 ± 0.0013** | 0.0092 ± 0.0013 |
| kl | -0.0029 ± 0.0004 | **0.0001 ± 0.0019** | 0.0111 ± 0.0017 |
| kldrop-p015 | -0.0009 ± 0.0010 | **0.0004 ± 0.0018** | 0.0121 ± 0.0039 |
| kldrop-p050 | -0.0025 ± 0.0009 | **-0.0002 ± 0.0006** | 0.0133 ± 0.0029 |
| kllowmed | -0.0019 ± 0.0005 | **-0.0001 ± 0.0013** | 0.0110 ± 0.0007 |


## Composite attacks (multiple perturbations per sample)

Each composite cell applies *K* random perturbations to every sample, drawn deterministically per `(cell_seed, sample_id)` from the full eval pool (not the training pool — composite is itself an OOD test). All component perturbations use the cell-level severity.

| Composite | K_text | K_image |
|---|---:|---:|
| `composite_2text` | 2 | 0 |
| `composite_2image` | 0 | 2 |
| `composite_text_image` | 1 | 1 |
| `composite_2text_2image` | 2 | 2 |

### composite_2text

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0474 ± 0.0012 | 0.0731 ± 0.0042 | 0.1112 ± 0.0054 | **0.0783 ± 0.0037** | 0.4036 ± 0.0069 | 0.3185 ± 0.0074 |
| augonly | 0.0334 ± 0.0092 | 0.0431 ± 0.0152 | 0.0645 ± 0.0181 | **0.0444 ± 0.0123** | 0.2367 ± 0.0267 | 0.1932 ± 0.0126 |
| kl | 0.0363 ± 0.0018 | 0.0515 ± 0.0046 | 0.0690 ± 0.0028 | **0.0487 ± 0.0040** | 0.2258 ± 0.0028 | 0.1923 ± 0.0025 |
| kldrop-p015 | 0.0336 ± 0.0028 | 0.0535 ± 0.0039 | 0.0741 ± 0.0098 | **0.0497 ± 0.0072** | 0.2124 ± 0.0367 | 0.1736 ± 0.0314 |
| kldrop-p050 | 0.0252 ± 0.0026 | 0.0353 ± 0.0170 | 0.0484 ± 0.0177 | **0.0350 ± 0.0125** | 0.2022 ± 0.0254 | 0.1675 ± 0.0217 |
| kllowmed | 0.0362 ± 0.0017 | 0.0531 ± 0.0049 | 0.0736 ± 0.0062 | **0.0496 ± 0.0040** | 0.2261 ± 0.0060 | 0.1822 ± 0.0081 |

### composite_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0024 ± 0.0010 | 0.0169 ± 0.0030 | 0.0252 ± 0.0033 | **0.0236 ± 0.0007** | 0.1656 ± 0.0201 | 0.1240 ± 0.0116 |
| augonly | 0.0014 ± 0.0018 | 0.0107 ± 0.0011 | 0.0192 ± 0.0047 | **0.0150 ± 0.0055** | 0.1890 ± 0.0433 | 0.1280 ± 0.0329 |
| kl | 0.0017 ± 0.0017 | 0.0117 ± 0.0004 | 0.0225 ± 0.0025 | **0.0195 ± 0.0022** | 0.1633 ± 0.0222 | 0.1119 ± 0.0101 |
| kldrop-p015 | 0.0024 ± 0.0002 | 0.0159 ± 0.0018 | 0.0271 ± 0.0011 | **0.0228 ± 0.0012** | 0.1745 ± 0.0205 | 0.1180 ± 0.0075 |
| kldrop-p050 | 0.0012 ± 0.0019 | 0.0137 ± 0.0046 | 0.0277 ± 0.0057 | **0.0200 ± 0.0090** | 0.1928 ± 0.0140 | 0.1320 ± 0.0007 |
| kllowmed | 0.0023 ± 0.0014 | 0.0142 ± 0.0005 | 0.0245 ± 0.0034 | **0.0208 ± 0.0032** | 0.1554 ± 0.0177 | 0.1005 ± 0.0092 |

### composite_text_image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0293 ± 0.0021 | 0.0503 ± 0.0035 | 0.1054 ± 0.0050 | **0.0579 ± 0.0045** | 0.3623 ± 0.0096 | 0.2535 ± 0.0121 |
| augonly | 0.0158 ± 0.0060 | 0.0295 ± 0.0132 | 0.0676 ± 0.0120 | **0.0381 ± 0.0061** | 0.2692 ± 0.0083 | 0.1863 ± 0.0102 |
| kl | 0.0178 ± 0.0021 | 0.0356 ± 0.0017 | 0.0706 ± 0.0024 | **0.0379 ± 0.0031** | 0.2486 ± 0.0078 | 0.1744 ± 0.0101 |
| kldrop-p015 | 0.0214 ± 0.0025 | 0.0365 ± 0.0040 | 0.0693 ± 0.0041 | **0.0430 ± 0.0036** | 0.2296 ± 0.0387 | 0.1651 ± 0.0270 |
| kldrop-p050 | 0.0137 ± 0.0058 | 0.0254 ± 0.0134 | 0.0569 ± 0.0125 | **0.0345 ± 0.0089** | 0.2459 ± 0.0316 | 0.1703 ± 0.0190 |
| kllowmed | 0.0187 ± 0.0032 | 0.0376 ± 0.0007 | 0.0740 ± 0.0020 | **0.0387 ± 0.0033** | 0.2340 ± 0.0003 | 0.1686 ± 0.0068 |

### composite_2text_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0512 ± 0.0034 | 0.0946 ± 0.0085 | 0.1399 ± 0.0101 | **0.0958 ± 0.0042** | 0.4689 ± 0.0212 | 0.3705 ± 0.0170 |
| augonly | 0.0309 ± 0.0092 | 0.0600 ± 0.0164 | 0.1023 ± 0.0176 | **0.0617 ± 0.0151** | 0.3818 ± 0.0208 | 0.2737 ± 0.0190 |
| kl | 0.0330 ± 0.0046 | 0.0650 ± 0.0025 | 0.1056 ± 0.0010 | **0.0670 ± 0.0062** | 0.3391 ± 0.0141 | 0.2550 ± 0.0136 |
| kldrop-p015 | 0.0353 ± 0.0023 | 0.0671 ± 0.0057 | 0.1007 ± 0.0094 | **0.0720 ± 0.0058** | 0.3078 ± 0.0428 | 0.2333 ± 0.0412 |
| kldrop-p050 | 0.0237 ± 0.0121 | 0.0581 ± 0.0142 | 0.0851 ± 0.0170 | **0.0558 ± 0.0223** | 0.3408 ± 0.0300 | 0.2610 ± 0.0296 |
| kllowmed | 0.0344 ± 0.0035 | 0.0683 ± 0.0013 | 0.1152 ± 0.0034 | **0.0712 ± 0.0057** | 0.3331 ± 0.0083 | 0.2459 ± 0.0079 |


## Deployment-honest threshold: dev-tuned τ applied to test

Per-split-tuned `τ` (the default elsewhere in this file) uses *test* labels to pick the threshold — mild data leakage. The cleaner methodology fixes τ on dev and applies that fixed τ to test predictions. AUROC is threshold-independent and is identical across both choices; only macro-F1 and accuracy change.

| Recipe | τ (dev) mean | Clean F1 @ τ_dev | Clean F1 @ τ_test (default) | Clean Acc @ τ_dev | Clean Acc @ τ_test |
|---|---:|---:|---:|---:|---:|
| clean | 0.323 ± 0.033 | 0.6595 ± 0.0056 | 0.6757 ± 0.0047 | 0.6643 ± 0.0065 | 0.6895 ± 0.0052 |
| augonly | 0.440 ± 0.008 | 0.6482 ± 0.0226 | 0.6517 ± 0.0212 | 0.6623 ± 0.0216 | 0.6662 ± 0.0245 |
| kl | 0.367 ± 0.040 | 0.6602 ± 0.0123 | 0.6760 ± 0.0098 | 0.6668 ± 0.0128 | 0.6917 ± 0.0098 |
| kldrop-p015 | 0.373 ± 0.012 | 0.6688 ± 0.0086 | 0.6817 ± 0.0099 | 0.6768 ± 0.0066 | 0.7075 ± 0.0023 |
| kldrop-p050 | 0.413 ± 0.068 | 0.6356 ± 0.0172 | 0.6561 ± 0.0299 | 0.6467 ± 0.0098 | 0.6732 ± 0.0295 |
| kllowmed | 0.373 ± 0.037 | 0.6516 ± 0.0012 | 0.6747 ± 0.0049 | 0.6558 ± 0.0033 | 0.6980 ± 0.0033 |

Reading: "Clean F1 @ τ_dev" is the deployment-honest number — what you would get if you tuned τ once on dev and shipped to a test-like distribution. The split-tuned column uses test labels and is therefore an upper bound on real-world F1. AUROC columns elsewhere in this file are unaffected.


## Appendix — dominated recipes (for audit)

Recipes that earlier phases tested but that were strictly Pareto-dominated by a sibling recipe (e.g. `kldrop` p=0.30 by `kldrop-p015`, see Phase 9b § 4). Numbers retained for reviewer comparison against the earlier phase reports; omitted from the main tables above.

| Recipe | n seeds | Clean AUROC | Worst-cell text Δ | Image-only AUROC | Composite_2text med Δ |
|---|---:|---:|---:|---:|---:|
| kldrop | 3 | 0.7126 ± 0.0333 | 0.0547 ± 0.0155 | 0.6550 ± 0.0155 | 0.0391 ± 0.0145 |

