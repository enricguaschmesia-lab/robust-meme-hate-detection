# Phase 5 — Robust vs clean training comparison

All multimodal numbers aggregate the 3 seeds (mean ± σ). Each row is one training recipe.

## Clean accuracy & naturalistic worst-case

| Recipe | n seeds | Clean AUROC | Clean F1 | Worst-cell text ΔAUROC | Worst-cell image ΔAUROC |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 0.7471 ± 0.0056 | 0.7012 ± 0.0065 | 0.1288 ± 0.0018 | 0.0317 ± 0.0060 |
| augonly | 3 | 0.7158 ± 0.0310 | 0.6737 ± 0.0263 | 0.0832 ± 0.0219 | 0.0159 ± 0.0039 |
| kl | 3 | 0.7454 ± 0.0081 | 0.6927 ± 0.0046 | 0.0855 ± 0.0014 | 0.0194 ± 0.0019 |
| kldrop-p015 | 3 | 0.7478 ± 0.0081 | 0.6912 ± 0.0080 | 0.0866 ± 0.0073 | 0.0284 ± 0.0024 |
| kldrop-p050 | 3 | 0.7172 ± 0.0338 | 0.6686 ± 0.0275 | 0.0676 ± 0.0184 | 0.0279 ± 0.0082 |
| kllowmed | 3 | 0.7444 ± 0.0054 | 0.6943 ± 0.0078 | 0.0885 ± 0.0062 | 0.0244 ± 0.0018 |

## Mean ΔAUROC at high severity, per family

| Recipe | text | image-pixel | image-photometric | image-geometric | typographic |
|---|---:|---:|---:|---:|---:|
| clean | 0.0887 ± 0.0016 | 0.0210 ± 0.0021 | 0.0098 ± 0.0007 | 0.0072 ± 0.0032 | 0.0113 ± 0.0039 |
| augonly | 0.0576 ± 0.0169 | 0.0116 ± 0.0041 | 0.0050 ± 0.0034 | 0.0010 ± 0.0038 | 0.0103 ± 0.0054 |
| kl | 0.0614 ± 0.0006 | 0.0151 ± 0.0019 | 0.0082 ± 0.0012 | 0.0040 ± 0.0006 | 0.0132 ± 0.0030 |
| kldrop-p015 | 0.0611 ± 0.0038 | 0.0192 ± 0.0012 | 0.0076 ± 0.0001 | 0.0083 ± 0.0014 | 0.0145 ± 0.0038 |
| kldrop-p050 | 0.0443 ± 0.0163 | 0.0185 ± 0.0063 | 0.0072 ± 0.0021 | 0.0061 ± 0.0042 | 0.0151 ± 0.0093 |
| kllowmed | 0.0639 ± 0.0045 | 0.0185 ± 0.0007 | 0.0092 ± 0.0010 | 0.0049 ± 0.0020 | 0.0127 ± 0.0028 |

## White-box PGD AUROC vs ε

| Recipe | clean AUROC | PGD ε=1/255 | PGD ε=2/255 | PGD ε=4/255 | PGD ε=8/255 |
|---|---:|---:|---:|---:|---:|
| clean | 0.7488 ± 0.0042 | 0.0126 ± 0.0007 | 0.0003 ± 0.0000 | 0.0000 | 0.0000 |
| augonly | 0.7140 ± 0.0291 | 0.0168 ± 0.0019 | 0.0006 ± 0.0002 | 0.0000 | 0.0000 |
| kl | 0.7458 ± 0.0102 | 0.0200 ± 0.0045 | 0.0009 ± 0.0005 | 0.0000 ± 0.0000 | 0.0000 |
| kldrop-p015 | 0.7449 ± 0.0097 | 0.0148 ± 0.0017 | 0.0004 ± 0.0003 | 0.0000 ± 0.0000 | 0.0000 |
| kldrop-p050 | 0.7099 ± 0.0322 | 0.0116 ± 0.0004 | 0.0002 ± 0.0000 | 0.0000 | 0.0000 |
| kllowmed | 0.7461 ± 0.0059 | 0.0213 ± 0.0009 | 0.0010 ± 0.0003 | 0.0000 | 0.0000 |

## Modality ablation (clean dev forwards)

Direct test of the "dead image branch" failure mode: did `KL_image_branch` lift `image_only` AUROC?

| Recipe | multimodal AUROC | text_only AUROC | image_only AUROC |
|---|---:|---:|---:|
| clean | 0.7471 ± 0.0056 | 0.6440 ± 0.0055 | 0.5822 ± 0.0029 |
| augonly | 0.7158 ± 0.0310 | 0.6344 ± 0.0081 | 0.6059 ± 0.0110 |
| kl | 0.7454 ± 0.0081 | 0.6429 ± 0.0016 | 0.6045 ± 0.0074 |
| kldrop-p015 | 0.7478 ± 0.0081 | 0.6472 ± 0.0053 | 0.6382 ± 0.0098 |
| kldrop-p050 | 0.7172 ± 0.0338 | 0.6395 ± 0.0125 | 0.6513 ± 0.0098 |
| kllowmed | 0.7444 ± 0.0054 | 0.6427 ± 0.0018 | 0.5968 ± 0.0010 |

## Per-example fully-robust counts (out of 500 dev)

Two threat models, reported separately:

* **Naturalistic-only**: example is clean-correct AND survives every naturalistic perturbation cell (8 text × 3 + 11 image × 3 = 57 cells).
* **Natural + white-box PGD**: same, plus survives PGD ε=4/255. PGD has near-100 % ASR by design (worst-case oracle) so this number is dominated by it; reported only for completeness.

| Recipe | Naturalistic seed0 | seed1 | seed2 | mean | Nat + PGD seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 167 | 133 | 143 | 147.67 | 0 | 2 | 0 | 0.67 |
| augonly | 165 | 168 | 143 | 158.67 | 1 | 3 | 1 | 1.67 |
| kl | 183 | 192 | 182 | 185.67 | 2 | 4 | 0 | 2.00 |
| kldrop-p015 | 166 | 221 | 200 | 195.67 | 1 | 1 | 0 | 0.67 |
| kldrop-p050 | 198 | 171 | 194 | 187.67 | 0 | 0 | 2 | 0.67 |
| kllowmed | 178 | 173 | 172 | 174.33 | 2 | 2 | 4 | 2.67 |


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
| kldrop-p015 | 0.0492 ± 0.0031 | 0.0291 ± 0.0019 | -0.0201 | 0.0082 ± 0.0010 | 0.0048 ± 0.0006 | -0.0035 |
| kldrop-p050 | 0.0361 ± 0.0126 | 0.0212 ± 0.0085 | -0.0148 | 0.0074 ± 0.0028 | 0.0042 ± 0.0015 | -0.0032 |
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
| kldrop-p015 | 0.0229 ± 0.0011 | **0.0411 ± 0.0031** | 0.0611 ± 0.0038 |
| kldrop-p050 | 0.0164 ± 0.0060 | **0.0308 ± 0.0110** | 0.0443 ± 0.0163 |
| kllowmed | 0.0225 ± 0.0018 | **0.0431 ± 0.0030** | 0.0639 ± 0.0045 |

### image-pixel

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0079 ± 0.0011 | **0.0117 ± 0.0017** | 0.0210 ± 0.0021 |
| augonly | 0.0036 ± 0.0022 | **0.0058 ± 0.0025** | 0.0116 ± 0.0041 |
| kl | 0.0042 ± 0.0010 | **0.0069 ± 0.0014** | 0.0151 ± 0.0019 |
| kldrop-p015 | 0.0068 ± 0.0009 | **0.0100 ± 0.0028** | 0.0192 ± 0.0012 |
| kldrop-p050 | 0.0044 ± 0.0021 | **0.0093 ± 0.0037** | 0.0185 ± 0.0063 |
| kllowmed | 0.0055 ± 0.0012 | **0.0098 ± 0.0004** | 0.0185 ± 0.0007 |

### image-photometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0017 ± 0.0001 | **0.0045 ± 0.0005** | 0.0098 ± 0.0007 |
| augonly | 0.0002 ± 0.0008 | **0.0011 ± 0.0019** | 0.0050 ± 0.0034 |
| kl | 0.0010 ± 0.0008 | **0.0031 ± 0.0011** | 0.0082 ± 0.0012 |
| kldrop-p015 | 0.0012 ± 0.0003 | **0.0032 ± 0.0001** | 0.0076 ± 0.0001 |
| kldrop-p050 | 0.0005 ± 0.0005 | **0.0024 ± 0.0012** | 0.0072 ± 0.0021 |
| kllowmed | 0.0010 ± 0.0008 | **0.0035 ± 0.0012** | 0.0092 ± 0.0010 |

### image-geometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0005 ± 0.0011 | **0.0057 ± 0.0006** | 0.0072 ± 0.0032 |
| augonly | -0.0020 ± 0.0003 | **0.0036 ± 0.0012** | 0.0010 ± 0.0038 |
| kl | -0.0019 ± 0.0012 | **0.0046 ± 0.0016** | 0.0040 ± 0.0006 |
| kldrop-p015 | 0.0005 ± 0.0006 | **0.0064 ± 0.0009** | 0.0083 ± 0.0014 |
| kldrop-p050 | 0.0001 ± 0.0008 | **0.0069 ± 0.0004** | 0.0061 ± 0.0042 |
| kllowmed | -0.0005 ± 0.0007 | **0.0056 ± 0.0010** | 0.0049 ± 0.0020 |

### typographic

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0015 ± 0.0010 | **0.0033 ± 0.0043** | 0.0113 ± 0.0039 |
| augonly | -0.0008 ± 0.0010 | **0.0009 ± 0.0024** | 0.0103 ± 0.0054 |
| kl | -0.0013 ± 0.0004 | **0.0035 ± 0.0026** | 0.0132 ± 0.0030 |
| kldrop-p015 | -0.0004 ± 0.0010 | **0.0038 ± 0.0008** | 0.0145 ± 0.0038 |
| kldrop-p050 | 0.0004 ± 0.0014 | **0.0036 ± 0.0008** | 0.0151 ± 0.0093 |
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

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0616 ± 0.0017 | 0.0870 ± 0.0012 | 0.1202 ± 0.0043 | **0.0947 ± 0.0019** | 0.3383 ± 0.0127 | 0.2847 ± 0.0056 |
| augonly | 0.0378 ± 0.0098 | 0.0582 ± 0.0148 | 0.0747 ± 0.0174 | **0.0584 ± 0.0206** | 0.2333 ± 0.0173 | 0.1896 ± 0.0206 |
| kl | 0.0432 ± 0.0016 | 0.0635 ± 0.0038 | 0.0804 ± 0.0015 | **0.0649 ± 0.0043** | 0.2322 ± 0.0019 | 0.1942 ± 0.0086 |
| kldrop-p015 | 0.0422 ± 0.0032 | 0.0637 ± 0.0050 | 0.0795 ± 0.0055 | **0.0626 ± 0.0036** | 0.2244 ± 0.0208 | 0.1841 ± 0.0117 |
| kldrop-p050 | 0.0318 ± 0.0115 | 0.0457 ± 0.0124 | 0.0558 ± 0.0212 | **0.0465 ± 0.0144** | 0.1893 ± 0.0352 | 0.1663 ± 0.0270 |
| kllowmed | 0.0423 ± 0.0050 | 0.0637 ± 0.0044 | 0.0857 ± 0.0036 | **0.0690 ± 0.0067** | 0.2537 ± 0.0072 | 0.2066 ± 0.0187 |

### composite_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0024 ± 0.0028 | 0.0043 ± 0.0036 | 0.0344 ± 0.0027 | **0.0163 ± 0.0011** | 0.1711 ± 0.0065 | 0.1236 ± 0.0047 |
| augonly | 0.0018 ± 0.0023 | 0.0037 ± 0.0023 | 0.0206 ± 0.0110 | **0.0052 ± 0.0049** | 0.1659 ± 0.0210 | 0.1154 ± 0.0228 |
| kl | 0.0025 ± 0.0013 | 0.0021 ± 0.0043 | 0.0271 ± 0.0018 | **0.0103 ± 0.0032** | 0.1533 ± 0.0154 | 0.1024 ± 0.0041 |
| kldrop-p015 | 0.0016 ± 0.0009 | 0.0003 ± 0.0005 | 0.0301 ± 0.0041 | **0.0106 ± 0.0031** | 0.1739 ± 0.0224 | 0.1124 ± 0.0169 |
| kldrop-p050 | 0.0030 ± 0.0023 | 0.0085 ± 0.0011 | 0.0279 ± 0.0082 | **0.0113 ± 0.0028** | 0.1762 ± 0.0140 | 0.1229 ± 0.0132 |
| kllowmed | 0.0032 ± 0.0015 | 0.0041 ± 0.0029 | 0.0307 ± 0.0027 | **0.0123 ± 0.0022** | 0.1607 ± 0.0141 | 0.1083 ± 0.0080 |

### composite_text_image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0356 ± 0.0027 | 0.0575 ± 0.0027 | 0.1035 ± 0.0086 | **0.0669 ± 0.0040** | 0.3090 ± 0.0073 | 0.2304 ± 0.0109 |
| augonly | 0.0274 ± 0.0013 | 0.0357 ± 0.0149 | 0.0708 ± 0.0205 | **0.0411 ± 0.0109** | 0.2380 ± 0.0125 | 0.1708 ± 0.0079 |
| kl | 0.0294 ± 0.0006 | 0.0416 ± 0.0063 | 0.0725 ± 0.0026 | **0.0476 ± 0.0002** | 0.2312 ± 0.0067 | 0.1678 ± 0.0020 |
| kldrop-p015 | 0.0292 ± 0.0013 | 0.0431 ± 0.0024 | 0.0782 ± 0.0027 | **0.0462 ± 0.0018** | 0.2366 ± 0.0084 | 0.1716 ± 0.0169 |
| kldrop-p050 | 0.0230 ± 0.0025 | 0.0321 ± 0.0130 | 0.0599 ± 0.0196 | **0.0375 ± 0.0121** | 0.2336 ± 0.0243 | 0.1538 ± 0.0137 |
| kllowmed | 0.0310 ± 0.0017 | 0.0418 ± 0.0059 | 0.0768 ± 0.0084 | **0.0498 ± 0.0024** | 0.2454 ± 0.0164 | 0.1784 ± 0.0049 |

### composite_2text_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0627 ± 0.0006 | 0.1019 ± 0.0047 | 0.1394 ± 0.0041 | **0.1044 ± 0.0025** | 0.3805 ± 0.0274 | 0.3302 ± 0.0101 |
| augonly | 0.0391 ± 0.0153 | 0.0629 ± 0.0227 | 0.0946 ± 0.0290 | **0.0590 ± 0.0253** | 0.3195 ± 0.0138 | 0.2438 ± 0.0144 |
| kl | 0.0428 ± 0.0022 | 0.0716 ± 0.0048 | 0.1100 ± 0.0027 | **0.0738 ± 0.0063** | 0.3106 ± 0.0158 | 0.2432 ± 0.0054 |
| kldrop-p015 | 0.0441 ± 0.0038 | 0.0731 ± 0.0064 | 0.1154 ± 0.0080 | **0.0736 ± 0.0056** | 0.3071 ± 0.0219 | 0.2394 ± 0.0192 |
| kldrop-p050 | 0.0335 ± 0.0121 | 0.0607 ± 0.0211 | 0.0941 ± 0.0259 | **0.0542 ± 0.0232** | 0.3045 ± 0.0320 | 0.2325 ± 0.0273 |
| kllowmed | 0.0441 ± 0.0053 | 0.0763 ± 0.0058 | 0.1140 ± 0.0012 | **0.0724 ± 0.0011** | 0.3204 ± 0.0050 | 0.2508 ± 0.0102 |


## Deployment-honest threshold: dev-tuned τ applied to test

Per-split-tuned `τ` (the default elsewhere in this file) uses *test* labels to pick the threshold — mild data leakage. The cleaner methodology fixes τ on dev and applies that fixed τ to test predictions. AUROC is threshold-independent and is identical across both choices; only macro-F1 and accuracy change.

| Recipe | τ (dev) mean | Clean F1 @ τ_dev | Clean F1 @ τ_test (default) | Clean Acc @ τ_dev | Clean Acc @ τ_test |
|---|---:|---:|---:|---:|---:|
| clean | 0.323 ± 0.033 | 0.6883 ± 0.0115 | 0.7012 ± 0.0065 | 0.6890 ± 0.0110 | 0.7013 ± 0.0065 |
| augonly | 0.440 ± 0.008 | 0.6716 ± 0.0288 | 0.6737 ± 0.0263 | 0.6720 ± 0.0289 | 0.6740 ± 0.0263 |
| kl | 0.367 ± 0.040 | 0.6854 ± 0.0053 | 0.6927 ± 0.0046 | 0.6857 ± 0.0053 | 0.6933 ± 0.0041 |
| kldrop-p015 | 0.373 ± 0.012 | 0.6902 ± 0.0093 | 0.6912 ± 0.0080 | 0.6907 ± 0.0098 | 0.6930 ± 0.0070 |
| kldrop-p050 | 0.413 ± 0.068 | 0.6508 ± 0.0231 | 0.6686 ± 0.0275 | 0.6533 ± 0.0222 | 0.6693 ± 0.0268 |
| kllowmed | 0.373 ± 0.037 | 0.6822 ± 0.0031 | 0.6943 ± 0.0078 | 0.6833 ± 0.0039 | 0.6950 ± 0.0073 |

Reading: "Clean F1 @ τ_dev" is the deployment-honest number — what you would get if you tuned τ once on dev and shipped to a test-like distribution. The split-tuned column uses test labels and is therefore an upper bound on real-world F1. AUROC columns elsewhere in this file are unaffected.


## Appendix — dominated recipes (for audit)

Recipes that earlier phases tested but that were strictly Pareto-dominated by a sibling recipe (e.g. `kldrop` p=0.30 by `kldrop-p015`, see Phase 9b § 4). Numbers retained for reviewer comparison against the earlier phase reports; omitted from the main tables above.

| Recipe | n seeds | Clean AUROC | Worst-cell text Δ | Image-only AUROC | Composite_2text med Δ |
|---|---:|---:|---:|---:|---:|
| kldrop | 3 | 0.7211 ± 0.0312 | 0.0736 ± 0.0162 | 0.6411 ± 0.0047 | 0.0498 ± 0.0136 |

