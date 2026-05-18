# Phase 5 — Robust vs clean training comparison

All multimodal numbers aggregate the 3 seeds (mean ± σ). Each row is one training recipe.

## Clean accuracy & naturalistic worst-case

| Recipe | n seeds | Clean AUROC | Clean F1 | Worst-cell text ΔAUROC | Worst-cell image ΔAUROC |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 0.7418 ± 0.0057 | 0.6954 ± 0.0112 | 0.1149 ± 0.0056 | 0.0381 ± 0.0028 |
| augonly | 3 | 0.7123 ± 0.0215 | 0.6691 ± 0.0082 | 0.0828 ± 0.0121 | 0.0306 ± 0.0021 |
| kl | 3 | 0.7367 ± 0.0076 | 0.6875 ± 0.0056 | 0.0898 ± 0.0038 | 0.0381 ± 0.0026 |
| kldrop | 3 | 0.7051 ± 0.0254 | 0.6550 ± 0.0173 | 0.0673 ± 0.0083 | 0.0360 ± 0.0071 |
| kldrop-p015 | 3 | 0.7345 ± 0.0020 | 0.6848 ± 0.0012 | 0.0807 ± 0.0038 | 0.0456 ± 0.0021 |
| kldrop-p050 | 3 | 0.7002 ± 0.0312 | 0.6539 ± 0.0254 | 0.0654 ± 0.0112 | 0.0413 ± 0.0068 |
| kllowmed | 3 | 0.7347 ± 0.0045 | 0.6924 ± 0.0021 | 0.0926 ± 0.0078 | 0.0415 ± 0.0053 |

## Mean ΔAUROC at high severity, per family

| Recipe | text | image-pixel | image-photometric | image-geometric | typographic |
|---|---:|---:|---:|---:|---:|
| clean | 0.0854 ± 0.0040 | 0.0247 ± 0.0054 | 0.0090 ± 0.0046 | 0.0058 ± 0.0057 | 0.0155 ± 0.0017 |
| augonly | 0.0587 ± 0.0097 | 0.0205 ± 0.0036 | 0.0077 ± 0.0019 | 0.0054 ± 0.0030 | 0.0092 ± 0.0040 |
| kl | 0.0617 ± 0.0039 | 0.0237 ± 0.0016 | 0.0068 ± 0.0013 | 0.0049 ± 0.0028 | 0.0141 ± 0.0014 |
| kldrop | 0.0471 ± 0.0075 | 0.0201 ± 0.0072 | 0.0049 ± 0.0017 | 0.0000 ± 0.0008 | 0.0125 ± 0.0031 |
| kldrop-p015 | 0.0584 ± 0.0033 | 0.0274 ± 0.0022 | 0.0065 ± 0.0010 | 0.0012 ± 0.0020 | 0.0173 ± 0.0016 |
| kldrop-p050 | 0.0443 ± 0.0109 | 0.0202 ± 0.0074 | 0.0038 ± 0.0029 | -0.0023 ± 0.0019 | 0.0117 ± 0.0045 |
| kllowmed | 0.0651 ± 0.0047 | 0.0246 ± 0.0021 | 0.0064 ± 0.0004 | 0.0056 ± 0.0006 | 0.0157 ± 0.0012 |

## White-box PGD AUROC vs ε

| Recipe | clean AUROC | PGD ε=1/255 | PGD ε=2/255 | PGD ε=4/255 | PGD ε=8/255 |
|---|---:|---:|---:|---:|---:|
| clean | 0.7338 ± 0.0074 | 0.0123 ± 0.0009 | 0.0004 ± 0.0001 | 0.0000 | 0.0000 |
| augonly | 0.6983 ± 0.0274 | 0.0189 ± 0.0013 | 0.0008 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 |
| kl | 0.7293 ± 0.0095 | 0.0194 ± 0.0030 | 0.0009 ± 0.0003 | 0.0000 ± 0.0000 | 0.0000 |
| kldrop | 0.6979 ± 0.0314 | 0.0181 ± 0.0030 | 0.0008 ± 0.0002 | 0.0000 | 0.0000 |
| kldrop-p015 | 0.7326 ± 0.0024 | 0.0156 ± 0.0022 | 0.0007 ± 0.0001 | 0.0000 | 0.0000 |
| kldrop-p050 | 0.6915 ± 0.0337 | 0.0140 ± 0.0032 | 0.0006 ± 0.0002 | 0.0000 | 0.0000 |
| kllowmed | 0.7263 ± 0.0063 | 0.0199 ± 0.0013 | 0.0010 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 |

## Modality ablation (clean dev forwards)

Direct test of the "dead image branch" failure mode: did `KL_image_branch` lift `image_only` AUROC?

| Recipe | multimodal AUROC | text_only AUROC | image_only AUROC |
|---|---:|---:|---:|
| clean | 0.7418 ± 0.0057 | 0.6249 ± 0.0031 | 0.5925 ± 0.0035 |
| augonly | 0.7123 ± 0.0215 | 0.6221 ± 0.0026 | 0.6040 ± 0.0029 |
| kl | 0.7367 ± 0.0076 | 0.6236 ± 0.0040 | 0.6106 ± 0.0039 |
| kldrop | 0.7051 ± 0.0254 | 0.6281 ± 0.0088 | 0.6360 ± 0.0141 |
| kldrop-p015 | 0.7345 ± 0.0020 | 0.6287 ± 0.0046 | 0.6383 ± 0.0034 |
| kldrop-p050 | 0.7002 ± 0.0312 | 0.6229 ± 0.0050 | 0.6407 ± 0.0169 |
| kllowmed | 0.7347 ± 0.0045 | 0.6258 ± 0.0046 | 0.6019 ± 0.0057 |

## Per-example fully-robust counts (out of 500 dev)

Two threat models, reported separately:

* **Naturalistic-only**: example is clean-correct AND survives every naturalistic perturbation cell (8 text × 3 + 11 image × 3 = 57 cells).
* **Natural + white-box PGD**: same, plus survives PGD ε=4/255. PGD has near-100 % ASR by design (worst-case oracle) so this number is dominated by it; reported only for completeness.

| Recipe | Naturalistic seed0 | seed1 | seed2 | mean | Nat + PGD seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 77 | 91 | 96 | 88.00 | 0 | 0 | 0 | 0.00 |
| augonly | 85 | 86 | 69 | 80.00 | 1 | 1 | 3 | 1.67 |
| kl | 103 | 98 | 116 | 105.67 | 1 | 1 | 1 | 1.00 |
| kldrop | 88 | 85 | 88 | 87.00 | 0 | 0 | 1 | 0.33 |
| kldrop-p015 | 91 | 109 | 100 | 100.00 | 0 | 0 | 1 | 0.33 |
| kldrop-p050 | 118 | 106 | 94 | 106.00 | 2 | 1 | 1 | 1.33 |
| kllowmed | 97 | 95 | 113 | 101.67 | 1 | 2 | 1 | 1.33 |


## Attack-type generalisation: in-pool vs held-out (OOD)

Training pool (text, n=5): `censoring, char_deletion, char_swap, keyboard_typo, leetspeak`  
Held-out text (n=3): `case_noise, punctuation, spacing`  
Training pool (image, n=6): `blur, brightness_down, compression, gaussian_noise, occlusion, typographic`  
Held-out image (n=5): `brightness_up, contrast_down, contrast_up, crop, translation`

Mean ΔAUROC (clean − attacked) across all severities of the cells in each partition. Lower is better. *Caveat*: held-out text attacks (`spacing/punctuation/case_noise`) are the weakest text attacks on the clean ckpt (ΔAUROC ≤ 0.02), so the text-side OOD column has limited signal. The image-side OOD column is the meaningful generalisation test.

| Recipe | text in-pool | text OOD | Δ(OOD-in) text | image in-pool | image OOD | Δ(OOD-in) image |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0691 ± 0.0036 | 0.0411 ± 0.0004 | -0.0280 | 0.0099 ± 0.0034 | 0.0048 ± 0.0037 | -0.0051 |
| augonly | 0.0478 ± 0.0078 | 0.0272 ± 0.0047 | -0.0206 | 0.0075 ± 0.0016 | 0.0037 ± 0.0018 | -0.0038 |
| kl | 0.0496 ± 0.0037 | 0.0284 ± 0.0014 | -0.0212 | 0.0078 ± 0.0009 | 0.0034 ± 0.0019 | -0.0044 |
| kldrop | 0.0375 ± 0.0044 | 0.0227 ± 0.0039 | -0.0148 | 0.0064 ± 0.0018 | 0.0013 ± 0.0009 | -0.0051 |
| kldrop-p015 | 0.0451 ± 0.0027 | 0.0277 ± 0.0021 | -0.0175 | 0.0087 ± 0.0012 | 0.0025 ± 0.0009 | -0.0062 |
| kldrop-p050 | 0.0353 ± 0.0071 | 0.0204 ± 0.0059 | -0.0149 | 0.0058 ± 0.0028 | 0.0002 ± 0.0020 | -0.0056 |
| kllowmed | 0.0522 ± 0.0037 | 0.0296 ± 0.0017 | -0.0227 | 0.0086 ± 0.0007 | 0.0037 ± 0.0003 | -0.0049 |

Reading: Δ(OOD−in) > 0 means the recipe defends in-pool attacks more than held-out ones — i.e. augmentation has *memorised* its training pool rather than learning a transferable defence. Δ near 0 (or negative) is the generalisation signal we want.


## Per-severity sensitivity (single-perturbation cells)

Mean ΔAUROC at each severity, per family. **Medium severity is the internet-realistic threat number** (the level a typical adversarial user would reach with an off-the-shelf editor); high severity is a stress-test ceiling that often borders on label-preserving limits; low is included for monotonicity sanity-checking.

### text

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0278 ± 0.0017 | **0.0626 ± 0.0015** | 0.0854 ± 0.0040 |
| augonly | 0.0192 ± 0.0036 | **0.0423 ± 0.0067** | 0.0587 ± 0.0097 |
| kl | 0.0191 ± 0.0027 | **0.0443 ± 0.0021** | 0.0617 ± 0.0039 |
| kldrop | 0.0148 ± 0.0017 | **0.0341 ± 0.0039** | 0.0471 ± 0.0075 |
| kldrop-p015 | 0.0171 ± 0.0021 | **0.0403 ± 0.0022** | 0.0584 ± 0.0033 |
| kldrop-p050 | 0.0135 ± 0.0025 | **0.0312 ± 0.0067** | 0.0443 ± 0.0109 |
| kllowmed | 0.0199 ± 0.0029 | **0.0463 ± 0.0017** | 0.0651 ± 0.0047 |

### image-pixel

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0051 ± 0.0034 | **0.0153 ± 0.0059** | 0.0247 ± 0.0054 |
| augonly | 0.0042 ± 0.0010 | **0.0102 ± 0.0025** | 0.0205 ± 0.0036 |
| kl | 0.0047 ± 0.0011 | **0.0099 ± 0.0008** | 0.0237 ± 0.0016 |
| kldrop | 0.0030 ± 0.0017 | **0.0082 ± 0.0034** | 0.0201 ± 0.0072 |
| kldrop-p015 | 0.0051 ± 0.0008 | **0.0119 ± 0.0016** | 0.0274 ± 0.0022 |
| kldrop-p050 | 0.0024 ± 0.0019 | **0.0075 ± 0.0034** | 0.0202 ± 0.0074 |
| kllowmed | 0.0043 ± 0.0009 | **0.0111 ± 0.0004** | 0.0246 ± 0.0021 |

### image-photometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0008 ± 0.0009 | **0.0046 ± 0.0031** | 0.0090 ± 0.0046 |
| augonly | 0.0009 ± 0.0008 | **0.0038 ± 0.0019** | 0.0077 ± 0.0019 |
| kl | 0.0005 ± 0.0003 | **0.0029 ± 0.0013** | 0.0068 ± 0.0013 |
| kldrop | -0.0000 ± 0.0002 | **0.0029 ± 0.0012** | 0.0049 ± 0.0017 |
| kldrop-p015 | 0.0002 ± 0.0001 | **0.0034 ± 0.0016** | 0.0065 ± 0.0010 |
| kldrop-p050 | -0.0005 ± 0.0010 | **0.0017 ± 0.0024** | 0.0038 ± 0.0029 |
| kllowmed | 0.0007 ± 0.0002 | **0.0027 ± 0.0003** | 0.0064 ± 0.0004 |

### image-geometric

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0045 ± 0.0033 | **0.0031 ± 0.0048** | 0.0058 ± 0.0057 |
| augonly | 0.0034 ± 0.0014 | **0.0013 ± 0.0031** | 0.0054 ± 0.0030 |
| kl | 0.0032 ± 0.0017 | **0.0005 ± 0.0049** | 0.0049 ± 0.0028 |
| kldrop | 0.0014 ± 0.0014 | **-0.0016 ± 0.0012** | 0.0000 ± 0.0008 |
| kldrop-p015 | 0.0022 ± 0.0005 | **-0.0006 ± 0.0018** | 0.0012 ± 0.0020 |
| kldrop-p050 | 0.0006 ± 0.0023 | **-0.0009 ± 0.0023** | -0.0023 ± 0.0019 |
| kllowmed | 0.0038 ± 0.0006 | **0.0029 ± 0.0020** | 0.0056 ± 0.0006 |

### typographic

| Recipe | low | **medium** (realistic) | high (stress) |
|---|---:|---:|---:|
| clean | 0.0001 ± 0.0047 | **0.0011 ± 0.0018** | 0.0155 ± 0.0017 |
| augonly | -0.0012 ± 0.0027 | **-0.0025 ± 0.0015** | 0.0092 ± 0.0040 |
| kl | -0.0019 ± 0.0023 | **-0.0021 ± 0.0022** | 0.0141 ± 0.0014 |
| kldrop | -0.0010 ± 0.0043 | **-0.0025 ± 0.0015** | 0.0125 ± 0.0031 |
| kldrop-p015 | -0.0038 ± 0.0020 | **-0.0019 ± 0.0049** | 0.0173 ± 0.0016 |
| kldrop-p050 | -0.0019 ± 0.0038 | **-0.0056 ± 0.0037** | 0.0117 ± 0.0045 |
| kllowmed | -0.0010 ± 0.0012 | **-0.0020 ± 0.0018** | 0.0157 ± 0.0012 |


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
| clean | 0.0549 ± 0.0042 | 0.0901 ± 0.0022 | 0.1302 ± 0.0107 | **0.0983 ± 0.0058** | 0.3324 ± 0.0229 | 0.2913 ± 0.0084 |
| augonly | 0.0380 ± 0.0067 | 0.0633 ± 0.0135 | 0.0913 ± 0.0021 | **0.0723 ± 0.0040** | 0.2391 ± 0.0123 | 0.2022 ± 0.0029 |
| kl | 0.0388 ± 0.0031 | 0.0660 ± 0.0034 | 0.0911 ± 0.0038 | **0.0675 ± 0.0032** | 0.2316 ± 0.0048 | 0.1929 ± 0.0103 |
| kldrop | 0.0310 ± 0.0087 | 0.0511 ± 0.0068 | 0.0685 ± 0.0018 | **0.0590 ± 0.0032** | 0.2129 ± 0.0053 | 0.1843 ± 0.0132 |
| kldrop-p015 | 0.0384 ± 0.0015 | 0.0594 ± 0.0038 | 0.0751 ± 0.0025 | **0.0637 ± 0.0035** | 0.2276 ± 0.0245 | 0.2111 ± 0.0069 |
| kldrop-p050 | 0.0290 ± 0.0114 | 0.0455 ± 0.0094 | 0.0610 ± 0.0071 | **0.0509 ± 0.0076** | 0.2130 ± 0.0124 | 0.1802 ± 0.0183 |
| kllowmed | 0.0397 ± 0.0007 | 0.0645 ± 0.0035 | 0.0955 ± 0.0059 | **0.0744 ± 0.0022** | 0.2478 ± 0.0056 | 0.2152 ± 0.0086 |

### composite_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0182 ± 0.0031 | 0.0108 ± 0.0078 | 0.0353 ± 0.0080 | **0.0095 ± 0.0101** | 0.1773 ± 0.0079 | 0.1206 ± 0.0076 |
| augonly | 0.0124 ± 0.0030 | 0.0081 ± 0.0016 | 0.0223 ± 0.0051 | **0.0056 ± 0.0036** | 0.1797 ± 0.0352 | 0.1197 ± 0.0160 |
| kl | 0.0108 ± 0.0028 | 0.0059 ± 0.0011 | 0.0154 ± 0.0005 | **0.0074 ± 0.0038** | 0.1570 ± 0.0074 | 0.1067 ± 0.0140 |
| kldrop | 0.0102 ± 0.0023 | 0.0051 ± 0.0038 | 0.0136 ± 0.0037 | **0.0002 ± 0.0072** | 0.1691 ± 0.0172 | 0.1113 ± 0.0160 |
| kldrop-p015 | 0.0119 ± 0.0021 | 0.0067 ± 0.0035 | 0.0185 ± 0.0064 | **0.0067 ± 0.0026** | 0.1722 ± 0.0044 | 0.1284 ± 0.0045 |
| kldrop-p050 | 0.0110 ± 0.0039 | 0.0050 ± 0.0041 | 0.0134 ± 0.0026 | **-0.0014 ± 0.0061** | 0.1909 ± 0.0056 | 0.1341 ± 0.0064 |
| kllowmed | 0.0110 ± 0.0023 | 0.0041 ± 0.0004 | 0.0191 ± 0.0041 | **0.0082 ± 0.0050** | 0.1547 ± 0.0078 | 0.1008 ± 0.0127 |

### composite_text_image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0294 ± 0.0038 | 0.0496 ± 0.0103 | 0.0760 ± 0.0124 | **0.0675 ± 0.0038** | 0.3075 ± 0.0123 | 0.2433 ± 0.0039 |
| augonly | 0.0213 ± 0.0051 | 0.0332 ± 0.0088 | 0.0559 ± 0.0038 | **0.0583 ± 0.0019** | 0.2300 ± 0.0037 | 0.1884 ± 0.0107 |
| kl | 0.0157 ± 0.0022 | 0.0360 ± 0.0048 | 0.0502 ± 0.0049 | **0.0504 ± 0.0053** | 0.2083 ± 0.0048 | 0.1580 ± 0.0123 |
| kldrop | 0.0136 ± 0.0061 | 0.0243 ± 0.0092 | 0.0488 ± 0.0035 | **0.0476 ± 0.0058** | 0.2282 ± 0.0017 | 0.1766 ± 0.0070 |
| kldrop-p015 | 0.0104 ± 0.0016 | 0.0325 ± 0.0019 | 0.0510 ± 0.0079 | **0.0535 ± 0.0011** | 0.2169 ± 0.0179 | 0.1829 ± 0.0121 |
| kldrop-p050 | 0.0119 ± 0.0030 | 0.0184 ± 0.0100 | 0.0449 ± 0.0065 | **0.0467 ± 0.0088** | 0.2313 ± 0.0139 | 0.1860 ± 0.0015 |
| kllowmed | 0.0200 ± 0.0007 | 0.0372 ± 0.0054 | 0.0551 ± 0.0061 | **0.0519 ± 0.0011** | 0.2171 ± 0.0064 | 0.1748 ± 0.0109 |

### composite_2text_2image

| Recipe | low ΔAUROC | medium ΔAUROC | high ΔAUROC | **mixed** ΔAUROC | high ASR | mixed ASR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.0446 ± 0.0030 | 0.0798 ± 0.0072 | 0.1447 ± 0.0125 | **0.1006 ± 0.0123** | 0.3843 ± 0.0106 | 0.3277 ± 0.0116 |
| augonly | 0.0322 ± 0.0024 | 0.0617 ± 0.0042 | 0.1043 ± 0.0185 | **0.0664 ± 0.0099** | 0.3178 ± 0.0077 | 0.2580 ± 0.0004 |
| kl | 0.0311 ± 0.0019 | 0.0602 ± 0.0046 | 0.1065 ± 0.0024 | **0.0667 ± 0.0059** | 0.3188 ± 0.0161 | 0.2413 ± 0.0028 |
| kldrop | 0.0213 ± 0.0021 | 0.0469 ± 0.0087 | 0.0951 ± 0.0130 | **0.0537 ± 0.0097** | 0.3290 ± 0.0247 | 0.2316 ± 0.0274 |
| kldrop-p015 | 0.0280 ± 0.0034 | 0.0612 ± 0.0074 | 0.1073 ± 0.0039 | **0.0689 ± 0.0019** | 0.3307 ± 0.0294 | 0.2442 ± 0.0021 |
| kldrop-p050 | 0.0223 ± 0.0025 | 0.0460 ± 0.0088 | 0.0913 ± 0.0186 | **0.0538 ± 0.0093** | 0.3269 ± 0.0357 | 0.2424 ± 0.0154 |
| kllowmed | 0.0320 ± 0.0009 | 0.0656 ± 0.0054 | 0.1190 ± 0.0065 | **0.0709 ± 0.0014** | 0.3353 ± 0.0089 | 0.2545 ± 0.0203 |

