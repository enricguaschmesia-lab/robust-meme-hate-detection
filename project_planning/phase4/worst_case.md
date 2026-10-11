# Phase 4 — Worst-case + modality fragility

Answers DoD bullets 5 ("worst-case across families") and 6 ("which modality is most fragile").

## Worst cells — Image-only (`baseline-image-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| image-pixel | blur (high) | 0.5546 | +0.0732 | 0.2886 |
| image-photometric | brightness_up (high) | 0.6171 | +0.0107 | 0.1342 |
| image-geometric | crop (medium) | 0.6150 | +0.0128 | 0.2148 |

## Worst cells — Text-only (`baseline-text-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_deletion (high) | 0.5263 | +0.1058 | 0.4333 |

## Worst cells — Unknown-only (`robust-augonly-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6364 | +0.0866 | 0.2249 |
| image-pixel | blur (high) | 0.6940 | +0.0291 | 0.1775 |
| image-photometric | brightness_up (high) | 0.7102 | +0.0129 | 0.1095 |
| image-geometric | occlusion (high) | 0.7105 | +0.0125 | 0.0710 |

## Worst cells — Unknown-only (`robust-augonly-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6362 | +0.0953 | 0.2463 |
| image-pixel | blur (high) | 0.6979 | +0.0335 | 0.1988 |
| image-photometric | brightness_up (high) | 0.7162 | +0.0153 | 0.1068 |
| image-geometric | occlusion (high) | 0.7076 | +0.0239 | 0.1157 |

## Worst cells — Unknown-only (`robust-augonly-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6159 | +0.0665 | 0.2340 |
| image-pixel | blur (high) | 0.6533 | +0.0291 | 0.2918 |
| image-photometric | brightness_down (high) | 0.6708 | +0.0115 | 0.0912 |
| image-geometric | occlusion (high) | 0.6634 | +0.0189 | 0.0942 |

## Worst cells — Unknown-only (`robust-kl-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6392 | +0.0869 | 0.2203 |
| image-pixel | blur (high) | 0.6899 | +0.0361 | 0.1913 |
| image-photometric | brightness_up (high) | 0.7149 | +0.0112 | 0.0870 |
| image-geometric | occlusion (high) | 0.7125 | +0.0135 | 0.0841 |

## Worst cells — Unknown-only (`robust-kl-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6449 | +0.0953 | 0.2334 |
| image-pixel | blur (high) | 0.6984 | +0.0418 | 0.2104 |
| image-photometric | brightness_up (high) | 0.7265 | +0.0137 | 0.0922 |
| image-geometric | occlusion (high) | 0.7215 | +0.0187 | 0.0807 |

## Worst cells — Unknown-only (`robust-kl-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6565 | +0.0873 | 0.2235 |
| image-pixel | blur (high) | 0.7075 | +0.0363 | 0.2059 |
| image-photometric | contrast_up (high) | 0.7336 | +0.0102 | 0.0529 |
| image-geometric | occlusion (high) | 0.7274 | +0.0164 | 0.0706 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6636 | +0.0755 | 0.2358 |
| image-pixel | blur (high) | 0.6923 | +0.0468 | 0.2017 |
| image-photometric | brightness_up (high) | 0.7257 | +0.0135 | 0.1136 |
| image-geometric | occlusion (high) | 0.7282 | +0.0109 | 0.0795 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6423 | +0.0819 | 0.2441 |
| image-pixel | blur (high) | 0.6812 | +0.0430 | 0.2059 |
| image-photometric | brightness_down (high) | 0.7139 | +0.0103 | 0.0912 |
| image-geometric | occlusion (high) | 0.7084 | +0.0158 | 0.0853 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6141 | +0.0611 | 0.2062 |
| image-pixel | blur (high) | 0.6489 | +0.0262 | 0.2000 |
| image-photometric | brightness_down (high) | 0.6662 | +0.0090 | 0.0969 |
| image-geometric | occlusion (high) | 0.6616 | +0.0136 | 0.0750 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_deletion (high) | 0.6576 | +0.0757 | 0.2135 |
| image-pixel | blur (high) | 0.6888 | +0.0445 | 0.1930 |
| image-photometric | contrast_up (high) | 0.7199 | +0.0134 | 0.0643 |
| image-geometric | occlusion (high) | 0.7233 | +0.0100 | 0.0789 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | punctuation (high) | 0.6525 | +0.0849 | 0.2209 |
| image-pixel | blur (high) | 0.6888 | +0.0485 | 0.2297 |
| image-photometric | contrast_up (high) | 0.7285 | +0.0088 | 0.0640 |
| image-geometric | occlusion (high) | 0.7240 | +0.0133 | 0.0843 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6515 | +0.0814 | 0.1901 |
| image-pixel | blur (high) | 0.6891 | +0.0439 | 0.2105 |
| image-photometric | brightness_up (high) | 0.7242 | +0.0088 | 0.0936 |
| image-geometric | occlusion (high) | 0.7148 | +0.0182 | 0.1140 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6540 | +0.0755 | 0.2315 |
| image-pixel | blur (high) | 0.6822 | +0.0473 | 0.2018 |
| image-photometric | brightness_up (high) | 0.7156 | +0.0140 | 0.1128 |
| image-geometric | occlusion (high) | 0.7200 | +0.0095 | 0.0712 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6517 | +0.0671 | 0.2372 |
| image-pixel | blur (high) | 0.6763 | +0.0425 | 0.2222 |
| image-photometric | brightness_up (high) | 0.7093 | +0.0095 | 0.0991 |
| image-geometric | occlusion (high) | 0.7045 | +0.0143 | 0.0841 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6486 | +0.0865 | 0.1988 |
| image-pixel | blur (high) | 0.6964 | +0.0386 | 0.2161 |
| image-photometric | contrast_up (high) | 0.7250 | +0.0101 | 0.0576 |
| image-geometric | occlusion (high) | 0.7171 | +0.0180 | 0.1066 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | punctuation (high) | 0.6541 | +0.0768 | 0.1958 |
| image-pixel | blur (high) | 0.6871 | +0.0438 | 0.2552 |
| image-photometric | brightness_up (high) | 0.7230 | +0.0080 | 0.1009 |
| image-geometric | occlusion (high) | 0.7167 | +0.0143 | 0.0861 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6568 | +0.0772 | 0.2265 |
| image-pixel | blur (high) | 0.6976 | +0.0364 | 0.2294 |
| image-photometric | contrast_up (high) | 0.7219 | +0.0121 | 0.0676 |
| image-geometric | occlusion (high) | 0.7204 | +0.0136 | 0.0912 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6527 | +0.0700 | 0.2344 |
| image-pixel | blur (high) | 0.6751 | +0.0475 | 0.2315 |
| image-photometric | brightness_up (high) | 0.7119 | +0.0108 | 0.0979 |
| image-geometric | occlusion (high) | 0.7095 | +0.0132 | 0.0831 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | punctuation (high) | 0.6459 | +0.0761 | 0.2202 |
| image-pixel | blur (high) | 0.6775 | +0.0445 | 0.2738 |
| image-photometric | brightness_up (high) | 0.7137 | +0.0083 | 0.0982 |
| image-geometric | occlusion (high) | 0.7073 | +0.0147 | 0.0714 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (medium) | 0.6061 | +0.0500 | 0.1736 |
| image-pixel | blur (high) | 0.6243 | +0.0318 | 0.2412 |
| image-photometric | brightness_down (high) | 0.6506 | +0.0054 | 0.0997 |
| image-geometric | occlusion (high) | 0.6471 | +0.0090 | 0.0740 |

## Worst cells — Unknown-only (`robust-kldrop-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_deletion (high) | 0.6497 | +0.0710 | 0.1772 |
| image-pixel | blur (high) | 0.6764 | +0.0443 | 0.2132 |
| image-photometric | brightness_up (high) | 0.7089 | +0.0119 | 0.1141 |
| image-geometric | occlusion (high) | 0.7077 | +0.0130 | 0.0781 |

## Worst cells — Unknown-only (`robust-kldrop-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | punctuation (high) | 0.6502 | +0.0752 | 0.2173 |
| image-pixel | blur (high) | 0.6888 | +0.0366 | 0.2530 |
| image-photometric | brightness_down (high) | 0.7150 | +0.0103 | 0.1012 |
| image-geometric | occlusion (high) | 0.7104 | +0.0150 | 0.0833 |

## Worst cells — Unknown-only (`robust-kldrop-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (medium) | 0.6134 | +0.0559 | 0.1956 |
| image-pixel | blur (high) | 0.6422 | +0.0270 | 0.2334 |
| image-photometric | brightness_down (high) | 0.6605 | +0.0087 | 0.0820 |
| image-geometric | occlusion (high) | 0.6563 | +0.0129 | 0.0726 |

## Worst cells — Unknown-only (`robust-kllowmed-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6459 | +0.0826 | 0.2406 |
| image-pixel | blur (high) | 0.6932 | +0.0352 | 0.1884 |
| image-photometric | brightness_up (high) | 0.7167 | +0.0117 | 0.0783 |
| image-geometric | occlusion (high) | 0.7146 | +0.0138 | 0.0725 |

## Worst cells — Unknown-only (`robust-kllowmed-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6370 | +0.1016 | 0.2356 |
| image-pixel | blur (high) | 0.6973 | +0.0413 | 0.1782 |
| image-photometric | brightness_up (high) | 0.7278 | +0.0108 | 0.0977 |
| image-geometric | occlusion (high) | 0.7184 | +0.0202 | 0.0862 |

## Worst cells — Unknown-only (`robust-kllowmed-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6437 | +0.0935 | 0.2615 |
| image-pixel | blur (high) | 0.6891 | +0.0481 | 0.1897 |
| image-photometric | brightness_up (high) | 0.7284 | +0.0088 | 0.0805 |
| image-geometric | occlusion (high) | 0.7199 | +0.0172 | 0.0747 |

## Worst cells — Multimodal (seed 0)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6272 | +0.1221 | 0.3162 |
| image-pixel | blur (high) | 0.7095 | +0.0397 | 0.1966 |
| image-photometric | brightness_up (high) | 0.7259 | +0.0233 | 0.1083 |
| image-geometric | crop (low) | 0.7324 | +0.0169 | 0.1111 |

## Worst cells — Image-only (`typographic-baseline-image-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|

## Worst cells — Text-only (`typographic-baseline-text-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|

## Worst cells — Unknown-only (`typographic-stage1-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|

## Worst cells — Unknown-only (`typographic-stage1-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|

## Worst cells — Unknown-only (`typographic-stage1-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|

## Per-modality fragility — average ΔAUROC at high severity

Average drop in AUROC (clean − attacked) across high-severity cells of each family.
A larger drop ⇒ more fragile on that attack family. "n/a" means the model has no
input modality the attack family touches (e.g. text-only model under image attacks).

| Model | Text high-sev | Image-pixel high-sev | Image-photometric high-sev | Image-geometric high-sev |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | +0.0854 | +0.0247 | +0.0090 | +0.0058 |
| baseline-image-seed0 | n/a | +0.0258 | +0.0048 | +0.0102 |
| baseline-text-seed0 | +0.0651 | n/a | n/a | n/a |
| robust-augonly-seed0 | +0.0618 | +0.0212 | +0.0064 | +0.0017 |
| robust-augonly-seed1 | +0.0687 | +0.0246 | +0.0104 | +0.0091 |
| robust-augonly-seed2 | +0.0456 | +0.0158 | +0.0062 | +0.0054 |
| robust-kl-seed0 | +0.0602 | +0.0214 | +0.0064 | +0.0066 |
| robust-kl-seed1 | +0.0670 | +0.0247 | +0.0085 | +0.0071 |
| robust-kl-seed2 | +0.0579 | +0.0249 | +0.0054 | +0.0010 |
| robust-kldrop-p010-seed0 | +0.0551 | +0.0291 | +0.0075 | -0.0034 |
| robust-kldrop-p010-seed1 | +0.0592 | +0.0255 | +0.0080 | +0.0024 |
| robust-kldrop-p010-seed2 | +0.0383 | +0.0097 | +0.0032 | -0.0002 |
| robust-kldrop-p015-seed0 | +0.0543 | +0.0287 | +0.0072 | +0.0028 |
| robust-kldrop-p015-seed1 | +0.0625 | +0.0293 | +0.0071 | -0.0016 |
| robust-kldrop-p015-seed2 | +0.0584 | +0.0242 | +0.0051 | +0.0025 |
| robust-kldrop-p020-seed0 | +0.0546 | +0.0296 | +0.0066 | +0.0015 |
| robust-kldrop-p020-seed1 | +0.0489 | +0.0231 | +0.0061 | +0.0010 |
| robust-kldrop-p020-seed2 | +0.0598 | +0.0225 | +0.0063 | +0.0023 |
| robust-kldrop-p025-seed1 | +0.0553 | +0.0264 | +0.0058 | -0.0010 |
| robust-kldrop-p025-seed2 | +0.0559 | +0.0216 | +0.0079 | +0.0001 |
| robust-kldrop-p050-seed0 | +0.0480 | +0.0288 | +0.0070 | +0.0000 |
| robust-kldrop-p050-seed1 | +0.0554 | +0.0212 | +0.0044 | -0.0021 |
| robust-kldrop-p050-seed2 | +0.0295 | +0.0107 | -0.0000 | -0.0047 |
| robust-kldrop-seed0 | +0.0510 | +0.0274 | +0.0046 | +0.0010 |
| robust-kldrop-seed1 | +0.0537 | +0.0226 | +0.0072 | -0.0008 |
| robust-kldrop-seed2 | +0.0365 | +0.0104 | +0.0031 | -0.0002 |
| robust-kllowmed-seed0 | +0.0592 | +0.0221 | +0.0070 | +0.0049 |
| robust-kllowmed-seed1 | +0.0707 | +0.0244 | +0.0062 | +0.0055 |
| robust-kllowmed-seed2 | +0.0653 | +0.0272 | +0.0060 | +0.0063 |
| typographic-baseline-image-seed0 | n/a | n/a | n/a | n/a |
| typographic-baseline-text-seed0 | n/a | n/a | n/a | n/a |
| typographic-stage1-seed0 | n/a | n/a | n/a | n/a |
| typographic-stage1-seed1 | n/a | n/a | n/a | n/a |
| typographic-stage1-seed2 | n/a | n/a | n/a | n/a |

## Per-modality fragility — white-box PGD ε=4/255

| Model | Clean AUROC | PGD AUROC | ΔAUROC | ASR |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | 0.7338 ± 0.0074 | 0.0000 | 0.7338 ± 0.0074 | 0.9990 ± 0.0014 |
| baseline-image-seed0 | 0.6232 | 0.0000 | 0.6232 | 1.0000 |
| robust-augonly-seed0 | 0.7154 | 0.0000 | 0.7154 | 0.9970 |
| robust-augonly-seed1 | 0.7198 | 0.0000 | 0.7198 | 0.9970 |
| robust-augonly-seed2 | 0.6597 | 0.0000 | 0.6597 | 0.9904 |
| robust-kl-seed0 | 0.7158 | 0.0000 | 0.7158 | 0.9970 |
| robust-kl-seed1 | 0.7364 | 0.0000 | 0.7364 | 0.9970 |
| robust-kl-seed2 | 0.7355 | 0.0000 | 0.7355 | 0.9970 |
| robust-kldrop-p010-seed0 | 0.7392 | 0.0000 | 0.7392 | 0.9942 |
| robust-kldrop-p010-seed1 | 0.7156 | 0.0000 | 0.7156 | 0.9940 |
| robust-kldrop-p010-seed2 | 0.6602 | 0.0000 | 0.6602 | 0.9903 |
| robust-kldrop-p015-seed0 | 0.7297 | 0.0000 | 0.7297 | 1.0000 |
| robust-kldrop-p015-seed1 | 0.7355 | 0.0000 | 0.7355 | 1.0000 |
| robust-kldrop-p015-seed2 | 0.7325 | 0.0000 | 0.7325 | 1.0000 |
| robust-kldrop-p020-seed0 | 0.7250 | 0.0000 | 0.7250 | 1.0000 |
| robust-kldrop-p020-seed1 | 0.7096 | 0.0000 | 0.7096 | 0.9970 |
| robust-kldrop-p020-seed2 | 0.7356 | 0.0000 | 0.7356 | 1.0000 |
| robust-kldrop-p025-seed0 | 0.7216 | 0.0000 | 0.7216 | 0.9970 |
| robust-kldrop-p025-seed1 | 0.7270 | 0.0000 | 0.7270 | 1.0000 |
| robust-kldrop-p025-seed2 | 0.7325 | 0.0000 | 0.7325 | 1.0000 |
| robust-kldrop-p050-seed0 | 0.7164 | 0.0000 | 0.7164 | 0.9940 |
| robust-kldrop-p050-seed1 | 0.7142 | 0.0000 | 0.7142 | 0.9970 |
| robust-kldrop-p050-seed2 | 0.6439 | 0.0000 | 0.6439 | 0.9966 |
| robust-kldrop-seed0 | 0.7162 | 0.0000 | 0.7162 | 0.9970 |
| robust-kldrop-seed1 | 0.7238 | 0.0000 | 0.7238 | 1.0000 |
| robust-kldrop-seed2 | 0.6538 | 0.0000 | 0.6538 | 0.9902 |
| robust-kllowmed-seed0 | 0.7206 | 0.0000 | 0.7206 | 0.9970 |
| robust-kllowmed-seed1 | 0.7350 | 0.0000 | 0.7350 | 0.9970 |
| robust-kllowmed-seed2 | 0.7233 | 0.0000 | 0.7233 | 0.9970 |

