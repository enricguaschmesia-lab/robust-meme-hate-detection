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
| robust-kldrop-seed0 | 0.7162 | 0.0000 | 0.7162 | 0.9970 |
| robust-kldrop-seed1 | 0.7238 | 0.0000 | 0.7238 | 1.0000 |
| robust-kldrop-seed2 | 0.6538 | 0.0000 | 0.6538 | 0.9902 |
| robust-kllowmed-seed0 | 0.7206 | 0.0000 | 0.7206 | 0.9970 |
| robust-kllowmed-seed1 | 0.7350 | 0.0000 | 0.7350 | 0.9970 |
| robust-kllowmed-seed2 | 0.7233 | 0.0000 | 0.7233 | 0.9970 |

