# Phase 4 — Worst-case + modality fragility

Answers DoD bullets 5 ("worst-case across families") and 6 ("which modality is most fragile").

## Worst cells — Unknown-only (`robust-augonly-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6337 | +0.0895 | 0.2448 |
| image-pixel | blur (high) | 0.7094 | +0.0139 | 0.1504 |
| image-photometric | brightness_up (high) | 0.7128 | +0.0105 | 0.0841 |
| image-geometric | crop (medium) | 0.7144 | +0.0088 | 0.1357 |

## Worst cells — Unknown-only (`robust-augonly-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6432 | +0.1063 | 0.2457 |
| image-pixel | compression (high) | 0.7281 | +0.0214 | 0.1335 |
| image-photometric | brightness_up (high) | 0.7364 | +0.0131 | 0.1037 |
| image-geometric | occlusion (high) | 0.7401 | +0.0094 | 0.0810 |

## Worst cells — Unknown-only (`robust-augonly-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6208 | +0.0538 | 0.2094 |
| image-pixel | compression (high) | 0.6622 | +0.0124 | 0.1828 |
| image-photometric | brightness_up (high) | 0.6691 | +0.0055 | 0.0953 |
| image-geometric | translation (medium) | 0.6715 | +0.0031 | 0.0938 |

## Worst cells — Unknown-only (`robust-kl-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6482 | +0.0865 | 0.2369 |
| image-pixel | blur (high) | 0.7179 | +0.0168 | 0.1599 |
| image-photometric | brightness_up (high) | 0.7240 | +0.0107 | 0.0988 |
| image-geometric | crop (high) | 0.7247 | +0.0100 | 0.1526 |

## Worst cells — Unknown-only (`robust-kl-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6608 | +0.0865 | 0.2262 |
| image-pixel | compression (high) | 0.7269 | +0.0203 | 0.1196 |
| image-photometric | brightness_up (high) | 0.7325 | +0.0148 | 0.0951 |
| image-geometric | crop (medium) | 0.7375 | +0.0098 | 0.1282 |

## Worst cells — Unknown-only (`robust-kl-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6707 | +0.0836 | 0.2278 |
| image-pixel | blur (high) | 0.7332 | +0.0211 | 0.2149 |
| image-photometric | brightness_up (high) | 0.7418 | +0.0124 | 0.0802 |
| image-geometric | occlusion (high) | 0.7453 | +0.0090 | 0.0759 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6691 | +0.0853 | 0.2303 |
| image-pixel | blur (high) | 0.7201 | +0.0343 | 0.2046 |
| image-photometric | brightness_up (high) | 0.7366 | +0.0178 | 0.0830 |
| image-geometric | occlusion (high) | 0.7427 | +0.0117 | 0.0858 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6516 | +0.0871 | 0.2691 |
| image-pixel | blur (high) | 0.7147 | +0.0240 | 0.1928 |
| image-photometric | brightness_up (high) | 0.7261 | +0.0126 | 0.0806 |
| image-geometric | crop (high) | 0.7259 | +0.0128 | 0.1683 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6238 | +0.0520 | 0.1925 |
| image-pixel | compression (high) | 0.6641 | +0.0117 | 0.1273 |
| image-photometric | brightness_up (high) | 0.6661 | +0.0097 | 0.0792 |
| image-geometric | crop (medium) | 0.6692 | +0.0065 | 0.1382 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_deletion (high) | 0.6559 | +0.0848 | 0.1810 |
| image-pixel | blur (high) | 0.7156 | +0.0251 | 0.1577 |
| image-photometric | brightness_up (high) | 0.7256 | +0.0150 | 0.0934 |
| image-geometric | crop (high) | 0.7233 | +0.0173 | 0.1664 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6628 | +0.0963 | 0.2664 |
| image-pixel | blur (high) | 0.7296 | +0.0295 | 0.2507 |
| image-photometric | brightness_up (high) | 0.7473 | +0.0119 | 0.0940 |
| image-geometric | crop (high) | 0.7416 | +0.0175 | 0.1724 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6650 | +0.0788 | 0.2153 |
| image-pixel | blur (high) | 0.7132 | +0.0305 | 0.2168 |
| image-photometric | brightness_up (high) | 0.7334 | +0.0104 | 0.0853 |
| image-geometric | crop (high) | 0.7317 | +0.0121 | 0.1691 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_deletion (high) | 0.6560 | +0.0835 | 0.1886 |
| image-pixel | blur (high) | 0.7171 | +0.0224 | 0.1740 |
| image-photometric | brightness_up (high) | 0.7262 | +0.0134 | 0.0921 |
| image-geometric | crop (high) | 0.7248 | +0.0148 | 0.1740 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6512 | +0.0840 | 0.2485 |
| image-pixel | blur (medium) | 0.7123 | +0.0229 | 0.1337 |
| image-photometric | brightness_up (high) | 0.7229 | +0.0123 | 0.0930 |
| image-geometric | crop (medium) | 0.7213 | +0.0139 | 0.1410 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6615 | +0.0794 | 0.2221 |
| image-pixel | blur (high) | 0.7130 | +0.0279 | 0.2135 |
| image-photometric | brightness_up (high) | 0.7319 | +0.0089 | 0.0817 |
| image-geometric | crop (medium) | 0.7284 | +0.0125 | 0.1504 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6564 | +0.0807 | 0.1873 |
| image-pixel | blur (high) | 0.7165 | +0.0205 | 0.1608 |
| image-photometric | brightness_up (high) | 0.7254 | +0.0117 | 0.0855 |
| image-geometric | crop (high) | 0.7186 | +0.0184 | 0.1681 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6628 | +0.0901 | 0.2151 |
| image-pixel | blur (high) | 0.7182 | +0.0348 | 0.2806 |
| image-photometric | brightness_up (high) | 0.7386 | +0.0144 | 0.0883 |
| image-geometric | crop (high) | 0.7376 | +0.0154 | 0.1809 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6724 | +0.0781 | 0.2152 |
| image-pixel | compression (high) | 0.7272 | +0.0233 | 0.1392 |
| image-photometric | contrast_up (high) | 0.7426 | +0.0079 | 0.0603 |
| image-geometric | crop (medium) | 0.7376 | +0.0129 | 0.1549 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6640 | +0.0765 | 0.2009 |
| image-pixel | blur (high) | 0.7044 | +0.0361 | 0.2434 |
| image-photometric | brightness_up (high) | 0.7233 | +0.0172 | 0.0953 |
| image-geometric | crop (high) | 0.7281 | +0.0124 | 0.2038 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6573 | +0.0844 | 0.2075 |
| image-pixel | blur (high) | 0.7107 | +0.0309 | 0.2695 |
| image-photometric | brightness_up (high) | 0.7279 | +0.0138 | 0.0980 |
| image-geometric | crop (high) | 0.7258 | +0.0158 | 0.1960 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6275 | +0.0420 | 0.1519 |
| image-pixel | compression (high) | 0.6527 | +0.0167 | 0.1456 |
| image-photometric | brightness_up (high) | 0.6598 | +0.0096 | 0.0886 |
| image-geometric | translation (medium) | 0.6582 | +0.0113 | 0.0918 |

## Worst cells — Unknown-only (`robust-kldrop-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6547 | +0.0801 | 0.1905 |
| image-pixel | blur (high) | 0.7076 | +0.0272 | 0.1758 |
| image-photometric | brightness_up (high) | 0.7218 | +0.0129 | 0.0871 |
| image-geometric | crop (high) | 0.7154 | +0.0194 | 0.1566 |

## Worst cells — Unknown-only (`robust-kldrop-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6613 | +0.0892 | 0.2045 |
| image-pixel | blur (high) | 0.7169 | +0.0336 | 0.2429 |
| image-photometric | brightness_up (high) | 0.7366 | +0.0139 | 0.0866 |
| image-geometric | crop (high) | 0.7364 | +0.0141 | 0.1761 |

## Worst cells — Unknown-only (`robust-kldrop-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | censoring (high) | 0.6266 | +0.0513 | 0.1906 |
| image-pixel | compression (high) | 0.6637 | +0.0142 | 0.1406 |
| image-photometric | brightness_up (high) | 0.6685 | +0.0095 | 0.0859 |
| image-geometric | translation (medium) | 0.6711 | +0.0069 | 0.1047 |

## Worst cells — Unknown-only (`robust-kllowmed-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6487 | +0.0886 | 0.2515 |
| image-pixel | blur (high) | 0.7122 | +0.0251 | 0.1599 |
| image-photometric | brightness_up (high) | 0.7251 | +0.0123 | 0.0959 |
| image-geometric | crop (medium) | 0.7275 | +0.0098 | 0.1294 |

## Worst cells — Unknown-only (`robust-kllowmed-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6543 | +0.0961 | 0.2496 |
| image-pixel | blur (high) | 0.7284 | +0.0219 | 0.1546 |
| image-photometric | brightness_up (high) | 0.7385 | +0.0119 | 0.0823 |
| image-geometric | crop (medium) | 0.7408 | +0.0095 | 0.1319 |

## Worst cells — Unknown-only (`robust-kllowmed-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6647 | +0.0808 | 0.2341 |
| image-pixel | blur (high) | 0.7194 | +0.0261 | 0.1936 |
| image-photometric | brightness_up (high) | 0.7325 | +0.0130 | 0.0838 |
| image-geometric | crop (high) | 0.7298 | +0.0157 | 0.2124 |

## Worst cells — Multimodal (seed 0)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6231 | +0.1311 | 0.3484 |
| image-pixel | blur (high) | 0.7141 | +0.0402 | 0.1636 |
| image-photometric | brightness_up (high) | 0.7381 | +0.0161 | 0.0903 |
| image-geometric | crop (high) | 0.7303 | +0.0240 | 0.1975 |

## Per-modality fragility — average ΔAUROC at high severity

Average drop in AUROC (clean − attacked) across high-severity cells of each family.
A larger drop ⇒ more fragile on that attack family. "n/a" means the model has no
input modality the attack family touches (e.g. text-only model under image attacks).

| Model | Text high-sev | Image-pixel high-sev | Image-photometric high-sev | Image-geometric high-sev |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | +0.0887 | +0.0210 | +0.0098 | +0.0072 |
| robust-augonly-seed0 | +0.0623 | +0.0114 | +0.0063 | +0.0025 |
| robust-augonly-seed1 | +0.0755 | +0.0166 | +0.0084 | +0.0047 |
| robust-augonly-seed2 | +0.0349 | +0.0067 | +0.0003 | -0.0043 |
| robust-kl-seed0 | +0.0609 | +0.0125 | +0.0066 | +0.0046 |
| robust-kl-seed1 | +0.0623 | +0.0170 | +0.0093 | +0.0033 |
| robust-kl-seed2 | +0.0610 | +0.0157 | +0.0088 | +0.0040 |
| robust-kldrop-p010-seed0 | +0.0605 | +0.0220 | +0.0107 | +0.0071 |
| robust-kldrop-p010-seed1 | +0.0589 | +0.0186 | +0.0074 | +0.0076 |
| robust-kldrop-p010-seed2 | +0.0337 | +0.0088 | +0.0035 | -0.0000 |
| robust-kldrop-p015-seed0 | +0.0610 | +0.0175 | +0.0074 | +0.0092 |
| robust-kldrop-p015-seed1 | +0.0658 | +0.0198 | +0.0076 | +0.0095 |
| robust-kldrop-p015-seed2 | +0.0565 | +0.0203 | +0.0076 | +0.0063 |
| robust-kldrop-p020-seed0 | +0.0603 | +0.0161 | +0.0064 | +0.0071 |
| robust-kldrop-p020-seed1 | +0.0566 | +0.0174 | +0.0069 | +0.0083 |
| robust-kldrop-p020-seed2 | +0.0577 | +0.0194 | +0.0073 | +0.0052 |
| robust-kldrop-p025-seed0 | +0.0573 | +0.0156 | +0.0052 | +0.0092 |
| robust-kldrop-p025-seed1 | +0.0628 | +0.0195 | +0.0089 | +0.0075 |
| robust-kldrop-p025-seed2 | +0.0568 | +0.0179 | +0.0056 | +0.0035 |
| robust-kldrop-p050-seed0 | +0.0537 | +0.0252 | +0.0097 | +0.0084 |
| robust-kldrop-p050-seed1 | +0.0578 | +0.0202 | +0.0074 | +0.0097 |
| robust-kldrop-p050-seed2 | +0.0214 | +0.0102 | +0.0046 | +0.0002 |
| robust-kldrop-seed0 | +0.0560 | +0.0192 | +0.0071 | +0.0102 |
| robust-kldrop-seed1 | +0.0623 | +0.0199 | +0.0081 | +0.0076 |
| robust-kldrop-seed2 | +0.0307 | +0.0087 | +0.0038 | +0.0001 |
| robust-kllowmed-seed0 | +0.0622 | +0.0176 | +0.0082 | +0.0045 |
| robust-kllowmed-seed1 | +0.0700 | +0.0186 | +0.0088 | +0.0026 |
| robust-kllowmed-seed2 | +0.0594 | +0.0194 | +0.0105 | +0.0076 |

## Per-modality fragility — white-box PGD ε=4/255

| Model | Clean AUROC | PGD AUROC | ΔAUROC | ASR |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | 0.7488 ± 0.0042 | 0.0000 | 0.7488 ± 0.0042 | 0.9966 ± 0.0025 |
| robust-augonly-seed0 | 0.7242 | 0.0000 | 0.7242 | 1.0000 |
| robust-augonly-seed1 | 0.7434 | 0.0000 | 0.7434 | 0.9956 |
| robust-augonly-seed2 | 0.6744 | 0.0000 | 0.6744 | 0.9952 |
| robust-kl-seed0 | 0.7345 | 0.0000 | 0.7345 | 0.9971 |
| robust-kl-seed1 | 0.7438 | 0.0000 | 0.7438 | 0.9899 |
| robust-kl-seed2 | 0.7592 | 0.0000 | 0.7592 | 1.0000 |
| robust-kldrop-p010-seed0 | 0.7542 | 0.0000 | 0.7542 | 1.0000 |
| robust-kldrop-p010-seed1 | 0.7356 | 0.0000 | 0.7356 | 0.9971 |
| robust-kldrop-p010-seed2 | 0.6737 | 0.0000 | 0.6737 | 0.9969 |
| robust-kldrop-p015-seed0 | 0.7327 | 0.0000 | 0.7327 | 0.9970 |
| robust-kldrop-p015-seed1 | 0.7565 | 0.0000 | 0.7565 | 1.0000 |
| robust-kldrop-p015-seed2 | 0.7454 | 0.0000 | 0.7454 | 1.0000 |
| robust-kldrop-p020-seed0 | 0.7357 | 0.0000 | 0.7357 | 0.9971 |
| robust-kldrop-p020-seed1 | 0.7306 | 0.0000 | 0.7306 | 0.9971 |
| robust-kldrop-p020-seed2 | 0.7416 | 0.0000 | 0.7416 | 1.0000 |
| robust-kldrop-p025-seed0 | 0.7316 | 0.0000 | 0.7316 | 0.9955 |
| robust-kldrop-p025-seed1 | 0.7480 | 0.0000 | 0.7480 | 0.9986 |
| robust-kldrop-p025-seed2 | 0.7494 | 0.0000 | 0.7494 | 0.9986 |
| robust-kldrop-p050-seed0 | 0.7333 | 0.0000 | 0.7333 | 0.9985 |
| robust-kldrop-p050-seed1 | 0.7319 | 0.0000 | 0.7319 | 1.0000 |
| robust-kldrop-p050-seed2 | 0.6644 | 0.0000 | 0.6644 | 0.9968 |
| robust-kldrop-seed0 | 0.7270 | 0.0000 | 0.7270 | 0.9940 |
| robust-kldrop-seed1 | 0.7455 | 0.0000 | 0.7455 | 1.0000 |
| robust-kldrop-seed2 | 0.6759 | 0.0000 | 0.6759 | 0.9984 |
| robust-kllowmed-seed0 | 0.7377 | 0.0000 | 0.7377 | 0.9971 |
| robust-kllowmed-seed1 | 0.7497 | 0.0000 | 0.7497 | 0.9971 |
| robust-kllowmed-seed2 | 0.7507 | 0.0000 | 0.7507 | 0.9971 |

