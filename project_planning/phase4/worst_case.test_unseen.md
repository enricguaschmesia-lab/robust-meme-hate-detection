# Phase 4 — Worst-case + modality fragility

Answers DoD bullets 5 ("worst-case across families") and 6 ("which modality is most fragile").

## Worst cells — Unknown-only (`robust-augonly-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6510 | +0.0715 | 0.2350 |
| image-pixel | blur (high) | 0.6991 | +0.0234 | 0.1662 |
| image-photometric | brightness_up (high) | 0.7061 | +0.0164 | 0.0835 |
| image-geometric | crop (medium) | 0.7073 | +0.0152 | 0.1325 |

## Worst cells — Unknown-only (`robust-augonly-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6553 | +0.0780 | 0.2361 |
| image-pixel | blur (high) | 0.7106 | +0.0227 | 0.1637 |
| image-photometric | brightness_up (high) | 0.7161 | +0.0172 | 0.0943 |
| image-geometric | occlusion (high) | 0.7223 | +0.0111 | 0.0775 |

## Worst cells — Unknown-only (`robust-augonly-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_swap (high) | 0.6294 | +0.0395 | 0.2098 |
| image-pixel | blur (high) | 0.6513 | +0.0176 | 0.3317 |
| image-photometric | brightness_up (high) | 0.6595 | +0.0094 | 0.1188 |
| image-geometric | crop (medium) | 0.6555 | +0.0134 | 0.1916 |

## Worst cells — Unknown-only (`robust-kl-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6648 | +0.0619 | 0.2239 |
| image-pixel | blur (high) | 0.6999 | +0.0268 | 0.1597 |
| image-photometric | brightness_up (high) | 0.7100 | +0.0167 | 0.0853 |
| image-geometric | crop (medium) | 0.7119 | +0.0148 | 0.1349 |

## Worst cells — Unknown-only (`robust-kl-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6729 | +0.0636 | 0.2018 |
| image-pixel | blur (high) | 0.7113 | +0.0253 | 0.1652 |
| image-photometric | brightness_up (high) | 0.7218 | +0.0147 | 0.0855 |
| image-geometric | crop (medium) | 0.7217 | +0.0149 | 0.1250 |

## Worst cells — Unknown-only (`robust-kl-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6815 | +0.0704 | 0.2034 |
| image-pixel | blur (high) | 0.7262 | +0.0257 | 0.2275 |
| image-photometric | brightness_up (high) | 0.7354 | +0.0165 | 0.0723 |
| image-geometric | occlusion (high) | 0.7421 | +0.0098 | 0.0829 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6903 | +0.0666 | 0.2098 |
| image-pixel | blur (high) | 0.7171 | +0.0398 | 0.2069 |
| image-photometric | brightness_up (high) | 0.7330 | +0.0238 | 0.0893 |
| image-geometric | crop (medium) | 0.7423 | +0.0146 | 0.1361 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6684 | +0.0611 | 0.2586 |
| image-pixel | blur (high) | 0.6993 | +0.0301 | 0.2243 |
| image-photometric | brightness_up (high) | 0.7121 | +0.0173 | 0.1078 |
| image-geometric | crop (medium) | 0.7120 | +0.0174 | 0.1420 |

## Worst cells — Unknown-only (`robust-kldrop-p010-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_swap (high) | 0.6256 | +0.0371 | 0.1949 |
| image-pixel | blur (high) | 0.6359 | +0.0268 | 0.2363 |
| image-photometric | brightness_up (high) | 0.6552 | +0.0075 | 0.0971 |
| image-geometric | translation (medium) | 0.6538 | +0.0089 | 0.1114 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6598 | +0.0742 | 0.1864 |
| image-pixel | blur (high) | 0.7015 | +0.0325 | 0.1525 |
| image-photometric | brightness_up (high) | 0.7112 | +0.0227 | 0.0840 |
| image-geometric | crop (medium) | 0.7172 | +0.0168 | 0.1130 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6779 | +0.0778 | 0.2444 |
| image-pixel | blur (high) | 0.7227 | +0.0330 | 0.2486 |
| image-photometric | brightness_up (high) | 0.7362 | +0.0195 | 0.1035 |
| image-geometric | crop (medium) | 0.7394 | +0.0163 | 0.1338 |

## Worst cells — Unknown-only (`robust-kldrop-p015-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6843 | +0.0557 | 0.1618 |
| image-pixel | blur (high) | 0.7077 | +0.0323 | 0.1959 |
| image-photometric | brightness_up (high) | 0.7252 | +0.0147 | 0.0767 |
| image-geometric | crop (medium) | 0.7275 | +0.0124 | 0.1285 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6596 | +0.0706 | 0.2208 |
| image-pixel | blur (high) | 0.6987 | +0.0316 | 0.1895 |
| image-photometric | brightness_up (high) | 0.7081 | +0.0222 | 0.0882 |
| image-geometric | crop (medium) | 0.7146 | +0.0157 | 0.1385 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6641 | +0.0586 | 0.2347 |
| image-pixel | blur (high) | 0.6912 | +0.0315 | 0.2115 |
| image-photometric | brightness_up (high) | 0.7051 | +0.0175 | 0.0981 |
| image-geometric | crop (medium) | 0.7057 | +0.0170 | 0.1359 |

## Worst cells — Unknown-only (`robust-kldrop-p020-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6810 | +0.0541 | 0.1730 |
| image-pixel | blur (high) | 0.7059 | +0.0291 | 0.1989 |
| image-photometric | brightness_up (high) | 0.7212 | +0.0138 | 0.0682 |
| image-geometric | crop (medium) | 0.7227 | +0.0124 | 0.1436 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6628 | +0.0674 | 0.2001 |
| image-pixel | blur (high) | 0.6974 | +0.0328 | 0.1704 |
| image-photometric | brightness_up (high) | 0.7079 | +0.0223 | 0.0870 |
| image-geometric | crop (medium) | 0.7132 | +0.0170 | 0.1334 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6833 | +0.0661 | 0.2164 |
| image-pixel | blur (high) | 0.7128 | +0.0366 | 0.2982 |
| image-photometric | brightness_up (high) | 0.7269 | +0.0225 | 0.1018 |
| image-geometric | crop (medium) | 0.7319 | +0.0175 | 0.1516 |

## Worst cells — Unknown-only (`robust-kldrop-p025-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6859 | +0.0562 | 0.1745 |
| image-pixel | blur (high) | 0.7160 | +0.0261 | 0.1773 |
| image-photometric | brightness_up (high) | 0.7249 | +0.0173 | 0.0778 |
| image-geometric | crop (medium) | 0.7284 | +0.0138 | 0.1359 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6755 | +0.0591 | 0.2130 |
| image-pixel | blur (high) | 0.6964 | +0.0382 | 0.2592 |
| image-photometric | brightness_up (high) | 0.7151 | +0.0195 | 0.1047 |
| image-geometric | crop (medium) | 0.7209 | +0.0137 | 0.1617 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6750 | +0.0618 | 0.2272 |
| image-pixel | blur (high) | 0.6950 | +0.0417 | 0.3120 |
| image-photometric | brightness_up (high) | 0.7143 | +0.0225 | 0.1057 |
| image-geometric | crop (medium) | 0.7181 | +0.0186 | 0.1546 |

## Worst cells — Unknown-only (`robust-kldrop-p050-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_swap (high) | 0.6391 | +0.0249 | 0.1362 |
| image-pixel | blur (high) | 0.6333 | +0.0308 | 0.2431 |
| image-photometric | brightness_up (high) | 0.6554 | +0.0086 | 0.0990 |
| image-geometric | translation (high) | 0.6488 | +0.0152 | 0.1211 |

## Worst cells — Unknown-only (`robust-kldrop-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6610 | +0.0637 | 0.1994 |
| image-pixel | blur (high) | 0.6907 | +0.0340 | 0.1871 |
| image-photometric | brightness_up (high) | 0.7028 | +0.0219 | 0.0921 |
| image-geometric | crop (medium) | 0.7072 | +0.0175 | 0.1501 |

## Worst cells — Unknown-only (`robust-kldrop-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6784 | +0.0675 | 0.2011 |
| image-pixel | blur (high) | 0.7119 | +0.0341 | 0.2757 |
| image-photometric | brightness_up (high) | 0.7246 | +0.0214 | 0.0970 |
| image-geometric | crop (medium) | 0.7296 | +0.0164 | 0.1428 |

## Worst cells — Unknown-only (`robust-kldrop-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | char_swap (high) | 0.6343 | +0.0329 | 0.1599 |
| image-pixel | blur (high) | 0.6394 | +0.0278 | 0.2534 |
| image-photometric | brightness_up (high) | 0.6587 | +0.0085 | 0.0927 |
| image-geometric | translation (high) | 0.6551 | +0.0120 | 0.1239 |

## Worst cells — Unknown-only (`robust-kllowmed-seed0`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6649 | +0.0640 | 0.2176 |
| image-pixel | blur (high) | 0.6981 | +0.0308 | 0.1448 |
| image-photometric | brightness_up (high) | 0.7102 | +0.0188 | 0.0735 |
| image-geometric | crop (medium) | 0.7150 | +0.0140 | 0.1225 |

## Worst cells — Unknown-only (`robust-kllowmed-seed1`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6638 | +0.0735 | 0.2185 |
| image-pixel | blur (high) | 0.7076 | +0.0297 | 0.1526 |
| image-photometric | brightness_up (high) | 0.7221 | +0.0152 | 0.0845 |
| image-geometric | crop (medium) | 0.7240 | +0.0133 | 0.1196 |

## Worst cells — Unknown-only (`robust-kllowmed-seed2`)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6746 | +0.0658 | 0.2016 |
| image-pixel | blur (high) | 0.7087 | +0.0316 | 0.1916 |
| image-photometric | brightness_up (high) | 0.7212 | +0.0192 | 0.0940 |
| image-geometric | crop (medium) | 0.7273 | +0.0130 | 0.1631 |

## Worst cells — Multimodal (seed 0)

| Family | Worst cell | Attacked AUROC | ΔAUROC | ASR |
|---|---|---:|---:|---:|
| text | leetspeak (high) | 0.6231 | +0.1193 | 0.4149 |
| image-pixel | blur (high) | 0.7066 | +0.0358 | 0.1759 |
| image-photometric | brightness_up (high) | 0.7197 | +0.0226 | 0.0883 |
| image-geometric | crop (medium) | 0.7242 | +0.0182 | 0.1644 |

## Per-modality fragility — average ΔAUROC at high severity

Average drop in AUROC (clean − attacked) across high-severity cells of each family.
A larger drop ⇒ more fragile on that attack family. "n/a" means the model has no
input modality the attack family touches (e.g. text-only model under image attacks).

| Model | Text high-sev | Image-pixel high-sev | Image-photometric high-sev | Image-geometric high-sev |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | +0.0734 | +0.0236 | +0.0110 | +0.0074 |
| robust-augonly-seed0 | +0.0494 | +0.0178 | +0.0093 | +0.0074 |
| robust-augonly-seed1 | +0.0539 | +0.0181 | +0.0090 | +0.0071 |
| robust-augonly-seed2 | +0.0239 | +0.0093 | +0.0030 | +0.0076 |
| robust-kl-seed0 | +0.0442 | +0.0174 | +0.0101 | +0.0066 |
| robust-kl-seed1 | +0.0461 | +0.0177 | +0.0081 | +0.0076 |
| robust-kl-seed2 | +0.0518 | +0.0187 | +0.0093 | +0.0039 |
| robust-kldrop-p010-seed0 | +0.0463 | +0.0243 | +0.0134 | +0.0082 |
| robust-kldrop-p010-seed1 | +0.0431 | +0.0216 | +0.0105 | +0.0098 |
| robust-kldrop-p010-seed2 | +0.0219 | +0.0115 | +0.0038 | +0.0067 |
| robust-kldrop-p015-seed0 | +0.0496 | +0.0215 | +0.0109 | +0.0095 |
| robust-kldrop-p015-seed1 | +0.0532 | +0.0219 | +0.0094 | +0.0103 |
| robust-kldrop-p015-seed2 | +0.0428 | +0.0189 | +0.0088 | +0.0059 |
| robust-kldrop-p020-seed0 | +0.0463 | +0.0207 | +0.0101 | +0.0086 |
| robust-kldrop-p020-seed1 | +0.0406 | +0.0221 | +0.0105 | +0.0098 |
| robust-kldrop-p020-seed2 | +0.0414 | +0.0176 | +0.0077 | +0.0054 |
| robust-kldrop-p025-seed0 | +0.0431 | +0.0225 | +0.0104 | +0.0104 |
| robust-kldrop-p025-seed1 | +0.0470 | +0.0248 | +0.0112 | +0.0112 |
| robust-kldrop-p025-seed2 | +0.0417 | +0.0168 | +0.0088 | +0.0041 |
| robust-kldrop-p050-seed0 | +0.0386 | +0.0239 | +0.0099 | +0.0113 |
| robust-kldrop-p050-seed1 | +0.0423 | +0.0275 | +0.0100 | +0.0121 |
| robust-kldrop-p050-seed2 | +0.0142 | +0.0150 | +0.0053 | +0.0110 |
| robust-kldrop-seed0 | +0.0404 | +0.0231 | +0.0103 | +0.0100 |
| robust-kldrop-seed1 | +0.0473 | +0.0237 | +0.0107 | +0.0108 |
| robust-kldrop-seed2 | +0.0207 | +0.0129 | +0.0051 | +0.0091 |
| robust-kllowmed-seed0 | +0.0454 | +0.0204 | +0.0119 | +0.0066 |
| robust-kllowmed-seed1 | +0.0534 | +0.0202 | +0.0087 | +0.0058 |
| robust-kllowmed-seed2 | +0.0508 | +0.0218 | +0.0118 | +0.0068 |

## Per-modality fragility — white-box PGD ε=4/255

| Model | Clean AUROC | PGD AUROC | ΔAUROC | ASR |
|---|---:|---:|---:|---:|
| multimodal (3-seed mean) | 0.7309 ± 0.0009 | 0.0000 ± 0.0000 | 0.7309 ± 0.0009 | 0.9985 ± 0.0006 |
| robust-augonly-seed0 | 0.7129 | 0.0000 | 0.7129 | 0.9993 |
| robust-augonly-seed1 | 0.7272 | 0.0000 | 0.7272 | 0.9985 |
| robust-augonly-seed2 | 0.6639 | 0.0000 | 0.6639 | 0.9976 |
| robust-kl-seed0 | 0.7195 | 0.0000 | 0.7195 | 0.9985 |
| robust-kl-seed1 | 0.7288 | 0.0000 | 0.7288 | 0.9978 |
| robust-kl-seed2 | 0.7527 | 0.0000 | 0.7527 | 0.9986 |
| robust-kldrop-p010-seed0 | 0.7483 | 0.0000 | 0.7483 | 1.0000 |
| robust-kldrop-p010-seed1 | 0.7202 | 0.0000 | 0.7202 | 0.9993 |
| robust-kldrop-p010-seed2 | 0.6584 | 0.0000 | 0.6584 | 0.9976 |
| robust-kldrop-p015-seed0 | 0.7210 | 0.0000 | 0.7210 | 0.9992 |
| robust-kldrop-p015-seed1 | 0.7507 | 0.0000 | 0.7507 | 1.0000 |
| robust-kldrop-p015-seed2 | 0.7346 | 0.0000 | 0.7346 | 1.0000 |
| robust-kldrop-p020-seed0 | 0.7188 | 0.0000 | 0.7188 | 0.9993 |
| robust-kldrop-p020-seed1 | 0.7126 | 0.0000 | 0.7126 | 0.9993 |
| robust-kldrop-p020-seed2 | 0.7287 | 0.0000 | 0.7287 | 1.0000 |
| robust-kldrop-p025-seed0 | 0.7175 | 0.0000 | 0.7175 | 0.9985 |
| robust-kldrop-p025-seed1 | 0.7403 | 0.0000 | 0.7403 | 1.0000 |
| robust-kldrop-p025-seed2 | 0.7366 | 0.0000 | 0.7366 | 0.9993 |
| robust-kldrop-p050-seed0 | 0.7211 | 0.0000 | 0.7211 | 1.0000 |
| robust-kldrop-p050-seed1 | 0.7248 | 0.0000 | 0.7248 | 1.0000 |
| robust-kldrop-p050-seed2 | 0.6579 | 0.0000 | 0.6579 | 0.9961 |
| robust-kldrop-seed0 | 0.7116 | 0.0000 | 0.7116 | 0.9977 |
| robust-kldrop-seed1 | 0.7370 | 0.0000 | 0.7370 | 0.9993 |
| robust-kldrop-seed2 | 0.6617 | 0.0000 | 0.6617 | 0.9984 |
| robust-kllowmed-seed0 | 0.7209 | 0.0000 | 0.7209 | 0.9978 |
| robust-kllowmed-seed1 | 0.7311 | 0.0000 | 0.7311 | 0.9986 |
| robust-kllowmed-seed2 | 0.7377 | 0.0000 | 0.7377 | 0.9985 |

