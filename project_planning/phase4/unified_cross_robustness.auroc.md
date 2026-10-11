# Unified robust models — cross-robustness comparison (AUROC)

Three models trained with `train.stage1_robust_unified` (seeds 0/1/2, mean±std). `nat` = naturalistic only (frozen encoders); `adv` = PGD adversarial (vision unfrozen); `both` = combined.

- **Clean** = clean AUROC (best threshold).
- **Nat** = mean attacked AUROC over all naturalistic cells; **Nat-ASR** = mean attack-success-rate.
- **PGD@k** = AUROC under white-box PGD at ε=k/255; **PGD8-ASR** = ASR at ε=8/255.


## Split: `dev`

| Model | Clean | Nat | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.705±0.025 | 0.689±0.023 | 0.099±0.003 | 0.018±0.003 | 0.001±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.624±0.005 | 0.607±0.005 | 0.100±0.006 | 0.622±0.007 | 0.621±0.007 | 0.620±0.008 | 0.617±0.010 | 0.010±0.007 |
| `both` (both) | 0.623±0.001 | 0.609±0.002 | 0.088±0.003 | 0.622±0.001 | 0.622±0.001 | 0.621±0.001 | 0.619±0.001 | 0.000±0.000 |

## Split: `test_seen`

| Model | Clean | Nat | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.721±0.031 | 0.704±0.026 | 0.092±0.005 | 0.015±0.001 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.652±0.009 | 0.632±0.006 | 0.099±0.005 | 0.651±0.010 | 0.650±0.010 | 0.649±0.011 | 0.647±0.012 | 0.008±0.005 |
| `both` (both) | 0.652±0.008 | 0.634±0.006 | 0.089±0.005 | 0.651±0.008 | 0.650±0.008 | 0.650±0.008 | 0.648±0.008 | 0.005±0.003 |

## Split: `test_unseen`

| Model | Clean | Nat | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.713±0.033 | 0.699±0.029 | 0.095±0.003 | 0.018±0.002 | 0.001±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.616±0.008 | 0.602±0.004 | 0.113±0.014 | 0.615±0.009 | 0.614±0.009 | 0.612±0.010 | 0.610±0.012 | 0.006±0.004 |
| `both` (both) | 0.614±0.007 | 0.602±0.005 | 0.090±0.004 | 0.613±0.008 | 0.612±0.008 | 0.612±0.008 | 0.610±0.008 | 0.003±0.002 |
