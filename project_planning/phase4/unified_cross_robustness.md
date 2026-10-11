# Unified robust models — cross-robustness comparison

Three models trained with `train.stage1_robust_unified` (seeds 0/1/2, mean±std). `nat` = naturalistic only (frozen encoders); `adv` = PGD adversarial (vision unfrozen); `both` = combined.

- **Clean** = clean macro-F1 (best threshold).
- **Nat-F1** = mean attacked macro-F1 over all naturalistic cells; **Nat-ASR** = mean attack-success-rate.
- **PGD@k** = macro-F1 under white-box PGD at ε=k/255; **PGD8-ASR** = ASR at ε=8/255.


## Split: `dev`

| Model | Clean | Nat-F1 | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.655±0.017 | 0.632±0.017 | 0.099±0.003 | 0.073±0.010 | 0.015±0.002 | 0.003±0.002 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.594±0.005 | 0.574±0.007 | 0.100±0.006 | 0.591±0.008 | 0.591±0.007 | 0.591±0.008 | 0.589±0.009 | 0.010±0.007 |
| `both` (both) | 0.593±0.003 | 0.578±0.002 | 0.088±0.003 | 0.594±0.002 | 0.594±0.002 | 0.593±0.003 | 0.593±0.003 | 0.000±0.000 |

## Split: `test_seen`

| Model | Clean | Nat-F1 | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.673±0.026 | 0.652±0.020 | 0.092±0.005 | 0.065±0.002 | 0.012±0.005 | 0.002±0.002 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.611±0.007 | 0.583±0.002 | 0.099±0.005 | 0.609±0.008 | 0.608±0.009 | 0.607±0.009 | 0.606±0.010 | 0.008±0.005 |
| `both` (both) | 0.611±0.007 | 0.587±0.005 | 0.089±0.005 | 0.610±0.008 | 0.609±0.008 | 0.609±0.009 | 0.607±0.009 | 0.005±0.003 |

## Split: `test_unseen`

| Model | Clean | Nat-F1 | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |
|---|---|---|---|---|---|---|---|---|
| `nat` (naturalistic) | 0.658±0.031 | 0.639±0.027 | 0.095±0.003 | 0.073±0.004 | 0.014±0.002 | 0.001±0.000 | 0.000±0.000 | 1.000±0.000 |
| `adv` (adversarial) | 0.579±0.004 | 0.557±0.007 | 0.113±0.014 | 0.578±0.004 | 0.578±0.005 | 0.577±0.006 | 0.576±0.006 | 0.006±0.004 |
| `both` (both) | 0.576±0.005 | 0.564±0.005 | 0.090±0.004 | 0.575±0.005 | 0.575±0.005 | 0.574±0.005 | 0.574±0.005 | 0.003±0.002 |
