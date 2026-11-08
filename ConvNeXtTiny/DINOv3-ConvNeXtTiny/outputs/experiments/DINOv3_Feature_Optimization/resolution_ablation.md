# DINOv3 resolution ablation: 320 vs 448 vs 512

R0 is the completed P2 DINOv3 ConvNeXt-Tiny GAP baseline. R1 and R2 use the same raw Tiny initialization, stage-freeze policy, Study Logit Mean, BCE, seed and early-stopping criteria. Test split was not read.

## Best checkpoints

| Group | Resolution | Best Epoch | Stop Epoch | Stop Reason | Macro-AUROC | Formal train images/s | Train BCE at best | Valid BCE at best |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| R0 | 320 | 5 | 8 | three_condition_early_stop | 0.794882 | 23.58 | nan | nan |
| R1 | 448 | 6 | 9 | three_condition_early_stop | 0.799318 | 22.32 | 0.263042 | 0.318269 |
| R2 | 512 | 6 | 9 | three_condition_early_stop | 0.802116 | 22.99 | 0.264244 | 0.311968 |

## Subject-cluster paired Bootstrap

| Difference | Delta Macro-AUROC | Subject bootstrap 95% CI |
|---|---:|---:|
| R1 − R0 | +0.004436 | [-0.014100, +0.021090] |
| R2 − R0 | +0.007234 | [-0.010093, +0.021461] |
| R2 − R1 | +0.002798 | [-0.005994, +0.010473] |

## Interpretation limits

- Bootstrap replicates: 2000; seed: 42; validation Studies: 859.
- The best epoch is selected using the same validation set. Confidence intervals are conditional and do not replace an independent Test evaluation.
- Per-class resolution differences are in `per_class_auroc_comparison.csv`.
- Test split was not read.
