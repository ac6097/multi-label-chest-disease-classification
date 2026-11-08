# H2C final report (H2C_SingleWeight_1p50)

- Intervention: Single-image Study BCE weight 1.50 with normalized weighted Study mean (same implementation as H2).
- Optimization config: `{"mode": "single_study_weight", "single_weight": 1.5}`
- Best epoch / completed epochs / stop reason: 7 / 10 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.936492 | 0.728859 | 0.194044 |
| Valid | 0.800501 | 0.407754 | 0.331139 |
| Gap (Valid−Train) | — | — | +0.137095 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.994671 | 0.967262 | 0.071472 |
| Valid | 0.770104 | 0.390511 | 0.558614 |
| Gap (Valid−Train) | — | — | +0.487142 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.009095 | 0.010705 |
| stage4 | 0.008806 | 0.011111 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

