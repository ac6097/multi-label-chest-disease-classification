# H2 final report (H2_Single_Study_Loss_Weight)

- Intervention: Single-image Study BCE weight 1.25 with normalized weighted Study mean.
- Optimization config: `{"mode": "single_study_weight", "single_weight": 1.25}`
- Best epoch / completed epochs / stop reason: 6 / 9 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.903954 | 0.622972 | 0.234986 |
| Valid | 0.805038 | 0.424958 | 0.311098 |
| Gap (Valid−Train) | — | — | +0.076112 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.986839 | 0.920506 | 0.097784 |
| Valid | 0.765304 | 0.388584 | 0.448960 |
| Gap (Valid−Train) | — | — | +0.351176 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.008268 | 0.010338 |
| stage4 | 0.007638 | 0.010391 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

