# H0 final report (H0_R5_Baseline)

- Intervention: R5 Global-only baseline (reused if config-identical).
- Optimization config: `{"mode": "baseline"}`
- Best epoch / completed epochs / stop reason: 6 / 9 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.903096 | 0.619838 | 0.236135 |
| Valid | 0.803469 | 0.428405 | 0.311185 |
| Gap (Valid−Train) | — | — | +0.075051 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.986568 | 0.921581 | 0.097966 |
| Valid | 0.775794 | 0.392911 | 0.453414 |
| Gap (Valid−Train) | — | — | +0.355448 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.008271 | 0.010345 |
| stage4 | 0.007639 | 0.010417 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

