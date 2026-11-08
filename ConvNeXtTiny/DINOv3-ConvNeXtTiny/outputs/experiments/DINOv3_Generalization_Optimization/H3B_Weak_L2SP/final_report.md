# H3B final report (H3B_Weak_L2SP)

- Intervention: Weak L2-SP (0.1 x H3 lambda) on the identical Stage3/4 pretrained parameter set.
- Optimization config: `{"mode": "l2sp", "lambda_sp": 2892779.942582381, "lambda_source": "0.1 x H3 calibrated lambda (2.89278e6)", "parameter_normalization": "mean squared deviation over regularized pretrained parameter elements (same as H3)"}`
- Best epoch / completed epochs / stop reason: 13 / 19 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.837969 | 0.475081 | 0.290968 |
| Valid | 0.791497 | 0.413933 | 0.316011 |
| Gap (Valid−Train) | — | — | +0.025044 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.893925 | 0.604037 | 0.243246 |
| Valid | 0.780670 | 0.403659 | 0.330165 |
| Gap (Valid−Train) | — | — | +0.086920 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.001139 | 0.001315 |
| stage4 | 0.000664 | 0.000813 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

