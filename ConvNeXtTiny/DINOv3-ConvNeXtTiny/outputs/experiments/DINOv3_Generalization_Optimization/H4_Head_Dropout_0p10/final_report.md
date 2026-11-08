# H4 final report (H4_Head_Dropout_0p10)

- Intervention: Dropout p=0.10 on the final pooled feature immediately before the Global-only linear classifier; no L2-SP, no sampler or other head changes.
- Optimization config: `{"mode": "head_dropout", "head_dropout": 0.1}`
- Best epoch / completed epochs / stop reason: 6 / 9 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.896072 | 0.597035 | 0.239685 |
| Valid | 0.803069 | 0.425843 | 0.313042 |
| Gap (Valid−Train) | — | — | +0.073357 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.979537 | 0.882196 | 0.118044 |
| Valid | 0.778455 | 0.394617 | 0.426254 |
| Gap (Valid−Train) | — | — | +0.308210 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.008292 | 0.010406 |
| stage4 | 0.007438 | 0.010061 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

