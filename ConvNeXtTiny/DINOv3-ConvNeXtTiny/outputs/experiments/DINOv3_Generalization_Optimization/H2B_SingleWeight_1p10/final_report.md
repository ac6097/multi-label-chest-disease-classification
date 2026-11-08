# H2B final report (H2B_SingleWeight_1p10)

- Intervention: Single-image Study BCE weight 1.10 with normalized weighted Study mean (same implementation as H2).
- Optimization config: `{"mode": "single_study_weight", "single_weight": 1.1}`
- Best epoch / completed epochs / stop reason: 6 / 9 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.904072 | 0.622586 | 0.234647 |
| Valid | 0.803685 | 0.427324 | 0.311292 |
| Gap (Valid−Train) | — | — | +0.076645 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.986997 | 0.922559 | 0.097680 |
| Valid | 0.767252 | 0.389223 | 0.448062 |
| Gap (Valid−Train) | — | — | +0.350382 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.008273 | 0.010345 |
| stage4 | 0.007639 | 0.010408 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

