# H1 final report (H1_AP_Study_Sampling)

- Intervention: Study-level weighted sampling toward AP-only Studies (target 55%).
- Optimization config: `{"mode": "ap_sampling", "target_ap_fraction": 0.55, "ap_weight": 1.2}`
- Best epoch / completed epochs / stop reason: 4 / 7 / `three_condition_early_stop`.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.868471 | 0.544083 | 0.266316 |
| Valid | 0.785835 | 0.388620 | 0.329965 |
| Gap (Valid−Train) | — | — | +0.063649 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.961411 | 0.840616 | 0.153405 |
| Valid | 0.755185 | 0.365224 | 0.438363 |
| Gap (Valid−Train) | — | — | +0.284958 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.006358 | 0.009189 |
| stage4 | 0.006086 | 0.009729 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

