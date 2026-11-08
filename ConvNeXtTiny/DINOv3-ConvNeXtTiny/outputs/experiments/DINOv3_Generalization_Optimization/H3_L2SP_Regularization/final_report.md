# H3 final report (H3_L2SP_Regularization)

- Intervention: L2-SP penalty on trainable Stage3/4 pretrained parameters.
- Optimization config: `{"mode": "l2sp"}`
- Best epoch / completed epochs / stop reason: 12 / 12 / `resource_budget_stop`.

> **Convergence limitation**: training was stopped at epoch 12 by the total experiment time budget. The three-condition early-stopping rule did NOT trigger and validation AUROC was still slowly improving at the stop epoch. This experiment is not a converged run; its comparison with H0 is indicative only.

## Best checkpoint (full deterministic no-augmentation inference)

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.793639 | 0.399611 | 0.315458 |
| Valid | 0.763786 | 0.372637 | 0.332165 |
| Gap (Valid−Train) | — | — | +0.016706 |

## Last checkpoint

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.793639 | 0.399611 | 0.315458 |
| Valid | 0.763786 | 0.372637 | 0.332165 |
| Gap (Valid−Train) | — | — | +0.016706 |

## Parameter drift vs original DINOv3 (Best / Last)

| Stage | Best relative drift | Last relative drift |
|---|---:|---:|
| stage3 | 0.000321 | 0.000321 |
| stage4 | 0.000227 | 0.000227 |

## Artifacts

- `training_history.csv`: per-epoch raw train BCE, validation BCE, Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.
- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) and validation (deterministic) BCE.
- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: full deterministic per-class Train/Valid BCE and positive/negative split.
- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` (AP-only / PA+Lateral / single / multi), `training_curves.png`.
- `best.pth` / `last.pth` follow the project checkpoint convention.

- Test labels were never read; Test split was not used for selection.

