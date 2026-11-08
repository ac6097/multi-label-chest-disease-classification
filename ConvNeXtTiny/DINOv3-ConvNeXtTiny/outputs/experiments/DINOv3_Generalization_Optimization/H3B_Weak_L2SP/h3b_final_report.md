# H3-B final report (Weak L2-SP)

- H3-B changes only λ: 2892779.94258 = 0.1 × H3 λ (28927799.4258).
- Status: completed 19/20 epochs; stop reason `three_condition_early_stop`; Best epoch 13; Best Validation Macro-AUROC 0.791497.
- Trainable parameters: 26,592,778 (Global-only, same as H0/H3); regularized L2-SP parameters: 26,583,552.

## Convergence record (last three epochs)

| Epoch | Train raw BCE | Valid raw BCE | Valid Macro-AUROC | Valid Macro-AUPRC | L2-SP term | Stage3 drift | Stage4 drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.268941 | 0.322026 | 0.784327 | 0.405758 | 0.0185794 | 0.001259 | 0.000747 |
| 18 | 0.259695 | 0.326178 | 0.783923 | 0.405459 | 0.0197721 | 0.001290 | 0.000783 |
| 19 | 0.251222 | 0.330165 | 0.780670 | 0.403659 | 0.0207805 | 0.001315 | 0.000813 |

- Stop reason: **three_condition_early_stop**；最后 epoch = 19，Best epoch = 13。
- 是否尚未收敛迹象：最后两轮 AUROC 未继续上升；按原规则正常结束。

## Best checkpoint full deterministic inference

| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| Train | 0.837969 | 0.475081 | 0.290968 |
| Valid | 0.791497 | 0.413933 | 0.316011 |
| Gap (Valid−Train BCE) | — | — | +0.025044 |

## Per-disease Train/Valid BCE at Best checkpoint

| Disease | H0 Train | H0 Valid | H3 Train | H3 Valid | H3B Train | H3B Valid |
|---|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.125810 | 0.124876 | 0.137533 | 0.121995 | 0.132633 | 0.122918 |
| Pneumothorax | 0.108250 | 0.141200 | 0.162515 | 0.157552 | 0.144545 | 0.141984 |
| Consolidation | 0.150230 | 0.178681 | 0.175162 | 0.190039 | 0.163075 | 0.182187 |
| Pneumonia | 0.216491 | 0.256572 | 0.258776 | 0.268491 | 0.244323 | 0.263429 |
| Edema | 0.204797 | 0.276348 | 0.278735 | 0.283880 | 0.254817 | 0.276341 |
| Cardiomegaly | 0.325708 | 0.422168 | 0.422396 | 0.443446 | 0.393754 | 0.426170 |
| Atelectasis | 0.337646 | 0.434288 | 0.434993 | 0.472142 | 0.404419 | 0.449360 |
| Lung Opacity | 0.411262 | 0.503647 | 0.491505 | 0.513734 | 0.475689 | 0.507163 |
| Pleural Effusion | 0.220422 | 0.357672 | 0.366441 | 0.418887 | 0.306853 | 0.364209 |
| No Finding | 0.260731 | 0.416401 | 0.426526 | 0.451482 | 0.389568 | 0.426351 |
| Overall BCE | 0.236135 | 0.311185 | 0.315458 | 0.332165 | 0.290968 | 0.316011 |

## Parameter / feature drift (Best checkpoints)

| Experiment | Stage3 relative drift | Stage4 relative drift | Stage4 GAP cosine vs pretrained | classifier-input cosine |
|---|---:|---:|---:|---:|
| H0 | 0.008271 | 0.007639 | 0.1340 | 0.2087 |
| H3 | 0.000321 | 0.000227 | 0.0091 | 0.1080 |
| H3B | 0.001139 | 0.000664 | -0.0031 | 0.0692 |

## Grouped Validation metrics (Best)

| Experiment | Group | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---|---:|---:|---:|
| H0 | AP-only | 0.724279 | 0.403985 | 0.157224 |
| H0 | PA+Lateral | 0.868411 | 0.465589 | 0.079337 |
| H0 | Single-image | 0.727877 | 0.402549 | 0.171979 |
| H0 | Multi-image | 0.849957 | 0.447634 | 0.085522 |
| H3 | AP-only | 0.684753 | 0.347450 | 0.149669 |
| H3 | PA+Lateral | 0.825530 | 0.391041 | 0.081607 |
| H3 | Single-image | 0.694360 | 0.361311 | 0.161945 |
| H3 | Multi-image | 0.804185 | 0.362657 | 0.088618 |
| H3B | AP-only | 0.712055 | 0.390836 | 0.155770 |
| H3B | PA+Lateral | 0.859758 | 0.429549 | 0.077841 |
| H3B | Single-image | 0.719658 | 0.399281 | 0.168390 |
| H3B | Multi-image | 0.838607 | 0.409887 | 0.084929 |

### Positive Study / independent positive Subject support per group

（每个 Study 至少一个标签为阳性，因此 group 的 any-positive 恒等于全部；下表按十类疾病给出阳性 Study / 独立阳性 Subject 的 min–median–max，逐疾病精确计数见 `grouped_performance_comparison.csv` 的 `n_positive_studies` / `n_positive_subjects`。）

| Split | Group | Studies | Subjects | Positive Studies (min/median/max) | Independent positive Subjects (min/median/max) |
|---|---|---:|---:|---:|---:|
| train | Overall | 18513 | 11055 | 622 / 3144.5 / 6683 | 572 / 2267.0 / 5724 |
| train | AP-only | 9331 | 5236 | 432 / 1847.0 / 3224 | 395 / 1492.5 / 2066 |
| train | PA+Lateral | 8455 | 6842 | 160 / 804.5 / 4587 | 160 / 739.5 / 4095 |
| train | Single-image | 8507 | 4999 | 395 / 1683.0 / 2963 | 364 / 1382.5 / 1952 |
| train | Multi-image | 10006 | 7838 | 227 / 1078.5 / 5035 | 224 / 974.5 / 4474 |
| valid | Overall | 859 | 228 | 24 / 146.0 / 306 | 13 / 44.5 / 166 |
| valid | AP-only | 442 | 96 | 15 / 91.5 / 187 | 11 / 34.5 / 47 |
| valid | PA+Lateral | 388 | 181 | 7 / 33.0 / 211 | 6 / 24.0 / 130 |
| valid | Single-image | 391 | 94 | 15 / 76.5 / 162 | 10 / 33.5 / 44 |
| valid | Multi-image | 468 | 198 | 9 / 50.5 / 231 | 8 / 32.0 / 141 |

## Paired bootstrap (Best, Validation, 2000 subject-cluster draws)

- H3B − H0: Δ Macro-AUROC = -0.011971, 95% CI [-0.025593961775302886, 0.0011581733822822497]；Δ Macro-AUPRC = -0.014472，CI [-0.04140774682164192, 0.005163355916738477]。
- H3B − H3: Δ Macro-AUROC = +0.027712, 95% CI [0.012805868685245515, 0.038153466582298276]。
- 这些区间是验证集重复使用条件下的探索性统计，不是独立 Test 结论。

## Artifacts

- `training_history.csv` / `l2sp_gradient_history.csv`（逐 epoch 总体指标、L2-SP 项与 λ·梯度范数）、`per_epoch_class_loss.csv`（逐 epoch 逐类 Validation BCE 完整；Train 逐类仅最后一个 epoch 保留，属产物限制，不伪造中间值；Best/Last 的完整确定性 Train/Valid 逐类 BCE 见 `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`）；
- `per_class_metrics.csv`（逐 epoch 十类 Validation AUROC/AUPRC/BCE）；
- `best_checkpoint.pth` / `last_checkpoint.pth`（另保留项目惯例 `best.pth` / `last.pth`）；
- `per_class_loss_comparison.csv`、`per_class_auroc_comparison.csv`、`parameter_drift_comparison.csv`、`convergence_curves.png`；
- `l2sp_implementation_audit.md` / `.json`；
- 未读取 Test 标签，未用 Test 选择 λ 或 checkpoint。

