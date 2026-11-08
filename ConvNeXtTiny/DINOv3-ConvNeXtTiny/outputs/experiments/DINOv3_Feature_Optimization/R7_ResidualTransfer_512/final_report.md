# R7 final report

- Group directory: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R7_ResidualTransfer_512`
- Best epoch: 6; best Validation Macro-AUROC: 0.802901; stop reason: `three_condition_early_stop`.
- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, channels-last, identical checkpoint loading for every model.
- Validation Macro-AUROC under this protocol: 0.802901.

## Inference ablation

| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |
|---|---:|---:|---:|---:|
| R7_Fused | 0.802901 | 0.427874 | +0.000000 | +0.000000 |
| R7_gamma0 | 0.802953 | 0.427953 | +0.000052 | +0.000080 |

## Paired Subject bootstrap (fresh predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|
| R7_Fused_minus_R5 | -0.000568 | [-0.0011, -0.0001] | -0.000531 | [-0.0047, +0.0009] |
| R7_Fused_minus_R7_gamma0 | -0.000052 | [-0.0002, +0.0000] | -0.000080 | [-0.0003, +0.0001] |

## Per-disease AUROC (best checkpoint)

| Disease | Pos/Neg | AUROC | 95% CI |
|---|---:|---:|---|
| Enlarged Cardiomediastinum | 24/835 | 0.7065 | [+0.5332, +0.8404] |
| Pneumothorax | 37/822 | 0.8548 | [+0.7652, +0.9245] |
| Consolidation | 46/813 | 0.7884 | [+0.7099, +0.8426] |
| Pneumonia | 68/791 | 0.7234 | [+0.6556, +0.7879] |
| Edema | 115/744 | 0.8851 | [+0.8437, +0.9288] |
| Cardiomegaly | 177/682 | 0.7763 | [+0.7222, +0.8302] |
| Atelectasis | 193/666 | 0.7923 | [+0.7527, +0.8355] |
| Lung Opacity | 214/645 | 0.7223 | [+0.6742, +0.7673] |
| Pleural Effusion | 260/599 | 0.9082 | [+0.8603, +0.9450] |
| No Finding | 306/553 | 0.8717 | [+0.8312, +0.9050] |

## Inference cost

- Parameters: 28,122,731; FP32 weights: 112.5 MB.
- Validation throughput: 11.46 Study/s; mean 87.3 ms/Study; peak 0.37 GiB.
- Submission package: not exported

## Saved-prediction replay diagnostic

- Status: `matched`; saved vs fresh unified inference max |Δprobability| = 0.000000; Macro-AUROC 0.802901 (saved) vs 0.802901 (fresh), Δ=+0.000000.
- All paired comparisons in this report use the fresh unified-protocol predictions, not the historical saved file. The R5 historical replay difference (max |Δp| ≈ 0.00935) was already recorded in the project before this task and remains an unresolved reproducibility limitation.

## Diagnostic interpretation

- Branch/attention diagnostics are model-behaviour measurements, not verified lesion localization.
- The Validation split has been used for repeated checkpoint selection and multiple comparisons across the project; all confidence intervals are exploratory and do not replace an independent test set.
- Test labels were never read; the Test split was not used for tuning.

