# R10 final report

- Group directory: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R10_Stage4Guided_CrossStageAttention_512`
- Best epoch: 6; best Validation Macro-AUROC: 0.803098; stop reason: `three_condition_early_stop`.
- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, channels-last, identical checkpoint loading for every model.
- Validation Macro-AUROC under this protocol: 0.803098.

## Inference ablation

| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |
|---|---:|---:|---:|---:|
| R10_Full | 0.803098 | 0.435072 | +0.000000 | +0.000000 |
| R10_BranchOff | 0.802291 | 0.433969 | -0.000807 | -0.001103 |
| R10_NoStage4Conditioning | 0.802385 | 0.434400 | -0.000713 | -0.000672 |

## Paired Subject bootstrap (fresh predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|
| R10_Full_minus_R5 | -0.000371 | [-0.003585, +0.002651] | +0.006667 | [-0.003363, +0.012246] |
| R10_Full_minus_R10_BranchOff | +0.000807 | [-0.001073, +0.002668] | +0.001103 | [-0.002624, +0.005223] |
| R10_Full_minus_R10_NoStage4Conditioning | +0.000713 | [-0.001094, +0.002507] | +0.000672 | [-0.003161, +0.004749] |

## Per-disease AUROC (best checkpoint)

| Disease | Pos/Neg | AUROC | 95% CI |
|---|---:|---:|---|
| Enlarged Cardiomediastinum | 24/835 | 0.7025 | [0.5308, 0.8395] |
| Pneumothorax | 37/822 | 0.8585 | [0.7704, 0.9255] |
| Consolidation | 46/813 | 0.7852 | [0.7051, 0.8396] |
| Pneumonia | 68/791 | 0.7218 | [0.6541, 0.7887] |
| Edema | 115/744 | 0.8842 | [0.8431, 0.9277] |
| Cardiomegaly | 177/682 | 0.7792 | [0.7248, 0.8329] |
| Atelectasis | 193/666 | 0.7923 | [0.7520, 0.8363] |
| Lung Opacity | 214/645 | 0.7230 | [0.6776, 0.7675] |
| Pleural Effusion | 260/599 | 0.9116 | [0.8679, 0.9465] |
| No Finding | 306/553 | 0.8727 | [0.8335, 0.9054] |

## Per-disease branch-off diagnostic

| Disabled disease branch | Δ Macro-AUROC vs Full | Disabled disease AUROC Δ |
|---|---:|---:|
| Enlarged Cardiomediastinum | -0.000432 | -0.004316 |
| Pneumothorax | -0.000012 | -0.000115 |
| Consolidation | +0.000180 | +0.001805 |
| Pneumonia | -0.000174 | -0.001738 |
| Edema | +0.000011 | +0.000111 |
| Cardiomegaly | +0.000045 | +0.000451 |
| Atelectasis | +0.000033 | +0.000327 |
| Lung Opacity | -0.000234 | -0.002340 |
| Pleural Effusion | -0.000229 | -0.002286 |
| No Finding | +0.000003 | +0.000033 |

## Inference cost

- Parameters: 28,029,694; FP32 weights: 112.1 MB.
- Validation throughput: 10.69 Study/s; mean 93.5 ms/Study; peak 0.37 GiB.
- Submission package: verified, zip 104.3 MB, standalone max |Δp| 6.82e-03

## Saved-prediction replay diagnostic

- Status: `matched`; saved vs fresh unified inference max |Δprobability| = 0.000000; Macro-AUROC 0.803098 (saved) vs 0.803098 (fresh), Δ=+0.000000.
- All paired comparisons in this report use the fresh unified-protocol predictions, not the historical saved file. The R5 historical replay difference (max |Δp| ≈ 0.00935) was already recorded in the project before this task and remains an unresolved reproducibility limitation.

## Diagnostic interpretation

- Branch/attention diagnostics are model-behaviour measurements, not verified lesion localization.
- The Validation split has been used for repeated checkpoint selection and multiple comparisons across the project; all confidence intervals are exploratory and do not replace an independent test set.
- Test labels were never read; the Test split was not used for tuning.

