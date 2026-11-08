# R9 final report

- Group directory: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R9_DiseaseSpecific_QueryAttention_512`
- Best epoch: 6; best Validation Macro-AUROC: 0.802962; stop reason: `three_condition_early_stop`.
- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, channels-last, identical checkpoint loading for every model.
- Validation Macro-AUROC under this protocol: 0.802963.

## Inference ablation

| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |
|---|---:|---:|---:|---:|
| R9_Full | 0.802963 | 0.427698 | +0.000000 | +0.000000 |
| R9_BranchOff | 0.802989 | 0.427534 | +0.000026 | -0.000164 |
| R9_Uniform | 0.802963 | 0.427699 | +0.000000 | +0.000001 |

## Paired Subject bootstrap (fresh predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|
| R9_Full_minus_R5 | -0.000506 | [-0.000972, -0.000018] | -0.000707 | [-0.004972, +0.001103] |
| R9_Full_minus_R9_BranchOff | -0.000026 | [-0.000139, +0.000089] | +0.000164 | [-0.000031, +0.000599] |
| R9_Full_minus_R9_Uniform | -0.000000 | [-0.000010, +0.000005] | -0.000001 | [-0.000008, +0.000000] |

## Per-disease AUROC (best checkpoint)

| Disease | Pos/Neg | AUROC | 95% CI |
|---|---:|---:|---|
| Enlarged Cardiomediastinum | 24/835 | 0.7061 | [0.5317, 0.8414] |
| Pneumothorax | 37/822 | 0.8556 | [0.7665, 0.9250] |
| Consolidation | 46/813 | 0.7886 | [0.7105, 0.8425] |
| Pneumonia | 68/791 | 0.7228 | [0.6555, 0.7871] |
| Edema | 115/744 | 0.8852 | [0.8435, 0.9288] |
| Cardiomegaly | 177/682 | 0.7763 | [0.7222, 0.8303] |
| Atelectasis | 193/666 | 0.7924 | [0.7526, 0.8354] |
| Lung Opacity | 214/645 | 0.7226 | [0.6749, 0.7676] |
| Pleural Effusion | 260/599 | 0.9086 | [0.8615, 0.9452] |
| No Finding | 306/553 | 0.8714 | [0.8314, 0.9051] |

## Per-disease branch-off diagnostic

| Disabled disease branch | Δ Macro-AUROC vs Full | Disabled disease AUROC Δ |
|---|---:|---:|
| Enlarged Cardiomediastinum | +0.000000 | +0.000000 |
| Pneumothorax | -0.000008 | -0.000082 |
| Consolidation | +0.000016 | +0.000160 |
| Pneumonia | +0.000012 | +0.000121 |
| Edema | +0.000002 | +0.000018 |
| Cardiomegaly | -0.000009 | -0.000087 |
| Atelectasis | +0.000008 | +0.000078 |
| Lung Opacity | +0.000001 | +0.000011 |
| Pleural Effusion | +0.000002 | +0.000016 |
| No Finding | +0.000003 | +0.000027 |

## Inference cost

- Parameters: 27,929,726; FP32 weights: 111.7 MB.
- Validation throughput: 10.74 Study/s; mean 93.1 ms/Study; peak 0.37 GiB.
- Submission package: verified, zip 103.9 MB, standalone max |Δp| 7.28e-03

## Saved-prediction replay diagnostic

- Status: `matched`; saved vs fresh unified inference max |Δprobability| = 0.000000; Macro-AUROC 0.802962 (saved) vs 0.802963 (fresh), Δ=+0.000001.
- All paired comparisons in this report use the fresh unified-protocol predictions, not the historical saved file. The R5 historical replay difference (max |Δp| ≈ 0.00935) was already recorded in the project before this task and remains an unresolved reproducibility limitation.

## Diagnostic interpretation

- Branch/attention diagnostics are model-behaviour measurements, not verified lesion localization.
- The Validation split has been used for repeated checkpoint selection and multiple comparisons across the project; all confidence intervals are exploratory and do not replace an independent test set.
- Test labels were never read; the Test split was not used for tuning.

