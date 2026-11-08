# R8 final report

- Group directory: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R8_DiseaseSpecific_StageLogitFusion_512`
- Best epoch: 6; best Validation Macro-AUROC: 0.803521; stop reason: `three_condition_early_stop`.
- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, channels-last, identical checkpoint loading for every model.
- Validation Macro-AUROC under this protocol: 0.803521.

## Inference ablation

| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |
|---|---:|---:|---:|---:|
| R8_Full | 0.803521 | 0.428488 | +0.000000 | +0.000000 |
| R8_BranchOff | 0.803444 | 0.428286 | -0.000076 | -0.000202 |

## Paired Subject bootstrap (fresh predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|
| R8_Full_minus_R5 | +0.000052 | [-0.0005, +0.0005] | +0.000083 | [-0.0029, +0.0016] |
| R8_Full_minus_R8_BranchOff | +0.000076 | [-0.0000, +0.0002] | +0.000202 | [-0.0005, +0.0011] |
| R8_BranchOff_minus_R5 | -0.000025 | [-0.0005, +0.0004] | -0.000119 | [-0.0030, +0.0014] |

## Per-disease AUROC (best checkpoint)

| Disease | Pos/Neg | AUROC | 95% CI |
|---|---:|---:|---|
| Enlarged Cardiomediastinum | 24/835 | 0.7088 | [+0.5348, +0.8429] |
| Pneumothorax | 37/822 | 0.8550 | [+0.7645, +0.9246] |
| Consolidation | 46/813 | 0.7883 | [+0.7108, +0.8420] |
| Pneumonia | 68/791 | 0.7242 | [+0.6560, +0.7888] |
| Edema | 115/744 | 0.8854 | [+0.8439, +0.9289] |
| Cardiomegaly | 177/682 | 0.7769 | [+0.7234, +0.8306] |
| Atelectasis | 193/666 | 0.7927 | [+0.7531, +0.8355] |
| Lung Opacity | 214/645 | 0.7235 | [+0.6755, +0.7680] |
| Pleural Effusion | 260/599 | 0.9089 | [+0.8613, +0.9458] |
| No Finding | 306/553 | 0.8716 | [+0.8311, +0.9049] |

## Per-disease branch-off diagnostic

| Disabled disease branch | Δ Macro-AUROC vs Full | Disabled disease AUROC Δ |
|---|---:|---:|
| Enlarged Cardiomediastinum | -0.000077 | -0.000773 |
| Pneumothorax | +0.000002 | +0.000016 |
| Consolidation | +0.000009 | +0.000094 |
| Pneumonia | -0.000009 | -0.000093 |
| Edema | +0.000004 | +0.000035 |
| Cardiomegaly | +0.000001 | +0.000008 |
| Atelectasis | +0.000001 | +0.000008 |
| Lung Opacity | -0.000008 | -0.000080 |
| Pleural Effusion | +0.000001 | +0.000010 |
| No Finding | +0.000000 | +0.000003 |

## Inference cost

- Parameters: 27,832,446; FP32 weights: 111.3 MB.
- Validation throughput: 11.68 Study/s; mean 85.6 ms/Study; peak 0.37 GiB.
- Submission package: verified, zip 103.6 MB, standalone max |Δp| 7.27e-03

## Saved-prediction replay diagnostic

- Status: `matched`; saved vs fresh unified inference max |Δprobability| = 0.000000; Macro-AUROC 0.803521 (saved) vs 0.803521 (fresh), Δ=+0.000000.
- All paired comparisons in this report use the fresh unified-protocol predictions, not the historical saved file. The R5 historical replay difference (max |Δp| ≈ 0.00935) was already recorded in the project before this task and remains an unresolved reproducibility limitation.

## Diagnostic interpretation

- Branch/attention diagnostics are model-behaviour measurements, not verified lesion localization.
- The Validation split has been used for repeated checkpoint selection and multiple comparisons across the project; all confidence intervals are exploratory and do not replace an independent test set.
- Test labels were never read; the Test split was not used for tuning.

The per-disease branch-off diagnostic isolates each disease's auxiliary logit contribution inside the trained model; it is not a training-time causal effect.

