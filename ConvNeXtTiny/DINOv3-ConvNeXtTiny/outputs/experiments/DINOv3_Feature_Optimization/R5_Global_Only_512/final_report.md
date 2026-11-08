# R5 final report

- Group directory: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R5_Global_Only_512`
- Best epoch: 6; best Validation Macro-AUROC: 0.803514; stop reason: `three_condition_early_stop`.
- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, channels-last, identical checkpoint loading for every model.
- Validation Macro-AUROC under this protocol: 0.803469.

## Inference ablation

| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |
|---|---:|---:|---:|---:|
| R5_baseline | 0.803469 | 0.428405 | +0.000000 | +0.000000 |

## Paired Subject bootstrap (fresh predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|

## Per-disease AUROC (best checkpoint)

| Disease | Pos/Neg | AUROC | 95% CI |
|---|---:|---:|---|
| Enlarged Cardiomediastinum | 24/835 | 0.7079 | [+0.5334, +0.8417] |
| Pneumothorax | 37/822 | 0.8552 | [+0.7658, +0.9250] |
| Consolidation | 46/813 | 0.7885 | [+0.7093, +0.8432] |
| Pneumonia | 68/791 | 0.7242 | [+0.6557, +0.7884] |
| Edema | 115/744 | 0.8854 | [+0.8439, +0.9291] |
| Cardiomegaly | 177/682 | 0.7769 | [+0.7230, +0.8305] |
| Atelectasis | 193/666 | 0.7926 | [+0.7528, +0.8358] |
| Lung Opacity | 214/645 | 0.7233 | [+0.6752, +0.7685] |
| Pleural Effusion | 260/599 | 0.9088 | [+0.8610, +0.9455] |
| No Finding | 306/553 | 0.8720 | [+0.8314, +0.9054] |

## Inference cost

- Parameters: 27,827,818; FP32 weights: 111.3 MB.
- Validation throughput: 11.55 Study/s; mean 86.6 ms/Study; peak 0.37 GiB.
- Submission package: not exported

## Saved-prediction replay diagnostic

- Status: `probability_mismatch`; saved vs fresh unified inference max |Δprobability| = 0.009352; Macro-AUROC 0.803514 (saved) vs 0.803469 (fresh), Δ=-0.000045.
- All paired comparisons in this report use the fresh unified-protocol predictions, not the historical saved file. The R5 historical replay difference (max |Δp| ≈ 0.00935) was already recorded in the project before this task and remains an unresolved reproducibility limitation.

## Diagnostic interpretation

- Branch/attention diagnostics are model-behaviour measurements, not verified lesion localization.
- The Validation split has been used for repeated checkpoint selection and multiple comparisons across the project; all confidence intervals are exploratory and do not replace an independent test set.
- Test labels were never read; the Test split was not used for tuning.

