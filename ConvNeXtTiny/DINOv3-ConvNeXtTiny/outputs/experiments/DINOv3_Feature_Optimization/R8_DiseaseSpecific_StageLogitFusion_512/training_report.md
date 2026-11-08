# R8 training report

- Backbone: DINOv3 ConvNeXt-Tiny DINOv3ConvNeXtTinyR8DiseaseSpecificStageLogitFusion; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.803521.
- Best Train BCE / Valid BCE: 0.262573 / 0.311232.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.142532 / 0.454420 / 0.773732.
- Mean formal train throughput: 21.75 images/s.
- Mean DataLoader wait: 1046.8 seconds/epoch.
- Trainable / frozen parameters: 26,597,406 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371810 | 0.343266 | 0.733951 | 0.343125 | 0.159950 | 21.90 | False | False | False | False |
| 2 | 0.327346 | 0.330548 | 0.769521 | 0.375873 | 0.208822 | 21.27 | False | False | False | False |
| 3 | 0.312292 | 0.321000 | 0.786312 | 0.401048 | 0.283905 | 21.92 | False | False | False | False |
| 4 | 0.299378 | 0.346040 | 0.786599 | 0.409979 | 0.237106 | 21.38 | True | False | False | False |
| 5 | 0.284604 | 0.309927 | 0.797812 | 0.413917 | 0.252338 | 20.98 | True | False | False | False |
| 6 | 0.262573 | 0.311232 | 0.803521 | 0.428488 | 0.313264 | 21.00 | True | False | False | False |
| 7 | 0.230908 | 0.338504 | 0.796495 | 0.412759 | 0.323213 | 21.95 | True | False | False | False |
| 8 | 0.189009 | 0.393697 | 0.789249 | 0.413858 | 0.347621 | 22.51 | True | True | False | False |
| 9 | 0.142532 | 0.454420 | 0.773732 | 0.391438 | 0.370969 | 22.87 | True | True | True | True |

## R8 auxiliary branch

LayerNorm(384) + Linear(384,10) on Stage3 GAP, zero-initialized head, z_k = z4_k + sigmoid(theta_k) * z3_k

The original R5 Global Head and main path are unchanged; the added state namespaces are `stage3_branch, fusion_alpha`. Per-epoch branch diagnostics are in `branch_diagnostics.csv`, `alpha_by_epoch.csv`, `stage3_logit_contribution.csv`.
